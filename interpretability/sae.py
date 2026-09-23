import torch
from torch import nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=768, expansion_factor=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = input_dim * expansion_factor
        self.encoder = nn.Linear(input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, input_dim)
        nn.init.normal_(self.encoder.weight, std=self.latent_dim**-0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, std=input_dim**-0.5)
        nn.init.zeros_(self.decoder.bias)

    def decode(self, latent):
        return self.decoder(latent)

    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


class VanillaReLUSAE(SparseAutoencoder):
    def encode(self, inputs):
        return F.relu(self.encoder(inputs))

    def forward(self, inputs):
        latent = self.encode(inputs)
        return self.decode(latent), latent


class BatchTopKSAE(SparseAutoencoder):
    def __init__(
        self,
        input_dim=768,
        expansion_factor=32,
        k=32,
        input_unit_norm=True,
        n_batches_to_dead=5,
    ):
        super().__init__(input_dim=input_dim, expansion_factor=expansion_factor)
        self.k = k
        self.input_unit_norm = input_unit_norm
        self.n_batches_to_dead = n_batches_to_dead
        self.register_buffer("batches_not_active", torch.zeros(self.latent_dim))
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.encoder.weight)
            nn.init.zeros_(self.encoder.bias)
            self.decoder.weight.copy_(self.encoder.weight.t())
            nn.init.zeros_(self.decoder.bias)
            self.normalize_decoder()

    def encode(self, inputs):
        normalized_inputs, _, _ = self.preprocess_inputs(inputs)
        preactivations = self.preactivations(normalized_inputs)
        return self.batch_topk(preactivations)

    def preprocess_inputs(self, inputs):
        if not self.input_unit_norm:
            return inputs, None, None
        input_mean = inputs.mean(dim=-1, keepdim=True)
        input_std = inputs.std(dim=-1, keepdim=True)
        return (inputs - input_mean) / (input_std + 1e-5), input_mean, input_std

    def postprocess_outputs(self, reconstruction, input_mean, input_std):
        if not self.input_unit_norm:
            return reconstruction
        return reconstruction * input_std + input_mean

    def batch_topk(self, preactivations):
        flat = preactivations.flatten()
        active_count = min(self.k * preactivations.shape[0], flat.numel())
        selected = torch.topk(flat, active_count, sorted=False)
        return torch.zeros_like(flat).scatter(0, selected.indices, selected.values).reshape(
            preactivations.shape
        )

    def preactivations(self, inputs):
        return F.relu(F.linear(inputs - self.decoder.bias, self.encoder.weight))

    def update_inactive_features(self, latent):
        active = latent.sum(dim=0) > 0
        self.batches_not_active.add_((~active).float())
        self.batches_not_active[active] = 0

    def auxiliary_loss(self, inputs, reconstruction, preactivations, aux_k=512):
        dead = self.batches_not_active >= self.n_batches_to_dead
        dead_count = int(dead.sum().item())
        if dead_count == 0:
            return inputs.new_zeros(())
        count = min(aux_k, dead_count)
        selected = torch.topk(preactivations[:, dead], count, dim=-1)
        auxiliary_latent = torch.zeros_like(preactivations[:, dead]).scatter(
            1, selected.indices, selected.values
        )
        auxiliary_reconstruction = F.linear(
            auxiliary_latent, self.decoder.weight[:, dead], bias=None
        )
        return F.mse_loss(auxiliary_reconstruction, inputs - reconstruction)

    def compute_loss(
        self,
        inputs,
        l1_coefficient=0.0,
        aux_coefficient=1 / 32,
        aux_k=512,
        update_activity=False,
        return_outputs=True,
    ):
        normalized_inputs, input_mean, input_std = self.preprocess_inputs(inputs)
        preactivations = self.preactivations(normalized_inputs)
        latent = self.batch_topk(preactivations)
        normalized_reconstruction = self.decode(latent)
        reconstruction_loss = F.mse_loss(normalized_reconstruction, normalized_inputs)
        sparsity_loss = latent.abs().sum(dim=-1).mean()
        if update_activity:
            self.update_inactive_features(latent)
        auxiliary_loss = self.auxiliary_loss(
            normalized_inputs, normalized_reconstruction, preactivations, aux_k
        )
        total_loss = (
            reconstruction_loss
            + l1_coefficient * sparsity_loss
            + aux_coefficient * auxiliary_loss
        )
        if not return_outputs:
            return total_loss, reconstruction_loss, sparsity_loss, auxiliary_loss
        reconstruction = self.postprocess_outputs(
            normalized_reconstruction, input_mean, input_std
        )
        return total_loss, reconstruction_loss, sparsity_loss, auxiliary_loss, reconstruction, latent

    def project_decoder_gradient(self):
        if self.decoder.weight.grad is None:
            return
        normalized = F.normalize(self.decoder.weight, dim=0)
        radial_gradient = (self.decoder.weight.grad * normalized).sum(
            dim=0, keepdim=True
        ) * normalized
        self.decoder.weight.grad.sub_(radial_gradient)

    def forward(self, inputs):
        _, _, _, _, reconstruction, latent = self.compute_loss(inputs)
        return reconstruction, latent


def extract_penultimate_patch_activations(model, images):
    """Return layer-11 patch activations, excluding the CLS token."""
    model.eval()
    with torch.no_grad():
        outputs = model.vit(pixel_values=images, output_hidden_states=True)
    return outputs.hidden_states[-2][:, 1:, :]


def sae_loss(inputs, reconstruction, latent, l1_coefficient=1e-3):
    reconstruction_loss = (reconstruction - inputs).pow(2).sum(dim=-1).mean()
    sparsity_loss = latent.abs().sum(dim=-1).mean()
    total = reconstruction_loss + l1_coefficient * sparsity_loss
    return total, reconstruction_loss, sparsity_loss


def corresponding_patch_cosine_similarity(clean_latent, corrupted_latent):
    if clean_latent.shape != corrupted_latent.shape:
        raise ValueError("Clean and corrupted SAE activations must have equal shapes.")
    return F.cosine_similarity(clean_latent, corrupted_latent, dim=-1, eps=1e-8)
