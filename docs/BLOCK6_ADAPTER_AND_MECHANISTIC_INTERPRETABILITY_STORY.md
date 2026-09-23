# From Mechanistic Diagnosis to a Deployable Block-6 Robustness Adapter

## Scope

This document explains the project up to the full Block-6 adapter and corruption-routing work. It connects:

1. how we localized recoverable corruption damage;
2. why we selected Block 6 rather than the apparently most recoverable late layer;
3. how the residual adapter was designed and trained;
4. how its correction travels from patch tokens to the final prediction;
5. what attention, value, residual-stream, MLP, head, and circuit analyses revealed;
6. how the method generalized to independent and out-of-distribution data;
7. how we attempted to detect corruption and route between repair experts;
8. what is supported, what failed, and what remains limited.

The later SAE Blur-subspace and low-rank shared-subspace studies are intentionally excluded.

---

## 1. Research Question

Common corruptions such as Gaussian noise and blur sharply reduce ViT accuracy. Full adversarial or corruption fine-tuning can improve robustness, but changes the whole backbone and is computationally expensive.

Our central question became:

> Can mechanistic interpretability identify where corruption damage is still recoverable, allowing a small frozen-backbone intervention to repair the representation?

We did not begin by arbitrarily attaching an adapter to a convenient layer. We first used causal restoration to identify recoverable locations, then performed a controlled deployable layer comparison, and finally analyzed why the selected location works.

---

## 2. Frozen Backbone and Representation Flow

The backbone is `google/vit-base-patch16-224`:

- 12 Transformer blocks;
- hidden width 768;
- 196 image patch tokens plus one CLS token;
- 1,000-class ImageNet classifier.

For an image, the simplified computation is:

```text
Image
  -> patch embedding
  -> Blocks 1-6
  -> Block-6 patch-token repair
  -> Blocks 7-12
  -> final CLS token
  -> classifier
```

The ViT backbone remains frozen. Only the small repair module is trained.

---

## 3. First Localization: Oracle Causal Restoration

### 3.1 Question

At which layer would correcting a corrupted hidden state allow the remaining frozen ViT to recover the prediction?

For a paired clean and Noise-4 image, we extracted hidden states at layer `l`:

```text
h_corrupt(l), h_clean(l)
```

We then formed a partial oracle restoration:

```text
h_test(l) = h_corrupt(l) + alpha * (h_clean(l) - h_corrupt(l))
```

and passed `h_test(l)` through the remaining frozen blocks.

This is not deployable because it uses the matching clean image. Its purpose is causal diagnosis: it asks whether the downstream network can use correctly repaired information from that location.

### 3.2 Oracle result

On 2,000 Noise-4 images, baseline accuracy was 68.70%. With `alpha=0.25`:

| Restored layer | Accuracy | Gain | Recovered | Damaged | Margin change |
|---:|---:|---:|---:|---:|---:|
| 1 | 73.30% | +4.60 pp | 124 | 32 | +0.485 |
| 3 | 73.60% | +4.90 pp | 130 | 32 | +0.530 |
| 6 | 74.75% | +6.05 pp | 143 | 22 | +0.677 |
| 9 | 74.70% | +6.00 pp | 134 | 14 | +0.661 |
| **11** | **75.00%** | **+6.30 pp** | **135** | **9** | **+0.691** |
| 12 | 75.00% | +6.30 pp | 132 | 6 | +0.636 |

Layer 11 was the strongest late-layer oracle location. This showed that substantial prediction information remained recoverable near the end of the network.

However, oracle recoverability and deployable repairability are different. The oracle supplies the correct clean direction; a real adapter must infer a correction from the corrupted representation alone.

---

## 4. Deployable Layer Search

We trained the same 741,120-parameter residual adapter after Blocks 6, 8, 9, 10, 11, and 12.

Every comparison used:

- the frozen ViT;
- the same adapter architecture;
- the same Noise-4 generation;
- the same training objective and schedule;
- zero initialization;
- three independently trained seeds;
- locked seed-specific train and validation partitions.

### Results

| Adapter location | Noise-4 change | Clean change | Interpretation |
|---:|---:|---:|---|
| **After Block 6** | **+2.20 pp** | -1.28 pp | Largest deployable gain |
| After Block 8 | -4.85 pp | -4.77 pp | Harmful |
| After Block 9 | -8.38 pp | -6.23 pp | Harmful |
| After Block 10 | -0.13 pp | -2.03 pp | Not useful |
| After Block 11 | +1.05 pp | -0.45 pp | Useful but weaker |
| After Block 12 | 0.00 pp | 0.00 pp | Expected negative control |

Block 12 cannot use a patch-only edit because no later attention layer remains to transfer the edited patch information into CLS.

### Main conclusion

> Layer 11 has high theoretical recoverability when given the correct oracle direction, but Block 6 has the highest practical repairability when the correction must be predicted from the corrupted image alone.

Block 6 provides the best balance:

- corruption damage is represented strongly enough to estimate a correction;
- useful image information has not yet been lost;
- six later blocks remain to transform repaired patch information into the CLS decision.

---

## 5. The Block-6 Residual Adapter

### 5.1 Architecture

Let the Block-6 patch-token matrix be:

```text
H in R^(196 x 768)
```

The adapter predicts:

```text
DeltaH = Linear(H) + PositionEmbedding
H_corrected = H + DeltaH
```

Only patch tokens are edited. The CLS token is copied unchanged at Block 6.

| Component | Shape or role |
|---|---|
| Linear weight | `768 x 768` |
| Linear bias | `768` |
| Patch-position embedding | `196 x 768` |
| Trainable parameters | **741,120** |
| FP32 size | **2.83 MiB** |
| Analytical overhead | **0.1156 GMAC/image** |

The module learns an additive correction rather than replacing the representation. This preserves the original ViT state and changes only what the adapter predicts is useful.

### 5.2 Training information

During training, paired clean and corrupted views were available. They provided:

```text
target residual = H_clean - H_corrupt
```

At inference, the method receives only the current image. It does not use:

- a clean counterpart;
- corruption labels;
- test labels;
- backbone updates.

### 5.3 Objective

The selected loss combined:

1. residual prediction toward the clean-minus-corrupted patch difference;
2. corrupted-image classification loss;
3. preservation of already-correct corrupted predictions;
4. clean-identity loss requiring approximately zero correction on clean features.

Conceptually:

```text
L = L_residual
  + 0.05 * L_classification
  + 0.05 * L_preservation
  + identity_weight * L_clean_identity
```

### 5.4 Hyperparameters

| Setting | Value |
|---|---:|
| Epochs | 5 |
| Early-stopping patience | 2 |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
| Smooth-L1 beta | 1.0 |
| Classification weight | 0.05 |
| Preservation weight | 0.05 |
| Training correction scale | 0.5 |
| Evaluation correction scale | 1.0 |
| Selected clean-identity weight | **0.2** |
| Independent adapter seeds | 3 |

---

## 6. Solving the Initial Clean-Accuracy Problem

The first Block-6 adapter improved Noise-4 by +2.20 pp but reduced clean accuracy by -1.28 pp. The diagnosis was simple: the objective taught the adapter how to repair corrupted activations but did not explicitly teach it to leave normal activations unchanged.

We added clean-identity regularization:

```text
adapter(H_clean) approximately 0
```

| Identity weight | Noise-4 gain | Clean change |
|---:|---:|---:|
| 0 | +2.20 pp | -1.28 pp |
| **0.2** | **+3.13 pp** | **+0.03 pp** |
| 1 | +2.90 pp | +0.13 pp |
| 5 | +2.17 pp | +0.18 pp |
| 20 | +1.68 pp | +0.07 pp |

Identity weight 0.2 was selected on development data and frozen before independent evaluation.

This established that the original clean loss was not an unavoidable Block-6 trade-off. It was an objective-design problem.

---

## 7. Independent Adapter Evaluation

The three frozen Noise-trained adapters were evaluated once on 10,000 ImageNetV2 images.

| Condition | Base ViT | Adapter mean | Change | Seed changes |
|---|---:|---:|---:|---|
| Clean | 68.37% | 68.28% | -0.09 pp | -0.16, -0.07, -0.03 |
| Noise-4 | 57.25% | 59.81% | **+2.56 pp** | +2.72, +2.61, +2.36 |
| Blur-4 | 50.82% | 52.14% | **+1.32 pp** | +1.12, +1.40, +1.44 |

All Noise and Blur improvements were significant under paired tests. Every clean confidence interval crossed zero, so there was no evidence of meaningful clean degradation.

The Blur gain was especially important because the adapter had been trained on Noise, not Blur. It showed that the repair was not purely memorizing one pixel corruption.

---

## 8. Out-of-Distribution Generalization

### 8.1 ImageNet-A

The frozen adapter was evaluated on all 7,500 ImageNet-A images. ImageNet-A was not used for training, feature selection, or hyperparameter tuning.

Under the standard masked-200 evaluation:

| Condition | Base | Mean adapter gain |
|---|---:|---:|
| Native ImageNet-A | 25.20% | **+1.28 pp** |
| Noise-4 ImageNet-A | 14.03% | **+2.28 pp** |
| Blur-4 ImageNet-A | 9.85% | **+0.90 pp** |

Every seed improved every condition significantly.

### 8.2 Mixed Noise-and-Blur adapter

We later trained one Block-6 adapter on a balanced mixture of Noise-4 and Blur-4 while retaining identity weight 0.2.

| Condition | Three-seed mean change |
|---|---:|
| Clean | +0.25 pp, nonsignificant |
| Noise-4 | **+3.22 pp** |
| Blur-4 | **+7.17 pp** |

This showed that one adapter can learn multiple repair modes without sacrificing the original Noise improvement.

### 8.3 Unseen corruptions and ImageNet-Sketch

The frozen mixed adapter was tested across seven additional corruption families and severities 1-5.

- ImageNet online benchmark mean gain: **+1.21 pp** across all corruption-severity combinations.
- ImageNet-Sketch cross-domain mean gain: **+1.16 pp**.
- Clean Sketch accuracy also improved in that run, although ImageNet-Sketch later became a repeatedly studied domain and should no longer be treated as pristine.

These are controlled online corruptions, not an official ImageNet-C reproduction.

---

## 9. How the Block-6 Repair Travels Downstream

The adapter changes patch tokens at Block 6, while CLS is initially unchanged. We propagated corrected representations through Blocks 7-12.

### Full adapter trajectory on Noise-4

- Block-6 patch cosine to clean increased by `+0.0565`.
- Relative patch error decreased by `0.1676`.
- CLS remained unchanged at the intervention point.
- Mean margin gain appeared at Block 7: `+0.0279`.
- It increased to `+0.1784` by Block 11.
- It reached `+0.3123` by Block 12.
- Mean true-class-logit gain at Block 12 was `+0.2477`.
- Final development accuracy improved by `+3.13 pp`.

### Patch-versus-CLS mediation

Experiment 122 replaced either corrected patches or corrected CLS with the corresponding baseline tokens.

| Stage | Gain lost when CLS reverted | Gain lost when patches reverted |
|---:|---:|---:|
| Block 6 | 0.00 pp | 3.00 pp |
| Block 7 | 0.03 pp | 3.00 pp |
| Block 8 | 0.20 pp | 2.80 pp |
| Block 9 | 0.30 pp | 2.67 pp |
| Block 10 | 0.40 pp | 2.87 pp |
| Block 11 | 0.57 pp | 1.73 pp |
| Block 12 | 3.00 pp | 0.00 pp |

The useful correction remains primarily patch-carried through Block 10. Blocks 11-12 then integrate it into the CLS token read by the classifier.

The supported computational story is:

```text
Block-6 patch correction
  -> distributed patch processing through Blocks 7-10
  -> late attention integration in Blocks 11-12
  -> improved CLS representation
  -> increased true-class margin
  -> improved top-1 prediction
```

Later blocks do not apply another adapter. They progressively convert the one Block-6 patch intervention into decision evidence.

---

## 10. Why Block 6 Is Repairable

### 10.1 Attention routing versus value content

At Block 6, we separately restored clean attention probabilities and clean value vectors using paired-clean oracle patches.

On 3,000 ImageNet validation images:

| Noise-4 intervention | Accuracy gain |
|---|---:|
| Clean routing, corrupted values | +0.37 pp, nonsignificant |
| Corrupted routing, clean values | **+3.37 pp** |
| Clean routing and values | +3.47 pp |
| Complete clean Block-6 output | +9.77 pp |

For Noise, corrupted content/value information was much more important than routing alone.

For Blur:

| Blur-4 intervention | Accuracy gain |
|---|---:|
| Clean routing, corrupted values | +0.43 pp, nonsignificant |
| Corrupted routing, clean values | **+3.00 pp** |
| Clean routing and values | **+4.90 pp** |
| Complete clean Block-6 output | +15.57 pp |

Blur also primarily damaged content, but routing became useful after content was restored. ImageNet-A replicated the central finding: value restoration was consistently useful, while routing alone was not.

### 10.2 Incoming residual, attention, and MLP

We decomposed Block 6 as:

```text
z = x + a + m
```

where `x` is the incoming residual stream, `a` is the attention output, and `m` is the MLP output.

Shapley attribution of complete Block-6 oracle recovery:

| Component | Noise attribution | Blur attribution |
|---|---:|---:|
| Incoming residual `x` | **70.3%** | **75.2%** |
| Attention output `a` | 15.8% | 18.2% |
| MLP output `m` | 13.9% | 6.6% |

Therefore, Block 6 is not the main place where corruption first appears. Most recoverable damage is already present when Block 6 begins. Block 6 is valuable because it is a strategically useful correction boundary.

---

## 11. Where the Damage Starts

Repeating the residual/attention/MLP decomposition at Blocks 1-5 showed:

- Block 1's MLP was the earliest dominant source/amplifier.
- For Noise, it accounted for 74.6% of the earliest full-block oracle recovery.
- For Blur, it also accounted for 74.6%.
- By Block 2, much of that damage had entered the incoming residual stream.
- Blocks 2-5 transformed and propagated the damage rather than independently creating most of it.

This does not mean Block 1 is the best intervention location.

We trained the exact same full adapter after Block 1:

| Location | Noise gain | Clean change |
|---|---:|---:|
| Block 1 | +0.93 pp | **-7.38 pp** |
| Block 6 | +2.20 pp | -1.28 pp before identity regularization |

The Block-1 adapter changed generic early features used by both clean and corrupted images. Later blocks amplified this clean distortion. Block 6 was safer because corruption damage had become more separable while enough downstream computation remained.

This distinction is central:

> The earliest damage-generating layer is not necessarily the best deployable repair layer.

---

## 12. Head and Circuit Analysis

### 12.1 Candidate-head discovery

We traced attention routing, CLS routing, value/content recovery, and MLP recovery through Blocks 7-12. Candidate heads included:

- Block 7 Head 11;
- Block 8 Head 9;
- Block 7 Head 0;
- Block 9 Head 4;
- Block 10 Head 3;
- several later Block-12 heads.

These were discovery candidates, not yet causal claims.

### 12.2 Necessity controls

We suppressed candidate heads and neurons and compared them with matched random removals. Small targeted sets did not consistently destroy the Noise repair more than random controls. Increasing the head-set size also failed to establish one stable tiny circuit.

Therefore we rejected the overly strong claim that a handful of named heads uniquely carry the complete repair.

### 12.3 Block-7 Q/K/V decomposition

At Block 7, reverting individual pathways showed different corruption structures:

| Reverted pathway | Mean Noise gain lost | Mean Blur gain lost |
|---|---:|---:|
| Q | 0.27 pp | 0.63 pp |
| K | 0.10 pp | 0.23 pp |
| V | **0.67 pp** | **0.97 pp** |
| Q+K+V / attention output | 0.10 pp | **2.33 pp** |
| MLP output | 0.83 pp | 1.23 pp |

The Noise result was seed-variable and distributed. Blur showed a much clearer dependence on the complete Block-7 attention pathway.

### 12.4 Reproducible block-level circuit

The later rank-128 Noise analysis tested reversion of attention and MLP outputs at every downstream block.

| Reverted component | Mean Noise repair lost |
|---|---:|
| Block 7 | 0.87 pp |
| Block 8 | 1.23 pp |
| Block 9 | 0.47 pp |
| Block 10 | 0.40 pp |
| Block 11 | 1.43 pp |
| Block 12 | 1.83 pp |
| **Blocks 11-12 together** | **2.73 pp** |
| All Blocks 7-12 | 2.80 pp |

Blocks 11-12 attention alone carried a mean 2.43 pp of the 2.80 pp repair effect, whereas their MLPs carried 0.93 pp.

The defensible Noise circuit is therefore coarse and distributed:

> Block-6 patch correction -> redundant processing through Blocks 7-10 -> dominant attention-mediated integration in Blocks 11-12 -> final CLS decision.

This is a stronger and more reproducible conclusion than assigning the mechanism to one attention head.

---

## 13. The Router: What It Is and What It Is Not

Two related but different routing questions were investigated.

### 13.1 Family router

The first router selected between a Noise expert and a Blur expert. It used only the current image and summarized early patch activations from Blocks 1-3.

Configuration:

- input: channel-wise means of patch tokens from Blocks 1-3;
- router parameters: **2,305**;
- frozen ViT and frozen experts;
- no clean counterpart;
- no corruption label at inference.

This router learned seen Noise/Blur families extremely accurately on development data and approached the oracle family choice there. However, a family router assumes that an input should be assigned to one known expert. It does not by itself answer whether the input is clean or whether an unseen corruption should be repaired.

### 13.2 Corruption detector

To answer the supervisor's question—whether internal activity can automatically indicate that an attack/corruption is occurring—we constructed features from:

- signed per-block attention-head means;
- head dispersion and variability;
- patch-token moments;
- image gradients and frequency statistics;
- later, four-view consistency under small benign transformations.

The single-view representation contained 152 features. Clean photo and clean Sketch distributions were modeled with regularized covariance, and a threshold was calibrated at the 95th clean-validation percentile.

Clean-only single-view results on 15 severity-4 corruptions:

- clean false-positive rate: 3.10% in the initial run;
- Gaussian, shot, and impulse noise recall: 100%;
- defocus blur recall: 81.7%;
- weak recall on zoom blur, elastic transform, pixelation, and JPEG.

On a paired fresh-image comparison:

| Detector | Clean FPR | Mean recall | Mean AUROC |
|---|---:|---:|---:|
| Single-view internal statistics | 3.81% | 40.01% | 0.830 |
| Four-view consistency | 6.62% | **45.34%** | **0.848** |
| Clean-calibrated fusion | 5.42% | 42.99% | 0.845 |

The simple fusion did not beat the four-view detector overall. Detection was excellent for strong noise but remained weak for subtle or digital corruptions such as JPEG.

### 13.3 Unknown-family detector

A small histogram-gradient-boosted classifier was then trained on the same 152 features using Noise and Blur examples only. It was evaluated on eight algorithms from completely unseen families.

| Detector | Clean false-alarm rate | Mean recall over all 15 |
|---|---:|---:|
| Clean-only baseline | 4.64% | 44.61% |
| **Fixed Noise+Blur detector** | **4.84%** | **69.55%** |
| All-corruptions-known reference | 4.84% | 83.82% |

For the eight unseen algorithms:

| Unseen family | Fixed detector | Clean-only baseline |
|---|---:|---:|
| Weather | 54.9% | 29.8% |
| Appearance | 65.0% | 45.0% |
| Digital/geometry | 16.9% | 7.1% |
| **All unseen algorithms** | **43.2%** | **25.1%** |

This is meaningful transfer, but not universal detection. Unknown JPEG recall was only 15.3%, and elastic-transform recall was 6.8%.

### 13.4 Abstaining selector

We combined detection and expert routing into three possible decisions:

```text
no repair / Noise expert / Blur expert
```

The selector frequently abstained on clean images, but abstention had two opposing effects:

- it avoided some adapter-induced errors;
- it also skipped a similar number of useful adapter recoveries.

Across seven corruptions, selector minus always-on mixed adapter averaged `-0.13 pp`, with a 95% interval of `[-0.74,+0.46] pp`. Therefore, the selector was not proven better than the simpler always-on mixed adapter.

The current conclusion is:

> Internal ViT statistics can detect many corruptions and generalize partially to unseen families, but they do not yet support a universally reliable clean/corrupt gate. The always-on identity-preserving mixed adapter remains the stronger deployment reference.

---

## 14. What Mechanistic Interpretability Contributed

The contribution is not simply “we trained an adapter.” MI determined the design and interpretation at every stage.

| MI question | Experiment answer | Engineering consequence |
|---|---|---|
| Where is correction theoretically useful? | Late layers, especially Layer 11, are highly oracle-recoverable | Established that corruption damage is causally reversible |
| Where can correction be inferred from corrupted input? | Block 6 wins the matched deployable layer test | Adapter placed after Block 6 |
| Why was clean accuracy initially harmed? | Adapter lacked a clean-identity constraint | Added identity loss 0.2 |
| What token carries the repair? | Patch tokens through Block 10, then CLS late | Patch-only intervention retained |
| Does Block 6 create the damage? | 70-75% arrives in its incoming residual | Block 6 reframed as a repair boundary |
| Where does damage first arise? | Block-1 MLP is the earliest major amplifier | Tested, then rejected Block 1 as unsafe deployment point |
| Is routing or content more damaged? | Value/content dominates; routing interaction is stronger for Blur | Explained corruption-specific attention behavior |
| Is there a tiny head circuit? | Candidate heads exist, but small-head necessity is unstable | Avoided an unsupported single-head claim |
| What is the robust downstream circuit? | Distributed Blocks 7-10, late attention bottleneck at Blocks 11-12 | Established a reproducible coarse causal circuit |
| Can internal signals detect attacks? | Strong for noise, partial transfer to unseen families, weak for some digital attacks | Router retained as promising but not superior deployment component |

MI changed the research from arbitrary architecture search into a sequence of falsifiable causal questions.

---

## 15. Data-Leakage and Statistical Safeguards

The corrected protocol uses separate purposes for data:

| Purpose | Allocation |
|---|---|
| Mechanism/feature development | dedicated ImageNet validation ranges |
| Adapter seed 0 train/validation | `[15000,22000)` split internally |
| Adapter seed 1 train/validation | `[22000,29000)` split internally |
| Adapter seed 2 train/validation | `[29000,36000)` split internally |
| Reserve mechanistic analyses | later ImageNet validation ranges |
| Frozen full-adapter confirmation | ImageNetV2 |
| OOD evaluation | ImageNet-A and frozen ImageNet-Sketch manifests |

Controls and reporting rules included:

- frozen backbone;
- zero-initialized independently trained adapters;
- three seeds;
- identical sample order and corruption seeds for paired comparisons;
- paired bootstrap confidence intervals;
- exact McNemar tests;
- recovered-versus-damaged prediction counts;
- no clean counterpart at inference;
- wrong-image activation controls in oracle patching;
- matched random head, neuron, and subspace controls where appropriate;
- explicit distinction between development, reused mechanistic, and independent evaluation data.

Some later ImageNet-Sketch experiments used newly hash-excluded images from a previously studied domain. They are image-level confirmations, not pristine new-domain benchmarks.

---

## 16. Final Connected Story

The full sequence is:

```text
Corruption reduces ViT accuracy
  -> oracle restoration proves that internal damage is recoverable
  -> late Layer 11 looks strongest with perfect clean information
  -> matched deployable adapters show Block 6 is actually easiest to repair
  -> a patch-token residual adapter is inserted after Block 6
  -> clean-identity regularization removes the clean-accuracy penalty
  -> independent ImageNetV2 confirms Noise and Blur improvements
  -> ImageNet-A confirms cross-domain robustness
  -> propagation analysis shows repaired patches carry the benefit through Block 10
  -> Blocks 11-12 attention integrates the benefit into CLS
  -> component patching shows value/content damage dominates routing-only damage
  -> early-block analysis identifies Block-1 MLP as the earliest amplifier
  -> Block-1 adapter control proves earliest damage is not the safest repair point
  -> head searches find candidates but reject an unstable tiny-circuit claim
  -> block-level reversion establishes a distributed downstream repair circuit
  -> internal-statistics routers detect strong corruptions and partially generalize
  -> abstaining routing does not yet beat the always-on mixed adapter
```

The central supported claim is:

> Corruption damage begins early, but Block 6 is the most useful deployable correction boundary. A small residual adapter repairs patch-token information there, after which distributed processing and late attention convert the repair into a stronger CLS decision. Mechanistic interpretability identified the layer, token pathway, damaged components, downstream circuit, and current limits of automatic corruption routing.

---

## 17. Primary Artifacts

- Layer localization: `scripts/experiment19_noise_layer_localization.py`
- Matched layer adapters: `scripts/experiment46_layer_specific_adapters.py`
- Clean-preserving Block-6 adapter: `scripts/experiment50_block6_clean_preservation.py`
- Blur transfer: `scripts/experiment51_block6_blur_transfer.py`
- Propagation: `scripts/experiment52_block6_repair_propagation.py`
- Independent ImageNetV2 confirmation: `scripts/experiment53_block6_independent_confirmation.py`
- ImageNet-A evaluation: `scripts/experiment60_block6_imageneta_ood.py`
- Attention/content patching: `scripts/experiment61_block6_attention_content_patching.py`
- ImageNet-A mechanism replication: `scripts/experiment62_block6_imageneta_mechanism.py`
- Residual/attention/MLP decomposition: `scripts/experiment63_block6_residual_mlp_decomposition.py`
- Early damage origin: `scripts/experiment64_early_block_damage_origin.py`
- Block-1 propagation control: `scripts/experiment65_block1_adapter_propagation.py`
- Mixed adapter: `scripts/experiment67_mixed_noise_blur_block6_adapter.py`
- OOD corruption evaluation: `scripts/experiment68_unseen_online_corruptions.py`
- Sketch evaluation: `scripts/experiment69_imagenet_sketch_frozen_generalization.py`
- Candidate circuit discovery: `scripts/experiment71_downstream_circuit_recovery_discovery.py`
- Circuit necessity: `scripts/experiment72_downstream_circuit_necessity.py`
- Head-set sweep: `scripts/experiment74_attention_circuit_size_sweep.py`
- Block-7 Q/K/V mediation: `scripts/experiment75_block7_qkv_path_mediation.py`
- Learned family router: `scripts/experiment77_learned_moe_router.py`
- Internal-statistics detector: `scripts/experiment114_multisignal_clean_detector.py`
- Multi-view detector: `scripts/experiment116_multiview_clean_detector.py`
- Paired detector comparison: `scripts/experiment117_paired_detector_fusion.py`
- Abstaining selector: `scripts/experiment119_abstaining_repair_selector.py`
- Selector audit: `scripts/experiment120_selector_error_audit.py`
- Unknown-family detector: `scripts/experiment121_leave_family_out_corruption_detector.py`
- Patch/CLS mediation: `scripts/experiment122_noise_patch_cls_mediation.py`
- Distributed block circuit: `scripts/experiment126_noise_distributed_block_circuit.py`
