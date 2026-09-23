# Codex Working Instructions

## Scope and First Read

These instructions apply to the entire repository.

Before doing research work:

1. Read `PROJECT_HANDOFF.md` completely.
2. Inspect the exact script and saved `summary.json` for any experiment being discussed.
3. Run `git status --short`; the working tree contains valuable modified and untracked research artifacts.
4. Do not infer that an experiment is missing merely because it is untracked by Git.

## Research Invariants

- Do not change or fine-tune the ViT backbone unless the user explicitly starts a separate fine-tuning study.
- Keep `google/vit-base-patch16-224` frozen for the current method and comparisons.
- Do not replace original hidden states with lossy SAE reconstructions. Apply SAE edits as residual decoder deltas.
- Paired clean/corrupted images are allowed for development diagnosis and feature discovery only. Deployment evaluation must not use the clean counterpart.
- Preserve the Block-11 intervention point for the current backbone unless running a clearly labeled layer-localization study.
- Do not assume Block 11, SAE feature IDs, thresholds, or gate weights transfer to another model. Rediscover them per backbone.
- Preserve three independently trained adapter seeds in confirmatory evaluations.
- Preserve paired sample order, corruption seeds, and the same held-out images when comparing methods.
- Keep the locked primary gate at 16 harmful SAE features and scale 2 unless an experiment is explicitly testing alternatives.
- Always include clean accuracy when reporting corruption improvement.
- Distinguish percentage points (`pp`) from relative percent changes.

## Project Structure

- `scripts/`: numbered research experiments and training/evaluation entry points.
- `interpretability/`: SAE, attention, logit-lens, and CFA utilities.
- `corruption/`: online Gaussian blur/noise implementations.
- `data/`: dataset wrappers and preprocessing.
- `training/`: model training/fine-tuning utilities from the broader project.
- `evaluation/`: common evaluation helpers.
- `checkpoints/`: SAE, CFA statistics, and other saved model artifacts.
- `results/sae/`: numbered experiment outputs, summaries, outcome arrays, and adapter/gate checkpoints.
- `external_data/`: ImageNetV2 matched-frequency data used for independent evaluation.
- `Dataset/`: local ImageNet-derived datasets/subsets. This is already large; do not duplicate it.
- `reports/` and `docs/`: earlier research reports and notes.
- `references/`: copied external source material required for continuity.
- `third_party/CFA/`: local CFA source/reference implementation.

## Environment

- Repository root: `/media/dr-yougart/Iqrar/vit_mi`
- Primary conda environment: `/home/dr-yougart/miniconda3/envs/vit`
- Python command:

  `PYTHONPATH=. /home/dr-yougart/miniconda3/envs/vit/bin/python scripts/<script>.py ...`

- Full evaluations require CUDA. Confirm scripts print `Device: cuda`.
- Use the system/base environment only for lightweight shell inspection, not scientific runs.
- Hugging Face may warn about missing `HF_TOKEN`; local cached model weights have worked. Do not download a different model silently.

## Important Data and Artifact Locations

- ImageNetV2: `external_data/imagenetv2-matched-frequency-format-val`
- Other local datasets/subsets: `Dataset/`
- Base paper: `references/gao_mechanistic_analysis_adversarial_finetuning_vits.pdf`
- Main Blur paper-style SAE: `checkpoints/sae/blur4_base_vanilla_paper`
- Earlier Blur SAE: `checkpoints/sae/blur4_base_vanilla`
- Harmful-feature source: `results/sae/experiment17_noise_bidirectional_repair/noise_bidirectional_development_gpu/summary.json`
- Three adapter checkpoints:
  - `results/sae/experiment25_multiseed_confirmation/imagenetv2_multiseed_confirmation/seed_0/classification_weight_0.05.pt`
  - `results/sae/experiment25_multiseed_confirmation/imagenetv2_multiseed_confirmation/seed_1/classification_weight_0.05.pt`
  - `results/sae/experiment25_multiseed_confirmation/imagenetv2_multiseed_confirmation/seed_2/classification_weight_0.05.pt`
- Nine completed gate replications: `results/sae/experiment36_sae_abnormality_gate/replication_*`

Do not mix SAE checkpoints when comparing cosine values or feature IDs. Read each `summary.json` configuration first.

## Important Scripts

- SAE training: `scripts/train_sae_level4.py`, `scripts/train_clean_sae.py`
- Paper-style SAE comparison: `scripts/compare_sae_level4.py`
- Failure-conditioned SAE changes: `scripts/analyze_base_sae_feature_changes.py`
- Causal layer localization: `scripts/experiment19_noise_layer_localization.py`
- Residual predictor/adapter: `scripts/experiment23_noise_patch_residual_predictor.py`, `scripts/experiment24_classification_aware_residual_predictor.py`
- Three-seed adapter confirmation: `scripts/experiment25_multiseed_independent_confirmation.py`
- Transfer: `scripts/experiment27_corruption_transfer.py`
- CFA severity comparison: `scripts/experiment30_cfa_imagenetv2_severity_sweep.py`
- SAE mediation/random controls: `scripts/experiment31_sae_adapter_causal_mediation.py`, `scripts/experiment32_multirandom_sae_subspace_test.py`, `scripts/experiment33_corruption_amplification_test.py`
- SAE-guided gate: `scripts/experiment36_sae_abnormality_gate.py`
- Gate grid: `scripts/experiment37_gate_validation_grid.py`

## Naming and Output Conventions

- Use `experimentNN_descriptive_name.py` for a new numbered experiment.
- Use a unique, descriptive `--run-name`; most scripts intentionally use `exist_ok=False`.
- Full runs should include `full`, condition, and seed/control details in the run name.
- Smoke runs should begin with `smoke_` and use tiny sample counts.
- Every scientific run should save a `summary.json` containing configuration, paths, seed, device, metrics, and limitations/status.
- Save paired per-image outcomes as compressed NPZ when later confidence intervals or McNemar tests may be needed.
- Never overwrite or delete completed output directories to reuse a name.

## Validation Workflow

1. Read applicable code and adjacent experiment patterns.
2. Make minimal changes with `apply_patch`.
3. Run `python -m py_compile` on changed Python files.
4. Run a tiny CPU/GPU smoke test with a unique `smoke_` name.
5. Inspect the smoke `summary.json`.
6. Only then launch the full CUDA run.
7. For long jobs, use unbuffered Python (`python -u`) and a visible PTY/session.
8. After completion, count expected summaries and verify no process remains.

## Experiments That Must Not Be Repeated by Default

Do not rerun these expensive completed evaluations unless the user explicitly requests replication or a verified artifact is missing:

- Experiment 15 independent frozen ImageNetV2 confirmation.
- Experiment 25 three-seed adapter confirmation.
- Experiment 27 fixed adapter corruption transfer.
- Experiment 30 full CFA severity sweep.
- Experiment 32 three-seed, 20-control random-subspace evaluations.
- Experiment 33 paired corruption-amplification analysis.
- Experiment 34 full extreme-corruption stress test.
- Experiment 36 original full gate run and all nine `replication_*` runs.
- Experiment 37 full 4x4 gate validation grid.

Use their existing JSON/NPZ outputs for new tables, confidence intervals, and plots.

## Statistical and Reporting Rules

- Prefer paired image-level tests because methods evaluate the same images.
- Report recovered and damaged predictions, not only accuracy gain.
- Use exact McNemar/binomial tests where appropriate.
- Bootstrap paired differences over images for confidence intervals.
- Do not treat repeated gate seeds on the same images as independent image samples.
- For random controls, report the number of controls, empirical rank/percentile, and minimum attainable empirical p-value.
- Separate development selection from held-out evaluation.
- Label clean-paired restoration as an oracle diagnostic, never an inference-time method.
- State that the controlled CFA comparison is not an official ImageNet-C reproduction.

## Current Next Task

The immediate recommended task is a non-expensive aggregation script for the nine completed Experiment 36 runs. It should produce per-run/per-adapter tables, paired confidence intervals for harmful gate versus ungated adapter, random-control ranks, JSON/CSV output, and a publication-quality plot without rerunning inference.

After that, propose a reusable model-configurable Layer Recovery Locator based on Experiment 19. Validate that it reproduces Layer 11 on the current ViT before testing a second backbone.

## Safety

- Never run `git reset`, `git clean`, broad `rm`, or delete results/checkpoints.
- Do not commit or create branches unless explicitly requested.
- Do not modify unrelated code or retrofit old results.
- Do not copy or version another multi-gigabyte dataset when an in-repository copy already exists.
- If electricity/process interruption occurs, count completed `summary.json` files and restart only incomplete unique runs.

