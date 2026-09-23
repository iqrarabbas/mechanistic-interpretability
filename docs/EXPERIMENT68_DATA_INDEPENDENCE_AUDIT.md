# Experiment 68 Data-Independence Audit

## Decision

Experiment 68 is a frozen-adapter corruption stress test, not an independent final benchmark. Its images were not used to fit the Experiment 67 mixed adapter, and its seven corruption transformations were not used for adapter training. However, the ImageNet validation interval `[39000, 42000)` had already appeared in diagnostic and model-development experiments.

The reported mean gain of `+1.21 pp` may be cited as exploratory evidence only. It must not be presented as final generalization evidence until the frozen protocol is repeated on images that were never used for training, feature selection, architecture selection, hyperparameter selection, mechanistic diagnosis, or previous evaluation.

## Earlier Uses of the Evaluation Images

| Experiment | Earlier interval | Overlap with `[39000,42000)` | Role |
|---|---:|---:|---|
| 7 | `[40000,45000)` | `[40000,42000)` | Noise fine-tuned-model feature analysis |
| 18 | `[40000,45000)` | `[40000,42000)` | Clean-gated Noise repair development |
| 19 | `[40000,42000)` | `[40000,42000)` | Oracle layer localization; influenced intervention-location research |
| 20 | `[40000,42000)` | `[40000,42000)` | Oracle direction/capacity diagnosis |
| 56 | `[36000,46000)` | Entire interval | Leakage-free causal-subspace replication and causal interpretation |
| 57 | `[36000,46000)` | Entire interval | Rank-8 Noise propagation evaluation |
| 58 | `[36000,46000)` | Entire interval | Rank-8 clean/Blur evaluation |

Several smoke tests also touched individual images in this interval. Earlier experiments ending at index `40000` overlap only the first image boundary under half-open interval semantics if their implementation was inconsistent; under the recorded `[start,end)` convention they do not overlap `[40000,42000)` and overlap `[39000,42000)` only on `[39000,40000)` where applicable.

## Influence Assessment

- **No direct fitting leakage:** Experiment 67 trained only on the locked seed-specific ranges `[15000,22000)`, `[22000,29000)`, and `[29000,36000)`.
- **No corruption leakage:** brightness, contrast, JPEG, pixelation, defocus, shot noise, and impulse noise were not included in Experiment 67 training.
- **Indirect development exposure exists:** previous results on these images informed the project's layer, architecture, compression, and mechanistic decisions before Experiment 68 was run.
- **Conclusion:** paired statistics within Experiment 68 remain mathematically valid for that sample, but the sample is not an untouched confirmatory set.

## Locked Fresh Evaluation Protocol

Before obtaining fresh data, freeze the following choices:

- Backbone: `google/vit-base-patch16-224`, frozen.
- Adapters: the three completed Experiment 67 mixed Noise-4/Blur-4 Block-6 adapters, frozen.
- Comparator: the three completed Experiment 50 Noise-only Block-6 adapters, frozen.
- Corruption families and severity tables: exactly those saved by Experiment 68.
- Evaluation: clean plus all seven families at severities 1–5.
- Statistics: per-image paired outcomes, paired bootstrap 95% confidence intervals, exact McNemar tests, recovered/damaged prediction counts, per-seed results, and three-seed aggregate.
- Prohibited actions: no tuning, feature selection, architecture changes, threshold selection, or checkpoint selection after inspecting fresh-set results.

## Fresh-Data Requirement

No eligible untouched labeled dataset is currently available in the repository:

- all 50,000 ImageNet validation images have been used during development or diagnosis;
- ImageNetV2 has already been used for frozen evaluations;
- ImageNet-A has already been used for mechanistic replication;
- `Dataset/imagenet_train_subset` currently contains zero images and an accidental literal `n*` directory.

The proper next run therefore requires a newly populated, class-labelled ImageNet training subset or another ImageNet-1K-compatible dataset that has never been opened during this project. Its file list and SHA-256 hashes must be frozen before inference. The fresh result must be reported regardless of whether it confirms or contradicts Experiment 68.
