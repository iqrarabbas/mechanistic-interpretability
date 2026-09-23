# Final Method Specification and Freeze Rules

## 1. Purpose

The supervisor correctly observed that the project currently contains several systems: a full residual adapter, low-rank adapters, LoRA baselines, SAE interventions, repair subspaces, routers, and corruption-specific experts. A paper cannot present all of them as competing primary methods.

This document defines one provisional primary method and assigns every other component a precise role. It also defines the decision rule that will freeze the final intervention layer after Experiment 130.

---

## 2. Provisional Primary Method

The primary candidate is:

> **A clean-identity-preserving mixed-corruption patch-token residual adapter inserted at one frozen ViT block.**

The intervention is:

`h'_patch = h_patch + alpha * A(h_patch)`

where:

- `h_patch` is the corrupted image's patch-token representation at the selected block;
- `A` is the learned residual adapter;
- `alpha=1.0` at evaluation;
- the CLS token is not directly modified;
- all ViT parameters remain frozen;
- the corrected representation passes through the remaining original ViT blocks.

The method uses only the test image at inference. It does not require:

- the clean counterpart;
- labels;
- an SAE;
- corruption identity;
- test-time gradients;
- batch-level adaptation statistics.

---

## 3. Locked Components

The following choices are frozen unless Experiment 130 directly invalidates the intervention location:

| Component | Frozen choice |
|---|---|
| Backbone | `google/vit-base-patch16-224` |
| Backbone updates | None |
| Adapter architecture | Full patch-token residual adapter |
| Adapter parameters | 741,120 |
| Training corruptions | Equal mixture of Gaussian Noise-4 and Gaussian Blur-4 |
| Identity weight | `0.2` |
| Classification weight | `0.05` |
| Preservation weight | `0.05` |
| Training correction scale | `0.5` |
| Evaluation correction scale | `1.0` |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Scheduler | Cosine annealing |
| Maximum epochs | 5 |
| Early-stopping patience | 2 |
| Seeds | 3 disjoint train/validation partitions |

No further hyperparameter sweep is permitted on final evaluation data.

---

## 4. Component Still Awaiting Freeze

The adapter's block location is provisional.

- Current candidate: **Block 6**.
- Required deciding evidence: **Experiment 130**, which compares Blocks 1–11 under the same mixed-corruption and identity-preserving objective.
- Block 12 is excluded because patch-only correction after the final Transformer block cannot change the already-computed CLS representation.

### 4.1 Primary selection score

For every layer, report three-seed means for:

- clean accuracy change;
- Noise-4 accuracy change;
- Blur-4 accuracy change.

The primary robustness score is:

`mean(Noise-4 gain, Blur-4 gain)`

### 4.2 Clean-safety constraint

A candidate layer is eligible only if:

1. its mean clean change is not worse than `-0.5 pp`; and
2. no seed exhibits a statistically significant clean degradation after correction for the layer comparisons.

The `-0.5 pp` threshold is specified before inspecting Experiment 130's final accuracy table.

### 4.3 Final layer rule

1. Remove layers that violate the clean-safety constraint.
2. Among eligible layers, select the highest primary robustness score.
3. If two layers differ by less than `0.25 pp`, prefer the earlier layer only if it has no worse clean result and a clear mechanistic justification; otherwise prefer the smaller clean cost.
4. Report the complete layer table, not only the winner.

### 4.4 Confirmation rule

Experiment 130 is a development-selection experiment. After choosing the layer:

- freeze the layer and all hyperparameters;
- do not revise the method using confirmation results;
- evaluate the frozen winner on data not used for layer selection;
- retain all three independently trained seeds.

---

## 5. Training Objective

For corruption `c`, the adapter minimizes:

`L_c = L_residual + lambda_cls L_cls + lambda_pres L_pres + lambda_id L_id`

The final mixed loss is:

`L = 0.5 L_noise + 0.5 L_blur`

### 5.1 Residual loss

The adapter predicts the paired clean-minus-corrupted patch-token difference using Smooth L1 loss.

This paired clean information is used only during adapter training. It is not needed at inference.

### 5.2 Classification loss

The corrected corrupted representation is propagated through all remaining frozen ViT blocks. Cross-entropy encourages the correction to improve the final prediction rather than merely reconstruct the clean hidden state.

### 5.3 Preservation loss

For corrupted inputs that the base model already classifies correctly, the loss penalizes reductions in the true-class margin. This reduces unnecessary damage to already-correct predictions.

### 5.4 Clean-identity loss

The adapter is encouraged to output zero on clean hidden states. This is the principal clean-safety mechanism and is why the layer sweep must use the same identity loss at every location.

---

## 6. Deployment Algorithm

For a new image:

1. Run the frozen ViT through the selected intervention block.
2. Extract its patch tokens.
3. Compute the adapter correction from those patch tokens.
4. Add the correction residually at scale `1.0`.
5. Preserve the original CLS token at the intervention boundary.
6. Run the corrected sequence through the remaining frozen ViT blocks.
7. Return the original classifier's prediction.

The primary method always applies the adapter. A detector or router is not currently part of the primary method.

---

## 7. Role of Every Other Project Component

| Component | Final role | Not claimed as |
|---|---|---|
| SAE | Mechanism discovery and diagnostic representation | Required deployment module |
| Oracle clean restoration | Upper-bound/localization analysis | Deployable method |
| Full residual adapter | Primary method | Backbone fine-tuning |
| Rank-8/16/32/64/128 adapters | Efficiency ablations | Separate primary methods |
| Frozen PCA/subspace maps | Mechanistic compression and intervention ablations | Universal replacement for the adapter |
| Q/V LoRA | Parameter-matched PEFT baseline | Mechanistically proven winner |
| K-only/MLP-only/wrong-block modules | Mechanism-opposed controls | Candidate final systems |
| Noise-only and Blur-only adapters | Specialization ablations | Preferred deployment configuration |
| Two-expert router | Oracle/conditional-routing ablation | Full mixture-of-experts contribution |
| Corruption detector | Optional repair-or-abstain safety study | Universal corruption detector |
| CFA-style method | Controlled comparison | Official CFA/ImageNet-C reproduction |

---

## 8. Mechanistic Contribution to the Method

The role of mechanistic interpretability is specific and testable:

1. **Diagnosis:** corruption alters task-relevant activation magnitudes and dominant features.
2. **Localization:** oracle restoration identifies where corrupted information remains recoverable.
3. **Practical intervention test:** matched adapters determine where corrupted representations can be repaired without clean-state access.
4. **Trajectory analysis:** downstream blocks convert patch-token correction into improved CLS evidence.
5. **Architecture controls:** attention, MLP, LoRA, low-rank, SAE, hidden-space, and random controls identify which intervention families are effective.

The method is therefore not described as "an adapter discovered entirely by SAE." The accurate claim is:

> Mechanistic analysis motivated the repair location, token scope, residual intervention form, and controls; the final architecture was selected through matched deployable experiments.

---

## 9. Primary Evaluation Table

Every final comparison must include:

| Category | Required metric |
|---|---|
| Clean | Base and corrected top-1 accuracy |
| Seen corruptions | Noise-4 and Blur-4 accuracy |
| Held-out families | Per-family and average accuracy |
| Severity | Levels 1–5 where available |
| Distribution shift | Frozen ImageNetV2/A/Sketch results with dataset-status caveats |
| Paired statistics | Recovered, damaged, McNemar p-value, bootstrap CI |
| Efficiency | Trainable parameters, stored size, FLOPs, latency |
| Deployment | Test-time gradients, batch dependence, clean counterpart, labels |

All methods in the main baseline table must use the same backbone, images, ordering, corruption seeds, and evaluation preprocessing.

---

## 10. Main Claim After Successful Confirmation

If Experiment 130 and independent confirmation support the selected layer, the main method claim may be:

> A mechanistically localized, identity-preserving residual adapter improves common-corruption robustness in a frozen ViT without requiring test-time optimization, batch statistics, corruption labels, or clean counterparts, while preserving clean accuracy.

This claim does not imply:

- state-of-the-art performance before baseline comparison;
- certified or adaptive adversarial robustness;
- universal corruption detection;
- transfer of the same absolute block index to other backbones.

---

## 11. Failure and Pivot Rules

### If Block 6 wins

Freeze Block 6 and proceed to standardized baselines, held-out families, and cross-backbone procedural replication.

### If another layer wins

Update the primary method to that layer. Present Block 6 as the earlier provisional result and explain why identity-preserving mixed training changed the ranking.

### If no layer is clean-safe

The full always-on adapter cannot be the primary method. Pivot to either:

- a lower-rank clean-safe adapter; or
- a repair-or-abstain method with independently validated clean safety.

### If baselines dominate

Reframe the work as an analysis paper emphasizing corruption mechanisms, origin/repair dissociation, and cross-model mechanistic replication rather than raw robustness leadership.

---

## 12. Freeze Checklist

The final method is frozen only when all boxes below are satisfied:

- [ ] Experiment 130 completed for Blocks 1–11 and all three seeds.
- [ ] Complete clean/Noise/Blur layer table inspected.
- [ ] Layer selected by the predeclared rule.
- [ ] Hyperparameters unchanged after selection.
- [ ] Primary method checkpoint paths recorded.
- [ ] Parameters and inference algorithm verified.
- [ ] Confirmation dataset declared before evaluation.
- [ ] Baseline fairness protocol frozen.
- [ ] No confirmation result used to retune the method.

---

## 13. Authoritative Artifacts

- Setup and leakage protocol: `docs/EXPERIMENTAL_SETUP_AND_LEAKAGE_PROTOCOL.md`
- Supervisor matrix: `docs/SUPERVISOR_RESPONSE_MATRIX.md`
- Current layer sweep: `scripts/experiment130_identity_matched_layer_sweep.py`
- Provisional mixed Block-6 method: `scripts/experiment67_mixed_noise_blur_block6_adapter.py`
- Split manifest: `configs/split_manifest_supervisor_v1.json`
- Provisional mixed-adapter results: `results/sae/experiment67_mixed_noise_blur_block6/full_3seed_noise_blur_identity0p2_v1/summary.json`
