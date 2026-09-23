import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment48_cost_accuracy_tradeoff"
RESOURCE_SUMMARY = (
    PROJECT_ROOT / "results" / "sae" / "experiment47_resource_audit"
    / "frozen_methods_pre_latency_v2" / "summary.json"
)
GATE_SUMMARY = (
    PROJECT_ROOT / "results" / "sae" / "experiment41_disjoint_gate_development"
    / "full_top8_top16_3seed_v1" / "summary.json"
)
DIRECT_SUMMARY = (
    PROJECT_ROOT / "results" / "sae" / "experiment43_direct_latent_repair"
    / "full_3seed_matched_direct_repair_optimized_v2" / "summary.json"
)
HYBRID_SUMMARY = (
    PROJECT_ROOT / "results" / "sae" / "experiment45_sae_hidden_subspace"
    / "full_3seed_rank16_normalized_v1" / "summary.json"
)


def load(path):
    return json.loads(path.read_text())


def mean_baselines(validation):
    records = list(validation.values())
    return {
        "clean": sum(record["baseline_clean_accuracy"] for record in records) / len(records),
        "noise": sum(record["baseline_noise4_accuracy"] for record in records) / len(records),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        "# Leakage-Free Development Cost–Accuracy Trade-off",
        "",
        "| Method | Noise-4 accuracy | Noise gain | Clean change | Positive seeds | Trainable params | Added GMAC | Evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['noise4_accuracy_percent']:.2f}% | "
            f"{row['noise4_gain_pp']:+.2f} pp | {row['clean_change_pp']:+.2f} pp | "
            f"{row['positive_seeds']} | {row['trainable_parameters']:,} | "
            f"{row['added_gmac_per_image']:.4f} | {row['evidence_status']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The full adapter has the largest stable gain and remains the primary engineering method.",
        "- The 16D raw repair has the strongest efficiency/clean-preservation trade-off, but its per-seed paired tests were not individually significant.",
        "- The SAE hybrid is positive in every seed and beats random SAE directions, but does not beat the strongest raw-hidden control.",
        "- The leakage-free SAE gate remains below the ungated full adapter and adds substantial frozen-SAE inference cost.",
        "- These are development results, not final ImageNetV2 claims. Exact CUDA latency remains pending.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Join leakage-free accuracy and resource results")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    resources = load(RESOURCE_SUMMARY)
    gate = load(GATE_SUMMARY)
    direct = load(DIRECT_SUMMARY)
    hybrid = load(HYBRID_SUMMARY)
    resource_map = {row["method"]: row for row in resources["methods"]}
    baseline = mean_baselines(direct["validation"])

    adapter_noise_gains = [
        direct["validation"][f"seed_{seed}"]["methods"]["adapter"]["noise_vs_baseline"]["accuracy_difference"]
        for seed in range(3)
    ]
    adapter_clean_gains = [
        direct["validation"][f"seed_{seed}"]["methods"]["adapter"]["clean_vs_baseline"]["accuracy_difference"]
        for seed in range(3)
    ]
    gate_selection = gate["selection"]["gate_top16"]
    gate_noise_gain = gate_selection["mean_noise4_accuracy"] - baseline["noise"]
    gate_clean_gain = gate_selection["mean_clean_accuracy"] - baseline["clean"]

    definitions = [
        {
            "method": "Frozen ViT baseline",
            "noise_gain": 0.0,
            "clean_gain": 0.0,
            "seed_gains": [0.0, 0.0, 0.0],
            "resource": "Frozen ViT baseline",
            "status": "Reference",
        },
        {
            "method": "Full 768D residual adapter",
            "noise_gain": sum(adapter_noise_gains) / 3,
            "clean_gain": sum(adapter_clean_gains) / 3,
            "seed_gains": adapter_noise_gains,
            "resource": "Full 768D residual adapter",
            "status": "Supported development method",
        },
        {
            "method": "16D raw-hidden repair",
            "noise_gain": direct["comparison"]["raw_hidden"]["mean_noise4_gain_vs_baseline"],
            "clean_gain": direct["comparison"]["raw_hidden"]["mean_clean_gain_vs_baseline"],
            "seed_gains": direct["comparison"]["raw_hidden"]["noise4_gains_by_seed"],
            "resource": "16D raw-hidden repair",
            "status": "Promising; paired CIs cross zero",
        },
        {
            "method": "16D SAE-discovered hybrid",
            "noise_gain": hybrid["comparison"]["harmful_sae_decoder"]["mean_noise4_gain_vs_baseline"],
            "clean_gain": hybrid["comparison"]["harmful_sae_decoder"]["mean_clean_gain_vs_baseline"],
            "seed_gains": hybrid["comparison"]["harmful_sae_decoder"]["noise4_gains_by_seed"],
            "resource": "16D shared-input subspace repair",
            "status": "Suggestive; not best matched basis",
        },
        {
            "method": "16D high-variance raw subspace",
            "noise_gain": hybrid["comparison"]["high_variance_hidden"]["mean_noise4_gain_vs_baseline"],
            "clean_gain": hybrid["comparison"]["high_variance_hidden"]["mean_clean_gain_vs_baseline"],
            "seed_gains": hybrid["comparison"]["high_variance_hidden"]["noise4_gains_by_seed"],
            "resource": "16D shared-input subspace repair",
            "status": "Highest small mean; fails one seed",
        },
        {
            "method": "16-feature SAE direct repair",
            "noise_gain": direct["comparison"]["harmful_sae"]["mean_noise4_gain_vs_baseline"],
            "clean_gain": direct["comparison"]["harmful_sae"]["mean_clean_gain_vs_baseline"],
            "seed_gains": direct["comparison"]["harmful_sae"]["noise4_gains_by_seed"],
            "resource": "16D raw-hidden repair",
            "status": "Not supported",
        },
        {
            "method": "Leakage-free top-16 SAE gate",
            "noise_gain": gate_noise_gain,
            "clean_gain": gate_clean_gain,
            "seed_gains": [
                gate["validation"][f"seed_{seed}"]["methods"]["gate_top16"]["noise_vs_baseline"]["accuracy_difference"]
                for seed in range(3)
            ],
            "resource": "17-parameter SAE gate",
            "status": "Below ungated adapter; not supported",
        },
    ]
    rows = []
    for definition in definitions:
        resource = resource_map[definition["resource"]]
        rows.append({
            "method": definition["method"],
            "noise4_accuracy_percent": 100 * (baseline["noise"] + definition["noise_gain"]),
            "noise4_gain_pp": 100 * definition["noise_gain"],
            "clean_accuracy_percent": 100 * (baseline["clean"] + definition["clean_gain"]),
            "clean_change_pp": 100 * definition["clean_gain"],
            "positive_seeds": f"{sum(value > 0 for value in definition['seed_gains'])}/3",
            "noise4_gains_by_seed_pp": json.dumps([100 * value for value in definition["seed_gains"]]),
            "trainable_parameters": resource["trainable_parameters"],
            "frozen_auxiliary_parameters": resource["frozen_auxiliary_parameters"],
            "added_gmac_per_image": resource["added_macs_per_image"] / 1e9,
            "checkpoint_mib": resource["checkpoint_mib"],
            "inference_requires_sae": resource["inference_requires_sae"],
            "evidence_status": definition["status"],
        })
    summary = {
        "configuration": {
            "accuracy_sources": [str(path.resolve()) for path in [GATE_SUMMARY, DIRECT_SUMMARY, HYBRID_SUMMARY]],
            "resource_source": str(RESOURCE_SUMMARY.resolve()),
            "baseline_clean_accuracy": baseline["clean"],
            "baseline_noise4_accuracy": baseline["noise"],
            "split_status": "leakage-free disjoint development validation",
            "imageNetV2_accessed_for_new_analysis": False,
            "inference_rerun": False,
            "latency_status": "pending",
        },
        "methods": rows,
        "limitations": [
            "Methods share the same split protocol, but were trained in separate experiments.",
            "Small-repair paired confidence intervals generally include zero.",
            "The SAE gate is an addition to the full adapter, not a standalone correction.",
            "Exact CUDA latency is not yet included.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(output_dir / "cost_accuracy_table.csv", rows)
    write_markdown(output_dir / "cost_accuracy_table.md", rows)
    print((output_dir / "cost_accuracy_table.md").read_text())
    print(f"Saved cost-accuracy trade-off to {output_dir}")


if __name__ == "__main__":
    main()
