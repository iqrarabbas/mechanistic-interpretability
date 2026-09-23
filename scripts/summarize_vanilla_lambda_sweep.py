import json
from pathlib import Path


ROOT = Path(__file__).parent.parent / "checkpoints" / "sae"


def main():
    rows = []
    for metadata_path in sorted(
        ROOT.glob("blur4_base_vanilla_paper_pilot_lambda_*/training.json")
    ):
        metadata = json.loads(metadata_path.read_text())
        best_epoch = min(
            metadata["history"],
            key=lambda epoch: epoch["validation"]["loss"],
        )
        validation = best_epoch["validation"]
        rows.append(
            (
                metadata["l1_coefficient"],
                best_epoch["epoch"],
                validation["loss"],
                validation["reconstruction"],
                validation["sparsity"],
                validation["active_features"],
            )
        )

    if not rows:
        raise SystemExit("No completed lambda pilot checkpoints found.")

    print("lambda\tepoch\ttotal\treconstruction\tl1\tactive_features")
    for row in rows:
        print(
            f"{row[0]:.6g}\t{row[1]}\t{row[2]:.4f}\t{row[3]:.4f}"
            f"\t{row[4]:.4f}\t{row[5]:.2f}"
        )


if __name__ == "__main__":
    main()
