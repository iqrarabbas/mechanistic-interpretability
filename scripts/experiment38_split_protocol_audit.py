import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "split_manifest_supervisor_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "protocol" / "split_audit_supervisor_v1.json"


def intersection(left, right):
    start = max(left["start"], right["start"])
    end = min(left["end"], right["end"])
    return None if start >= end else [start, end]


def audit_protocol(manifest, protocol_name):
    protocol = manifest[protocol_name]
    datasets = manifest["datasets"]
    splits = protocol["splits"]
    invalid_ranges = []
    forbidden_overlaps = []
    allowed_within_group_overlaps = []

    for split in splits:
        dataset = datasets[split["dataset"]]
        if split["start"] < 0 or split["end"] <= split["start"] or split["end"] > dataset["size"]:
            invalid_ranges.append(split)

    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            if left["dataset"] != right["dataset"]:
                continue
            overlap = intersection(left, right)
            if overlap is None:
                continue
            record = {
                "dataset": left["dataset"],
                "left": left["name"],
                "right": right["name"],
                "left_group": left["isolation_group"],
                "right_group": right["isolation_group"],
                "intersection": overlap,
                "images": overlap[1] - overlap[0],
            }
            if left["isolation_group"] == right["isolation_group"]:
                allowed_within_group_overlaps.append(record)
            else:
                forbidden_overlaps.append(record)

    return {
        "protocol": protocol_name,
        "status": protocol["status"],
        "split_count": len(splits),
        "invalid_ranges": invalid_ranges,
        "forbidden_overlaps": forbidden_overlaps,
        "allowed_within_group_overlaps": allowed_within_group_overlaps,
        "passes_isolation": not invalid_ranges and not forbidden_overlaps,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit current and proposed research split isolation")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-proposed-pass", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    report = {
        "manifest": str(args.manifest.resolve()),
        "manifest_version": manifest["manifest_version"],
        "current": audit_protocol(manifest, "current_protocol"),
        "proposed": audit_protocol(manifest, "proposed_protocol"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved split audit to {args.output}")

    if args.require_proposed_pass and not report["proposed"]["passes_isolation"]:
        raise SystemExit("Proposed protocol contains invalid ranges or forbidden overlap")


if __name__ == "__main__":
    main()
