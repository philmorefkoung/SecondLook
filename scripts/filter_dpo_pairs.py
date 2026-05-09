"""Filter mined DPO pairs for high-signal preferences.

DPO v2 regressed because the 221 mined pairs included many ambiguous edge
cases (SFT v2 was 89% correct on train; the 11% it got wrong were the
hardest, often genuinely uncertain cases). Filtering for "real bugs"
(high-confidence wrongs, decisive GT, decisive detector prob) keeps the
preference signal clean.

Filter criteria (default, AND-combined):
  1. SFT v2's confidence in the WRONG answer >= --min-rej-conf (default 0.7).
     The model was SURE - so this is a real failure, not uncertainty.
  2. GT match distance is unambiguous: very near a GT lesion (confirm case)
     OR very far from any (reject case).
       confirm: gt_match_dist_vox <= --confirm-max-dist (default 3.0 vox)
       reject:  gt_match_dist_vox >= --reject-min-dist  (default 12.0 vox)
  3. Detector probability is decisive: very high OR very low.
       --high-prob >= 0.50 (default), --low-prob <= 0.25 (default)

Note: gt_match_dist_vox is not in the original metadata; we recompute it
from coord_zyx + study GT centroids on disk if needed (skipped here -
filter only on what's already in the meta dict).

Usage:
  python scripts/filter_dpo_pairs.py \
      --in data/dpo_v2/mined_pairs.jsonl \
      --metadata data/sft_v3_train/metadata.jsonl \
      --out data/dpo_v3/filtered_pairs.jsonl
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _parse_confidence(rejected_text: str) -> float | None:
    """Pull confidence out of SFT v2's structured-JSON output."""
    m = re.search(r"\{.*\}", rejected_text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return float(obj.get("confidence", 0.5))
    except Exception:
        return None


def _build_metadata_lookup(metadata_path: Path) -> dict[tuple, dict]:
    out = {}
    for line in metadata_path.open():
        r = json.loads(line)
        key = (r["study_id"], tuple(int(c) for c in r["coord_zyx"]))
        out[key] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--metadata", default=None,
                    help="Optional original metadata.jsonl for additional fields like "
                         "match_dist_vox (used in confirm/reject distance filters).")
    ap.add_argument("--out", default="data/dpo_v3/filtered_pairs.jsonl")
    ap.add_argument("--min-rej-conf", type=float, default=0.7,
                    help="Minimum confidence in the WRONG answer to keep the pair.")
    ap.add_argument("--confirm-max-dist", type=float, default=3.0,
                    help="For GT=confirm pairs, keep only if matched GT lesion "
                         "centroid is within this distance (clean positive).")
    ap.add_argument("--reject-min-dist", type=float, default=12.0,
                    help="For GT=reject pairs, keep only if nearest GT lesion is at "
                         "least this far away (clean negative).")
    ap.add_argument("--high-prob", type=float, default=0.50)
    ap.add_argument("--low-prob",  type=float, default=0.25)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta_lookup = {}
    if args.metadata:
        meta_lookup = _build_metadata_lookup(Path(args.metadata))
        print(f"Loaded {len(meta_lookup)} metadata records for distance lookup")

    n_total = 0
    n_kept = 0
    drops = Counter()
    with out_path.open("w") as fout:
        for line in Path(args.in_path).open():
            p = json.loads(line)
            n_total += 1
            meta = p["meta"]
            gt = meta["gt_verdict"]

            # Confidence filter
            rej_conf = _parse_confidence(p["rejected"])
            if rej_conf is None or rej_conf < args.min_rej_conf:
                drops["confidence_too_low"] += 1
                continue

            # Distance filter (uses metadata if provided; else skipped)
            if meta_lookup:
                key = (meta["study_id"], tuple(int(c) for c in meta["coord_zyx"]))
                m = meta_lookup.get(key, {})
                gt_dist = float(m.get("gt_label", {}).get("match_dist_vox", float("inf")))
                if gt == "confirm" and gt_dist > args.confirm_max_dist:
                    drops["confirm_dist_too_high"] += 1
                    continue
                if gt == "reject" and gt_dist < args.reject_min_dist:
                    drops["reject_dist_too_low"] += 1
                    continue

            # Detector-probability filter (decisive only)
            prob = float(meta["detector_prob"])
            if not (prob >= args.high_prob or prob <= args.low_prob):
                drops["prob_borderline"] += 1
                continue

            fout.write(json.dumps(p) + "\n")
            n_kept += 1

    print(f"\nKept {n_kept}/{n_total} pairs ({n_kept/max(1,n_total):.1%})")
    print("Drop reasons:")
    for k, v in drops.most_common():
        print(f"  {k:>22}: {v}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
