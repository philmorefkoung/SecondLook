"""Balance DPO preference pairs by GT verdict direction.

DPO v3 collapsed to over-rejection partly because (a) the cleaner data let
beta=0.1 push too hard, and (b) the combined 236-pair set still has a 1.3:1
reject-direction skew. This script downsamples the majority direction to
match the minority, producing perfectly-balanced training pairs.

Usage:
  python scripts/balance_dpo_pairs.py \
      --inputs data/dpo_v1/train_pairs.jsonl data/dpo_v3/filtered_pairs.jsonl \
      --out data/dpo_v4/balanced_pairs.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="data/dpo_v4/balanced_pairs.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for p in args.inputs:
        for line in Path(p).open():
            all_pairs.append(json.loads(line))
    print(f"Loaded {len(all_pairs)} pairs total")
    print(f"  by gt_verdict: {dict(Counter(r['meta']['gt_verdict'] for r in all_pairs))}")

    by_dir: dict[str, list] = {"confirm": [], "reject": []}
    for r in all_pairs:
        gt = r["meta"]["gt_verdict"]
        if gt in by_dir:
            by_dir[gt].append(r)

    n_min = min(len(by_dir["confirm"]), len(by_dir["reject"]))
    rng = random.Random(args.seed)
    confirm_pairs = by_dir["confirm"]
    reject_pairs = by_dir["reject"]
    if len(confirm_pairs) > n_min:
        rng.shuffle(confirm_pairs); confirm_pairs = confirm_pairs[:n_min]
    if len(reject_pairs) > n_min:
        rng.shuffle(reject_pairs); reject_pairs = reject_pairs[:n_min]
    balanced = confirm_pairs + reject_pairs
    rng.shuffle(balanced)
    print(f"Balanced to {n_min}/{n_min} = {len(balanced)} total")

    with out_path.open("w") as f:
        for r in balanced:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
