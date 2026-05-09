"""Build SFT-ready training records from a labelled metadata.jsonl.

Implements philosophy C:
  - For agreements (vlm_verdict == gt_verdict): preserve the VLM's structured
    output as the target (the reasoning is already correct in the model's voice).
  - For disagreements: re-prompt the same model with the GT verdict revealed,
    asking it to produce a corrected structured output that explains WHY the GT
    verdict is right. This rewrites the failed reasoning into a teaching example.

Output: train.jsonl + val.jsonl in Anthropic messages format
        (messages = [user(image+prompt), assistant(json verdict)]).

Class balancing: oversample positives in TRAIN to a configurable ratio (default
1:2 positive:negative) so the SFT'd model doesn't collapse to "always reject".

Usage:
  ANTHROPIC_API_KEY=... \
  python scripts/finalize_sft_targets.py \
      --in data/sft_v1 \
      --out data/sft_v1_finalized \
      --val-frac 0.1 \
      --neg-pos-ratio 2.0
"""
from __future__ import annotations
import argparse
import base64
import json
import random
import re
import time
from collections import Counter
from pathlib import Path


CORRECTOR_MODEL = "claude-sonnet-4-6"
CORRECTOR_MAX_TOKENS = 600
CORRECTOR_TEMPERATURE = 0.0


def _verifier_user_prompt(coord, prob) -> str:
    """Reconstructs the prompt the original verifier saw. Must match the
    structured-output schema used at eval time exactly so SFT training data
    matches inference-time prompting."""
    return (
        "You are an auditable second-finding verifier for brain MRI metastases.\n"
        "Your job is to judge whether the candidate at the centre of every tile is a real "
        "additional metastasis (not a vessel, CSF, artefact, or duplicate of the seed).\n"
        "\n"
        "The single PNG below is laid out as a 4 row x N modality grid:\n"
        "  Row 0: candidate AXIAL crops (modalities in order)\n"
        "  Row 1: candidate CORONAL crops\n"
        "  Row 2: candidate SAGITTAL crops\n"
        "  Row 3: SEED lesion AXIAL crops (for direct comparison)\n"
        "Red outline (rows 0-2) marks the proposal model's predicted candidate extent.\n"
        "Green outline (row 3) marks the seed lesion segmentation.\n"
        "The candidate is at the EXACT CENTRE of every candidate tile - localize before reasoning.\n"
        f"Candidate centroid (z,y,x)={coord}, detector probability={prob:.3f}.\n"
        "\n"
        "Respond with JSON only, matching this schema EXACTLY:\n"
        "{\n"
        '  "decision": "accept" | "reject" | "uncertain",\n'
        '  "evidence_for": [string, ...],\n'
        '  "evidence_against": [string, ...],\n'
        '  "seed_similarity": number in [0,1],\n'
        '  "mimic_risk": "low" | "medium" | "high",\n'
        '  "confidence": number in [0,1],\n'
        '  "reason": string (1-2 sentences)\n'
        "}\n"
    )


def _gt_revelation_prompt(coord, prob, gt_verdict, original_vlm_label) -> str:
    """For disagreements: tell the model the ground truth and ask for corrected reasoning."""
    target_decision = "accept" if gt_verdict == "confirm" else "reject"
    flip_hint = (
        "Specifically: the candidate IS a real additional metastasis (verified by an expert "
        "neuroradiologist's voxel-level annotation). Look harder at the candidate tiles for "
        "signs of enhancement, FLAIR correlate, or rounded morphology that you may have missed."
        if target_decision == "accept" else
        "Specifically: the candidate is NOT a real additional metastasis (no overlap with any "
        "expert-annotated lesion within 10 voxels). Look for mimic features: vessel-like linear "
        "morphology, CSF-following intensity, motion artefact, or implausible location."
    )
    return (
        _verifier_user_prompt(coord, prob)
        + "\n"
        "GROUND-TRUTH HINT (this example is from a labelled training set; the correct answer is\n"
        f"known): the correct decision is \"{target_decision}\".\n"
        f"{flip_hint}\n"
        "Produce the structured JSON output that JUSTIFIES this correct decision based on what "
        "is actually visible in the panels. Do not just restate the GT - cite specific visual "
        "evidence."
    )


def _build_user_content(image_b64: str, prompt: str) -> list[dict]:
    return [
        {"type": "image",
         "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
        {"type": "text", "text": prompt},
    ]


def _structured_target_from_vlm_label(vlm_label: dict) -> dict:
    """Convert cached vlm_label fields back into the structured JSON target."""
    decision_map = {"confirm": "accept", "reject": "reject", "uncertain": "uncertain"}
    return {
        "decision": decision_map.get(vlm_label.get("verdict") or "uncertain", "uncertain"),
        "evidence_for": list(vlm_label.get("evidence_for") or []),
        "evidence_against": list(vlm_label.get("evidence_against") or []),
        "seed_similarity": float(vlm_label.get("seed_similarity") or 0.0),
        "mimic_risk": vlm_label.get("mimic_risk") or "low",
        "confidence": float(vlm_label.get("confidence") or 0.5),
        "reason": (vlm_label.get("rationale") or "")[:300],
    }


def _call_corrector(client, image_b64: str, prompt: str) -> dict:
    resp = client.messages.create(
        model=CORRECTOR_MODEL,
        max_tokens=CORRECTOR_MAX_TOKENS,
        temperature=CORRECTOR_TEMPERATURE,
        messages=[{"role": "user", "content": _build_user_content(image_b64, prompt)}],
    )
    text = "".join(getattr(b, "text", "") for b in resp.content)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"decision": "uncertain", "confidence": 0.5,
                "evidence_for": [], "evidence_against": [],
                "seed_similarity": 0.0, "mimic_risk": "low",
                "reason": text[:200]}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {"decision": "uncertain", "confidence": 0.5,
                "evidence_for": [], "evidence_against": [],
                "seed_similarity": 0.0, "mimic_risk": "low",
                "reason": text[:200]}
    return obj


def _make_record(image_b64: str, user_prompt: str, target: dict, meta: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": _build_user_content(image_b64, user_prompt)},
            {"role": "assistant",
             "content": [{"type": "text", "text": json.dumps(target, ensure_ascii=False)}]},
        ],
        "meta": meta,
    }


def _gt_verdict_of(rec: dict) -> str:
    return rec["gt_label"]["verdict"]


def _split_study_grouped(records: list[dict], val_frac: float, seed: int = 42):
    """Group by study_id so a single study cannot bleed across train/val."""
    rng = random.Random(seed)
    studies = sorted({r["study_id"] for r in records})
    rng.shuffle(studies)
    n_val = max(1, int(round(len(studies) * val_frac))) if val_frac > 0 else 0
    val_studies = set(studies[:n_val])
    train = [r for r in records if r["study_id"] not in val_studies]
    val = [r for r in records if r["study_id"] in val_studies]
    return train, val, val_studies


def _balance_classes(records: list[dict], neg_pos_ratio: float, seed: int = 42):
    rng = random.Random(seed)
    pos = [r for r in records if _gt_verdict_of(r) == "confirm"]
    neg = [r for r in records if _gt_verdict_of(r) == "reject"]
    if not pos or not neg:
        return records
    target_neg = int(round(len(pos) * neg_pos_ratio))
    if len(neg) > target_neg:
        rng.shuffle(neg)
        neg = neg[:target_neg]
    out = pos + neg
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dirs", nargs="+", required=True,
                    help="One or more directories, each with metadata.jsonl + images/.")
    ap.add_argument("--out", default="data/sft_v1_finalized")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--neg-pos-ratio", type=float, default=2.0)
    ap.add_argument("--limit-disagreements", type=int, default=0,
                    help="Cap the number of disagreements re-prompted (0 = all). "
                         "Useful for cost control.")
    ap.add_argument("--no-correct-disagreements", action="store_true",
                    help="Skip Opus/Sonnet correction; use GT verdict + minimal reason.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each record carries its own input dir so image paths resolve correctly
    raw: list[tuple[Path, dict]] = []
    for d_str in args.in_dirs:
        d = Path(d_str)
        meta_path = d / "metadata.jsonl"
        if not meta_path.exists():
            print(f"  [skip] {d}: no metadata.jsonl")
            continue
        n_d = 0
        for line in meta_path.open():
            r = json.loads(line)
            raw.append((d, r))
            n_d += 1
        print(f"Loaded {n_d} records from {meta_path}")
    print(f"Total: {len(raw)} records across {len(args.in_dirs)} dirs")

    # Three buckets:
    #   - proposal_only: no VLM verdict at all (vlm.verdict is None) -> gt_only target
    #   - agreement: vlm.verdict == gt.verdict -> imitation target
    #   - disagreement: vlm.verdict != gt.verdict -> corrector target (or gt_only)
    proposal_only = [t for t in raw if t[1]["vlm_label"]["verdict"] is None]
    has_vlm = [t for t in raw if t[1]["vlm_label"]["verdict"] is not None]
    agreements = [t for t in has_vlm if t[1]["agreement"]]
    disagreements = [t for t in has_vlm if not t[1]["agreement"]]
    print(f"  proposal_only: {len(proposal_only)}")
    print(f"  agreements:    {len(agreements)}")
    print(f"  disagreements: {len(disagreements)}")

    def _gt_only_target(r):
        return {
            "decision": "accept" if r["gt_label"]["verdict"] == "confirm" else "reject",
            "evidence_for": [], "evidence_against": [],
            "seed_similarity": 0.0, "mimic_risk": "low",
            "confidence": 0.95,
            "reason": "Ground-truth label; reasoning omitted.",
        }

    def _wrap(in_dir: Path, r: dict, philosophy: str, target: dict) -> dict:
        return {
            "in_dir": str(in_dir),
            "image_path": r["image_path"],
            "coord_zyx": r["coord_zyx"],
            "detector_prob": r["detector_prob"],
            "study_id": r["study_id"],
            "gt_label": r["gt_label"],
            "vlm_label": r["vlm_label"],
            "philosophy": philosophy,
            "target": target,
        }

    finalized: list[dict] = []

    # Bucket 1: agreements -> imitate VLM output
    for in_dir, r in agreements:
        target = _structured_target_from_vlm_label(r["vlm_label"])
        if r["gt_label"]["verdict"] == "confirm":
            target["decision"] = "accept"
        elif r["gt_label"]["verdict"] == "reject":
            target["decision"] = "reject"
        finalized.append(_wrap(in_dir, r, "agreement_imitation", target))

    # Bucket 2: proposal-only -> GT label, minimal reason (no VLM to imitate)
    for in_dir, r in proposal_only:
        finalized.append(_wrap(in_dir, r, "proposal_only_gt", _gt_only_target(r)))

    # Bucket 3: disagreements -> corrector model (or gt_only if --no-correct)
    if disagreements:
        if args.no_correct_disagreements:
            for in_dir, r in disagreements:
                finalized.append(_wrap(in_dir, r, "disagreement_gt_only", _gt_only_target(r)))
        else:
            import anthropic
            client = anthropic.Anthropic()
            n = args.limit_disagreements or len(disagreements)
            print(f"Re-prompting {n} disagreement(s) with corrector model "
                  f"({CORRECTOR_MODEL}) ... this calls the API.")
            for i, (in_dir, r) in enumerate(disagreements[:n]):
                img_path = in_dir / r["image_path"]
                if not img_path.exists():
                    print(f"  [{i+1}/{n}] missing image {img_path}, skipping")
                    continue
                img_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
                prompt = _gt_revelation_prompt(
                    r["coord_zyx"], r["detector_prob"],
                    r["gt_label"]["verdict"], r["vlm_label"],
                )
                try:
                    target = _call_corrector(client, img_b64, prompt)
                except Exception as e:
                    print(f"  [{i+1}/{n}] {r['study_id']}#{r['candidate_index_in_study']}: {e}")
                    continue
                if r["gt_label"]["verdict"] == "confirm":
                    target["decision"] = "accept"
                else:
                    target["decision"] = "reject"
                finalized.append(_wrap(in_dir, r, "disagreement_corrected", target))
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{n}] corrected so far")
                time.sleep(0.05)

    # Train/val split (study-grouped)
    train_recs, val_recs, val_studies = _split_study_grouped(finalized, args.val_frac)
    print(f"Study-grouped split: {len(train_recs)} train / {len(val_recs)} val "
          f"(val studies: {sorted(val_studies)})")

    # Class balance only for train
    pre_balance_n = len(train_recs)
    train_recs = _balance_classes(train_recs, args.neg_pos_ratio)
    print(f"After balancing (neg:pos = {args.neg_pos_ratio}): "
          f"{len(train_recs)} train (was {pre_balance_n})")

    # Emit messages-format JSONL
    train_out = out_dir / "train.jsonl"
    val_out = out_dir / "val.jsonl"
    with train_out.open("w") as f:
        for r in train_recs:
            img_b64 = base64.b64encode((Path(r["in_dir"]) / r["image_path"]).read_bytes()).decode("ascii")
            user_prompt = _verifier_user_prompt(r["coord_zyx"], r["detector_prob"])
            meta = {k: r[k] for k in ("study_id", "gt_label", "vlm_label", "philosophy",
                                      "coord_zyx", "detector_prob")}
            f.write(json.dumps(_make_record(img_b64, user_prompt, r["target"], meta)) + "\n")
    with val_out.open("w") as f:
        for r in val_recs:
            img_b64 = base64.b64encode((Path(r["in_dir"]) / r["image_path"]).read_bytes()).decode("ascii")
            user_prompt = _verifier_user_prompt(r["coord_zyx"], r["detector_prob"])
            meta = {k: r[k] for k in ("study_id", "gt_label", "vlm_label", "philosophy",
                                      "coord_zyx", "detector_prob")}
            f.write(json.dumps(_make_record(img_b64, user_prompt, r["target"], meta)) + "\n")

    print()
    print(f"Wrote {train_out} ({train_out.stat().st_size//1024} KB) "
          f"and {val_out} ({val_out.stat().st_size//1024} KB)")
    print()
    print("Train philosophy mix:")
    for k, v in Counter(r["philosophy"] for r in train_recs).most_common():
        print(f"  {k}: {v}")
    print()
    print("Train GT label mix:")
    for k, v in Counter(r["gt_label"]["verdict"] for r in train_recs).most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
