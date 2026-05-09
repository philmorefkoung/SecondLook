"""VLM verifier tool: structured second-finding judgement.

Backends accept either a raw list of panels OR an EvidenceCard (preferred).
The VerifierVerdict carries the full structured-output schema:
  decision (confirm | reject | uncertain), confidence, evidence_for,
  evidence_against, seed_similarity, mimic_risk, reason.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Protocol, Any
import base64
import io

import numpy as np


@dataclass
class VerifierVerdict:
    label: Literal["confirm", "reject", "uncertain"]
    confidence: float
    rationale: str
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    seed_similarity: float = 0.0
    mimic_risk: Literal["low", "medium", "high"] = "low"


class VLMBackend(Protocol):
    def verify(self, panels: list[np.ndarray], context: dict[str, Any]) -> VerifierVerdict: ...


class VerifierTool:
    def __init__(self, backend: VLMBackend):
        self.backend = backend

    def verify(self, panels, *, modalities, candidate_meta) -> VerifierVerdict:
        ctx = {"modalities": list(modalities), **candidate_meta}
        return self.backend.verify(panels, ctx)


# ---------- Heuristic stub (deterministic, no network) ----------

class HeuristicVLM:
    """Inspects centred multi-modal tiles and returns a structured verdict.

    Uses very simple visual heuristics: enhancement at the tile centre
    (T1post - T1pre) drives the decision; eccentricity-from-centre of the
    high-intensity blob drives mimic_risk (elongated => vessel-like).
    Designed for unit tests; not clinically valid.
    """
    def __init__(self, enhancement_thr: float = 0.3):
        self.thr = enhancement_thr

    def verify(self, panels, context):
        mods = context.get("modalities", [])
        try:
            t1pre = panels[mods.index("T1pre")]
            t1post = panels[mods.index("T1post")]
        except (ValueError, IndexError):
            return VerifierVerdict("uncertain", 0.5, "missing T1pre/T1post panels")
        cy, cx = t1post.shape[0] // 2, t1post.shape[1] // 2
        delta = float(t1post[cy, cx] - t1pre[cy, cx])
        if delta >= self.thr:
            return VerifierVerdict(
                label="confirm", confidence=min(1.0, 0.5 + delta),
                rationale=f"enhancement delta={delta:.2f}",
                evidence_for=["centre enhancement vs T1pre"],
                seed_similarity=0.5, mimic_risk="low",
            )
        if delta <= -self.thr:
            return VerifierVerdict(
                label="reject", confidence=min(1.0, 0.5 + abs(delta)),
                rationale=f"hypointense delta={delta:.2f}",
                evidence_against=["hypointensity on T1post vs T1pre"],
                mimic_risk="medium",
            )
        return VerifierVerdict(
            label="uncertain", confidence=0.5,
            rationale=f"low contrast delta={delta:.2f}",
            mimic_risk="medium",
        )


# ---------- Anthropic VLM (composite + structured JSON) ----------

class AnthropicVLM:
    """Sends an evidence-card composite + structured-output prompt.

    Falls back to a panels-only prompt if no card is provided in context.
    Default temperature=0 so the same evidence yields the same verdict across
    runs - which lets us cache verdicts and sweep ranker/scoring configs
    offline without paying for repeated VLM calls.
    """
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 600,
                 temperature: float = 0.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = float(temperature)
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(max_retries=10)
        return self._client

    def verify(self, panels, context):
        card = context.get("evidence_card")
        if card is not None and getattr(card, "composite_png_b64", ""):
            return self._verify_card(card, context)
        return self._verify_panels(panels, context)

    def _verify_card(self, card, context):
        client = self._client_lazy()
        content = [
            {"type": "text", "text": _structured_prompt(context)},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": card.composite_png_b64,
            }},
            {"type": "text", "text": card.metadata_text},
        ]
        resp = client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )
        return _parse_structured(_extract_text(resp))

    def _verify_panels(self, panels, context):
        client = self._client_lazy()
        content = [{"type": "text", "text": _structured_prompt(context, has_seed_strip=False)}]
        for arr, mod in zip(panels, context.get("modalities", [])):
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png",
                           "data": _arr_to_png_b64(arr)},
            })
            content.append({"type": "text", "text": f"^ {mod} (centred on candidate)"})
        resp = client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )
        return _parse_structured(_extract_text(resp))


def _structured_prompt(context: dict[str, Any], has_seed_strip: bool = True) -> str:
    layout = (
        "The single PNG below is laid out as a 4 row x N modality grid:\n"
        "  Row 0: candidate AXIAL crops (modalities in order)\n"
        "  Row 1: candidate CORONAL crops\n"
        "  Row 2: candidate SAGITTAL crops\n"
        "  Row 3: SEED lesion AXIAL crops (for direct comparison)\n"
        if has_seed_strip else
        "Each image is a centred 2D tile of the candidate in one MRI sequence.\n"
    )
    coord = context.get("coord", "?")
    prob = context.get("prob", 0.0)
    return (
        "You are an auditable second-finding verifier for brain MRI metastases.\n"
        "Your job is to judge whether the candidate at the centre of every tile is a real "
        "additional metastasis (not a vessel, CSF, artefact, or duplicate of the seed).\n"
        f"\n{layout}"
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
        "- evidence_for: visual features supporting an additional metastasis "
        "(e.g. \"ring enhancement\", \"FLAIR correlate\", \"smooth round border\").\n"
        "- evidence_against: features against (e.g. \"linear vessel-like morphology\", "
        "\"follows CSF on FLAIR\", \"motion artefact\", \"identical to seed - duplicate\").\n"
        "- seed_similarity: 1.0 = identical phenotype to seed strip, 0.0 = unrelated.\n"
        "- mimic_risk: probability this is a mimic rather than a metastasis.\n"
    )


def _parse_structured(text: str) -> VerifierVerdict:
    import json, re
    label: Literal["confirm", "reject", "uncertain"] = "uncertain"
    conf = 0.5
    evidence_for: list[str] = []
    evidence_against: list[str] = []
    seed_sim = 0.0
    mimic_risk: Literal["low", "medium", "high"] = "low"
    rationale = (text or "").strip()[:300]

    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            d = str(obj.get("decision", "")).lower()
            if d.startswith("acc") or d.startswith("conf"):
                label = "confirm"
            elif d.startswith("rej"):
                label = "reject"
            else:
                label = "uncertain"
            conf = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
            evidence_for = [str(x)[:120] for x in (obj.get("evidence_for") or [])][:8]
            evidence_against = [str(x)[:120] for x in (obj.get("evidence_against") or [])][:8]
            seed_sim = max(0.0, min(1.0, float(obj.get("seed_similarity", 0.0))))
            mr = str(obj.get("mimic_risk", "low")).lower()
            mimic_risk = mr if mr in ("low", "medium", "high") else "low"
            rationale = str(obj.get("reason", rationale))[:300]
        except Exception:
            pass
    return VerifierVerdict(
        label=label, confidence=conf, rationale=rationale,
        evidence_for=evidence_for, evidence_against=evidence_against,
        seed_similarity=seed_sim, mimic_risk=mimic_risk,
    )


def _extract_text(resp) -> str:
    parts = []
    for b in getattr(resp, "content", []) or []:
        t = getattr(b, "text", None)
        if t:
            parts.append(t)
    return "".join(parts)


def _arr_to_png_b64(arr: np.ndarray) -> str:
    from PIL import Image
    a = arr.astype(np.float32)
    a = (a - a.min()) / max(1e-6, (a.max() - a.min()))
    img = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------- Local Qwen-VL backend (base or LoRA-adapted) ----------

class LocalQwenVLM:
    """Local Qwen2.5-VL backend for the verifier - base model + optional LoRA adapter.

    Loads the model once on first verify() call; subsequent calls reuse it.
    Generates the structured-JSON verdict via transformers.generate (greedy
    decoding by default for reproducibility). No network cost.
    """
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        adapter_path: str | None = None,
        max_new_tokens: int = 400,
        temperature: float = 0.0,
        device_map: str = "auto",
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.device_map = device_map
        self._model = None
        self._processor = None

    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(self.base_model)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model, torch_dtype=torch.bfloat16, device_map=self.device_map,
        )
        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(base, self.adapter_path)
        else:
            self._model = base
        self._model.eval()

    def verify(self, panels, context):
        if self._model is None:
            self._load()
        card = context.get("evidence_card")
        if card is not None and getattr(card, "composite_png_b64", ""):
            return self._verify_card(card, context)
        return self._verify_panels(panels, context)

    def _verify_card(self, card, context):
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        img = Image.open(io.BytesIO(base64.b64decode(card.composite_png_b64))).convert("RGB")
        prompt = _structured_prompt(context, has_seed_strip=True) + "\n\n" + card.metadata_text
        messages = [{"role": "user",
                     "content": [{"type": "image", "image": img},
                                  {"type": "text", "text": prompt}]}]
        return self._generate(messages)

    def _verify_panels(self, panels, context):
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        msg_content = [{"type": "text", "text": _structured_prompt(context, has_seed_strip=False)}]
        for arr, mod in zip(panels, context.get("modalities", [])):
            a = arr.astype(np.float32)
            a = (a - a.min()) / max(1e-6, (a.max() - a.min()))
            img = Image.fromarray((a * 255).astype(np.uint8), mode="L").convert("RGB")
            msg_content.append({"type": "image", "image": img})
            msg_content.append({"type": "text", "text": f"^ {mod}"})
        return self._generate([{"role": "user", "content": msg_content}])

    def _generate(self, messages):
        import torch
        from qwen_vl_utils import process_vision_info
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self._processor(text=[text], images=image_inputs,
                                   return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature if self.temperature > 0 else 1.0,
                pad_token_id=self._processor.tokenizer.pad_token_id,
            )
        gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        gen_text = self._processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return _parse_structured(gen_text)
