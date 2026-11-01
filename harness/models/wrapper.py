"""LLMClassifier: classify by scoring each candidate label's log-probability.

The prompt is rendered with the model's chat template plus the assistant-turn
prefix; each label string is appended and its token log-probs are summed.
A softmax over labels gives `probs`; the argmax is the prediction.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn.functional as F

from harness.config import ModelConfig, load_models, load_tasks
from harness.models.prompts import build_messages

log = logging.getLogger(__name__)

FALLBACK_TEMPLATE_SUFFIX = "Assistant:"


def pick_dtype() -> torch.dtype:
    """bf16 on GPUs that support it, fp16 on other GPUs, fp32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model_and_tokenizer(hf_id: str, load_4bit: bool = False, dtype: torch.dtype | None = None):
    """Load a causal LM (falling back to image-text-to-text for multimodal models)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    dtype = dtype or pick_dtype()
    tok = AutoTokenizer.from_pretrained(hf_id, token=token)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        try:  # multimodal models sometimes keep the template on the processor only
            from transformers import AutoProcessor
            tok.chat_template = getattr(AutoProcessor.from_pretrained(hf_id, token=token), "chat_template", None)
        except Exception:
            pass

    kwargs: dict[str, Any] = {"dtype": dtype, "token": token}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(hf_id, **kwargs)
    model.eval()
    return model, tok


def _forward_last_logits(model, input_ids, attention_mask, keep: int) -> torch.Tensor:
    """Logits for the last `keep` positions only (avoids materializing B x T x V)."""
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, logits_to_keep=keep, use_cache=False)
    except TypeError:
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, use_cache=False)
    return out.logits[:, -keep:, :]


def score_candidates(model, prompts: list[list[int]], cands: list[list[int]], pad_id: int,
                     batch_size: int = 16) -> torch.Tensor:
    """Return [N, C] summed log-probs of each candidate continuation after each prompt.

    Sequences are left-padded, so every candidate ends at the last position and
    only the last max_len(cand)+1 logits are needed.
    """
    device = next(model.parameters()).device
    C, K = len(cands), max(map(len, cands)) + 1
    # Targets aligned to the kept logits slice [-K:-1]; candidate of length m fills the last m slots.
    tgt = torch.zeros(C, K - 1, dtype=torch.long)
    mask = torch.zeros(C, K - 1, dtype=torch.bool)
    for j, c in enumerate(cands):
        tgt[j, K - 1 - len(c):] = torch.tensor(c)
        mask[j, K - 1 - len(c):] = True
    tgt, mask = tgt.to(device), mask.to(device)

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    scores = torch.empty(len(prompts), C)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        seqs = [prompts[i] + c for i in idx for c in cands]
        T = max(map(len, seqs))
        ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for r, seq in enumerate(seqs):
            ids[r, T - len(seq):] = torch.tensor(seq)
            att[r, T - len(seq):] = 1
        logits = _forward_last_logits(model, ids.to(device), att.to(device), K)
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)                 # [B*C, K-1, V]
        B = len(idx)
        t, m = tgt.repeat(B, 1), mask.repeat(B, 1)
        tok_lp = lp.gather(-1, t.unsqueeze(-1)).squeeze(-1) * m
        scores[idx] = tok_lp.sum(-1).view(B, C).cpu()
    return scores


class LLMClassifier:
    """Zero-shot / few-shot / fine-tuned classifier via label scoring."""

    def __init__(self, model_key: str, adapter_path: str | None = None, load_4bit: bool = False):
        models = load_models()
        self.cfg: ModelConfig = models.get(model_key) or ModelConfig(key=model_key, hf_id=model_key)
        self.model_key = model_key
        self.model_id = self.cfg.hf_id
        self.tasks = load_tasks()
        self.model, self.tokenizer = load_model_and_tokenizer(self.cfg.hf_id, load_4bit=load_4bit)
        self.has_template = bool(self.tokenizer.chat_template)
        self.adapter_path: str | None = None
        self._label_ids: dict[str, list[list[int]]] = {}
        if adapter_path:
            self.set_adapter(adapter_path)

    # ------------------------------------------------------------ adapters

    def set_adapter(self, path: str | None) -> None:
        """Attach a LoRA adapter (or detach with None), keeping the base model loaded."""
        if path == self.adapter_path:
            return
        from peft import PeftModel

        if isinstance(self.model, PeftModel):
            self.model = self.model.unload()
        if path:
            self.model = PeftModel.from_pretrained(self.model, path)
        self.model.eval()
        self.adapter_path = path

    # ------------------------------------------------------------ encoding

    def render(self, messages: list[dict[str, str]]) -> str:
        """Render messages with the chat template and the assistant-turn prefix."""
        if self.has_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.cfg.chat_template_kwargs)
        # Plain fallback for base models without a template (e.g. tiny test models).
        turns = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages]
        return "\n\n".join(turns) + "\n\n" + FALLBACK_TEMPLATE_SUFFIX

    def encode_prompt(self, text: str, task: str, demos: list[dict] | None = None) -> list[int]:
        tc = self.tasks[task]
        prompt = self.render(build_messages(tc.prompt_version, text, tc.labels, demos))
        return self.tokenizer(prompt, add_special_tokens=not self.has_template).input_ids

    def label_ids(self, task: str) -> list[list[int]]:
        """Token ids of each verbalizer, in the order of tasks.yaml labels."""
        if task not in self._label_ids:
            prefix = "" if self.has_template else " "
            self._label_ids[task] = [self.tokenizer(prefix + v, add_special_tokens=False).input_ids
                                     for v in self.tasks[task].labels.values()]
        return self._label_ids[task]

    # ------------------------------------------------------------ predict

    @torch.inference_mode()
    def predict(self, texts: list[str], task: str, regime: str,
                demos: list[dict] | None = None, batch_size: int = 16) -> list[dict]:
        """Returns [{"label": str, "probs": {label: float}}]"""
        if regime == "fewshot" and not demos:
            raise ValueError("fewshot regime needs demos")
        use_demos = demos if regime == "fewshot" else None
        prompts = [self.encode_prompt(t, task, use_demos) for t in texts]
        scores = score_candidates(self.model, prompts, self.label_ids(task),
                                  self.tokenizer.pad_token_id, batch_size)
        probs = scores.softmax(-1)
        names = self.tasks[task].label_names
        return [{"label": names[int(p.argmax())],
                 "probs": {n: round(float(x), 6) for n, x in zip(names, p)}} for p in probs]
