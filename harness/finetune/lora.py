"""LoRA fine-tuning CLI with the same prompt format as zero-shot; loss on label tokens only.

    python -m harness.finetune.lora --model qwen --task sentiment --train clean
    python -m harness.finetune.lora --model qwen --task sentiment --train clean \
        --train-perturbed typo:0.10,dialect:all            # RQ3 augmented variant

The best checkpoint by validation macro-F1 is saved to
outputs/adapters/{model}/{task}/{train_variant}/.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time

import numpy as np
import torch

from harness.config import DIALECTS, Condition, load_tasks
from harness.data.build_sets import subprocess_dialect
from harness.data.loaders import load_tweeteval, perturb_rows, stratified_subsample
from harness.metrics import macro_f1
from harness.run import adapter_dir

log = logging.getLogger("harness.finetune")

# Attention + MLP projections of the language model (excludes vision/audio towers).
TARGET_MODULES = r"^(?!.*(vision|audio)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def parse_perturbed(spec: str | None) -> list[Condition]:
    """'typo:0.10,dialect:all' -> conditions ('dialect:all' expands to every dialect)."""
    conds: list[Condition] = []
    for part in filter(None, (spec or "").split(",")):
        if part.strip() == "dialect:all":
            conds += [Condition("dialect", dialect=d) for d in DIALECTS]
        else:
            conds.append(Condition.parse(part.strip()))
    return conds


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(clf, rows: list[dict], task: str, max_len: int) -> tuple[list[tuple[list[int], list[int]]], int]:
    """(input_ids, labels) pairs with -100 on prompt tokens; drops examples over max_len."""
    lab = dict(zip(clf.tasks[task].label_names, clf.label_ids(task)))
    out, skipped = [], 0
    for r in rows:
        p, y = clf.encode_prompt(r["text"], task), lab[r["label"]]
        if len(p) + len(y) > max_len:
            skipped += 1
            continue
        out.append((p + y, [-100] * len(p) + y))
    return out, skipped


def collate(batch, pad_id: int, device) -> dict[str, torch.Tensor]:
    """Right-pad a micro-batch (no generation, so default position ids are correct)."""
    T = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), T), -100, dtype=torch.long)
    att = torch.zeros((len(batch), T), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)], lab[i, :len(y)], att[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    return {"input_ids": ids.to(device), "labels": lab.to(device), "attention_mask": att.to(device)}


def evaluate(clf, rows: list[dict], task: str, batch_size: int) -> float:
    clf.model.eval()
    preds = clf.predict([r["text"] for r in rows], task, "zeroshot", batch_size=batch_size)
    clf.model.train()
    return macro_f1([{**r, "pred": p["label"]} for r, p in zip(rows, preds)])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--train", default="clean", choices=["clean"], help="Base training data.")
    ap.add_argument("--train-perturbed", help="Augment with perturbed copies, e.g. typo:0.10,dialect:all")
    ap.add_argument("--variant", help="Adapter name (default: clean, or augmented with --train-perturbed).")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--effective-batch", type=int, default=32)
    ap.add_argument("--val-size", type=int, default=500, help="Held out from train, stratified.")
    ap.add_argument("--max-train", type=int, help="Stratified cap on clean train size (quick runs).")
    ap.add_argument("--eval-every", type=int, default=0, help="Eval every N optimizer steps (0 = per epoch).")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA.")
    ap.add_argument("--dialect-python", help="Interpreter with Multi-VALUE for dialect augmentation.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import get_linear_schedule_with_warmup

    from harness.models.wrapper import LLMClassifier, pick_dtype

    variant = args.variant or ("augmented" if args.train_perturbed else "clean")
    out = adapter_dir(args.output_dir, args.model, args.task, variant)
    if (out / "adapter_config.json").exists() and not args.overwrite:
        log.info("adapter exists at %s; skipping (use --overwrite)", out)
        return
    seed_everything(args.seed)
    task = load_tasks()[args.task]

    # ---- data: hold out val from train, then optionally augment train
    rows = load_tweeteval(task, "train")
    val =stratified_subsample(rows, args.val_size, args.seed)
    val_ids = {r["id"] for r in val}
    train = [r for r in rows if r["id"] not in val_ids]
    train = stratified_subsample(train, args.max_train, args.seed)
    dialect_fn = subprocess_dialect(args.dialect_python) if args.dialect_python else None
    aug = [r for c in parse_perturbed(args.train_perturbed)
           for r in perturb_rows(train, c, args.seed, dialect_fn=dialect_fn)]
    train_all = train + aug
    log.info("train: %d clean + %d perturbed, val: %d", len(train), len(aug), len(val))

    # ---- model
    clf = LLMClassifier(args.model, load_4bit=args.load_4bit)
    model = clf.model
    model.config.use_cache = False
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    clf.model, clf.adapter_path = model, "training"

    data, skipped = encode(clf, train_all, args.task, args.max_len)
    log.info("encoded %d examples (%d over max_len skipped)", len(data), skipped)

    accum = max(1, args.effective_batch // args.micro_batch)
    steps_per_epoch = math.ceil(len(data) / (args.micro_batch * accum))
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * total), total)
    dtype = pick_dtype()
    device = next(model.parameters()).device
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

    # LoRA B is zero-initialized, so step 0 equals the zero-shot model.
    zeroshot = evaluate(clf, val, args.task, args.eval_batch)
    history = [{"step": 0, "val_macro_f1": zeroshot}]
    log.info("zero-shot val macro-F1 = %.4f", zeroshot)
    best, best_step = -1.0, 0

    def maybe_save(step: int) -> None:
        """Keep the best trained checkpoint by val macro-F1."""
        nonlocal best, best_step
        f1 = evaluate(clf, val, args.task, args.eval_batch)
        history.append({"step": step, "val_macro_f1": f1})
        log.info("step %d: val macro-F1 = %.4f (best so far %.4f, zero-shot %.4f)", step, f1, best, zeroshot)
        if f1 > best:
            best, best_step = f1, step
            model.save_pretrained(out)
            log.info("saved best adapter -> %s", out)

    gen = torch.Generator().manual_seed(args.seed)
    step, t0 = 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        order = torch.randperm(len(data), generator=gen).tolist()
        micro = [order[i:i + args.micro_batch] for i in range(0, len(order), args.micro_batch)]
        for mi, idx in enumerate(micro):
            batch = collate([data[i] for i in idx], clf.tokenizer.pad_token_id, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                loss = model(**batch).loss / accum
            scaler.scale(loss).backward()
            if (mi + 1) % accum == 0 or mi == len(micro) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 10 == 0:
                    log.info("epoch %d step %d/%d loss %.4f (%.1fs)", epoch, step, total,
                             loss.item() * accum, time.time() - t0)
                if args.eval_every and step % args.eval_every == 0:
                    maybe_save(step)
        if not args.eval_every or step % args.eval_every:
            maybe_save(step)

    meta = {"model": args.model, "model_id": clf.model_id, "task": args.task, "variant": variant,
            "train_perturbed": args.train_perturbed, "n_train": len(data), "n_val": len(val),
            "skipped_over_max_len": skipped, "zeroshot_val_macro_f1": zeroshot,
            "best_val_macro_f1": best, "best_step": best_step, "history": history,
            "hparams": {k: v for k, v in vars(args).items() if k not in ("model", "task")}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("best val macro-F1 %.4f at step %d (zero-shot %.4f)", best, best_step, zeroshot)


if __name__ == "__main__":
    main()
