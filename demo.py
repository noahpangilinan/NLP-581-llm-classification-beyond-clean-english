"""Console demo: perturb a sentence and compare clean vs perturbed predictions.

    python demo.py --model qwen --task sentiment [--adapter PATH] [--all-models] [--load-4bit]

Commands at the prompt: type a sentence, `:sample` for a random test tweet, `:quit` to exit.
"""

from __future__ import annotations

import argparse
import random

from harness.config import load_models, load_tasks
from harness.perturb.typo import typo

try:
    from rich.console import Console
    from rich.table import Table
    console: Console | None = Console()
except ImportError:
    console = None

PERTURBATIONS = ["typo", "natural_typo", "dialect", "none"]


def out(msg: str = "") -> None:
    # markup=False: tweets may contain [brackets] that rich would parse as tags.
    console.print(msg, markup=False, highlight=False) if console else print(msg)


def bar(p: float, width: int = 24) -> str:
    n = round(p * width)
    return "█" * n + "·" * (width - n)


def ask(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val or default


def perturb(text: str, kind: str, severity: float, seed: int) -> str:
    if kind == "typo":
        return typo(text, severity, seed)
    if kind == "natural_typo":
        from harness.perturb.natural_typo import load_typo_map, natural_typo
        return natural_typo(text, severity, seed, load_typo_map())
    if kind == "dialect":
        from harness.perturb.dialect import dialect
        name = ask("dialect (aave / indian_english / singlish)", "aave")
        return dialect(text, name, seed)
    return text


def show(results: dict[str, tuple[dict, dict]], labels: list[str]) -> None:
    """results: model -> (clean_pred, perturbed_pred)."""
    if console:
        t = Table(show_header=True, header_style="bold")
        t.add_column("model")
        t.add_column("label")
        t.add_column("clean")
        t.add_column("perturbed")
        for m, (c, p) in results.items():
            for i, l in enumerate(labels):
                t.add_row(m if i == 0 else "", l,
                          f"{bar(c['probs'][l])} {c['probs'][l]:.2f}{' ◀' if c['label'] == l else ''}",
                          f"{bar(p['probs'][l])} {p['probs'][l]:.2f}{' ◀' if p['label'] == l else ''}")
            t.add_section()
        console.print(t)
        return
    for m, (c, p) in results.items():
        print(f"\n{m}: clean -> {c['label']}, perturbed -> {p['label']}")
        for l in labels:
            print(f"  {l:14s} clean {bar(c['probs'][l])} {c['probs'][l]:.2f}   "
                  f"perturbed {bar(p['probs'][l])} {p['probs'][l]:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--task", default="sentiment", choices=sorted(load_tasks()))
    ap.add_argument("--adapter", help="LoRA adapter path (applies to --model only).")
    ap.add_argument("--all-models", action="store_true", help="Load every model in models.yaml side by side.")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    from harness.models.wrapper import LLMClassifier

    keys = list(load_models()) if args.all_models else [args.model]
    clfs = {}
    for k in keys:
        out(f"loading {k} ...")
        clfs[k] = LLMClassifier(k, adapter_path=args.adapter if k == args.model else None,
                                load_4bit=args.load_4bit)
    labels = load_tasks()[args.task].label_names
    samples: list[dict] = []
    rng = random.Random(args.seed)
    turn = 0

    while True:
        try:
            text = input("\ntext (:sample, :quit)> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in (":quit", ":q"):
            break
        if text == ":sample":
            if not samples:
                from harness.data.loaders import load_tweeteval
                samples = load_tweeteval(load_tasks()[args.task], "test")
            s = rng.choice(samples)
            text = s["text"]
            out(f"sample (gold: {s['label']}): {text}")
        if not text:
            continue
        kind = ask(f"perturbation ({' / '.join(PERTURBATIONS)})", "typo")
        severity = float(ask("severity p", "0.20")) if kind in ("typo", "natural_typo") else 0.0
        turn += 1
        try:
            pert = perturb(text, kind, severity, f"{args.seed}:{turn}")
        except (ImportError, FileNotFoundError, ValueError) as e:
            out(f"cannot apply {kind}: {e}")
            continue
        out(f"\nclean:     {text}\nperturbed: {pert}")
        results = {}
        for k, clf in clfs.items():
            c, p = clf.predict([text, pert], args.task, "zeroshot", batch_size=2)
            results[k] = (c, p)
        show(results, labels)


if __name__ == "__main__":
    main()
