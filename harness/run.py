"""Grid runner: model x task x regime x condition -> prediction JSONL files.

    python -m harness.run --grid configs/grid_main.yaml [--dry-run] [--overwrite]

Outputs go to outputs/predictions/{model}/{task}/{regime}/{condition}.jsonl.
Runs whose output exists are skipped, so an interrupted grid can be resumed.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from harness.config import Condition, GridConfig, RegimeConfig, load_grid, load_tasks, resolve

log = logging.getLogger("harness.run")


@dataclass(frozen=True)
class Run:
    model: str
    task: str
    regime: RegimeConfig
    condition: Condition

    def out_path(self, output_dir: str | Path) -> Path:
        return (resolve(output_dir) / "predictions" / self.model / self.task
                / self.regime.slug / f"{self.condition.name}.jsonl")

    def adapter_path(self, output_dir: str | Path) -> Path | None:
        if self.regime.name != "finetune":
            return None
        return adapter_dir(output_dir, self.model, self.task, self.regime.train_variant)


def adapter_dir(output_dir: str | Path, model: str, task: str, variant: str) -> Path:
    return resolve(output_dir) / "adapters" / model / task / variant


def expand(grid: GridConfig) -> list[Run]:
    """Expand the grid. Order groups runs by model, then adapter, to minimize reloads."""
    return [Run(m, t, r, c) for m in grid.models for r in grid.regimes
            for t in grid.tasks for c in grid.conditions_for(t)]


def predict_with_oom_retry(clf, texts: list[str], task: str, regime: str, demos, batch_size: int,
                           label: str) -> tuple[list[dict], int]:
    """Predict in chunks; on CUDA OOM halve the batch size and retry the chunk."""
    import torch

    out: list[dict] = []
    bs, i, t0 = batch_size, 0, time.time()
    while i < len(texts):
        chunk = texts[i:i + bs * 4]
        try:
            out += clf.predict(chunk, task, regime, demos=demos, batch_size=bs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs //= 2
            log.warning("CUDA OOM; retrying with batch_size=%d", bs)
            continue
        i += len(chunk)
        rate = i / max(time.time() - t0, 1e-9)
        log.info("  %s: %d/%d (%.1f ex/s)", label, i, len(texts), rate)
    return out, bs


def run_grid(grid: GridConfig, runs: list[Run], seed: int) -> None:
    import torch

    from harness.data.loaders import get_condition_rows, load_tweeteval
    from harness.data.schema import write_jsonl
    from harness.models.fewshot import sample_demos
    from harness.models.wrapper import LLMClassifier

    tasks = load_tasks()
    train_cache: dict[str, list[dict]] = {}
    for model_key, model_runs in groupby(runs, key=lambda r: r.model):
        model_runs = list(model_runs)
        log.info("loading model %s (%d runs)", model_key, len(model_runs))
        clf = LLMClassifier(model_key, load_4bit=grid.load_4bit)
        bs = grid.batch_size
        for run in model_runs:
            label = f"{run.model}/{run.task}/{run.regime.slug}/{run.condition.name}"
            adapter = run.adapter_path(grid.output_dir)
            if adapter is not None and not (adapter / "adapter_config.json").exists():
                log.warning("skip %s: no adapter at %s (train it with harness.finetune.lora)", label, adapter)
                continue
            try:
                rows = get_condition_rows(tasks[run.task], run.condition, data_dir=grid.data_dir,
                                          seed=seed, limit=grid.max_examples)
            except (FileNotFoundError, ImportError) as e:
                log.warning("skip %s: data unavailable (%s)", label, e)
                continue
            clf.set_adapter(str(adapter) if adapter else None)

            demos = None
            if run.regime.name == "fewshot":
                if run.task not in train_cache:
                    train_cache[run.task] = load_tweeteval(tasks[run.task], "train")
                demos = sample_demos(train_cache[run.task], run.regime.k, seed,
                                     labels=tasks[run.task].label_names)

            log.info("run %s (%d examples)", label, len(rows))
            preds, bs = predict_with_oom_retry(clf, [r["text"] for r in rows], run.task,
                                               run.regime.name, demos, bs, label)
            meta = {"model_id": clf.model_id, "regime": run.regime.slug, "k": run.regime.k,
                    "prompt_version": tasks[run.task].prompt_version, "seed": seed,
                    "adapter": str(adapter) if adapter else None,
                    "demo_ids": [d["id"] for d in demos] if demos else None}
            write_jsonl(({**r, "pred": p["label"], "probs": p["probs"], **meta}
                         for r, p in zip(rows, preds)), run.out_path(grid.output_dir))
        del clf
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="configs/grid_smoke.yaml", help="Grid YAML (default: smoke grid).")
    ap.add_argument("--dry-run", action="store_true", help="Print the expanded grid and run count.")
    ap.add_argument("--overwrite", action="store_true", help="Re-run even if outputs exist.")
    ap.add_argument("--models", nargs="+", help="Restrict to these model keys.")
    ap.add_argument("--load-4bit", action="store_true", help="Load models in 4-bit (overrides grid).")
    ap.add_argument("--batch-size", type=int, help="Override grid batch_size.")
    ap.add_argument("--max-examples", type=int, help="Override grid max_examples.")
    args = ap.parse_args(argv)

    grid = load_grid(args.grid)
    grid.load_4bit = grid.load_4bit or args.load_4bit
    grid.batch_size = args.batch_size or grid.batch_size
    grid.max_examples = args.max_examples or grid.max_examples
    if args.models:
        grid.models = [m for m in grid.models if m in args.models]

    out_dir = resolve(grid.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "run.log")])
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    runs = expand(grid)
    todo = [r for r in runs if args.overwrite or not r.out_path(grid.output_dir).exists()]
    if args.dry_run:
        for r in runs:
            status = "todo" if r in todo else "done"
            print(f"[{status}] {r.model:8s} {r.task:10s} {r.regime.slug:20s} {r.condition.name}")
        print(f"\n{len(runs)} runs in grid, {len(todo)} to run, {len(runs) - len(todo)} already done.")
        return
    log.info("%d runs in grid, %d to run (%d skipped: output exists)", len(runs), len(todo), len(runs) - len(todo))
    if todo:
        run_grid(grid, todo, grid.seed)
    log.info("done")


if __name__ == "__main__":
    main()
