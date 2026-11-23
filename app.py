from __future__ import annotations

import argparse
import hashlib
import random
import re

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from harness.config import DIALECTS, load_models, load_tasks  # noqa: E402
from harness.perturb.typo import typo  # noqa: E402

PERTURBATIONS = {
    "None (control)": "none",
    "Synthetic typos": "typo",
    "Natural typos (real-world misspellings)": "natural_typo",
    "Dialect (Multi-VALUE)": "dialect",
}
DIALECT_LABELS = {"aave": "AAVE", "indian_english": "Indian English", "singlish": "Singlish"}

CLEAN_COLOR, PERT_COLOR = "#4C78A8", "#F58518"

# classifiers


class StubClassifier:
    """Keyword-based fake model so the UI can be developed and tested without a GPU.

    Typos break the keyword matches, so perturbing a sentence visibly changes the output.
    """

    POS = {"love", "loved", "great", "good", "amazing", "fantastic", "happy", "best", "awesome"}
    NEG = {"hate", "hated", "bad", "awful", "terrible", "worst", "sad", "horrible", "trash"}

    def predict(self, texts, task, regime, demos=None, batch_size=16):
        from harness.config import load_tasks as _lt

        names = _lt()[task].label_names
        out = []
        for t in texts:
            words = set(re.findall(r"[a-z']+", t.lower()))
            score = len(words & self.POS) - len(words & self.NEG)
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            raw = {n: 0.2 + (seed >> (4 * i) & 0xF) / 60 for i, n in enumerate(names)}
            favored = {"positive": score > 0, "negative": score < 0, "offensive": score < 0,
                       "neutral": score == 0, "not_offensive": score >= 0}
            for n in names:
                if favored.get(n):
                    raw[n] += 1.5
            z = sum(raw.values())
            probs = {n: round(v / z, 6) for n, v in raw.items()}
            out.append({"label": max(probs, key=probs.get), "probs": probs})
        return out


_CLF_CACHE: dict[tuple, object] = {}
SETTINGS = {"stub": False, "load_4bit": False}


def get_classifier(model_key: str, adapter: str | None):
    """Load a model once and reuse it across requests."""
    if SETTINGS["stub"]:
        return StubClassifier()
    key = (model_key, adapter or None, SETTINGS["load_4bit"])
    if key not in _CLF_CACHE:
        from harness.models.wrapper import LLMClassifier

        _CLF_CACHE.clear()  
        _CLF_CACHE[key] = LLMClassifier(model_key, adapter_path=adapter or None,
                                        load_4bit=SETTINGS["load_4bit"])
    return _CLF_CACHE[key]


# perturbation


def apply_perturbation(text: str, kind: str, severity: float, dialect_name: str, seed: int, turn: int) -> str:
    seed_str = f"{seed}:{turn}"
    if kind == "typo":
        return typo(text, severity, seed_str)
    if kind == "natural_typo":
        from harness.perturb.natural_typo import load_typo_map, natural_typo

        return natural_typo(text, severity, seed_str, load_typo_map())
    if kind == "dialect":
        from harness.perturb.dialect import dialect

        return dialect(text, dialect_name, seed_str)
    return text


# rendering


def make_chart(labels: list[str], clean: dict, pert: dict):
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=110)
    x = range(len(labels))
    w = 0.38
    cb = ax.bar([i - w / 2 for i in x], [clean["probs"][l] for l in labels], w,
                label="Clean", color=CLEAN_COLOR)
    pb = ax.bar([i + w / 2 for i in x], [pert["probs"][l] for l in labels], w,
                label="Perturbed", color=PERT_COLOR)
    for bars in (cb, pb):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([l.replace("_", " ") for l in labels])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("probability")
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def verdict(clean: dict, pert: dict, changed: bool) -> str:
    c, p = clean["label"], pert["label"]
    cc, pc = clean["probs"][c], pert["probs"][p]
    if not changed:
        return (f"### Text unchanged\nNo perturbation was applied (or it left the text as is). "
                f"Prediction: **{c}** ({cc:.0%}).")
    if c != p:
        return (f"### 🔄 Prediction flipped\n**{c}** ({cc:.0%}) → **{p}** ({pc:.0%}). "
                f"The model changed its answer because of the perturbation.")
    shift = pert["probs"][c] - cc
    return (f"### ✅ Prediction held\nStill **{c}**; confidence moved from {cc:.0%} to "
            f"{pert['probs'][c]:.0%} ({shift:+.0%}).")


def toggle_controls(kind_label: str):
    kind = PERTURBATIONS[kind_label]
    return (gr.update(visible=kind in ("typo", "natural_typo")),
            gr.update(visible=kind == "dialect"))


# handlers

_SAMPLES: dict[str, list[dict]] = {}


def random_sample(task: str):
    try:
        if task not in _SAMPLES:
            from harness.data.loaders import load_tweeteval

            _SAMPLES[task] = load_tweeteval(load_tasks()[task], "test")
        s = random.Random().choice(_SAMPLES[task])
    except Exception as e:  # no network / datasets missing
        raise gr.Error(f"Couldn't load TweetEval samples ({type(e).__name__}: {e}). "
                       "Type your own sentence instead.")
    return s["text"], f"Gold label for this tweet: **{s['label']}**"


def run(text, task, model_key, adapter, kind_label, severity, dialect_label, seed, state_turn):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type a sentence first (or press 'Random tweet').")
    kind = PERTURBATIONS[kind_label]
    dialect_name = next((k for k, v in DIALECT_LABELS.items() if v == dialect_label), "aave")
    turn = int(state_turn) + 1
    try:
        pert = apply_perturbation(text, kind, float(severity), dialect_name, int(seed), turn)
    except (ImportError, FileNotFoundError, ValueError) as e:
        raise gr.Error(f"Can't apply '{kind_label}': {e}")
    try:
        clf = get_classifier(model_key, (adapter or "").strip() or None)
        labels = load_tasks()[task].label_names
        clean_p, pert_p = clf.predict([text, pert], task, "zeroshot", batch_size=2)
    except ImportError as e:
        raise gr.Error(f"Model dependencies missing ({e}). Run with --stub to test the UI "
                       "without a model.")
    except Exception as e:
        raise gr.Error(f"Model error: {type(e).__name__}: {e}")
    changed = pert != text
    fig = make_chart(labels, clean_p, pert_p)
    plt.close(fig)
    table = [[l, f"{clean_p['probs'][l]:.3f}", f"{pert_p['probs'][l]:.3f}",
              f"{pert_p['probs'][l] - clean_p['probs'][l]:+.3f}"] for l in labels]
    return text, pert, verdict(clean_p, pert_p, changed), fig, table, turn


# UI


def build_ui(default_model: str, default_task: str) -> gr.Blocks:
    models = list(load_models())
    tasks = sorted(load_tasks())
    with gr.Blocks(title="Beyond Clean English") as demo:
        gr.Markdown(
            "# Same Meaning, Different Words\n"
            "How much does an LLM's classification change when the *same* sentence is written "
            "with typos or in another English dialect? Type a sentence, pick a perturbation, "
            "and compare the model's prediction on the clean vs perturbed text."
            + ("\n\n> ⚠️ **Stub mode:** predictions come from a fake keyword model, not a real LLM."
               if SETTINGS["stub"] else ""))
        turn = gr.State(0)
        with gr.Row():
            with gr.Column(scale=5):
                text = gr.Textbox(label="Sentence", lines=3,
                                  placeholder="e.g. I absolutely loved this movie, the acting was fantastic")
                with gr.Row():
                    sample_btn = gr.Button("🎲 Random tweet")
                    run_btn = gr.Button("Run", variant="primary")
                gold = gr.Markdown()
                kind = gr.Radio(list(PERTURBATIONS), value="Synthetic typos", label="Perturbation")
                severity = gr.Slider(0.0, 0.5, value=0.20, step=0.05,
                                     label="Severity (fraction of words perturbed)", visible=True)
                dialect_dd = gr.Dropdown(list(DIALECT_LABELS.values()), value="AAVE",
                                         label="Dialect", visible=False)
                with gr.Accordion("Model settings", open=False):
                    task = gr.Dropdown(tasks, value=default_task, label="Task")
                    model = gr.Dropdown(models, value=default_model, label="Model")
                    adapter = gr.Textbox(label="LoRA adapter path (optional)", placeholder="outputs/adapters/...")
                    seed = gr.Number(value=13, precision=0, label="Seed")
            with gr.Column(scale=6):
                verdict_md = gr.Markdown()
                with gr.Row():
                    clean_out = gr.Textbox(label="Clean", interactive=False, lines=3)
                    pert_out = gr.Textbox(label="Perturbed", interactive=False, lines=3)
                plot = gr.Plot(label="Label probabilities")
                table = gr.Dataframe(headers=["label", "clean", "perturbed", "Δ"], interactive=False,
                                     label="Probabilities")

        kind.change(toggle_controls, kind, [severity, dialect_dd])
        sample_btn.click(random_sample, task, [text, gold])
        task.change(lambda: "", None, gold)
        inputs = [text, task, model, adapter, kind, severity, dialect_dd, seed, turn]
        outputs = [clean_out, pert_out, verdict_md, plot, table, turn]
        run_btn.click(run, inputs, outputs)
        text.submit(run, inputs, outputs)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--task", default="sentiment")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--stub", action="store_true", help="Fake model: test the UI without a GPU.")
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    SETTINGS.update(stub=args.stub, load_4bit=args.load_4bit)
    build_ui(args.model, args.task).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()