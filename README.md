# Same Meaning, Different Words — robustness evaluation harness

Measures how much LLM classification drops under typos, dialect variation, and
code-switching, for three ~3–4B instruct models in three regimes (zero-shot,
in-context learning, LoRA fine-tuning) on TweetEval sentiment and offensive-language
detection. Course project for NLP 581.

| | Main grid (`configs/grid_main.yaml`) |
|---|---|
| Models | Qwen3-4B-Instruct-2507, Llama-3.2-3B-Instruct, Gemma 4 E4B-it (text only) |
| Tasks | Sentiment, offensive language |
| Regimes | Zero-shot, 4-shot, 8-shot, LoRA on clean data, LoRA on perturbed data |
| Conditions | Clean; synthetic and natural typos at 5/10/20% of words; AAVE, Indian English and Singlish via Multi-VALUE; Hinglish and Spanglish code-switching (sentiment only) |

Research questions:
- **RQ1:** how much does each kind of non-standard English hurt each model and regime, and which is most damaging?
- **RQ2:** does the drop under synthetic typos predict the drop under real ones (Spearman across matched conditions)?
- **RQ3:** does fine-tuning on perturbed text, rather than clean text, recover the loss?

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export HF_TOKEN=...        # needed for meta-llama/Llama-3.2-3B-Instruct (gated)
pytest                     # CPU-only; downloads sshleifer/tiny-gpt2
```

**Multi-VALUE (dialects)** clashes with the main env, so it lives in its own venv and
is driven through a subprocess (`harness/perturb/dialect.py` has no harness imports):

```bash
python -m venv .venv-value
.venv-value/bin/pip install -r requirements-value.txt
```

## Pipeline

```bash
# 1. Natural typo map (GitHub Typo Corpus, via HF mirror; original S3 link is dead)
python -m harness.perturb.natural_typo                 # -> data/cache/typo_map.json

# 2. Build evaluation sets (+ spot-check CSVs) -> data/processed/{task}/{condition}.jsonl
python -m harness.data.build_sets --task sentiment offensive --grid configs/grid_main.yaml \
    --dialect-python .venv-value/bin/python

# 3. Run the grid (resumable; finished runs are skipped)
python -m harness.run --grid configs/grid_main.yaml --dry-run
python -m harness.run --grid configs/grid_main.yaml [--load-4bit] [--models qwen]

# 4. LoRA adapters for the finetune regimes
python -m harness.finetune.lora --model qwen --task sentiment --train clean
python -m harness.finetune.lora --model qwen --task sentiment --train clean \
    --train-perturbed typo:0.10,dialect:all --dialect-python .venv-value/bin/python   # RQ3

# 5. Tables + plots
python -m harness.report --out outputs/reports/

# Demo
python demo.py --model qwen --task sentiment [--adapter PATH] [--all-models] [--load-4bit]
```

`configs/grid_smoke.yaml` (1 model, 1 task, zero-shot + 4-shot, clean + typo 0.10,
50 examples) is the end-to-end integration test.

## Layout

| Path | What |
|---|---|
| `configs/` | `models.yaml`, `tasks.yaml` (labels, prompt versions, metrics, data sources), grids |
| `harness/config.py` | YAML → dataclasses; `Condition` naming (`typo_0.10`, `dialect_aave`, …) |
| `harness/data/` | `schema.py` (JSONL schema + validator), `loaders.py`, `build_sets.py` |
| `harness/perturb/` | `typo.py`, `natural_typo.py`, `dialect.py`, `keyboard.py`, `tokens.py` |
| `harness/models/` | `prompts.py`, `fewshot.py`, `wrapper.py` (`LLMClassifier`) |
| `harness/finetune/lora.py` | LoRA/QLoRA training CLI |
| `harness/metrics.py` | metrics, flip rate, dialect FPR, bootstrap CIs, RQ2 Spearman |
| `harness/run.py`, `harness/report.py` | grid runner, reporting |
| `outputs/` (gitignored) | `predictions/`, `adapters/`, `reports/`, `run.log` |

## Data interface

One JSON object per line:

```json
{"id": "sent-test-00042-typo-p0.10", "source_id": "sent-test-00042", "task": "sentiment",
 "split": "test", "text": "...", "label": "positive",
 "perturbation": "typo", "severity": 0.10, "dialect": null, "origin": "synthetic"}
```

Prebuilt files dropped at `data/processed/{task}/{condition}.jsonl` take precedence;
anything missing that can be derived from the clean set is generated on the fly and
cached. Clean rows have `source_id == id` and `origin: natural`. `natural_typo` rows
use `origin: natural` (real misspellings); typo and Multi-VALUE rows are `synthetic`.

## Design notes

- **Label scoring, not generation.** Each label string's summed token log-prob after the
  chat-template assistant prefix; softmax over labels → `probs`. Left-padded batches
  with explicit `position_ids`, and only the last `max_label_len+1` logits are kept
  (`logits_to_keep`), so large vocabularies don't blow up memory.
- **Determinism.** Every perturbation is seeded per row with `"{seed}:{condition}:{source_id}"`;
  subsampling is seeded and label-stratified, and row-aligned sets select the same
  `source_id`s. Generated sets always hold the canonical 1000 rows and are subsampled
  on read, so small runs never truncate the cache.
- **Resumability.** All JSONL is written atomically (tmp + rename); a run is done iff its
  output file exists.
- **Typo definition.** Each ASCII alphabetic word of length ≥ 2 is perturbed with prob. *p*
  by a uniformly chosen keyboard substitution / deletion / neighbor insertion /
  transposition. URLs, @mentions, #hashtags, emoji, and tokens with digits are untouched.
- **Natural typos.** The HF mirror has no `is_typo` flag, so real-word edits
  (e.g. going→doing) are filtered by dropping misspellings that are common words in
  corrected text. Words not in the map are left unchanged, so the effective rate is below *p*.
- **VALUE-SST-2** is binary; metrics macro-average over the gold labels present.

## Local verification (RTX 4060 8 GB, Windows)

Qwen3-4B needs `--load-4bit` on 8 GB cards; bf16 fits on T4/L4/A100.
