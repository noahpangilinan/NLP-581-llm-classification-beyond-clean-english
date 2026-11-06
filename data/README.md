# LING-581 Data & Perturbation Pipeline (`gyan-data`)

Dataset curation, synthetic typo perturbation, multilingual code-switching standardization, and mega-dataset assembly pipeline for **LING-581 Final Project**:  
> ***Same Meaning, Different Words: Measuring and Mitigating the Robustness of Sentiment Classifiers to Typos, Dialects, and Code-Switching***  
> *Ethan Piccus, Owen Davis, Gyan Mistry, Noah Pangilinan*

---

## Directory Structure

The repository is organized into distinct subdirectories:

```
gyan-data/
├── typos/                          # Typo curation and perturbation pipeline
│   ├── fetch_typo_corpus.py        # Compiles crate-ci/typos dictionary
│   ├── curate_clean_dataset.py     # Curates verified typo-free English baseline
│   ├── typo_injector.py            # Core crate-ci typo injection engine
│   ├── inject_typos.py             # CLI to generate perturbed datasets & switch logs
│   └── data/
│       ├── crate_ci_typos_dict.json
│       ├── crate_ci_english_words.txt
│       ├── crate_ci_known_typos.txt
│       ├── clean/                  # Baseline amazon_polarity_clean.csv
│       └── perturbed/              # amazon_polarity_typos_{low,medium,high}.csv + switch logs
│
├── codeswitching/                  # Code-switching curation and standardization
│   ├── curate_codeswitch_datasets.py # Curates CoVoSwitch, ParsingProject, SwitchLingua
│   └── data/                       # Standard [id,text,label] CSVs + *_metadata.csv
│       ├── covoswitch_multilingual.csv
│       ├── parsing_project_codeswitch.csv
│       └── codeswitch_all_combined.csv
│
├── mega/                           # Unified benchmark dataset & audit sheet
│   ├── build_mega_dataset.py       # Combines all clean, typo, and code-switching splits
│   └── data/
│       ├── ling581_mega_dataset.csv  # 6,470-row master evaluation dataset
│       └── manual_label_check_50.csv # Stratified 50-row manual verification sheet
│
└── tests/                          # Modular test suites (50/50 passing)
    ├── typos/                      # Unit tests for typo injection & curation
    ├── codeswitching/              # Unit tests for code-switching parsing
    └── mega/                       # Unit tests for mega dataset assembly
```

---

## Quick Start

### 1. Typo Pipeline (`typos/`)

#### A. Fetch & Compile the `crate-ci/typos` Dictionary
```bash
python3 typos/fetch_typo_corpus.py
```
Caches `typos/data/crate_ci_typos_dict.json` locally (also run automatically on first use of `inject_typos.py`).

#### B. Curate Clean Baseline Dataset (`amazon_polarity`, Apache-2.0)
```bash
python3 typos/curate_clean_dataset.py --num-samples 1000
```
Outputs to `typos/data/clean/amazon_polarity_clean.csv`.

#### C. Inject Typos Across Severity Levels
Generate `low` (5%), `medium` (15%), and `high` (30%) perturbed splits plus their companion switch logs:
```bash
python3 typos/inject_typos.py \
  --input typos/data/clean/amazon_polarity_clean.csv \
  --output "typos/data/perturbed/amazon_polarity_typos_{severity}.csv" \
  --severities low,medium,high \
  --seed 42
```

---

### 2. Code-Switching Pipeline (`codeswitching/`)

Curates standard `id,text,label` CSVs (`label` left blank `""`) plus companion `*_metadata.csv` files from **CoVoSwitch** (13 `*_en` pairs) and **ParsingProject** (`es_en`, plus optional `--include-guarani`):
```bash
python3 codeswitching/curate_codeswitch_datasets.py
```
Outputs to `codeswitching/data/`.

> **Unlocking `Shelton1013/SwitchLingua_text` (gated on Hugging Face):**
> 1. Accept access at https://huggingface.co/datasets/Shelton1013/SwitchLingua_text
> 2. Run `hf auth login` (or `export HF_TOKEN=hf_...`), then re-run `python3 codeswitching/curate_codeswitch_datasets.py` (or pass `--switchlingua-local-dir /path/to/csvs`).

---

### 3. Mega Dataset & Manual Audit (`mega/`)

Combines all clean, typo-perturbed (`low`/`medium`/`high`), and code-switching splits into `mega/data/ling581_mega_dataset.csv` and exports a stratified 50-row check sheet (`mega/data/manual_label_check_50.csv`):
```bash
python3 mega/build_mega_dataset.py
```

---

## Output Schemas

### 1. Perturbed Dataset (`amazon_polarity_typos_<severity>.csv`)
Preserves the exact columns of the input file (no extra columns added):

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `int` | Example ID matching the clean dataset |
| `text` | `str` | Perturbed review text with `crate-ci/typos` injected |
| `label` | `int` | Ground-truth sentiment (`0` = negative, `1` = positive) |

### 2. Companion Word-Switch Dataset (`amazon_polarity_typos_<severity>_switches.csv`)
Each row records a single word switch event:

| Column | Type | Description |
| :--- | :--- | :--- |
| `row_index` | `int` | 0-based row index in the clean/perturbed dataset |
| `word_index` | `int` | 0-based whitespace-token index within the sentence (`text.split()[word_index]`) |
| `original_word` | `str` | Clean alphabetic word before perturbation (e.g., `Great`) |
| `perturbed_word` | `str` | Switched typo word with casing preserved (e.g., `Graet`) |
| `original_token` | `str` | Full whitespace token including attached punctuation before perturbation (e.g., `"Great!"`) |
| `perturbed_token` | `str` | Full whitespace token including attached punctuation after perturbation (e.g., `"Graet!"`) |
| `orig_char_start` | `int` | Start character offset of `original_word` in clean `text` |
| `orig_char_end` | `int` | End character offset of `original_word` in clean `text` |
| `pert_char_start` | `int` | Start character offset of `perturbed_word` in perturbed `text` |
| `pert_char_end` | `int` | End character offset of `perturbed_word` in perturbed `text` |
| `edit_type` | `str` | Operation class: `transposition`, `deletion`, `insertion`, `substitution`, or `multi_edit` |
| `edit_distance` | `int` | Character-level Levenshtein distance between `original_word` and `perturbed_word` |
| `num_typo_candidates` | `int` | Number of candidate typos for `original_word` in `crate-ci/typos` |
| `sentence_word_count` | `int` | Total whitespace tokens in the example |
| `eligible_word_count` | `int` | Number of tokens in the example present in `crate-ci/typos` |
| `label` | `int/str` | Sentiment label of the row (if present in input dataset) |
| `target_rate` | `float` | Target word perturbation rate (e.g., `0.05`, `0.15`, `0.30`) |
| `seed` | `int` | Random seed used for reproducibility |

### 3. Unified Mega Dataset (`mega/data/ling581_mega_dataset.csv`)
Combines all splits while retaining `id,text,label` compatibility alongside provenance metadata:

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `int` | Global 0-based row ID across the mega dataset |
| `source_id` | `int/str` | Row ID within the source split |
| `phenomenon` | `str` | `clean`, `typos`, or `codeswitch` |
| `variant` | `str` | `clean_baseline`, `typos_low`, `typos_medium`, `typos_high`, `covoswitch`, `parsing_project`, or `switchlingua` |
| `source_dataset` | `str` | Upstream dataset identifier |
| `lang_pair` | `str` | ISO pair (`en`, `es_en`, `ca_en`, `de_en`, `ar_en`, etc.) |
| `severity` | `str` | `none`, `low`, `medium`, or `high` |
| `text` | `str` | Evaluation input text (clean, typo-perturbed, or code-switched) |
| `clean_text` | `str` | Parallel unperturbed English text when available (`clean`, `typos`, `covoswitch`) |
| `label` | `int/str` | `0`/`1` for labeled splits; empty `""` for unlabeled code-switching rows |

---

## Python API Usage

```python
from typos.typo_injector import TypoInjector

injector = TypoInjector(severity="medium", seed=42)
perturbed_text, switches = injector.perturb_text(
    "This album was an amazing experience from start to finish!",
    row_index=0,
    label=1,
)
print(perturbed_text)
for s in switches:
    print(f"word[{s.word_index}]: {s.original_word} -> {s.perturbed_word} ({s.edit_type})")
```

---

## Running Tests

Run all unit tests across the modular subdirectories:
```bash
python3 -m unittest discover -s tests -v
```
