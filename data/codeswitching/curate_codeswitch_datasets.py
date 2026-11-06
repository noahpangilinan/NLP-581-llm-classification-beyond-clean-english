#!/usr/bin/env python3
"""Curate and standardize multilingual code-switching datasets.

Supports three code-switching sources for LING-581 Final Project:
  1. sophiayk20/covoswitch (Hugging Face)
     - Multilingual parallel English (`en`) <-> Code-Switched (`csw`) sentences
       across 13 language pairs (ar_en, ca_en, cy_en, de_en, et_en, fa_en,
       id_en, lv_en, mn_en, sl_en, sv_en, ta_en, tr_en).
  2. N3mika/ParsingProject (GitHub)
     - Natural spoken Spanish-English (`spanish_english.csv`, Miami corpus) and
       optional Spanish-Guarani (`spanish_guarani.csv`) utterances with token-level
       language tags.
  3. Shelton1013/SwitchLingua_text (Hugging Face, gated auto-approval)
     - Large-scale multilingual code-switching CSVs across Spanish-English and
       14+ additional language pairs (requires `HF_TOKEN` / `hf auth login` after
       accepting repository access on Hugging Face, or a local directory).

For each curated dataset, this script writes:
  - A standard `[id, text, label]` CSV (`label` left empty `""` for unlabeled
    code-switching inputs) matching `data/clean/amazon_polarity_clean.csv`.
  - A companion `*_metadata.csv` preserving source IDs, language pairs, parallel
    clean English text (when available), and linguistic metadata.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

DEFAULT_CODESWITCH_DIR = Path(__file__).resolve().parent / "data" / "codeswitch"

COVOSWITCH_REPO = "sophiayk20/covoswitch"
COVOSWITCH_ALL_CONFIGS: tuple[str, ...] = (
    "ca_en",
    "de_en",
    "ar_en",
    "tr_en",
    "id_en",
    "fa_en",
    "sv_en",
    "ta_en",
    "cy_en",
    "et_en",
    "lv_en",
    "mn_en",
    "sl_en",
)

PARSING_PROJECT_SPA_ENG_URL = (
    "https://raw.githubusercontent.com/N3mika/ParsingProject/main/data/spanish_english.csv"
)
PARSING_PROJECT_SPA_GUA_URL = (
    "https://raw.githubusercontent.com/N3mika/ParsingProject/main/data/spanish_guarani.csv"
)

SWITCHLINGUA_REPO = "Shelton1013/SwitchLingua_text"
SWITCHLINGUA_FILES: dict[str, str] = {
    "es_en": "Spanish_eng.csv",
    "fr_en": "French_eng.csv",
    "de_en": "German_Eng.csv",
    "ar_en": "Arabic_eng.csv",
    "zh_en": "Chinese_eng.csv",
    "hi_en": "Hindi_eng.csv",
    "it_en": "Italian_eng.csv",
    "ja_en": "Japanese_eng.csv",
    "ko_en": "Korean_eng.csv",
    "ms_en": "Malay_eng.csv",
    "tl_en": "Philippines_eng.csv",
    "ru_en": "Russian_eng.csv",
    "vi_en": "Vietnamese_eng.csv",
    "yue_en": "Can_Eng.csv",
    "de_fr": "German_French.csv",
}

SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:?!%])")
MULTI_SPACE_RE = re.compile(r"\s{2,}")
STRAY_MID_PERIOD_RE = re.compile(r"\.\s+[a-z]")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to a UTF-8 CSV file with exact column ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_standard_and_meta(
    standard_path: Path,
    meta_path: Path,
    meta_rows: list[dict[str, Any]],
    meta_columns: list[str],
) -> None:
    """Save both the `[id, text, label]` CSV and the rich metadata CSV."""
    std_rows = [
        {"id": idx, "text": r["text"], "label": r.get("label", "")}
        for idx, r in enumerate(meta_rows)
    ]
    numbered_meta = []
    for idx, r in enumerate(meta_rows):
        copy_r = dict(r)
        copy_r["id"] = idx
        copy_r.setdefault("label", "")
        numbered_meta.append(copy_r)

    write_csv(standard_path, std_rows, ["id", "text", "label"])
    write_csv(meta_path, numbered_meta, meta_columns)


# ---------------------------------------------------------------------------
# 1. CoVoSwitch (sophiayk20/covoswitch)
# ---------------------------------------------------------------------------


def is_valid_covoswitch_pair(
    csw: str,
    en: str,
    l2: str,
    min_words: int = 6,
    max_words: int = 45,
) -> bool:
    """Return True if `csw` is a clean, non-trivial code-switched sentence."""
    csw_s = csw.strip()
    en_s = en.strip()
    l2_s = l2.strip()
    if not csw_s or not en_s:
        return False
    # Must actually mix languages rather than copying monolingual English or L2 verbatim
    if csw_s.lower() == en_s.lower() or (l2_s and csw_s.lower() == l2_s.lower()):
        return False
    words = csw_s.split()
    if not (min_words <= len(words) <= max_words):
        return False
    # Filter occasional MT alignment artifacts where a period precedes a lowercase continuation
    if STRAY_MID_PERIOD_RE.search(csw_s) and not STRAY_MID_PERIOD_RE.search(en_s):
        return False
    return True


def curate_covoswitch(
    output_dir: Path = DEFAULT_CODESWITCH_DIR,
    configs: list[str] | None = None,
    per_pair_samples: int = 100,
    split: str = "test",
    min_words: int = 6,
    max_words: int = 45,
    unique_english: bool = True,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Stream and curate multilingual parallel code-switching rows from CoVoSwitch."""
    target_configs = configs if configs is not None else list(COVOSWITCH_ALL_CONFIGS)
    seen_csw: set[str] = set()
    seen_en: set[str] = set()
    collected: list[dict[str, Any]] = []

    print(
        f"[CoVoSwitch] Streaming {len(target_configs)} language pairs "
        f"({per_pair_samples} samples/pair, split='{split}')..."
    )

    for cfg in target_configs:
        l2_key = cfg.split("_")[0]
        ds = load_dataset(COVOSWITCH_REPO, cfg, split=split, streaming=True)
        pair_count = 0

        for row in ds:
            if pair_count >= per_pair_samples:
                break
            trans = row.get("translation") or {}
            csw = str(trans.get("csw", "")).strip()
            en = str(trans.get("en", "")).strip()
            l2 = str(trans.get(l2_key, "")).strip()

            if not is_valid_covoswitch_pair(
                csw, en, l2, min_words=min_words, max_words=max_words
            ):
                continue

            csw_norm = csw.lower()
            en_norm = en.lower()
            if csw_norm in seen_csw:
                continue
            if unique_english and en_norm in seen_en:
                continue

            seen_csw.add(csw_norm)
            seen_en.add(en_norm)
            collected.append(
                {
                    "source": "covoswitch",
                    "source_row_id": row.get("id", ""),
                    "lang_pair": cfg,
                    "split": split,
                    "text": csw,
                    "clean_text": en,
                    "l2_text": l2,
                    "label": "",
                }
            )
            pair_count += 1

        print(f"  - {cfg}: collected {pair_count} rows")

    std_path = output_dir / "covoswitch_multilingual.csv"
    meta_path = output_dir / "covoswitch_multilingual_metadata.csv"
    meta_cols = [
        "id",
        "source",
        "source_row_id",
        "lang_pair",
        "split",
        "text",
        "clean_text",
        "l2_text",
        "label",
    ]
    write_standard_and_meta(std_path, meta_path, collected, meta_cols)
    print(f"[CoVoSwitch] Saved {len(collected)} total rows -> {std_path}")
    return std_path, meta_path, collected


# ---------------------------------------------------------------------------
# 2. ParsingProject (N3mika/ParsingProject)
# ---------------------------------------------------------------------------


def detokenize_ud_sentence(sentence: str) -> str:
    """Clean UD-tokenized spacing and multi-word underscores for classifier input."""
    # Replace underscore-joined multiword transcriptions (e.g. It's_Been_a_Day -> It's Been a Day)
    cleaned = sentence.replace("_", " ").strip()
    cleaned = SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned


def parse_parsing_project_csv(
    raw_csv: str,
    lang_pair: str,
    primary_langs: tuple[str, str],
    min_tokens_per_lang: int = 2,
    min_words: int = 6,
    max_words: int = 45,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Filter and normalize rows from N3mika/ParsingProject CSV data."""
    lang_a, lang_b = primary_langs
    reader = csv.DictReader(io.StringIO(raw_csv))
    seen_texts: set[str] = set()
    rows_out: list[dict[str, Any]] = []

    for row in reader:
        if max_samples is not None and len(rows_out) >= max_samples:
            break

        raw_sent = str(row.get("sentence", "")).strip()
        raw_tok_info = str(row.get("token_info", "")).strip()
        if not raw_sent or not raw_tok_info:
            continue

        try:
            tok_list = ast.literal_eval(raw_tok_info)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(tok_list, list):
            continue

        count_a = 0
        count_b = 0
        word_count = 0
        for tok in tok_list:
            if not isinstance(tok, (list, tuple)) or len(tok) < 3:
                continue
            tag = str(tok[2]).lower()
            if tag in ("0", "other"):
                continue
            word_count += 1
            if tag == lang_a or tag.startswith(f"{lang_a}-"):
                count_a += 1
            elif tag == lang_b or tag.startswith(f"{lang_b}-"):
                count_b += 1

        if not (min_words <= word_count <= max_words):
            continue
        if count_a < min_tokens_per_lang or count_b < min_tokens_per_lang:
            continue

        clean_sent = detokenize_ud_sentence(raw_sent)
        norm_key = clean_sent.lower()
        if norm_key in seen_texts:
            continue
        seen_texts.add(norm_key)

        rows_out.append(
            {
                "source": "parsing_project",
                "source_row_id": row.get("sentence_id", ""),
                "filename": row.get("filename", ""),
                "lang_pair": lang_pair,
                "lang_a_tokens": count_a,
                "lang_b_tokens": count_b,
                "word_count": word_count,
                "text": clean_sent,
                "raw_sentence": raw_sent,
                "clean_text": "",
                "label": "",
            }
        )

    return rows_out


def fetch_url_text(url: str, timeout: int = 30) -> str:
    """Download UTF-8 text from a public HTTP(S) URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ling581-codeswitch-curator/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def curate_parsing_project(
    output_dir: Path = DEFAULT_CODESWITCH_DIR,
    include_guarani: bool = False,
    min_tokens_per_lang: int = 2,
    min_words: int = 6,
    max_words: int = 45,
    max_samples: int | None = None,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Download and curate Spanish-English (and optional Spanish-Guarani) data."""
    print(f"[ParsingProject] Downloading Spanish-English from:\n  {PARSING_PROJECT_SPA_ENG_URL}")
    raw_spa_eng = fetch_url_text(PARSING_PROJECT_SPA_ENG_URL)
    collected = parse_parsing_project_csv(
        raw_spa_eng,
        lang_pair="es_en",
        primary_langs=("spa", "eng"),
        min_tokens_per_lang=min_tokens_per_lang,
        min_words=min_words,
        max_words=max_words,
        max_samples=max_samples,
    )
    print(f"  - es_en: collected {len(collected)} genuine mid-sentence switches")

    if include_guarani:
        print(f"[ParsingProject] Downloading Spanish-Guarani from:\n  {PARSING_PROJECT_SPA_GUA_URL}")
        raw_spa_gua = fetch_url_text(PARSING_PROJECT_SPA_GUA_URL)
        gua_rows = parse_parsing_project_csv(
            raw_spa_gua,
            lang_pair="es_gn",
            primary_langs=("es", "gn"),
            min_tokens_per_lang=min_tokens_per_lang,
            min_words=min_words,
            max_words=max_words,
            max_samples=max_samples,
        )
        print(f"  - es_gn: collected {len(gua_rows)} genuine mid-sentence switches")
        collected.extend(gua_rows)

    std_path = output_dir / "parsing_project_codeswitch.csv"
    meta_path = output_dir / "parsing_project_codeswitch_metadata.csv"
    meta_cols = [
        "id",
        "source",
        "source_row_id",
        "filename",
        "lang_pair",
        "lang_a_tokens",
        "lang_b_tokens",
        "word_count",
        "text",
        "raw_sentence",
        "clean_text",
        "label",
    ]
    write_standard_and_meta(std_path, meta_path, collected, meta_cols)
    print(f"[ParsingProject] Saved {len(collected)} total rows -> {std_path}")
    return std_path, meta_path, collected


# ---------------------------------------------------------------------------
# 3. SwitchLingua_text (Shelton1013/SwitchLingua_text)
# ---------------------------------------------------------------------------


def extract_switchlingua_texts(raw_val: Any) -> list[str]:
    """Extract one or more utterance strings from SwitchLingua text/list cells."""
    if raw_val is None:
        return []
    if isinstance(raw_val, list):
        return [str(x).strip() for x in raw_val if str(x).strip()]
    s = str(raw_val).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
    return [s]


def parse_switchlingua_file(
    file_path: Path,
    lang_pair: str,
    max_per_pair: int = 150,
    min_words: int = 5,
    max_words: int = 60,
) -> list[dict[str, Any]]:
    """Parse a SwitchLingua CSV or JSON/JSONL file into standardized metadata rows."""
    suffix = file_path.suffix.lower()
    raw_records: list[dict[str, Any]] = []

    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            raw_records = list(reader)
    elif suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            raw_records = [json.loads(line) for line in f if line.strip()]
    elif suffix == ".json":
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            loaded = json.load(f)
            raw_records = loaded if isinstance(loaded, list) else [loaded]
    else:
        raise ValueError(f"Unsupported SwitchLingua file type: {file_path}")

    candidate_cols = (
        "data_generation_result",
        "text",
        "sentence",
        "utterance",
        "cs_text",
        "content",
        "instances",
    )

    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for row_idx, rec in enumerate(raw_records):
        if len(out) >= max_per_pair:
            break

        # Locate text field
        texts: list[str] = []
        lower_keys = {k.lower(): k for k in rec.keys() if k is not None}
        for cand in candidate_cols:
            if cand in lower_keys:
                texts = extract_switchlingua_texts(rec[lower_keys[cand]])
                if texts:
                    break
        if not texts:
            # Fallback: first non-empty string column longer than 15 chars
            for k, v in rec.items():
                if k and k.lower() not in ("summary", "notes", "issues", "observations"):
                    cand_texts = extract_switchlingua_texts(v)
                    if cand_texts and len(cand_texts[0].split()) >= min_words:
                        texts = cand_texts
                        break

        for utt in texts:
            if len(out) >= max_per_pair:
                break
            cleaned = MULTI_SPACE_RE.sub(" ", utt).strip()
            wc = len(cleaned.split())
            if not (min_words <= wc <= max_words):
                continue
            norm = cleaned.lower()
            if norm in seen:
                continue
            seen.add(norm)

            out.append(
                {
                    "source": "switchlingua",
                    "source_row_id": rec.get("id", row_idx),
                    "lang_pair": lang_pair,
                    "topic": rec.get("topic", ""),
                    "cs_type": rec.get("cs_type", ""),
                    "cs_function": rec.get("cs_function", ""),
                    "cs_ratio": rec.get("cs_ratio", ""),
                    "score": rec.get("score", ""),
                    "text": cleaned,
                    "clean_text": "",
                    "label": "",
                }
            )

    return out


def curate_switchlingua(
    output_dir: Path = DEFAULT_CODESWITCH_DIR,
    lang_pairs: list[str] | None = None,
    per_pair_samples: int = 150,
    local_dir: Path | None = None,
    hf_token: str | None = None,
    strict: bool = False,
) -> tuple[Path | None, Path | None, list[dict[str, Any]]]:
    """Curate multilingual SwitchLingua_text splits from HF Hub or a local directory."""
    target_pairs = (
        lang_pairs
        if lang_pairs is not None
        else ["es_en", "fr_en", "de_en", "ar_en", "zh_en", "hi_en", "it_en", "ko_en"]
    )
    resolved_token = hf_token or os.environ.get("HF_TOKEN")
    collected: list[dict[str, Any]] = []

    print(
        f"[SwitchLingua] Loading {len(target_pairs)} language pairs "
        f"({'local_dir=' + str(local_dir) if local_dir else 'HF Hub: ' + SWITCHLINGUA_REPO})..."
    )

    for pair in target_pairs:
        filename = SWITCHLINGUA_FILES.get(pair)
        if not filename:
            print(f"  - Skipping unknown SwitchLingua pair '{pair}'")
            continue

        file_path: Path | None = None
        if local_dir is not None:
            cand = local_dir / filename
            if cand.exists():
                file_path = cand
            else:
                print(f"  - Missing local file {cand}, skipping '{pair}'")
                continue
        else:
            try:
                downloaded = hf_hub_download(
                    repo_id=SWITCHLINGUA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=resolved_token,
                )
                file_path = Path(downloaded)
            except (GatedRepoError, HfHubHTTPError) as exc:
                msg = (
                    f"\n[SwitchLingua] Access blocked (gated repository on Hugging Face: {SWITCHLINGUA_REPO}).\n"
                    "  To include SwitchLingua_text, complete these two steps and re-run:\n"
                    "    1. Visit https://huggingface.co/datasets/Shelton1013/SwitchLingua_text and click 'Agree / Access repository'.\n"
                    "    2. Authenticate locally with `hf auth login` (or `export HF_TOKEN=hf_...`),\n"
                    "       OR pass `--switchlingua-local-dir /path/to/downloaded_csvs`.\n"
                )
                if strict:
                    raise RuntimeError(msg) from exc
                print(msg)
                return None, None, []

        pair_rows = parse_switchlingua_file(
            file_path=file_path,
            lang_pair=pair,
            max_per_pair=per_pair_samples,
        )
        print(f"  - {pair} ({filename}): collected {len(pair_rows)} rows")
        collected.extend(pair_rows)

    if not collected:
        print("[SwitchLingua] No rows collected.")
        return None, None, []

    std_path = output_dir / "switchlingua_multilingual.csv"
    meta_path = output_dir / "switchlingua_multilingual_metadata.csv"
    meta_cols = [
        "id",
        "source",
        "source_row_id",
        "lang_pair",
        "topic",
        "cs_type",
        "cs_function",
        "cs_ratio",
        "score",
        "text",
        "clean_text",
        "label",
    ]
    write_standard_and_meta(std_path, meta_path, collected, meta_cols)
    print(f"[SwitchLingua] Saved {len(collected)} total rows -> {std_path}")
    return std_path, meta_path, collected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Curate and standardize multilingual code-switching datasets from "
            "CoVoSwitch, ParsingProject, and SwitchLingua_text."
        )
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_CODESWITCH_DIR,
        help=f"Output directory for code-switching CSVs (default: {DEFAULT_CODESWITCH_DIR}).",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="covoswitch,parsing_project,switchlingua",
        help="Comma-separated sources to curate (default: covoswitch,parsing_project,switchlingua).",
    )
    parser.add_argument(
        "--covoswitch-configs",
        type=str,
        default=",".join(COVOSWITCH_ALL_CONFIGS),
        help="Comma-separated CoVoSwitch configs (default: all 13 *_en pairs).",
    )
    parser.add_argument(
        "--covoswitch-per-pair",
        type=int,
        default=100,
        help="Samples per language pair for CoVoSwitch (default: 100 -> 1,300 rows across 13 pairs).",
    )
    parser.add_argument(
        "--include-guarani",
        action="store_true",
        help="Also include Spanish-Guarani (es_gn) from N3mika/ParsingProject.",
    )
    parser.add_argument(
        "--parsing-max-samples",
        type=int,
        default=None,
        help="Optional cap on ParsingProject rows (default: all valid switches, ~1,172 rows).",
    )
    parser.add_argument(
        "--switchlingua-pairs",
        type=str,
        default="es_en,fr_en,de_en,ar_en,zh_en,hi_en,it_en,ko_en",
        help="Comma-separated SwitchLingua language pairs to pull.",
    )
    parser.add_argument(
        "--switchlingua-per-pair",
        type=int,
        default=150,
        help="Max rows per language pair for SwitchLingua_text (default: 150).",
    )
    parser.add_argument(
        "--switchlingua-local-dir",
        type=Path,
        default=None,
        help="Optional local folder containing downloaded SwitchLingua CSV files.",
    )
    parser.add_argument(
        "--strict-gated",
        action="store_true",
        help="Fail with nonzero exit status if gated SwitchLingua_text is not yet authenticated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}

    all_cs_rows: list[dict[str, Any]] = []

    if "covoswitch" in sources:
        cfgs = [c.strip() for c in args.covoswitch_configs.split(",") if c.strip()]
        _, _, rows = curate_covoswitch(
            output_dir=args.output_dir,
            configs=cfgs,
            per_pair_samples=args.covoswitch_per_pair,
        )
        all_cs_rows.extend(rows)

    if "parsing_project" in sources:
        _, _, rows = curate_parsing_project(
            output_dir=args.output_dir,
            include_guarani=args.include_guarani,
            max_samples=args.parsing_max_samples,
        )
        all_cs_rows.extend(rows)

    if "switchlingua" in sources:
        sl_pairs = [p.strip() for p in args.switchlingua_pairs.split(",") if p.strip()]
        _, _, rows = curate_switchlingua(
            output_dir=args.output_dir,
            lang_pairs=sl_pairs,
            per_pair_samples=args.switchlingua_per_pair,
            local_dir=args.switchlingua_local_dir,
            strict=args.strict_gated,
        )
        all_cs_rows.extend(rows)

    if all_cs_rows:
        combined_std = args.output_dir / "codeswitch_all_combined.csv"
        combined_meta = args.output_dir / "codeswitch_all_combined_metadata.csv"
        write_standard_and_meta(
            combined_std,
            combined_meta,
            all_cs_rows,
            ["id", "source", "lang_pair", "text", "clean_text", "label"],
        )
        print(
            f"\n[Combined Code-Switching] Saved {len(all_cs_rows):,} total rows:\n"
            f"  -> Standard [id,text,label]: {combined_std}\n"
            f"  -> Unified CS metadata:      {combined_meta}"
        )


if __name__ == "__main__":
    main()
