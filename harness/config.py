"""Load YAML configs into dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

PERTURBATIONS = ("none", "typo", "natural_typo", "dialect", "code_switch")
TYPO_TYPES = ("typo", "natural_typo")
DIALECTS = ("aave", "indian_english", "singlish")


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the repo root unless it is absolute."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class ModelConfig:
    key: str
    hf_id: str
    gated: bool = False
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class TaskConfig:
    name: str
    labels: dict[str, str]          # gold label -> verbalizer string
    prompt_version: str
    metric: str
    extra_metrics: list[str]
    id_prefix: str
    source: dict[str, Any]
    natural: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    @property
    def label_names(self) -> list[str]:
        return list(self.labels)


@dataclass(frozen=True)
class RegimeConfig:
    name: str                       # zeroshot | fewshot | finetune
    k: int | None = None
    train_variant: str | None = None

    @property
    def slug(self) -> str:
        if self.name == "fewshot":
            return f"fewshot_k{self.k}"
        if self.name == "finetune":
            return f"finetune_{self.train_variant}"
        return self.name


@dataclass(frozen=True)
class Condition:
    """One concrete evaluation condition, i.e. one JSONL file per task."""

    perturbation: str
    severity: float | None = None
    dialect: str | None = None
    lang: str | None = None

    @property
    def name(self) -> str:
        if self.perturbation == "none":
            return "none"
        if self.perturbation in TYPO_TYPES:
            return f"{self.perturbation}_{self.severity:.2f}"
        if self.perturbation == "dialect":
            return f"dialect_{self.dialect}"
        return f"code_switch_{self.lang}" if self.lang else "code_switch"

    @classmethod
    def parse(cls, spec: str) -> "Condition":
        """Parse 'none', 'typo:0.10', 'dialect:aave', or 'code_switch:hinglish'."""
        pert, _, arg = spec.partition(":")
        if pert in TYPO_TYPES:
            return cls(pert, severity=float(arg))
        if pert == "dialect":
            return cls(pert, dialect=arg)
        if pert == "code_switch":
            return cls(pert, lang=arg or None)
        if pert == "none":
            return cls("none")
        raise ValueError(f"Unknown condition spec: {spec!r}")


def expand_condition_specs(specs: list[dict[str, Any]], task: str) -> list[Condition]:
    """Expand grid condition entries (with list-valued fields) for one task."""
    out: list[Condition] = []
    for spec in specs:
        if "tasks" in spec and task not in spec["tasks"]:
            continue
        pert = spec["perturbation"]
        if pert not in PERTURBATIONS:
            raise ValueError(f"Unknown perturbation: {pert}")
        if pert in TYPO_TYPES:
            out += [Condition(pert, severity=float(s)) for s in _as_list(spec["severity"])]
        elif pert == "dialect":
            out += [Condition(pert, dialect=d) for d in _as_list(spec.get("dialect", list(DIALECTS)))]
        elif pert == "code_switch":
            langs = _as_list(spec.get("lang"))
            out += [Condition(pert, lang=l) for l in langs] if langs else [Condition(pert)]
        else:
            out.append(Condition("none"))
    return out


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


@dataclass
class GridConfig:
    seed: int
    models: list[str]
    tasks: list[str]
    regimes: list[RegimeConfig]
    conditions: list[dict[str, Any]]
    max_examples: int | None = None
    batch_size: int = 16
    load_4bit: bool = False
    data_dir: str = "data/processed"
    output_dir: str = "outputs"

    def conditions_for(self, task: str) -> list[Condition]:
        return expand_condition_specs(self.conditions, task)


def load_models(path: str | Path = CONFIG_DIR / "models.yaml") -> dict[str, ModelConfig]:
    return {k: ModelConfig(key=k, **v) for k, v in read_yaml(path).items()}


def load_tasks(path: str | Path = CONFIG_DIR / "tasks.yaml") -> dict[str, TaskConfig]:
    return {k: TaskConfig(name=k, **v) for k, v in read_yaml(path).items()}


def load_grid(path: str | Path) -> GridConfig:
    raw = read_yaml(path)
    raw["regimes"] = [RegimeConfig(**r) for r in raw["regimes"]]
    return GridConfig(**raw)
