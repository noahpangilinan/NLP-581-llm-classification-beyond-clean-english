import json

import pytest

from harness.config import Condition, TaskConfig
from harness.data.loaders import perturb_rows, stratified_subsample
from harness.data.schema import SchemaError, make_row, validate_file, validate_row, write_jsonl

LABELS = ["negative", "neutral", "positive"]


def _row(**kw):
    base = dict(id="sent-test-00042", source_id="sent-test-00042", task="sentiment", split="test",
                text="hello world", label="positive")
    base.update(kw)
    return make_row(**base)


def test_valid_rows():
    validate_row(_row(), LABELS)
    validate_row(_row(id="x-typo", perturbation="typo", severity=0.1, origin="synthetic"), LABELS)
    validate_row(_row(id="x-d", perturbation="dialect", dialect="aave", origin="synthetic"), LABELS)
    validate_row(_row(id="cs", source_id=None, perturbation="code_switch"), LABELS)


@pytest.mark.parametrize("bad", [
    {"perturbation": "typo"},                                  # missing severity
    {"perturbation": "none", "severity": 0.1},                 # severity on non-typo
    {"perturbation": "dialect"},                               # no dialect name
    {"perturbation": "shout"},
    {"origin": "made_up"},
    {"label": "angry"},
    {"severity": True, "perturbation": "typo"},
])
def test_invalid_rows(bad):
    with pytest.raises(SchemaError):
        validate_row(_row(**bad), LABELS)


def test_missing_field():
    r = _row()
    del r["origin"]
    with pytest.raises(SchemaError):
        validate_row(r)


def test_validate_file_duplicates(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl([_row(), _row()], p)
    with pytest.raises(SchemaError):
        validate_file(p)
    write_jsonl([_row(), _row(id="b")], p)
    assert validate_file(p, LABELS) == 2
    assert all(json.loads(l) for l in p.read_text(encoding="utf-8").splitlines())


def _clean(n=300):
    return [_row(id=f"s-{i:05d}", source_id=f"s-{i:05d}", text=f"this is tweet number {i}",
                 label=LABELS[(i * 7) % 3 if i % 5 else 0]) for i in range(n)]


def test_stratified_subsample():
    rows = _clean()
    sub = stratified_subsample(rows, 50, seed=13)
    assert len(sub) == 50
    assert sub == stratified_subsample(rows, 50, seed=13)
    frac = lambda rs, l: sum(r["label"] == l for r in rs) / len(rs)
    for l in LABELS:
        assert abs(frac(sub, l) - frac(rows, l)) < 0.03
    assert [r["id"] for r in sub] == sorted(r["id"] for r in sub)   # original order kept
    assert stratified_subsample(rows, None, 1) == rows


def test_perturb_rows_aligned_and_valid():
    clean = _clean(40)
    pert = perturb_rows(clean, Condition("typo", severity=0.2), seed=13)
    assert [p["source_id"] for p in pert] == [c["id"] for c in clean]
    for p in pert:
        validate_row(p, LABELS)
        assert p["id"].endswith("-typo-p0.20") and p["origin"] == "synthetic"
    # Subsampling clean and perturbed with the same seed selects the same source_ids.
    a = stratified_subsample(clean, 10, 5)
    b = stratified_subsample(pert, 10, 5)
    assert [r["source_id"] for r in a] == [r["source_id"] for r in b]


def test_condition_names_roundtrip():
    for spec, name in [("none", "none"), ("typo:0.1", "typo_0.10"), ("natural_typo:0.05", "natural_typo_0.05"),
                       ("dialect:aave", "dialect_aave"), ("code_switch:hinglish", "code_switch_hinglish")]:
        assert Condition.parse(spec).name == name
