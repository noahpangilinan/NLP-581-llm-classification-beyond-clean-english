"""Prompting, few-shot sampling, and label scoring on a tiny CPU model."""

import pytest

from harness.models.fewshot import sample_demos
from harness.models.prompts import build_messages

LABELS = {"negative": "negative", "neutral": "neutral", "positive": "positive"}


def _train(n=60):
    names = list(LABELS)
    return [{"id": f"t{i}", "text": f"train tweet {i}", "label": names[i % 3], "perturbation": "none"}
            for i in range(n)]


def test_prompt_contains_parts():
    msgs = build_messages("sentiment_v1", "I love it", LABELS)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    c = msgs[0]["content"]
    assert "negative, neutral, positive" in c and "I love it" in c and c.endswith("Answer:")


def test_demos_become_turns():
    demos = sample_demos(_train(), 4, seed=13)
    msgs = build_messages("sentiment_v1", "x", LABELS, demos)
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 4 + ["user"]
    assert msgs[1]["content"] in LABELS.values()


@pytest.mark.parametrize("k", [4, 8])
def test_fewshot_balanced_and_deterministic(k):
    d1 = sample_demos(_train(), k, seed=13)
    assert d1 == sample_demos(_train(), k, seed=13)
    assert len(d1) == k
    counts = [sum(d["label"] == l for d in d1) for l in LABELS]
    assert max(counts) - min(counts) <= 1


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("transformers")
    from harness.models.wrapper import LLMClassifier
    try:
        return LLMClassifier("sshleifer/tiny-gpt2")
    except OSError as e:  # offline
        pytest.skip(f"tiny model unavailable: {e}")


def test_label_scoring_valid_and_batch_invariant(tiny):
    texts = ["I love this so much", "worst day ever", "ok", "a much longer tweet " * 5]
    batched = tiny.predict(texts, "sentiment", "zeroshot", batch_size=3)
    single = [tiny.predict([t], "sentiment", "zeroshot", batch_size=1)[0] for t in texts]
    for b, s in zip(batched, single):
        assert b["label"] in LABELS
        assert sum(b["probs"].values()) == pytest.approx(1.0, abs=1e-4)
        assert b["label"] == max(b["probs"], key=b["probs"].get)
        for l in LABELS:
            assert b["probs"][l] == pytest.approx(s["probs"][l], abs=1e-4)


def test_multi_token_labels(tiny):
    # "not offensive" spans several tokens; probs must still be over the two gold labels.
    out = tiny.predict(["you are great"], "offensive", "zeroshot")
    assert set(out[0]["probs"]) == {"not_offensive", "offensive"}
    assert any(len(ids) > 1 for ids in tiny.label_ids("offensive"))


def test_fewshot_requires_demos(tiny):
    with pytest.raises(ValueError):
        tiny.predict(["x"], "sentiment", "fewshot")
    demos = sample_demos(_train(), 4, seed=1)
    assert tiny.predict(["x"], "sentiment", "fewshot", demos=demos)[0]["label"] in LABELS
