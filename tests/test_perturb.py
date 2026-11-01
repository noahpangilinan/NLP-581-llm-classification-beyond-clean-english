import random

import pytest

from harness.perturb.keyboard import NEIGHBORS, neighbors
from harness.perturb.natural_typo import natural_typo, word_pairs
from harness.perturb.typo import typo

TEXT = "@user This movie was absolutely GREAT!!! See https://t.co/abc #MustWatch 10/10 😀 lol"


def _corpus(n_words: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    words = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 9)))
             for _ in range(n_words)]
    return " ".join(words)


def test_keyboard_symmetric_and_case():
    for k, vs in NEIGHBORS.items():
        for v in vs:
            assert k in NEIGHBORS[v]
    assert set(NEIGHBORS["s"]) == set("weadzx")
    assert neighbors("S") == NEIGHBORS["s"].upper()
    assert neighbors("1") == ""


@pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 1.0])
def test_typo_deterministic(p):
    assert typo(TEXT, p, 7) == typo(TEXT, p, 7)
    assert typo(TEXT, p, "13:typo:sent-test-00001") == typo(TEXT, p, "13:typo:sent-test-00001")


def test_typo_seed_changes_output():
    text = _corpus(200)
    assert typo(text, 0.2, 1) != typo(text, 0.2, 2)


def test_typo_p0_identity():
    assert typo(TEXT, 0.0, 123) == TEXT
    assert typo(_corpus(500), 0, 5) == _corpus(500)


def test_protected_tokens_untouched():
    out = typo(TEXT, 1.0, 3)
    toks = out.split(" ")
    for protected in ("@user", "https://t.co/abc", "#MustWatch", "10/10", "😀"):
        assert protected in toks
    assert out != TEXT  # but ordinary words did change
    assert len(out.split()) == len(TEXT.split())


def test_typo_preserves_case():
    # Substituted/inserted characters copy the case of the character they neighbor.
    for s in range(50):
        assert typo("HELLO", 1.0, s).isupper()


@pytest.mark.parametrize("p", [0.05, 0.10, 0.20])
def test_typo_rate_approx_p(p):
    text = _corpus(10_000, seed=1)
    out = typo(text, p, 42)
    a, b = text.split(" "), out.split(" ")
    assert len(a) == len(b) == 10_000
    rate = sum(x != y for x, y in zip(a, b)) / len(a)
    # Binomial sd at p=0.2, n=10k is 0.004; allow ~4 sd.
    assert abs(rate - p) < 0.016, rate


def test_single_letter_words_untouched():
    assert typo("a i a i a", 1.0, 0) == "a i a i a"


TYPO_MAP = {"great": ["graet", "grate"], "movie": ["moive"], "absolutely": ["absolutly"]}


def test_natural_typo_deterministic_and_case():
    out = natural_typo(TEXT, 1.0, 3, TYPO_MAP)
    assert out == natural_typo(TEXT, 1.0, 3, TYPO_MAP)
    assert "moive" in out and "absolutly" in out
    assert "GRAET" in out or "GRATE" in out
    assert "@user" in out and "#MustWatch" in out


def test_natural_typo_p0_and_unknown_words():
    assert natural_typo(TEXT, 0.0, 3, TYPO_MAP) == TEXT
    assert natural_typo("zzz qqq", 1.0, 3, TYPO_MAP) == "zzz qqq"


def test_build_typo_map_drops_real_word_edits():
    from harness.perturb.natural_typo import build_typo_map
    edits = [("I am goign home", "I am going home"),
             ("I am doing home", "I am going home"),          # real-word edit
             ("doing it now", "doing it now")] * 3            # 'doing' is in corrected vocab
    assert build_typo_map(edits) == {"going": ["goign"]}


def test_word_pairs():
    assert word_pairs("This is teh best.", "This is the best.") == [("the", "teh")]
    assert word_pairs("totally different", "nothing alike here") == []
