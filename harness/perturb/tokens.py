"""Find perturbable words while protecting URLs, mentions, hashtags, emoji, numbers."""

from __future__ import annotations

import re
from collections.abc import Callable

_WS = re.compile(r"(\s+)")
_WORD = re.compile(r"[A-Za-z]+")
_PROTECTED = re.compile(r"^(https?://|www\.|@|#)|\d", re.IGNORECASE)


def is_protected(token: str) -> bool:
    """A whitespace token is protected if it is a URL, @mention, #hashtag, or has digits."""
    return bool(_PROTECTED.search(token))


def map_words(text: str, fn: Callable[[str], str], min_len: int = 2) -> str:
    """Apply `fn` to every ASCII alphabetic run of length >= min_len in unprotected tokens.

    Whitespace and non-alphabetic characters (punctuation, emoji) are preserved,
    and words are visited left to right so seeded RNG use is deterministic.
    """
    parts = _WS.split(text)
    for i, tok in enumerate(parts):
        if not tok or tok.isspace() or is_protected(tok):
            continue
        parts[i] = _WORD.sub(lambda m: fn(m.group()) if len(m.group()) >= min_len else m.group(), tok)
    return "".join(parts)
