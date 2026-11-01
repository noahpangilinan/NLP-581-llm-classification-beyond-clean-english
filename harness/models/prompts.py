"""Versioned prompt templates. The version string is recorded in every output."""

from __future__ import annotations

from typing import Any

TEMPLATES: dict[str, dict[str, str]] = {
    "sentiment_v1": {
        "task": "sentiment",
        "instruction": "Classify the sentiment of the tweet below.",
    },
    "offensive_v1": {
        "task": "offensive",
        "instruction": (
            "Decide whether the tweet below is offensive. A tweet is offensive if it "
            "contains insults, threats, profanity, or targeted abuse."
        ),
    },
}


def user_message(version: str, text: str, verbalizers: list[str]) -> str:
    """Instruction, allowed labels, the input text, and 'Answer:'."""
    t = TEMPLATES[version]
    return (
        f"{t['instruction']}\n"
        f"Allowed labels: {', '.join(verbalizers)}.\n"
        "Respond with the label only.\n\n"
        f"Tweet: {text}\n"
        "Answer:"
    )


def build_messages(version: str, text: str, labels: dict[str, str],
                   demos: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    """Chat messages for one example. Demos become prior user/assistant turns.

    `labels` maps gold label -> verbalizer; demos are rows with `text` and `label`.
    """
    verbs = list(labels.values())
    msgs: list[dict[str, str]] = []
    for d in demos or []:
        msgs.append({"role": "user", "content": user_message(version, d["text"], verbs)})
        msgs.append({"role": "assistant", "content": labels[d["label"]]})
    msgs.append({"role": "user", "content": user_message(version, text, verbs)})
    return msgs
