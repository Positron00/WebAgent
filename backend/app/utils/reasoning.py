"""Remove the reasoning some completion models write before their actual answer."""

# Text some models put between their reasoning and the actual answer
# (seen with Together models: "...Let's craft final answer. assistant to=user**Title**").
REASONING_END_MARKERS = ("assistant to=user", "<|channel|>final<|message|>", "assistantfinal")


def strip_reasoning(text: str) -> str:
    """Keep only what follows the last reasoning marker; return the text unchanged if there is none."""
    for marker in REASONING_END_MARKERS:
        if marker in text:
            answer = text.rsplit(marker, 1)[1].strip()
            if answer:
                text = answer
    return text.strip()
