"""Prompt handling on top of FLUX's own T5: bracket parsing and edit-token indices."""
import re

_BRACKET = re.compile(r"\[([^\]]*)\]")


def strip_brackets(prompt: str) -> str:
    return " ".join(_BRACKET.sub(r"\1", prompt).split())


def bracketed_word_positions(prompt: str) -> list[int]:
    """Word indices (in strip_brackets(prompt).split()) that sit inside [...]."""
    positions, idx = [], 0
    for part in re.split(r"(\[[^\]]*\])", prompt):
        words = part.strip("[]").split()
        if part.startswith("["):
            positions += range(idx, idx + len(words))
        idx += len(words)
    return positions


def word_token_spans(tokenizer, words: list[str]) -> list[tuple[int, int]]:
    """[start, end) token span per word, assuming sentencepiece splits on whitespace first."""
    spans, pos = [], 0
    for w in words:
        n = len(tokenizer(w, add_special_tokens=False).input_ids)
        spans.append((pos, pos + n))
        pos += n
    full = tokenizer(" ".join(words), add_special_tokens=False).input_ids
    if len(full) != pos:
        raise ValueError(f"per-word tokenisation ({pos}) != full ({len(full)}) for {' '.join(words)!r}")
    return spans


def token_indices_for_words(tokenizer, prompt: str, word_positions: list[int], max_length: int = 512) -> list[int]:
    words = strip_brackets(prompt).split()
    spans = word_token_spans(tokenizer, words)
    idx = [t for p in word_positions for t in range(*spans[p]) if t < max_length]
    return sorted(set(idx))


def edit_token_indices(tokenizer, prompt: str, extra_words: list[str] = ()) -> list[int]:
    """T5 token indices of bracketed words in `prompt`, plus any word equal to one of `extra_words`."""
    words = strip_brackets(prompt).split()
    positions = set(bracketed_word_positions(prompt))
    extra = {w.lower() for w in extra_words}
    positions |= {i for i, w in enumerate(words) if w.lower().strip(".,") in extra}
    return token_indices_for_words(tokenizer, prompt, sorted(positions))
