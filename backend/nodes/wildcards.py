"""Wildcards in prompts (report item F21): `{a|b|c}` picks one alternative and
`__name__` one line of `wildcards/name.txt` in the data folder (subfolders as
`__colors/warm__`). They're resolved per seed by the samplers -- image i of a
batch with seed + i -- so a batch varies, and any single image reproduces from
its own seed (Save Image embeds that seed with batch_size 1).

Only braces with a `|` inside are wildcards, so a literal `{word}` is left
alone; A1111 weighting uses ( ) and [ ], never braces. Alternatives may be
empty (`{red |}`) and may nest (`{a|{b|c}}`); a wildcard file's lines may
contain more wildcards. Lines that are empty or start with `#` are skipped.

No torch import: the samplers call this before anything is encoded."""

import random
import re
from pathlib import Path
from typing import List, Optional

from .. import paths

# Innermost {...|...}: no braces inside, at least one "|".
_ALTERNATIVES = re.compile(r"\{([^{}]*\|[^{}]*)\}")
_FILE_WILDCARD = re.compile(r"__([A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-]+)*)__")
# A wildcard file can name another wildcard (or itself): stop after this many
# rounds instead of looping forever.
MAX_ROUNDS = 32
MAX_WILDCARD_FILE_BYTES = 1024 * 1024


class WildcardError(ValueError):
    """A missing or unreadable wildcard file, or wildcards that never settle."""


def has_wildcards(text: Optional[str]) -> bool:
    return bool(text) and bool(_ALTERNATIVES.search(text) or _FILE_WILDCARD.search(text))


def wildcard_dir() -> Path:
    return paths.WILDCARD_DIR


def wildcard_file(name: str) -> Path:
    """`wildcards/<name>.txt`; `name` is already limited by the pattern to
    letters, digits, `_`, `-` and `/`, and safe_path() rejects anything that
    still resolves outside the folder."""
    return paths.safe_path(wildcard_dir(), f"{name}.txt")


def wildcard_options(name: str) -> List[str]:
    path = wildcard_file(name)
    try:
        if path.stat().st_size > MAX_WILDCARD_FILE_BYTES:
            raise WildcardError(f"wildcard file {path.name} is larger than 1 MB")
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise WildcardError(
            f"__{name}__: no wildcard file {name}.txt in {wildcard_dir()} (one option per line)"
        ) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise WildcardError(f"__{name}__: can't read {path}: {exc}") from None
    options = [line.strip() for line in text.splitlines()]
    options = [line for line in options if line and not line.startswith("#")]
    if not options:
        raise WildcardError(f"__{name}__: {name}.txt has no options (one per line, # for comments)")
    return options


def referenced_files(text: Optional[str]) -> List[str]:
    """Names of the wildcard files `text` mentions directly (for IS_CHANGED)."""
    return sorted(set(_FILE_WILDCARD.findall(text or "")))


def resolve(text: Optional[str], seed: int, stream: str = "positive") -> Optional[str]:
    """`text` with every wildcard replaced, the same way every time for the
    same seed. `stream` keeps the positive and negative prompt's choices
    independent: editing the negative doesn't reshuffle the positive."""
    if not has_wildcards(text):
        return text
    rng = random.Random(f"pixelflow-wildcards:{stream}:{int(seed)}")
    for _round in range(MAX_ROUNDS):
        changed = False
        match = _FILE_WILDCARD.search(text)
        while match:
            choice = rng.choice(wildcard_options(match.group(1)))
            text = text[:match.start()] + choice + text[match.end():]
            changed = True
            match = _FILE_WILDCARD.search(text, match.start() + len(choice))
        match = _ALTERNATIVES.search(text)
        while match:
            choice = rng.choice(match.group(1).split("|"))
            text = text[:match.start()] + choice + text[match.end():]
            changed = True
            match = _ALTERNATIVES.search(text)
        if not changed or not has_wildcards(text):
            return text
    raise WildcardError(f"wildcards still unresolved after {MAX_ROUNDS} rounds (a file that names itself?)")
