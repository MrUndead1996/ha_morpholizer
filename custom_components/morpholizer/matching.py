"""Bounded, conservative fuzzy name matching primitives."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FuzzySettings:
    """Opt-in confidence settings for fuzzy normalization."""

    enabled: bool = False
    score_cutoff: float = 90.0
    margin: float = 8.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        for name, value, minimum, maximum in (
            ("score_cutoff", self.score_cutoff, 85.0, 100.0),
            ("margin", self.margin, 5.0, 30.0),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")


# Hard ceilings make work independent of arbitrarily large input/catalogs. The
# caller must reject a request exceeding any ceiling rather than scan a prefix.
MAX_TEXT_CHARS = 4096
MAX_QUERY_TOKENS = 128
MAX_NAME_TOKENS = 8
MAX_NAME_CHARS = 128
MAX_CANDIDATES = 4096
MAX_COMPARISONS = 250_000


def normalized_similarity(left: str, right: str) -> float:
    """Return RapidFuzz Levenshtein similarity on a 0..100 scale."""
    from rapidfuzz.distance import Levenshtein

    return 100.0 * Levenshtein.normalized_similarity(left, right)
