"""Small, bounded Russian morphology adapter (independent of Home Assistant)."""

from dataclasses import dataclass
from functools import lru_cache
from threading import RLock


@dataclass(frozen=True, slots=True)
class WordAnalysis:
    lemma: str
    pos: str | None
    score: float
    known: bool


class Morphology:
    """Thread-safe pymorphy3 facade. Construct this in an executor."""

    def __init__(self) -> None:
        from pymorphy3 import MorphAnalyzer

        self._analyzer = MorphAnalyzer(lang="ru")
        self._lock = RLock()
        # Keep the cache owned by this instance: a decorated bound method
        # would put ``self`` in a process-global cache and retain analyzers
        # after an integration reload.
        self._cached_analyze = lru_cache(maxsize=4096)(self._analyze_uncached)
        self._cached_forms = lru_cache(maxsize=4096)(self._forms_uncached)

    def nominal_forms(self, word: str) -> tuple[str, ...]:
        """Known noun/adjective forms only; never predict a typo's paradigm."""
        return self._cached_forms(word)

    def _forms_uncached(self, word: str) -> tuple[str, ...]:
        with self._lock:
            parses = self._analyzer.parse(word)
            forms = {word}
            for parse in parses:
                if (
                    not parse.is_known
                    or parse.score < 0.05
                    or parse.tag.POS not in {"NOUN", "ADJF", "ADJS", "COMP"}
                ):
                    continue
                for form in parse.lexeme:
                    if form.tag.POS == parse.tag.POS:
                        forms.add(str(form.word))
                    if len(forms) > 64:
                        return (word,)
            return tuple(sorted(forms))

    def analyze(self, word: str) -> tuple[WordAnalysis, ...]:
        """Return analyses ordered by pymorphy's confidence, cached by input."""
        return self._cached_analyze(word)

    def _analyze_uncached(self, word: str) -> tuple[WordAnalysis, ...]:
        with self._lock:
            parses = self._analyzer.parse(word)
        return tuple(
            WordAnalysis(
                lemma=str(parse.normal_form),
                pos=str(parse.tag.POS) if parse.tag.POS else None,
                score=float(parse.score),
                known=bool(parse.is_known),
            )
            for parse in parses
        )
