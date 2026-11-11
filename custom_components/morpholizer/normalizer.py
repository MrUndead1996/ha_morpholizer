"""Conservative, explainable morphology-based entity-name normalization."""

from dataclasses import dataclass
from itertools import pairwise
from threading import RLock

from .morphology import Morphology


@dataclass(frozen=True, slots=True)
class NameCandidate:
    entity_id: str
    name: str


@dataclass(frozen=True, slots=True)
class NormalizationDecision:
    reason: str
    method: str
    span: tuple[int, int]
    source: str
    candidates: tuple[NameCandidate, ...]


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    original_text: str
    text: str
    replacements: tuple[NormalizationDecision, ...]
    decisions: tuple[NormalizationDecision, ...]


@dataclass(frozen=True, slots=True)
class _Token:
    value: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Prepared:
    tokens: tuple[str, ...]
    separators: tuple[str, ...]
    candidate: NameCandidate
    analyses: tuple[tuple[tuple[str, str], ...], ...]


@dataclass(frozen=True, slots=True)
class _Proposal:
    start: int
    end: int
    candidate: NameCandidate
    method: str
    reason: str = "matched"


_NOMINAL = {"NOUN", "ADJF", "ADJS", "COMP"}
_NON_NOMINAL = {"VERB", "INFN", "PRED"}


def _fold(value: str) -> str:
    return value.casefold().replace("ё", "е")


def _tokens(text: str) -> tuple[_Token, ...]:
    result: list[_Token] = []
    start: int | None = None
    for i, char in enumerate(text):
        if char.isalnum():
            if start is None:
                start = i
        elif start is not None:
            result.append(_Token(text[start:i], start, i))
            start = None
    if start is not None:
        result.append(_Token(text[start:], start, len(text)))
    return tuple(result)


def _cyrillic(value: str) -> bool:
    return bool(value) and all(
        "а" <= char.casefold() <= "я" or char.casefold() == "ё" for char in value
    )


def _separator_key(value: str) -> str:
    # Whitespace may vary, but punctuation/symbols are structural and exact.
    return " ".join(value.split()) if value.isspace() else value


def _phrase_separators(text: str, phrase: tuple[_Token, ...]) -> tuple[str, ...]:
    return tuple(_separator_key(text[left.end : right.start]) for left, right in pairwise(phrase))


class Normalizer:
    """Normalize only unambiguous inflected forms of supplied entity names."""

    def __init__(self, morphology: Morphology) -> None:
        self._morphology = morphology
        self._prepared: tuple[_Prepared, ...] = ()
        self._prepared_key: tuple[tuple[str, str], ...] | None = None
        self._prepare_lock = RLock()

    def _prepare(self, candidates: tuple[NameCandidate, ...]) -> tuple[_Prepared, ...]:
        key = tuple((c.entity_id, c.name) for c in candidates)
        with self._prepare_lock:
            if key == self._prepared_key:
                return self._prepared
            prepared: list[_Prepared] = []
            seen: set[tuple[str, str]] = set()
            for candidate in candidates:
                raw_tokens = _tokens(candidate.name)
                if not raw_tokens:
                    continue
                words = tuple(_fold(t.value) for t in raw_tokens)
                identity = (candidate.entity_id, candidate.name.casefold())
                if identity in seen:
                    continue
                seen.add(identity)
                separators = _phrase_separators(candidate.name, raw_tokens)
                token_analyses = []
                for token in raw_tokens:
                    if not _cyrillic(token.value):
                        token_analyses.append(())
                        continue
                    token_analyses.append(
                        tuple(
                            (_fold(a.lemma), a.pos)
                            for a in self._morphology.analyze(token.value)
                            if a.known and a.pos in _NOMINAL and a.score >= 0.05
                        )
                    )
                prepared.append(_Prepared(words, separators, candidate, tuple(token_analyses)))
            self._prepared_key = key
            self._prepared = tuple(prepared)
            return self._prepared

    def normalize(self, text: str, candidates: tuple[NameCandidate, ...]) -> NormalizationResult:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        prepared = self._prepare(tuple(candidates))
        tokens = _tokens(text)
        proposals: list[_Proposal] = []
        for i in range(len(tokens)):
            for target in prepared:
                count = len(target.tokens)
                if i + count > len(tokens):
                    continue
                phrase = tokens[i : i + count]
                if _phrase_separators(text, phrase) != target.separators:
                    continue
                source_words = tuple(_fold(t.value) for t in phrase)
                if source_words == target.tokens:
                    proposals.append(_Proposal(i, i + count, target.candidate, "exact"))
                    continue
                matches = True
                ambiguous_analysis = False
                for source, target_word, target_analyses in zip(
                    phrase, target.tokens, target.analyses
                ):
                    # Literal non-Russian tokens (brands, model numbers, etc.)
                    # are allowed only unchanged; never morphologically infer them.
                    if _fold(source.value) == target_word:
                        continue
                    if not (_cyrillic(source.value) and _cyrillic(target_word)):
                        matches = False
                        break
                    source_analyses = tuple(
                        (_fold(a.lemma), a.pos)
                        for a in self._morphology.analyze(source.value)
                        if a.known and a.pos in _NOMINAL and a.score >= 0.05
                    )
                    credible = tuple(
                        a
                        for a in self._morphology.analyze(source.value)
                        if a.known and a.score >= 0.05
                    )
                    # Forms such as «мой» have credible nominal and verb
                    # readings. Fail closed rather than rewriting a possible
                    # command or predicate as an entity mention.
                    if any(a.pos in _NON_NOMINAL for a in credible):
                        ambiguous_analysis = True
                    # Require compatible part of speech and common credible
                    # dictionary lemma, not just a matching raw target spelling.
                    if not any(
                        source_lemma == target_lemma and source_pos == target_pos
                        for source_lemma, source_pos in source_analyses
                        for target_lemma, target_pos in target_analyses
                    ):
                        matches = False
                        break
                if matches:
                    proposals.append(
                        _Proposal(
                            i,
                            i + count,
                            target.candidate,
                            "morphology",
                            "ambiguous_analysis" if ambiguous_analysis else "matched",
                        )
                    )

        # Resolve overlapping proposal components as a whole. Exact aliases
        # outrank morphology; otherwise only maximal phrases can claim a span.
        remaining = set(range(len(proposals)))
        decisions: list[NormalizationDecision] = []
        replacements: list[NormalizationDecision] = []
        edits: list[tuple[int, int, str]] = []
        while remaining:
            seed = remaining.pop()
            component = {seed}
            changed = True
            while changed:
                changed = False
                for index in tuple(remaining):
                    p = proposals[index]
                    if any(
                        p.start < proposals[j].end and proposals[j].start < p.end for j in component
                    ):
                        remaining.remove(index)
                        component.add(index)
                        changed = True
            group = [proposals[index] for index in sorted(component)]
            exact = [p for p in group if p.method == "exact"]
            pool = exact if exact else group
            longest = max(p.end - p.start for p in pool)
            maximal = [p for p in pool if p.end - p.start == longest]
            starts = {p.start for p in maximal}
            ends = {p.end for p in maximal}
            # Overlapping maximal phrases with distinct spans are unresolved;
            # protect their union so no shorter subphrase is partially changed.
            if len(starts) != 1 or len(ends) != 1:
                first = min(p.start for p in group)
                last = max(p.end for p in group)
                a, b = tokens[first].start, tokens[last - 1].end
                found = tuple(dict.fromkeys(p.candidate for p in maximal))
                decisions.append(
                    NormalizationDecision(
                        "overlapping_matches",
                        "exact" if exact else "morphology",
                        (a, b),
                        text[a:b],
                        found,
                    )
                )
                continue
            first, last = next(iter(starts)), next(iter(ends))
            a, b = tokens[first].start, tokens[last - 1].end
            found = tuple(dict.fromkeys(p.candidate for p in maximal))
            method = "exact" if exact else "morphology"
            if any(p.reason == "ambiguous_analysis" for p in maximal):
                decisions.append(
                    NormalizationDecision(
                        "ambiguous_analysis",
                        method,
                        (a, b),
                        text[a:b],
                        found,
                    )
                )
                continue
            if len(found) > 1:
                decisions.append(
                    NormalizationDecision(
                        "ambiguous_candidates",
                        method,
                        (a, b),
                        text[a:b],
                        found,
                    )
                )
                continue
            candidate = found[0]
            source = text[a:b]
            # Restore a unique alias's ё spelling too: HA must receive the real
            # alias, not an е-equivalent that only our matcher accepts. Casing
            # differences alone never require a rewrite.
            restore_yo = source.casefold() != candidate.name.casefold() and _fold(source) == _fold(
                candidate.name
            )
            if method == "morphology" or restore_yo:
                decision = NormalizationDecision("matched", method, (a, b), source, found)
                decisions.append(decision)
                if source != candidate.name:
                    replacements.append(decision)
                    edits.append((a, b, candidate.name))

        normalized = text
        for start, end, value in sorted(edits, reverse=True):
            normalized = normalized[:start] + value + normalized[end:]
        return NormalizationResult(
            text,
            normalized,
            tuple(sorted(replacements, key=lambda decision: decision.span)),
            tuple(sorted(decisions, key=lambda decision: decision.span)),
        )
