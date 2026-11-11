"""Conservative, explainable morphology-based entity-name normalization."""

from dataclasses import dataclass
from itertools import pairwise
from threading import RLock

from .matching import (
    MAX_CANDIDATES,
    MAX_COMPARISONS,
    MAX_NAME_CHARS,
    MAX_NAME_TOKENS,
    MAX_QUERY_TOKENS,
    MAX_TEXT_CHARS,
    FuzzySettings,
    normalized_similarity,
)
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
    score: float | None = None
    runner_up_score: float | None = None


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

    def __init__(self, morphology: Morphology, fuzzy: FuzzySettings | None = None) -> None:
        self._morphology = morphology
        self._fuzzy = fuzzy if fuzzy is not None else FuzzySettings()
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
        if self._fuzzy.enabled:
            fuzzy_decisions, fuzzy_edits = self._fuzzy_normalize(text, tokens, prepared, proposals)
            decisions.extend(fuzzy_decisions)
            replacements.extend(
                d
                for d in fuzzy_decisions
                if d.reason == "matched" and text[d.span[0] : d.span[1]] != d.candidates[0].name
            )
            edits.extend(fuzzy_edits)
            normalized = text
            for start, end, value in sorted(edits, reverse=True):
                normalized = normalized[:start] + value + normalized[end:]
        return NormalizationResult(
            text,
            normalized,
            tuple(sorted(replacements, key=lambda decision: decision.span)),
            tuple(sorted(decisions, key=lambda decision: decision.span)),
        )

    def _fuzzy_normalize(
        self,
        text: str,
        tokens: tuple[_Token, ...],
        prepared: tuple[_Prepared, ...],
        proposals: list[_Proposal],
    ) -> tuple[list[NormalizationDecision], list[tuple[int, int, str]]]:
        """Match only unclaimed spans; all earlier proposals protect their span."""
        comparison_budget = sum(max(0, len(tokens) - len(target.tokens) + 1) for target in prepared)
        if (
            len(text) > MAX_TEXT_CHARS
            or len(tokens) > MAX_QUERY_TOKENS
            or len(prepared) > MAX_CANDIDATES
            or comparison_budget > MAX_COMPARISONS
            or any(
                len(p.tokens) > MAX_NAME_TOKENS or len(p.candidate.name) > MAX_NAME_CHARS
                for p in prepared
            )
        ):
            return [_resource_limit_decision(text)], []
        protected = [(p.start, p.end) for p in proposals]
        comparisons = 0
        found: list[tuple[int, int, NameCandidate, float]] = []
        for i in range(len(tokens)):
            for target in prepared:
                count = len(target.tokens)
                if i + count > len(tokens) or any(
                    i < end and start < i + count for start, end in protected
                ):
                    continue
                phrase = tokens[i : i + count]
                if _phrase_separators(text, phrase) != target.separators:
                    continue
                # A neighboring model/number is a likely qualifier belonging
                # to the name, not safe context to leave behind after a partial
                # fuzzy rewrite (e.g. "термостта 2").
                if (i > 0 and _literal_qualifier(tokens[i - 1].value)) or (
                    i + count < len(tokens) and _literal_qualifier(tokens[i + count].value)
                ):
                    continue
                source_words = tuple(_fold(t.value) for t in phrase)
                if source_words == target.tokens:
                    continue
                # Literal tokens (including embedded model digits) are exact.
                changed = [
                    n
                    for n, (src, dst) in enumerate(zip(phrase, target.tokens))
                    if _fold(src.value) != dst
                ]
                if not changed or any(
                    not (_cyrillic(phrase[n].value) and _cyrillic(target.tokens[n]))
                    for n in changed
                ):
                    continue
                safe = True
                for n in changed:
                    source = phrase[n].value
                    if len(source) < 6 or len(target.tokens[n]) < 6:
                        safe = False
                        break
                    analyses = tuple(
                        a for a in self._morphology.analyze(source) if a.known and a.score >= 0.05
                    )
                    target_word = _tokens(target.candidate.name)[n].value
                    target_analyses = tuple(
                        a
                        for a in self._morphology.analyze(target_word)
                        if a.known and a.score >= 0.05
                    )
                    # Both ends must be safe nominal readings. Distinct known
                    # lemmas are not spelling variants (машина != машинка).
                    if (
                        _unsafe_lexical_reading(analyses)
                        or _unsafe_lexical_reading(target_analyses)
                        or _known_lemmas_conflict(analyses, target_analyses)
                    ):
                        safe = False
                        break
                if not safe:
                    continue
                comparisons += 1
                if comparisons > MAX_COMPARISONS:
                    return [_resource_limit_decision(text)], []
                left = " ".join(source_words)
                right = " ".join(target.tokens)
                score = normalized_similarity(left, right)
                found.append((i, i + count, target.candidate, score))
        # Collapse aliases by entity before measuring confidence. A name shared
        # by distinct entities is deliberately not eligible for selection.
        groups: dict[tuple[int, int], dict[str, tuple[NameCandidate, float]]] = {}
        for start, end, candidate, score in found:
            group = groups.setdefault((start, end), {})
            old = group.get(candidate.entity_id)
            if old is None or (score, candidate.name.casefold(), candidate.name) > (
                old[1],
                old[0].name.casefold(),
                old[0].name,
            ):
                group[candidate.entity_id] = (candidate, score)
        decisions: list[NormalizationDecision] = []
        edits: list[tuple[int, int, str]] = []
        remaining = set(groups)
        while remaining:
            seed = remaining.pop()
            component = {seed}
            changed_component = True
            while changed_component:
                changed_component = False
                for span in tuple(remaining):
                    if any(span[0] < end and start < span[1] for start, end in component):
                        remaining.remove(span)
                        component.add(span)
                        changed_component = True
            longest = max(end - start for start, end in component)
            maximal_spans = sorted(span for span in component if span[1] - span[0] == longest)
            if len(maximal_spans) > 1:
                first = min(start for start, _ in component)
                last = max(end for _, end in component)
                a, b = tokens[first].start, tokens[last - 1].end
                candidates_at_max = tuple(
                    sorted(
                        {
                            candidate
                            for span in maximal_spans
                            for candidate, _ in groups[span].values()
                        },
                        key=lambda candidate: (
                            candidate.entity_id,
                            candidate.name.casefold(),
                            candidate.name,
                        ),
                    )
                )
                decisions.append(
                    NormalizationDecision(
                        "overlapping_matches", "fuzzy", (a, b), text[a:b], candidates_at_max
                    )
                )
                continue
            start, end = maximal_spans[0]
            group = groups[(start, end)]
            ranked = sorted(
                group.values(), key=lambda x: (-x[1], x[0].entity_id, x[0].name.casefold())
            )
            best, score = ranked[0]
            runner = ranked[1][1] if len(ranked) > 1 else None
            a, b = tokens[start].start, tokens[end - 1].end
            if score < self._fuzzy.score_cutoff:
                if score >= self._fuzzy.score_cutoff - 5.0:
                    decisions.append(
                        NormalizationDecision(
                            "below_threshold", "fuzzy", (a, b), text[a:b], (best,), score, runner
                        )
                    )
                continue
            # If the chosen spelling is an exact alias for another entity,
            # fuzzy ranking must not break that exact-name ambiguity.
            exact_owners = tuple(
                dict.fromkeys(
                    p.candidate for p in prepared if _fold(p.candidate.name) == _fold(best.name)
                )
            )
            if len({candidate.entity_id for candidate in exact_owners}) > 1:
                decisions.append(
                    NormalizationDecision(
                        "ambiguous_candidates",
                        "fuzzy",
                        (a, b),
                        text[a:b],
                        exact_owners,
                        score,
                        runner,
                    )
                )
                continue
            if runner is not None and (score - runner) < self._fuzzy.margin:
                decisions.append(
                    NormalizationDecision(
                        "ambiguous_candidates",
                        "fuzzy",
                        (a, b),
                        text[a:b],
                        tuple(x[0] for x in ranked),
                        score,
                        runner,
                    )
                )
            else:
                decisions.append(
                    NormalizationDecision(
                        "matched", "fuzzy", (a, b), text[a:b], (best,), score, runner
                    )
                )
                edits.append((a, b, best.name))
        return decisions, edits


def _literal_qualifier(value: str) -> bool:
    """Whether a neighboring token looks like a model/number qualifier."""
    return any(char.isdigit() for char in value) or any(
        "a" <= char.casefold() <= "z" for char in value
    )


def _unsafe_lexical_reading(analyses: tuple[object, ...]) -> bool:
    """Fail closed on credible verb/function readings and competing lemmas."""
    return bool(analyses) and (
        any(a.pos in _NON_NOMINAL or a.pos not in _NOMINAL for a in analyses)
        or len({_fold(a.lemma) for a in analyses}) > 1
    )


def _known_lemmas_conflict(source: tuple[object, ...], target: tuple[object, ...]) -> bool:
    source_lemmas = {_fold(a.lemma) for a in source}
    target_lemmas = {_fold(a.lemma) for a in target}
    return bool(source_lemmas and target_lemmas and source_lemmas.isdisjoint(target_lemmas))


def _resource_limit_decision(text: str) -> NormalizationDecision:
    return NormalizationDecision("resource_limit", "fuzzy", (0, len(text)), text, ())
