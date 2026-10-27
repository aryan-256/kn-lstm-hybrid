"""
N-gram language model: count tables + 5 smoothing variants (Task 1).

Shared counting + 3 smoothing families:
  - Add-k (k in {1, 0.5, 0.1})  -- 3 variants, same formula, different k
  - Good-Turing (r <= 5 re-estimated, backoff for r=0 -- see PLAN.md 3.3)
  - Interpolated Kneser-Ney (d=0.75 fixed, + outer lambda blend w/ unigram,
    lambda grid-searched on validation -- see PLAN.md 3.4)

Design: build ONE set of n-gram count tables per order (1..4) from the train
scoring-stream, then every smoothing method reads from those same tables.
Never recount per smoothing variant.

All models expose the same two methods so Task 1's PPL loop, Task 3's hybrid,
and the BLiMP scorer can all treat them interchangeably:
  - logprob(context: tuple[str,...], word: str) -> float   (natural log)
  - a `order` attribute (the n in n-gram, e.g. 3 for trigram)
"""

import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Count tables
# ---------------------------------------------------------------------------

class NgramCounts:
    """Count tables for orders 1..max_order, built from one token stream.

    counts[n] is a dict: n-gram tuple (length n) -> count
    context_counts[n] is a dict: (n-1)-gram tuple -> total count of that
        context (i.e. sum of counts[n] over all n-grams sharing that prefix)
        -- stored separately to avoid recomputing sums repeatedly.
    """

    def __init__(self, stream: List[str], max_order: int = 4):
        self.max_order = max_order
        self.counts: Dict[int, Dict[Tuple[str, ...], int]] = {
            n: defaultdict(int) for n in range(1, max_order + 1)
        }
        self.vocab_tokens = set()

        for n in range(1, max_order + 1):
            for i in range(len(stream) - n + 1):
                ngram = tuple(stream[i:i + n])
                self.counts[n][ngram] += 1

        self.vocab_tokens = set(stream)
        self.vocab_size = len(self.vocab_tokens)

        # context_counts[n][context] = sum over all words w of counts[n][context + (w,)]
        # equivalently: context_counts[n] == counts[n-1] restricted to contexts
        # that actually appear as an (n-1)-gram -- but safer to just recompute
        # from counts[n] directly so it's always consistent with counts[n].
        self.context_counts: Dict[int, Dict[Tuple[str, ...], int]] = {
            n: defaultdict(int) for n in range(1, max_order + 1)
        }
        for n in range(1, max_order + 1):
            for ngram, c in self.counts[n].items():
                context = ngram[:-1]
                self.context_counts[n][context] += c

        # Count-of-counts tables per order, for Good-Turing: N_r = number of
        # distinct n-grams of this order that occurred exactly r times.
        self.count_of_counts: Dict[int, Counter] = {}
        for n in range(1, max_order + 1):
            self.count_of_counts[n] = Counter(self.counts[n].values())


# ---------------------------------------------------------------------------
# Add-k smoothing
# ---------------------------------------------------------------------------

class AddKModel:
    """Add-k (Laplace generalization) smoothing for a fixed n-gram order."""

    def __init__(self, ngram_counts: NgramCounts, order: int, k: float):
        self.nc = ngram_counts
        self.order = order
        self.k = k
        self.V = ngram_counts.vocab_size

    def logprob(self, context: Tuple[str, ...], word: str) -> float:
        context = context[-(self.order - 1):] if self.order > 1 else ()
        ngram = context + (word,)
        c_ngram = self.nc.counts[self.order].get(ngram, 0)
        c_context = self.nc.context_counts[self.order].get(context, 0)
        prob = (c_ngram + self.k) / (c_context + self.k * self.V)
        return math.log(prob)


# ---------------------------------------------------------------------------
# Good-Turing smoothing
# ---------------------------------------------------------------------------

class GoodTuringModel:
    """Good-Turing discounting for r <= gt_max; backoff to (order-1) MLE for
    unseen (r=0) n-grams (see PLAN.md assumption 3.3).

    Adjusted count: r* = (r+1) * N_{r+1} / N_r  for r in [1, gt_max]
    For r=0 (unseen n-gram): no direct GT estimate exists (N_0 is not a count
    we observe in the usual formula), so we fall back to a simple backoff:
    distribute a held-out mass (estimated as N_1 / N_total, i.e. the
    Good-Turing estimate for total unseen mass) proportionally via the
    (order-1)-gram MLE distribution. This is a standard simple Katz-style
    backoff, not full Katz backoff with its own discounting at every level --
    kept deliberately simple per the project's "minimal effort, ship it"
    principle, documented explicitly in the report.
    """

    def __init__(self, ngram_counts: NgramCounts, order: int, gt_max: int = 5):
        self.nc = ngram_counts
        self.order = order
        self.gt_max = gt_max
        self.V = ngram_counts.vocab_size

        coc = ngram_counts.count_of_counts[order]
        self.N_total = sum(r * n for r, n in coc.items())  # total token count at this order

        # Precompute adjusted counts r* for r = 1..gt_max
        self.r_star = {}
        for r in range(1, gt_max + 1):
            Nr = coc.get(r, 0)
            Nr1 = coc.get(r + 1, 0)
            if Nr == 0:
                self.r_star[r] = float(r)  # no data to adjust, fall back to raw count
            else:
                self.r_star[r] = (r + 1) * Nr1 / Nr if Nr1 > 0 else float(r)

        # Mass reserved for unseen n-grams (r=0): N_1 / N_total (standard GT)
        N1 = coc.get(1, 0)
        self.unseen_mass = (N1 / self.N_total) if self.N_total > 0 else 1e-10

        # Lower-order model for backing off unseen n-grams (order-1 MLE,
        # or uniform if order==1)
        if order > 1:
            self._lower = GoodTuringModel(ngram_counts, order - 1, gt_max) \
                if order - 1 >= 1 else None
        else:
            self._lower = None

    def _raw_count(self, context, word):
        ngram = context + (word,)
        return self.nc.counts[self.order].get(ngram, 0)

    def logprob(self, context: Tuple[str, ...], word: str) -> float:
        context = context[-(self.order - 1):] if self.order > 1 else ()
        r = self._raw_count(context, word)
        c_context = self.nc.context_counts[self.order].get(context, 0)

        if r == 0:
            # Backoff: unseen_mass spread over OOV-given-context, approximated
            # via lower-order distribution (or uniform at unigram level).
            if self.order == 1 or self._lower is None:
                prob = self.unseen_mass / max(self.V, 1)
            else:
                lower_context = context[1:] if len(context) > 0 else ()
                lower_logprob = self._lower.logprob(lower_context, word)
                prob = self.unseen_mass * math.exp(lower_logprob)
            prob = max(prob, 1e-12)  # floor to avoid log(0)
            return math.log(prob)
        elif r <= self.gt_max:
            r_adj = self.r_star[r]
            prob = r_adj / c_context if c_context > 0 else r_adj / max(self.V, 1)
            return math.log(max(prob, 1e-12))
        else:
            # r > gt_max: use raw MLE, undiscounted (standard GT practice --
            # only low counts are unreliable enough to need discounting)
            prob = r / c_context
            return math.log(prob)


# ---------------------------------------------------------------------------
# Interpolated Kneser-Ney (+ outer unigram blend, per PLAN.md 3.4)
# ---------------------------------------------------------------------------

class KneserNeyModel:
    """Standard recursive interpolated KN with fixed discount d, PLUS an
    outer global interpolation with the unigram distribution:

        P_final(w | context) = lam * P_KN(w | context) + (1-lam) * P_unigram(w)

    lam is tuned by the caller via grid search on validation perplexity (see
    `tune_kn_lambda` below) and passed in at construction time.

    See PLAN.md assumption 3.4 for why this two-part structure was chosen to
    resolve the spec's ambiguous "d=0.75 AND separately tune lambda" wording.
    """

    def __init__(self, ngram_counts: NgramCounts, order: int, d: float = 0.75,
                 lam: float = 1.0):
        self.nc = ngram_counts
        self.order = order
        self.d = d
        self.lam = lam
        self.V = ngram_counts.vocab_size

        # Precompute unigram MLE distribution for the outer blend.
        unigram_counts = ngram_counts.counts[1]
        total_unigrams = sum(unigram_counts.values())
        self._unigram_prob = {
            w: c / total_unigrams for (w,), c in unigram_counts.items()
        }
        self._unigram_floor = 1e-10  # for words with zero unigram count (shouldn't happen if w in vocab)

        # Precompute continuation counts for true KN lower-order smoothing:
        # continuation count of a word w at order n = number of distinct
        # (n)-gram contexts of order (n+1) in which w appears as the last
        # token. Needed for orders < max_order when used as the recursive
        # backoff base. We precompute this once per order pair actually used.
        self._continuation_cache = {}

        # PERFORMANCE: precompute, for every order from 1..self.order, the
        # number of distinct words following each context -- i.e.
        # distinct_following_table[n][context] = |{w : (context,w) in counts[n]}|.
        # Built once here (O(number of n-grams), not per logprob() call) so
        # `_distinct_following` below is an O(1) dict lookup instead of a full
        # linear scan over the n-gram table on every single token scored.
        # This matters a great deal at real corpus scale (~1.2M tokens):
        # without this, KN perplexity would be impractically slow.
        self._distinct_following_table: Dict[int, Dict[Tuple[str, ...], int]] = {}
        for n in range(1, self.order + 1):
            table = defaultdict(set)
            for ngram in ngram_counts.counts[n]:
                prefix, last = ngram[:-1], ngram[-1]
                table[prefix].add(last)
            self._distinct_following_table[n] = {k: len(v) for k, v in table.items()}

    def _continuation_prob(self, word: str, order: int) -> float:
        """P_continuation(word) at the given order: fraction of distinct
        (order+1)-gram types that end in `word`, used as the lowest-order KN
        base distribution (standard interpolated KN definition)."""
        cache_key = order
        if cache_key not in self._continuation_cache:
            higher = self.nc.counts.get(order + 1, {})
            # distinct contexts (order-grams) that precede each distinct
            # continuation word
            continuation_counts = defaultdict(set)
            for ngram in higher:
                prefix, last = ngram[:-1], ngram[-1]
                continuation_counts[last].add(prefix)
            total_types = sum(len(v) for v in continuation_counts.values())
            self._continuation_cache[cache_key] = (
                {w: len(ctxs) for w, ctxs in continuation_counts.items()},
                total_types,
            )
        type_counts, total_types = self._continuation_cache[cache_key]
        if total_types == 0:
            return 1.0 / max(self.V, 1)
        return type_counts.get(word, 0) / total_types

    def _kn_recursive(self, context: Tuple[str, ...], word: str, order: int) -> float:
        """Standard recursive interpolated KN probability (returns a plain
        probability, not log), for a given order, recursing down to the
        continuation-based unigram base case."""
        if order == 1:
            return self._continuation_prob(word, 1)

        ctx = context[-(order - 1):] if order > 1 else ()
        ngram = ctx + (word,)
        c_ngram = self.nc.counts[order].get(ngram, 0)
        c_context = self.nc.context_counts[order].get(ctx, 0)

        if c_context == 0:
            # Never saw this context at all -- fall back entirely to lower order
            return self._kn_recursive(context, word, order - 1)

        # Number of distinct word types following this context (for the
        # normalizing backoff weight lambda(context))
        # We need: distinct continuations of `ctx` at this order.
        distinct_continuations = self._distinct_following(ctx, order)

        discounted = max(c_ngram - self.d, 0.0) / c_context
        backoff_weight = (self.d * distinct_continuations) / c_context
        lower = self._kn_recursive(context, word, order - 1)

        return discounted + backoff_weight * lower

    def _distinct_following(self, context: Tuple[str, ...], order: int) -> int:
        """Number of distinct words w such that (context + (w,)) was seen at
        this order -- used for the KN backoff normalizing weight. O(1)
        lookup into the table precomputed in __init__ (see PERFORMANCE note
        there) -- do not revert this to a linear scan."""
        return self._distinct_following_table.get(order, {}).get(context, 0)

    def logprob(self, context: Tuple[str, ...], word: str) -> float:
        p_kn = self._kn_recursive(context, word, self.order)
        p_uni = self._unigram_prob.get(word, self._unigram_floor)
        prob = self.lam * p_kn + (1 - self.lam) * p_uni
        return math.log(max(prob, 1e-12))


# ---------------------------------------------------------------------------
# Perplexity evaluation (shared by all 5 variants)
# ---------------------------------------------------------------------------

def perplexity(model, stream: List[str], order: int) -> float:
    """Compute perplexity of `model` over `stream`, using sliding windows of
    the given order. `stream` should already include <s>/</s> boundaries
    (i.e. it's the output of tokenizer.build_scoring_stream).

    Convention (PLAN.md 3.2): every token is scored INCLUDING </s>, EXCLUDING
    <s> as a predicted token (it only ever appears as context).
    """
    from tokenizer import BOS

    total_logprob = 0.0
    n_tokens = 0

    for i in range(len(stream)):
        word = stream[i]
        if word == BOS:
            continue  # never score <s> itself, it's context-only
        context_start = max(0, i - (order - 1))
        context = tuple(stream[context_start:i])
        total_logprob += model.logprob(context, word)
        n_tokens += 1

    avg_neg_logprob = -total_logprob / n_tokens
    return math.exp(avg_neg_logprob)


# ---------------------------------------------------------------------------
# KN lambda tuning (grid search on validation, per PLAN.md 3.4)
# ---------------------------------------------------------------------------

def tune_kn_lambda(ngram_counts: NgramCounts, order: int, val_stream: List[str],
                    d: float = 0.75, grid=None) -> Tuple[float, float]:
    """Grid search lam over `grid` (default {0.1,...,0.9}), minimizing
    validation perplexity. Returns (best_lam, best_val_ppl)."""
    if grid is None:
        grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    best_lam, best_ppl = None, float("inf")
    for lam in grid:
        model = KneserNeyModel(ngram_counts, order, d=d, lam=lam)
        ppl = perplexity(model, val_stream, order)
        if ppl < best_ppl:
            best_lam, best_ppl = lam, ppl
    return best_lam, best_ppl


if __name__ == "__main__":
    # Smoke test on fabricated data -- NOT real DailyDialog, just verifies
    # the module runs end-to-end and produces sane (finite, >1) perplexities
    # before we touch Colab.
    import sys
    sys.path.insert(0, "/home/claude")
    from tokenizer import build_scoring_stream, Vocab, build_vocab, tokenize, join_dialogue

    fake_dialogues = [
        ["I like cats.", "Cats are nice.", "Do you like dogs too?"],
        ["I like dogs.", "Dogs are loyal.", "Cats are independent."],
        ["Do you like cats or dogs?", "I like both animals."],
        ["Cats are independent.", "Dogs are loyal and nice."],
    ] * 5  # repeat for slightly less sparse counts in this tiny smoke test

    token_lists = [tokenize(join_dialogue(d)) for d in fake_dialogues]
    vocab = build_vocab(token_lists, max_size=1000, min_count=1)
    stream = build_scoring_stream(fake_dialogues, vocab)

    print("Stream length:", len(stream), "Vocab size:", len(vocab))

    nc = NgramCounts(stream, max_order=4)

    print("\n--- Add-k (k=1), trigram ---")
    addk = AddKModel(nc, order=3, k=1.0)
    print("PPL:", perplexity(addk, stream, order=3))

    print("\n--- Good-Turing, trigram ---")
    gt = GoodTuringModel(nc, order=3, gt_max=5)
    print("PPL:", perplexity(gt, stream, order=3))

    print("\n--- Kneser-Ney, trigram (lam=1.0, i.e. pure KN, no unigram blend) ---")
    kn = KneserNeyModel(nc, order=3, d=0.75, lam=1.0)
    print("PPL:", perplexity(kn, stream, order=3))

    print("\n--- KN lambda tuning (using the same stream as 'validation' -- smoke test only) ---")
    best_lam, best_ppl = tune_kn_lambda(nc, order=3, val_stream=stream)
    print("Best lambda:", best_lam, "Best PPL:", best_ppl)

    print("\nAll smoke tests ran without error.")
