"""
Task 3: linear interpolation of Task 1's 4-gram KN model and Task 2's
AWD-LSTM, per PLAN.md Section 5 (Task 3).

P_hybrid(w_i | context) = lambda * P_KN(w_i | context) + (1-lambda) * P_AWD(w_i | context)

Core requirement (PLAN.md 3.7): both models must be scored over the EXACT
SAME sequence of positions for this interpolation to be well-defined. We
reuse the same scoring-stream convention from tokenizer.build_scoring_stream
that both Task 1 and Task 2 already evaluate perplexity with, so this is
already guaranteed by construction as long as both models are called on the
same `stream` list.

Design: extract per-token log-probs from EACH model SEPARATELY into plain
Python lists of floats (same length, same order, position i in one list
corresponds to the same token as position i in the other). Then interpolate
in plain Python/numpy -- simple, easy to verify, and fast enough at this
corpus scale (test stream is ~108K tokens).
"""

import math
from typing import List

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Per-token log-prob extraction -- n-gram side
# ---------------------------------------------------------------------------

def ngram_token_logprobs(model, stream: List[str], order: int) -> List[float]:
    """Returns one log-prob per SCORED token in `stream` (i.e. every token
    except <s>, matching the convention in ngram_lm.perplexity /
    tokenizer.build_scoring_stream). Length == number of non-<s> tokens in
    `stream`.
    """
    from tokenizer import BOS

    logprobs = []
    for i in range(len(stream)):
        word = stream[i]
        if word == BOS:
            continue
        context_start = max(0, i - (order - 1))
        context = tuple(stream[context_start:i])
        logprobs.append(model.logprob(context, word))
    return logprobs


# ---------------------------------------------------------------------------
# Per-token log-prob extraction -- LSTM side
# ---------------------------------------------------------------------------

def lstm_token_logprobs(model, stream: List[str], vocab, bos_id: int, device,
                         chunk_size: int = 2048) -> List[float]:
    """Returns one log-prob per SCORED token in `stream` (every token except
    <s>), in the SAME order as ngram_token_logprobs would produce for the
    same stream -- i.e. skip the log-prob that would have been assigned to
    predicting a <s> token (since <s> is never a prediction target, only
    ever appears as context), exactly mirroring ngram_lm.perplexity's and
    awd_lstm.evaluate_stream_exact's shared convention.

    This is almost identical to awd_lstm.evaluate_stream_exact, except it
    RETURNS the full list of per-token log-probs instead of collapsing them
    into a single perplexity number -- we need the individual values here to
    interpolate position-by-position with the n-gram model's values.
    """
    ids = vocab.encode(stream)
    data = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)  # (N, 1)
    hidden = model.init_hidden(1, device)

    logprobs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, chunk_size):
            seq_len = min(chunk_size, data.size(0) - 1 - i)
            inp = data[i:i + seq_len]
            tgt = data[i + 1:i + 1 + seq_len].squeeze(1)

            logits, hidden = model(inp, hidden)
            from awd_lstm import repackage_hidden
            hidden = repackage_hidden(hidden)

            log_probs = F.log_softmax(logits.squeeze(1), dim=-1)
            target_lps = log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)

            valid_mask = (tgt != bos_id)
            valid_lps = target_lps[valid_mask]
            logprobs.extend(valid_lps.cpu().tolist())

    return logprobs


# ---------------------------------------------------------------------------
# Interpolation + perplexity from precomputed per-token log-probs
# ---------------------------------------------------------------------------

def interpolate_perplexity(kn_logprobs: List[float], lstm_logprobs: List[float],
                            lam: float) -> float:
    """Given two aligned lists of per-token NATURAL-LOG probabilities (same
    length, position i = same token in both), compute the perplexity of the
    linearly-interpolated hybrid:

        P_hybrid(token_i) = lam * exp(kn_logprobs[i]) + (1-lam) * exp(lstm_logprobs[i])

    Interpolation happens in PROBABILITY space (not log space) -- this is
    standard for linear interpolation of LMs, since the whole point is a
    weighted average of probabilities, not of log-probabilities (which
    would be a geometric-mean-like blend instead, a different and NOT what
    the spec's formula P_hybrid = lambda*P_KN + (1-lambda)*P_AWD specifies).
    """
    assert len(kn_logprobs) == len(lstm_logprobs), (
        f"Misaligned token streams: KN has {len(kn_logprobs)} scored tokens, "
        f"LSTM has {len(lstm_logprobs)} -- these MUST match exactly for "
        f"interpolation to be valid. Check that both were computed from the "
        f"identical `stream` list."
    )

    total_logprob = 0.0
    n = len(kn_logprobs)
    for kn_lp, lstm_lp in zip(kn_logprobs, lstm_logprobs):
        p_kn = math.exp(kn_lp)
        p_lstm = math.exp(lstm_lp)
        p_hybrid = lam * p_kn + (1 - lam) * p_lstm
        p_hybrid = max(p_hybrid, 1e-12)  # floor to avoid log(0)
        total_logprob += math.log(p_hybrid)

    return math.exp(-total_logprob / n)


def tune_hybrid_lambda(kn_val_logprobs: List[float], lstm_val_logprobs: List[float],
                        grid=None):
    """Grid search lambda in [0.0, 1.0] (step 0.1 by default, per spec),
    minimizing validation perplexity. Returns (best_lambda, best_val_ppl,
    all_results) where all_results is a list of (lambda, val_ppl) pairs for
    plotting the full curve.
    """
    if grid is None:
        grid = [round(x * 0.1, 1) for x in range(0, 11)]  # 0.0, 0.1, ..., 1.0

    results = []
    best_lam, best_ppl = None, float("inf")
    for lam in grid:
        ppl = interpolate_perplexity(kn_val_logprobs, lstm_val_logprobs, lam)
        results.append((lam, ppl))
        if ppl < best_ppl:
            best_lam, best_ppl = lam, ppl
    return best_lam, best_ppl, results


if __name__ == "__main__":
    # Smoke test with fabricated aligned log-prob lists (no torch/real models
    # needed for this part -- pure arithmetic verification).
    import random
    random.seed(42)

    # Simulate: KN is decent, LSTM is better, on a small fake "stream"
    n_tokens = 200
    kn_lps = [-random.uniform(2.0, 5.0) for _ in range(n_tokens)]
    lstm_lps = [-random.uniform(1.0, 4.0) for _ in range(n_tokens)]  # systematically better

    print("KN-only perplexity (lambda=1.0):", interpolate_perplexity(kn_lps, lstm_lps, 1.0))
    print("LSTM-only perplexity (lambda=0.0):", interpolate_perplexity(kn_lps, lstm_lps, 0.0))

    best_lam, best_ppl, all_results = tune_hybrid_lambda(kn_lps, lstm_lps)
    print("\nFull lambda sweep:")
    for lam, ppl in all_results:
        print(f"  lambda={lam}: PPL={ppl:.3f}")
    print(f"\nBest lambda: {best_lam}, Best PPL: {best_ppl:.3f}")

    # Sanity check: lambda=1.0 result should EXACTLY match a pure KN-only
    # perplexity computed directly (not through the hybrid formula), since
    # at lambda=1.0 the LSTM term is multiplied by (1-1.0)=0.
    direct_kn_only = math.exp(-sum(kn_lps) / len(kn_lps))
    hybrid_at_1 = interpolate_perplexity(kn_lps, lstm_lps, 1.0)
    print(f"\nSanity check -- direct KN-only PPL: {direct_kn_only:.6f}, "
          f"hybrid(lambda=1.0) PPL: {hybrid_at_1:.6f}, "
          f"match: {abs(direct_kn_only - hybrid_at_1) < 1e-6}")

    # Test misalignment detection
    try:
        interpolate_perplexity(kn_lps, lstm_lps[:-1], 0.5)
        print("ERROR: misalignment was NOT detected!")
    except AssertionError as e:
        print("\nMisalignment correctly detected:", str(e)[:80], "...")
