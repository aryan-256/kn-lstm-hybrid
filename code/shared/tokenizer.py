"""
Shared tokenizer + vocabulary module for NLP Assignment 2, Project 1 (DailyDialog).

Used identically by Task 1 (n-grams), Task 2 (AWD-LSTM), Task 3 (hybrid), and the
BLiMP evaluation harness. Build the vocab ONCE from DailyDialog's train split and
reuse the same Vocab object everywhere — never rebuild it per-task.

Design choices (see PLAN.md Section 3 for the full rationale):
- Punctuation-aware whitespace tokenizer: splits words from attached punctuation
  (e.g. "hello!" -> ["hello", "!"]) but does not split contractions internally
  (e.g. "don't" stays as ["don't"]) since the spec doesn't ask for that level of
  linguistic splitting and DailyDialog is casual/contracted speech.
- Lowercase everything.
- Vocabulary built from train only, capped at 30,000 most frequent tokens.
  Tokens occurring <2 times in train, and any OOV token seen later, map to <UNK>.
- Special tokens: <unk>, <s>, </s>, <pad> (pad reserved for Task 2 batching; not
  used by Task 1's counting).
"""

import re
import json
from collections import Counter
from typing import List, Dict, Iterable


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

# Matches: contractions/words with internal apostrophes (don't, it's) as one
# token, plain word tokens, or any single punctuation/symbol character.
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+(?:\.[0-9]+)?|[^\sA-Za-z0-9]")


def _normalize_apostrophes(text: str) -> str:
    """Collapse all Unicode apostrophe/quote variants to a single straight
    ASCII apostrophe (') before any other processing.

    Real DailyDialog text mixes the straight apostrophe (U+0027, ') with the
    Unicode "right single quotation mark" (U+2019, ’) and occasionally other
    look-alikes, inconsistently -- sometimes within the same split. Left
    unnormalized, contraction detection silently fails for every curly-quote
    instance (e.g. "I \u2019 m here" would NOT be caught by the spacing-fix
    regex below, since that regex only matches the ASCII character), leaking
    junk single-character tokens ('m', 't', 's', etc.) into the vocabulary.
    This must run BEFORE `_normalize_spacing`.
    """
    return text.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")


def _normalize_spacing(text: str) -> str:
    """Fix known DailyDialog spacing artifacts before tokenizing.

    Real DailyDialog utterances frequently have spaces on BOTH sides of the
    apostrophe inside a contraction, e.g. "Let ' s go" instead of "Let's go".
    Left alone, this breaks contraction detection entirely: the word, the
    apostrophe, and the suffix end up as three spurious separate tokens
    ("let", "'", "s") instead of one "let's" token, polluting the vocabulary
    with junk single-character tokens. This regex collapses "<word> ' <sfx>"
    into "<word>'<sfx>" whenever the trailing part looks like a common
    contraction suffix (s, t, re, ve, ll, d, m), so "let ' s" -> "let's",
    "they ' re" -> "they're", etc. Requires a preceding word character so it
    doesn't fire on a bare leading apostrophe/quote mark.

    This is a documented, deliberate preprocessing deviation -- state it in
    the report's preprocessing subsection.
    """
    text = re.sub(
        r"(\w)\s*'\s+(s|t|re|ve|ll|d|m)\b",
        r"\1'\2",
        text,
        flags=re.IGNORECASE,
    )
    return text


def tokenize(text: str) -> List[str]:
    """Punctuation-aware whitespace tokenizer + lowercase.

    Applies DailyDialog-specific spacing normalization (see
    `_normalize_spacing`) before the regex tokenizer runs.

    Example:
        >>> tokenize("Hello! Don't you think it's 3.5 degrees out?")
        ['hello', '!', "don't", 'you', 'think', "it's", '3.5', 'degrees', 'out', '?']
        >>> tokenize("Good . Let ' s go now .")
        ['good', '.', "let's", 'go', 'now', '.']
        >>> tokenize("I \u2019 m here .")
        ["i'm", 'here', '.']
    """
    text = _normalize_apostrophes(text)
    text = _normalize_spacing(text)
    text = text.lower()
    return _TOKEN_RE.findall(text)


def join_dialogue(utterances: List[str]) -> str:
    """Join a DailyDialog `dialog` list into one document.

    Per spec: concatenate utterances in order, separated by a single space.
    NO speaker tags, no turn markers.
    """
    return " ".join(u.strip() for u in utterances)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"
PAD = "<pad>"

SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]  # fixed order -> fixed ids 0,1,2,3


class Vocab:
    """Fixed vocabulary built from a token frequency count.

    Attributes:
        token2id: dict token -> int id
        id2token: list, index = id
        freq: Counter of raw training-set token frequencies (pre-cap, pre-UNK
              collapse) -- kept for diagnostics / report stats, not used at
              runtime.
    """

    def __init__(self, token2id: Dict[str, int], id2token: List[str], freq: Counter):
        self.token2id = token2id
        self.id2token = id2token
        self.freq = freq

    def __len__(self):
        return len(self.id2token)

    def encode(self, tokens: Iterable[str]) -> List[int]:
        unk_id = self.token2id[UNK]
        return [self.token2id.get(t, unk_id) for t in tokens]

    def encode_str(self, tokens: Iterable[str]) -> List[str]:
        """Map OOV tokens to the literal '<unk>' string (used by n-gram code
        that works with token strings rather than ids)."""
        vocab_set = set(self.token2id.keys())
        return [t if t in vocab_set else UNK for t in tokens]

    def decode(self, ids: Iterable[int]) -> List[str]:
        return [self.id2token[i] for i in ids]

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id2token": self.id2token,
                    "freq": dict(self.freq),
                },
                f,
                ensure_ascii=False,
                indent=1,
            )

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        id2token = obj["id2token"]
        token2id = {t: i for i, t in enumerate(id2token)}
        freq = Counter(obj["freq"])
        return cls(token2id, id2token, freq)


def build_vocab(
    train_token_lists: Iterable[List[str]],
    max_size: int = 30000,
    min_count: int = 2,
) -> Vocab:
    """Build a Vocab from an iterable of already-tokenized training documents.

    Args:
        train_token_lists: iterable where each element is the token list for
            one training document (one DailyDialog dialogue, already joined +
            tokenized). Pass the full train split here, nothing held out.
        max_size: cap on vocabulary size INCLUDING special tokens. Spec says
            "30,000 most frequent tokens" -- we interpret this as 30,000
            regular (non-special) tokens, with the 4 special tokens added on
            top, since the special tokens are not part of the natural-language
            frequency count the spec is capping. State this in the report if
            asked; trivial to change `max_size` semantics here if TA disagrees.
        min_count: tokens occurring fewer than this many times in train are
            collapsed into <UNK> even if they'd otherwise fit under max_size.

    Returns:
        Vocab object. Deterministic tie-breaking: when multiple tokens have
        identical frequency at the cutoff boundary, ties are broken
        alphabetically (ascending) for full reproducibility across runs/seeds.
    """
    freq = Counter()
    for tokens in train_token_lists:
        freq.update(tokens)

    # Keep tokens meeting min_count, then cap to max_size most frequent,
    # alphabetical tie-break for determinism.
    eligible = [(tok, c) for tok, c in freq.items() if c >= min_count]
    eligible.sort(key=lambda x: (-x[1], x[0]))  # freq desc, then alpha asc
    kept = [tok for tok, _ in eligible[:max_size]]

    id2token = list(SPECIAL_TOKENS) + kept
    token2id = {t: i for i, t in enumerate(id2token)}

    return Vocab(token2id, id2token, freq)


# ---------------------------------------------------------------------------
# Scoring-stream builder (shared by Task 1 n-gram eval AND Task 2 LSTM eval,
# so Task 3's per-position interpolation is well-defined -- see PLAN.md 3.7)
# ---------------------------------------------------------------------------

def build_scoring_stream(
    dialogues: List[List[str]],
    vocab: Vocab,
) -> List[str]:
    """Turn a list of dialogues (each a list of raw utterance strings) into
    ONE flat list of vocab-mapped token strings, with <s> ... </s> wrapped
    around each dialogue (per-dialogue boundaries, NOT per-utterance -- see
    PLAN.md assumption 3.2).

    This exact function must be used by both the n-gram scorer and the LSTM
    scorer at evaluation time so both models score an identical sequence of
    positions. Do not reimplement this logic separately in task2/task3 code.
    """
    stream: List[str] = []
    for utterances in dialogues:
        doc = join_dialogue(utterances)
        toks = tokenize(doc)
        toks = vocab.encode_str(toks)
        stream.append(BOS)
        stream.extend(toks)
        stream.append(EOS)
    return stream


if __name__ == "__main__":
    # Minimal smoke test using fabricated data (NOT real DailyDialog -- just
    # verifies the module's internal logic before we touch Colab/real data).
    fake_dialogues = [
        ["Hello there!", "Hi, how are you?", "I'm fine, thanks."],
        ["Don't forget the meeting.", "I won't. See you at 3.5pm!"],
        ["Hello there!", "Hi, how are you?"],  # repeat to test frequency counts
    ]

    tokenized_docs = [tokenize(join_dialogue(d)) for d in fake_dialogues]
    for d, toks in zip(fake_dialogues, tokenized_docs):
        print(join_dialogue(d), "->", toks)

    vocab = build_vocab(tokenized_docs, max_size=30000, min_count=2)
    print("\nVocab size (incl. specials):", len(vocab))
    print("Tokens kept:", [t for t in vocab.id2token if t not in SPECIAL_TOKENS])
    print("'hello' in vocab:", "hello" in vocab.token2id)
    print("'meeting' in vocab (freq 1, should be UNK'd):", "meeting" in vocab.token2id)

    stream = build_scoring_stream(fake_dialogues, vocab)
    print("\nScoring stream:")
    print(stream)

    # Save/load round trip
    vocab.save("/tmp/test_vocab.json")
    reloaded = Vocab.load("/tmp/test_vocab.json")
    assert reloaded.id2token == vocab.id2token
    print("\nSave/load round-trip OK.")
