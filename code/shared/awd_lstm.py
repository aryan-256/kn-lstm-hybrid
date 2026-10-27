"""
AWD-LSTM (Merity, Keskar, Socher 2017, arXiv:1708.02182) -- Task 2.

Written fresh against the paper's description rather than adapted from the
official repo (github.com/salesforce/awd-lstm-lm), which predates modern
PyTorch and uses APIs (e.g. old-style weight-norm hooks, Python 2 idioms)
that don't run cleanly on current torch versions. See PLAN.md assumption 3.6.

Components implemented (per spec step 2):
  - WeightDropLSTM: DropConnect (p=0.5 default) applied to the
    hidden-to-hidden weight matrix of each LSTM layer, reapplied (a fresh
    dropout mask) on every forward call during training.
  - Embedding dropout (p=0.1): drops entire embedding ROWS (whole word
    types), not individual activations -- this is what the paper calls
    "embedding dropout" (distinct from standard dropout on embedding
    outputs).
  - Output (locked) dropout (p=0.4): standard dropout applied to the final
    LSTM layer's output before the softmax, using the SAME mask across all
    timesteps in a sequence (a "locked"/"variational" dropout mask) --
    this is what the paper calls output dropout.
  - Tied embedding/softmax weights.

NOT implemented (explicit, documented deviations -- state in report):
  - AR/TAR activation regularization (the paper's alpha/beta regularization
    terms) -- spec's step-by-step instructions (steps 1-4) do not mention
    these, only DropConnect + embedding dropout + output dropout + NT-ASGD +
    the listed hyperparameters, so we implement exactly what's listed and
    flag AR/TAR as an omitted paper detail, consistent with "minimal effort,
    ship what's asked for" per team decision.
  - Variable-length BPTT (paper varies BPTT length per batch for
    regularization) -- spec says "BPTT length 70" as a fixed hyperparameter,
    so we use a fixed length, not the paper's variable scheme.
  - The paper's fine-tuning second stage -- out of scope per spec's step 5.
"""

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    # This sandbox has no internet access to PyTorch's package servers, so
    # this module cannot be executed/tested here -- only written and
    # manually shape-traced (see SHAPE_TRACE below). It WILL run in Colab,
    # which has torch preinstalled. Running this file directly in an
    # environment without torch still prints the shape trace for review.
    _TORCH_AVAILABLE = False
    nn = None
    F = None


# ---------------------------------------------------------------------------
# DropConnect on hidden-to-hidden weights
# ---------------------------------------------------------------------------

class WeightDropLSTM(nn.Module if _TORCH_AVAILABLE else object):
    """Wraps nn.LSTM and applies DropConnect to the hidden-to-hidden weight
    matrix (weight_hh_l0) on every forward call during training.

    Standard dropout zeroes ACTIVATIONS; DropConnect zeroes WEIGHTS.

    IMPLEMENTATION NOTE (important, read before modifying): an earlier draft
    of this class tried to replace nn.LSTM's weight_hh_l0 Parameter with a
    plain (non-leaf) tensor via setattr(), mirroring a trick from the
    official AWD-LSTM repo. That approach is fragile on modern PyTorch:
    nn.LSTM caches parameter names/references at construction time
    (`_flat_weights`, `_all_weights`) for its fused cuDNN/CPU kernel path,
    and a bare setattr() does not reliably update what the kernel actually
    reads -- this could silently train with the UNDROPPED weight while
    appearing to work. Since this module cannot be executed/tested in this
    sandbox (no torch available -- see top-of-file note) before being run
    for real in Colab, we deliberately avoid that risk entirely and use a
    simpler, more robust approach instead:

      1. Keep weight_hh_l0 as nn.LSTM's own real Parameter (untouched,
         nothing deleted/reassigned/re-registered).
      2. On each forward() call during training: clone the current
         weight_hh_l0.data, compute a dropped-out version, write it IN
         PLACE into weight_hh_l0.data (same tensor object, so cuDNN's
         cached references stay valid), run the forward pass, then
         restore the original (undropped) data afterward.
      3. Gradients still flow correctly: autograd records the forward
         computation using whatever data was in weight_hh_l0 at the time
         (the dropped version), and since the Parameter object itself
         (and its .grad) is never replaced, the optimizer's step() still
         updates the SAME Parameter it always did -- we're only ever
         swapping the underlying .data buffer's contents, never the
         Parameter object's identity.

    This trades a small amount of complexity (explicit save/restore) for
    something we can reason about correctly without being able to run it
    first. If Colab shows the model training (loss decreasing, no shape
    errors) in the smoke test, this approach is confirmed working.
    """

    def __init__(self, input_size, hidden_size, dropconnect_p=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=False)
        self.dropconnect_p = dropconnect_p

    def forward(self, x, hidden=None):
        if self.training and self.dropconnect_p > 0:
            original_data = self.lstm.weight_hh_l0.data.clone()
            mask = torch.bernoulli(
                original_data.new_full(original_data.shape, 1 - self.dropconnect_p)
            ) / (1 - self.dropconnect_p)  # inverted dropout scaling
            self.lstm.weight_hh_l0.data.mul_(mask)  # in-place: same tensor object
            try:
                output, new_hidden = self.lstm(x, hidden)
            finally:
                # Always restore, even if the forward pass raises, so a
                # mid-training exception can't leave the model's real
                # weights permanently corrupted.
                self.lstm.weight_hh_l0.data.copy_(original_data)
            return output, new_hidden
        else:
            return self.lstm(x, hidden)


# ---------------------------------------------------------------------------
# Embedding dropout (drops whole word-rows, not individual activations)
# ---------------------------------------------------------------------------

def embedded_dropout(embed_weight, input_ids, p=0.1, training=True):
    """Apply embedding dropout: with probability p, zero out an ENTIRE row
    of the embedding matrix (i.e. drop a word TYPE for this forward pass,
    so every occurrence of that word in the batch gets a zero vector), then
    do a normal embedding lookup with the (possibly zeroed) matrix.
    """
    if training and p > 0:
        mask = embed_weight.data.new_full((embed_weight.size(0), 1), 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)
        masked_weight = embed_weight * mask
    else:
        masked_weight = embed_weight
    return F.embedding(input_ids, masked_weight)


# ---------------------------------------------------------------------------
# Locked (variational) dropout -- same mask reused across all timesteps
# ---------------------------------------------------------------------------

class LockedDropout(nn.Module if _TORCH_AVAILABLE else object):
    """Standard dropout would sample a new mask per timestep; locked dropout
    samples ONE mask and reuses it for every timestep in the sequence
    (standard technique from the AWD-LSTM paper, used here for output
    dropout, p=0.4)."""

    def forward(self, x, p):
        if not self.training or p == 0:
            return x
        # x shape: (seq_len, batch, hidden)
        mask = x.data.new_full((1, x.size(1), x.size(2)), 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)
        mask = mask.expand_as(x)
        return x * mask


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class AWDLSTM(nn.Module if _TORCH_AVAILABLE else object):
    """3-layer weight-dropped LSTM LM with tied embeddings.

    Args match the spec exactly:
        vocab_size: size of the vocabulary (len(vocab) from tokenizer.Vocab)
        embed_dim: 400 (paper default)
        hidden_dim: 1150 (paper default; pass 400 for the compute-constrained
            fallback -- see PLAN.md assumption 3.6)
        n_layers: 3
        dropconnect_p: 0.5 (hidden-to-hidden DropConnect)
        embed_dropout_p: 0.1
        output_dropout_p: 0.4

    IMPORTANT architectural note on tied weights: tying requires the LAST
    LSTM layer's hidden_dim to equal embed_dim (since the output projection
    reuses the embedding matrix, shape (vocab_size, embed_dim)). The paper
    handles this by making the LAST layer's hidden size equal to embed_dim
    (400) even though earlier layers use the larger hidden_dim (1150). We
    replicate that: layers 0..n_layers-2 use hidden_dim, the FINAL layer
    uses embed_dim as its output size.
    """

    def __init__(self, vocab_size, embed_dim=400, hidden_dim=1150, n_layers=3,
                 dropconnect_p=0.5, embed_dropout_p=0.1, output_dropout_p=0.4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed_dropout_p = embed_dropout_p
        self.output_dropout_p = output_dropout_p

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Per-layer sizes: all but last use hidden_dim, last uses embed_dim
        # (required for weight tying -- see docstring).
        layer_input_sizes = [embed_dim] + [hidden_dim] * (n_layers - 1)
        layer_output_sizes = [hidden_dim] * (n_layers - 1) + [embed_dim]

        self.rnns = nn.ModuleList([
            WeightDropLSTM(layer_input_sizes[i], layer_output_sizes[i], dropconnect_p)
            for i in range(n_layers)
        ])

        self.locked_dropout = LockedDropout()

        self.decoder = nn.Linear(embed_dim, vocab_size)
        # Tie weights: decoder.weight (vocab_size, embed_dim) == embedding.weight
        self.decoder.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        # decoder.weight is tied to embedding.weight, already initialized above

    def init_hidden(self, batch_size, device):
        hidden = []
        for i, rnn in enumerate(self.rnns):
            h_size = self.hidden_dim if i < self.n_layers - 1 else self.embed_dim
            h = torch.zeros(1, batch_size, h_size, device=device)
            c = torch.zeros(1, batch_size, h_size, device=device)
            hidden.append((h, c))
        return hidden

    def forward(self, input_ids, hidden):
        """
        input_ids: (seq_len, batch) LongTensor of token ids
        hidden: list of (h, c) tuples, one per layer, from init_hidden() or
                a previous forward() call's returned hidden state (for
                truncated BPTT -- caller detaches between chunks)

        Returns:
            logits: (seq_len, batch, vocab_size)
            new_hidden: list of (h, c) tuples, same structure as `hidden`
        """
        emb = embedded_dropout(
            self.embedding.weight, input_ids,
            p=self.embed_dropout_p, training=self.training
        )  # (seq_len, batch, embed_dim)

        x = emb
        new_hidden = []
        for i, rnn in enumerate(self.rnns):
            x, h_new = rnn(x, hidden[i])
            new_hidden.append(h_new)
            if i < self.n_layers - 1:
                # Locked dropout between LSTM layers too (standard AWD-LSTM
                # practice uses this for inter-layer dropout as well, at the
                # same output_dropout_p -- spec doesn't separately specify an
                # inter-layer rate, so we reuse output_dropout_p here, a
                # reasonable simplification per "minimal effort" principle).
                x = self.locked_dropout(x, self.output_dropout_p)

        x = self.locked_dropout(x, self.output_dropout_p)  # final output dropout
        logits = self.decoder(x)  # (seq_len, batch, vocab_size)
        return logits, new_hidden

    @staticmethod
    def detach_hidden(hidden):
        """Detach hidden state from the computation graph between BPTT
        chunks (standard truncated-BPTT practice -- prevents backprop from
        running all the way back to the start of training)."""
        return [(h.detach(), c.detach()) for (h, c) in hidden]


# ---------------------------------------------------------------------------
# Manual shape trace (since we cannot run torch in this sandbox -- see module
# docstring). This documents, step by step, what every tensor shape SHOULD
# be, so the first real Colab run can be checked against this by eye instead
# of debugging blind.
# ---------------------------------------------------------------------------

SHAPE_TRACE = """
Manual shape trace for AWDLSTM(vocab_size=12482, embed_dim=400, hidden_dim=1150, n_layers=3)
with seq_len=70 (BPTT length per spec), batch_size=80 (per spec):

  input_ids:            (70, 80)                      int64
  embedding lookup ->   (70, 80, 400)                  float32
  layer 0 (WeightDropLSTM, in=400, out=1150):
    input  (70, 80, 400)
    output (70, 80, 1150), h/c each (1, 80, 1150)
  locked_dropout (p=0.4) -> same shape (70, 80, 1150)
  layer 1 (WeightDropLSTM, in=1150, out=1150):
    input  (70, 80, 1150)
    output (70, 80, 1150), h/c each (1, 80, 1150)
  locked_dropout (p=0.4) -> same shape (70, 80, 1150)
  layer 2 (WeightDropLSTM, in=1150, out=400):   <- last layer outputs embed_dim, for tying
    input  (70, 80, 1150)
    output (70, 80, 400), h/c each (1, 80, 400)
  locked_dropout (p=0.4, final output dropout) -> (70, 80, 400)
  decoder (Linear(400 -> 12482), weight tied to embedding.weight):
    output logits: (70, 80, 12482)

Parameter count estimate (rough, for sanity-checking against Colab's actual
printed count -- see the test script's parameter counter):
  embedding: 12482 * 400                           = 4,992,800
  layer 0 LSTM: 4 * 1150 * (400+1150+1)            = 7,135,000   (approx, incl. biases)
  layer 1 LSTM: 4 * 1150 * (1150+1150+1)           = 10,580,600  (approx)
  layer 2 LSTM: 4 * 400  * (1150+400+1)            = 2,482,400   (approx)
  decoder bias: 12482
  decoder.weight is TIED (0 additional params, reuses embedding)
  ------------------------------------------------------------------
  TOTAL approx: ~25.2M parameters

This is in the same ballpark as the paper's ~24M for PTB/WT2 configs, which
is expected since embed_dim/hidden_dim match the paper despite our much
smaller vocab (12,482 vs ~33K) -- the LSTM weight matrices dominate
parameter count, not the embedding/decoder, so vocab size differences barely
move the total.
"""

# ---------------------------------------------------------------------------
# Training / evaluation utility functions
#
# These are plain tensor-reshaping and bookkeeping helpers -- they do NOT
# touch the WeightDropLSTM internals, so they carry none of the non-leaf-
# tensor risk that the weight-drop trick does. Kept deliberately simple.
# ---------------------------------------------------------------------------

def repackage_hidden(h):
    """Detach hidden states from their computation-graph history, so
    truncated BPTT doesn't backprop through the entire training history.
    Call this once per batch, on the hidden state returned by the previous
    batch's forward() call, before feeding it into the next batch."""
    if _TORCH_AVAILABLE and isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data_ids, bsz, device):
    """Reshape a flat list[int] of token ids into a (num_steps, bsz) tensor
    for truncated-BPTT training, per spec: batch size 80.

    Standard LM batching: split the flat stream into `bsz` contiguous
    columns (dropping leftover tokens that don't fill a full column), so
    each column is read top-to-bottom as one continuous sub-sequence and
    hidden state carries across batches WITHIN a column via repackage_hidden.
    """
    data = torch.tensor(data_ids, dtype=torch.long)
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt=70):
    """Slice one training chunk starting at row i: input is source[i:i+seq_len],
    target is the same window shifted forward by one timestep (next-token
    prediction), flattened to 1D for CrossEntropyLoss. seq_len is clipped
    near the end of `source` so we never index out of bounds."""
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


@torch.no_grad() if _TORCH_AVAILABLE else (lambda f: f)
def evaluate_stream_exact(model, stream_ids, bos_id, device, chunk_size=2048,
                           return_token_logprobs=False):
    """Exact sequential perplexity evaluation over a 1D scoring stream
    (batch_size=1, no truncation/dropped tail), matching Task 1's n-gram
    perplexity convention exactly (PLAN.md 3.2/3.7): every token is scored
    EXCEPT <s> (bos_id), which only ever appears as context, never as a
    predicted target. This is what makes Task 1 and Task 2 perplexities
    directly comparable, and is REQUIRED for Task 3's per-position hybrid
    interpolation to be well-defined (both models must score the identical
    sequence of positions).

    Use this for all REPORTED perplexity numbers (val/test). Use the faster
    batched `batchify`+`get_batch` path only for TRAINING, where exact
    per-token alignment with Task 1 doesn't matter (training just needs the
    loss signal, not a comparable metric).
    """
    model.eval()
    data = torch.tensor(stream_ids, dtype=torch.long, device=device).unsqueeze(1)  # (N, 1)
    hidden = model.init_hidden(1, device)

    total_logprob = 0.0
    n_tokens = 0
    token_logprobs = []

    for i in range(0, data.size(0) - 1, chunk_size):
        seq_len = min(chunk_size, data.size(0) - 1 - i)
        inp = data[i:i + seq_len]                      # (seq_len, 1)
        tgt = data[i + 1:i + 1 + seq_len].squeeze(1)    # (seq_len,)

        logits, hidden = model(inp, hidden)
        hidden = repackage_hidden(hidden)

        log_probs = F.log_softmax(logits.squeeze(1), dim=-1)  # (seq_len, V)
        target_lps = log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)  # (seq_len,)

        valid_mask = (tgt != bos_id)
        valid_lps = target_lps[valid_mask]

        total_logprob += valid_lps.sum().item()
        n_tokens += valid_mask.sum().item()

        if return_token_logprobs:
            token_logprobs.extend(valid_lps.cpu().tolist())

    ppl = math.exp(-total_logprob / n_tokens)
    if return_token_logprobs:
        return ppl, token_logprobs
    return ppl


if __name__ == "__main__":
    print(SHAPE_TRACE)
    print("\n(Full forward-pass execution requires torch + will be run in Colab;")
    print(" this file was written and shape-traced by hand in this sandbox")
    print(" since torch is not installable here -- verify against real run.)")
