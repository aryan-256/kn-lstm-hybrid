"""
NT-ASGD: Non-monotonically Triggered Averaged SGD (Merity, Keskar, Socher 2017,
arXiv:1708.02182, Algorithm 1) -- Task 2 optimizer, per spec step 3.

Written as a thin wrapper AROUND plain torch.optim.SGD, rather than a custom
Optimizer subclass -- deliberately simple, since the failure modes we most want
to avoid here are (a) silently averaging the wrong tensors, (b) forgetting to
swap averaged weights back in before evaluation, (c) off-by-one errors in the
"non-monotonic interval" trigger check. Keeping this as an explicit wrapper
class (not hiding behavior inside a subclassed .step()) makes every one of
those three steps an obvious, separately-callable, separately-inspectable
method -- easier to reason about without being able to execute it here.

Usage pattern (exact contract the training loop must follow):

    nt_asgd = NTASGD(model.parameters(), lr=30.0, n=5, weight_decay=1.2e-6)

    for epoch in range(1, n_epochs + 1):
        train_one_epoch(model, ..., optimizer=nt_asgd.optimizer)  # train as normal

        val_ppl = evaluate(...)                      # eval on RAW (non-averaged) weights is fine pre-trigger
        triggered_this_epoch = nt_asgd.step_epoch(val_ppl)   # call ONCE per epoch, after computing val_ppl

        if nt_asgd.is_averaging:
            nt_asgd.swap_to_averaged(model)           # put averaged weights into model for eval/checkpointing
            val_ppl_averaged = evaluate(...)          # re-evaluate using averaged weights (the ones you report)
            nt_asgd.swap_to_raw(model)                # put raw weights back for the NEXT epoch's training step

IMPORTANT: once is_averaging is True, you must call swap_to_averaged() before
any evaluation/checkpoint-saving you intend to report, and swap_to_raw()
before resuming training -- training must always continue on the raw
(non-averaged) parameters; only EVALUATION should see the averaged ones.
Mixing this up (e.g. training on averaged weights) silently breaks the
method's theoretical justification even though nothing would crash.
"""

import copy

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class NTASGD:
    """Wraps torch.optim.SGD; after each epoch, decides whether to start
    averaging parameters across subsequent steps, per the paper's
    non-monotonic trigger condition.

    Args:
        params: model.parameters() (same as any torch optimizer)
        lr: initial learning rate (spec: 30)
        n: non-monotonic interval -- trigger averaging once current val PPL
           is worse than the min of the last `n` val PPLs (paper default: 5,
           not specified explicitly in our spec, so we use the paper's own
           default value here and document this as filling a spec gap)
        weight_decay: passed through to the underlying SGD
    """

    def __init__(self, params, lr=30.0, n=5, weight_decay=0.0):
        self.params = list(params)  # materialize once; both SGD and we need
                                     # a stable, ORDERED list of the same
                                     # Parameter objects
        self.optimizer = torch.optim.SGD(self.params, lr=lr, weight_decay=weight_decay)
        self.n = n

        self.val_ppl_history = []
        self.is_averaging = False
        self.trigger_epoch = None

        # Running average state: for each param, accumulate a running SUM
        # of its values across every optimizer.step() call made AFTER
        # averaging was triggered, plus a count of how many steps have been
        # accumulated -- average = running_sum / step_count. We store this
        # as plain tensors (not Parameters), one per entry in self.params,
        # matched by LIST POSITION (self.params never changes order/length
        # after __init__, so position-matching is safe and avoids relying
        # on object identity across copies).
        self._running_sum = None   # list[Tensor], same shapes as self.params
        self._avg_step_count = 0

        # Saved raw (non-averaged) parameter values, used to restore model
        # state after a temporary swap_to_averaged() for evaluation.
        self._raw_backup = None

    def step_batch(self):
        """Call once per training batch, AFTER loss.backward() and gradient
        clipping, in place of optimizer.step(). Performs the normal SGD
        step, then (if averaging has been triggered) accumulates the
        post-step parameter values into the running average.
        """
        self.optimizer.step()

        if self.is_averaging:
            if self._running_sum is None:
                self._running_sum = [p.data.clone() for p in self.params]
                self._avg_step_count = 1
            else:
                for i, p in enumerate(self.params):
                    self._running_sum[i].add_(p.data)
                self._avg_step_count += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step_epoch(self, val_ppl):
        """Call once per epoch, after computing validation perplexity for
        that epoch (on the RAW, non-averaged weights -- averaging has no
        effect on training-time eval until swap_to_averaged() is called
        explicitly). Checks the non-monotonic trigger condition and flips
        self.is_averaging to True the first time it's satisfied.

        Returns True if averaging was JUST triggered this call (useful for
        logging "NT-ASGD triggered at epoch N" once), False otherwise.
        """
        just_triggered = False

        if not self.is_averaging:
            if len(self.val_ppl_history) >= self.n:
                recent_min = min(self.val_ppl_history[-self.n:])
                if val_ppl > recent_min:
                    self.is_averaging = True
                    just_triggered = True
                    # Averaging starts accumulating from the NEXT batch's
                    # step (step_batch handles first-call initialization of
                    # _running_sum lazily) -- we deliberately do NOT seed
                    # _running_sum with the current weights here, since the
                    # first accumulated step will do that on its own.

        self.val_ppl_history.append(val_ppl)
        return just_triggered

    def get_averaged_params(self):
        """Returns the current running-average parameter values as a list
        of tensors (same order as self.params). Raises if averaging hasn't
        been triggered/accumulated yet -- callers should check
        self.is_averaging and that at least one step_batch() has run since
        triggering before calling this."""
        if self._running_sum is None or self._avg_step_count == 0:
            raise RuntimeError(
                "get_averaged_params() called before any post-trigger "
                "optimizer steps have been accumulated. Check "
                "self.is_averaging and ensure at least one training batch "
                "has run since triggering."
            )
        return [s / self._avg_step_count for s in self._running_sum]

    def swap_to_averaged(self, model):
        """Temporarily replace model's parameter VALUES (in place, via
        .data) with the running-averaged values, for evaluation/
        checkpointing. Saves the raw values first so swap_to_raw() can
        restore them exactly. Safe to call only when self.is_averaging is
        True and at least one step has been accumulated (see
        get_averaged_params)."""
        averaged = self.get_averaged_params()
        self._raw_backup = [p.data.clone() for p in self.params]
        for p, avg_val in zip(self.params, averaged):
            p.data.copy_(avg_val)

    def swap_to_raw(self, model):
        """Restore the raw (non-averaged) parameter values saved by the most
        recent swap_to_averaged() call. MUST be called before resuming
        training after an averaged-weights evaluation -- training must
        always continue on raw weights, never on the averaged copy."""
        if self._raw_backup is None:
            raise RuntimeError(
                "swap_to_raw() called without a matching prior "
                "swap_to_averaged() call -- nothing to restore."
            )
        for p, raw_val in zip(self.params, self._raw_backup):
            p.data.copy_(raw_val)
        self._raw_backup = None

    def state_dict(self):
        """For checkpointing: includes the underlying SGD's state plus all
        NT-ASGD-specific bookkeeping needed to resume correctly mid-average."""
        return {
            'optimizer_state': self.optimizer.state_dict(),
            'val_ppl_history': self.val_ppl_history,
            'is_averaging': self.is_averaging,
            'trigger_epoch': self.trigger_epoch,
            'running_sum': self._running_sum,
            'avg_step_count': self._avg_step_count,
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.val_ppl_history = state['val_ppl_history']
        self.is_averaging = state['is_averaging']
        self.trigger_epoch = state['trigger_epoch']
        self._running_sum = state['running_sum']
        self._avg_step_count = state['avg_step_count']


if __name__ == "__main__":
    print(__doc__)
    print("\n(Requires torch to execute -- not runnable in this sandbox;")
    print(" logic reasoned through manually, verify in Colab.)")
