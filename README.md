# kn-lstm-hybrid

N-gram smoothing, AWD-LSTM reproduction, and a Kneser-Ney + LSTM hybrid language model, trained and evaluated on the [DailyDialog](https://huggingface.co/datasets/daily_dialog) corpus.

This repository contains the code and report for Project 1 ("N-gram Language Modeling, Smoothing, and Neural LM Reproduction") of CS60075 Assignment II, assigned dataset: DailyDialog.

## What's here

Three tasks on the same dataset:

1. **Task 1 — Smoothed n-gram language model.** Add-k, Good-Turing, and interpolated Kneser-Ney smoothing over trigram and 4-gram counts.
2. **Task 2 — AWD-LSTM reproduction.** A from-scratch implementation of Merity et al. (2017), *Regularizing and Optimizing LSTM Language Models* (arXiv:1708.02182), trained with NT-ASGD.
3. **Task 3 — Hybrid model.** Linear interpolation of the Task 1 Kneser-Ney model and the Task 2 AWD-LSTM, with the blend weight tuned on validation perplexity.

All three are evaluated on held-out perplexity and on [BLiMP](https://huggingface.co/datasets/nyu-mll/blimp) (nine paradigms, spanning morphology, syntax, and semantics).

**Headline result:** the hybrid achieves a test perplexity of 23.09, against 33.06 for Kneser-Ney alone and 32.85 for the AWD-LSTM alone — roughly a 30% relative improvement over either individual model, despite the two models performing almost identically on their own.

The full writeup, including the per-paradigm BLiMP analysis and worked qualitative examples, is in [`report/NLP_A2_Project1_Report.docx`](report/NLP_A2_Project1_Report.docx).

## Repository structure

```
code/
  shared/
    tokenizer.py       # tokenization, vocabulary, shared scoring-stream construction
    ngram_lm.py         # n-gram counting + Add-k / Good-Turing / Kneser-Ney smoothing
    awd_lstm.py          # AWD-LSTM architecture, training utilities, exact-stream evaluator
  task1/
    task1_ngram.ipynb        # Task 1, self-contained, no external state required
  task2/
    nt_asgd.py                # NT-ASGD optimizer wrapper
    task2_awd_lstm.ipynb      # Task 2, trains the AWD-LSTM from scratch
  task3/
    hybrid.py                  # per-token log-prob extraction + interpolation
    task3_hybrid.ipynb         # Task 3, downloads the Task 2 checkpoint, no retraining needed
report/
  NLP_A2_Project1_Report.docx
requirements.txt
```

## Running the notebooks

Each notebook under `code/task*/` is self-contained: it writes its required `.py` modules to the Colab runtime's local disk directly from cells inside the notebook, downloads DailyDialog and BLiMP fresh from HuggingFace, and (for Task 3) downloads the trained Task 2 checkpoint from this repository's [Releases](../../releases) page. No Google Drive access, local file uploads, or manual setup is required.

Open any notebook directly in Colab:

- Task 1: `https://colab.research.google.com/github/aryan-256/kn-lstm-hybrid/blob/main/code/task1/task1_ngram.ipynb`
- Task 2: `https://colab.research.google.com/github/aryan-256/kn-lstm-hybrid/blob/main/code/task2/task2_awd_lstm.ipynb`
- Task 3: `https://colab.research.google.com/github/aryan-256/kn-lstm-hybrid/blob/main/code/task3/task3_hybrid.ipynb`

Run each notebook's cells top to bottom in a fresh runtime (Runtime → Run all). Task 1 runs entirely on CPU in under a minute. Task 2 requires a GPU runtime (Runtime → Change runtime type → GPU) and takes roughly 2 hours to train for 50 epochs; Task 3 requires a GPU runtime but does not retrain anything, since it downloads the Task 2 checkpoint directly.

### Running Task 3 without retraining Task 2

Task 3 depends on the trained AWD-LSTM from Task 2. Rather than retraining it, Task 3's notebook downloads the final trained weights directly from this repository's release assets:

```
https://github.com/aryan-256/kn-lstm-hybrid/releases/download/v1.0-checkpoints/awd_lstm_final_averaged.pt
```

This is handled automatically in Task 3's notebook; no manual download is needed.

## Dataset

[DailyDialog](https://huggingface.co/datasets/daily_dialog): 11,118 training dialogues, 1,000 validation, 1,000 test. Downloaded automatically by each notebook via direct HTTPS requests to the dataset's Hugging Face Hub Parquet export (the standard `datasets` library loader for this dataset relies on a Python loading script that is no longer supported by recent versions of the `datasets` library, so the notebooks fetch the Parquet files directly instead).

## Evaluation: BLiMP

Nine BLiMP paradigms were selected, spanning all three of BLiMP's broad categories:

- **Morphology:** `anaphor_gender_agreement`, `determiner_noun_agreement_1`, `regular_plural_subject_verb_agreement_1`
- **Syntax:** `wh_island`, `npi_present_1`, `left_branch_island_simple_question`
- **Semantics:** `principle_A_domain_1`, `drop_argument`, `existential_there_quantifiers_1`

The same nine paradigms, downloaded automatically from the Hugging Face Hub, are used identically across all three tasks.

## Results summary

| Model | Test Perplexity | BLiMP Accuracy |
|---|---|---|
| Task 1: 4-gram Kneser-Ney | 33.06 | 56.1% |
| Task 2: AWD-LSTM | 32.85 | 68.4% |
| Task 3: Hybrid (λ = 0.4) | 23.09 | 64.7% |

See the report for the full per-paradigm breakdown, the training curve, the λ grid search, and a discussion of what the hybrid's perplexity-versus-BLiMP trade-off suggests about the two base models.

## Notable implementation details and deviations

A full account of every deviation from the assignment specification, with justification, is in the report. The most significant:

- **Kneser-Ney's interpolation weight.** The assignment specifies both a fixed discount (d = 0.75) and a separately tuned interpolation weight λ, which is redundant under the standard interpolated-KN formula. We implemented standard recursive interpolated KN with d = 0.75 fixed, and treated λ as an additional outer blend with the unigram distribution, tuned by grid search on validation.
- **AWD-LSTM regularization.** DropConnect, embedding dropout, and output dropout are implemented as specified. Activation and temporal activation regularization (AR/TAR), part of the original paper but not listed in the assignment's step-by-step instructions, were not implemented.
- **NT-ASGD's non-monotonic interval.** Not specified by the assignment; we used the paper's own default of five epochs.
- **DropConnect implementation.** Applied by mutating the relevant weight tensor's underlying data in place before each forward pass and restoring it afterward, rather than by replacing the parameter object itself — this avoids a known PyTorch failure mode in which a dropout-masked weight, if substituted directly into a module's parameter registry, is rejected by the optimizer as a non-leaf tensor.

## Requirements

See `requirements.txt`. All notebooks were developed and run on Google Colab, which preinstalls PyTorch, pandas, and matplotlib. No additional package installation is required; datasets and BLiMP paradigms are fetched as Parquet files and read directly with pandas, so the `datasets` library itself is not needed.
