# Triage data

Curated data for the question-routing classifier (`backend/scripts/train_triage.py`, `eval_triage.py`). Design: `planning/specs/agentic-triage/`.

| File | Rows | Use |
|---|---|---|
| `categories.json` | 15 categories | Category id, the label shown on hover ("Looks like: …"), class (`direct`, `agentic`, or `mixed` for follow-ups) and whether it is a hard case |
| `train.jsonl` | 2,496 (1,265 direct, 1,231 agentic) | Fine-tuning, training and threshold picking (80/20 split inside the training script) |
| `test.jsonl` | 155 | Release gate only. Never used for training or threshold picking |
| `probe.jsonl` | 40 | A second check in a different style: hard questions written after training, none closer than 0.87 to a training row. Not part of the gate |
| `collected/` | — | Shadow decisions and override clicks written by the server when collection is on. Git-ignored, deleted after 30 days. **Contains real user questions** |

Each row: `{"text", "label": "agentic" | "direct", "category", "source": "seed" | "generated" | "collected"}`. Follow-up rows hold the joined form `previous question → follow-up`, the same text the server builds for short follow-ups.

## How the data was made

All rows are `source: "generated"`: written by AI, not by hand. The spec asks for hand-written seed and test sets; **a person still has to review both files** (tasks 1.2–1.4) before the gate result can be fully trusted.

**Version a (600 train rows).** 322 rows across all categories, then 280 more aimed at the hard cases. The test set was written by a separate agent that never saw `train.jsonl`; near-copies of training rows were replaced.

**Version c (2,496 train rows, about 4x).** 1,896 rows added on 2026-10-11:

- About 100–300 more per category, weighted toward the misses of version a: casually phrased research (297 in total), reports (198), follow-ups (357, half of them routine follow-ups to a research question such as "→ what does IPO stand for?"), and opinion-style chat (189).
- A new category, `general_knowledge` (116 rows: settled history, literature, geography and science facts), plus 30 topic-matched questions that need current facts ("Who is the current president of France?"). Without it, the fine-tuned encoder learned "sounds like an event" = research and sent "When did World War II end?" to research.
- Removed: rows the rules already answer, exact duplicates, follow-ups of 6 words or more (the server joins only shorter ones), 15 rows with embedding similarity ≥ 0.9 to a test row, and 6 paraphrases of test questions the writer had seen in an earlier error list.
- `--check-data` passes. The test set is unchanged, so versions a and c are compared on the same gate. It has no `general_knowledge` rows; add some, written separately, when a person reviews the test set.

## Current result: `triage-2026-10-c`

Version c fine-tunes MiniLM on contrastive pairs (SetFit-style; see `scripts/train_triage.py --finetune`), then fits the logistic regression on its embeddings. Version b is the same 4x data with the frozen encoder, trained only for comparison.

| Version | Gate (155 rows): research recall / routine direct | `probe.jsonl` (40 rows): research / routine | Scores between 0.2 and 0.8 |
|---|---|---|---|
| a: 600 rows, frozen | 0.931 / 0.952 (fails) | 16/20 / 19/20 | 68% |
| b: 2,496 rows, frozen | 0.986 / 0.952 (passes) | 18/20 / 19/20 | 38% |
| **c: 2,496 rows, fine-tuned (shipped)** | **1.000 / 1.000 (passes)** | **20/20 / 19/20** | **0%** |

Read the numbers with care:

- The test set and the training data come from the same kind of generator, so the gate is easier than real traffic. `probe.jsonl` is a second check written in a different style; its only miss on version c is "is 97 prime" (sent to research).
- Version c's scores are almost always near 0 or 1. When it is wrong, it is confidently wrong, the `webagent_triage_uncertain_total` metric will stay near zero, and the optional LLM tie-breaker would rarely run. The override button and the override rate are the safety net.
- Fine-tuning ran on the 80% fit split only, so the threshold (0.5) is picked on rows the encoder never saw.

uat and prod still run with `triage.enabled: false` until the shadow week and a human review of the data; the gate itself now passes. Dev uses the classifier in shadow mode.

## Commands

Run from `backend/`:

```bash
python scripts/eval_triage.py --check-data
python scripts/train_triage.py --version triage-2026-10-d --finetune   # ~1 minute on an Apple GPU
python scripts/eval_triage.py --write-metadata   # exit code 1 if the gate fails
python scripts/eval_triage.py --data data/triage/probe.jsonl   # second check, not the gate
```

Without `--finetune` the encoder stays frozen and training takes seconds. The fine-tuned encoder is written to `models/triage/encoder/`; its weights are stored with Git LFS (`.gitattributes`), so run `git lfs pull` after cloning.
