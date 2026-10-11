# Triage data

Curated data for the question-routing classifier (`backend/scripts/train_triage.py`, `eval_triage.py`). Design: `planning/specs/agentic-triage/`.

| File | Rows | Use |
|---|---|---|
| `categories.json` | 14 categories | Category id, the label shown on hover ("Looks like: …"), class (`direct`, `agentic`, or `mixed` for follow-ups) and whether it is a hard case |
| `train.jsonl` | 600 | Training and threshold picking (80/20 split inside the training script) |
| `test.jsonl` | 155 | Release gate only. Never used for training or threshold picking |
| `collected/` | — | Shadow decisions and override clicks written by the server when collection is on. Git-ignored, deleted after 30 days. **Contains real user questions** |

Each row: `{"text", "label": "agentic" | "direct", "category", "source": "seed" | "generated" | "collected"}`. Follow-up rows hold the joined form `previous question → follow-up`, the same text the server builds for short follow-ups.

## How the current data was made (version `triage-2026-10-a`)

All rows are `source: "generated"`: written by AI agents, not by hand. The spec asks for hand-written seed and test sets; **a person still has to review both files** (task 1.2–1.4) before the gate result can be trusted.

- `train.jsonl`: 322 rows across all categories (at least 22 each), then 280 more aimed at the hard cases (casual research, follow-ups, reports, short time-sensitive questions, long pasted text). Two rows that collided with test rows were removed.
- `test.jsonl`: written by a separate agent that never saw `train.jsonl`. 23 of its first 160 rows turned out to be near-copies of training rows (classic textbook questions) and were removed; 18 fresh rows replaced them.
- `--check-data` passes: schema, known categories, labels matching the category class, no duplicates, no test row with embedding similarity ≥ 0.9 to a training row, and at least 20 train / 10 test rows per hard case.

Prompt used for the training rows (abridged):

> Write questions a chat-assistant user would type, one JSON object per line with `text`, `label`, `category`, `source: "generated"`. `direct` = one quick LLM call is enough (stable knowledge, coding, math, rewriting pasted text, chit-chat). `agentic` = needs web research or current data (news, prices, comparisons with current sources, reports). Cover every category in `categories.json`. Follow-ups use `previous → follow-up` with a follow-up under 6 words, labelled by what the follow-up needs. Long routine rows are 60–200 words of pasted text with a rewrite/summarize request. Vary length, wording and casing; include near-misses. Don't start with "research", "find sources", "write a report", etc. (rules handle those). No personal data.

## Current result

`triage-2026-10-a` **fails the gate**: research recall 0.931 (target 0.95), routine routed direct 0.952 (target 0.80), precision 0.944, at threshold 0.5. Misses are casually phrased research and reports; false alarms are direct follow-ups after a research question ("→ what does IPO stand for?"). Validation recall (0.98) is higher than test recall because the validation split comes from the same generated data, so the threshold lands on the 0.5 cap.

Because it fails, uat and prod run with `triage.enabled: false`, and `startup_checks` refuses a classifier whose `metadata.json` doesn't record `test.passed: true`. Dev uses it in shadow mode to collect real questions.

Next steps, in order: review and rewrite rows by hand; add reviewed shadow-week questions to `train.jsonl` (never `test.jsonl`); retrain and re-run the gate; if the frozen encoder still plateaus, try SetFit (phase 4, REQ-024).

## Commands

Run from `backend/`:

```bash
python scripts/eval_triage.py --check-data
python scripts/train_triage.py --version triage-2026-10-b
python scripts/eval_triage.py --write-metadata   # exit code 1 if the gate fails
```
