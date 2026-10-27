# ContextDefender curated data (SEC-027)

Training, release-gate and probe data for the ContextDefender classifier, built from public datasets by `scripts/build_security_dataset.py` (after `scripts/fetch_security_sources.py` downloads them). Design: `planning/specs/agentic-security/`.

**Content warning:** the attack rows contain real prompt-injection, jailbreak and harmful-request text from published research datasets. They are classifier training data only.

| File | Rows | Use |
|---|---|---|
| `train.jsonl` | 27,545 | Training and calibration |
| `test.jsonl` | 6,784 | Release gate only (`scripts/eval_context_defender.py`) |
| `probe.jsonl` | 697 | Second check: held-out source (treycsa) plus a 2% hash sample |
| `collected/`, `escalations/`, `tokens/` | — | Runtime folders, git-ignored |
| `local/` | — | Reviewed real web pages from a shadow week (`scripts/import_reviewed_pages.py`), git-ignored: third-party text, never committed |

Each row: `{"text", "source", "labels": {injection, harm, sensitive_data, jailbreak: none|low|high}, "origin"}`, plus `"span": [start, end]` on rows where an injection was inserted into a benign page (13,387 rows: 10,433 train, 2,725 test, 229 probe). `origin` is `<dataset>:<role>`. Rows have one positive axis or none (benign).

## Composition (35,026 rows)

Injections hidden in pages and tool data, and harmful requests, are over-represented, with matching benign pages, tool data and safe prompts. Since 2026-10-12, Aegis 2.0 "unsafe" prompts are labelled by category, ordinary questions are increased so threshold selection has enough of them, and fake secrets come in 30 contexts × 21 formats with placeholder twins (see "How it is built").

| Group | Rows | Origin |
|---|---|---|
| Benign | 14,196 | WildChat user turns (2,523) and assistant replies as pages (2,500), S-Labs/deepset benign prompts (1,000), Aegis 2.0 safe prompts (775) and soft-category prompts (403), triage questions (700), NVIDIA agent records with the injection removed (600), TrustAIRLab regular prompts (388), configs and commands with placeholders instead of secrets (318), XSTest safe prompts (250), generated program output (100), JBB benign behaviours (100), AgentDojo clean environment data (39, probe), InjecAgent clean tool responses (17, test), treycsa benign (7, probe); SEP task data without its probe (499), NotInject prompts with trigger words (339, a look-alike set), BIPIA emails without attacks (48).; SQuAD Wikipedia passages without instructions (600).; benign text full of instructions to the reader: Cosmopedia how-to articles (1,200) and web articles (600), OpenAssistant answers (1,200). |
| Injection | 14,991 | Inserted into benign pages: S-Labs/deepset injections (4,000), BIPIA attacks (748: train attacks in train, test attacks in test), LLMail emails (1,500). In agent tool data: NVIDIA records with an injected field (1,200), InjecAgent tool responses (536: split by template and attacker instruction, 340 train and 196 test, the test cases using a template and an instruction training never saw), AgentDojo environment data (80, probe only). Standalone: LLMail emails (994), S-Labs/deepset injections in the user's message and history (600), treycsa (10, probe only) BIPIA-style task instructions in data: SEP probes (1,499; about 30% of probes only in test) and BIPIA attacks in BIPIA's own emails (525; train attacks in train emails, test attacks in test emails). TaskTracker-style task drift: TaskTracker's own instructions in SQuAD passages (1,799; train and test passages and instructions kept apart) and Alpaca instructions in WildChat pages (1,500). |
| Harm | 2,780 | Aegis 2.0 prompts in dangerous categories (2,301), HarmBench standard behaviours (191), XSTest unsafe contrast prompts (200), JBB harmful behaviours (100) |
| Jailbreak | 1,100 | TrustAIRLab in-the-wild jailbreak prompts, deduplicated, as user turns, history and web pages |
| Sensitive data | 1,959 | Generated fake credentials in 30 contexts and 21 formats (900), Nemotron-PII synthetic documents (800), Aegis 2.0 requests for someone's personal data (262) |

By source: user_turn 10,281, web_result 7,012, document 6,017, retrieved_doc 5,295, tool_output 2,748, agent_output 2,275, history 1,317, unlabelled 81.

Per-group targets are the `QUOTAS` in `scripts/build_security_dataset.py` (their sum is the default `--total`); deduplication and the near-duplicate split move the final counts slightly.

## How it is built

- **Filtering:** WildChat rows are English, not flagged toxic, not redacted; benign candidates that match injection or jailbreak wording (the fast patterns, or terms such as "DAN" and "developer mode") are dropped to avoid mislabelled benign rows.
- **Long texts** keep their first and last 1,500 characters.
- **Harm from Aegis 2.0, by category** (decided 2026-10-12, `aegis_group()` in the builder): Aegis calls a prompt "unsafe" for a broad set of categories, but WebAgent declines only dangerous requests. An unsafe prompt (under 2,000 characters, no jailbreak wording) is **harm** if any of its categories is criminal planning, violence, hate, harassment, weapons, self-harm, fraud, controlled substances, sexual (minor), threat, illegal activity or malware; otherwise **sensitive data** if PII/privacy (`aegis:privacy`, a request for someone's personal data); otherwise **benign** if every category is needs caution, unauthorized advice, political/misinformation, immoral/unethical or profanity (`aegis:soft`, counted as a look-alike); anything else (adult sexual content, copyright, high-risk government decisions, manipulation, other) is ambiguous and not used. Safe prompts are added as benign user turns.
- **Indirect injection:** injection text (up to 600 characters) is inserted at a random point inside a benign WildChat assistant reply trimmed to 2,400 characters, which then acts as a web page, document or agent output. The row's `span` marks the inserted text; training labels only the chunks that contain it as injection, and the rest of the page as benign.
- **Splits by near-duplicate cluster:** rows with MiniLM cosine similarity ≥ 0.9 are linked, and each cluster goes to one split (hash of its text, about 80/18/2). Test and probe therefore contain no paraphrase of a training row. BIPIA's train attacks go only to train and its test attacks only to test; a forced row whose cluster lands in another split is dropped rather than moved, so no attack string is on both sides. treycsa is held out entirely for probe, which measures transfer to a source the model never saw. Probe is a ~2% sample, so a source missing from it by chance gets one row moved from test (never from train).
- `python scripts/eval_context_defender.py --check-data` passes: schema, no duplicates, no near-duplicates across splits, minimum rows per source, and positives for every realistic axis/source combination (`REQUIRED_CELLS` in `scripts/security_data.py`).

Rebuild (from `backend/`):

```bash
python scripts/fetch_security_sources.py --out /tmp/secsrc
python scripts/build_security_dataset.py --raw /tmp/secsrc
python scripts/eval_context_defender.py --check-data
```

## Sources, licences and attribution

| Dataset | Licence | Notes |
|---|---|---|
| [microsoft/llmail-inject-challenge](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) | MIT | Phase 1 emails, deduplicated; 6,000 sampled |
| [microsoft/BIPIA](https://github.com/microsoft/BIPIA) | MIT | Text and code attack sets (train attacks only in training, test attacks only in test), and the EmailQA contexts as carriers (from OpenAI Evals, MIT); the code (Stack Exchange) and table (WikiTableQuestions) contexts are CC BY-SA and not used. The README says the code and data are for research on indirect prompt injection; accepted for this use (2026-10-11) |
| [treycsa/realistic-prompt-injections](https://huggingface.co/datasets/treycsa/realistic-prompt-injections) | MIT | 17 rows, probe only |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | |
| [S-Labs/prompt-injection-dataset](https://huggingface.co/datasets/S-Labs/prompt-injection-dataset) | MIT | Upstream sources not named by the publisher |
| [TrustAIRLab/in-the-wild-jailbreak-prompts](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts) | MIT | Carries a "research purposes only" disclaimer; accepted for this use (2026-10-11). Shen et al., "Do Anything Now", CCS 2024 |
| [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | MIT | The card asks users to also cite AdvBench and TDC/HarmBench |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | MIT (repository) | `harmbench_behaviors_text_all.csv`, standard behaviours. Mazeika et al., 2024 |
| [XSTest](https://huggingface.co/datasets/natolambert/xstest-v2-copy) | CC BY 4.0 (prompts) | Röttger et al., "XSTest", NAACL 2024. Prompts only; model completions are not used |
| [nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | CC BY 4.0 | Prompts and their human safety labels only; responses are not used. Ghosh et al., "Aegis2.0", NAACL 2025 |
| [nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1) | CC BY 4.0 | Fully synthetic agent environments (generated with NVIDIA's open Nemotron models) with an injection in one record field; we use the record as tool output with the injection's span, and the same record without it as a benign twin. © NVIDIA, used under CC BY 4.0 |
| [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | MIT | Tool responses with attacker instructions (`test_cases_dh_base.json`, `test_cases_ds_base.json`), split by template and instruction (cases mixing a training and a test template or instruction are not used). Zhan et al., "InjecAgent", ACL Findings 2024. Cases were generated with GPT-4; accepted for this non-commercial use (2026-10-12) |
| [AgentDojo](https://github.com/ethz-spylab/agentdojo) | MIT | Suite environment data with the "important instructions" attack (injection-task goals) at each injection point, probe only; the Python sources are parsed, not executed. Debenedetti et al., "AgentDojo", NeurIPS 2024 Datasets and Benchmarks |
| [SEP](https://github.com/egozverev/Should-It-Be-Executed-Or-Processed) | MIT | Task data with a probe instruction added at its start or end, and the same data without it; split by probe (about 30% of the 100 probes only in test). The task data was generated with GPT-4 (accepted 2026-10-12). Zverev et al., "Can LLMs Separate Instructions From Data?", ICLR 2025 |
| [leolee99/NotInject](https://huggingface.co/datasets/leolee99/NotInject) | MIT | Benign prompts containing injection trigger words, used as benign rows against over-blocking. Li et al., "InjecGuard", 2024 |
| [microsoft/TaskTracker](https://github.com/microsoft/TaskTracker) | MIT | Its secondary-task lists (`saved_injections.txt` for training, `saved_injections_test.txt` for test only, trivia questions and translated tasks split by hash), inserted TaskTracker-style into SQuAD passages. Abdelnabi et al., "Get my drift? Catching LLM Task Drift with Activation Deltas", SaTML 2025. TaskTracker's own rebuilt dataset is not used |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | CC BY-SA 4.0 | Wikipedia paragraphs as carriers for TaskTracker's instructions, and as benign passages; split by paragraph. Rajpurkar et al., 2018. Share-alike: rows built from them must stay CC BY-SA 4.0 if shared, so they are never combined with CC BY-NC text |
| [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | CC BY-NC 4.0 | Instructions without an input (generated with text-davinci-003), inserted into WildChat pages only; non-commercial use (WebAgent is non-commercial). Taori et al., 2023 |
| [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | Apache-2.0 | Synthetic how-to articles (`wikihow` config, written by Mixtral-8x7B-Instruct from wikiHow titles only) and web articles (`web_samples_v2`), as benign pages full of instructions to the reader; weighted 3× on injection when training. Ben Allal et al., 2024 |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache-2.0 | English assistant replies written by volunteers, as benign instruction-rich text; weighted 3× on injection. Köpf et al., "OpenAssistant Conversations", 2023 |
| [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | CC BY 4.0 | Synthetic documents with PII spans |
| [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | ODC-BY | Zhao et al., "WildChat", ICLR 2024 |
| WebAgent triage questions | Project data | `data/triage/train.jsonl` |
| Generated rows | Project data | Fake secrets and program output from `build_security_dataset.py` |

WebAgent is non-commercial (confirmed 2026-10-12). Reviewed and not used (2026-10-12): Meta's WASP (no standard licence), rogue-security/prompt-injections-benchmark (CC BY-NC, gated), BIPIA-GPT (CC BY-SA), 3nesdeniz/agentic-prompt-injection (generation method not stated), neuralchemy and rgeada/tool-response-injections (part of the source data is marked "Research" only), Microsoft TaskTracker's rebuilt dataset (its rows mix CC BY-SA passages with CC BY-NC instructions; we build the same kind of rows without mixing them), HotpotQA (not needed), databricks-dolly-15k (CC BY-SA; Apache-2.0 sources covered the same need), wikiHow itself (CC BY-NC-SA; Cosmopedia's synthetic how-tos used instead), Open-Prompt-Injection (its data includes Gigaword, under an LDC licence). Excluded earlier: qualifire/prompt-injections-benchmark (CC BY-NC 4.0, non-commercial), ai4privacy PII sets with restrictive licences, and walledai/AdvBench (gated; would need an accepted licence on a logged-in account).
