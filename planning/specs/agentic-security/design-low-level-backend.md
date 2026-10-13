# Low-level design: backend

Paths are relative to `backend/`. Requirement IDs refer to [`requirements.md`](requirements.md).

## 1. Module layout

| File | New? | Purpose |
|---|---|---|
| `app/security/types.py` | new | `Source`, `Outcome`, `ContextItem`, `ItemScores`, `Decision`, `HopResult`, `ToolRequest` |
| `app/security/patterns.py` | new | Load `config/security/patterns.json`; `scan(text) -> set[str]`; argument checks for tools |
| `app/security/context_defender.py` | new | Router, chunker, featurizers, heads, calibration, cache; `score(items) -> list[ItemScores]` |
| `app/security/decision.py` | new | Policy loading, annotation map, schema validation, policy tests, `decide(request) -> Decision` |
| `app/security/hop_guard.py` | new | `check_hop(agent, items, state)` (checkpoint 1); `kept_text(hop, ref, default)` for prompt building; the context variable the safety net reads |
| `app/security/hop_integration.py` | new | `run_with_hop(agent, state, items, build_prompt, invoke)`: check, build the prompt from kept items, call the model in one step |
| `app/security/errors.py` | new | `GuardStop` (stop the task or request) and `GuardDeclined` (safety net refused a model call); both subclass `BaseException` so agents' `except Exception` blocks can't swallow them |
| `app/security/thresholds.py` | new | Default policy thresholds and `merged()` with a model's own (SEC-028a) |
| `app/security/encoder.py` | new | The base MiniLM, loaded once per worker and shared by ContextDefender and alignment |
| `app/security/guarded_llm.py` | new | `GuardedModel` wrapper used by the LLM factory (safety net) |
| `app/security/tool_guard.py` | new | `guard_tool(action, args, state)` (checkpoint 2) |
| `app/security/code_checks.py` | new | AST findings for generated code |
| `app/security/alignment.py` | new | Cosine between arguments and task text |
| `app/security/judge.py` | new | LLM judge for `verify` at checkpoint 2 |
| `app/security/tokens.py` | new | Consent and approval tokens, `TokenStore` |
| `app/security/ledger.py` | new | Read/write `state.context["security"]`; merge for parallel hops |
| `app/security/records.py` | new | Collection, escalation queue, cleanup |
| `app/api/endpoints/guard.py` | new | `POST /api/guard/input` |
| `config/security/` | new | `policies.cedar`, `schema.cedarschema`, `policy_tests.yaml`, `patterns.json`, `source_weights.yaml` |
| `models/security/` | new | `heads.joblib`, `metadata.json` |
| `data/security/` | new | `train.jsonl`, `test.jsonl`, `probe.jsonl`, `README.md`; `collected/`, `escalations/`, `tokens/` git-ignored |
| `scripts/train_context_defender.py`, `scripts/eval_context_defender.py`, `scripts/security_data.py` | new | Training, gate, shared row loading and `REQUIRED_CELLS` |
| `scripts/fetch_security_sources.py`, `scripts/build_security_dataset.py` | new | Download the public source datasets; build `train`/`test`/`probe` from them (section 7.2) |
| `app/services/llm.py` | changed | Factory returns guarded models; `get_trusted_llm()`; coding-agent models |
| `app/agents/*.py` | changed | Labelled items + `check_hop`; `guard_tool` at tool sites; coding agents use the factory |
| `app/graph/workflows.py` | changed | Merge security ledgers after `run_research_agents` |
| `app/api/endpoints/task.py` | changed | User-turn check before the task; consent and approval tokens |
| `app/services/task_manager.py` | changed | `security` block on results; escalation stops the task |
| `app/services/triage.py` | changed | Tie-breaker uses `get_trusted_llm()` |
| `app/core/config_def.py`, `config.py`, `config/*.yaml` | changed | `SecurityGuardConfig`, `security_guard:` section, env overrides |
| `app/core/metrics.py`, `startup_checks.py`, `main.py` | changed | Metrics, checks, router mount, cleanup task |
| `Dockerfile` | changed | Bake the base MiniLM into `/opt/models` again (for ContextDefender) |
| `requirements.txt`, `constraints.txt` | changed | `cedarpy` |

## 2. Types (`app/security/types.py`)

```python
Source = Literal["user_turn", "history", "agent_output", "web_result", "retrieved_doc",
                 "document", "tool_output", "image", "unlabelled"]
UNTRUSTED = {"web_result", "retrieved_doc", "document", "tool_output", "unlabelled"}
Outcome = Literal["allow", "verify", "consent", "decline", "escalate"]
SEVERITY = ["allow", "verify", "consent", "decline", "escalate"]
ToolAction = Literal["tool::web_search", "tool::internal_retrieve", "tool::document_extract", "tool::execute_code"]

@dataclass(frozen=True)
class ContextItem:
    source: Source
    text: str                         # never logged outside dev
    ref: Optional[str] = None         # URL, document title or agent name; goes into reports
    tainted: bool = False             # agent_output from a tainted hop

@dataclass(frozen=True)
class ItemScores:
    axes: Dict[str, int]              # injection, harm, sensitive_data, jailbreak → 0–100
    top_axis: str
    threat_risk: int
    pattern_hits: FrozenSet[str]
    chunks: Tuple[Dict[str, int], ...]  # per-chunk axis scores, for dropping chunks
    scores_available: bool
    model_version: str

@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    policy_ids: Tuple[str, ...]       # @id values; empty for the baseline permit
    reason: str                       # human-readable, never contains item text
    shadow: bool
    latency_ms: float
    error: bool = False               # the decision point failed (SEC-061); callers apply the fallback

@dataclass
class HopResult:
    items: List[ContextItem]          # what the agent may put in its prompt
    decisions: List[Decision]
    tainted: bool
    dropped: List[Dict[str, Any]]     # {source, ref, policy_id, reason}
```

## 3. LLM factory (`app/services/llm.py`, SEC-001, SEC-002, SEC-032)

- `get_agent_llm(name)` returns `GuardedModel(inner, agent=name)`. `get_llm(model)` and `get_llm_without_tags(model)` become private (`_build_llm`, `_build_llm_without_tags`) and are no longer imported by agents.
- New `get_agent_llm("coding_assistant")` / `("coding_review")` honour an OpenAI provider in the agent's YAML config (`provider: openai`, `model: …`), replacing the `ChatOpenAI(...)` and direct `openai` client in the coding agents. Their retry and fallback logic moves into the factory.
- `internal_research.py`'s `get_llm("gpt-3.5-turbo")` retry is replaced by a second `get_agent_llm("internal_research")` call (the retry was for a tags error, which the factory now handles).
- `get_trusted_llm()` returns the unguarded model; allowed only in `app/services/triage.py`, `app/security/judge.py` and `app/utils/diagnostics.py`, whose health check sends a fixed prompt (SEC-002). `get_llm` and `get_llm_without_tags` become private to `services/llm.py`.

`GuardedModel` (`app/security/guarded_llm.py`) is a LangChain `Runnable` that delegates to the inner model. In `invoke`/`ainvoke`:

1. Read `CURRENT_HOP` (a `contextvars.ContextVar` set by `check_hop` through `mark_hop(agent, decisions)` and cleared after one model call). If it's set for this agent, call through. `acheck_hop` marks the hop again in the caller's context, because the scoring thread's context doesn't flow back.
2. If not, build one `unlabelled` item from the prompt with the registered template text removed (`register_template(agent, template_text)` is called by each agent at import for each of its prompt constants, kept as a list per agent; the static parts are the text between `{placeholder}`s, split with the regex `\{[^{}]*\}`), run checkpoint 1 on it, count `webagent_guard_unlabelled_total{agent}`, log a warning, and call through. In act mode decline, escalate and consent raise `GuardDeclined`; in shadow mode, or when the decision point errored, the call goes through.

## 4. Checkpoint 1 (`app/security/hop_guard.py`, SEC-030–SEC-039)

```python
def check_hop(agent: str, items: List[ContextItem], state: Optional[WorkflowState]) -> HopResult
async def acheck_hop(...) -> HopResult      # runs scoring in the threadpool
```

Steps:

1. If the guard is disabled: return all items, no decisions.
2. Score all non-image items in one batch (`ContextDefender.score`). Images get `ItemScores(scores_available=False)` with `source="image"`.
3. For each item, build a Cedar request: principal `Agent::"<agent>"` (`"intake"` for the `/api/task` user-turn check, `"web_chat"` for `/api/guard/input`), action `Action::"converse"`, resource `Task::"<task id>"` (or `Chat::"<decision id>"` for the direct chat), and the context (section 6.2) with the item's scores, `source`, `pattern_hits`, `tainted` (task taint so far), `consented`, `has_image`.
4. Apply outcomes (in act mode; in shadow mode record and keep everything):
   - user turn or history with verify from `user-harm-verify` or `user-jailbreak-verify`: ask `judge.judge_user_turn(text, axis)` (harm first). Flagged → decline (harm) or consent (jailbreak; allow if already consented); cleared → allow, no taint; judge off or failed → verify stands. The judged decision is appended to the hop's decisions (also in shadow mode, not applied).
   - user turn or history with consent/decline/escalate: raise `GuardStop(decision)` (the caller turns it into 409/403, or into a failed task when it happens mid-task).
   - untrusted with decline: drop chunks whose own score crossed the policy (the decision is re-run per chunk only for declined items) or the whole item if more than half its chunks are flagged; record in `dropped`; hop tainted.
   - untrusted with verify: keep; hop tainted. With escalate: drop the item; escalation record; hop tainted.
5. Taint: hop tainted if step 4 tainted it, any kept item has `tainted=True`, the task is already tainted (taint is task-wide: once any hop is tainted, every later hop is), or the user consented.
6. Decision-point error (`Decision.error`): fail open for `user_turn` and `history` (keep, log), fail closed for untrusted sources (drop the item, hop tainted). SEC-061.
7. Write the hop to the ledger (`ledger.add_hop`; escalations are recorded even without a state, as on `/api/guard/input`), set `CURRENT_HOP` with the decisions for LangSmith metadata, return the `HopResult`.

Chunk dropping re-decides each chunk with its own scores and its own pattern hits; flagged token spans are removed with `text_without_chunks`, so overlapping windows don't duplicate the kept text.

Per-agent integration (phase 3), for example `web_research.py`:

```python
items = [ContextItem("web_result", r["content"], ref=r["url"]) for r in search_results]
hop = await acheck_hop("web_research", items + task_items(state), state)
kept = [i for i in hop.items if i.source == "web_result"]
prompt_text = self.prompt.format(search_results=self._format_items(kept), ...)
```

Agents with one variable per item use `kept_text(hop, ref, original)`, which returns the kept (possibly trimmed) text, `"[Left out by the security check.]"` when the item was dropped, or the original when it wasn't checked.

`task_items(state)` returns the `user_turn` and `history` items once per task (cached by hash, so re-scoring is free). An agent's output is wrapped as `ContextItem("agent_output", text, ref=agent, tainted=hop.tainted)` by `ledger.output_item` when the next agent reads it.

**User turn before the task (SEC-033).** `create_task` builds `user_turn` and `history` items from the request and calls `check_hop("intake", …)` before `register_pending`. `GuardStop` becomes:

| Outcome | Response |
|---|---|
| consent | `409` |
| decline | `403` |
| escalate | `403` + escalation record |

The body is top-level (not FastAPI's `detail` wrapper), built by `refusal_body()` and shared with `/api/guard/input`: `{"code": "consent_required" | "declined" | "escalated", "outcome", "reason", "policy_id", "policy_ids", "consent_token"?}`.

A request with a valid `consent_token` for this exact turn (bound to the hash of the query and history) is redeemed once, sets `consented = true` and marks the task tainted. `create_task` also stores the chat mode in `context["chat_mode"]` for the policy context's `mode`.

**Parallel hops (SEC-039).** `run_research_agents` gives each agent a deep copy of the state; after `gather`, `ledger.merge(state, [results])` unions hops, ORs taint and sums `declined_actions_in_task`.

## 5. Checkpoint 2 (`app/security/tool_guard.py`, SEC-040–SEC-047)

```python
async def guard_tool(action: ToolAction, args: Dict[str, Any], state: WorkflowState) -> Decision
```

1. Build `ToolRequest` features:
   - search/retrieve: `patterns.scan_args(args["query"], user_turn=state.query)` → `pattern_hits` from `api_key`, `private_key`, `password`, `email`, `phone`, `long_token`, `url`, `site_operator`, `query_too_long` (matches present in the user turn are left out); `alignment` (section 5.1).
   - document: `"document_provenance"` in `pattern_hits` if `args["document_id"] != state.inputs.get("document_id")`.
   - code: `code_checks.findings(args["code"])` → `code_findings`; `code_execution_enabled`, `sandbox`.
2. Context adds `source = "tool"`, `tainted` (ledger), `approved_tools` (from the task's approval tokens), `declined_actions_in_task`, `mode`, `env`. The principal is the calling agent, the resource the task.
3. `decide()` (context building and the judge run in a thread with `asyncio.to_thread`). In shadow mode, record and return allow. On a decision-point error: decline code execution and document extraction, allow search and retrieval (SEC-061).
4. `verify` → `judge.check(task_text, action, args)` if the judge is on, else decline.
5. Apply: decline/consent → record in `security.skipped` and increment `declined_actions_in_task` for decline. Consent also issues an approval token, `tokens.issue("approval", {"query_hash": …}, policy_id, action=action)`, listed in `security.approvals` for the web app's "Approve and re-run" (SEC-051). Escalate → raise `GuardStop`; the task fails with `error_code: "escalated"` (a decline that stops the task gives `"declined"`).

Call sites (SEC-040):

| File | Call | Action |
|---|---|---|
| `web_research.py` | `perform_web_search(query=search_query, …)` (main) | `tool::web_search` |
| `web_research.py` | `perform_web_search(query=simplified_query, …)` | `tool::web_search` |
| `web_research.py` | `perform_web_search(query=additional_query, …)` (follow-up) | `tool::web_search` |
| `internal_research.py` | `self.retriever.retrieve(search_query, …)` (both calls) | `tool::internal_retrieve` |
| `document_extraction_agent.py` | loading `state.inputs["document_id"]` | `tool::document_extract` |
| `coding_review.py` | `run_python_file(...)` in `execute_python_code`, every attempt | `tool::execute_code` |

### 5.1 Alignment (`alignment.py`)

`alignment = round(100 × max(0, cos(embed(args_text), embed(task_text))))`, with ContextDefender's encoder; `task_text` is `research_plan.standalone_query` or `state.query`. Embeddings of the task text are cached per task.

### 5.2 Code checks (`code_checks.py`)

`ast.parse` the code (failure → `{"unparseable"}`), then walk the tree:

| Finding | Triggers |
|---|---|
| `network` | imports of `socket`, `http`, `urllib`, `requests`, `httpx`, `aiohttp`, `ftplib`, `smtplib`, `paramiko` |
| `process` | `subprocess`, `os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `pty` |
| `filesystem` | `open(...)` with a write mode on an absolute path or a path containing `..`; `shutil.rmtree`; `os.remove` outside the work folder |
| `eval` | `eval`, `exec`, `compile`, `__import__`, `importlib` |
| `native` | `ctypes`, `cffi` |
| `env` | `os.environ`, `os.getenv` |

### 5.3 Judge (`judge.py`)

Tool calls (checkpoint 2): the task text and `"<action>: <args as JSON>"`, asking for exactly `ALIGNED` or `NOT_ALIGNED`. User turns (checkpoint 1, `judge_user_turn(text, axis)`): harm goes to OpenAI's moderation endpoint (`omni-moderation-latest`) by default, flagged when the highest of its 13 category scores reaches the loaded model's `harm_moderation_cutoff` (`context_defender.current_moderation_cutoff()`; moderation's `flagged` when unset), because a chat model may refuse to judge clearly harmful text and that is the content this judge sees; jailbreak goes to Llama Prompt Guard 2 (`app/security/prompt_guard.py`: local `transformers` model from `models/security/prompt_guard/`, P(malicious) as the highest over 512-token windows, flagged at `prompt_guard_threshold`) when `jailbreak_judge: prompt_guard`; uat/prod refuse to start with that setting and no model. `scripts/fetch_security_sources.py --prompt-guard 86M|22M` downloads it with the account's Hugging Face token (gated repository) into a temporary folder and copies it in only when complete, with `LICENSE`, `USE_POLICY.md` and the `NOTICE` the Llama 4 Community License requires. With `jailbreak_judge: llm`, jailbreak (and harm with `harm_judge: llm`) goes to the trusted chat model with the message alone, up to 4,000 characters, between `<<<MESSAGE` / `MESSAGE>>>` markers with an instruction to treat it as data, asking for exactly `HARMFUL` or `FINE` (harm) or `JAILBREAK` or `FINE` (jailbreak); a reply that refuses to judge (for example "I can't help with that") or a provider error naming a content policy counts as flagged; timeouts, network errors and other answers return `None`. Cached by (SHA-256 of the text, axis, backend). `train_context_defender.py --choose-moderation-cutoff [--moderation-cache file]` picks the cut-off on the model's held-out user turns: the lowest value in a fixed grid (0.001–0.9) for which direct declines + judge rate × flag rate on ordinary questions stays under 75% of the harm limit (texts go to the moderation endpoint in batches of 32; the cache keeps only SHA-256 → score). `eval_context_defender.py --with-judge` measures harm end to end with the moderation judge (sends the verify-band test turns to OpenAI). `get_trusted_llm().invoke` on a 2-thread pool with a 3 s timeout (same pattern as the triage tie-breaker). Cached per task by (action, args). Counted in `webagent_guard_judge_total{verdict}`.

## 6. Decision point (`app/security/decision.py`, SEC-010–SEC-014)

### 6.1 Loading

- Read `config/security/policies.cedar`. Split into policies (Cedar statements end with `;` outside strings), read `@id` and `@outcome` for each, and build `{f"policy{i}": (id, outcome)}`; the baseline `permit` maps to `("baseline-permit", "allow")`.
- Lint (SEC-014): every `forbid` has `@id` and `@outcome`; ids are unique; outcomes are valid; `consent` policies are the only ones that read `context.approved_tools` or `context.consented` (SEC-052); no `consent` policy for untrusted sources at `converse` (SEC-036).
- Validate each policy against `schema.cedarschema` (`cedarpy.validate_policies`).
- Run `policy_tests.yaml`: each row `{action, context, expected}` through `decide`; any mismatch is a load failure.

### 6.2 Schema and context

```cedarschema
entity Agent;
entity Task;
entity Chat;
type GuardContext = {
  source: String, injection: Long, harm: Long, sensitive_data: Long, jailbreak: Long, threat_risk: Long,
  top_axis: String, scores_available: Bool, pattern_hits: Set<String>, tainted: Bool, alignment: Long,
  code_findings: Set<String>, approved_tools: Set<String>, consented: Bool, has_image: Bool,
  declined_actions_in_task: Long, mode: String, env: String, code_execution_enabled: Bool, sandbox: String,
  injection_watch_at: Long, injection_drop_at: Long, harm_verify_at: Long, harm_decline_at: Long,
  jailbreak_verify_at: Long, jailbreak_consent_at: Long
};
action "converse", "tool::web_search", "tool::internal_retrieve", "tool::document_extract", "tool::execute_code"
  appliesTo { principal: [Agent], resource: [Task, Chat], context: GuardContext };
```

Checked on 2026-10-11 with `cedarpy` 4.8.7: this schema validates all 15 starting policies (the 10 in the plan and the 5 below, counting the baseline `permit`) with `cedarpy.validate_policies(policies, schema)`, and the set gives the expected outcome for 21 test cases.

`decide()` fills every attribute with its default before calling `cedarpy.is_authorized` (SEC-013). Matches of policies annotated `@shadow("true")` are set aside into `Decision.shadow_policy_ids` and don't count toward the outcome; if only shadow-only policies matched, the outcome is allow (SEC-012a). `policy_tests.yaml` rows may give `expected_shadow`. The six `*_at` attributes are the loaded model's thresholds (`context_defender.current_thresholds()`, from `metadata.json["thresholds"]` through `app/security/thresholds.py`), defaulting to 20 and 40 (injection watch and drop), 50 and 85 (harm verify and decline), 50 and 70 (jailbreak verify and consent); the checkpoint-1 policies compare scores with them instead of literal numbers (SEC-028a).

### 6.3 decide()

```python
def decide(principal: str, action: str, resource: str, context: dict) -> Decision
```

`is_authorized` → if allowed, `allow`; else the most severe `@outcome` among `diagnostics.reasons` (mapped through the annotation table). `reason` is the matching policy's `@reason` annotation if present, else a default per outcome. Errors in `diagnostics.errors` count as a decision-point error (SEC-061). Latency observed in `webagent_guard_latency_seconds{checkpoint}`.

### 6.4 Starting policies

The 10 policies in the plan (`baseline permit`, `untrusted-injection-watch`, `untrusted-injection-drop` (which also fires on the `injection_phrase`, `role_marker` and `hidden_text` patterns), `user-harm-decline`, `user-jailbreak-consent`, `search-secret-decline`, `tainted-search-verify`, `tainted-code-consent`, `repeated-declines-escalate`, `classifier-down-untrusted`), plus 5 tool policies below, 2 user-turn verify policies (`user-harm-verify`, `user-jailbreak-verify`: scores from the verify threshold up to the decline or consent threshold; SEC-035a) and one shadow-only policy (`untainted-search-misaligned`: search or retrieval with alignment below 30 in an untainted task; SEC-044a). 18 in all:

```cedar
@id("document-provenance-decline") @outcome("decline")
forbid (principal, action == Action::"tool::document_extract", resource)
when { context.pattern_hits.contains("document_provenance") };

@id("search-personal-data-verify") @outcome("verify")
forbid (principal, action in [Action::"tool::web_search", Action::"tool::internal_retrieve"], resource)
when { context.pattern_hits.containsAny(["email", "phone", "long_token"]) };

@id("search-url-verify") @outcome("verify")
forbid (principal, action in [Action::"tool::web_search", Action::"tool::internal_retrieve"], resource)
when { context.pattern_hits.containsAny(["url", "site_operator", "query_too_long"]) };

@id("code-dangerous-decline") @outcome("decline")
forbid (principal, action == Action::"tool::execute_code", resource)
when { context.code_findings.containsAny(["process", "native", "env", "unparseable"]) };

@id("code-disabled-decline") @outcome("decline")
forbid (principal, action == Action::"tool::execute_code", resource)
when { !context.code_execution_enabled };
```

## 7. ContextDefender (`app/security/context_defender.py`, SEC-020–SEC-028)

- **Router:** a chunk goes to TF-IDF if it parses as code (`ast.parse` succeeds and it has ≥ 2 statements), is ≥ 30% base64/hex characters, or < 50% of its tokens are in a 50,000-word vocabulary; else to MiniLM.
- **Chunker:** the MiniLM tokenizer; 96 tokens, stride 64 by default since `cd-2026-10-12-4` (200/150 before: on the same data injection recall went from 0.868 to 0.926, benign-page false positives halved, and CPU time per page stayed about 100 ms because shorter chunks are cheaper to encode). The window is stored in `heads.joblib` (`chunking`) and applied by `load_model` (`set_chunking`), so training and serving always agree.
- **Featurizers** (`featurize()`, shared by training and runtime): `all-MiniLM-L12-v2` since `cd-2026-10-13-2` (L6 before; same recipe gave injection recall 0.973 against 0.939 at the gate's limit, about 1.7× CPU per page), fine-tuned for ContextDefender and saved in `models/security/encoder/` (Git LFS; loaded with `get_encoder_at`, `local_files_only`), or the frozen base model from `app/security/encoder.py` when the bundle has no fine-tuned encoder (`SENTENCE_TRANSFORMERS_HOME`, `local_files_only` in uat/prod), `normalize_embeddings=True`; character 3–5-gram `TfidfVectorizer` (`char_wb`, `min_df=2`, at most 50,000 features, sublinear TF; fitted at training, kept sparse). With `combine_features` (the default), the MiniLM route is the embedding concatenated with the TF-IDF n-grams: the embedding alone underfit injections that fill only part of a chunk (recall 0.11 on its own training rows).
- **Fine-tuning** (`--finetune`): the training rows are split 85/15 by a hash of the row (of the inserted attack for page rows, so pages sharing one attack stay together). MiniLM is fine-tuned on the 85% with a 4-output linear head and `BCEWithLogitsLoss` (positive weight per axis, capped at 10), 2 epochs, AdamW (2e-5 encoder, 1e-3 head), warm-up then linear decay.
- **Hard negatives:** chunks from benign sets written to look risky weigh 3× on the axis they imitate (`HARD_NEGATIVES`: XSTest safe, JBB benign and Aegis safe for harm; TrustAIRLab regular prompts for jailbreak), in the fine-tuning loss and in the heads' fit; calibration stays unweighted.
- **Thresholds:** after calibration, `select_thresholds` scores the held-out rows (max over chunks, as at runtime) and picks per axis the lowest score whose false-positive rate's one-sided 95% Wilson upper bound is ≤ the gate's limit (benign untrusted items for injection, ordinary questions for harm and jailbreak), so with few held-out rows the acting scores go up (and for harm and jailbreak more is left to the judge); within 30–95 (injection) or 50–98 (harm, jailbreak), and below the harm and jailbreak scores the highest verify score reaching 92% recall on held-out user turns; written to `metadata.json["thresholds"]` with the held-out recall and false positives (SEC-028a).
- **Heads:** per axis and route, a binary `LogisticRegression` (risk present: `low` or `high`; `class_weight="balanced"`, `C=10`). With a fine-tuned encoder it is fitted on the 85% and Platt-calibrated (`cv="prefit"`) on the 15% neither it nor the encoder saw, because scores on the encoder's own training rows are overconfident. Without fine-tuning, `CalibratedClassifierCV` (sigmoid, 3-fold) on all rows. A training row with a `span` labels only the chunks containing the inserted injection (matched by alphanumeric anchors) as injection.
- **Pooling:** per axis, the max over chunks of the calibrated probability.
- **Cache:** `functools.lru_cache`-style dict keyed by (SHA-256 of text, model version), 10,000 entries, per worker.
- **Loading:** lazy, lock-guarded, 60 s back-off after a failure (as triage); `warm_up()` at startup; `torch.set_num_threads(1)`.
- **Artifacts:** `models/security/heads.joblib` (heads, TF-IDF vectorizer, vocabulary list, `chunking`, `combine_features`, `c`, `finetuned_encoder`), `models/security/encoder/` (fine-tuned SentenceTransformer, about 88 MB), `metadata.json` (version, encoder name, `finetuned_encoder`, data hash, calibration rows, test metrics, `test.passed`).
- **Memory:** with a fine-tuned encoder each worker holds two MiniLM copies (ContextDefender's and the base one for alignment), about 90 MB more than with the shared base encoder.
- **Errors:** a model error while scoring falls back to pattern-only scores with `scores_available = false`.

### 7.1 Scripts

- `scripts/train_context_defender.py [--data data/security/train.jsonl] [--out models/security] [--version …] [--finetune] [--epochs 2] [--embedding-only] [--C 10] [--chunk-size 96] [--chunk-stride 64]`: load rows, chunk with span-aware labels, route, optionally fine-tune the encoder, featurize each chunk once per route, fit and calibrate the heads, write artifacts.
- `scripts/eval_context_defender.py [--data …] [--check-data] [--write-metadata]`: per-axis and per-source recall/false positives at the model's thresholds (injection, harm, jailbreak) or 50 (sensitive data), ECE per head, the gate (SEC-028), exit code 1 on failure. `--check-data`: schema, label values, sources, near-duplicates across files, minimum rows per source, positives in every `REQUIRED_CELLS` (axis, source) cell.

### 7.2 Data scripts

- `scripts/fetch_security_sources.py --out <raw dir>`: downloads the public sets (Hugging Face `datasets`, large sets streamed and sampled; HarmBench and BIPIA from GitHub raw). Prints counts only.
- `scripts/build_security_dataset.py --raw <raw dir> [--total 10000]`: per-group quotas, filtering of benign rows that look like attacks, injections inserted into benign pages with their `span`, generated fake secrets and program output, then splits by near-duplicate cluster (MiniLM cosine ≥ 0.9, union-find) with BIPIA forced to test and treycsa to probe. Licences and composition: `data/security/README.md`.

## 8. Tokens (`app/security/tokens.py`, SEC-050–SEC-053)

```python
def issue(kind: Literal["consent", "approval"], binding: Dict[str, str], policy_id: str,
          ttl_s: int = 900, action: Optional[str] = None) -> str
def redeem(token: str, kind: str, binding: Dict[str, str]) -> Optional[Dict[str, Any]]   # payload or None
def query_hash(query: str) -> str
def consent_binding(query: str, history) -> Dict[str, str]
```

- Token: `base64url(json payload) + "." + base64url(hmac_sha256(SECRET_KEY, payload))`; payload `{kid, kind, binding_hash, policy_id, exp, jti}`. `binding` is `{"text_hash": …}` for consent and `{"query_hash": …, "action": …}` for approval.
- `redeem` checks the signature, expiry, kind and binding, then claims `jti` by creating `data/security/tokens/<jti>` with `O_CREAT | O_EXCL`; an existing file means reused, and a store that can't be written (`OSError`) means the token is refused. Cleanup deletes claims whose file is more than an hour old (well past the 15-minute TTL).
- `POST /api/task` accepts `consent_token` and `approval_tokens: List[str]`; an approval is accepted only for the same query (`query_hash`) and becomes its payload's `action` in `approved_tools`.

## 9. Endpoint: `POST /api/guard/input` (SEC-034)

```python
class GuardMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(max_length=100_000)

class GuardInputRequest(BaseModel):
    messages: List[GuardMessage] = Field(min_length=1, max_length=60)
    has_image: bool = False
    consent_token: Optional[str] = None

class GuardInputResponse(BaseModel):
    outcome: Literal["allow", "verify", "consent", "decline", "escalate"]
    reason: str
    policy_ids: List[str]
    consent_token: Optional[str] = None
    decision_id: str
```

The last `user` message is `user_turn`; earlier messages are `history`; `system` messages are ignored (the web app builds its own system prompt). Mounted like `/api/task` with `require_api_key`; rate-limited. Returns 200 for allow/verify, 409 for consent, 403 for decline/escalate, with the same body.

## 10. Configuration (SEC-062–SEC-064)

```python
class SecurityGuardConfig(BaseModel):
    enabled: bool = True
    input_mode: Literal["shadow", "act"] = "shadow"
    tool_mode: Literal["shadow", "act"] = "shadow"
    judge: bool = False
    harm_judge: Literal["moderation", "llm", "off"] = "moderation"   # checkpoint-1 judge for harm (SEC-035a)
    jailbreak_judge: Literal["llm", "off"] = "llm"
    collect: bool = False
    collect_retention_days: int = Field(default=30, gt=0)
    model_dir: str = "models/security"
    policy_dir: str = "config/security"
```

Environment overrides: `WEBAGENT_GUARD_ENABLED`, `WEBAGENT_GUARD_INPUT_MODE`, `WEBAGENT_GUARD_TOOL_MODE`, `WEBAGENT_GUARD_JUDGE`, `WEBAGENT_GUARD_COLLECT` (bad values warn and keep YAML, as for triage). `dev.yaml` sets `collect: true`.

## 11. Observability (SEC-080–SEC-083)

- Metrics in `app/core/metrics.py` as listed in SEC-080; the `shadow` label is `"true"`/`"false"`.
- Log line (logger `app.security`, INFO for non-allow): `guard checkpoint=1 action=converse outcome=decline policies=untrusted-injection-drop source=web_result top=injection:72 task=…`; `text="…"` only in dev.
- `task_manager` copies `state.context["security"]` into the saved result's `security` block (SEC-082).
- `GuardedModel` adds `metadata={"guard.outcome": …, "guard.policy_ids": …}` to the run config (SEC-083).
- `guard_tool` observes every search and retrieval's alignment in `webagent_guard_search_alignment{tainted}` (numbers only, no text), and `record_guard_decision` counts shadow-only matches in `webagent_guard_shadow_policy_total`; collection rows carry `shadow_policy_ids` and are written for allowed decisions that had shadow matches. Checkpoint-1 rows carry `item` (source, axis scores, top axis, threat risk, pattern hits, chunk count, model version; the URL for web results; text only in dev) and checkpoint-2 rows `context` (alignment, taint, pattern hits, code findings, approvals, declines so far). `check_hop` and `guard_tool` are timed whole in `webagent_guard_hop_seconds` and `webagent_guard_tool_seconds`. `scripts/review_security_collection.py` summarises the collection (counts only), writes a labelling sheet and estimates the real false-positive rate with a 95% upper bound (shadow-week runbook).
- `records.py` writes collection and escalations with `O_APPEND` single writes (as triage collection) and cleans up daily from the startup hook.

## 12. Startup (SEC-014, SEC-065)

`startup_checks.check_security_guard()`: with the guard enabled in uat/prod, policies load (parse, lint, validate, tests). When the model artifacts exist, the encoders they need must be present in any mode (`models/security/encoder` for a fine-tuned bundle, and the base MiniLM), and `metadata.json` must have `test.passed` only if `input_mode` is `act`; in shadow mode a failed or missing gate logs a warning, because shadow mode is how data is collected for a model that hasn't passed yet. The base encoder must be present (`encoder_files_present`). The `main.py` startup hook warms up ContextDefender and starts the daily cleanup of `security/collected`, `escalations` and `tokens`.

## 13. Docker

- Bake the base `sentence-transformers/all-MiniLM-L6-v2` into `/opt/models` and set `SENTENCE_TRANSFORMERS_HOME=/opt/models` (the step 3.2.0 removed), because ContextDefender uses the base encoder and the triage encoder is fine-tuned. About +90 MB.
- `models/security/` and `config/security/` come in with `COPY . .`; `data/security/` stays out (`.dockerignore`).

## 14. Tests

| File | Covers |
|---|---|
| `tests/test_llm_entry_points.py` | No direct model clients in `app/agents/`; `get_trusted_llm` only in the three allowed files (SEC-001, SEC-002) |
| `tests/test_security_decision.py` | Loading, annotation map, lint rules, schema validation, `policy_tests.yaml`, severity, defaults (SEC-010–SEC-014, SEC-052) |
| `tests/test_security_patterns.py` | Pattern table and near-misses; argument checks with the user-turn exception (SEC-021, SEC-041) |
| `tests/test_context_defender.py` | Router, chunker, pooling, cache, calibration shape, determinism, unavailable model (SEC-020–SEC-026) |
| `tests/test_security_hop_guard.py` | Outcomes per source, chunk dropping, taint propagation, parallel merge, safety net, images, shadow (SEC-030–SEC-039, SEC-062) |
| `tests/test_user_turn_judge.py` | The judge for borderline user turns: flag, clear, consented, failures, judge off, shadow, cache (SEC-035a) |
| `tests/test_security_tool_guard.py` | Every call site, each check, alignment, judge, outcomes, counter (SEC-040–SEC-047) |
| `tests/test_security_code_checks.py` | AST findings table (SEC-043) |
| `tests/test_security_tokens.py` | Sign, expiry, binding, reuse across two stores, tamper (SEC-050–SEC-053) |
| `tests/test_guard_endpoint.py` | Contract, auth, statuses, consent round trip (SEC-034) |
| `tests/test_task_guard.py` | User turn before task; 409/403; approval re-run; escalation fails the task; `security` block (SEC-033, SEC-046, SEC-051, SEC-082) |
| `tests/test_security_records.py` | Collection on/off, escalation lines, cleanup, no text outside dev (SEC-060, SEC-064, SEC-081) |
| `tests/test_train_context_defender.py` | Training smoke test with a fake encoder; gate exit codes; `--check-data` |
| `tests/redteam/` | Poisoned pages, hidden-text documents, forged history, code asking for the network: end-to-end with fakes (SNFR-03) |
| `tests/test_security_benign.py` | Benign regression set through both checkpoints (SNFR-04) |
