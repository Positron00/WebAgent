# Shadow week runbook (release 3.2.2, uat)

Goal: run the security guard on real uat traffic for about a week in shadow mode (it checks everything and records what it would do, but blocks nothing), then decide whether checkpoint 1 can act. Tasks 3.8 and 5.1–5.3 in [`tasks.md`](tasks.md); targets in [`requirements.md`](requirements.md) (SEC-028, SNFR-01, SNFR-05).

## 1. Before deploying

- [ ] `git lfs pull` (the ContextDefender encoder and dataset are LFS files; without them the image has pointer files and the guard can't score).
- [ ] Build and push the backend image: `docker build --build-arg APP_VERSION=3.2.2 -t webagent/backend:3.2.2 backend/`, then deploy with `backend/kubernetes/deploy.sh -e uat -v 3.2.2` (the script applies the manifests; it doesn't build). It needs `OPENAI_API_KEY`, `WEBAGENT_SECURITY_SECRET_KEY`, `WEBAGENT_API_KEY` and `GRAFANA_ADMIN_PASSWORD` in the environment.
- [ ] Environment: `WEBAGENT_ENV=uat`, a real `WEBAGENT_SECURITY_SECRET_KEY` (signs consent and approval tokens), `WEB_CONCURRENCY=2`. The guard itself uses `OPENAI_API_KEY` only if the harm judge is turned on.
- [ ] `backend/config/uat.yaml`, `security_guard:` — `enabled: true`, `input_mode: shadow`, `tool_mode: shadow`, `collect: true`, `judge: false`. (Set in 3.2.2.)
- [ ] Kubernetes: memory limit 2 GiB, request 1.5 GiB (`kubernetes/deployment.yaml`), one replica (collection writes to `/app/data`, which must be writable).
- [ ] Optional, decide before deploying: run triage's own pending shadow week at the same time (`triage: enabled: true, mode: shadow, collect: true` in `uat.yaml`). Not set by default.

## 2. Right after deploying

- [ ] The pod starts: startup checks pass (policies load; encoder files present).
- [ ] `/metrics` lists `webagent_guard_decisions_total`, `webagent_guard_hop_seconds`, `webagent_guard_tool_seconds`, `webagent_guard_search_alignment`, `webagent_contextdefender_score`.
- [ ] Run one agentic research task; a file appears in `/app/data/security/collected/<date>.jsonl`.
- [ ] `webagent_guard_unlabelled_total` stays at 0 (every LLM call went through a labelled checkpoint).

## 3. Every day

Prometheus (adjust the range to the period):

| What | Query | Target |
|---|---|---|
| Checkpoint 1 time per hop, p95 | `histogram_quantile(0.95, sum by (le) (rate(webagent_guard_hop_seconds_bucket[1d])))` | < 0.15 s (SNFR-01) |
| Checkpoint 2 time per tool call, p95 | `histogram_quantile(0.95, sum by (le) (rate(webagent_guard_tool_seconds_bucket[1d])))` | < 0.02 s without the judge |
| What checkpoint 1 would do | `sum by (outcome) (increase(webagent_guard_decisions_total{checkpoint="1"}[1d]))` | watch decline / consent / escalate |
| Real searches' alignment (untainted) | `histogram_quantile(0.1, sum by (le) (rate(webagent_guard_search_alignment_bucket{tainted="false"}[1d])))` | sets the shadow rule's threshold |
| Shadow rule matches | `increase(webagent_guard_shadow_policy_total[1d])` | how often `untainted-search-misaligned` would act |
| Memory per pod | `container_memory_working_set_bytes` (or `kubectl top pod`) | ≤ 1.6 GiB (2 GiB with 20% headroom, SNFR-05) |
| Unlabelled LLM calls | `increase(webagent_guard_unlabelled_total[1d])` | 0 |

And the collection summary (counts only, no text), from `backend/` on the pod or a copy of the folder:

```bash
python scripts/review_security_collection.py --dir /app/data/security/collected
```

## 4. End of the week: review

1. Write the labelling sheet: every untrusted item (web page, document, retrieved passage, tool output) the guard would have acted on, with its URL for web pages and its scores, never text:
   ```bash
   python scripts/review_security_collection.py --dir /app/data/security/collected --sheet review.csv
   ```
2. Open each URL and fill in `label` with `injection` or `benign`. Documents and tool output have no URL; their task id finds the run in the results folder.
3. Estimate the real false-positive rate, with a 95% upper bound, against every untrusted item judged that week:
   ```bash
   python scripts/review_security_collection.py --dir /app/data/security/collected --labels review_labelled.csv
   ```
4. Real injections go into the next training round (task 5.3), through a reviewed import, never by copying collected text by hand.
5. From the alignment distribution, pick the threshold for `untainted-search-misaligned`: the score that the lowest few percent of real untainted searches fall under, compared with InjecAgent's misaligned searches (median 9–16).

## 5. Deciding on act mode (checkpoint 1)

Switch `input_mode: act` in uat only if all of these hold:

- [ ] At least about 2,000 untrusted items were judged, and the false-positive rate's 95% upper bound from step 4.3 is ≤ 2% (the release gate's limit, now on real pages).
- [ ] Checkpoint 1 p95 < 150 ms per hop; checkpoint 2 p95 < 20 ms per tool call.
- [ ] Memory ≤ 1.6 GiB per pod with two workers.
- [ ] No unlabelled LLM calls; no guard errors in the logs; `scores_unavailable` near 0 in the summary.
- [ ] The user turns that would have been declined or asked for consent were read and judged reasonable (the summary counts them; their text is not collected outside dev).

Then: act in uat for a day or two, watch the same numbers, then prod. Checkpoint 2 (`tool_mode: act`) and the shadow search rule follow once its threshold is set.

## 6. Rolling back

Any time, without a new image: set `WEBAGENT_GUARD_INPUT_MODE=shadow` (or `WEBAGENT_GUARD_ENABLED=false`) and restart the pod.

## 7. What collection holds

Per decision: time, task id, checkpoint, outcome, policy ids, scores and pattern hits, web-result URLs, and tool arguments (search queries, code). User text is stored only as a hash outside dev. Files live in `/app/data/security/collected/` on the uat volume, are git-ignored, and are deleted after 30 days (`collect_retention_days`).
