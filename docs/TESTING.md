# Testing Guide

WebAgent has two test suites: Jest for the Next.js web app and pytest for the FastAPI backend. Neither calls a real LLM or web service; model replies and HTTP responses are faked, so the tests are fast, free and repeatable.

## Running Tests

```bash
# Web app (Jest)
npm test                      # all tests
npx jest src/__tests__/api    # one folder or file
npm run test:coverage         # with coverage; fails if coverage drops below the floor

# Backend (pytest, in the `agents` conda environment)
npm run test:backend          # all tests
npm run coverage:backend      # with a coverage report
python -m pytest backend/tests/test_report_parsing.py   # one file, from the repo root

# Type check and lint
npm run check-types
npm run lint
```

Backend tests never trace to LangSmith and never reach the network; `conftest.py` enforces both (see below).

## Web App Tests

Setup:

- **Jest** with **babel-jest** (config: `jest.config.js`, Babel config: `babel-test.config.js`)
- **React Testing Library** for components and contexts
- **jest-environment-jsdom** by default; API route tests use the Node environment (see below)

Layout:

```
src/
├── __tests__/
│   ├── api/                 # Next.js API routes (Node environment)
│   │   ├── chatRoute.test.ts      # /api/chat: validation, Together errors, fallback notice, image format
│   │   ├── taskProxy.test.ts      # /api/task proxy: forwarding, X-API-Key, backend down
│   │   ├── triageProxy.test.ts    # /api/triage proxy; "unreachable" header on 503
│   │   └── settingsRoute.test.ts  # /api/settings: default mode from .env.local, re-read on change
│   ├── contexts/
│   │   ├── ChatContext.test.tsx   # question stays visible; no invented sources; route, logs, research in depth
│   │   └── AppContext.mode.test.tsx  # Agentic mode: first paint, server default not saved, user choice saved
│   ├── utils/
│   │   ├── apiClient.test.ts
│   │   ├── apiClient.agentic.test.ts  # routing per mode, previous_messages, override, polling, fallback
│   │   ├── routeQuestion.test.ts  # every triage status code and what the web app does
│   │   └── agenticMode.test.ts    # saved mode and the one-time migration from the old on/off setting
│   └── Chat.test.tsx
└── components/chat/__tests__/     # MessageInput (+ .mode: send disabled, image note), MessageList,
                                   # AgenticModeToggle, RouteLabel (tooltips via Tooltip.tsx)
```

### Testing API routes

Next.js route handlers need the web `Request`/`Response` objects, so their test files start with:

```ts
/**
 * @jest-environment node
 */
```

`jest.setup.js` only installs browser mocks when `window` exists, so it works in both environments. Routes that read environment variables when they load (for example `WEBAGENT_BACKEND_URL` or `TOGETHER_API_KEY`) must set them first and then import the route inside `beforeAll`:

```ts
beforeAll(async () => {
  process.env.WEBAGENT_BACKEND_URL = 'http://backend.test:9000';
  ({ POST } = await import('@/app/api/task/route'));
});
```

### Browser API mocks

`jest.setup.js` mocks `localStorage`/`sessionStorage`, `matchMedia`, `SpeechRecognition`, `IntersectionObserver`, `ResizeObserver`, `URL.createObjectURL`, `scrollIntoView`, `next/image` and a default `fetch`. Tests that check requests replace `global.fetch` with their own `jest.fn()`.

### React contexts

Render components inside the real providers:

```tsx
render(
  <AppProvider>
    <ChatProvider>
      <ComponentUnderTest />
    </ChatProvider>
  </AppProvider>
);
```

Mock `@/utils/apiClient` to control what the "server" returns (see `ChatContext.test.tsx`).

### Coverage floor

`jest.config.js` sets `coverageThreshold` (lines/statements/branches/functions) at least 5 points below measured coverage, so `npm run test:coverage` fails if coverage drops. Raise the numbers when you add tests. CI enforces a backend coverage floor via `--cov-fail-under` in `.github/workflows/ci.yml`.

## Backend Tests

Layout (`backend/tests/`):

| File | What it checks |
|------|----------------|
| `test_workflow_state_flow.py` | Runs the real LangGraph workflow with stand-in agents: reports carry between steps, research rounds count 1→2→3 and stop |
| `test_report_parsing.py` | Team Manager and Supervisor JSON parsing, report building, fallbacks |
| `test_coding_agents.py` | Coding Assistant output parsing and Coding Review code execution, timeouts, fixes |
| `test_llm_selection.py` | Provider choice and fallback, which model goes to which provider, rate-limit queueing |
| `test_together_retry.py` | Retrying temporary Together errors (429, 5xx), not retrying permanent ones |
| `test_together_usage.py` | Token counts from Together reach LangChain callbacks / LangSmith |
| `test_result_store.py` | Saving and reloading task results; rejects unsafe task IDs |
| `test_log_context.py` | `[task <id>]` log tagging, including parallel agents |
| `test_setup_langsmith.py` | App LangSmith settings override ones exported in the shell |
| `test_task_status_shared.py` | Task status is visible to every uvicorn worker; stale "processing" records time out |
| `test_health_and_rate_limit.py` | `/health` needs no key and isn't throttled; rate limits are per user behind the proxy |
| `test_code_sandbox.py` | Generated code can't see API keys; docker mode has no network and resource limits |
| `test_deploy_config.py` | `WEBAGENT_ENV` from the shell wins over `.env.local`; paths and version inside the image |
| `test_followup_query.py` | Follow-up questions are searched as the supervisor's standalone rewrite; a document step planned without a document becomes research |
| `test_web_search_async.py` | Every Tavily call, including the retry, runs off the event loop |
| `test_web_research_prompt.py` | The web research prompt has no unfilled placeholders and ends with `ANALYSIS:`; reasoning before the answer is dropped; per-agent output limits |
| `test_startup_checks.py` | uat/prod refuse to start with unsafe settings or without a triage classifier that passed the release gate (unless triage is off); the command-line check the Docker image runs exits with a message |
| `test_mlflow_logging.py` | Repeated MLflow logging inside an active run doesn't fail |
| `test_mlflow_experiment_race.py` | A worker that loses the race to create the MLflow experiment keeps starting |
| `test_direct_answer.py` | The backend direct path: a prompt the text model answers, reasoning stripped from the reply, a note when an attached image can't be seen |
| `test_triage_rules.py`, `test_triage_classify.py` | Triage rules and near-misses; threshold, follow-up joining, fallback, kill switch, reasons, metrics, logs, tie-breaker (fake model) |
| `test_triage_endpoint.py`, `test_triage_config.py` | `POST /api/triage` contract, auth, size limits, rate limit; `WEBAGENT_TRIAGE_*` overrides |
| `test_triage_collect.py` | Collected questions: off unless enabled, written only under `collected/`, 30-day cleanup |
| `test_task_previous_messages.py`, `test_task_shadow.py`, `test_task_modes.py` | `/api/task` history, shadow mode, overrides, and the `off` / `auto` / `always` modes on both task endpoints |
| `test_train_triage.py` | Training script, threshold choice, contrastive pairs for fine-tuning, release-gate exit codes, `--check-data` |
| `test_triage_real_model.py` | The committed classifier with its fine-tuned encoder from `models/triage/encoder/`, including a settled-history question that must stay direct (marker `real_triage_model`; skipped if the encoder isn't on disk, e.g. LFS weights not pulled) |
| `unit/`, `integration/`, `diagnostics/` | Routing, security, API and diagnostics tests |

`backend/tests/conftest.py`:

- makes both `app.*` and `backend.*` imports work wherever pytest runs from
- points saved task results at a temporary folder, so tests never write to `backend/data/results/`
- points the model registry at a temporary file, so tests never modify tracked files
- turns LangSmith tracing off before the app loads (`.env.local` holds a real key)
- blocks every network connection except localhost; a test that tries one fails with
  "Test tried to open a network connection". Fake the service instead. To debug with the
  network on, set `WEBAGENT_TESTS_ALLOW_NETWORK=true`
- sets Hugging Face to offline mode and turns off NLTK auto-download
- runs triage in `act` mode with collection off and no classifier (model decisions fall back to agentic), and points `DATA_DIR` at a temporary folder, so tests never write collected questions. Tests set a fake model (`tests/fakes/triage.py`) or shadow mode themselves

### Faking the model

Replace the agent's model call with a fixed reply instead of calling a provider:

```python
async def fake_generate(prompt):
    return '{"code": "print(1)", "explanation": "demo"}'

monkeypatch.setattr(agent, "_generate_with_retry", fake_generate)
```

For whole workflows, patch the agent factories in `app.graph.workflows` (see `test_workflow_state_flow.py`).

## Best Practices

1. **Test behavior, not implementation**: check what the user or the next agent sees.
2. **One behavior per test**, with a name that says what should happen.
3. **Fake external services**: LLMs, Tavily, the backend, Together. Never depend on the network.
4. **Cover failure paths**: errors, timeouts, fallbacks and malformed model output.
5. **For a bug fix, write a test that fails without the fix** before relying on it.

## Troubleshooting

1. **Test times out**: look for an unresolved promise, or stub delays (`(client as any).delay = jest.fn().mockResolvedValue(undefined)`).
2. **`window is not defined` / `Request is not defined`**: wrong test environment; API route tests need `@jest-environment node`.
3. **Context errors**: wrap components in `AppProvider` and `ChatProvider`.
4. **Module import errors**: check `transformIgnorePatterns` and `moduleNameMapper` in `jest.config.js`.
5. **Backend `ModuleNotFoundError: backend`**: run pytest from the repo root, or check that `backend/tests/conftest.py` exists.

### ESM Module Issues

Some libraries like `react-markdown` ship ESM only. `jest.config.js` lists them in `transformIgnorePatterns` so Babel transforms them.

## Further Reading

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro)
- [pytest Documentation](https://docs.pytest.org/)
