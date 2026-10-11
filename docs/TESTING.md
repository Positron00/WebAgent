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

Set `LANGSMITH_TRACING_ENABLED=false` when running backend tests if you don't want them traced to LangSmith.

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
│   │   ├── chatRoute.test.ts      # /api/chat: validation, Together errors, fallback notice
│   │   └── taskProxy.test.ts      # /api/task proxy: forwarding, X-API-Key, backend down
│   ├── contexts/
│   │   └── ChatContext.test.tsx   # question stays visible after the answer or an error
│   ├── utils/
│   │   ├── apiClient.test.ts
│   │   └── apiClient.agentic.test.ts  # agentic request, polling, fallback
│   └── Chat.test.tsx
└── components/chat/__tests__/     # MessageInput, MessageList
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

`jest.config.js` sets `coverageThreshold` just below current coverage, so `npm run test:coverage` fails if coverage drops. Raise the numbers when you add tests.

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
| `unit/`, `integration/`, `diagnostics/` | Routing, security, API and diagnostics tests |

`backend/tests/conftest.py`:

- makes both `app.*` and `backend.*` imports work wherever pytest runs from
- points saved task results at a temporary folder, so tests never write to `backend/data/results/`

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
