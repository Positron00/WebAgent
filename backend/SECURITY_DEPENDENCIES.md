# Dependency advisories

CI audits the exact versions in `backend/constraints.txt` (`pip-audit`) and the web app's production
dependencies (`npm audit --omit=dev --audit-level=high`). A new advisory fails the build. The
advisories below stay open because the fix needs the package's next major version, which needs its
own migration. None of them is reachable in WebAgent, for the reason given. Review this list when
any of these packages moves to a new major version, and remove an entry once its fix ships.

Last reviewed: 2026-10-17 (release 3.10.0).

| Package (version) | Advisory | Fixed in | Why WebAgent is not affected |
|---|---|---|---|
| chromadb 0.6.3 | PYSEC-2026-3813, -3814, -3815 (tenant authorisation, code injection through a model repository with `trust_remote_code`) | none yet | Chroma runs embedded in the backend process (`langchain_chroma.Chroma` with a local directory). No Chroma server, tenants, RBAC provider or remote embedding repository is used. |
| langchain 0.3.30, langchain-anthropic 0.3.22 | PYSEC-2026-2192, PYSEC-2026-2556 (path confinement in file-search middleware and glob helpers) | 1.3.9, 1.4.6 | Those components aren't used; documents are read by our own loaders from the upload folder. |
| langchain-core 0.3.86 | PYSEC-2026-2193 (`prompts.loading` reads paths from deserialised configs) | 1.2.22 | Prompts are built in code; `load_prompt` and prompt files from configs are never used. |
| langchain-core 0.3.86, langchain-openai 0.3.35 | PYSEC-2026-2562, PYSEC-2026-76 (SSRF in image token counting) | 1.2.11, 1.1.14 | `get_num_tokens_from_messages` is never called and no message carries an `image_url`. |
| langchain-text-splitters 0.3.11 | PYSEC-2026-77 (`HTMLHeaderTextSplitter.split_text_from_url` follows redirects) | 1.1.2 | `split_text_from_url` is never called; web pages are fetched by Tavily, not by the splitter. |
| langgraph 0.2.76, langgraph-checkpoint 2.1.2 | PYSEC-2026-83, PYSEC-2026-1527, -2573, -2574 (checkpoint and cache deserialisation) | 1.0.10, 4.1.1 | No checkpointer, store or `CachePolicy` is configured; graphs run in memory per task. |
| langgraph-sdk 0.1.74 (and langgraph via it) | PYSEC-2026-2194, -2575, -4240 (SDK URL paths, auth decorators) | 0.4.4 | The SDK client for LangGraph API servers isn't used; there is no LangGraph server. |
| nltk 3.10.3 | PYSEC-2026-3740 (model-artifact paths outside the data root) | none yet | NLTK data is installed at build time; no caller-supplied model path reaches NLTK. |
| sentence-transformers 3.4.1 | PYSEC-2026-4164 (`import_module_class` bypass with a remote model repository) | 5.6.0 | Every encoder loads from a local folder or the build-time cache (`local_files_only`), never from a repository named at runtime. |
| transformers 4.57.6 | PYSEC-2025-217, PYSEC-2026-2288, -2289, -2290, -3929, -4174 (checkpoint conversion, `Trainer` RNG state, crafted `config.json`, custom generation code, `save_pretrained` paths) | 5.3.0 – 5.10.0 | The runtime loads only WebAgent's own committed models and the build-time MiniLM cache with `local_files_only=True` and no `trust_remote_code`; it never trains, converts or saves models. The training scripts run offline on the developer's machine. |

## Web app

`npm audit --omit=dev` reports no advisories on Next.js 15.5.28 and React 19. One override is in
`package.json`: Next.js bundles postcss 8.4.31, which has advisories (CSS stringify XSS, source-map
path reads). postcss only processes the app's own stylesheets at build time, but the override
(`"overrides": {"next": {"postcss": "^8.5.23"}}`) moves Next's copy to a fixed 8.5 release so the
audit stays clean. Remove it when Next.js ships a fixed postcss itself.
