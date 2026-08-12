# llm_runtimes integration notes

Additive routing layer that lets CodeScientist use two extra model backends
besides its usual provider APIs, via the vendored `llm_runtimes/` package
(an OpenAI-compatible local server):

- `claudecli-sonnet` / `claudecli-opus` / `claudecli-haiku` — Claude via the local `claude -p` CLI (subscription auth, no API key)
- `local-qwen` — a local model served by an in-process vLLM engine

Model names with these prefixes are translated at the litellm call sites to
`model="openai/<name>"` with `api_base` pointing at the local llm_runtimes
server and `api_key="llm-runtimes"`. All existing model names behave exactly
as before.

## Files changed

- `src/ExtractionUtils.py` — the central litellm call site for all of `src/`
  (every module goes through `getLLMResponseJSON` here). Added a thin
  `completion` wrapper around `litellm.completion` that translates
  `claudecli-*` / `local-*` model names and points them at
  `llm_runtimes.ensure_server()` (which starts, or discovers, the local server).
- `llm-proxy/llm-proxy-server.py` — the proxy used by generated experiments.
  Added `translate_llm_runtimes_request()`, applied where requests are
  forwarded to `litellm.completion`. Uses the `LLM_RUNTIMES_PORT` environment
  variable (default 8399) to build the api_base, since `llm_runtimes` may not
  be importable from the proxy's runtime environment.
- `llm-proxy/experiment-llm-cost.json` — zero-cost entries for the four new
  model names so the proxy's budget accounting does not apply its default
  ($20/1M) fallback rate to local models.
- `src/CodeScientistWebInterface.py` — the four new model names appended to
  every model-selection dropdown (`model_str` options lists).
- `codeblocks/llm_submit_proxy.py` — documentation comment listing the new
  model names so generated experiment code knows they exist.
- `RUNTIMES_NOTES.md` — this file.

`llm_runtimes/` itself is vendored as-is and was not modified.

## Usage per backend

### Usual API models (unchanged)

Nothing changes. Configure API keys as before (`api_keys.donotcommit.json`,
environment variables) and select any of the existing model names.

### claudecli-* (Claude via local CLI)

Requires the `claude` CLI installed and authenticated on the machine running
CodeScientist. For the `src/` pipeline no extra step is needed: the wrapper
calls `ensure_server()`, which lazily starts the server in-process (or reuses
one already bound to the port). To run the server standalone instead:

```
python -m llm_runtimes.server --port 8399
```

Then select `claudecli-sonnet`, `claudecli-opus`, or `claudecli-haiku` as the
model in the web interface, or pass it as `model` / `model_str` anywhere a
model name is accepted.

For the experiment llm-proxy path, the proxy expects the server to be
reachable at `http://127.0.0.1:${LLM_RUNTIMES_PORT:-8399}/v1`, so start the
standalone server first (command above) and, if using a non-default port,
export `LLM_RUNTIMES_PORT` before launching the proxy.

### local-* (local vLLM model)

Same routing as claudecli-*, but requires vLLM and a GPU. `local-qwen` maps
to the alias configured in `llm_runtimes/config.py`
(default `Qwen/Qwen3-8B-AWQ`; override with `LLM_RUNTIMES_LOCAL_HF_ID` etc.).
Optionally preload the model at server startup:

```
python -m llm_runtimes.server --port 8399 --preload-local
```

## Limitations

- **claudecli ignores temperature** — the `claude -p` CLI does not expose
  sampling controls, so `temperature` (and similar parameters) are accepted
  but have no effect for `claudecli-*` models.
- **Usage/cost accounting is zeroed** — the runtime server reports zeroed
  token usage for claudecli models and the cost table entries are 0.0, so
  running-cost totals and per-experiment budgets do not reflect these calls
  (they cost no API money, but token counts are also unavailable).
- **Modal-based execution untested** — CodeScientist executes generated
  experiments in Modal (cloud). That path needs Modal credentials and has not
  been exercised; additionally, a Modal container cannot reach a
  `127.0.0.1` llm_runtimes server on the host, so `claudecli-*` / `local-*`
  models inside Modal-run experiments would require a tunnel or a reachable
  server address. The routing here is verified for the local litellm call
  sites and the proxy translation logic only.
- **Only single-tool forced function calls emulated** — the runtime server's
  tool/function-calling emulation covers the "one tool, forced call" pattern;
  multi-tool selection or parallel tool calls are not supported.
- JSON response formatting relies on prompting rather than a strict
  `response_format` schema for the new backends (CodeScientist already
  prompts for fenced JSON, and its Claude code path skips
  `response_format`, so this matches existing behavior).
