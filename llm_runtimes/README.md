# llm_runtimes

Additive runtime layer that gives this scaffold two extra model backends beside
its usual API-based access. No upstream scaffolding is modified; the layer only
adds new model names.

| Model string | Backend | Auth |
|---|---|---|
| `claudecli-sonnet` / `claudecli-opus` / `claudecli-haiku` | local `claude -p` CLI | your Claude subscription (no API key) |
| `local-qwen` (or `local-<hf-model-id>`) | in-process vLLM engine | none (local GPU) |
| everything else | untouched upstream code path | provider API keys as before |

## How it works

A minimal OpenAI-compatible server (stdlib-only) starts lazily in a daemon
thread inside the scaffold's own process the first time a `claudecli-*` or
`local-*` model is used; other processes on the machine reuse the same port
(default 8399, `LLM_RUNTIMES_PORT`). The scaffold's existing OpenAI client code
path is pointed at it, so request/response semantics stay identical.

Standalone mode (e.g. to share one GPU engine across several scaffolds):

```bash
python -m llm_runtimes.server --port 8399 --preload-local
```

## Fine-tuning integration (local models)

The vLLM engine is in-process and reachable:

```python
import llm_runtimes
eng = llm_runtimes.get_engine()
eng.load_adapter("ckpt-500", "/path/to/lora/checkpoint")  # hot-swap LoRA
eng.sleep(); eng.wake()   # release/reclaim GPU memory around training steps
```

or over HTTP: `POST /admin/load_adapter {"name": "ckpt-500", "path": "..."}`.
Requests can pin an adapter with `"lora": "ckpt-500"` in the body.

## Env knobs

- `LLM_RUNTIMES_PORT` (8399), `LLM_RUNTIMES_LOCAL_GPU` (0)
- `LLM_RUNTIMES_LOCAL_HF_ID` (Qwen/Qwen3-8B-AWQ), `LLM_RUNTIMES_LOCAL_QUANT` (awq)
- `LLM_RUNTIMES_LOCAL_MAXLEN` (16384), `LLM_RUNTIMES_CLAUDE_TIMEOUT` (900)
- `LLM_RUNTIMES_CLAUDE_CWD` (/tmp; neutral cwd so no project config leaks into calls)

## Notes / limits

- `claude -p` ignores `temperature`; sampling is provider-controlled.
- Tool/function calling is emulated: the first tool's JSON schema is appended
  to the prompt and the JSON reply is wrapped as a `tool_calls` response.
- Token usage fields are zeroed (the CLI does not report comparable counts).
- Local default targets 11GB Turing GPUs: AWQ quant, fp16, 16k context.
