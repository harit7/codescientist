"""Configuration for llm_runtimes. Everything is env-overridable; no host-repo
config files are touched."""

import os

# Fixed default port so every scaffold on this machine can share one server.
DEFAULT_PORT = int(os.environ.get("LLM_RUNTIMES_PORT", "8399"))
DEFAULT_HOST = "127.0.0.1"

# claude -p defaults
CLAUDE_BIN = os.environ.get("LLM_RUNTIMES_CLAUDE_BIN", "claude")
CLAUDE_TIMEOUT = int(os.environ.get("LLM_RUNTIMES_CLAUDE_TIMEOUT", "900"))
# Neutral cwd so no project-level CLAUDE.md/skills leak into calls.
CLAUDE_CWD = os.environ.get("LLM_RUNTIMES_CLAUDE_CWD", "/tmp")

# Local model aliases: model string "local-<alias>" resolves here.
# Turing (SM 7.5) friendly defaults: fp16, AWQ quantization, modest context.
LOCAL_MODELS = {
    "qwen": {
        "hf_id": os.environ.get("LLM_RUNTIMES_LOCAL_HF_ID", "Qwen/Qwen3-8B-AWQ"),
        "quantization": os.environ.get("LLM_RUNTIMES_LOCAL_QUANT", "awq"),
        "dtype": "float16",
        "max_model_len": int(os.environ.get("LLM_RUNTIMES_LOCAL_MAXLEN", "16384")),
        "gpu_memory_utilization": float(
            os.environ.get("LLM_RUNTIMES_LOCAL_GPU_UTIL", "0.90")
        ),
    },
}
DEFAULT_LOCAL_ALIAS = "qwen"

# GPU pinning: which device the in-process engine uses (never touch busy GPUs).
LOCAL_GPU = os.environ.get("LLM_RUNTIMES_LOCAL_GPU", "0")


def is_runtime_model(model: str) -> bool:
    """True if this model string is handled by llm_runtimes rather than a
    provider API."""
    return isinstance(model, str) and (
        model.startswith("claudecli-") or model.startswith("local-")
    )


def resolve_local(model: str) -> dict:
    alias = model[len("local-"):] or DEFAULT_LOCAL_ALIAS
    if alias in LOCAL_MODELS:
        return LOCAL_MODELS[alias]
    # Unknown alias: treat it as a raw HF model id.
    base = dict(LOCAL_MODELS[DEFAULT_LOCAL_ALIAS])
    base["hf_id"] = alias
    base["quantization"] = None
    return base


def resolve_claude_model(model: str) -> str:
    """claudecli-sonnet -> sonnet ; claudecli -> sonnet (default)."""
    rest = model[len("claudecli-"):]
    return rest or "sonnet"
