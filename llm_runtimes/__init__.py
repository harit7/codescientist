"""llm_runtimes: additive runtime layer giving AI-scientist scaffolds two extra
model backends besides their usual API access:

  * ``claudecli-<model>``  -> Claude via the local `claude -p` CLI (subscription auth)
  * ``local-<alias>``      -> local model served by an in-process vLLM engine

Design: a minimal OpenAI-compatible HTTP server runs lazily inside the host
process (or is shared, if another process already bound the port). Host repos
keep their existing OpenAI client code path and just point base_url at
``ensure_server()``. Nothing in the host scaffolding changes.

Fine-tuning control: the vLLM engine lives in-process; use ``get_engine()`` or
the /admin/load_adapter endpoint to hot-swap LoRA checkpoints mid-run.
"""

from .config import DEFAULT_PORT, is_runtime_model
from .server import ensure_server
from .client import openai_client, chat

__all__ = [
    "ensure_server",
    "openai_client",
    "chat",
    "is_runtime_model",
    "DEFAULT_PORT",
]


def get_engine():
    """Return the in-process vLLM engine wrapper (starts it on first use)."""
    from .local_vllm import LocalVLLM

    return LocalVLLM.instance()
