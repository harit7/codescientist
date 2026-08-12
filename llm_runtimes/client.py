"""Convenience client helpers for host repos."""

from .server import ensure_server


def openai_client(**kwargs):
    """An openai.OpenAI client pointed at the in-process runtime server.

    Host repos can hand this to their existing OpenAI code path unchanged.
    """
    import openai

    return openai.OpenAI(base_url=ensure_server(), api_key="llm-runtimes", **kwargs)


def chat(model, messages, **kwargs):
    """One-shot helper: chat('claudecli-sonnet', [...]) -> str"""
    client = openai_client()
    resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return resp.choices[0].message.content
