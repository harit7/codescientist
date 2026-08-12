"""Claude backend via the local `claude -p` CLI (subscription auth, no API key).

Each chat-completion request becomes one headless claude invocation with tools
disabled, so it behaves like a plain text-in/text-out model endpoint.
"""

import json
import subprocess

from . import config

_NO_TOOLS_SYSTEM = (
    "You are being used as a raw text-generation API endpoint by an automated "
    "pipeline. Respond with the answer content only: no preamble, no tool use, "
    "no questions back. Follow the caller's formatting instructions exactly."
)


def _flatten(messages):
    """Turn an OpenAI-style message list into (system_text, prompt_text)."""
    system_parts, convo = [], []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):  # multimodal: keep text parts only
            content = "\n".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            convo.append(f"<assistant_reply>\n{content}\n</assistant_reply>")
        else:
            convo.append(content)
    return "\n\n".join(system_parts), "\n\n".join(convo)


def complete(messages, model="sonnet", response_format=None, timeout=None):
    """Run one claude -p call. Returns the assistant text."""
    system_text, prompt = _flatten(messages)
    sys_prompt = _NO_TOOLS_SYSTEM
    if system_text:
        sys_prompt += "\n\n" + system_text
    if response_format and response_format.get("type") == "json_object":
        sys_prompt += "\n\nRespond with a single valid JSON object and nothing else."

    cmd = [
        config.CLAUDE_BIN,
        "-p",
        "--model", model,
        "--output-format", "json",
        "--append-system-prompt", sys_prompt,
        "--disallowedTools", "*",
    ]
    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout or config.CLAUDE_TIMEOUT,
        cwd=config.CLAUDE_CWD,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude -p failed (rc={proc.returncode}): {proc.stderr[-2000:]}"
        )
    payload = _parse_envelope(proc.stdout)
    if payload is None:
        # Non-JSON output (banner-only, etc.): fall back to raw stdout.
        return proc.stdout.strip()
    if payload.get("is_error"):
        raise RuntimeError(f"claude -p error result: {str(payload)[:2000]}")
    text = (payload.get("result") or "").strip()
    if response_format and response_format.get("type") == "json_object":
        text = _strip_fences(text)
    return text


def _parse_envelope(stdout):
    """Parse the CLI's JSON envelope, tolerating banner lines around it."""
    for candidate in (stdout, stdout[stdout.find("{"):] if "{" in stdout else ""):
        if not candidate:
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(candidate)
            if isinstance(obj, dict) and ("result" in obj or "is_error" in obj):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _strip_fences(text):
    """Remove markdown code fences around a JSON payload, if present."""
    t = text.strip()
    if t.startswith("```"):
        first_brace = t.find("{")
        last_brace = t.rfind("}")
        if 0 <= first_brace < last_brace:
            return t[first_brace:last_brace + 1]
    return t
