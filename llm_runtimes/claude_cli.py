"""Claude backend via the local `claude -p` CLI (subscription auth, no API key).

Each chat-completion request becomes one headless claude invocation. Tools are
disabled for pure text requests. Requests containing OpenAI-style image parts
(data: URIs) are supported by writing the images to a temporary directory and
allowing claude only the Read tool, giving true vision capability.
"""

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile

from . import config

_NO_TOOLS_SYSTEM = (
    "You are being used as a raw text-generation API endpoint by an automated "
    "pipeline. Respond with the answer content only: no preamble, no questions "
    "back. Follow the caller's formatting instructions exactly."
)

_DATA_URI_RE = re.compile(r"^data:image/(?P<ext>[a-zA-Z0-9.+-]+);base64,(?P<b64>.*)$", re.S)


def _flatten(messages, image_dir):
    """Turn an OpenAI-style message list into (system_text, prompt_text,
    image_paths). Image parts are written into image_dir."""
    system_parts, convo, image_paths = [], [], []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            texts = []
            for part in content:
                ptype = part.get("type")
                if ptype == "text":
                    texts.append(part.get("text", ""))
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    match = _DATA_URI_RE.match(url)
                    if match:
                        ext = match.group("ext").split("+")[0]
                        path = os.path.join(
                            image_dir, f"image_{len(image_paths) + 1}.{ext}"
                        )
                        with open(path, "wb") as fh:
                            fh.write(base64.b64decode(match.group("b64")))
                        image_paths.append(path)
                        texts.append(f"[attached image file: {path}]")
            content = "\n".join(texts)
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            convo.append(f"<assistant_reply>\n{content}\n</assistant_reply>")
        else:
            convo.append(content)
    return "\n\n".join(system_parts), "\n\n".join(convo), image_paths


def complete(messages, model="sonnet", response_format=None, timeout=None):
    """Run one claude -p call. Returns the assistant text."""
    image_dir = tempfile.mkdtemp(prefix="llm_runtimes_img_")
    try:
        system_text, prompt, image_paths = _flatten(messages, image_dir)
        if not prompt.strip():
            # claude -p requires a non-empty prompt; some scaffolds send
            # system-message-only requests.
            prompt = "Follow the system instructions and produce the requested output."
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
        ]
        if image_paths:
            prompt += (
                "\n\nThe attached images listed above are image files on disk. "
                "Use the Read tool to view each of them before answering."
            )
            cmd += ["--allowedTools", "Read", "--add-dir", image_dir]
        else:
            cmd += ["--disallowedTools", "*"]

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
    finally:
        shutil.rmtree(image_dir, ignore_errors=True)


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
