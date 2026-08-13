"""Minimal OpenAI-compatible server exposing the claudecli-* and local-* models.

Runs lazily in a daemon thread inside whichever scaffold process first needs it;
later processes on the same machine detect the bound port and reuse it. Also
runnable standalone: python -m llm_runtimes.server --port 8399
"""

import json
import socket
import threading
import time
import uuid

from . import config


# --------------------------------------------------------------------------
# request handling (framework-free: stdlib http.server keeps deps tiny)
# --------------------------------------------------------------------------
def _handle_chat(body):
    model = body.get("model", "")
    messages = body.get("messages", [])
    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
    n = int(body.get("n", 1))
    stop = body.get("stop")
    response_format = body.get("response_format")
    tools = body.get("tools") or (
        [{"type": "function", "function": f} for f in body.get("functions", [])]
        if body.get("functions") else None
    )

    # Tool-forcing: flatten tool schema into the prompt, demand JSON back.
    forced_tool = None
    if tools:
        forced_tool = tools[0]["function"]
        schema = json.dumps(forced_tool.get("parameters", {}), indent=2)
        messages = list(messages) + [{
            "role": "user",
            "content": (
                f"Respond ONLY with a valid JSON object matching this schema "
                f"(no markdown fences, no commentary):\n{schema}"
            ),
        }]
        response_format = {"type": "json_object"}

    if model.startswith("claudecli-"):
        from . import claude_cli
        cm = config.resolve_claude_model(model)
        texts = [
            claude_cli.complete(messages, model=cm, response_format=response_format)
            for _ in range(n)
        ]
    elif model.startswith("local-"):
        from .local_vllm import LocalVLLM
        eng = LocalVLLM.instance(config.resolve_local(model))
        texts = eng.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            lora=body.get("lora"),
        )
        # Reasoning models (e.g. Qwen3) emit <think> blocks; scaffolds expect
        # clean assistant text.
        texts = [
            t.split("</think>", 1)[1].lstrip() if "<think>" in t and "</think>" in t
            else t
            for t in texts
        ]
    else:
        return 404, {"error": {"message": f"unknown runtime model: {model}"}}

    choices = []
    for i, t in enumerate(texts):
        if forced_tool:
            cleaned = t.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                cleaned = cleaned[cleaned.find("{"):cleaned.rfind("}") + 1]
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": forced_tool["name"], "arguments": cleaned},
                }],
            }
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": t}
            finish = "stop"
        choices.append({"index": i, "message": msg, "finish_reason": finish})

    return 200, {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
    }


def _sse_stream(payload):
    """Fake streaming: emit the finished completion as a short SSE stream."""
    base = {
        "id": payload["id"],
        "object": "chat.completion.chunk",
        "created": payload["created"],
        "model": payload["model"],
    }
    chunks = []
    for ch in payload["choices"]:
        delta_role = dict(base, choices=[{"index": ch["index"],
                                          "delta": {"role": "assistant"},
                                          "finish_reason": None}])
        content = ch["message"].get("content") or ""
        delta_body = dict(base, choices=[{"index": ch["index"],
                                          "delta": {"content": content},
                                          "finish_reason": None}])
        delta_end = dict(base, choices=[{"index": ch["index"], "delta": {},
                                         "finish_reason": ch["finish_reason"]}])
        chunks += [delta_role, delta_body, delta_end]
    out = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return out + "data: [DONE]\n\n"


# --------------------------------------------------------------------------
# stdlib HTTP server
# --------------------------------------------------------------------------
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence request logging
        pass

    def _send(self, code, obj, raw=None, content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        body = raw.encode() if raw is not None else json.dumps(obj).encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            models = ["claudecli-sonnet", "claudecli-opus", "claudecli-haiku"] + [
                f"local-{a}" for a in config.LOCAL_MODELS
            ]
            self._send(200, {"object": "list", "data": [
                {"id": m, "object": "model", "owned_by": "llm_runtimes"} for m in models
            ]})
        elif self.path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})

        if self.path.startswith("/v1/chat/completions"):
            try:
                code, payload = _handle_chat(body)
            except Exception as e:  # surface backend errors as API errors
                return self._send(500, {"error": {"message": str(e)[:4000]}})
            if code == 200 and body.get("stream"):
                return self._send(200, None, raw=_sse_stream(payload),
                                  content_type="text/event-stream")
            return self._send(code, payload)

        if self.path == "/admin/load_adapter":
            from .local_vllm import LocalVLLM
            info = LocalVLLM.instance().load_adapter(body["name"], body["path"])
            return self._send(200, info)

        self._send(404, {"error": "not found"})


def _port_open(port):
    with socket.socket() as s:
        return s.connect_ex((config.DEFAULT_HOST, port)) == 0


_server = None
_server_lock = threading.Lock()


def ensure_server(port=None):
    """Start (or discover) the runtime server; return its base_url."""
    global _server
    port = port or config.DEFAULT_PORT
    url = f"http://{config.DEFAULT_HOST}:{port}/v1"
    with _server_lock:
        if _server is not None or _port_open(port):
            return url
        srv = ThreadingHTTPServer((config.DEFAULT_HOST, port), _Handler)
        t = threading.Thread(target=srv.serve_forever, daemon=True,
                             name="llm_runtimes_server")
        t.start()
        _server = srv
    return url


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    ap.add_argument("--preload-local", action="store_true",
                    help="load the local vLLM model at startup")
    args = ap.parse_args()
    if args.preload_local:
        from .local_vllm import LocalVLLM
        LocalVLLM.instance()
    ensure_server(args.port)
    print(f"llm_runtimes server on http://{config.DEFAULT_HOST}:{args.port}/v1")
    threading.Event().wait()
