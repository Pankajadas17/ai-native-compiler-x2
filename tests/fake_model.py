"""An offline stand-in model for the agent harness tests: a tiny OpenAI-compatible /chat/completions server in a thread.

The harness talks to it exactly as it talks to Gemini (same loop, tools and end-of-turn hook), so the tests check the real
wiring without a key or a network. A script decides each reply from the conversation so far:

    def script(messages, request):            # -> ("tool", name, args) | ("text", content) | ("http", status, body)
        ...
    with FakeModel(script) as fm:
        run = harness.converse(harness.Gemini(model="m", base_url=fm.url), ...)
        fm.requests      # what the harness sent (messages, tools offered, model name)
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def calls_so_far(messages):
    """Names of the tool calls the model has made so far."""
    return [tc["function"]["name"] for m in messages if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])]


def texts_so_far(messages):
    """The model's plain-text replies so far."""
    return [m.get("content") for m in messages if m.get("role") == "assistant" and not m.get("tool_calls") and m.get("content")]


class FakeModel:
    def __init__(self, script):
        self.script, self.requests = script, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self._send(200, {"data": [{"id": "fake", "object": "model"}]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.requests.append(req)
                reply = outer.script(req.get("messages", []), req)
                if reply[0] == "http":
                    self._send(reply[1], reply[2])
                    return
                if reply[0] == "tool":
                    msg = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": f"call_{len(outer.requests)}", "type": "function",
                         "function": {"name": reply[1], "arguments": json.dumps(reply[2])}}]}
                    finish = "tool_calls"
                else:
                    msg, finish = {"role": "assistant", "content": reply[1]}, "stop"
                usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                if req.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    delta = dict(msg)
                    if "tool_calls" in delta:
                        delta["tool_calls"] = [dict(delta["tool_calls"][0], index=0)]
                    for c in ({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                              {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage}):
                        c.update({"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": req.get("model")})
                        self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self._send(200, {"id": "x", "object": "chat.completion", "created": int(time.time()), "model": req.get("model"),
                                     "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})

            def _send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def tools_offered(self):
        """Tool names the harness showed the model (union over all requests)."""
        return sorted({t.get("function", {}).get("name") for r in self.requests for t in r.get("tools", [])})
