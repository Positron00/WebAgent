#!/usr/bin/env python3
"""Fake Together completions server: dev only, for the 3.6.0 local drift and alerts check (OBS-020).

Never imported by the app and never deployed. It answers like Together's completions API so the
backend's Together client (pointed here through TOGETHER_API_BASE, OBS-021) can be made slower,
failing or rate-limited on purpose, without spending real credit:

    python scripts/fake_together.py --port 18200 --delay 0.5 --completion-tokens 80 \
        --error-rate 0.0 --rate-limit-rate 0.0

    # in the backend container's environment:
    TOGETHER_API_BASE=http://host.docker.internal:18200/v1

Routes: POST /v1/completions (also /completions), GET /health. Standard library only.
"""
import argparse
import json
import random
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Dev-only fake Together completions server (OBS-020).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Address to listen on (0.0.0.0 to reach it from Docker on Linux)")
    parser.add_argument("--port", type=int, default=18200)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to wait before each answer")
    parser.add_argument("--completion-tokens", type=int, default=80, help="completion_tokens reported in usage")
    parser.add_argument("--error-rate", type=float, default=0.0, help="Share of requests answered with HTTP 500")
    parser.add_argument("--rate-limit-rate", type=float, default=0.0, help="Share of requests answered with HTTP 429")
    args = parser.parse_args(argv)
    for name in ("error_rate", "rate_limit_rate"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.delay < 0 or args.completion_tokens < 0:
        parser.error("--delay and --completion-tokens must not be negative")
    return args


def make_handler(args):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FakeTogether/1.0"

        def _send(self, status, body, headers=None):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": {"message": "not found", "type": "not_found"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/completions", "/completions"):
                self._send(404, {"error": {"message": "not found", "type": "not_found"}})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._send(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
                return
            time.sleep(args.delay)
            roll = random.random()
            if roll < args.error_rate:
                self._send(500, {"error": {"message": "fake internal error", "type": "server_error"}})
                return
            if roll < args.error_rate + args.rate_limit_rate:
                self._send(429, {"error": {"message": "fake rate limit", "type": "rate_limit"}},
                           headers={"Retry-After": "1"})
                return
            prompt = str(request.get("prompt") or "")
            prompt_tokens = max(1, len(prompt.split()))
            completion_tokens = args.completion_tokens
            self._send(200, {
                "id": f"fake-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.get("model", "fake-model"),
                "choices": [{"index": 0, "text": "This is a fake completion.", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens},
            })

        def log_message(self, fmt, *log_args):
            sys.stderr.write("fake_together: " + (fmt % log_args) + "\n")

    return Handler


def main(argv=None):
    args = parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args))
    server.daemon_threads = True
    print(f"fake_together: dev only, listening on http://{args.host}:{args.port}/v1 "
          f"(delay {args.delay}s, error rate {args.error_rate}, rate-limit rate {args.rate_limit_rate})",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
