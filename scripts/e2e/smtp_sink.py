#!/usr/bin/env python3
"""Dev and CI only: an SMTP server with STARTTLS that saves each message for the end-to-end tests.

    python scripts/e2e/smtp_sink.py --port 2525 --out <dir> --cert cert.pem --key key.pem

Each message becomes <dir>/<n>.json: {"headers": {name: raw value, ...}, "body": decoded text}.
Headers are kept raw (not decoded), so a test sees exactly what a mail client would. It refuses
MAIL before STARTTLS, like a real submission server. Never deployed; standard library only.
"""
import argparse
import email
import email.policy
import itertools
import json
import pathlib
import socketserver
import ssl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=2525)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--cert", required=True)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(args.cert, args.key)
    counter = itertools.count(1)

    def save(raw: bytes) -> None:
        msg = email.message_from_bytes(raw, policy=email.policy.compat32)
        headers = {}
        for name, value in msg.items():
            headers[name] = value
        payload = msg.get_payload(decode=True) or b""
        record = {"headers": headers, "body": payload.decode(msg.get_content_charset() or "utf-8", "replace")}
        (args.out / f"{next(counter)}.json").write_text(json.dumps(record, indent=2))

    class Handler(socketserver.StreamRequestHandler):
        def send(self, line: str) -> None:
            self.wfile.write((line + "\r\n").encode())
            self.wfile.flush()

        def handle(self) -> None:
            self.send("220 e2e sink")
            tls = False
            while True:
                raw = self.rfile.readline()
                if not raw:
                    return
                verb = raw.decode(errors="replace").strip().split(" ")[0].upper()
                if verb in ("EHLO", "HELO"):
                    self.wfile.write(b"250-e2e\r\n" + (b"" if tls else b"250-STARTTLS\r\n") + b"250 OK\r\n")
                    self.wfile.flush()
                elif verb == "STARTTLS":
                    self.send("220 go ahead")
                    conn = context.wrap_socket(self.request, server_side=True)
                    self.request, self.rfile, self.wfile, tls = conn, conn.makefile("rb"), conn.makefile("wb"), True
                elif verb in ("MAIL", "RCPT", "RSET", "NOOP"):
                    self.send("250 OK" if tls else "530 STARTTLS first")
                elif verb == "DATA":
                    self.send("354 end with .")
                    lines = []
                    while True:
                        line = self.rfile.readline()
                        if line in (b".\r\n", b".\n", b""):
                            break
                        lines.append(line[1:] if line.startswith(b"..") else line)
                    save(b"".join(lines))
                    self.send("250 queued")
                elif verb == "QUIT":
                    self.send("221 bye")
                    return
                else:
                    self.send("502 not implemented")

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    Server((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
