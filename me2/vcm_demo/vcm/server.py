"""Push demo events to a browser.

The demo runs headless on the Pi, so the interface is a page served over the
network rather than anything on a local display. Events go out as Server-Sent
Events: one-way server to browser, which is all this needs, and unlike
WebSockets it is plain HTTP with no handshake or frame encoding. That keeps the
runtime's one real virtue intact - `demo.py` still imports nothing beyond
numpy, onnxruntime and sounddevice.

The audio callback thread must never block, so `emit` only appends to per
client queues and returns; a slow or vanished browser cannot stall recognition.
"""
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# A client that has not been drained for this many events is assumed gone: the
# browser tab was closed, or the network dropped. Its queue is discarded rather
# than grown without limit.
MAX_PENDING = 256


class EventServer:
    """Serve the demo page and stream events to whoever is watching."""

    def __init__(self, page: Path | None = None):
        """
        Args:
            page: HTML file to serve at /, defaults to ui.html beside demo.py
        """
        self.page = page or HERE / "ui.html"
        self.clients: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.latest: dict | None = None
        self.httpd: ThreadingHTTPServer | None = None

    def emit(self, event: dict) -> None:
        """
        Queue one event for every connected browser.

        Called from the audio loop, so it does no IO and never blocks.

        Args:
            event: JSON-serialisable payload, carrying at least a "type"
        """
        if event.get("type") == "result":
            self.latest = event

        with self.lock:
            for client in list(self.clients):
                if client.qsize() > MAX_PENDING:
                    self.clients.remove(client)
                    continue
                client.put_nowait(event)

    def _subscribe(self) -> queue.Queue:
        client: queue.Queue = queue.Queue()
        with self.lock:
            self.clients.append(client)
        return client

    def _unsubscribe(self, client: queue.Queue) -> None:
        with self.lock:
            if client in self.clients:
                self.clients.remove(client)

    def start(self, port: int = 8000, host: str = "0.0.0.0") -> str:
        """
        Start serving in a background thread.

        Args:
            port: TCP port to listen on (default: 8000)
            host: Interface to bind, default all so a phone on the same
                network can watch the Pi

        Returns:
            The URL to open
        """
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                """Silence the per-request logging; the demo owns the console."""

            def do_GET(self):
                if self.path.startswith("/events"):
                    return self._stream()
                if self.path in ("/", "/index.html"):
                    return self._page()
                self.send_error(404)

            def _page(self):
                body = server.page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                client = server._subscribe()
                try:
                    if server.latest:
                        self._send(server.latest)
                    while True:
                        try:
                            event = client.get(timeout=10)
                        except queue.Empty:
                            # A comment line keeps proxies and phones from
                            # closing an idle connection
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue
                        self._send(event)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    server._unsubscribe(client)

            def _send(self, event: dict):
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

        return f"http://localhost:{port}"

    def stop(self) -> None:
        """Shut the server down."""
        if self.httpd is not None:
            self.httpd.shutdown()
