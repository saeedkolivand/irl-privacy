"""Upload page for getting glasses footage onto the PC from the phone.

Open it in Safari over Tailscale and pick clips from the Photos library. Files land in
test-footage/real/. Like the panic endpoint it has no auth of its own: it binds to the Tailscale
address, so being able to reach it already means being on the tailnet.

    python upload.py                 # serve on the Tailscale IP
    python upload.py --selftest
"""
import argparse
import ipaddress
import re
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

DEST = Path(__file__).parent / "test-footage" / "real"
CHUNK = 1 << 20     # stream to disk a megabyte at a time; a 3 min 3K clip will not fit in RAM twice

PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>irl-privacy upload</title>
<style>
 body{font:16px system-ui;margin:0;padding:24px;background:#111;color:#eee}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 p{color:#999;margin:0 0 20px;font-size:14px}
 label{display:block;padding:28px;border:2px dashed #444;border-radius:12px;text-align:center;
       color:#bbd;background:#181818;cursor:pointer}
 input{display:none}
 li{list-style:none;padding:8px 0;border-bottom:1px solid #222;font-size:14px;
     display:flex;justify-content:space-between;gap:12px}
 .ok{color:#6c6}.err{color:#d66}.pending{color:#888}
 ul{padding:0;margin:20px 0 0}
</style>
<h1>irl-privacy</h1>
<p>Clips land in test-footage/real/ on the PC.</p>
<label for=f>Tap to choose videos</label>
<input id=f type=file accept="video/*" multiple>
<ul id=log></ul>
<script>
const log = document.getElementById('log');
document.getElementById('f').onchange = async e => {
  for (const file of e.target.files) {
    const li = document.createElement('li');
    li.innerHTML = `<span>${file.name}</span><span class=pending>sending…</span>`;
    log.prepend(li);
    const status = li.lastElementChild;
    try {
      const r = await fetch('/put/' + encodeURIComponent(file.name), {method: 'PUT', body: file});
      const t = await r.text();
      status.textContent = r.ok ? t.trim() : 'failed';
      status.className = r.ok ? 'ok' : 'err';
    } catch (err) { status.textContent = 'failed'; status.className = 'err'; }
  }
  e.target.value = '';
};
</script>
"""


def safe_name(raw):
    """Keep the basename only, and only characters that cannot escape the folder.

    Split on both slash directions by hand rather than Path(...).name: pathlib only treats
    backslash as a separator on Windows, so a backslash-laden name that should be stripped down
    to its basename passed through whole on Linux -- caught by CI running this on Linux while
    the server itself only ever runs on Windows."""
    base = unquote(raw).replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    return name[:120] or "clip.mp4"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/plain; charset=utf-8"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        self._send(404, "not here\n")

    def do_PUT(self):
        if not self.path.startswith("/put/"):
            return self._send(404, "not here\n")
        DEST.mkdir(parents=True, exist_ok=True)
        dest = DEST / safe_name(self.path[len("/put/"):])
        left = int(self.headers.get("Content-Length") or 0)
        if left <= 0:
            return self._send(400, "empty\n")
        got = 0
        with open(dest, "wb") as f:
            while got < left:
                chunk = self.rfile.read(min(CHUNK, left - got))
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
        if got < left:                      # truncated: better no file than half a file
            dest.unlink(missing_ok=True)
            return self._send(400, "incomplete\n")
        print(f"received {dest.name} ({got / 1e6:.1f} MB)", flush=True)
        self._send(200, f"{got / 1e6:.1f} MB\n")

    def log_message(self, *a):
        pass


def selftest():
    assert safe_name("clip.mp4") == "clip.mp4"
    assert safe_name("../../etc/passwd") == "passwd", "path traversal is stripped"
    assert safe_name("C:\\Users\\x\\IMG 0042.MOV") == "IMG_0042.MOV"
    assert safe_name("%2e%2e%2fsecret.mp4") == "secret.mp4", "url-encoded traversal too"
    assert safe_name("") == "clip.mp4" and safe_name("...") == "clip.mp4"
    assert "/" not in safe_name("a/b/c.mp4") and "\\" not in safe_name("a\\b.mp4")
    assert (ip := tailscale_ip()) is None or ip.startswith("100.")
    print("selftest ok")


def tailscale_ip():
    """This PC's tailnet address, so the phone-facing URLs never have to be typed by hand.

    Asks the routing table rather than the host name, which on Linux answers 127.0.1.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("100.100.100.100", 80))      # Tailscale's resolver: routed only when up
            ip = s.getsockname()[0]                 # a UDP connect() picks a route, sends nothing
        except OSError:
            return None
    return ip if ipaddress.ip_address(ip) in ipaddress.ip_network("100.64.0.0/10") else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=tailscale_ip(),
                   help="Tailscale IP of this PC (default: auto-detected)")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not a.host:
        p.error("no Tailscale IP found -- is Tailscale up? Otherwise pass --host 100.x.x.x")
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"upload page: http://{a.host}:{a.port}/\nsaving to {DEST}")
    try:
        ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
    except OSError as e:
        sys.exit(f"cannot bind {a.host}:{a.port} -- is Tailscale up? ({e})")


if __name__ == "__main__":
    main()
