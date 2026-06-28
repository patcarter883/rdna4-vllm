"""Tiny JSON-over-HTTP helper on urllib (stdlib only — no `requests` dependency, matching
gpu-lease's zero-dep ethos). Returns (status_code, parsed_body)."""
import json
import urllib.error
import urllib.request


def request(method, url, token=None, body=None, headers=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    # RunPod's GraphQL host sits behind Cloudflare, which bans the default urllib UA
    # (error 1010). A normal UA avoids it; harmless on the other providers.
    h = {"Accept": "application/json", "User-Agent": "Mozilla/5.0 (cloud-lease/0.1)"}
    if data is not None:
        h["Content-Type"] = "application/json"
    if token:
        h["Authorization"] = f"Bearer {token}"
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return e.code, parsed
    except urllib.error.URLError as e:
        return 0, {"error": str(e)}
