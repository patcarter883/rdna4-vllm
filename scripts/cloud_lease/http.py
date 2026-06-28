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


class HttpError(RuntimeError):
    """Raised by request_ok on a connection failure (status 0) or an unexpected status."""

    def __init__(self, method, url, status, body):
        self.status = status
        self.body = body
        super().__init__(f"{method} {url} -> [{status}] {body}")


def request_ok(method, url, token=None, body=None, ok=range(200, 300), headers=None, timeout=30):
    """Like request(), but RAISES HttpError on a connection-level failure (status 0, which a
    plain `st >= 300` check silently passes) or any status outside `ok`. Use for control-plane
    mutations (provision/destroy/key) that must fail loud; polls keep using request() so they
    can tolerate-and-retry transient errors."""
    st, resp = request(method, url, token, body, headers, timeout)
    if st == 0 or st not in ok:
        raise HttpError(method, url, st, resp)
    return st, resp
