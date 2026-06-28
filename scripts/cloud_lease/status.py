"""cloud-status — reconcile local lease records against each provider's live (cloud-lease-owned)
instances, and optionally reap leaks. The dollar-safety net the local flock gives for free and
the cloud doesn't (design §7).

Because list_live() is label-filtered, this ONLY ever sees instances cloud-lease created — a
user's unrelated nodes (e.g. a production video-streaming headend) are never listed or reaped.

Categories:
  OK      live instance + record + launcher pid alive
  ORPHAN  live instance + record but launcher pid DEAD (the kill -9 case) -> reap target
  STRAY   live cloud-lease instance with NO local record               -> reap target
  STALE   record but no live instance                                  -> record auto-cleaned
"""
import sys

from . import state


def reconcile(make_backend, providers):
    records = {(r["provider"], str(r["instance_id"])): r for r in state.all_records()}
    rows = []
    for prov in providers:
        try:
            live = make_backend(prov).list_live()
        except SystemExit as e:           # missing key etc. — skip, don't crash the whole status
            rows.append(["SKIP", prov, "-", str(e)])
            continue
        except Exception as e:            # noqa: BLE001
            rows.append(["ERROR", prov, "-", f"list_live failed: {e}"])
            continue
        for i in live:
            key = (prov, str(i["id"]))
            rec = records.pop(key, None)
            if rec is None:
                rows.append(["STRAY", prov, i["id"],
                             f"{i.get('label', '')} {i.get('ip', '')} — no local record"])
            elif not state.pid_alive(rec.get("pid", -1)):
                rows.append(["ORPHAN", prov, i["id"],
                             f"launcher pid {rec.get('pid')} dead — {i.get('label', '')}"])
            else:
                rows.append(["OK", prov, i["id"],
                             f"pid {rec.get('pid')} alive — {i.get('label', '')}"])
    for (prov, iid), rec in records.items():
        rows.append(["STALE", prov, iid, "record but no live instance — cleaning record"])
        state.remove(prov, iid)
    return rows


def run(make_backend, providers, reap=False):
    rows = reconcile(make_backend, providers)
    if not rows:
        print("cloud-status: no cloud-lease instances or records.")
        return 0
    print(f"{'STATUS':8} {'PROVIDER':8} {'ID':24} DETAIL")
    for status, prov, iid, detail in rows:
        print(f"{status:8} {prov:8} {str(iid):24} {detail}")

    reapable = [(p, iid) for (s, p, iid, _d) in rows if s in ("STRAY", "ORPHAN")]
    if not reapable:
        return 0
    if not reap:
        print(f"\ncloud-status: {len(reapable)} reapable (STRAY/ORPHAN). "
              f"Re-run with --reap to destroy them.", file=sys.stderr)
        return 0
    rc = 0
    for prov, iid in reapable:
        print(f"cloud-status: reaping {prov}/{iid}…", file=sys.stderr)
        try:
            make_backend(prov).destroy(iid)
            state.remove(prov, iid)
        except Exception as e:            # noqa: BLE001
            print(f"cloud-status: WARN reap {prov}/{iid} failed: {e}", file=sys.stderr)
            rc = 1
    return rc
