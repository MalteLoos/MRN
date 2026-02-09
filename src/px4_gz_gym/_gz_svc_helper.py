#!/usr/bin/env python3
"""
_gz_svc_helper.py — persistent gz-transport service helper.

Spawned once by ``GzStepController.__init__()`` as a child process.
Keeps a ``gz.transport.Node`` alive (no subscriptions) so that
``node.request()`` can return in < 1 ms instead of paying ~260 ms
subprocess startup on every call.

Communication is via stdin/stdout with a simple line protocol::

    stdin  → "STEP <n>\\n"   fire a multi_step + pause request
    stdin  → "QUIT\\n"       exit gracefully

    stdout ← "READY\\n"      warmup complete, accepting commands
    stdout ← "OK\\n"         request succeeded
    stdout ← "FAIL\\n"       request failed (node recreated for next call)
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from gz.msgs10.boolean_pb2 import Boolean  # noqa: E402
from gz.msgs10.world_control_pb2 import WorldControl  # noqa: E402
from gz.transport13 import Node  # noqa: E402


def _call(
    node: Node,
    service: str,
    req: WorldControl,
    timeout_ms: int,
) -> bool:
    """Single service call.  Returns True on success."""
    try:
        ok, resp = node.request(service, req, WorldControl, Boolean, timeout_ms)
        return ok and resp.data
    except Exception:
        return False


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: _gz_svc_helper.py <service> <warmup_timeout_ms> <call_timeout_ms>",
            file=sys.stderr,
        )
        sys.exit(1)

    service: str = sys.argv[1]
    warmup_timeout: int = int(sys.argv[2])
    call_timeout: int = int(sys.argv[3])

    node = Node()

    # ── warmup ──────────────────────────────────────────────
    # The first request() on a fresh process typically takes several
    # seconds while gz-transport discovers the service provider.
    # We absorb that cost here so that subsequent calls are fast.
    req = WorldControl()
    req.pause = True
    _call(node, service, req, warmup_timeout)

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    # ── command loop ────────────────────────────────────────
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            break

        # Parse command — currently only STEP is supported.
        req = WorldControl()
        req.pause = True  # always keep paused for stepping

        if line.startswith("STEP "):
            try:
                req.multi_step = int(line[5:])
            except ValueError:
                sys.stdout.write("FAIL\n")
                sys.stdout.flush()
                continue
        else:
            sys.stdout.write("FAIL\n")
            sys.stdout.flush()
            continue

        ok = _call(node, service, req, call_timeout)
        if not ok:
            # ZMQ socket is likely stuck after a recv timeout.
            # Recreate the node for the NEXT call.  Don't retry this
            # one because Gazebo may have already processed the steps
            # (the timeout is about ZMQ recv, not Gazebo execution).
            node = Node()

        sys.stdout.write("OK\n" if ok else "FAIL\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
