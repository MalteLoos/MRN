#!/usr/bin/env python3
"""
_gz_svc_helper.py — persistent gz-transport service helper.

Spawned once by ``GzStepController.__init__()`` as a child process.
Keeps a ``gz.transport.Node`` alive (no subscriptions) so that
``node.request()`` can return in < 1 ms instead of paying ~260 ms
subprocess startup on every call.

Communication is via stdin/stdout with a simple line protocol::

    stdin  → "STEP <n>\\n"                                    fire a multi_step + pause request
    stdin  → "POSE <name> <x> <y> <z> <qw> <qx> <qy> <qz>\\n" teleport a model
    stdin  → "QUIT\\n"                                        exit gracefully

    stdout ← "READY\\n"      warmup complete, accepting commands
    stdout ← "OK\\n"         request succeeded
    stdout ← "FAIL\\n"       request failed (node recreated for next call)
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from gz.msgs10.boolean_pb2 import Boolean  # noqa: E402
from gz.msgs10.pose_pb2 import Pose  # noqa: E402
from gz.msgs10.world_control_pb2 import WorldControl  # noqa: E402
from gz.transport13 import Node  # noqa: E402


def _call_wc(
    node: Node,
    service: str,
    req: WorldControl,
    timeout_ms: int,
) -> bool:
    """WorldControl service call.  Returns True on success."""
    try:
        ok, resp = node.request(service, req, WorldControl, Boolean, timeout_ms)
        return ok and resp.data
    except Exception:
        return False


def _call_pose(
    node: Node,
    service: str,
    req: Pose,
    timeout_ms: int,
) -> bool:
    """Pose (set_pose) service call.  Returns True on success."""
    try:
        ok, resp = node.request(service, req, Pose, Boolean, timeout_ms)
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

    # Derive the world name from the service path for set_pose
    # service = "/world/<name>/control"  →  world_name = <name>
    parts = service.strip("/").split("/")
    world_name = parts[1] if len(parts) >= 3 else "default"
    pose_service = f"/world/{world_name}/set_pose"

    node = Node()

    # ── warmup ──────────────────────────────────────────────
    # The first request() on a fresh process typically takes several
    # seconds while gz-transport discovers the service provider.
    # We absorb that cost here so that subsequent calls are fast.
    req = WorldControl()
    req.pause = True
    _call_wc(node, service, req, warmup_timeout)
    # Also warmup the set_pose service path
    _warmup_pose = Pose()
    _warmup_pose.name = "__warmup_nonexistent__"
    _call_pose(node, pose_service, _warmup_pose, 1000)

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    # ── command loop ────────────────────────────────────────
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            break

        ok = False

        if line.startswith("STEP "):
            # STEP <n>  — advance n physics steps (paused)
            req = WorldControl()
            req.pause = True
            try:
                req.multi_step = int(line[5:])
            except ValueError:
                sys.stdout.write("FAIL\n")
                sys.stdout.flush()
                continue
            ok = _call_wc(node, service, req, call_timeout)

        elif line.startswith("POSE "):
            # POSE <name> <x> <y> <z> <qw> <qx> <qy> <qz>
            tokens = line[5:].split()
            if len(tokens) != 8:
                sys.stdout.write("FAIL\n")
                sys.stdout.flush()
                continue
            try:
                pose_req = Pose()
                pose_req.name = tokens[0]
                pose_req.position.x = float(tokens[1])
                pose_req.position.y = float(tokens[2])
                pose_req.position.z = float(tokens[3])
                pose_req.orientation.w = float(tokens[4])
                pose_req.orientation.x = float(tokens[5])
                pose_req.orientation.y = float(tokens[6])
                pose_req.orientation.z = float(tokens[7])
                ok = _call_pose(node, pose_service, pose_req, call_timeout)
            except (ValueError, IndexError):
                sys.stdout.write("FAIL\n")
                sys.stdout.flush()
                continue

        else:
            sys.stdout.write("FAIL\n")
            sys.stdout.flush()
            continue

        if not ok:
            # ZMQ socket is likely stuck after a recv timeout.
            # Recreate the node for the NEXT call.
            node = Node()

        sys.stdout.write("OK\n" if ok else "FAIL\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
