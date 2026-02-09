"""
gz_step.py — deterministic Gazebo Harmonic stepping via gz-transport.

Talks directly to gz-sim over IPC (no ROS 2 dependency) to:
  • pause / unpause the world
  • advance exactly *n* physics steps  (multi_step)
  • read the simulation clock

The protobuf env-var is set automatically so the system-packaged
`gz.msgs10` works even when a newer pip `protobuf` is installed.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Ensure compatibility with the system-packaged gz.msgs10 protos
# which were generated with protoc 3.x.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from gz.msgs10.clock_pb2 import Clock  # noqa: E402
from gz.transport13 import Node  # noqa: E402


class GzStepController:
    """Pause / step / unpause a Gazebo Harmonic world.

    Implementation notes
    --------------------
    A persistent **helper subprocess** (``_gz_svc_helper.py``) keeps
    a ``gz.transport.Node`` alive for service calls.  Because it has
    **no subscriptions**, ``node.request()`` returns in < 1 ms
    (vs. ~260 ms per ``gz service`` CLI invocation).

    The helper communicates with the controller over stdin/stdout
    pipes.  ``step()`` writes a command and returns immediately
    (fire-and-forget); completion is confirmed via clock-topic
    subscription on a separate ``Node`` in the **main** process
    that is used only for pub-sub.

    If the helper is unavailable (startup delay, crash), all calls
    fall back transparently to ``gz service`` CLI subprocesses.

    **One-shot commands** (pause, unpause, reset) always go through
    the CLI since latency is not critical for them.

    IMPORTANT — proto3 + Gazebo interaction:
    In protobuf 3 every scalar field has a default value (``bool`` →
    ``False``, ``uint32`` → ``0``) and default values are **not**
    serialised on the wire.  Gazebo's ``SimulationRunner`` reads
    ``msg.pause()`` on *every* ``WorldControl`` it receives.  If
    we send a message that only sets ``multi_step`` the implicit
    ``pause = false`` **unpauses** the world, defeating lockstep.

    Fix: we **always** set ``pause = True`` together with
    ``multi_step`` so the world stays paused between steps.
    """

    def __init__(
        self,
        world_name: str = "default",
        timeout_ms: int = 5_000,
    ) -> None:
        self.world_name = world_name
        self.timeout_ms = timeout_ms

        self._service = f"/world/{world_name}/control"
        self._clock_topic = f"/world/{world_name}/clock"

        # ── clock subscription — pub-sub only, no service calls ──
        self._node = Node()
        self._sim_sec: float = 0.0
        self._sim_sec_prev: float = 0.0
        self._clock_lock = threading.Lock()
        self._clock_event = threading.Event()
        self._node.subscribe(Clock, self._clock_topic, self._on_clock)

        # Reusable CLI prefix for fallback service calls
        self._cli_prefix = [
            "gz",
            "service",
            "-s",
            self._service,
            "--reqtype",
            "gz.msgs.WorldControl",
            "--reptype",
            "gz.msgs.Boolean",
        ]

        # ── persistent helper subprocess for fast step() calls ──
        self._helper_proc: subprocess.Popen | None = None
        self._helper_ready = threading.Event()
        self._start_helper()
        atexit.register(self.close)

        # CLI fire-and-forget fallback (used when helper unavailable)
        self._step_proc: subprocess.Popen | None = None

    # ── public API ──────────────────────────────────────────

    def pause(self) -> bool:
        """Pause the simulation (blocking CLI call)."""
        return self._cli_call("pause: true")

    def unpause(self) -> bool:
        """Unpause (free-run) the simulation (blocking CLI call)."""
        return self._cli_call("pause: false")

    def step(self, n: int = 1) -> None:
        """Fire exactly *n* physics steps while keeping the world
        paused.

        Uses the persistent helper subprocess for low latency
        (< 1 ms).  Falls back to a one-shot ``gz service`` CLI
        call if the helper is unavailable.

        Use :meth:`step_and_wait` to block until the steps have
        actually completed (confirmed via clock subscription).
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not self._helper_send(f"STEP {n}"):
            self._fire(f"pause: true, multi_step: {n}")

    def step_and_wait(
        self,
        n: int = 1,
        step_size: float = 0.004,
        wall_timeout: float = 5.0,
    ) -> float:
        """Step *n* physics steps **and** block until the clock topic
        confirms that sim-time has advanced by at least ``n * step_size``.

        Returns the new simulation time (seconds).

        Parameters
        ----------
        n : int
            Number of physics steps.
        step_size : float
            Gazebo ``<max_step_size>`` in seconds (PX4 default world
            uses 0.004 s → 250 Hz).
        wall_timeout : float
            Max wall-clock seconds to wait for the clock to catch up.
        """
        t_before = self.sim_time
        expected = t_before + n * step_size - 1e-9

        self._clock_event.clear()
        self.step(n)

        # Spin until the clock confirms the expected sim-time.
        # The _clock_event is set by _on_clock whenever sim-time
        # advances.  We must check sim_time *before* clearing the
        # event to avoid a race where the signal arrives between
        # the clear and the check.
        deadline = time.monotonic() + wall_timeout
        while time.monotonic() < deadline:
            # 1) Check first — the callback may already have fired.
            if self.sim_time >= expected:
                return self.sim_time
            # 2) Wait for the next clock update.
            self._clock_event.wait(timeout=0.05)
            # 3) Check *again* before clearing so we never lose
            #    a signal that arrived while we were checking.
            if self.sim_time >= expected:
                return self.sim_time
            self._clock_event.clear()

        # Timed out – return whatever we have
        return self.sim_time

    def reset_world(self) -> bool:
        """Send a world-reset request (time-only reset, blocking CLI)."""
        return self._cli_call("pause: true, reset: {all: true}")

    @property
    def sim_time(self) -> float:
        """Latest simulation time in seconds."""
        with self._clock_lock:
            return self._sim_sec

    # ── internals ───────────────────────────────────────────

    def _cli_call(self, req_text: str) -> bool:
        """Blocking ``gz service`` call for one-shot commands."""
        try:
            r = subprocess.run(
                [
                    *self._cli_prefix,
                    "--timeout",
                    str(self.timeout_ms),
                    "--req",
                    req_text,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_ms / 1000 + 2,
            )
            return "data: true" in r.stdout
        except subprocess.TimeoutExpired:
            return False

    def _fire(self, req_text: str) -> None:
        """Non-blocking ``gz service`` call for step commands.

        The subprocess sends the WorldControl message and exits on its
        own after ``--timeout``.  We don't wait for it — clock events
        are the authoritative completion signal.
        """
        # Reap the previous subprocess (if it finished)
        if self._step_proc is not None:
            self._step_proc.poll()
        self._step_proc = subprocess.Popen(
            [
                *self._cli_prefix,
                "--timeout",
                "2000",
                "--req",
                req_text,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # ── persistent helper management ────────────────────────

    def _start_helper(self) -> None:
        """Spawn the persistent gz-transport service helper.

        The helper creates a ``Node`` with no subscriptions, does
        one warmup call (absorbs ZMQ discovery delay), then enters
        a command loop reading from stdin.
        """
        helper_script = Path(__file__).with_name("_gz_svc_helper.py")
        if not helper_script.exists():
            return
        try:
            self._helper_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(helper_script),
                    self._service,
                    str(self.timeout_ms),  # warmup timeout
                    "500",  # per-call timeout (ms)
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,  # line-buffered
            )
        except OSError:
            self._helper_proc = None
            return
        t = threading.Thread(target=self._drain_helper, daemon=True)
        t.start()

    def _drain_helper(self) -> None:
        """Background thread: read helper stdout, watch for READY.

        OK / FAIL responses for fire-and-forget ``STEP`` commands
        are silently consumed to prevent the pipe buffer from
        filling up.
        """
        proc = self._helper_proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                if line.strip() == "READY":
                    self._helper_ready.set()
        except (ValueError, OSError):
            pass
        # stdout closed → helper died
        self._helper_ready.clear()

    def _helper_send(self, cmd: str) -> bool:
        """Send a fire-and-forget command to the helper.

        Returns ``True`` if the command was written to the pipe,
        ``False`` if the helper is not available (caller should
        use the CLI fallback).
        """
        if self._helper_proc is None or not self._helper_ready.is_set():
            return False
        try:
            assert self._helper_proc.stdin is not None
            self._helper_proc.stdin.write(cmd + "\n")
            self._helper_proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            self._helper_proc = None
            self._helper_ready.clear()
            return False

    def close(self) -> None:
        """Shut down the helper subprocess gracefully."""
        proc = self._helper_proc
        if proc is None:
            return
        self._helper_proc = None
        self._helper_ready.clear()
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

    def _on_clock(self, msg: Clock) -> None:
        t = msg.sim
        new_t = t.sec + t.nsec * 1e-9
        with self._clock_lock:
            self._sim_sec_prev = self._sim_sec
            self._sim_sec = new_t
        # Only signal when sim-time actually advanced (Gazebo keeps
        # publishing clock while paused, but with the same sim value).
        if new_t > self._sim_sec_prev + 1e-9:
            self._clock_event.set()
