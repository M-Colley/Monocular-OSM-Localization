"""One wall-clock deadline for assisted work, owned cleanup and final publication.

The benchmark supplies the absolute monotonic deadline before Python starts.
Standalone callers may supply a relative budget. This never changes admission
rules: an expired optional component is a failed request, not a partial answer.
"""
from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import dataclass, field

DEADLINE_ENV = "MONOCULAR_OSM_DEADLINE_MONOTONIC_S"
DEFAULT_FINALIZATION_RESERVE_S = 120.0
DEFAULT_CLEANUP_RESERVE_S = 10.0


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


class DeadlineExpired(TimeoutError):
    def __init__(self, event):
        self.event = event
        super().__init__(f"assisted runtime deadline: {event['kind']} {event['reason']}")


class CleanupFailed(RuntimeError):
    """Never claim a normally completed parent while an owned birth is live."""


@dataclass
class RuntimeBudget:
    deadline: float
    finalization_reserve_s: float
    cleanup_reserve_s: float
    origin: str
    started: float
    clock: object = time.monotonic
    events: list = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg, *, environ=None, clock=None):
        environ = os.environ if environ is None else environ
        clock = time.monotonic if clock is None else clock
        now = clock()
        relative = getattr(cfg, "runtime_budget_s", None)
        configured_deadline = getattr(cfg, "runtime_deadline_monotonic_s", None)
        inherited = environ.get(DEADLINE_ENV)
        if relative is None and inherited is None and configured_deadline is None:
            return None
        final = positive(getattr(cfg, "runtime_finalization_reserve_s", DEFAULT_FINALIZATION_RESERVE_S), "runtime_finalization_reserve_s")
        cleanup = positive(getattr(cfg, "runtime_cleanup_reserve_s", DEFAULT_CLEANUP_RESERVE_S), "runtime_cleanup_reserve_s")
        limits = []
        if configured_deadline is not None:
            limits.append(positive(configured_deadline, "runtime_deadline_monotonic_s"))
        if relative is not None:
            limits.append(now + positive(relative, "runtime_budget_s"))
        if inherited is not None:
            try:
                limits.append(positive(float(inherited), DEADLINE_ENV))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("invalid inherited runtime deadline") from error
        return cls(min(limits), final, cleanup, "benchmark" if inherited is not None else "config", now, clock)

    @property
    def work_deadline(self):
        return self.deadline - self.finalization_reserve_s - self.cleanup_reserve_s

    def remaining_work(self):
        return max(0.0, self.work_deadline - self.clock())

    def require_work(self, kind):
        if self.remaining_work() <= 0:
            event = {"kind": kind, "reason": "not_started_budget_reserved_for_cleanup_and_publication",
                     "status": "not_started", "monotonic_s": self.clock()}
            self.events.append(event)
            raise DeadlineExpired(event)

    def snapshot(self):
        return {"schema": "assisted-runtime-budget-v1", "origin": self.origin,
                "deadline_monotonic_s": self.deadline, "started_monotonic_s": self.started,
                "work_deadline_monotonic_s": self.work_deadline,
                "finalization_reserve_s": self.finalization_reserve_s,
                "cleanup_reserve_s": self.cleanup_reserve_s, "events": list(self.events)}

    def require_publication_time(self):
        if self.clock() >= self.deadline:
            raise TimeoutError("mandatory assisted publication exceeded the shared overall deadline")


class OwnedProcessTree:
    """Signal only observed PID+birth pairs, not recycled/unrelated processes.

    Polling cannot certify grandchildren orphaned before being observed. The
    receipt deliberately claims only the tracked births, never a global scan.
    """
    def __init__(self, child, *, clock=None, process_factory=None):
        import psutil
        self.child = child
        self.clock = time.monotonic if clock is None else clock
        self.process = psutil.Process if process_factory is None else process_factory
        self.errors = (psutil.NoSuchProcess, psutil.ZombieProcess)
        self.births = {}
        self.root_birth = None
        try:
            self.root_birth = positive(self.process(child.pid).create_time(), "owned root birth")
            self.births[child.pid] = self.root_birth
        except self.errors:
            if child.poll() is None:
                raise CleanupFailed("cannot bind owned child birth")

    def matching(self, pid, birth):
        try:
            process = self.process(pid)
            return process if process.create_time() == birth else None
        except self.errors:
            return None

    def capture(self):
        root = self.matching(self.child.pid, self.root_birth)
        if root is not None:
            try:
                descendants = root.children(recursive=True)
            except self.errors:
                return  # Normal root exit raced this read; tracked births remain.
            for process in descendants:
                try:
                    birth = positive(process.create_time(), "owned descendant birth")
                    if birth >= self.root_birth:
                        self.births.setdefault(process.pid, birth)
                except self.errors:
                    pass

    def wait(self, deadline):
        while True:
            self.capture()
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("owned assisted child", 0)
            try:
                return self.child.wait(timeout=min(remaining, .25))
            except subprocess.TimeoutExpired:
                pass

    def stop(self, timeout):
        capture_error = None
        try:
            self.capture()
        except Exception as error:
            # Still stop the original Popen handle and already captured births.
            capture_error = error
        # The Popen handle targets the original root even if its PID is reused.
        if self.child.poll() is None:
            self.child.kill()
        for pid, birth in reversed(list(self.births.items())):
            if pid == self.child.pid:
                continue
            process = self.matching(pid, birth)
            if process is not None:
                try:
                    process.kill()
                except self.errors:
                    pass
        end = self.clock() + max(0.0, timeout)
        while True:
            alive = []
            for pid, birth in self.births.items():
                process = self.matching(pid, birth)
                if process is not None:
                    try:
                        if process.is_running() and process.status() != "zombie":
                            alive.append({"pid": pid, "create_time": birth})
                    except self.errors:
                        pass
            if not alive:
                self.child.wait(timeout=max(.001, end - self.clock()))
                if capture_error is not None:
                    raise CleanupFailed("owned process capture failed; tracked births stopped") from capture_error
                return {"owned_births": [{"pid": p, "create_time": b} for p, b in self.births.items()],
                        "all_tracked_owned_births_gone": True,
                        "ownership_scope": "observed_pid_births_only; unobserved_orphans_not_certified"}
            if self.clock() >= end:
                raise CleanupFailed(f"owned child cleanup exceeded reserve: {alive}")
            time.sleep(min(.02, max(0.0, end - self.clock())))
