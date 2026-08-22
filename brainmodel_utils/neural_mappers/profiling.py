"""Optional phase timing for mapper and consistency benchmarks."""

import threading
import time
from contextlib import contextmanager

_state = threading.local()


class PhaseTimes(dict):
    def add(self, name, seconds):
        self[name] = self.get(name, 0.0) + float(seconds)


def current_phase_times():
    return getattr(_state, "times", None)


@contextmanager
def record_phases():
    previous = getattr(_state, "times", None)
    times = PhaseTimes()
    _state.times = times
    try:
        yield times
    finally:
        _state.times = previous


def add_phase(name, seconds):
    times = current_phase_times()
    if times is not None:
        times.add(name, seconds)


@contextmanager
def phase(name):
    times = current_phase_times()
    start = time.perf_counter()
    try:
        yield
    finally:
        if times is not None:
            times.add(name, time.perf_counter() - start)
