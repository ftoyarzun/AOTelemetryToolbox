import math
import sys
import time
import threading
from typing import NamedTuple, Optional

import numpy as np

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

from aott.config import DATA_GRABBER_FILE

# Keys that must be present, and not "TODO", in the data grabber config file
REQUIRED_KEYS = [
    "shm.wfs.frames",
    "shm.wfs.valid_pixel_map",
    "shm.wfs.measurements",
    "shm.wfs.fps",
    "shm.wfs.gain",
    "shm.dm.commands",
    "shm.dm.m2c",
    "shm.loop.gain",
    "shm.loop.leak",
    "shm.loop.cmd",
    "shm.science.frames",
    "shm.science.dit",
    "shm.science.fps",
    "shm.science.gain",
    "shm.dm.dm_map",
    "acquisition.semaphore",
    "calibration.Z2C",
    "output.hdf5_dir",
    "output.report_dir",
]


def LoadConfig(path=DATA_GRABBER_FILE):
    """Read the data grabber TOML file and exit if a required value is missing."""

    with open(path, "rb") as f:
        config = tomllib.load(f)

    missing = []
    for key in REQUIRED_KEYS:
        value = config
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None or value == "TODO":
            missing.append(key)

    if missing:
        sys.exit(f"{path}: fill in these values first: " + ", ".join(missing))

    return config


class Stream(NamedTuple):
    """One shared-memory image to record.

    keep_every:        keep only every n-th sample. The loop still runs at full rate,
                       so the other streams and the timestamps are not affected.
    window:            extra get_data arguments, e.g. {"y": slice(0, 100), "x": slice(0, 100)}
    record_timestamps: also record shm.get_timestamp() for every kept sample, as Unix time
    """

    shm: object
    keep_every: int = 1
    window: Optional[dict] = None
    record_timestamps: bool = False


class Recording(NamedTuple):
    """Result of RecordStreams.

    timestamps:        Unix time [s] of every loop iteration, kept or not, taken
                       on this machine when the first stream delivers a frame
    data:              one array per stream, with the kept samples along the first axis
    stream_timestamps: one entry per stream, the shm.get_timestamp() of every kept
                       sample as Unix time [s], so they line up with `data`. None for
                       streams without record_timestamps. get_timestamp() returns a
                       naive datetime, which .timestamp() reads as local time of this
                       machine.
    """

    timestamps: np.ndarray
    data: list
    stream_timestamps: list


class SampleBuffer:
    """The samples of one stream, copied into one preallocated array as they
    arrive, so recording holds the data once in memory. The array grows by half
    when the capacity is reached."""

    def __init__(self, capacity):
        self.capacity = max(int(capacity), 1)
        self.array = None
        self.n = 0

    def append(self, sample):
        sample = np.asarray(sample)
        if self.array is None:
            self.array = np.empty((self.capacity,) + sample.shape, dtype=sample.dtype)
        elif self.n == len(self.array):
            grown = np.empty((len(self.array) * 3 // 2 + 1,) + self.array.shape[1:], dtype=self.array.dtype)
            grown[:self.n] = self.array
            self.array = grown
        self.array[self.n] = sample
        self.n += 1

    def data(self):
        """The recorded samples along the first axis, without copying, and without
        the length-1 axes of a sample (a (1, 1) scalar image gives shape (n,))."""
        if self.array is None:
            return np.empty(0)
        sample_shape = tuple(d for d in self.array.shape[1:] if d != 1)
        return self.array[:self.n].reshape((self.n,) + sample_shape)


def RecordStreams(streams, duration, sem_nb, rate=None):
    """Record `streams` in lockstep for `duration` seconds.

    streams[0] sets the pace: each iteration waits for a new frame from it, then
    reads the current value of the other streams. `rate` [Hz], the expected pace,
    sizes the preallocated buffers (duration * rate, plus 10 %); without it they
    start small and grow.
    """

    pacer = streams[0]
    expected = math.ceil(duration * rate * 1.1) + 16 if rate else 1024
    samples = [SampleBuffer(math.ceil(expected / stream.keep_every)) for stream in streams]
    shm_timestamps = [[] for _ in streams]
    timestamps = []

    start_time = time.monotonic()
    n = 0
    while time.monotonic() - start_time < duration:
        frame = pacer.shm.get_data(check=True, semNb=sem_nb, **(pacer.window or {}))
        timestamps.append(time.time())

        for k, stream in enumerate(streams):
            if n % stream.keep_every != 0:
                continue
            if k == 0:
                samples[k].append(frame)
            else:
                samples[k].append(
                    stream.shm.get_data(semNb=sem_nb, **(stream.window or {}))
                )
            if stream.record_timestamps:
                shm_timestamps[k].append(stream.shm.get_timestamp().timestamp())
        n += 1

    if not timestamps:
        raise RuntimeError(f"No frame received from the first stream in {duration} s")

    return Recording(
        np.array(timestamps),
        [s.data() for s in samples],
        [np.array(t) if stream.record_timestamps else None
         for stream, t in zip(streams, shm_timestamps)],
    )


def RecordInParallel(groups, duration, sem_nb, rates=None):
    """Run RecordStreams on each group of streams in its own thread, with
    `rates[i]` the expected rate of group i (see RecordStreams).

    Returns one Recording per group. An error raised in any thread is raised
    here, instead of leaving that group silently empty.
    """

    recordings = [None] * len(groups)
    errors = []
    rates = rates or [None] * len(groups)

    def Run(i, streams):
        try:
            recordings[i] = RecordStreams(streams, duration, sem_nb, rates[i])
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=Run, args=(i, streams), daemon=True)
        for i, streams in enumerate(groups)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        raise errors[0]

    return recordings
