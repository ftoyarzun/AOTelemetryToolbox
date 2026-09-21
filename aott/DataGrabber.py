import sys
import time
import threading
from typing import NamedTuple, Optional

import numpy as np

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

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
    "shm.science.frames",
    "shm.science.dit",
    "shm.science.fps",
    "shm.science.gain",
    "acquisition.semaphore",
    "calibration.dm_modes",
    "calibration.z_modes",
    "output.hdf5_dir",
    "output.report_dir",
]


def LoadConfig(path):
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
                       machine, as Telemetry_conversion does.
    """

    timestamps: np.ndarray
    data: list
    stream_timestamps: list


def RecordStreams(streams, duration, sem_nb):
    """Record `streams` in lockstep for `duration` seconds.

    streams[0] sets the pace: each iteration waits for a new frame from it, then
    reads the current value of the other streams.
    """

    pacer = streams[0]
    samples = [[] for _ in streams]
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
        [np.array(s).squeeze() for s in samples],
        [np.array(t) if stream.record_timestamps else None
         for stream, t in zip(streams, shm_timestamps)],
    )


def RecordInParallel(groups, duration, sem_nb):
    """Run RecordStreams on each group of streams in its own thread.

    Returns one Recording per group. An error raised in any thread is raised
    here, instead of leaving that group silently empty.
    """

    recordings = [None] * len(groups)
    errors = []

    def Run(i, streams):
        try:
            recordings[i] = RecordStreams(streams, duration, sem_nb)
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
