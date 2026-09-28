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

# The HDF5 destinations the analysis reads, "path" for a dataset and "path@Name" for an
# attribute. {camera} is the [acquisition] analysed_camera.
REQUIRED_DESTINATIONS = [
    "WFS/DM_commands",
    "WFS/DM_TimeStamps",
    "WFS/WFS_measurements",
    "WFS/loop_status",
    "WFS/DM_Map",
    "WFS/WFS_Images",
    "WFS/WFS_Images@FPS",
    "WFS@Loop_Gain",
    "WFS@Loop_Leak",
    "WFS@Loop_Freq",
    "Calibration/M2C",
    "Calibration/Z2C",
    "Science/{camera}/Science_PSFs",
    "Science/{camera}/PSF_TimeStamps",
    "Science/{camera}/Science_PSFs@FPS",
]

# Destinations the analysis lines up by index, one row per loop iteration: they must
# come from streams of one thread that keep every sample
LOOP_ALIGNED = ["WFS/DM_commands", "WFS/DM_TimeStamps", "WFS/WFS_measurements", "WFS/loop_status"]


def AsList(value):
    return value if isinstance(value, list) else [value]


def StreamConfig(config, ref):
    """The stream config named "thread.stream", or None."""
    thread, _, stream = ref.partition(".")
    return config.get("threads", {}).get(thread, {}).get("streams", {}).get(stream)


def Window(stream_config):
    """The window of a stream config as {"y": slice, "x": slice}, or None."""
    window = stream_config.get("window")
    return {"y": slice(*window["y"]), "x": slice(*window["x"])} if window else None


def Destinations(config):
    """(destination, owner) for every HDF5 destination the config writes. The owner is
    (thread, stream) for a stream's dataset, timestamps and mean, ("static", name) for a
    static entry's datasets and attrs."""
    found = []
    for thread_name, thread in config.get("threads", {}).items():
        for stream_name, stream in thread.get("streams", {}).items():
            for key in ("dataset", "timestamps", "mean"):
                if key in stream:
                    found.append((stream[key], (thread_name, stream_name)))
    for name, entry in config.get("static", {}).items():
        for key in ("dataset", "attr"):
            for destination in AsList(entry.get(key, [])):
                found.append((destination, ("static", name)))
    return found


def ConfigProblems(config):
    """Everything that stops the data grabber config from producing a file the analysis
    can read, as a list of messages (empty if none)."""

    problems = []
    acquisition = config.get("acquisition", {})
    for key in ("semaphore", "analysed_camera"):
        if acquisition.get(key, "TODO") == "TODO":
            problems.append(f"acquisition.{key} is missing")
    for key in ("hdf5_dir", "report_dir"):
        if config.get("output", {}).get(key, "TODO") == "TODO":
            problems.append(f"output.{key} is missing")

    threads = config.get("threads", {})
    statics = config.get("static", {})
    if not threads:
        problems.append("there is no [threads.<name>] section")
    for thread_name, thread in threads.items():
        streams = thread.get("streams", {})
        if thread.get("pacer") not in streams:
            problems.append(f"threads.{thread_name}.pacer must name one of its streams")
        if "rate" in thread and thread["rate"] not in statics:
            problems.append(f"threads.{thread_name}.rate must name a static entry")
        for stream_name, stream in streams.items():
            where = f"threads.{thread_name}.streams.{stream_name}"
            if stream.get("shm", "TODO") == "TODO":
                problems.append(f"{where}.shm is missing")
            if "dataset" not in stream:
                problems.append(f"{where}.dataset is missing")
            if "background" in stream and stream["background"] not in statics:
                problems.append(f"{where}.background must name a static entry")
    for name, entry in statics.items():
        where = f"static.{name}"
        if "value" not in entry and entry.get("source", "TODO") == "TODO":
            problems.append(f"{where}.source is missing")
        if "dataset" not in entry and "attr" not in entry:
            problems.append(f"{where} has no dataset or attr")
        if "window_of" in entry and StreamConfig(config, entry["window_of"]) is None:
            problems.append(f'{where}.window_of must name a stream as "thread.stream"')

    found = Destinations(config)
    destinations = [destination for destination, _ in found]
    for destination in sorted({d for d in destinations if destinations.count(d) > 1}):
        problems.append(f"{destination} is written by more than one entry")

    camera = acquisition.get("analysed_camera")
    missing = [d.format(camera=camera) for d in REQUIRED_DESTINATIONS if d.format(camera=camera) not in destinations]
    if missing:
        problems.append("the analysis needs these destinations: " + ", ".join(missing))

    owners = dict(found)
    aligned = [owners[d] for d in LOOP_ALIGNED if d in owners]
    if len({thread for thread, _ in aligned}) > 1 or any(thread == "static" for thread, _ in aligned):
        problems.append(", ".join(LOOP_ALIGNED) + " must be recorded by the same thread")
    for thread, stream in aligned:
        if thread != "static" and threads[thread]["streams"][stream].get("keep_every", 1) != 1:
            problems.append(f"threads.{thread}.streams.{stream} must keep every sample (keep_every = 1)")

    return problems


def LoadConfig(path=DATA_GRABBER_FILE):
    """Read the data grabber TOML file and exit if it can't produce a file the analysis reads."""

    with open(path, "rb") as f:
        config = tomllib.load(f)

    problems = ConfigProblems(config)
    if problems:
        sys.exit(f"{path}:\n  " + "\n  ".join(problems))

    return config


class Stream(NamedTuple):
    """One shared-memory image to record.

    keep_every:        keep only every n-th sample. The loop still runs at full rate,
                       so the other streams and the timestamps are not affected.
    window:            part of the image to keep, e.g. {"y": slice(0, 100), "x": slice(0, 100)}
    sliceable:         the shm crops to `window` itself (get_data(y=..., x=...)), which is
                       faster; otherwise the whole image is read and indexed here
    background:        array subtracted from every sample (already cropped to `window`),
                       which is then stored as float32
    record_timestamps: also record shm.get_timestamp() for every kept sample, as Unix time
    mean:              also compute the mean of every sample read, kept or not (the
                       stream is then read at every iteration)
    saturation_level:  also record the maximum raw value (before the background is
                       subtracted), to compare with this level
    """

    shm: object
    keep_every: int = 1
    window: Optional[dict] = None
    sliceable: bool = True
    background: Optional[np.ndarray] = None
    record_timestamps: bool = False
    mean: bool = False
    saturation_level: Optional[float] = None


def ReadRaw(stream, **kwargs):
    """One sample of `stream`: get_data(**kwargs), cropped to its window."""
    window = stream.window or {}
    if stream.sliceable:
        return stream.shm.get_data(**kwargs, **window)
    sample = stream.shm.get_data(**kwargs)
    if window:
        sample = np.asarray(sample)[window["y"], window["x"]]
    return sample


def ReadSample(stream, **kwargs):
    """One sample of `stream`: ReadRaw, background subtracted."""
    sample = ReadRaw(stream, **kwargs)
    if stream.background is not None:
        sample = np.subtract(sample, stream.background, dtype=np.float32)
    return sample


def CheckClock(shm_time, name):
    """Warn when an shm timestamp and this machine's clock differ by more than 1 s:
    get_timestamp() returns a naive datetime, read as local time, so a time-zone
    mismatch would shift every recorded time."""
    offset = shm_time - time.time()
    if abs(offset) > 1:
        print(f"WARNING: {name}: the shm timestamps are {offset:+.1f} s from this machine's clock")


class Recording(NamedTuple):
    """Result of RecordStreams.

    timestamps:        Unix time [s] of every loop iteration, kept or not: the
                       shm.get_timestamp() of the first stream's frame
    data:              one array per stream, with the kept samples along the first axis
    stream_timestamps: one entry per stream, the shm.get_timestamp() of every kept
                       sample as Unix time [s], so they line up with `data`. None for
                       streams without record_timestamps. get_timestamp() returns a
                       naive datetime, which .timestamp() reads as local time of this
                       machine.
    means:             one entry per stream, the mean of every sample read, or None
                       for streams without `mean`
    maxima:            one entry per stream, the maximum raw value, or None for
                       streams without `saturation_level`
    """

    timestamps: np.ndarray
    data: list
    stream_timestamps: list
    means: list
    maxima: list


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


def RecordStreams(streams, duration, sem_nb, rate=None, name="streams"):
    """Record `streams` in lockstep for `duration` seconds.

    streams[0] sets the pace: each iteration waits for a new frame from it, then
    reads the current value of the other streams. `rate` [Hz], the expected pace,
    sizes the preallocated buffers (duration * rate, plus 10 %); without it they
    start small and grow. `name` labels the printed messages.
    """

    expected = math.ceil(duration * rate * 1.1) + 16 if rate else 1024
    samples = [SampleBuffer(math.ceil(expected / stream.keep_every)) for stream in streams]
    shm_timestamps = [[] for _ in streams]
    sums = [None] * len(streams)
    counts = [0] * len(streams)
    maxima = [None] * len(streams)
    timestamps = []

    start_time = time.monotonic()
    n = 0
    while time.monotonic() - start_time < duration:
        for k, stream in enumerate(streams):
            kept = n % stream.keep_every == 0
            if k > 0 and not kept and not stream.mean:
                continue
            if k == 0:
                raw = ReadRaw(stream, check=True, semNb=sem_nb)
                timestamps.append(stream.shm.get_timestamp().timestamp())
                if n == 0:
                    CheckClock(timestamps[0], name)
            else:
                raw = ReadRaw(stream, semNb=sem_nb)
            if stream.saturation_level is not None:
                peak = np.max(raw)
                maxima[k] = peak if maxima[k] is None else max(maxima[k], peak)
            sample = raw if stream.background is None else np.subtract(raw, stream.background, dtype=np.float32)
            if stream.mean:
                if sums[k] is None:
                    sums[k] = np.zeros(np.shape(sample))
                sums[k] += sample
                counts[k] += 1
            if kept:
                samples[k].append(sample)
                if stream.record_timestamps:
                    shm_timestamps[k].append(timestamps[-1] if k == 0 else stream.shm.get_timestamp().timestamp())
        if n == 0:
            size = sum(s.array.nbytes for s in samples if s.array is not None)
            print(f"{name}: {size / 1e9:.2f} GB preallocated")
        n += 1

    if not timestamps:
        raise RuntimeError(f"{name}: no frame received from the first stream in {duration} s")

    return Recording(
        np.array(timestamps),
        [s.data() for s in samples],
        [np.array(t) if stream.record_timestamps else None
         for stream, t in zip(streams, shm_timestamps)],
        [np.squeeze(total / count) if total is not None else None for total, count in zip(sums, counts)],
        maxima,
    )


def RecordInParallel(threads, duration):
    """Run RecordStreams on each thread's streams in its own thread. `threads` maps
    a name to (streams, sem_nb, rate), see RecordStreams.

    Returns one Recording per name. An error raised in any thread is raised
    here, instead of leaving that thread silently empty.
    """

    recordings = {}
    errors = []

    def Run(name, streams, sem_nb, rate):
        try:
            recordings[name] = RecordStreams(streams, duration, sem_nb, rate, name)
        except Exception as e:
            errors.append(e)

    workers = [
        threading.Thread(target=Run, args=(name, *thread), daemon=True)
        for name, thread in threads.items()
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    if errors:
        raise errors[0]

    return recordings
