import time
from typing import BinaryIO, Union, Optional

import av
from PIL import Image


class VideoDecoder:
    """Minimal, robust frame extractor on top of PyAV."""
    _EPS = 1e-3

    def __init__(
        self,
        src: Union[str, BinaryIO],
        default_thread_type: Optional[str] = None,
        parallel_frames: bool = False,
        forward_gop: int = 10,
        backward_gop: int = 5,
    ):
        # Keep a file handle open if a path was provided (avoids odd issues on some platforms).
        self._fh: Optional[BinaryIO] = open(src, "rb") if isinstance(src, str) else None
        av_src = src

        # Ignore metadata errors (PyAV quirk).
        self.container = av.open(av_src, mode="r", metadata_errors="ignore")

        self.video_stream = self.container.streams.video[0]
        self.framerate = float(self.video_stream.guessed_rate)
        self._frame_duration = 1.0 / self.framerate
        self._half_frame_window = self._frame_duration / 2 + self._EPS

        self._default_thread_type = default_thread_type or self.video_stream.thread_type
        self.video_stream.thread_type = self._default_thread_type
        self._parallel_frames = parallel_frames

        self._min_seek_interval = forward_gop / self.framerate
        self._back_seek_step = backward_gop / self.framerate

    # -- lifecycle -------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        try:
            self.container.close()
        finally:
            if self._fh:
                self._fh.close()

    # -- public API ------------------------------------------------------------

    def frame_at_index(self, index: int, frame_seek_timeout_sec: float = 10.0) -> Image.Image:
        return self.frame_at_time(index / self.framerate, frame_seek_timeout_sec)

    def frame_at_time(self, t: float, frame_seek_timeout_sec: float = 10.0) -> Image.Image:
        return self.frames_at_times([t], frame_seek_timeout_sec)[0]

    def frames_at_indices(self, indices: list[int], frame_seek_timeout_sec: float = 10.0) -> list[Image.Image]:
        return self.frames_at_times([i / self.framerate for i in indices], frame_seek_timeout_sec)

    def frames_at_times(self, times: list[float], frame_seek_timeout_sec: float = 10.0) -> list[Image.Image]:
        self._configure_threads(len(times))

        ordered = sorted(enumerate(times), key=lambda x: x[1])  # [(pos, t), ...]
        results: list[tuple[Image.Image, int]] = []
        frame_iter = None
        last_ts = float("-inf")
        last_frame = None  # av.VideoFrame

        for pos, target_ts in ordered:
            deadline = time.time() + frame_seek_timeout_sec

            # Reuse previous frame if targets fall in the same presentation interval.
            if self._same_frame(last_ts, target_ts) and results:
                results.append((results[-1][0].copy(), pos))
                continue

            seek_start = target_ts
            found = False

            while not found:
                if self._need_seek(target_ts, last_ts):
                    self._seek(seek_start)
                    frame_iter = None

                if frame_iter is None:
                    frame_iter = iter(self.container.decode(video=0))
                    min_ts_since_seek = float("inf")
                    prev_ts = None

                overshot = False

                # Scan forward until we hit the target or overshoot it.
                while True:
                    if time.time() > deadline:
                        raise TimeoutError(f"Timed out after {frame_seek_timeout_sec:.1f}s while decoding frame at t={target_ts:.3f}s.")
                    try:
                        frame = next(frame_iter)
                        ts = frame.time

                        if prev_ts is not None and ts < prev_ts:
                            raise RuntimeError("Video frames are not monotonic; file may be corrupted (e.g., CTTS invalid).")

                        prev_ts = ts
                        last_ts = ts
                        last_frame = frame
                        min_ts_since_seek = min(min_ts_since_seek, ts)

                        if self._same_frame(ts, target_ts):
                            results.append((frame.to_image(), pos))
                            found = True
                            break

                        if ts > target_ts and min_ts_since_seek < target_ts:
                            raise RuntimeError(
                                f"Target frame at t={target_ts:.3f}s appears missing "
                                f"(scanned from {min_ts_since_seek:.3f}s to {ts:.3f}s)."
                            )

                        if ts > target_ts:
                            overshot = True
                            break

                    except StopIteration:
                        # Allow the last frame to cover until EOF.
                        if last_ts > float("-inf") and target_ts > last_ts and abs(last_ts - target_ts) <= self._frame_duration + self._EPS:
                            results.append((last_frame.to_image(), pos))
                            found = True
                            break
                        # Otherwise, restart the outer loop (e.g., hit EOF while seeking).
                        frame_iter = None
                        break

                if found:
                    break

                if overshot:
                    seek_start = max(0.0, seek_start - self._back_seek_step)
                    continue

                # No progress made and no overshoot: give up if already at the start.
                if seek_start == 0.0:
                    raise RuntimeError(f"Could not find frame at t={target_ts:.3f}s.")
                seek_start = max(0.0, seek_start - self._back_seek_step)

        # Restore original order.
        results.sort(key=lambda x: x[1])
        return [img for img, _ in results]

    # -- helpers ---------------------------------------------------------------

    def _configure_threads(self, n_targets: int) -> None:
        if not self.video_stream.codec_context.is_open:
            self.video_stream.thread_type = (
                "AUTO" if (n_targets > 1 and self._parallel_frames) else self._default_thread_type
            )

    def _need_seek(self, target_ts: float, last_ts: float) -> bool:
        return (target_ts - last_ts) >= self._min_seek_interval or last_ts >= target_ts

    def _seek(self, t: float) -> None:
        offset = round(t / self.video_stream.time_base)
        self.container.seek(offset, backward=True, any_frame=False, stream=self.video_stream)

    def _same_frame(self, a: float, b: float) -> bool:
        return abs(a - b) <= self._half_frame_window

    # Backwards-compatible method names
    decode_frame_at_index = frame_at_index
    decode_frame_at_time = frame_at_time
    decode_frames_at_indexes = frames_at_indices
    decode_frames_at_times = frames_at_times

#----------------------------------------------------------------------------
