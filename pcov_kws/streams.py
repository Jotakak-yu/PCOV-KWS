import pyaudio
import queue
import threading
from typing import Callable
import numpy as np
from scipy.signal import resample

NoParameterFunction = Callable[[], None]
AudioFrameFunction = Callable[[], np.array]

# Sentinel used to unblock queue.get() when the stream is closing
_STOP_SENTINEL = object()


class CustomAudioStream:
    """Applies a sliding window over an audio source."""

    def __init__(
        self,
        open_stream: Callable[[], None],
        close_stream: Callable[[], None],
        get_next_frame: Callable[[], np.array],
        window_length_secs=1,
        sliding_window_secs: float = 1/8,
        sample_rate=16000,
    ):
        self._open_stream = open_stream
        self._close_stream = close_stream
        self._get_next_frame = get_next_frame
        self._sample_rate = sample_rate
        self._window_size = int(window_length_secs * sample_rate)
        self._sliding_window_size = int(sliding_window_secs * sample_rate)
        self._out_audio = np.zeros(self._window_size)
        self.stop = False

    def start_stream(self):
        self._out_audio = np.zeros(self._window_size)
        self._open_stream()
        # Pre-fill the sliding window buffer
        for _ in range(self._sample_rate // self._sliding_window_size - 1):
            self.getFrame()

    def close_stream(self):
        self.stop = True
        self._close_stream()
        self._out_audio = np.zeros(self._window_size)

    def getFrame(self):
        """Return the current window after sliding in one new chunk."""
        if self.stop:
            return None

        new_frame = self._get_next_frame()
        if new_frame is None:
            return None

        assert new_frame.shape == (self._sliding_window_size,), \
            "audio frame size from src doesn't match sliding_window_secs"

        self._out_audio = np.append(
            self._out_audio[self._sliding_window_size:],
            new_frame,
        )
        return self._out_audio


class SimpleMicStream(CustomAudioStream):
    """Microphone stream using pyaudio callback mode.
    """

    def __init__(self, window_length_secs=1, sliding_window_secs: float = 1/8,
                 custom_channels=2, custom_rate=48000, custom_device_index=None):

        self._pa = pyaudio.PyAudio()
        self._custom_channels = custom_channels
        self._custom_rate = custom_rate
        target_rate = 16000

        CHUNK = int(sliding_window_secs * custom_rate)
        print(f"Chunk size (captured at {custom_rate}Hz): {CHUNK}")

        # maxsize caps memory usage; oldest frames are dropped when the
        # consumer (main thread) falls behind.
        self._audio_queue: queue.Queue = queue.Queue(maxsize=64)

        def _audio_callback(in_data, frame_count, time_info, status):
            """Called by PortAudio from a background thread on every chunk."""
            try:
                self._audio_queue.put_nowait(in_data)
            except queue.Full:
                pass  # Drop oldest-equivalent by not enqueueing
            return (None, pyaudio.paContinue)

        self._mic_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=custom_channels,
            rate=custom_rate,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=custom_device_index,
            stream_callback=_audio_callback,
        )
        self._mic_stream.stop_stream()

        self._silent_frame = np.zeros(int(sliding_window_secs * target_rate), dtype=np.int16)

        def get_next_frame():
            # Poll with a short timeout so that Streamlit's StopException can
            # propagate through Python between polls.  Total wait is still
            # bounded; if no audio arrives for 4 s we return a silent frame.
            if self.stop:
                return None
            deadline = 4.0
            step    = 0.05
            elapsed = 0.0
            while elapsed < deadline:
                if self.stop:
                    return None
                try:
                    data = self._audio_queue.get(timeout=step)
                except queue.Empty:
                    elapsed += step
                    continue

                # Sentinel injected by _stop_mic to unblock this loop
                if data is _STOP_SENTINEL:
                    return None

                arr = np.frombuffer(data, dtype=np.int16)
                if custom_channels > 1:
                    arr = np.mean(arr.reshape(-1, custom_channels), axis=1).astype(np.int16)
                if target_rate != custom_rate:
                    new_length = int(len(arr) * target_rate / custom_rate)
                    arr = resample(arr, new_length).astype(np.int16)
                return arr

            return self._silent_frame.copy()

        CustomAudioStream.__init__(
            self,
            open_stream=self._mic_stream.start_stream,
            close_stream=self._stop_mic,
            get_next_frame=get_next_frame,
            window_length_secs=window_length_secs,
            sliding_window_secs=sliding_window_secs,
            sample_rate=target_rate,
        )

    def _stop_mic(self):
        """Stop pyaudio and unblock any pending get_next_frame call."""
        # Put the sentinel first so the polling loop exits immediately
        try:
            self._audio_queue.put_nowait(_STOP_SENTINEL)
        except queue.Full:
            pass
        try:
            if self._mic_stream.is_active():
                self._mic_stream.stop_stream()
            self._mic_stream.close()
        except Exception:
            pass
        try:
            self._pa.terminate()
        except Exception:
            pass