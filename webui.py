import streamlit as st
import os
import time
import atexit
import numpy as np
from pcov_kws import samples_loc
from pcov_kws.streams import SimpleMicStream
from pcov_kws.engine import HotwordDetector, MultiHotwordDetector
from pcov_kws.audio_processing import Resnet50_Arc_loss, TDResNeXt_SP2_loss
from pcov_kws.audio_processing import MODEL_TYPE_MAPPER
import json
import torch
import pyaudio

# Streamlit uses ScriptControlException (BaseException subclass) for flow control.
# RerunException must propagate freely; only StopException should trigger cleanup.
try:
    from streamlit.runtime.scriptrunner.script_runner import StopException
except ImportError:
    # Fallback for different Streamlit versions
    StopException = None

# Early session-state initialisation (must run before any widget or callback
# that references these keys, including across st.rerun cycles).
st.session_state.setdefault('multi_wake_word_mode', False)
st.session_state.setdefault('recording_step', 0)
st.session_state.setdefault('path', "")
st.session_state.setdefault('word_name', "")
st.session_state.setdefault('record_counter', 0)
st.session_state.setdefault('recordings', [])
st.session_state.setdefault('detector', None)
st.session_state.setdefault('command_detector', None)
st.session_state.setdefault('detector_config', None)

# ---------------------------------------------------------------------------
# VAD Setup
# ---------------------------------------------------------------------------
VAD_SAMPLE_RATE = 16000  # Silero VAD expected sample rate
VAD_CHUNK_SIZE  = 512    # Required chunk size at 16 kHz

VAD_ENABLED = False
vad_model = None
try:
    vad_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=True
    )
    VAD_ENABLED = True
    st.sidebar.success("Silero VAD loaded successfully.")
except Exception as e:
    st.sidebar.warning(f"Could not load Silero VAD. VAD disabled. Error: {e}")


def check_vad(frame: np.ndarray, threshold: float) -> bool:
    """Return True if any VAD chunk in *frame* has speech probability above *threshold*."""
    audio = frame.astype(np.float32) / 32768.0 if frame.dtype == np.int16 \
        else np.clip(frame.astype(np.float32), -1.0, 1.0)

    n_chunks = len(audio) // VAD_CHUNK_SIZE
    if n_chunks == 0:
        return False

    for chunk in audio[: n_chunks * VAD_CHUNK_SIZE].reshape(n_chunks, VAD_CHUNK_SIZE):
        if vad_model(torch.from_numpy(chunk), VAD_SAMPLE_RATE).item() > threshold:
            return True
    return False

# ---------------------------------------------------------------------------
# Threshold persistence
# ---------------------------------------------------------------------------
THRESHOLDS_FILE      = "configs/command_thresholds.json"
WAKE_THRESHOLDS_FILE = "configs/wake_thresholds.json"


def load_thresholds(filepath, model_name=None):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            st.warning("Invalid thresholds file format. Using defaults.")
            return {}
        return data.get(model_name, {}) if model_name else data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_thresholds(filepath, thresholds_dict, model_name=None):
    try:
        all_thresholds = {}
        try:
            with open(filepath, 'r') as f:
                all_thresholds = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if model_name:
            all_thresholds[model_name] = thresholds_dict
        else:
            all_thresholds = thresholds_dict
        with open(filepath, 'w') as f:
            json.dump(all_thresholds, f, indent=4)
    except Exception as e:
        st.error(f"Error saving thresholds: {e}")


def update_wake_threshold(key):
    wake_word = key.replace("wake_", "")
    st.session_state.wake_word_thresholds[wake_word] = st.session_state[key]
    save_thresholds(WAKE_THRESHOLDS_FILE, st.session_state.wake_word_thresholds, model_key)


def update_threshold(key):
    command_key = key.replace("thresh_", "")
    st.session_state.command_thresholds[command_key] = st.session_state[key]
    save_thresholds(THRESHOLDS_FILE, st.session_state.command_thresholds, model_key)

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
current_detector   = None
command_detector   = None
wake_word_detected = False
end_command_words  = ['结束', '再见', '拜拜']
timeout            = 10
timeout_message    = "Command timeout. Please wake the device again."
last_command_time  = None

# ---------------------------------------------------------------------------
# Sidebar – model & detection settings
# ---------------------------------------------------------------------------
model_options = {
    'TDResNeXt_SP2_loss': {
        'model': TDResNeXt_SP2_loss,
        'window_length_secs': 1.5,
        'default_sliding_window_secs': 0.75,
        'default_threshold': 0.81
    },
    'Resnet_Arc_loss': {
        'model': Resnet50_Arc_loss,
        'window_length_secs': 1.5,
        'default_sliding_window_secs': 0.75,
        'default_threshold': 0.7
    },
}
chosen_model = st.sidebar.selectbox("Model", options=list(model_options.keys()))

window_length_secs          = model_options[chosen_model]['window_length_secs']
default_sliding_window_secs = model_options[chosen_model]['default_sliding_window_secs']
default_threshold           = model_options[chosen_model]['default_threshold']

sliding_window_secs = st.sidebar.slider(
    'Sliding window (s) — increase if detection is poor',
    min_value=0.0, max_value=window_length_secs,
    value=default_sliding_window_secs, step=0.01
)
base_model = model_options[chosen_model]['model']()

# Wake word selection
wake_word_options = ['Hi_Siri', 'OK_Google']
if st.session_state.multi_wake_word_mode:
    default_wake = wake_word_options[:2] if len(wake_word_options) >= 2 else wake_word_options
    chosen_wake_words = st.sidebar.multiselect(
        'Wake words', options=wake_word_options, default=default_wake
    )
    if len(chosen_wake_words) < 2:
        st.sidebar.warning("Multi-wake-word mode requires at least 2 selections.")
        chosen_wake_words = wake_word_options[:2] if len(wake_word_options) >= 2 else wake_word_options
else:
    chosen_wake_word  = st.sidebar.selectbox('Wake word', options=wake_word_options)
    chosen_wake_words = [chosen_wake_word]

st.sidebar.subheader("Wake word mode")
multi_wake_word_mode = st.sidebar.checkbox(
    "Enable multi-wake-word mode", value=st.session_state.multi_wake_word_mode
)
if multi_wake_word_mode != st.session_state.multi_wake_word_mode:
    st.session_state.multi_wake_word_mode = multi_wake_word_mode
    st.rerun()

# Command word selection
command_word_options = ['lights_on', 'lights_off']
chosen_command_words = st.sidebar.multiselect(
    'Command words', options=command_word_options, default=command_word_options
)

# ---------------------------------------------------------------------------
# Threshold management
# ---------------------------------------------------------------------------
model_key = chosen_model
command_thresholds = load_thresholds(THRESHOLDS_FILE, model_key)
wake_thresholds    = load_thresholds(WAKE_THRESHOLDS_FILE, model_key)

default_global_threshold = model_options[chosen_model]['default_threshold']
for cmd in chosen_command_words:
    if cmd not in command_thresholds:
        command_thresholds[cmd] = default_global_threshold

st.session_state.setdefault('command_thresholds', command_thresholds)
st.session_state.setdefault('current_model', model_key)
st.session_state.setdefault('wake_word_thresholds', wake_thresholds)

# Refresh thresholds when model changes
if st.session_state.get('current_model') != model_key:
    st.session_state.command_thresholds   = command_thresholds
    st.session_state.wake_word_thresholds = wake_thresholds
    st.session_state.current_model        = model_key

# Wake word threshold sliders
st.sidebar.subheader("Wake word thresholds")
wake_word_thresholds = {}
for wake_word in chosen_wake_words:
    if wake_word not in st.session_state.wake_word_thresholds:
        st.session_state.wake_word_thresholds[wake_word] = default_global_threshold
    wake_word_thresholds[wake_word] = st.sidebar.slider(
        wake_word, min_value=0.5, max_value=1.0,
        value=st.session_state.wake_word_thresholds.get(wake_word, default_global_threshold),
        step=0.01, key=f"wake_{wake_word}",
        on_change=update_wake_threshold, args=(f"wake_{wake_word}",)
    )

# Prune stale wake word entries
updated_wake_thresholds = {
    ww: st.session_state.wake_word_thresholds.get(ww, default_global_threshold)
    for ww in chosen_wake_words
}
if updated_wake_thresholds != st.session_state.wake_word_thresholds:
    st.session_state.wake_word_thresholds = updated_wake_thresholds
    save_thresholds(WAKE_THRESHOLDS_FILE, st.session_state.wake_word_thresholds, model_key)

relaxation_time = st.sidebar.slider(
    'Relaxation time (s) — too short may cause missed detections',
    min_value=0.1, max_value=5.0, value=2.0, step=0.1
)
debounce_time = st.sidebar.slider(
    'Debounce time (s)',
    min_value=0.5, max_value=5.0, value=2.0, step=0.1,
    help="Minimum interval between successive triggers. Should be >= relaxation time to prevent double-firing from overlapping sliding windows."
)
# Effective debounce is at least as long as relaxation to guarantee no double triggers
effective_debounce = max(debounce_time, relaxation_time)

# VAD controls
use_vad       = st.sidebar.checkbox("Enable Voice Activity Detection (VAD)", value=True, disabled=not VAD_ENABLED)
vad_threshold = st.sidebar.slider(
    'VAD threshold', min_value=0.1, max_value=0.9, value=0.3, step=0.05, disabled=not use_vad
)

# Command word threshold sliders
st.sidebar.subheader("Command word thresholds")
for cmd in chosen_command_words:
    slider_key = f"thresh_{cmd}"
    st.sidebar.slider(
        cmd, min_value=0.5, max_value=1.0,
        value=st.session_state.command_thresholds.get(cmd, default_global_threshold),
        step=0.01, key=slider_key,
        on_change=update_threshold, args=(slider_key,)
    )

# Prune stale command threshold entries
updated_thresholds = {
    cmd: st.session_state.command_thresholds.get(cmd, default_global_threshold)
    for cmd in chosen_command_words
}
if updated_thresholds != st.session_state.command_thresholds:
    st.session_state.command_thresholds = updated_thresholds
    save_thresholds(THRESHOLDS_FILE, st.session_state.command_thresholds, model_key)

if use_vad and vad_model is not None:
    vad_model.reset_states()

# ---------------------------------------------------------------------------
# Resolve model path key
# ---------------------------------------------------------------------------
model_key_for_path = next(
    (k for k, v in MODEL_TYPE_MAPPER.items() if isinstance(base_model, v)), None
)
if model_key_for_path is None:
    raise ValueError(f'No path key found for model {chosen_model} in MODEL_TYPE_MAPPER')

# ---------------------------------------------------------------------------
# Detector (re-)initialisation — only when config changes
# ---------------------------------------------------------------------------
current_config = (
    model_key,
    tuple(sorted(chosen_wake_words)),
    tuple(sorted(chosen_command_words)),
    relaxation_time,
    tuple(sorted(st.session_state.wake_word_thresholds.items())),
    tuple(sorted(st.session_state.command_thresholds.items()))
)

if current_config != st.session_state.detector_config:
    with st.spinner("Updating detector configuration..."):
        st.session_state.detector_config = current_config

        if current_detector is not None:
            current_detector.stop()

        # Build wake-word detectors
        detector_collection = []
        for ww in chosen_wake_words:
            ref_path  = os.path.join(samples_loc, model_key_for_path, f"{ww}.json")
            threshold = st.session_state.wake_word_thresholds.get(ww, default_global_threshold)
            print(f"Wake word: {ww}, path: {ref_path}, threshold: {threshold}")
            detector_collection.append(HotwordDetector(
                hotword=ww, model=base_model,
                reference_file=ref_path,
                threshold=threshold, relaxation_time=relaxation_time
            ))

        # Build command-word detectors
        command_detector_collection = []
        for hw in chosen_command_words:
            ref_path  = os.path.join(samples_loc, model_key_for_path, f"{hw}.json")
            threshold = st.session_state.command_thresholds.get(hw, default_global_threshold)
            print(f"Command: {hw}, path: {ref_path}, threshold: {threshold}")
            command_detector_collection.append(HotwordDetector(
                hotword=hw, model=base_model,
                reference_file=ref_path,
                threshold=threshold, relaxation_time=relaxation_time
            ))

        detector = (
            MultiHotwordDetector(detector_collection, model=base_model,
                                 relaxation_time=relaxation_time, continuous=True)
            if len(detector_collection) > 1
            else (detector_collection[0] if detector_collection else None)
        )
        command_detector = (
            MultiHotwordDetector(command_detector_collection, model=base_model,
                                 relaxation_time=relaxation_time, continuous=True)
            if len(command_detector_collection) > 1
            else (command_detector_collection[0] if command_detector_collection else None)
        )

        st.session_state.detector         = detector
        st.session_state.command_detector = command_detector
        st.info("Detector configuration updated.")

detector         = st.session_state.detector
command_detector = st.session_state.command_detector

if not detector:
    st.warning("Wake-word detector not initialised. Please select a wake word.")
    st.stop()

current_detector = detector
current_detector.start()

# ---------------------------------------------------------------------------
# Microphone stream setup
# ---------------------------------------------------------------------------
p = pyaudio.PyAudio()
default_device_index = p.get_default_input_device_info()['index']
device_info          = p.get_device_info_by_index(default_device_index)
mic_sample_rate      = int(device_info['defaultSampleRate'])
mic_channels         = int(device_info['maxInputChannels'])
p.terminate()

print(f"Microphone sample rate: {mic_sample_rate}, channels: {mic_channels}")

mic_stream = SimpleMicStream(
    window_length_secs=window_length_secs,
    sliding_window_secs=sliding_window_secs,
    custom_channels=mic_channels,
    custom_rate=mic_sample_rate,
    custom_device_index=default_device_index
)
mic_stream.start_stream()

active_wake_words = (
    ', '.join(d.hotword for d in detector.detector_collection)
    if isinstance(detector, MultiHotwordDetector)
    else detector.hotword
)
st.write(f"Active wake word(s): {active_wake_words}")

res_placeholder         = st.empty()
instruction_placeholder = st.empty()
last_trigger_timestamp  = 0


def _reset_to_wake_mode():
    """Switch back to wake-word listening and reset VAD state."""
    global wake_word_detected, current_detector, last_command_time, last_trigger_timestamp
    wake_word_detected = False
    last_command_time  = None
    current_detector   = detector
    current_detector.start()
    # Prevent overlapping audio from immediately triggering the wake word
    # right after switching back from command mode.
    now = time.time()
    last_trigger_timestamp = now
    current_detector.reset_activation_timer(now)
    if use_vad and vad_model is not None:
        vad_model.reset_states()


def _cleanup():
    """Release microphone and detector resources."""
    try:
        mic_stream.stop = True   # signal getFrame() to return None immediately
        mic_stream.close_stream()
    except Exception:
        pass
    try:
        if current_detector is not None and hasattr(current_detector, 'stop'):
            current_detector.stop()
    except Exception:
        pass


atexit.register(_cleanup)

# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------
try:
    while True:
        if not current_detector.is_running:
            break
        frame = mic_stream.getFrame()
        if frame is None:
            st.write("The detector has stopped.")
            break

        # VAD gate: skip frames that contain no speech
        if use_vad and VAD_ENABLED and vad_model is not None:
            try:
                process_frame = check_vad(frame, vad_threshold)
            except Exception as e:
                st.warning(f"VAD error: {e}")
                process_frame = True
        else:
            process_frame = True

        # Timeout must be checked even when VAD suppresses the frame
        if wake_word_detected and last_command_time is not None:
            if time.time() - last_command_time > timeout:
                print(f"Command timeout: {time.time() - last_command_time:.2f}s")
                instruction_placeholder.empty()
                res_placeholder.text(timeout_message)
                _reset_to_wake_mode()
                continue

        if not process_frame:
            continue

        if not wake_word_detected:
            # --- Wake-word detection ---
            match_found      = False
            detected_hotword = ""
            if isinstance(current_detector, MultiHotwordDetector):
                result = current_detector.findBestMatch(frame)
                if result and result[0] is not None:
                    match_found      = True
                    detected_hotword = result[0].hotword
            else:
                result = current_detector.scoreFrame(frame)
                if result is not None and result["match"]:
                    match_found      = True
                    detected_hotword = current_detector.hotword

            if match_found:
                now = time.time()
                if now - last_trigger_timestamp > effective_debounce:
                    last_trigger_timestamp = now
                    print(result)
                    wake_word_detected = True
                    res_placeholder.empty()
                    instruction_placeholder.text(
                        f"Wake word '{detected_hotword}' detected. What can I do for you?"
                    )
                    if command_detector:
                        current_detector  = command_detector
                        current_detector.start()
                        # Prevent the overlapping audio from immediately triggering
                        # a command match right after wake-word detection.
                        current_detector.reset_activation_timer(now)
                        last_trigger_timestamp = now
                        last_command_time = time.time()
                        if use_vad and vad_model is not None:
                            vad_model.reset_states()
                    else:
                        instruction_placeholder.text("Wake word detected, but no command words configured.")

        else:
            # --- Command-word detection ---
            if current_detector and command_detector:
                cmd_match_found = False
                cmd_result = None
                detected_command = ""
                if isinstance(current_detector, MultiHotwordDetector):
                    cmd_result = current_detector.findBestMatch(frame)
                    if cmd_result and cmd_result[0] is not None:
                        cmd_match_found = True
                        detected_command = cmd_result[0].hotword
                else:
                    cmd_result = current_detector.scoreFrame(frame)
                    if cmd_result is not None and cmd_result["match"]:
                        cmd_match_found = True
                        detected_command = current_detector.hotword

                if cmd_match_found:
                    now = time.time()
                    if now - last_trigger_timestamp > effective_debounce:
                        last_trigger_timestamp = now
                        print(cmd_result)

                        if detected_command in end_command_words:
                            instruction_placeholder.empty()
                            res_placeholder.empty()
                            instruction_placeholder.text("Session ended. Please wake the device again.")
                            _reset_to_wake_mode()
                            continue
                        else:
                            confidence = (
                                cmd_result[1] if isinstance(cmd_result, tuple)
                                else cmd_result.get("confidence", 0.0)
                            )
                            instruction_placeholder.empty()
                            res_placeholder.text(f'{detected_command}, Confidence {confidence:0.4f}')
                            last_command_time = time.time()

except KeyboardInterrupt:
    print("Shutting down (KeyboardInterrupt)...")
except SystemExit:
    print("Shutting down (SystemExit)...")
except Exception:
    # Catch unexpected runtime errors but NOT Streamlit's ScriptControlException
    # (RerunException / StopException inherit from BaseException, not Exception).
    print("Shutting down (unexpected error)...")
finally:
    _cleanup()