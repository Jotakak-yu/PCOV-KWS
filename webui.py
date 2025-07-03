import streamlit as st
import os
import time
import numpy as np
from pcov_kws import samples_loc
from pcov_kws.streams import SimpleMicStream
from pcov_kws.audio_utils import create_wav_header
from pcov_kws.engine import HotwordDetector, MultiHotwordDetector
from pcov_kws.audio_processing import (Resnet50_Arc_loss,
                                           TCResNet14, EfficientWord, 
                                           TCResNet14_Arc_loss,
                                           TDResNeXt_SP2_loss
)
from pcov_kws.audio_processing import (
    ModelType,
    MODEL_TYPE_MAPPER
)
import json 
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# --- VAD Setup ---
VAD_ENABLED = False
vad_model = None
utils = None # Initialize utils
try:
    # Silero VAD model loading
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False, # Set True for first time/updates
                                      onnx=True) # Use True if you have onnxruntime installed and prefer it
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils
    VAD_ENABLED = True
    st.sidebar.success("Silero VAD loaded successfully.")
except Exception as e:
    st.sidebar.warning(f"Could not load Silero VAD. VAD disabled. Error: {e}")
# --- End VAD Setup ---

# --- Threshold Persistence Setup ---
THRESHOLDS_FILE = "configs/command_thresholds.json"
WAKE_THRESHOLDS_FILE = "configs/wake_thresholds.json"  # 新增唤醒词阈值文件

# Function to load thresholds
def load_thresholds(filepath, model_name=None):
    try:
        with open(filepath, 'r') as f:
            thresholds = json.load(f)
            # Basic validation: check if it's a dictionary
            if not isinstance(thresholds, dict):
                st.warning("Invalid thresholds file format. Using defaults.")
                return {}
            
            # If model_name is provided, get model-specific thresholds
            if model_name:
                # If model doesn't have thresholds yet, return empty dict
                return thresholds.get(model_name, {})
            return thresholds
    except (FileNotFoundError, json.JSONDecodeError):
        return {} # Return empty dict if file not found or invalid

# Function to save thresholds
def save_thresholds(filepath, thresholds_dict, model_name=None):
    try:
        # Load existing thresholds
        all_thresholds = {}
        try:
            with open(filepath, 'r') as f:
                all_thresholds = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        # If model_name is provided, update only that model's thresholds
        if model_name:
            all_thresholds[model_name] = thresholds_dict
        else:
            all_thresholds = thresholds_dict
            
        with open(filepath, 'w') as f:
            json.dump(all_thresholds, f, indent=4)
    except Exception as e:
        st.error(f"Error saving thresholds: {e}")

# 为唤醒词添加更新阈值的回调函数
def update_wake_threshold(key):
    """唤醒词阈值的回调函数，更新session state并保存到文件。"""
    # 从key中提取唤醒词名称
    wake_word = key.replace("wake_", "")
    # 更新词典和session state
    st.session_state.wake_word_thresholds[wake_word] = st.session_state[key]
    # 保存更新后的词典
    save_thresholds(WAKE_THRESHOLDS_FILE, st.session_state.wake_word_thresholds, model_key)
# --- End Threshold Persistence Setup ---

current_detector = None
command_detector = None
wake_word_detected = False
end_command_words = ['结束', '再见', '拜拜']  # 定义结束词，可以是多个
timeout = 10  # 定义超时时间（秒）
timeout_message = "等待命令词超时，请重新唤醒"  # 定义超时后显示的信息

last_command_time = None  # 记录最后一次收到命令词的时间

# 初始化 session_state
st.session_state.setdefault('recording_step', 0)
st.session_state.setdefault('path', "")
st.session_state.setdefault('word_name', "")
st.session_state.setdefault('record_counter', 0)
st.session_state.setdefault('recordings', [])
st.session_state.setdefault('multi_wake_word_mode', False)
st.session_state.setdefault('use_ans', False)
st.session_state.setdefault('ans_pipeline', None)
st.session_state.setdefault('detector', None)
st.session_state.setdefault('command_detector', None)
st.session_state.setdefault('detector_config', None)

# Model Selection
model_options = {
    'TDResNeXt_SP2_loss': {'model': TDResNeXt_SP2_loss, 'window_length_secs': 1.5, 'default_sliding_window_secs': 0.75, 'default_threshold': 0.81},
    'Resnet_Arc_loss': {'model': Resnet50_Arc_loss, 'window_length_secs': 1.5, 'default_sliding_window_secs': 0.75, 'default_threshold': 0.7},
    'TCResnet14_Arc_loss': {'model': TCResNet14_Arc_loss, 'window_length_secs': 1.0, 'default_sliding_window_secs': 0.5, 'default_threshold': 0.78},
    'EfficientNet': {'model': EfficientWord, 'window_length_secs': 1.0, 'default_sliding_window_secs': 0.5, 'default_threshold': 0.85},
    'TCResNet14': {'model': TCResNet14, 'window_length_secs': 1.0, 'default_sliding_window_secs': 0.5, 'default_threshold': 0.85},
}
chosen_model = st.sidebar.selectbox("切换模型", options=list(model_options.keys()))

window_length_secs = model_options[chosen_model]['window_length_secs']
default_sliding_window_secs = model_options[chosen_model]['default_sliding_window_secs']
default_threshold = model_options[chosen_model]['default_threshold']
sliding_window_secs = st.sidebar.slider('滑动窗口长度（默认为切片长度的一半，若检测效果较差，可适当调高）', min_value=0.0, max_value=window_length_secs, value=default_sliding_window_secs, step=0.01)
base_model = model_options[chosen_model]['model']()

WAKE_WORD_PATH = "wakewords"

# 调整唤醒词
wake_word_options = ['Hi_Siri', 'mycroft', '小爱同学', '小度小度', 'OK_Google']
if st.session_state.multi_wake_word_mode:
    # 在多唤醒词模式下，使用multiselect
    chosen_wake_words = st.sidebar.multiselect('选择唤醒词', options=wake_word_options, default=[wake_word_options[0]])
    if not chosen_wake_words:
        st.sidebar.warning("至少需要选择一个唤醒词")
        chosen_wake_words = [wake_word_options[0]]
else:
    # 在单唤醒词模式下，使用selectbox
    chosen_wake_word = st.sidebar.selectbox('调整唤醒词', options=wake_word_options)
    chosen_wake_words = [chosen_wake_word]

# 添加多唤醒词模式开关
st.sidebar.subheader("唤醒词模式设置")
multi_wake_word_mode = st.sidebar.checkbox("启用多唤醒词模式", value=st.session_state.multi_wake_word_mode)
if multi_wake_word_mode != st.session_state.multi_wake_word_mode:
    st.session_state.multi_wake_word_mode = multi_wake_word_mode
    st.experimental_rerun()

# 调整命令词
command_word_options = ['开灯', '关灯', '上一首', '切歌', '继续播放', '暂停', '开启空调', '关闭空调', '调高音量', '减小音量', '再见']
chosen_command_words = st.sidebar.multiselect('调整命令词', options=command_word_options, default=command_word_options)

# Load existing thresholds for the current model
model_key = chosen_model  # Using the model name as the key
command_thresholds = load_thresholds(THRESHOLDS_FILE, model_key)
# 加载已有的唤醒词阈值
wake_thresholds = load_thresholds(WAKE_THRESHOLDS_FILE, model_key)

# Ensure all currently selected command words have an entry
default_global_threshold = model_options[chosen_model]['default_threshold']
for cmd in chosen_command_words:
    if cmd not in command_thresholds:
        command_thresholds[cmd] = default_global_threshold # Use model default for new words

# Store thresholds in session state to manage updates via callbacks
st.session_state.setdefault('command_thresholds', command_thresholds)
st.session_state.setdefault('current_model', model_key)
# 为唤醒词阈值添加到session state
st.session_state.setdefault('wake_word_thresholds', wake_thresholds)

# If model has changed, update the thresholds
if st.session_state.get('current_model') != model_key:
    st.session_state.command_thresholds = command_thresholds
    st.session_state.wake_word_thresholds = wake_thresholds
    st.session_state.current_model = model_key

def update_threshold(key):
    """Callback function to update session state and save file."""
    # Update the main dictionary from session state widget value
    command_key = key.replace("thresh_", "") # Extract command word from widget key
    st.session_state.command_thresholds[command_key] = st.session_state[key]
    # Save the updated dictionary
    save_thresholds(THRESHOLDS_FILE, st.session_state.command_thresholds, model_key)

# Wake word threshold settings
st.sidebar.subheader("唤醒词触发阈值")
wake_word_thresholds = {}

for wake_word in chosen_wake_words:
    # 为每个唤醒词确保在session state中存在阈值记录
    if wake_word not in st.session_state.wake_word_thresholds:
        st.session_state.wake_word_thresholds[wake_word] = default_global_threshold
    
    # 使用保存的阈值创建滑动条
    wake_word_thresholds[wake_word] = st.sidebar.slider(
        f'{wake_word}',
        min_value=0.5,
        max_value=1.0,
        value=st.session_state.wake_word_thresholds.get(wake_word, default_global_threshold),
        step=0.01,
        key=f"wake_{wake_word}",
        on_change=update_wake_threshold,
        args=(f"wake_{wake_word}",)
    )

# 清理不再使用的唤醒词阈值，仅保留当前选中的
updated_wake_thresholds = {wake_word: st.session_state.wake_word_thresholds.get(wake_word, default_global_threshold)
                          for wake_word in chosen_wake_words}
if updated_wake_thresholds != st.session_state.wake_word_thresholds:
    st.session_state.wake_word_thresholds = updated_wake_thresholds
    save_thresholds(WAKE_THRESHOLDS_FILE, st.session_state.wake_word_thresholds, model_key)

relaxation_time = st.sidebar.slider('间隔时间（过短可能导致检测效果不佳）', min_value=0.1, max_value=5.0, value=2.0, step=0.1)

debounce_time = st.sidebar.slider(
    '触发后去抖时间（秒）',
    min_value=0.1,
    max_value=3.0,
    value=0.5,
    step=0.1,
    help="在一次成功触发后，必须经过此设定的时间才能再次触发。可有效防止单次发音被识别为两次。"
)

# VAD Toggle
use_vad = st.sidebar.checkbox("启用语音活动检测 (VAD)", value=True, disabled=not VAD_ENABLED)
vad_threshold = st.sidebar.slider('VAD 触发阈值', min_value=0.1, max_value=0.9, value=0.3, step=0.05, disabled=not use_vad)

# ANS Toggle
use_ans = st.sidebar.checkbox("启用降噪", value=st.session_state.get('use_ans', False))
if use_ans != st.session_state.use_ans:
    st.session_state.use_ans = use_ans
    st.experimental_rerun()

# Display sliders for individual command words
st.sidebar.subheader("命令词触发阈值")
# Keep track of commands displayed to remove sliders for deselected words later
current_slider_commands = set()
for cmd in chosen_command_words:
    current_slider_commands.add(cmd)
    slider_key = f"thresh_{cmd}"
    st.sidebar.slider(
        f'{cmd}',
        min_value=0.5,
        max_value=1.0,
        # Get value from session state, fallback to default if somehow missing
        value=st.session_state.command_thresholds.get(cmd, default_global_threshold),
        step=0.01,
        key=slider_key, # Unique key for each slider
        on_change=update_threshold, # Callback to save changes
        args=(slider_key,) # Pass the key to the callback
    )

# Optional: Clean up thresholds for commands that were deselected
# Creates a new dict with only the currently selected commands
# This prevents the JSON file from growing indefinitely with old commands
updated_thresholds = {cmd: st.session_state.command_thresholds.get(cmd, default_global_threshold)
                        for cmd in chosen_command_words}
if updated_thresholds != st.session_state.command_thresholds:
    st.session_state.command_thresholds = updated_thresholds
    save_thresholds(THRESHOLDS_FILE, st.session_state.command_thresholds, model_key)

# Reset VAD state if using VAD
if use_vad and vad_model is not None:
    vad_model.reset_states()

# Load ANS model if enabled
if st.session_state.use_ans and st.session_state.ans_pipeline is None:
    with st.spinner("正在加载降噪模型..."):
        try:
            st.session_state.ans_pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model='iic/speech_zipenhancer_ans_multiloss_16k_base'
            )
        except Exception as e:
            st.error(f"加载降噪模型失败: {e}")
            st.session_state.use_ans = False
            st.session_state.ans_pipeline = None
            st.experimental_rerun()

# From model instance find corresponding key
model_key_for_path = None
for key, value in MODEL_TYPE_MAPPER.items():
    if isinstance(base_model, value):
        model_key_for_path = key
        break
# Ensure found corresponding key
if model_key_for_path is None:
    raise ValueError(f'No corresponding key found for model {chosen_model} in MODEL_TYPE_MAPPER')

# Create a config tuple to check if we need to re-create detectors
current_config = (
    model_key,
    tuple(sorted(chosen_wake_words)),
    tuple(sorted(chosen_command_words)),
    relaxation_time,
    tuple(sorted(st.session_state.wake_word_thresholds.items())),
    tuple(sorted(st.session_state.command_thresholds.items()))
)

# If config has changed, or detectors are not initialized, create them
if current_config != st.session_state.detector_config:
    with st.spinner("正在更新检测器配置..."):
        st.session_state.detector_config = current_config
        
        # This stop/start logic might need review depending on how state is managed
        # If switching models/words, need to ensure proper cleanup/re-init
        if current_detector is not None:
            current_detector.stop()

        # Construct lookup reference_file path
        detector_collection = []
        # 1. Add Wake Word Detector(s)
        for wake_word in chosen_wake_words:
            wake_reference_file_path = os.path.join(samples_loc, model_key_for_path, f"{wake_word}.json")
            # 使用session state中的阈值，而非临时字典中的值
            wake_threshold = st.session_state.wake_word_thresholds.get(wake_word, default_global_threshold)
            print(f"Wake Word: {wake_word}, Path: {wake_reference_file_path}, Threshold: {wake_threshold}")
            hw_wake = HotwordDetector(
                hotword=wake_word,
                model=base_model,
                reference_file=wake_reference_file_path,
                threshold=wake_threshold,
                relaxation_time=relaxation_time
            )
            detector_collection.append(hw_wake)

        # 2. Add Command Word Detectors (use individual thresholds)
        command_detector_collection = []
        for hotword in chosen_command_words:
            reference_file_path = os.path.join(samples_loc, model_key_for_path, f"{hotword}.json")
            # Get the specific threshold for this command, fallback to default if needed
            cmd_threshold = st.session_state.command_thresholds.get(hotword, default_global_threshold)
            print(f"Command: {hotword}, Path: {reference_file_path}, Threshold: {cmd_threshold}")

            hw = HotwordDetector(
                hotword=hotword,
                model=base_model,
                reference_file=reference_file_path,
                threshold=cmd_threshold, # Use the command-specific threshold
                relaxation_time=relaxation_time
            )
            command_detector_collection.append(hw)

        # Create wake word detector(s)
        if len(detector_collection) > 1:
            detector = MultiHotwordDetector(
                detector_collection,
                model=base_model,
                relaxation_time=relaxation_time,
                continuous=True
            )
        else:
            detector = detector_collection[0] if detector_collection else None

        # Create command word detector
        if len(command_detector_collection) > 0:
            command_detector = MultiHotwordDetector(
                command_detector_collection,
                model=base_model,
                relaxation_time=relaxation_time,
                continuous=True
            )
        else:
            command_detector = None  # No commands selected or loaded
        
        st.session_state.detector = detector
        st.session_state.command_detector = command_detector
        st.info("检测器已更新。")

detector = st.session_state.detector
command_detector = st.session_state.command_detector

if not detector:
    st.warning("唤醒词检测器未初始化，请选择唤醒词。")
    st.stop()

current_detector = detector  # Start with the wake word detector
if current_detector:
    current_detector.start()

import pyaudio

# 获取当前默认输入设备的信息
p = pyaudio.PyAudio()
default_device_index = p.get_default_input_device_info()['index']
device_info = p.get_device_info_by_index(default_device_index)
mic_sample_rate = int(device_info['defaultSampleRate'])
mic_channels = int(device_info['maxInputChannels'])
p.terminate()

# 打印采样率和通道数，便于调试
print(f"当前麦克风采样率: {mic_sample_rate}, 通道数: {mic_channels}")

# 使用获取到的采样率和通道数初始化 SimpleMicStream
mic_stream = SimpleMicStream(
    window_length_secs=window_length_secs,
    sliding_window_secs=sliding_window_secs,
    custom_channels=mic_channels,
    custom_rate=mic_sample_rate,
    custom_device_index=default_device_index
)
mic_stream.start_stream()

st.write(f"当前唤醒词：  {', '.join([d.hotword for d in detector.detector_collection]) if isinstance(detector, MultiHotwordDetector) else detector.hotword}")
res_placeholder = st.empty()
instruction_placeholder = st.empty()  # Create new placeholder for displaying instructions
last_trigger_timestamp = 0 # Initialize timestamp for debouncing

while True:
    if not current_detector.is_running:
        break
    frame = mic_stream.getFrame()
    if frame is None:
        st.write("The detector has stopped.")
        break

    # --- ANS processing ---
    if st.session_state.use_ans and st.session_state.ans_pipeline:
        try:
            frame_bytes = frame.tobytes()
            wav_chunk = create_wav_header(frame_bytes, sample_rate=16000, num_channels=1, bits_per_sample=16)
            
            result = st.session_state.ans_pipeline(wav_chunk)
            output_pcm = result['output_pcm']
            
            processed_frame = np.frombuffer(output_pcm, dtype=np.int16)
            
            if len(processed_frame) != len(frame):
                # Pad or truncate to maintain frame size for downstream components
                new_frame = np.zeros_like(frame)
                copy_len = min(len(frame), len(processed_frame))
                new_frame[:copy_len] = processed_frame[:copy_len]
                frame = new_frame
            else:
                frame = processed_frame
        except Exception as e:
            st.warning(f"降噪处理失败: {e}")


    # --- VAD Check ---
    process_frame = True # Default to processing
    if use_vad and VAD_ENABLED and vad_model is not None:
        try:
            # Ensure frame is float32 numpy array for VAD processing
            if frame.dtype == np.int16:
                frame_float32 = frame.astype(np.float32) / 32768.0
            else:
                frame_float32 = frame.astype(np.float32)

            frame_float32 = np.clip(frame_float32, -1.0, 1.0)

            sample_rate = 16000 # Silero VAD standard sample rate
            chunk_size = 512    # VAD model expects audio chunks of this size for 16k SR

            contains_speech = False
            # Iterate over the larger frame in smaller, VAD-compatible chunks
            for i in range(0, len(frame_float32), chunk_size):
                chunk = frame_float32[i : i + chunk_size]

                # If the last chunk is smaller than the required size, skip it
                if len(chunk) < chunk_size:
                    continue

                audio_tensor = torch.from_numpy(chunk)
                speech_prob = vad_model(audio_tensor, sample_rate).item()

                if speech_prob > vad_threshold:
                    contains_speech = True
                    break  # Speech detected, no need to check further chunks

            if not contains_speech:
                process_frame = False

        except Exception as e:
            st.warning(f"Error during VAD processing: {e}")
            # Fallback to processing the frame if VAD fails
            process_frame = True
    # --- End VAD Check ---

    if process_frame: # Only process if VAD allows or is disabled
        if not wake_word_detected:
            if current_detector: # Check if detector exists
                match_found = False
                result = None
                detected_hotword = ""
                # For MultiHotwordDetector, use findBestMatch instead of scoreFrame
                if isinstance(current_detector, MultiHotwordDetector):
                    result = current_detector.findBestMatch(frame)
                    if result and result[0] is not None:
                        match_found = True
                        detected_hotword = result[0].hotword
                else:
                    # Single wake word detector
                    result = current_detector.scoreFrame(frame)
                    if result is not None and result["match"]:
                        match_found = True
                        detected_hotword = current_detector.hotword
                
                if match_found:
                    current_time = time.time()
                    if current_time - last_trigger_timestamp > debounce_time:
                        last_trigger_timestamp = current_time
                        print(result)
                        wake_word_detected = True
                        
                        if isinstance(current_detector, MultiHotwordDetector):
                            instruction_placeholder.text(f"你好，唤醒词 {detected_hotword} 已激活，请告诉我要做什么")
                        else:
                            instruction_placeholder.text("你好，请告诉我要做什么")

                        # Switch detector only if command_detector is valid
                        if command_detector:
                            current_detector = command_detector
                            current_detector.start()
                            last_command_time = time.time()
                            # Reset VAD state after wake word is detected
                            if use_vad and vad_model is not None:
                                vad_model.reset_states()
                        else:
                            instruction_placeholder.text("唤醒成功，但未配置命令词。")

        else: # Wake word detected, listening for commands
            if current_detector and command_detector: # Ensure we are in command mode
                result = current_detector.findBestMatch(frame)
                if result and result[0] is not None: # Check if findBestMatch returned a valid result
                    current_time = time.time()
                    if current_time - last_trigger_timestamp > debounce_time:
                        last_trigger_timestamp = current_time
                        print(result)
                        # Extract command name more robustly
                        detected_command_detector = result[0]
                        detected_command = detected_command_detector.hotword

                        if detected_command in end_command_words:  # Check for end command
                            wake_word_detected = False
                            instruction_placeholder.empty()
                            res_placeholder.empty()
                            instruction_placeholder.text("已结束，请重新唤醒")
                            # Switch back to wake word detector
                            current_detector = detector # detector holds the wake word detector
                            # Start the wake word detector again - This is crucial
                            current_detector.start()
                            # Reset VAD state after conversation ends
                            if use_vad and vad_model is not None:
                                vad_model.reset_states()
                            continue
                        else:
                            res_placeholder.text(f'{detected_command}, Confidence {result[1]:0.4f}')
                            last_command_time = time.time()  # Update time on valid command
                # Explicitly check for timeout regardless of detection result
                elif last_command_time is not None and time.time() - last_command_time > timeout:
                    # Log timeout for debugging
                    print(f"Command timeout triggered: {time.time() - last_command_time:.2f}s > {timeout}s")
                    wake_word_detected = False
                    instruction_placeholder.empty()
                    res_placeholder.text(timeout_message)
                    # Switch back to wake word detector
                    current_detector = detector # detector holds the wake word detector
                    # Start the wake word detector again - This is crucial
                    current_detector.start()
                    # Reset VAD state on timeout
                    if use_vad and vad_model is not None:
                        vad_model.reset_states()
                    # Reset last command time to avoid immediate re-triggering
                    last_command_time = None
    
    # Add timeout check outside of process_frame condition to ensure it's always checked
    elif wake_word_detected and last_command_time is not None and time.time() - last_command_time > timeout:
        # Log timeout for debugging
        print(f"Timeout triggered outside process_frame: {time.time() - last_command_time:.2f}s > {timeout}s")
        wake_word_detected = False
        instruction_placeholder.empty()
        res_placeholder.text(timeout_message)
        # Switch back to wake word detector
        current_detector = detector
        current_detector.start()
        # Reset VAD state on timeout
        if use_vad and vad_model is not None:
            vad_model.reset_states()
        # Reset last command time
        last_command_time = None

# Cleanup when loop exits (e.g., script stopped)
if current_detector is not None:
    # Check if stop method exists before calling
    if hasattr(current_detector, 'stop') and callable(current_detector.stop):
        current_detector.stop()