import pyaudio
import librosa
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pcov_kws.streams import SimpleMicStream


CHUNK = 1000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1.5
WAVE_OUTPUT_FILENAME = "voice.wav"

p = pyaudio.PyAudio()

def record_audio():
    try:
        inp_stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

        print("Recording...")  # 调试信息
        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = inp_stream.read(CHUNK)
            print(f"Chunk {i}: {len(data)} bytes")  # 调试信息
            if data:
                frames.append(np.frombuffer(data, dtype=np.int16))
            else:
                print(f"Empty chunk received at iteration {i}")

        inp_stream.stop_stream()
        inp_stream.close()

        if not frames:
            raise ValueError("No audio data was recorded - check your microphone")
        
        return np.concatenate(frames)
    
    except Exception as e:
        print(f"Recording error: {str(e)}")
        if 'inp_stream' in locals():
            inp_stream.stop_stream()
            inp_stream.close()
        raise


def playFrame(inpFrame):
    if inpFrame.dtype != np.float32:
        inpFrame = inpFrame.astype(np.float32) / 32768.0
    
    print(f"Playing frame with dtype: {inpFrame.dtype}, shape: {inpFrame.shape}")
    converterFrame = librosa.resample(inpFrame, orig_sr=16000, target_sr=48000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                         channels=1,
                         rate=48000,
                         output=True,
                         #output_device_index=1
                         )
    #stream = p.open(format = p.get_format_from_width(1), channels = 1, rate = 16000, output = True)
    stream.write(converterFrame.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

# mic_stream = SimpleMicStream(window_length_secs=1.5, sliding_window_secs=1.5, custom_rate=16000, custom_channels=1)

input("Press enter to record and wait for speak now:")

frame = record_audio()
input("Press Enter to play")
playFrame(frame)