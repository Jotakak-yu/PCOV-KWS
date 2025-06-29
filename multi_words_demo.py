import os
from pcov_kws.streams import SimpleMicStream
from pcov_kws.engine import HotwordDetector, MultiHotwordDetector
from pcov_kws.audio_processing import Resnet50_Arc_loss, TDResNeXt_SP2_loss
from pcov_kws import samples_loc
from pcov_kws.audio_processing import (
    ModelType,
    MODEL_TYPE_MAPPER
)
import random
import numpy as np
import soundfile as sf
import librosa

SAMPLE_RATE = 16000
AUDIO_WINDOW = 1.5 #seconds
AUDIO_LENGTH = int(AUDIO_WINDOW * SAMPLE_RATE)

base_model = TDResNeXt_SP2_loss()

model_key = None
for key, value in MODEL_TYPE_MAPPER.items():
    if isinstance(base_model, value):
        model_key = key
        break
hotwords = ["mycroft", "lights_on", "lights_off"]

# 可选参数配置，方便后续扩展
hotword_params = {
    "threshold": 0.8,
    "relaxation_time": 1.2
}

# 动态创建所有 HotwordDetector 实例
hotword_detectors = []
for hotword in hotwords:
    detector = HotwordDetector(
        hotword=hotword,
        model=base_model,
        reference_file=os.path.join(samples_loc, model_key, f"{hotword}.json"),
        **hotword_params
    )
    hotword_detectors.append(detector)

multi_hotword_detector = MultiHotwordDetector(
    hotword_detectors,
    model=base_model,
    continuous=True,
)
multi_hotword_detector.start()

mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=0.75
)
mic_stream.start_stream()

#print("Say ", mycroft_hw.hotword)
print("Say one hotword among :", " ".join([x.hotword for x in multi_hotword_detector.detector_collection]))
while True :
    frame = mic_stream.getFrame()
    best_match = multi_hotword_detector.findBestMatch(frame)
    if best_match[0]!=None :
        print(best_match[0].hotword, best_match[1])

