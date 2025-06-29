import os
from pcov_kws.streams import SimpleMicStream
from pcov_kws.engine import HotwordDetector, MultiHotwordDetector
from pcov_kws.audio_processing import Resnet50_Arc_loss
from pcov_kws import samples_loc
import random
import numpy as np
import soundfile as sf
import librosa

SAMPLE_RATE = 16000
AUDIO_WINDOW = 1.5 #seconds
AUDIO_LENGTH = int(AUDIO_WINDOW * SAMPLE_RATE)

base_model = Resnet50_Arc_loss()

mycroft_hw = HotwordDetector(
    hotword="mycroft",
    model = base_model,
    reference_file="pcov_kws/sample_refs/resarc/mycroft.json",
    threshold=0.7,
    relaxation_time=2
)

lights_on = HotwordDetector(
    hotword="lights_on",
    model = base_model,
    reference_file="pcov_kws/sample_refs/resarc/lights_on.json",
    threshold=0.7,
    relaxation_time=2    
)


lights_off = HotwordDetector(
    hotword="lights_off",
    model = base_model,
    reference_file="pcov_kws/sample_refs/resarc/lights_off.json",
    threshold=0.7,
    relaxation_time=2    
)

multi_hotword_detector = MultiHotwordDetector(
    [mycroft_hw, lights_on, lights_off],
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

