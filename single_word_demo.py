import os
from pcov_kws.streams import SimpleMicStream
from pcov_kws.engine import HotwordDetector

from pcov_kws.audio_processing import TDResNeXt_SP2_loss, Resnet50_Arc_loss

from pcov_kws import samples_loc
from pcov_kws.audio_processing import (
    ModelType,
    MODEL_TYPE_MAPPER
)
base_model = TDResNeXt_SP2_loss()

# Find the corresponding key in reverse from the model instance
model_key = None
for key, value in MODEL_TYPE_MAPPER.items():
    if isinstance(base_model, value):
        model_key = key
        break

# Make sure find the corresponding key
if model_key is None:
    raise ValueError(f'No corresponding key found for model {base_model} in MODEL_TYPE_MAPPER')

hotword="Hi_Siri"

reference_file = os.path.join(samples_loc, model_key, f"{hotword}.json")
print(reference_file)
hisiri_hw = HotwordDetector(
    hotword=hotword,
    model = base_model,
    reference_file = os.path.join(samples_loc, model_key, f"{hotword}.json"),
    threshold=0.8,
    relaxation_time=1.2
)

mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=0.75, #一般为window_length_secs的1/2
)
  
mic_stream.start_stream()

print(f"Say {hotword} ")
while True :
    frame = mic_stream.getFrame()
    hisiri_hw.start()
    result = hisiri_hw.scoreFrame(frame)
    if result==None :
        #no voice activity
        continue
    if(result["match"]):
        print("Wakeword uttered",result["confidence"])
