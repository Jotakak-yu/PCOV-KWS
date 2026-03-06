import os
import numpy as np
import random
import onnxruntime as rt
from pcov_kws.audio_utils import logfbank, compute_fbank_kaldi_native

LIB_FOLDER_LOCATION = os.path.dirname(os.path.realpath(__file__))

class ModelRawBackend:
    def __init__(self):
        self.window_length = None
        self.window_frames = None
        pass

    def _randomCrop(self, x: np.array, length=16000) -> np.array:
        assert (x.shape[0] > self.window_frames)
        frontBits = random.randint(0, x.shape[0] - length)
        return x[frontBits:frontBits + length]

    def _addPadding(self, x: np.array, length=16000) -> np.array:
        assert (x.shape[0] < length)
        bitCountToBeAdded = length - x.shape[0]
        frontBits = random.randint(0, bitCountToBeAdded)
        new_x = np.append(np.zeros(frontBits), x)
        new_x = np.append(new_x, np.zeros(bitCountToBeAdded - frontBits))
        return new_x

    def _removeExistingPadding(self, x: np.array) -> np.array:
        lastZeroBitBeforeAudio = 0
        firstZeroBitAfterAudio = len(x)
        for i in range(len(x)):
            if x[i] == 0:
                lastZeroBitBeforeAudio = i
            else:
                break
        for i in range(len(x) - 1, 1, -1):
            if x[i] == 0:
                firstZeroBitAfterAudio = i
            else:
                break
        return x[lastZeroBitBeforeAudio:firstZeroBitAfterAudio]

    def fixPaddingIssues(self, x: np.array) -> np.array:
        x = self._removeExistingPadding(x)
        if (x.shape[0] > self.window_frames):
            return self._randomCrop(x, length=self.window_frames)
        elif (x.shape[0] < self.window_frames):
            return self._addPadding(x, length=self.window_frames)
        else:
            return x

    def scoreVector(self, inp_vector: np.array, embeddings: np.array) -> np.array:
        raise NotImplementedError("Vector scoring attempted on raw model backend")

    def audioToVector(self, inpAudio: np.array) -> np.array:
        raise NotImplementedError("Vector Convertion on raw model backend invoked")

class Resnet50_Arc_loss(ModelRawBackend):
    def __init__(self):

        self.window_length = 1.5
        self.window_frames = int(self.window_length * 16000)

        self.onnx_sess = rt.InferenceSession(
            os.path.join(
                LIB_FOLDER_LOCATION, "models/pt_onnx/resnet50_quant_dynamic.onnx"),
            sess_options=rt.SessionOptions(),
            providers=["CPUExecutionProvider"]
        )

        self.input_name: str = self.onnx_sess.get_inputs()[0].name
        self.output_name: str = self.onnx_sess.get_outputs()[0].name

        self.audioToVector(np.float32(np.zeros(self.window_frames, )))  # warmup inference

    def compute_logfbank_features(self, inpAudio: np.array) -> np.array:
        """
        This assumes a mono channel input
        """
        return logfbank(
            inpAudio,
            samplerate=16000,
            winlen=0.025,
            winstep=0.01,
            nfilt=64,
            nfft=512,
            preemph=0.0
        )

    def scoreVector(self, inp_vector: np.array, embeddings: np.array) -> np.array:
        cosine_similarity = np.matmul(embeddings, inp_vector.T)
        confidence_scores = (cosine_similarity + 1) / 2
        return confidence_scores.max()

    def audioToVector(self, inpAudio: np.array) -> np.array:
        assert inpAudio.shape == (self.window_frames,)  # 1.5 sec long window
        features = self.compute_logfbank_features(inpAudio)

        output = self.onnx_sess.run(
            [self.output_name],
            {
                self.input_name: np.float32(
                    np.expand_dims(
                        features,
                        axis=(0, 1)  # adding channel and batch dimension
                    )
                )
            }
        )[0]

        return output

class TDResNeXt_SP2_loss(ModelRawBackend):
    def __init__(self):
        self.window_length = 1.5
        self.window_frames = int(self.window_length * 16000)

        self.onnx_sess = rt.InferenceSession(
            os.path.join(
                LIB_FOLDER_LOCATION, "models/pt_onnx/tdsp2_quant_dynamic.onnx"),
            sess_options=rt.SessionOptions(),
            providers=["CPUExecutionProvider"]
        )

        self.input_name: str = self.onnx_sess.get_inputs()[0].name
        self.output_name: str = self.onnx_sess.get_outputs()[0].name

        self.audioToVector(np.float32(np.zeros(self.window_frames, )))  # warmup inference

    def compute_logfbank_features(self, inpAudio: np.array) -> np.array:
        """
        This assumes a mono channel input
        """
        return compute_fbank_kaldi_native(inpAudio, 16000)

    def scoreVector(self, inp_vector: np.array, embeddings: np.array) -> np.array:
        cosine_similarity = np.matmul(embeddings, inp_vector.T)
        confidence_scores = (cosine_similarity + 1) / 2
        return confidence_scores.max()

    def audioToVector(self, inpAudio: np.array) -> np.array:
        assert inpAudio.shape == (self.window_frames,)  # 1.0 sec long window
        features = self.compute_logfbank_features(inpAudio)

        output = self.onnx_sess.run(
            [self.output_name],
            {
                self.input_name: np.float32(
                    np.expand_dims(
                        features,
                        axis=(0, 1)  # adding channel and batch dimension
                    )
                )
            }
        )[0]

        return output

from enum import Enum
class ModelType(str, Enum):
    resarc = "resarc"
    tdsp2 = "tdsp2"

MODEL_TYPE_MAPPER = {
    "resarc": Resnet50_Arc_loss,
    "tdsp2": TDResNeXt_SP2_loss,
}
