import tflite_runtime.interpreter as tflite
import os
import numpy as np
import random
import onnxruntime as rt
from pcov_kws.audio_utils import logfbank, compute_fbank_kaldi_native
# from Quant_fbank import optimized_mel_calculation_graph

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

class TCResNet14(ModelRawBackend):
    def __init__(self):
        self.window_length = 1.0  # 1 second
        self.window_frames = int(self.window_length * 16000)

        self.logmelcalc_interpreter = tflite.Interpreter(
            model_path=os.path.join(LIB_FOLDER_LOCATION, "models/first_iteration_siamese/#melcalc.tflite"
                                    )
        )
        self.logmelcalc_interpreter.allocate_tensors()
        
        self.input_index = self.logmelcalc_interpreter.get_input_details()[0]["index"]
        print("Input Index", self.input_index)
        self.output_details = self.logmelcalc_interpreter.get_output_details()
        print("Output Details", self.output_details)

        self.baseModel_interpreter = tflite.Interpreter(
            model_path=os.path.join(LIB_FOLDER_LOCATION, "models/first_iteration_siamese/#Quant_static_int8_baseModel_1.0.tflite")
        )
        self.baseModel_interpreter.allocate_tensors()

        self.base_model_inp = self.baseModel_interpreter.get_input_details()
        self.base_model_out = self.baseModel_interpreter.get_output_details()

    def scoreVector(self, inp_vec, embeddings):
        """
        Returns a float with confidence of match 0 - 1
        """
        assert inp_vec.shape == (1, 128), \
        print(inp_vec.shape)

        distances = np.sqrt(
            np.sum(
                (inp_vec - embeddings) ** 2,
                axis=1
            )
        )

        distances[distances > 0.3] = 0.3
        top3 = (0.3 - np.sort(distances)[:3]) / 0.3
        out = 0.0
        for i in top3:
            out += (1 - out) * i

        return out

    def audioToVector(self, inpAudio: np.array) -> np.array:
        """
        Converts 16000Hz sampled 1 sec of audio to vector embedding
        Input:  inpAudio  : np.array of shape (16000,)

        Output: 1 vector embedding of shape (128,1)
        """
        assert (inpAudio.shape == (self.window_frames,))

        self.logmelcalc_interpreter.set_tensor(
            self.input_index,
            np.expand_dims(
                inpAudio / inpAudio.max(),
                axis=0
            ).astype("float32")
        )
        self.logmelcalc_interpreter.invoke()
        logmel_output = self.logmelcalc_interpreter.get_tensor(self.output_details[0]['index'])

        # logmel_output = optimized_mel_calculation_graph(
        #     inpAudio,
        #     use_norm=True,
        #     feature_type='logfbank',
        #     use_pre_emphasis=True,
        #     convert_to_float=True,
        #     saturate=False,
        #     unsigned=True
        # ).numpy()   

        if self.base_model_inp[0]['dtype'] == np.int8:
            input_scale, input_zero_point = self.base_model_inp[0]["quantization"]
            q_normed_input = logmel_output / input_scale + input_zero_point
            self.q_logmel_output = q_normed_input.astype(self.base_model_inp[0]["dtype"])
        self.baseModel_interpreter.set_tensor(
            self.base_model_inp[0]["index"],
            np.expand_dims(self.q_logmel_output, axis=(0, -1))
        )
        self.baseModel_interpreter.invoke()
        q_output_data = self.baseModel_interpreter.get_tensor(self.base_model_out[0]['index'])
        if self.base_model_out[0]['dtype'] == np.int8:
            output_scale, output_zero_point = self.base_model_out[0]["quantization"]
            output_data = output_scale * (q_output_data - output_zero_point)

        print(output_data.shape)

        return output_data


class EfficientWord(ModelRawBackend):
    def __init__(self):
        self.window_length = 1.0  # 1 second
        self.window_frames = int(self.window_length * 16000)
        self.logmelcalc_interpreter = tflite.Interpreter(
            model_path=os.path.join(LIB_FOLDER_LOCATION, "models/first_iteration_siamese/logmelcalc.tflite"
                                    )
        )

        self.logmelcalc_interpreter.allocate_tensors()
        
        self.input_index = self.logmelcalc_interpreter.get_input_details()[0]["index"]
        # print("Input Index", self.input_index)
        self.output_details = self.logmelcalc_interpreter.get_output_details()
        # print("Output Details", self.output_details)

        self.baseModel_interpreter = tflite.Interpreter(
            model_path=os.path.join(LIB_FOLDER_LOCATION, "models/first_iteration_siamese/baseModel.tflite")
        )

        self.baseModel_interpreter.allocate_tensors()

        self.base_model_inp = self.baseModel_interpreter.get_input_details()
        self.base_model_out = self.baseModel_interpreter.get_output_details()

    def scoreVector(self, inp_vec, embeddings):
        """
        **Use this directly only if u know what you are doing**

        Returns a float with confidence of match 0 - 1
        """

        assert inp_vec.shape == (1, 128), \
            "Inp vector should be of shape (1,128)"
        # print(inp_vec.shape)

        distances = np.sqrt(
            np.sum(
                (inp_vec - embeddings) ** 2,
                axis=1
            )
        )

        distances[distances > 0.3] = 0.3
        top3 = (0.3 - np.sort(distances)[:3]) / 0.3
        out = 0.0
        for i in top3:
            out += (1 - out) * i

        return out

    def audioToVector(self, inpAudio: np.array) -> np.array:
        """
        Converts 16000Hz sampled 1 sec of audio to vector embedding
        Inp Parameters :

            inpAudio  : np.array of shape (16000,)

        Out Parameters :

            1 vector embedding of shape (1, embedding_size)

        """
        assert (inpAudio.shape == (self.window_frames,))

        self.logmelcalc_interpreter.set_tensor(
            self.input_index,
            np.expand_dims(
                inpAudio / inpAudio.max(),
                axis=0
            ).astype("float32")
        )
        self.logmelcalc_interpreter.invoke()
        self.logmel_output = self.logmelcalc_interpreter.get_tensor(self.output_details[0]['index'])
        self.baseModel_interpreter.set_tensor(
            self.base_model_inp[0]["index"],
            np.expand_dims(self.logmel_output, axis=(0, -1)).astype("float32")
        )
        self.baseModel_interpreter.invoke()
        output_data = self.baseModel_interpreter.get_tensor(self.base_model_out[0]['index'])

        return output_data



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

class TCResNet14_Arc_loss(ModelRawBackend):
    def __init__(self):
        self.window_length = 1.0
        self.window_frames = int(self.window_length * 16000)

        self.onnx_sess = rt.InferenceSession(
            os.path.join(
                LIB_FOLDER_LOCATION, "models/pt_onnx/dstcres_quant_dynamic.onnx"),
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
    effnet = "effnet"
    tcres14 = "tcres14"
    resarc = "resarc"
    tcarc14 = "tcarc14"
    tdsp2 = "tdsp2"


MODEL_TYPE_MAPPER = {
    "effnet": EfficientWord,
    "tcres14": TCResNet14,
    "resarc": Resnet50_Arc_loss,
    "tcarc14": TCResNet14_Arc_loss,
    "tdsp2": TDResNeXt_SP2_loss,
}
