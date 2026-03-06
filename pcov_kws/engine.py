import json
from os.path import isfile, join
import numpy as np

from typing import Tuple, List, Union

from pcov_kws.audio_processing import ModelRawBackend
from pcov_kws import RATE
from time import time as current_time_in_sec
import logging

from pcov_kws.audio_processing import MODEL_TYPE_MAPPER


class HotwordDetector:

    def __init__(
            self,
            hotword: str,
            model: ModelRawBackend,
            reference_file: str,
            threshold: float = 0.7,
            relaxation_time=0.8,
            continuous=True,
            verbose=False):

        assert isfile(reference_file), \
            "Reference File Path Invalid"

        assert 0 < threshold < 1, \
            "Threshold must be between 0 and 1"

        with open(reference_file, 'r') as f:
            data = json.load(f)
        self.embeddings = np.array(data["embeddings"]).astype(np.float32)

        assert self.embeddings.shape[0] > 3, \
            "Minimum of 4 samples is required"

        assert MODEL_TYPE_MAPPER[data["model_type"]] == type(model)
        self.model = model

        self.hotword = hotword
        self.threshold = threshold
        self.continuous = continuous

        self.relaxation_time = relaxation_time
        self.verbose = verbose

        self.__last_activation_time = 0.0
        self.is_running = False

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False

    def reset_activation_timer(self, timestamp: float = None):
        """Reset the internal cooldown timer to *timestamp* (default: now)."""
        self.__last_activation_time = timestamp if timestamp is not None else current_time_in_sec()

    def __repr__(self):
        return f"Hotword: {self.hotword}"

    def scoreVector(self, inp_vec: np.array) -> float:
        return self.model.scoreVector(inp_vec, self.embeddings)

    def scoreFrame(
            self,
            inp_audio_frame: np.array,
            unsafe: bool = False) -> Union[dict, None]:
        """Score an audio frame against the reference embeddings.

        Returns dict with match/confidence/rms, or None if no voice activity.
        """
        current_time = current_time_in_sec()
        if (current_time - self.__last_activation_time) < self.relaxation_time:
            return {"match": False, "confidence": 0.0}

        rms_value = np.sqrt(np.mean(np.square(inp_audio_frame.astype(np.float32))))

        if not unsafe:
            max_val = inp_audio_frame.max()
            if max_val != 0:
                upperPoint = max((inp_audio_frame / max_val)[:RATE // 10])
                if upperPoint > 0.2:
                    return None
            else:
                return None  # Silent frame

        if not self.is_running:
            return None

        score = self.scoreVector(self.model.audioToVector(inp_audio_frame))

        is_match = score >= self.threshold
        if is_match:
            self.__last_activation_time = current_time

        return {
            "match": is_match,
            "confidence": score,
            "rms": rms_value
        }


HotwordDetectorArray = List[HotwordDetector]
MatchInfo = Tuple[HotwordDetector, float]
MatchInfoArray = List[MatchInfo]


class MultiHotwordDetector:

    def __init__(
        self,
        detector_collection: HotwordDetectorArray,
        model: ModelRawBackend,
        relaxation_time: float = 0.8,
        continuous=True
    ):
        assert len(detector_collection) > 1, \
            "Pass at least 2 HotwordDetector instances"

        for d in detector_collection:
            assert isinstance(d, HotwordDetector), \
                "All elements must be HotwordDetector instances"

        self.model = model
        self.detector_collection = detector_collection
        self.continuous = continuous
        self.is_running = False

        self.relaxation_time = relaxation_time
        self.__last_activation_time = 0.0

    def start(self):
        self.is_running = True
        for detector in self.detector_collection:
            detector.start()

    def stop(self):
        self.is_running = False
        for detector in self.detector_collection:
            detector.stop()

    def reset_activation_timer(self, timestamp: float = None):
        """Reset the internal cooldown timer to *timestamp* (default: now).

        Also resets the timer on all child detectors.
        """
        t = timestamp if timestamp is not None else current_time_in_sec()
        self.__last_activation_time = t
        for detector in self.detector_collection:
            detector.reset_activation_timer(t)

    def findBestMatch(
            self,
            inp_audio_frame: np.array,
            unsafe: bool = False
    ) -> MatchInfo:
        """Return (detector, score) for the best matched hotword, or (None, 0.0)."""
        current_time = current_time_in_sec()
        if (current_time - self.__last_activation_time) < self.relaxation_time:
            return (None, 0.0)

        embedding = self.model.audioToVector(inp_audio_frame)

        best_match_detector: HotwordDetector = None
        best_match_score: float = 0.0

        for detector in self.detector_collection:
            if not detector.is_running:
                continue

            score = self.model.scoreVector(embedding, detector.embeddings)

            if score < detector.threshold:
                continue

            if score > best_match_score:
                best_match_score = score
                best_match_detector = detector

        if best_match_detector is not None:
            self.__last_activation_time = current_time

        return (best_match_detector, best_match_score)

    def findAllMatches(
            self,
            inp_audio_frame: np.array,
            unsafe: bool = False
    ) -> MatchInfoArray:
        """Return a list of (detector, score) for all matched hotwords, sorted by score descending."""
        if self.continuous and (not unsafe):
            max_val = inp_audio_frame.max()
            if max_val == 0:
                return []
            upperPoint = max((inp_audio_frame / max_val)[:1600])
            if upperPoint > 0.2:
                return []

        embedding = self.model.audioToVector(inp_audio_frame)

        matches: MatchInfoArray = []

        for detector in self.detector_collection:
            if not detector.is_running:
                continue
            score = self.model.scoreVector(embedding, detector.embeddings)
            if score < detector.threshold:
                continue
            matches.append((detector, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
