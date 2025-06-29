import json
from os.path import isfile , join
import numpy as np
import pyaudio

from typing import Tuple , List ,Union

from pcov_kws.audio_processing import ModelRawBackend
from pcov_kws import RATE
from time import time as current_time_in_sec
import logging

from pcov_kws.audio_processing import MODEL_TYPE_MAPPER

class HotwordDetector:

    def __init__(
            self,
            hotword:str,
            model:ModelRawBackend,
            reference_file:str,
            threshold:float=0.7,
            relaxation_time=0.8,
            continuous=True,
            verbose = False):
        
        assert isfile(reference_file), \
            "Reference File Path Invalid"

        assert threshold>0 and threshold<1, \
            "Threshold can be only between 0 and 1"

        data = json.loads(open(reference_file,'r').read())
        self.embeddings = np.array(data["embeddings"]).astype(np.float32)

        #self.model_type = data.get()
        assert self.embeddings.shape[0]>3, \
            "Minimum of 4 samples is required"
        
        assert MODEL_TYPE_MAPPER[data["model_type"]]==type(model)
        self.model = model

        self.hotword = hotword
        self.threshold = threshold
        self.continuous = continuous

        self.relaxation_time = relaxation_time
        self.verbose = verbose
        
        self.__last_activation_time = 0.0 # Initialize to 0 to allow immediate first trigger
        self.is_running = False

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False


    def __repr__(self):
        return f"Hotword: {self.hotword}"

    def __crossedRelaxationTime(self):
        current_time = current_time_in_sec()
        print("gap :",current_time - self.__last_activation_time)
        return (current_time-self.__last_activation_time) > self.relaxation_time

    def scoreVector(self,inp_vec:np.array) -> float :
        # This method now only returns the raw score.
        # Relaxation/cooldown logic is handled by the calling methods (scoreFrame or findBestMatch).
        score =  self.model.scoreVector(inp_vec, self.embeddings)
        return score
          
    def scoreFrame(
            self,
            inp_audio_frame:np.array,
            unsafe:bool = False) -> float :
        """
        Converts given audio frame to embedding and checks for similarity
        with given reference file

        Inp Parameters:

            inp_audio_frame : np.array of 1channel 1 sec 16000Hz sampled audio 
            frame
            unsafe : bool value, set to False by default to prevent engine
            processing continuous speech or silence, to minimalize false positives

        **Note : change unsafe to True only if you know what you are doing**

        Out Parameters:

            {
                "match":True or False,
                "confidence":float value
            }
                 or 
            None when no voice activity is identified
        """
        current_time = current_time_in_sec()
        # Cooldown check for single detector mode
        if (current_time - self.__last_activation_time) < self.relaxation_time:
            return {"match": False, "confidence": 0.0}

        if(not unsafe):
            upperPoint = max(
                (
                    inp_audio_frame/inp_audio_frame.max()
                )[:RATE//10]
            )
            if(upperPoint > 0.2):
                return None

        #assert inp_audio_frame.shape == (RATE,), \
        #    f"Audio frame needs to be a 1 sec {RATE}Hz sampled vector"

        if not self.is_running:
            return None
        else:
            score = self.scoreVector(
                self.model.audioToVector(
                    inp_audio_frame
                )
            )
            
            is_match = score >= self.threshold
            if is_match:
                self.__last_activation_time = current_time

            return {
                "match": is_match,
                "confidence": score
            }

HotwordDetectorArray = List[HotwordDetector]
MatchInfo = Tuple[HotwordDetector,float]
MatchInfoArray = List[MatchInfo]

class MultiHotwordDetector :


    def __init__(
        self,
        detector_collection:HotwordDetectorArray,
        model:ModelRawBackend,
        relaxation_time:float=0.8,
        continuous=True
    ):
        """
        Input: detector_collection : List of HotwordDetector instances
        """
        assert len(detector_collection)>1, \
            "Pass atleast 2 HotwordDetector instances"

        for detector in detector_collection :
            assert isinstance(detector,HotwordDetector), \
                "Mixed Array received, send HotwordDetector only array"
            
        self.model = model

        self.detector_collection = detector_collection
        self.continous = continuous
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

    def findBestMatch(
            self,
            inp_audio_frame:np.array,
            unsafe:bool=False
            ) -> MatchInfo :
        """
        Output:(detector,score) : returns detector of best matched hotword with its score
        """
        current_time = current_time_in_sec()
        if (current_time - self.__last_activation_time) < self.relaxation_time:
            return (None, 0.0)

        embedding = self.model.audioToVector(inp_audio_frame)

        best_match_detector:HotwordDetector = None
        best_match_score:float = 0.0

        for detector in self.detector_collection :
            if not detector.is_running:
                continue
            
            # Use the model's scoring directly, bypassing HotwordDetector's internal logic
            score = self.model.scoreVector(embedding, detector.embeddings)

            if(score < detector.threshold):
                continue

            if(score>best_match_score):
                best_match_score = score
                best_match_detector = detector
        
        if best_match_detector is not None:
            self.__last_activation_time = current_time

        return (best_match_detector,best_match_score)

    def findAllMatches(
            self,
            inp_audio_frame:np.array,
            unsafe:bool=False
            ) -> MatchInfoArray :
        """
        Returns the best match hotword for a given audio frame
        within respective thresholds , returns None if found none

        Inp Parameters:

            inp_audio_frame : 1 sec 16000Hz frq sampled audio frame

            unsafe : bool value, set to False by default to prevent engine
            processing continuous speech , to minimalize false positives

        Note : change unsafe to True only if you know what you are doing

        Out Parameters:

            [ (detector,score) ,... ] : returns list of matched detectors 
            with respective scores

        """
        #assert inp_audio_frame.shape == (RATE,), \
        #    f"Audio frame needs to be a 1 sec {RATE}Hz sampled vector"


        if self.continous and (not unsafe):
            upperPoint = max(
                (
                    inp_audio_frame/inp_audio_frame.max()
                )[:1600]
            )
            if(upperPoint > 0.2 or upperPoint==0):
                return None , None

        embedding = self.model.audioToVector(inp_audio_frame)

        matches:MatchInfoArray = []

        best_match_score = 0.0
        for detector in self.detector_collection :
            if not detector.is_running:
                continue
            score = detector.getMatchScoreVector(embedding)
            print(detector,score,end="|")
            if(score<detector.threshold):
                continue
            if(len(matches)>0):
                for i in range(len(matches)):
                    if matches[i][1] > score :
                        matches.insert(i,(detector,score))
                        break
                else:
                    matches.append(i,(detector,score))
            else:
                matches.append(
                        (detector,score)
                        )
        print()
        return matches
