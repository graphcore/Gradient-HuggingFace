# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import logging
import time
import os
import numpy as np
import librosa
from ssf.application import SSFApplicationInterface
from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import to_pipelined
from transformers import WhisperProcessor, WhisperForConditionalGeneration

logger = logging.getLogger("whisper")

class MyApplication(SSFApplicationInterface):
    def __init__(self):
        logger.info("App init...")
        
        model_spec = "openai/whisper-tiny.en"
        self.processor = WhisperProcessor.from_pretrained(model_spec)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_spec)
        self.ipu_config = IPUConfig(ipus_per_replica=2, executable_cache_dir="./exe_cache")
        self.pipelined_model = to_pipelined(self.model, self.ipu_config)
        self.pipelined_model = self.pipelined_model.parallelize(for_generation=True).half()
    
    
    def transcribe(self, data, rate):
        input_features = self.processor(data, return_tensors="pt", sampling_rate=rate).input_features.half()
        sample_output = self.pipelined_model.generate(input_features, max_length=448, min_length=3)
        transcription = self.processor.batch_decode(sample_output, skip_special_tokens=True)[0]
        return transcription
        
    # BUILD method for SSF
    def build(self) -> int:
        return 0

    # STARTUP method for SSF
    def startup(self) -> int:
        logger.info("App starting... Executing 1 request for warmup")
        t0 = time.time()
        self.transcribe(np.random.rand(199760), 16000)
        logger.info(f"MyApp started in, {time.time()-t0} s")
        return 0
    
    # REQUEST method for SSF
    def request(self, params: dict, meta: dict) -> dict:
        logger.info("Receiving request...")
        try:
            data, samplerate = librosa.load(params['audio_file'], sr=16000) 
        except:
            raise Exception("Error loading audio")
        try:
            result = {"result": self.transcribe(data, samplerate)} 
        except:
            raise Exception("Error processing audio")
        logger.info(f"MyApp returning result={result}")
        return result
    
    # SHUTDOWN method for SSF
    def shutdown(self) -> int:
        logger.info("MyApp shutdown")
        self.pipelined_model.detachFromDevice()
        return 0

    # IS_HEALTHY method for SSF
    def is_healthy(self) -> bool:
        return True


# Factory method enforced by SSF
def create_ssf_application_instance() -> SSFApplicationInterface:
    logger.info("Create MyApplication instance")
    return MyApplication()
