# Generic imports
from datasets import load_dataset
# import matplotlib
# import librosa
# import IPython
# import random

# IPU-specific imports
from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import to_pipelined

# HF-related imports
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import os
os.environ["POPART_LOG_LEVEL"]="INFO"
os.environ["POPLAR_LOG_LEVEL"]="INFO"

model_spec = "openai/whisper-tiny.en"

# Instantiate processor and model
processor = WhisperProcessor.from_pretrained(model_spec)
model = WhisperForConditionalGeneration.from_pretrained(model_spec)

# Adapt whisper-tiny to run on the IPU
ipu_config = IPUConfig(ipus_per_replica=2)
pipelined_model = to_pipelined(model, ipu_config)
pipelined_model = pipelined_model.parallelize(for_generation=True).half()

# load the dataset and read an example sound file
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
test_sample = ds[2]
sample_rate = test_sample['audio']['sampling_rate']

def transcribe(data, rate):
    input_features = processor(data, return_tensors="pt", sampling_rate=rate).input_features.half()

    # This triggers a compilation, unless a precompiled model is available.
    sample_output = pipelined_model.generate(input_features, max_length=448, min_length=3)
    transcription = processor.batch_decode(sample_output, skip_special_tokens=True)[0]
    return transcription


test_transcription = transcribe(test_sample["audio"]["array"], sample_rate)

print("Compilation succeeded, end of test.")
pipelined_model.detachFromDevice()

