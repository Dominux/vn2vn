import wave

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import numpy as np

torch_dtype = torch.bfloat16 # set your preferred type here



def gen_speech2text():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        setattr(torch.distributed, "is_initialized", lambda : False) # monkey patching
    device = torch.device(device)

    whisper = WhisperForConditionalGeneration.from_pretrained(
        "antony66/whisper-large-v3-russian", torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
        # add attn_implementation="flash_attention_2" if your GPU supports it
    )

    processor = WhisperProcessor.from_pretrained("antony66/whisper-large-v3-russian")

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=whisper,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=256,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # read your wav file into variable wav. For example:

    with wave.open('/tmp/vn2vn/6d0d5ffe-cccc-4268-8858-6518c0fa85ca/input_audio.wav', 'rb') as f:
        samples = f.getnframes()
        audio = f.readframes(samples)

    # Convert buffer to float32 using NumPy
    audio = np.frombuffer(audio, dtype=np.float32)
    # audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0
    max_int16 = 2**15
    audio = audio / max_int16

    # get the transcription
    asr = asr_pipeline(audio, generate_kwargs={"language": "russian", "max_new_tokens": 256}, return_timestamps=False)

    print(asr['text'])
