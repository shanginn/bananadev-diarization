from pyannote.audio import Pipeline
import os

Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=os.getenv('HF_TOKEN'),
)