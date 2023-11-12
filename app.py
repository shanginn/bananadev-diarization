import os
import base64

from potassium import Potassium, Request, Response
from io import BytesIO
from pyannote.audio import Pipeline

app = Potassium("diarization")


@app.init
def init():
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token=os.getenv('HF_TOKEN'),
    )

    context = {
        "pipeline": pipeline
    }

    return context


@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    print('Request received')

    pipeline = context.get("pipeline")
    print('Pipeline loaded')

    base64file = request.json.get("file")
    file = BytesIO(base64.b64decode(base64file))

    print('File loaded')
    diarization = pipeline(file)
    print('Diarization done')

    output = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        output.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    print(diarization)

    return Response(json={"output": output}, status=200)


if __name__ == "__main__":
    app.serve()