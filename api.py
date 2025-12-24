from fastapi import FastAPI, UploadFile, File
import shutil
from cv_pipeline.probe_embedder import ProbeEmbedder
from cv_pipeline import shared_state

app = FastAPI()

PROBE_IMAGE_PATH = "cv_pipeline/uploaded_person.jpg"
WEIGHTS_PATH = "cv_pipeline/osnet_x0_25_msmt17.pt"


@app.post("/upload_probe")
async def upload_probe(file: UploadFile = File(...)):
    # Save uploaded image
    with open(PROBE_IMAGE_PATH, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract embedding
    probe = ProbeEmbedder(WEIGHTS_PATH)
    shared_state.PROBE_EMBEDDING = probe.extract(PROBE_IMAGE_PATH)

    # Reset result
    shared_state.PROBE_RESULT = {
        "matched": False,
        "person_id": None,
        "similarity": None
    }

    return {"status": "probe_loaded"}


@app.get("/probe_status")
def probe_status():
    return shared_state.PROBE_RESULT
