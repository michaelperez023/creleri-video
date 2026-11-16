import os

from sam2.sam2_video_predictor import SAM2VideoPredictor
from torch import multiprocessing
from transformers import AutoModelForCausalLM, AutoProcessor

# Molmo ships its model code in the Hub repo (trust_remote_code). Set MOLMO_REVISION to a commit
# hash, e.g. cab33fb7f1a40091911f81165f8481920621948f, to pin the code that runs; unset, the
# latest (or locally cached) revision is used.
MOLMO_REVISION = os.environ.get("MOLMO_REVISION") or None

models = {}


def init(device_queue: multiprocessing.Queue, worker_ready):
    device = device_queue.get()
    print("initializing worker process on device", device)

    try:
        models["sam_model"] = SAM2VideoPredictor.from_pretrained(
            "facebook/sam2-hiera-large", device=device
        )

        models["molmo_processor"] = AutoProcessor.from_pretrained(
            "allenai/Molmo-7B-D-0924",
            revision=MOLMO_REVISION,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device,
        )

        models["molmo_model"] = AutoModelForCausalLM.from_pretrained(
            "allenai/Molmo-7B-D-0924",
            revision=MOLMO_REVISION,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device,
        )
    except BaseException:
        # The pool would replace this worker with one that blocks on the now-empty device
        # queue; breaking the barrier makes the parent fail instead of waiting forever.
        worker_ready.abort()
        raise

    worker_ready.wait()
