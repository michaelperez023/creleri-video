from transformers import AutoModelForCausalLM, AutoProcessor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from torch import multiprocessing

models = {}


def init(device_queue: multiprocessing.Queue, worker_ready):
    device = device_queue.get()
    print('initializing worker process on device', device)

    models["sam_model"] = SAM2VideoPredictor.from_pretrained(
        "facebook/sam2-hiera-large", device=device
    )

    models["molmo_processor"] = AutoProcessor.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device,
    )

    models["molmo_model"] = AutoModelForCausalLM.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device,
    )

    worker_ready.wait()
