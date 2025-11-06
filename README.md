
# CReLeRI: Explainable, Concept-Centric Video Analysis System

Authors:
Michael Francis Perez, Yichi Yang, Yuheng Zha, Enze Ma, Danish Tamboli, Haodi Ma, Reza Shahriari, Vyom Pathak, Dzmitry Kasinets, Rohith Venkatakrishnan, Daisy (Zhe) Wang, Jaime Ruiz, Eric D. Ragan, Zhiting Hu, Eric Xing, and Jun-Yan Zhu.

In Proceedings of the 33rd ACM International Conference on Multimedia (ACM MM ’25), Dublin, Ireland.
https://doi.org/10.1145/3746027.3754479

# Overview
CReLeRI (Concept-centric Representation, Learning, Reasoning, and Interaction) is an open-source system for explainable action detection in untrimmed videos.
The system combines segmentation, vision-language reasoning, and spatial grounding to deliver interpretable and trustworthy video understanding.
Our goal is to reduce hallucinations, improve temporal precision, and enhance transparency in AI-driven video analysis.

# Key Features
Action Segmentation: Automatically partitions untrimmed videos by detecting both scene and action transitions. \
Action Recognition: Identifies actor-specific trajectories and predicts action–argument pairs via multimodal large models (Qwen2.5-VL + LLaMA3.1). \
3D Grounding: Links detected actions and objects to physical locations using GroundingDINO, Molmo, SAM2, and DepthPro. \
Explainable Visualization: Overlays segmentation masks and interaction points for human-interpretable evidence. \
Modular Backend: Built with FastAPI, Celery, and Redis for scalable asynchronous processing. \
Flexible Vocabulary: Easily redefine your own set of actions or arguments; only the segmentation model is fine-tuned.

# Advantages Over Existing Systems
Improved temporal resolution for long, untrimmed videos. \
Enhanced explainability and grounding, mitigating hallucinations. \
Open-source and extensible — all major components are pretrained; only segmentation is task-specific. \
Supports offline video retrieval and semantic search applications.

# Applications
Video retrieval & semantic search (offline or corpus-scale indexing) \
Video summarization \
Explainable action detection \
Dataset annotation and reasoning visualization

# Demo
Watch the demonstration video:  https://www.youtube.com/watch?v=2v0DDpFEq14 \
Paper on ACM Digital Library: https://doi.org/10.1145/3746027.3754479

# Install

1. temporal segmentation server
cd adatad_temp_seg \
conda env create -f environment.yml 

download best.pth from: https://drive.google.com/file/d/1GZl1UIDW8Qs0-I-sgeEW1NAGL8plXA3M/view?usp=sharing \
using some command like: \
pip install gdown \
gdown --id 1GZl1UIDW8Qs0-I-sgeEW1NAGL8plXA3M -O best.pth

#wget vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth \
#this file used to be available by visiting: \
https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md#model-weight-links \
#filling out this download request form:
https://docs.google.com/forms/d/e/1FAIpQLSd1SjKMtD8piL9uxGEUwicerxd46bs12QojQt92rzalnoI3JA/viewform \
#then converting the file by running: \
python tools/model_converters/convert_videomaev2.py \
    vit_g_hybrid_pt_1200e_k710_ft.pth pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth

#now, the download request form links to a public huggingface model: \
https://huggingface.co/OpenGVLab/VideoMAEv2-giant

2. object detection server
cd dfine_obj_det \
conda env create -f environment.yml

wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj365.pth \
cd configs/dfine \
wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth 

3. redis

cd redis \
wget https://download.redis.io/redis-stable.tar.gz \
tar -xzvf redis-stable.tar.gz \
cd redis-stable \
make

4. qwen, llama, and grounding servers 

cd vlm_action_pipeline \
conda env create -f qwen_environment.yml \
conda env create -f ufllama_environment.yml \
conda env create -f grounding_environment.yml 

cd demo_pipeline \
wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt

5. Web user interface

cd web_ui \
python -m venv webui \
source webui/bin/activate \
pip install -r requirements.txt

# Run

Starting creleri system

(UI, AdaTAD action segmentation, Dfine object detection, llama3.3, qwen2.5, grounding)

Notes: \
#assume a machine with 4× NVIDIA B200 GPUs and 4 CPU cores \
#Update the GPU configuration in: \
#grounding_api.py (or via CLI -d argument when launching the grounding service) \
#qwen2p5_api.py to match your hardware \
#One of the action-segmentation models used in this pipeline is no longer publicly available. You will need to substitute an alternative checkpoint or adapt the checkpoint that is available on hugging face. \
#The local Llama-3.3 configuration (non-UF deployment, llama32_api.py) has not been recently tested. If running outside of UF, ensure an updated model path/checkpoint and a fresh environment are configured. \
#install the environments for each API first before running them via the install.txt in each subdirectory. 

1. WebUI
#New terminal \
#assuming you have ffmpeg \
cd web_ui \
source webui/bin/activate \
python app.py 5004 \
#Go to: http://localhost:5004 \
#do port forwarding if UI is running on different machine than backend

2. action segmentation API - redis \
#new terminal \
cd redis/redis-stable \
src/redis-server --port 17253

3. action segmentation API - celery worker
#new terminal \
#assuming you have conda and ffmpeg \
conda activate opentad4 \
cd adatad_temp_seg \
CUDA_VISIBLE_DEVICES=1 python celery_worker.py -d 0

4. action segmentation API - api
#new terminal \
#assuming you have conda \
conda activate opentad4 \
cd adatad_temp_seg \
CUDA_VISIBLE_DEVICES=1 python api.py -p 17249

5. Dfine object detection API
#new terminal \
#assuming you have conda \
conda activate dfine2 \
cd dfine_obj_det/tools/inference \
CUDA_VISIBLE_DEVICES=0 python object_proposer_api.py

6. qwen2.5-VL API
#assuming you have conda and cuda/12.8.1 \
conda activate qwen3 \
cd vlm_action_pipeline/demo_pipeline \
export PYTHONNOUSERSITE=1 \
python qwen2p5_api.py

7. llama3.3 API
#assuming you have conda \
#if on UF campus \
conda activate ufllama \
cd vlm_action_pipeline/demo_pipeline \
python llama_3p3_api.py

#otherwise
python llama32_api.py

8. grounding API
#assuming you have conda and ffmpeg \
conda activate demo_pipeline \
cd vlm_action_pipeline/demo_pipeline \
python grounding_api.py -d 0 1 2 3


# Cite
If you use CReLeRI in your research, please cite:
```bibtex
@inproceedings{10.1145/3746027.3754479,
author = {Perez, Michael Francis and Yang, Yichi and Zha, Yuheng and Ma, Enze and Tamboli, Danish and Ma, Haodi and Shahriari, Reza and Pathak, Vyom and Kasinets, Dzmitry and Venkatakrishnan, Rohith and Wang, Daisy (Zhe) and Ruiz, Jaime and Ragan, Eric D. and Hu, Zhiting and Xing, Eric and Zhu, Jun-Yan},
title = {CReLeRI: Explainable, Concept-centric, Representation, Learning, Reasoning, and Interaction Video Analysis System},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746027.3754479},
doi = {10.1145/3746027.3754479},
abstract = {Existing video analysis models often lack explainability, perform poorly on long videos, and frequently hallucinate. Commercial solutions are closed-source and costly. We introduce CReLeRI, an open-source system for action detection in untrimmed videos. CReLeRI segments videos using scene and action transitions, detects actions and their arguments and grounds them in 3D space to improve interpretability and reduce hallucinations. The system promotes transparency and trust in AI-driven analysis of complex, real-world videos. A demonstration video is also available.},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {13528–13530},
numpages = {3},
keywords = {grounding, human-centered computing, interpretability, large language models, multimedia interaction, object detection, video action detection, vision-language models},
location = {Dublin, Ireland},
series = {MM '25}
}```