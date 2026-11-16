# Third-party notices

CReLeRI is released under the [GNU AGPL v3](LICENSE). It includes code from the projects
below, each under its own license, and downloads model weights that carry their own terms.

## Vendored code

| Component | Location | Upstream | License | Local changes |
|---|---|---|---|---|
| OpenTAD | `adatad_temp_seg/opentad/` | [sming256/OpenTAD](https://github.com/sming256/OpenTAD) @ `15f5c44` (Nov 2024) | Apache-2.0 ([copy](adatad_temp_seg/opentad/LICENSE)) | Added `datasets/base/single_video_dataset.py` and `cores/test_engine_single_video.py` for single-video inference, registered in the package `__init__` files; `torch.amp` updates in the Re2TAL backbones; the Align1D CUDA kernel dispatches on `scalar_type()` so it builds with PyTorch 2.9. Modified files start with a "Modified from OpenTAD" line. The `adatad_temp_seg/*-inference.py` configs are derived from OpenTAD configs. |
| D-FINE | `dfine_obj_det/` (`src/`, `configs/`, `reference/`, `tools/`, `train.py`) | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) @ `ee4a951` (Nov 2024) | Apache-2.0 ([copy](dfine_obj_det/LICENSE)) | None. `tools/inference/object_proposer_api.py` and `environment.yml` are CReLeRI files. |
| Grounded-SAM-2, bundling SAM 2 and Grounding DINO | `vlm_action_pipeline/grounded_sam_2/` | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) @ `e899ad9` (Sep 2024) | Apache-2.0; parts under BSD-3-Clause and MIT (see the `LICENSE*` files in that directory) | None. Upstream notebooks, demo media and tests are omitted. |
| Objects365 class names | `dfine_obj_det/tools/inference/Objects365.yaml` | [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | None. |
| Bootstrap 3.1.1 CSS | `web_ui/static/plugins/bootstrap/` | [twbs/bootstrap](https://github.com/twbs/bootstrap) | MIT | None. |
| SB Admin 2 theme | basis of `web_ui/static/css/style.css` | Iron Summit Media Strategies | Apache-2.0 | Restyled for CReLeRI. |
| binarytheme template | basis of `web_ui/static/css/main-style.css` | [binarytheme.com](http://binarytheme.com) | Free for personal and commercial use | Restyled for CReLeRI. |

The web UI also loads [D3](https://d3js.org) 6.7.0 (BSD-3-Clause), Font Awesome 5.3.1 (icons
CC BY 4.0, fonts SIL OFL 1.1, code MIT) and Google Fonts from public CDNs. The object proposer
uses [BoxMOT](https://github.com/mikel-brostrom/boxmot)'s StrongSORT tracker (AGPL-3.0) as a
pip dependency.

## Model weights

Weights are not part of this repository. Their licenses differ from the code license; check
them before any commercial use.

| Model | Used by | License / terms |
|---|---|---|
| AdaTAD checkpoint `best.pth`, fine-tuned from [VideoMAEv2-giant](https://huggingface.co/OpenGVLab/VideoMAEv2-giant) | segmentation worker | CC BY-NC 4.0 (inherited from VideoMAEv2) |
| [D-FINE-X Objects365](https://github.com/Peterande/D-FINE) | object proposer | Apache-2.0; the [Objects365](https://www.objects365.org) dataset terms may also apply |
| [OSNet-AIN x1.0, MSMT17](https://github.com/KaiyangZhou/deep-person-reid) (downloaded by BoxMOT) | tracker re-identification | MIT; trained on MSMT17, which is licensed for non-commercial research only |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | segmentation worker | Apache-2.0 |
| [Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) | captioning | Apache-2.0 |
| [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), served by your own endpoint | action parsing | Llama 3.3 Community License |
| [Grounding DINO base](https://huggingface.co/IDEA-Research/grounding-dino-base) | grounding | Apache-2.0 |
| [SAM 2 Hiera-L](https://huggingface.co/facebook/sam2-hiera-large) | grounding | Apache-2.0 |
| [Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924) | grounding | Apache-2.0 |
| [Depth Pro](https://github.com/apple/ml-depth-pro) | grounding | Apple Machine Learning Research Model license |
