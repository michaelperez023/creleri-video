import os
from datetime import datetime, timezone
import torch
from mmengine.config import Config
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import eval_one_epoch_single_video
from opentad.utils import set_seed
import decord
import requests
import json
import cv2
import argparse
from collections import OrderedDict
import pandas as pd
import ffmpeg
from celery import Celery
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

import logging, coloredlogs

REDIS_PORT = 17253
LLAMA_PORT = 5003
OBJ_DET_PORT = 17266
GROUNDING_PORT = 12000

celery_app = Celery('tasks', broker=f'redis://localhost:{REDIS_PORT}/0', backend=f'redis://localhost:{REDIS_PORT}/0')
    
# Global variables for configuration, model, and device
cfg = None
model = None
device = None
use_amp = None
sentence_transformer_model = None

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix
        if k.startswith('module.'):
            new_k = k[7:]  # len('module.') == 7
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

def initialize_globals(device_id):
    global cfg, model, device, use_amp, sentence_transformer_model, thumbnail_dir

    # Default configuration parameters
    config_path = "e2e_thumos_videomaev2_g_768x2_224_adapter-inference.py"
    checkpoint_path = "best.pth"
    seed = 42

    # Load configuration
    cfg = Config.fromfile(config_path)

    # Set up device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    # Load model and checkpoint
    model = build_detector(cfg.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict_ema", checkpoint["state_dict"])
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)

    # Initialize SentenceTransformer and AMP (automatic mixed precision)
    use_amp = getattr(cfg.solver, "amp", False)
    sentence_transformer_model = SentenceTransformer('all-mpnet-base-v2')

    thumbnail_dir = "./thumbnails"
    os.makedirs(thumbnail_dir, exist_ok=True)

def get_video_duration_decord(video_path):
    vr = decord.VideoReader(video_path)
    num_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = num_frames / fps
    return duration

def segments_overlap(seg1, seg2):
    """
    Returns True if segments overlap, False otherwise.
    """
    return max(seg1[0], seg2[0]) < min(seg1[1], seg2[1])

def select_non_overlapping_proposals(proposals):
    # Sort proposals by score descending
    proposals = sorted(proposals, key=lambda x: x['score'], reverse=True)
    selected = []
    for prop in proposals:
        # Check if prop overlaps with any selected segment
        overlaps = False
        for sel in selected:
            if segments_overlap(prop['segment'], sel['segment']):
                overlaps = True
                break
        if not overlaps:
            # Exclude the 'label' field if it exists
            selected_prop = {
                'segment': prop['segment'],
                'score': prop['score']
            }
            selected.append(selected_prop)
    return selected

def detect_transitions(video_path, threshold=30):
    cap = cv2.VideoCapture(video_path)
    frame_transitions = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    success, prev_frame = cap.read()
    if not success:
        print(f"Failed to read video {video_path}")
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for i in tqdm(range(1, frame_count), desc=f"Processing {os.path.basename(video_path)}"):
        success, curr_frame = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        mean_diff = frame_diff.mean()

        if mean_diff > threshold:
            transition_time = round(i / fps, 2)  # Convert frame number to time in seconds
            frame_transitions.append(transition_time)

        prev_gray = curr_gray

    cap.release()
    return frame_transitions

def adjust_segments_to_scene_changes(proposals, scene_changes, video_duration, min_duration=2):
    # Step 1: Create initial segments based on scene changes
    scene_changes = sorted(scene_changes)
    segments = []
    last_end = 0.0
    
    proposal_times = sorted([time for seg in proposals for time in seg['segment']])

    # Create segments between scene changes
    for sc in scene_changes:
        segments.append({'segment': [last_end, sc], 'score': 0.0})
        last_end = sc

    # Add the final segment up to the video duration
    if last_end < video_duration:
        segments.append({'segment': [last_end, video_duration], 'score': 0.0})

    # Step 2: For each initial segment, attempt to split based on proposals, enforcing min_duration
    final_segments = []
    for seg in segments:
        start, end = seg['segment']
        split_points = [start]

        # Identify valid split points based on proposal times
        for time in proposal_times:
            if time > start and time < end:
                if time - split_points[-1] >= min_duration and end - time >= min_duration:
                    split_points.append(time)
        split_points.append(end)

        # Create sub-segments
        for i in range(len(split_points) - 1):
            sub_start = split_points[i]
            sub_end = split_points[i + 1]
            final_segments.append({'segment': [sub_start, sub_end], 'score': 0.0})
            
    return final_segments

def segment_video(video_path):
    # Detect scene changes
    scene_changes = detect_transitions(video_path)
    print(f"Scene changes detected: {scene_changes}")
    video_duration = get_video_duration_decord(video_path)
    
    # Get action segmentation predictions
    cfg.dataset.test = dict(
        type='SingleVideoDataset',
        video_path=video_path,
        class_map='category_idx_51.txt',
        pipeline=cfg.dataset.test.pipeline,
        feature_stride=4,
        sample_stride=1,
        window_size=768,
        window_overlap_ratio=0.5,
        test_mode=True,
    )

    test_dataset = build_dataset(cfg.dataset.test)
    test_loader = build_dataloader(
        test_dataset,
        rank=0,
        world_size=1,
        **cfg.solver.test,
    )

    result = eval_one_epoch_single_video(
        test_loader,
        model,
        cfg,
        use_amp=use_amp,
        device=device
    )

    final_proposals = {}
    for video_id, proposals in result['results'].items():
        # Filter and select non-overlapping proposals
        pre_nms_thresh = 0.001
        proposals = [p for p in proposals if p['score'] > pre_nms_thresh]
        proposals = sorted(proposals, key=lambda x: x['score'], reverse=True)[:2000]
        selected_proposals = select_non_overlapping_proposals(proposals)
        selected_proposals = selected_proposals[:20]  # Limit number of segments

        print(f"Selected proposals: {selected_proposals}")
        
        final_proposals[video_id] = adjust_segments_to_scene_changes(
            selected_proposals, scene_changes, round(video_duration, 2), min_duration=1
        )

    return final_proposals

def llm_parse_actions(reparsed_captions: list, action_list: list):
    url = f"http://localhost:{LLAMA_PORT}/llama33"

    actions_with_descriptions = [
        f"'{action}': {description} (Example: {examples})"
        for action, description, examples in action_list
    ]

    message_list = []
    SYSTEM_TMPL = (
        "You are a strict classifier. Pick exactly one action from this set:\n"
        "[{actions}]\n\n"
        "Rules:\n"
        "• Output ONLY the action label, in lowercase, no quotes, no punctuation, no explanation.\n"
        "• If there is any uncertainty, output: none\n"
        "• Do not invent labels. Choose from the set or 'none' only."
    )
    
    for caption in reparsed_captions:
        message_list.append([
            {"role": "system", "content": SYSTEM_TMPL.format(actions=", ".join(actions_with_descriptions))},
            {"role": "user", "content": (
                "Caption:\n"
                f"{caption}\n\n"
                "Return exactly one label from the set above or 'none'."
            )}
        ])
    '''for caption in reparsed_captions:
        message_list.append([
            {"role": "system", "content": prompt.format(actions=", ".join(actions_with_descriptions))},
            {"role": "user",
             "content": "This is the caption describing the video:\n"
                        f"{caption}\n"
                        "Now, based on the caption, please select the action that best matches from the provided list."}
        ])'''

    payload = {
        "messages": message_list,                            # <-- real JSON (list[list[turn]])
        "constraint_words": [a[0] for a in action_list] + ["none"],  # <-- real JSON list
        "is_batch": True
    }

    r = requests.post(url, json=payload, timeout=180)        # <-- send JSON
    r.raise_for_status()
    data = r.json()

    # Your server returns list[str] for batch; your old code expected dicts with ['content']
    responses = data["response"]                             # list[str]
    assert len(responses) == len(reparsed_captions)
    return responses

def _coerce_json_obj(text: str) -> dict:
    """
    Extract the first JSON object from the model text, even if it added prose or code fences.
    Fallback to a minimal default if parsing fails.
    """
    # remove code fences if present
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()

    # try straight parse
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # find first {...} block
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # last resort
    return {"argument0": "none", "argument1": "none"}

def llm_parse_arguments(parsed_actions, reparsed_captions, action_dict):
    url = f"http://localhost:{LLAMA_PORT}/llama33"

    message_list = []
    for act, cap in zip(parsed_actions, reparsed_captions):
        if act == "none":
            continue

        prompt = (
            "You are a strict information extractor.\n"
            "Given a caption and an action, return a SINGLE JSON object with physical entities only.\n"
            "Rules:\n"
            '• Format: {"argument0":"...", "argument1":"...", "argument2":"..."}\n'
            "• argument0 = subject/actor (or 'none' if absent)\n"
            "• argument1 = direct object/recipient (or 'none' if absent)\n"
            "• argument2 = tool/produced/indirect entity (omit if none)\n"
            "• No inference beyond what is explicitly stated; do not invent entities.\n"
            "• Output ONLY the JSON object; no prose, no code fences, no extra text."
        )

        example_use, example_sentence = action_dict.get(act, ["none", "none"])
        user = (
            f"Action: {act}\n"
            f"Usage: {example_use}\n"
            f"Example: {example_sentence}\n"
            f"Caption:\n{cap}\n"
            "Return the JSON object now."
        )

        message_list.append([
            {"role": "system", "content": prompt},
            {"role": "user",   "content": user},
        ])

    # If no actionable items, return defaults aligned to captions
    if not message_list:
        return [{"argument0":"none","argument1":"none"} for _ in reparsed_captions]

    # --- SEND JSON, not form data ---
    payload = {"messages": message_list, "is_batch": True}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    if "response" not in data:
        raise RuntimeError(f"Bad response from server: {data}")

    texts = data["response"]  # list[str], one per non-'none' action
    parsed_from_model = [_coerce_json_obj(t) for t in texts]

    # Re-align to original order (insert defaults for 'none')
    final_answer = []
    idx = 0
    for act in parsed_actions:
        if act == "none":
            final_answer.append({"argument0":"none","argument1":"none"})
        else:
            # pop next parsed object
            final_answer.append(parsed_from_model[idx] if idx < len(parsed_from_model)
                                else {"argument0":"none","argument1":"none"})
            idx += 1

    assert len(final_answer) == len(reparsed_captions)
    return final_answer

def filter_caption(objs: list, vlm_captions: list):
    filtered_captions = []
    #print(f"Caption before filter: {vlm_captions}")
    url = f'http://localhost:{LLAMA_PORT}/llama33'
    messages_list = []
    indices_mapping = {}  # To keep track of indices
    idx_counter = 0

    for idx, captions in enumerate(vlm_captions):
        for caption in captions:
            if caption.strip().lower() in ["none.", "no action."]:
                filtered_captions.append(caption)
                continue  # Skip processing this caption
            obj = objs[idx]
            retry_times = 0
            prompt = (
                f"Filter the caption to retain only elements directly related to '{obj}', including any interactions, actions, or contexts involving '{obj}'. "
                f"This includes details about entities interacting with '{obj}', actions '{obj}' performs or receives, and the setting where '{obj}' is present. "
                f"If there is no information relevant to '{obj}', respond with 'None'. "
                f"Respond in JSON format only as {{\"{obj}\": \"filtered caption\"}}. Do not include any other sentences."
            )
            message = [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": f"Here is the caption: '{caption}'. Please ensure the response includes all relevant information involving '{obj}', including interactions, actions, and contextual settings related to '{obj}'."
                }
            ]
            messages_list.append(message)
            indices_mapping[idx_counter] = (idx, obj)
            idx_counter += 1
            #print("\nmessage: \n")
            #print(str(message))
    data = {'messages': messages_list,'is_batch': True}
    retry_times = 0
    
    while retry_times < 3:
        try: 
            response = requests.post(url, json=data)
            response_contents = response.json()['response']
            if len(response_contents) == len(messages_list):
                for idx, response_content in enumerate(response_contents):
                    caption_idx, obj = indices_mapping[idx]
                    content = response_content['content']
                    try:
                        filtered_caption = json.loads(content)[obj]
                        filtered_captions.append(filtered_caption)
                    except (json.JSONDecodeError, KeyError):
                        raise ValueError("Failed to parse response.")
            else:
                raise ValueError("Response length mismatch.")
            
            break
  
        except Exception as e:
            filtered_captions = []
            retry_times += 1
            print(f"filter_caption retry: {retry_times}, error: {e}")

    if not filtered_captions:
        filtered_captions = [caption for captions in vlm_captions for caption in captions]

    return filtered_captions

def group_predictions_with_dbscan(arguments: list, actions: list, eps=0.3, min_samples=1):
    """
    Groups action/argument pairs with similar semantic meanings using embeddings
    and DBSCAN clustering based on cosine similarity.
    """
    # Combine actions and arguments into a list of dictionaries
    predictions_with_indices = [
        {"original_idx": idx, "action": actions[idx], "arguments": arguments[idx]}
        for idx in range(len(arguments))
    ]

    # Filter out predictions where action is 'none'
    filtered_predictions = [
        entry for entry in predictions_with_indices if entry["action"].lower() != "none"
    ]

    if not filtered_predictions:
        print("No valid actions after filtering. Returning empty result.")
        return {}, []

    # Apply substitutions to avoid 'none' as argument values
    for entry in filtered_predictions:
        arguments = entry['arguments']
        # First Substitution
        if arguments.get('argument1') == 'none' and arguments.get('argument2', 'none') != 'none':
            arguments['argument1'] = arguments.get('argument2')
            arguments.pop('argument2', None)

        # Second Substitution
        if arguments.get('argument1') == 'none':
            arguments.pop('argument1', None)

        # Third Substitution
        if arguments.get('argument0') == 'none':
            arguments.pop('argument0', None)

        # Remove any remaining arguments with 'none' as value
        arguments = {k: v for k, v in arguments.items() if v != 'none'}

        # Update the entry's arguments
        entry['arguments'] = arguments

    # Convert each action/argument pair to a string format for embedding
    pair_texts = []
    for entry in filtered_predictions:
        action = entry['action']
        arguments = entry['arguments']

        # Access argument0 without removing it
        argument0 = arguments.get('argument0', None)

        # Get the remaining arguments without argument0
        other_arguments = [v for k, v in arguments.items() if k != 'argument0']

        # Build the components list
        components = ([argument0] if argument0 else []) + [action] + other_arguments
        text = ' '.join(components)
        pair_texts.append(text)

    print("Pair Texts for Embedding:")
    for idx, text in enumerate(pair_texts):
        print(f"Index: {filtered_predictions[idx]['original_idx']}, Text: '{text}'")

    # Generate embeddings for each action/argument pair
    embeddings = sentence_transformer_model.encode(pair_texts)

    # Calculate the cosine similarity matrix and convert it to a distance matrix
    cosine_sim_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - cosine_sim_matrix

    # Clip any small negative values to zero
    distance_matrix = np.clip(distance_matrix, 0, None)

    # Use DBSCAN with the precomputed distance matrix
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)

    clusters_dict = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # Skip noise points
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(idx)  # Use index into filtered_predictions

    # Find the most representative string for each cluster
    cluster_representatives = {}
    for label, indices in clusters_dict.items():
        cluster_embeddings = embeddings[indices]

        mean_embedding = np.mean(cluster_embeddings, axis=0)

        distances = np.linalg.norm(cluster_embeddings - mean_embedding, axis=1)
        representative_idx_in_cluster = np.argmin(distances)
        representative_filtered_idx = indices[representative_idx_in_cluster]

        cluster_representatives[label] = (indices, representative_filtered_idx)

    return cluster_representatives, filtered_predictions

def object_detection(video_path):
    url = f"http://localhost:{OBJ_DET_PORT}/process_video"

    data = {
        'video_path': video_path
    }

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            response_data = response.json()
            return (
                response_data.get("objects", []),
                response_data.get("captions", []),
                response_data.get("videos", [])
            )
        else:
            print(f"Error {response.status_code}: {response.text}")
            return ([], [], [])
    except Exception as e:
        print(f"Request error: {e}")
        return ([], [], [])
    
def _norm_cap(s):
    s = (s or "").strip()
    return "no action" if s.lower() in {"none", "none."} else s

def proc_one_video(video_path: str, action_list: list, action_dict: dict):
    objects, vlm_captions, videos = object_detection(video_path)
    print("objects:", objects)
    if not objects:
        print(f"Failed to propose objects for {video_path}, return.")
        return {}

    print("VLM captions (raw):", vlm_captions)

    # --- ALWAYS FLATTEN to 1:1 (object, caption) pairs ---
    objects_expanded = []
    vlm_captions_expanded = []
    for obj, caps in zip(objects, vlm_captions):
        # caps is typically a list of 3 strings from Qwen
        if isinstance(caps, list):
            for cap in caps:
                # just in case any nesting sneaks in
                if isinstance(cap, list):
                    for c in cap:
                        objects_expanded.append(obj)
                        vlm_captions_expanded.append(str(c))
                else:
                    objects_expanded.append(obj)
                    vlm_captions_expanded.append(str(cap))
        else:
            objects_expanded.append(obj)
            vlm_captions_expanded.append(str(caps))

    assert len(objects_expanded) == len(vlm_captions_expanded)
    assert all(isinstance(c, str) for c in vlm_captions_expanded)

    parsed_actions = llm_parse_actions(vlm_captions_expanded, action_list)
    print(f"Parsed actions: {parsed_actions}")
    parsed_arguments = llm_parse_arguments(parsed_actions,vlm_captions_expanded,action_dict)
    print(f"Parsed arguments: {parsed_arguments}")
    assert len(parsed_arguments) == len(parsed_actions) == len(vlm_captions_expanded)
    grouped_predictions_and_reps, filtered_predictions = group_predictions_with_dbscan(parsed_arguments, parsed_actions)

    if not filtered_predictions:
        print("No valid actions detected. Skipping clustering.")
        final_predictions = {}
        return final_predictions

    total_valid = sum(len(cluster) for _, (cluster, _) in grouped_predictions_and_reps.items())

    predictions = {}
    for label, (indices, representative_filtered_idx) in grouped_predictions_and_reps.items():
        cluster_size = len(indices)
        probability = cluster_size / total_valid if total_valid > 0 else 0

        representative_entry = filtered_predictions[representative_filtered_idx]
        representative_original_idx = representative_entry['original_idx']

        # Get original indices for printing
        indices_original = [filtered_predictions[idx]['original_idx'] for idx in indices]
        print(f"Label: {label}, Original Indices: {indices_original}, Representative original idx: {representative_original_idx}")

        # Build the pattern
        pattern = {
            "action": representative_entry["action"],
            **representative_entry["arguments"]
        }

        # Determine the number of arguments in the pattern (excluding 'action')
        num_arguments = len(pattern) - 1  # Subtract 1 for the 'action' key

        # Determine the key based on the number of elements (arguments + action)
        key = f"valid_parsed_actions_{num_arguments + 1}"  # +1 to include 'action'

        caption = vlm_captions_expanded[representative_original_idx]

        # Append the structured data to the result
        if key not in predictions:
            predictions[key] = []

        predictions[key].append({
            "pattern": pattern,
            "probability": probability,
            "caption": caption
        })

    # After populating 'predictions', sort the keys
    sorted_keys = sorted(predictions.keys(), key=lambda x: int(x.split('_')[-1]))

    # Build final_predictions with sorted keys
    final_predictions = OrderedDict()
    for key in sorted_keys:
        final_predictions[key] = predictions[key]

    return final_predictions

def get_subj_obj_act_from_segments(segmentation_result, video_file_path):    
    action_df = pd.read_csv("./action_categories.csv")
    action_list_v2 = list(zip(action_df['action_name'], action_df['"as in…"'], action_df['example sentence from propbank']))    
    action_dict = {row['action_name']: (row['"as in…"'], row['example sentence from propbank']) for index, row in action_df.iterrows()}

    all_results = []
    # Iterate over each segment and send it to the object detection API
    for video_id, segments in segmentation_result.items():
        video_basename = os.path.basename(video_id)
        for idx, segment in enumerate(segments):
            start_time, end_time = segment['segment']
            
            # Extract the specific segment from the video
            temp_video_path = os.path.join('./', f'segment_{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")}.mp4')

            vr = decord.VideoReader(video_file_path, ctx=decord.cpu())
            fps = vr.get_avg_fps()
            total_frames = len(vr)

            print(f"Processing segment from {start_time} to {end_time}")
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)
            if start_frame >= end_frame:
                print(f"Adjusted start_frame ({start_frame}) is not less than end_frame ({end_frame}). Skipping segment.")
                continue

            frames = vr.get_batch(range(start_frame, end_frame)).asnumpy()

            # Save the first frame as a thumbnail
            first_frame = frames[0]
            # Generate a unique filename for the thumbnail
            thumbnail_filename = f"thumbnail_{video_basename}_{idx}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.jpg"
            thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)
            cv2.imwrite(thumbnail_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

            # Include the thumbnail filename in your results
            segment['thumbnail'] = thumbnail_filename

            # Convert frames to video using OpenCV
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()

            # Log the path of the saved video segment for verification
            print(f"Segment saved at: {temp_video_path}")
            for _ in range(3):
                try:
                    results = proc_one_video(                                 video_path=os.path.abspath(temp_video_path), 
                        action_list=action_list_v2, 
                        action_dict=action_dict)
                    print(f"Results: {results}")
                    break
                except Exception as e:
                    print(f"Error processing video {temp_video_path}: {e}")
                    continue

            subject_object_pairs = []
            for key in results:
                for item in results[key]:
                    pattern = item['pattern']
                    caption = item['caption']
                    action = pattern.get('action', 'none')
                    arg0 = pattern.get('argument0', 'none')
                    arg1 = pattern.get('argument1', 'none')
                    arg2 = pattern.get('argument2', 'none')

                    # Build the argument names list
                    argument_names = []
                    if arg0 != 'none':
                        argument_names.append(arg0)
                    if arg1 != 'none':
                        argument_names.append(arg1)
                    if arg2 != 'none':
                        argument_names.append(arg2)

                    caption = caption if caption else ''

                    # Append the dictionary to the list
                    subject_object_pairs.append({
                        'argument_names': argument_names,
                        'action': action,
                        'caption': caption
                    })

            inputs = {
                "subject_object_pairs": subject_object_pairs,
            }
            
            url = f"http://localhost:{GROUNDING_PORT}"
            with open(temp_video_path, "rb") as file:
                response = requests.post(
                    f"{url}/ground", data={"inputs": json.dumps(inputs)}, files={"video": file}
                )
                response.raise_for_status()
            print(response.json())

            all_results.append({
                "results": results,
                "api_response": response.json(),
                "thumbnail": thumbnail_filename,
                "segment": segment  # Include segment info if needed
            })

            # Clean up the temporary file
            os.remove(temp_video_path)

    return all_results
    
def get_video_duration_ffmpeg(filepath):
    probe = ffmpeg.probe(filepath)
    duration = float(probe['format']['duration'])
    return duration

def generate_intervals(duration, interval_length=5):
    intervals = []
    start = 0.0
    
    while start < duration:
        end = min(start + interval_length, duration)
        intervals.append({"segment": [start, end], "score": 1})
        start = end  # Move to the next segment
    
    return intervals

@celery_app.task(name="tasks.process_video_task")
def process_video_task(file_path, segmentation_type):
    duration = get_video_duration_ffmpeg(file_path)
    print(f"Video duration: {duration} seconds")
    
    # Segment the video and generate proposals
    if segmentation_type == 2: # no segmentation
        proposals = {file_path: [{'segment': [0, duration], 'score': 1}]}
    elif segmentation_type == 1: # uniform segmentation
        proposals = {file_path: generate_intervals(duration)}
    elif segmentation_type == 0: # segmentation model
        proposals = segment_video(file_path)
    print(f"Proposals: {proposals}")
    
    results = get_subj_obj_act_from_segments(proposals, file_path)

    # Collect the thumbnails and segments
    thumbnails = []
    for item in results:
        thumbnail = item.get('thumbnail')
        segment_info = item.get('segment')
        thumbnails.append({
            "thumbnail": thumbnail,
            "segment": segment_info['segment']  # Start and end times
        })

    os.remove(file_path)

    print(f"Thumbnails: {thumbnails}")
    return {"proposals": proposals, "results": results, "thumbnails": thumbnails}

if __name__ == "__main__":
    celery_app.conf.update(
        worker_hijack_root_logger=False,
        worker_redirect_stdouts=False,
    )
    
    for name in ["celery", "celery.worker", "celery.app.trace", "celery.redirected"]:
        celery_logger = logging.getLogger(name)
        coloredlogs.install(
            level="INFO",
            logger=celery_logger,
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            level_styles={
                'debug':    {'color': 'black'},
                'info':     {'color': 'black'},
                'warning':  {'color': 'black', 'bold': True},
                'error':    {'color': 'black', 'bold': True, 'bright': True},
                'critical': {'color': 'black', 'bold': True, 'bright': True},
            },
        )

    # Parse command-line arguments for device ID
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=int, default=0, help="GPU device ID to use")
    args = parser.parse_args()
    device_id = args.device

    # Initialize global variables with the specified device ID
    initialize_globals(device_id)

    # Start the Celery worker
    celery_app.worker_main(['worker', '--loglevel=info', '--pool=solo'])
