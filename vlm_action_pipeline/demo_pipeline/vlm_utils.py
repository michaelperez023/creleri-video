from transformers import LogitsProcessor, LogitsProcessorList
from transformers import MllamaForConditionalGeneration, AutoProcessor
import transformers
import torch
import io
import string
import decord
import numpy as np
from PIL import Image
import requests
import json
import os

def clean_object_name(object_name: str):
    object_name = object_name.lower().strip()
    # remove punctuation
    object_name = object_name.translate(str.maketrans('', '', string.punctuation))

    return object_name

def get_first_mid_frame(video):
    video = decord.VideoReader(io.BytesIO(video))
    first_frame = video[0].asnumpy()
    mid_frame = video[len(video) // 2].asnumpy()

    first_frame = Image.fromarray(np.uint8(first_frame))
    mid_frame = Image.fromarray(np.uint8(mid_frame))

    return first_frame, mid_frame

class ObjectProposer:
    def __init__(self, model_name="meta-llama/Llama-3.2-11B-Vision-Instruct") -> None:
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def get_objects_in_video(self, video, threshold=0.7, messages=None):
        first_frame, mid_frame = get_first_mid_frame(video=video)

        default_messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "What objects are in this image? Please only response with JSON format. For example: {'objects': ['object1', 'object2']}"},
                
            ]
            }
        ]

        if messages is None:
            messages = default_messages

        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True)
        inputs = self.processor(images=[first_frame, mid_frame],
                        text=input_text, return_tensors="pt").to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=100)
        objects = self.processor.decode(
            output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        return objects


def query_vlm(video_path, obj_proposer: ObjectProposer, prompt="What is the {object} doing in the video?", return_hidden_state=False, training_object_json_path=None):
    url = 'http://127.0.0.1:5000/video_qa'

    output = []
    objects = []
    last_hidden_states = []
    for _ in range(3):
        try:
            objects = json.loads(obj_proposer.get_objects_in_video(video_path).replace("'", "\""))['objects']
            assert len(objects) > 0
            break
        except:
            continue
    
    objects = [clean_object_name(obj) for obj in objects]

    if training_object_json_path is not None:
        with open(training_object_json_path, 'r') as f:
            training_objects = json.load(f)
        # union of objects
        objects = list(set(objects) | set([clean_object_name(training_objects[os.path.basename(video_path)]['object_name'])]))

    for obj in objects:
        video_file = video_path
        question = prompt.format(object=obj)
        temperature = 0.2
        files = {'video': open(video_file, 'rb')}
        data = {'question': question, 'temperature': temperature}
        response = requests.post(url, files=files, data=data)
        print(response.json()["answer"])

        output.append(response.json()["answer"])
        if return_hidden_state:
            last_hidden_states.append(np.array(response.json()["last_hidden_states"]).mean(0).reshape([-1]))  # mean pooling or just select the first token

    if return_hidden_state:
        return output, objects, last_hidden_states
    else:
        return output, objects


class ConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, constraint_tokens):
        self.constraint_tokens = constraint_tokens

    def __call__(self, input_ids, scores):
        # Example: Modify scores to favor certain tokens
        for token in self.constraint_tokens:
            # Increasing the logit score for constraint tokens
            scores[:, token] += 10.0
        return scores

class VLMParser:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct") -> None:
        model_id = model_name

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def parse_vlm_response(self, vlm_response: str, action_list=[], prompt="You are given a list of actions: [{actions}]. Please select the action that exists in the caption. Please only answer with the action name in the list. If the action is not in the list, please type 'none'."):
        action_list = [action.replace("_", " ") for action in action_list]
        action_list = ["'"+action+"'" for action in action_list]
        messages = [
            {"role": "system", "content": prompt.format(
                actions=", ".join(action_list))},
            {"role": "user", "content": "This is the caption: \n" + vlm_response +
                "\n Now, please select the action that exists in the video: "},
        ]

        
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            constraint_tokens = [self.tokenizer(act)['input_ids'] for act in action_list] + [self.tokenizer("none")['input_ids']]
            return constraint_tokens


        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        return outputs[0]["generated_text"][-1]

    def parse_subject_object(self, caption: str, action_name: str, prompt="""You are given a caption from a video and an action name. Please select the subject and object that exists in the caption based on the action name. Please only answer with the JSON format. For example: {"subject": "subject_name", "object": "object_name"}. if there is no object in the caption given the action name, please replace 'object_name' with 'none'."""):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "This is the action: \n" + action_name + "This is the caption: \n" + caption +
                "\n Now, please output the subject and object in the caption: "},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
        )

        return outputs[0]["generated_text"][-1]
    
    def query_llm(self, messages, constraint_words: list=None):
        if constraint_words is not None:
            constraint_words = [word.strip() +" <|eot_id|>" for word in constraint_words] # <|end_of_text|>
            def prefix_allowed_tokens_fn(batch_id, input_ids):
                constraint_tokens = self.tokenizer(constraint_words, padding=True)['input_ids']
                return constraint_tokens
        else:
            prefix_allowed_tokens_fn = None    
        
        outputs = self.pipeline(messages, max_new_tokens=256, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
        return outputs[0]["generated_text"][-1]
    
    def query_llm_batch(self, messages, constraint_words: list=None):
        if constraint_words is not None:
            constraint_words = [word.strip() +" <|eot_id|>" for word in constraint_words]
            def prefix_allowed_tokens_fn(batch_id, input_ids):
                constraint_tokens = self.tokenizer(constraint_words, padding=True)['input_ids']
                return constraint_tokens
        else:
            prefix_allowed_tokens_fn = None
        
        outputs = self.pipeline(messages, max_new_tokens=256, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
        print(outputs)
        return [output[0]["generated_text"][-1] for output in outputs]