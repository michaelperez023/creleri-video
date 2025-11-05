"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image

import sys
import os
import cv2  # Added for video processing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.core import YAMLConfig


def process_image(model, device, file_path, threshold=0.4):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    # Filter outputs based on the threshold
    valid_indices = scores[0] > threshold
    filtered_boxes = boxes[0][valid_indices]
    filtered_labels = labels[0][valid_indices]

    # Print bounding boxes and labels
    print("Bounding Boxes:", filtered_boxes)
    print("Labels:", filtered_labels)


def load_label_mapping(label_yaml_path):
    """
    Load label mappings from a YAML file.
    :param label_yaml_path: Path to the YAML file containing the label mappings.
    :return: Dictionary mapping label IDs to names.
    """
    import yaml

    with open(label_yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data['names']

def process_video(model, device, file_path, threshold=0.4, label_mapping=None):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    label_counter = {}

    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, _, scores = output

        # Filter outputs based on the threshold
        valid_indices = scores[0] > threshold
        filtered_labels = labels[0][valid_indices]

        # Count occurrences of each label
        for label in filtered_labels.tolist():
            label_counter[label] = label_counter.get(label, 0) + 1

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()

    # Calculate label frame coverage
    coverage_threshold = 0.7  # At least 70% of frames
    valid_labels = [
        label for label, count in label_counter.items()
        if count / frame_count >= coverage_threshold
    ]

    # Map label IDs to names
    if label_mapping:
        valid_label_names = [label_mapping[label-1] for label in valid_labels]
    else:
        valid_label_names = valid_labels

    print("Labels with >=70% frame coverage:", valid_label_names)
    print("Video processing complete.")

def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # Check if the input file is an image or a video
    file_path = args.input
    threshold = args.threshold
    if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process as image
        process_image(model, device, file_path, threshold)
        print("Image processing complete.")
    else:
        # Process as video
        label_yaml_path = "/data/enze/D-FINE/tools/inference/Objects365.yaml"
        label_mapping = load_label_mapping(label_yaml_path)
        process_video(model, device, file_path, threshold=0.5,label_mapping=label_mapping)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help="Threshold for filtering scores")
    args = parser.parse_args()
    main(args)
