"""
dataset.py — PyTorch Dataset for DrunkSoberNet

Assumes frame extraction + face-crop + resize has ALREADY been done
(see preprocessing.py) and frames are saved to disk as:

    data/
        train/
            drunk/
                video_001/
                    frame_001.jpg
                    frame_002.jpg
                    ...
            sober/
                video_001/
                    ...
        val/
            drunk/
            sober/

`frames_to_tensor` is shared with app/streamlit_app.py so frame-loading
logic isn't duplicated between training and inference.
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_MAP = {"sober": 0, "drunk": 1}

# ImageNet normalization — required since the backbone uses ImageNet-pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

frame_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def frames_to_tensor(frame_paths):
    """
    Load an ordered list of frame image paths and stack them into a
    single sequence tensor.

    Args:
        frame_paths: list[str], already sorted in time order

    Returns:
        torch.Tensor of shape [T, 3, 224, 224]
    """
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        frames.append(frame_transform(img))
    return torch.stack(frames, dim=0)  # [T, 3, 224, 224]


class VideoDataset(Dataset):
    """
    Each item = one video's frame sequence + its drunk/sober label.
    """

    def __init__(self, data, split="train"):
        self.split_dir = os.path.join(data, split)
        self.samples = []  # list of (video_folder_path, label)

        for label_name, label_idx in LABEL_MAP.items():
            label_dir = os.path.join(self.split_dir, label_name)
            if not os.path.isdir(label_dir):
                continue

            for video_id in sorted(os.listdir(label_dir)):
                video_dir = os.path.join(label_dir, video_id)
                if os.path.isdir(video_dir):
                    self.samples.append((video_dir, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found under {self.split_dir}. "
                                f"Check that preprocessing.py has been run.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]

        frame_files = sorted(
            f for f in os.listdir(video_dir) if f.lower().endswith((".jpg", ".png"))
        )
        frame_paths = [os.path.join(video_dir, f) for f in frame_files]

        sequence = frames_to_tensor(frame_paths)  # [T, 3, 224, 224]
        return sequence, label


if __name__ == "__main__":
    # Quick sanity check — point this at a real data to test
    data = "data/frames"
    dataset = VideoDataset(data, split="train")
    print(f"Found {len(dataset)} videos")

    sequence, label = dataset[0]
    print("Sequence shape:", sequence.shape)  # e.g. torch.Size([50, 3, 224, 224])
    print("Label:", label)