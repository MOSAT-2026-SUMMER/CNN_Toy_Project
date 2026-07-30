"""
dataset.py — PyTorch Dataset for DrunkSoberNet

Assumes frame extraction + face-crop + resize has ALREADY been done
(see preprocessing.py) and frames are saved to disk as:

    frames_dir/
        drunk/
            IMG_3280/
                frame_001.jpg
                frame_002.jpg
                ...
            IMG_3275/
                ...
        sober/
            IMG_XXXX/
                ...

NOTE: there is no separate train/ and val/ folder on disk — this file
does the split itself (fixed seed, stratified by label) so results are
reproducible across both collaborators without touching Drive.

`frames_to_tensor` is shared with app/streamlit_app.py so frame-loading
logic isn't duplicated between training and inference.
"""

import os
import random
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

# Fixed seed so both collaborators get the identical train/val split
SPLIT_SEED = 42
VAL_RATIO = 0.2

# Videos have differing frame counts (e.g. 99 vs 100) which breaks default
# batch collation. Every item is resampled to this many frames so tensors
# stack cleanly into a batch.
NUM_FRAMES = 32


def _sample_frame_paths(frame_paths, num_frames=NUM_FRAMES):
    """
    Pick `num_frames` paths evenly spaced across the clip (with repeats if
    the clip is shorter than num_frames), preserving time order.
    """
    n = len(frame_paths)
    if n == 0:
        raise RuntimeError("Got an empty frame list — check preprocessing output.")

    indices = [round(i * (n - 1) / (num_frames - 1)) for i in range(num_frames)]
    return [frame_paths[i] for i in indices]


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


def _collect_all_samples(frames_dir):
    """
    Scan frames_dir/drunk and frames_dir/sober, one sample per video folder.
    Returns dict label_name -> list[(video_dir_path, label_idx)].
    """
    all_samples = {"sober": [], "drunk": []}  # kept separate for stratified split

    for label_name, label_idx in LABEL_MAP.items():
        label_dir = os.path.join(frames_dir, label_name)
        if not os.path.isdir(label_dir):
            continue

        for video_id in sorted(os.listdir(label_dir)):
            video_dir = os.path.join(label_dir, video_id)
            if os.path.isdir(video_dir):
                all_samples[label_name].append((video_dir, label_idx))

    return all_samples


def _stratified_split(all_samples, val_ratio, seed):
    """
    Split each label's video list into train/val separately, then combine.
    This keeps the drunk/sober ratio roughly equal in both splits, even
    if the two classes have different numbers of videos.
    """
    rng = random.Random(seed)
    train_samples, val_samples = [], []

    for label_name, samples in all_samples.items():
        samples = samples.copy()
        rng.shuffle(samples)

        n_val = max(1, int(len(samples) * val_ratio)) if len(samples) > 1 else 0
        val_samples.extend(samples[:n_val])
        train_samples.extend(samples[n_val:])

    return train_samples, val_samples


class VideoDataset(Dataset):
    """
    Each item = one video's frame sequence + its drunk/sober label.

    frames_dir should point directly at the folder containing drunk/
    and sober/ subfolders (e.g. ".../data/frames").
    """

    def __init__(self, frames_dir, split="train", val_ratio=VAL_RATIO, seed=SPLIT_SEED):
        all_samples = _collect_all_samples(frames_dir)

        total_videos = sum(len(v) for v in all_samples.values())
        if total_videos == 0:
            raise RuntimeError(
                f"No videos found under {frames_dir}/drunk or {frames_dir}/sober. "
                f"Check the path and that preprocessing.py has been run."
            )

        train_samples, val_samples = _stratified_split(all_samples, val_ratio, seed)

        if split == "train":
            self.samples = train_samples
        elif split == "val":
            self.samples = val_samples
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"'{split}' split is empty ({total_videos} total videos found). "
                f"Try a smaller val_ratio or add more data."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]

        frame_files = sorted(
            f for f in os.listdir(video_dir) if f.lower().endswith((".jpg", ".png"))
        )
        frame_paths = [os.path.join(video_dir, f) for f in frame_files]
        frame_paths = _sample_frame_paths(frame_paths)

        sequence = frames_to_tensor(frame_paths)  # [NUM_FRAMES, 3, 224, 224]
        return sequence, label


if __name__ == "__main__":
    # Quick sanity check — point this at a real frames_dir to test
    frames_dir = "data/frames"

    train_dataset = VideoDataset(frames_dir, split="train")
    val_dataset = VideoDataset(frames_dir, split="val")
    print(f"train: {len(train_dataset)} videos, val: {len(val_dataset)} videos")

    sequence, label = train_dataset[0]
    print("Sequence shape:", sequence.shape)  # e.g. torch.Size([50, 3, 224, 224])
    print("Label:", label)