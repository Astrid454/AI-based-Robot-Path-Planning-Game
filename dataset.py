import torch
import numpy as np
import os
#import cv2
import shutil
import uuid
#import matplotlib.pyplot as plt
#import random
from torch.utils.data import Dataset
from random import shuffle
from shutil import move


def split_dataset(
    datapath,
    train_p,
    val_p,
    test_p,
    train_name="train",
    test_name="test",
    validation_name="validation",
):
    datapath = os.path.abspath(datapath)
    assert abs(1 - train_p - val_p - test_p) < 1e-5, "Percentage must sum to 1"

    files = [
        f for f in os.listdir(datapath)
        if f.endswith(".pt") and os.path.isfile(os.path.join(datapath, f))
    ]

    n = len(files)
    train_id = int(round(n * train_p))
    val_id = int(round(n * (train_p + val_p)))

    if train_id < 1 or val_id < 1:
        train_id = val_id = n

    train_files = files[:train_id]
    val_files = files[train_id:val_id]
    test_files = files[val_id:]

    assert len(train_files) + len(val_files) + len(test_files) == n, "Error in splitting data"

    split_names = (train_name, validation_name, test_name)
    split_lists = (train_files, val_files, test_files)

    for folder_name, split in zip(split_names, split_lists):
        folder_path = os.path.join(datapath, folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        for filename in split:
            src = os.path.join(datapath, filename)
            dst = os.path.join(folder_path, filename)
            shutil.move(src, dst)


def chunk_folder(datapath, chunk_size=1000):
    datapath = os.path.abspath(datapath)
    base_name = "data_"

    current_chunk = []
    chunk_count = 0

    for entry in os.scandir(datapath):
        if os.path.isfile(entry.path):
            current_chunk.append(entry.name)

        if len(current_chunk) == chunk_size:
            dirname = base_name + str(chunk_count)
            os.mkdir(os.path.join(datapath, dirname))

            for fname in current_chunk:
                move(os.path.join(datapath, fname), os.path.join(datapath, dirname, fname))

            chunk_count += 1
            current_chunk = []

    if current_chunk:
        dirname = base_name + str(chunk_count)
        os.mkdir(os.path.join(datapath, dirname))
        for fname in current_chunk:
            move(os.path.join(datapath, fname), os.path.join(datapath, dirname, fname))


def walk_files(datapath):
    datapath = os.path.abspath(datapath)

    for entry in os.scandir(datapath):
        if os.path.isfile(entry.path):
            yield entry.path
        elif os.path.isdir(entry.path):
            for nested in walk_files(entry.path):
                yield nested



def collate_maps(batch):
    """
    Return:
      maps   -> [B, 1, H, W]
      starts -> [B, 2]
      goals  -> [B, 2]
      paths  -> [B, H, W]
    """
    samples = [item[0] for item in batch]

    maps = torch.stack([s.map for s in samples]).unsqueeze(1)
    starts = torch.stack([s.start for s in samples])
    goals = torch.stack([s.goal for s in samples])

    masks = []
    for s in samples:
        mask = torch.zeros_like(s.map, dtype=s.map.dtype)
        if s.path.numel() > 0:
            mask[s.path[:, 0], s.path[:, 1]] = 1.0
        masks.append(mask)

    paths = torch.stack(masks)
    return maps, starts, goals, paths


def collate_maps_with_info(batch):
    filenames = [item[1] for item in batch]
    path_array = [item[0].path for item in batch]

    out = list(collate_maps(batch))
    out.extend(filenames)
    out.extend(path_array)
    return out


def _onehot_point_mask(rc, h, w, device, dtype):
    m = torch.zeros((h, w), dtype=dtype, device=device)
    r = int(rc[0])
    c = int(rc[1])
    if 0 <= r < h and 0 <= c < w:
        m[r, c] = 1.0
    return m


def collate_maps_cnn(batch):
    """
    Input X:  [B, 3, H, W] = [occ, start_mask, goal_mask]
    Label Y:  [B, 1, H, W] = path_mask
    """
    samples = [item[0] for item in batch]

    occ = torch.stack([s.map for s in samples])  # [B,H,W]
    B, H, W = occ.shape
    device = occ.device
    dtype = occ.dtype

    start_masks = []
    goal_masks = []
    path_masks = []

    for s in samples:
        start_masks.append(_onehot_point_mask(s.start, H, W, device, dtype))
        goal_masks.append(_onehot_point_mask(s.goal, H, W, device, dtype))

        pm = torch.zeros((H, W), dtype=dtype, device=device)
        if s.path.numel() > 0:
            pm[s.path[:, 0], s.path[:, 1]] = 1.0
        path_masks.append(pm)

    start_masks = torch.stack(start_masks)            # [B,H,W]
    goal_masks  = torch.stack(goal_masks)             # [B,H,W]
    y = torch.stack(path_masks).unsqueeze(1)          # [B,1,H,W]

    x = torch.stack([occ, start_masks, goal_masks], dim=1)  # [B,3,H,W]
    return x, y


def build_grid(h, w, to_torch=True):
    coords = np.mgrid[0:h, 0:w].reshape(2, -1)
    grid = np.stack((coords[0], coords[1]), axis=1)
    if to_torch:
        grid = torch.tensor(grid)
    return grid


def neighbor_at_index(r, c, h, w, idx):
    grid = np.mgrid[max(0, r - 1):min(h, r + 2),
                    max(0, c - 1):min(w, c + 2)]
    neighbors = np.stack((grid[0].ravel(), grid[1].ravel())).T
    return neighbors[idx]


WHITE = np.array((255, 255, 255), dtype=np.uint8)
RED = np.array((0, 0, 255), dtype=np.uint8)
GREEN = np.array((0, 255, 0), dtype=np.uint8)
BLUE = np.array((255, 0, 0), dtype=np.uint8)

BLACK = (0, 0, 0)
PURPLE = (255, 0, 255)
DARK_BLUE = (139, 0, 0)


class MapSample(object):
    def __init__(self, map, start, goal, path, device=None):
        super(MapSample, self).__init__()

        if device is None:
            self._device = "cpu" 
        else:
            self._device = device

        self.map = torch.tensor(map, dtype=torch.float32, device=self._device)
        self.start = torch.tensor(start, dtype=torch.long, device=self._device)
        self.goal = torch.tensor(goal, dtype=torch.long, device=self._device)
        self.path = torch.tensor(path, dtype=torch.long, device=self._device)

    def to(self, device):
        self._device = device
        self.map = self.map.to(device)
        self.start = self.start.to(device)
        self.goal = self.goal.to(device)
        self.path = self.path.to(device)

    def save(self, path=None):
        self.to("cpu")
        if path is None:
            path = str(uuid.uuid4()) + ".pt"
        torch.save(self, path)

    @staticmethod
    def load(path):
        try:
            sample = torch.load(path, weights_only=False)
        except IOError as e:
            print(e)
            sample = None
        return sample

    def numpy(self):
        return (
            self.map.cpu().detach().numpy(),
            self.start.cpu().detach().numpy(),
            self.goal.cpu().detach().numpy(),
            self.path.cpu().detach().numpy(),
        )

    def bgr_map(self, start_color=GREEN, goal_color=DARK_BLUE, path_color=PURPLE):
        map_np, start_np, goal_np, path_np = self.numpy()
        return MapSample.get_bgr_map(map_np, start_np, goal_np, path_np,
                                     start_color=start_color, goal_color=goal_color, path_color=path_color)

    @staticmethod
    def get_bgr_map(map, start, goal, path,
                    start_color=GREEN, goal_color=DARK_BLUE, path_color=PURPLE,
                    remove_first_path=True):
        h, w = map.shape

        if remove_first_path and len(path) > 0:
            path = path[1:]

        if isinstance(path, (list, tuple)):
            path = np.array(path)

        bgr = np.ones((h, w, 3), dtype=np.uint8) * 255

        idx = np.argwhere(map > 0).reshape(-1, 2)
        if idx.size > 0:
            bgr[idx[:, 0], idx[:, 1]] = BLACK

        if isinstance(path, np.ndarray) and path.size > 0:
            bgr[path[:, 0], path[:, 1]] = path_color

        bgr[start[0], start[1]] = start_color
        bgr[goal[0], goal[1]] = goal_color

        return bgr


class MapsDataset(Dataset):
    def __init__(self, datapath, lazy=True):
        super(MapsDataset, self).__init__()

        datapath = os.path.abspath(datapath)

        self.samples = list(walk_files(datapath))
        shuffle(self.samples)

        self._lazy = lazy

        if not lazy:
            self.samples = [MapSample.load(p) for p in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._lazy:
            filepath = self.samples[idx]
            return MapSample.load(filepath), filepath
        else:
            sample = self.samples[idx]
            return sample, sample


