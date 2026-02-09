# CNN NEXT-MOVE TRAINING

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

def _onehot_point_mask(rc, h, w, device, dtype):
    m = torch.zeros((h, w), dtype=dtype, device=device)
    r = int(rc[0]); c = int(rc[1])
    if 0 <= r < h and 0 <= c < w:
        m[r, c] = 1.0
    return m

ACTIONS_4 = [
    (-1, 0),  # 0: UP
    ( 1, 0),  # 1: DOWN
    ( 0, 1),  # 2: RIGHT
    ( 0,-1),  # 3: LEFT
]
DELTA_TO_CLASS = {d: i for i, d in enumerate(ACTIONS_4)}

def _delta_to_class_4(curr_rc, next_rc):
    dr = int(next_rc[0]) - int(curr_rc[0])
    dc = int(next_rc[1]) - int(curr_rc[1])

    if dr == 0 and dc == 0:
        return 0  # default UP

    if abs(dr) >= abs(dc):
        return DELTA_TO_CLASS[(1, 0)] if dr > 0 else DELTA_TO_CLASS[(-1, 0)]
    else:
        return DELTA_TO_CLASS[(0, 1)] if dc > 0 else DELTA_TO_CLASS[(0, -1)]

def collate_maps_next_move_4(batch):
    """
      X: [B, 4, H, W] = [occ, start_mask, goal_mask, current_mask]
      y: [B] int in {0..3} = next move class
    """
    samples = [item[0] for item in batch]

    occ_raw = torch.stack([s.map for s in samples])  # [B,H,W]
    B, H, W = occ_raw.shape
    device = occ_raw.device
    dtype = occ_raw.dtype

    occ = (occ_raw > 0).to(dtype)  # [B,H,W]

    x_list = []
    y_list = []

    for s, occ_i in zip(samples, occ):
        if s.path is None or s.path.numel() < 4:
            curr = s.start
            nxt = s.goal
        else:
            path = s.path
            L = path.size(0)
            t = random.randint(0, L - 2)
            curr = path[t]
            nxt  = path[t + 1]

        start_m = _onehot_point_mask(s.start, H, W, device, dtype)
        goal_m  = _onehot_point_mask(s.goal,  H, W, device, dtype)
        curr_m  = _onehot_point_mask(curr,    H, W, device, dtype)

        x = torch.stack([occ_i, start_m, goal_m, curr_m], dim=0)  # [4,H,W]
        y = _delta_to_class_4(curr, nxt)

        x_list.append(x)
        y_list.append(y)

    X = torch.stack(x_list, dim=0)  # [B,4,H,W]
    y = torch.tensor(y_list, dtype=torch.long, device=device)  # [B]
    return X, y

class NextMoveCNN(nn.Module):
    """
    Input:  [B, 4, H, W]
    Output: [B, 4] logits for actions (UP/DOWN/RIGHT/LEFT)
    """
    def __init__(self, in_ch=4, base=32, num_actions=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),

            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(base, base*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base*2, base*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(base*2, base*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B,C,1,1]
            nn.Flatten(),             # [B,C]
            nn.Linear(base*4, base*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base*2, num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)

@torch.no_grad()
def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def make_loaders_next_move(root, batch_size=32, num_workers=0, train_limit = 20000, seed = 42):
    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "validation")
    test_dir  = os.path.join(root, "test")

    train_ds = MapsDataset(train_dir, lazy=True)
    val_ds   = MapsDataset(val_dir, lazy=True)
    test_ds  = MapsDataset(test_dir, lazy=True)

    if train_limit is not None and len(train_ds) > train_limit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_ds), size=train_limit, replace=False)
        train_ds = Subset(train_ds, idx.tolist())

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_maps_next_move_4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_maps_next_move_4,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_maps_next_move_4,
    )
    return train_loader, val_loader, test_loader

def train_one_epoch_next(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)  # [B,4]
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        bs = X.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(logits, y) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)

@torch.no_grad()
def evaluate_next(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        loss = loss_fn(logits, y)

        bs = X.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(logits, y) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)

def fit_next_move(
    root_dataset,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    num_workers=0,
    save_path="cnn_next_move_4dir_120k.pt",
    label_smoothing=0.05,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader = make_loaders_next_move(
        root_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        train_limit = 120000
    )

    model = NextMoveCNN(in_ch=4, base=32, num_actions=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch_next(model, train_loader, optimizer, loss_fn, device)
        va_loss, va_acc = evaluate_next(model, val_loader, loss_fn, device)

        print(f"[{ep:02d}/{epochs}] "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict()}, save_path)

    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_acc = evaluate_next(model, test_loader, loss_fn, device)
    print(f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f}")

    return model

def _is_free(map_np, r, c):
    H, W = map_np.shape
    if r < 0 or r >= H or c < 0 or c >= W:
        return False
    return map_np[r, c] == 0

@torch.no_grad()
def rollout_path_next_move_4dir(model, map_np, start_rc, goal_rc, max_steps=4096, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device).eval()

    H, W = map_np.shape
    occ = (torch.tensor(map_np, dtype=torch.float32) > 0).float()  # 1 obstacle, 0 free

    start = (int(start_rc[0]), int(start_rc[1]))
    goal  = (int(goal_rc[0]),  int(goal_rc[1]))

    path = [start]
    curr = start

    for _ in range(max_steps):
        if curr == goal:
            break

        s_mask = torch.zeros((H, W), dtype=torch.float32)
        g_mask = torch.zeros((H, W), dtype=torch.float32)
        c_mask = torch.zeros((H, W), dtype=torch.float32)
        s_mask[start[0], start[1]] = 1.0
        g_mask[goal[0],  goal[1]]  = 1.0
        c_mask[curr[0],  curr[1]]  = 1.0

        X = torch.stack([occ.cpu(), s_mask, g_mask, c_mask], dim=0).unsqueeze(0).to(device)  # [1,4,H,W]
        logits = model(X)[0]  # [4]

        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        order = np.argsort(-probs)  # best first

        moved = False
        for a in order:
            dr, dc = ACTIONS_4[int(a)]
            nr, nc = curr[0] + dr, curr[1] + dc
            if _is_free(map_np, nr, nc):
                curr = (nr, nc)
                path.append(curr)
                moved = True
                break

        if not moved:
            break

    return np.array(path, dtype=np.int32)

# Entry point for CNN training (used during experiments / Google Colab)
#model = fit_next_move(
#    root_dataset="/content/map_dataset",
#    epochs=8,
#    batch_size=64,
#    lr=1e-3,
#    num_workers=2,
#    save_path="/content/cnn_next_move_4dir_120k.pt"
#  )
