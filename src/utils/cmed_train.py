import os, re, math, json, random, glob, itertools, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import importlib

try:
    _tqdm_mod = importlib.import_module("tqdm.auto")
    tqdm = _tqdm_mod.tqdm
    trange = _tqdm_mod.trange
except Exception:
    class _TqdmShim:
        @staticmethod
        def write(msg):
            print(msg)

        def __call__(self, it, **kwargs):
            return it

    def _range_shim(*args, **kwargs):
        return range(*args)

    tqdm = _TqdmShim()
    trange = _range_shim

# Logging
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass

logger = logging.getLogger("cmed_train")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = TqdmLoggingHandler()
    _fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)

# Reproducibility
def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(10)

# Dataset root
DATA_ROOT = Path("/home/ubuntu/cmed_openface")
EMOTION_DIRS = ["anger", "disgust", "fear", "happy", "no emotion", "sad", "surprise"]

# Zone relabeling to mitigate imbalance while keeping substance for CTA for final model output
ZONE_CLASS_NAMES = ["Zone 1", "Zone 2", "Zone 3"]
EMOTION_TO_ZONE = {
    "no emotion": "Zone 1",
    "happy":      "Zone 2",
    "surprise":   "Zone 2",
    "anger":      "Zone 3",
    "sad":        "Zone 3",
    "disgust":    "Zone 3",
    "fear":       "Zone 3",
}

TARGET_FPS = 25.0

WINDOW_SECONDS = 0.6       # typical micro-expression span
WINDOW_SIZE = int(WINDOW_SECONDS * TARGET_FPS)  # frames per window
WINDOW_STRIDE = max(1, WINDOW_SIZE // 3)        # overlap

USE_DELTA = True
USE_DELTA2 = True
USE_AU_PRESENCE = False     # This can be noisy
MIN_SUCCESS_RATIO = 0.7     # discard sequences with too many failed frames if 'success' exists

# GPU Utilization (1 A10G NVIDIA GPU - g5.4xlarge AWS EC2 Instance)
BATCH_SIZE = 1024
NUM_WORKERS = 16
PIN_MEMORY = True
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Enable cuDNN autotune for best convolution performance
import torch.backends.cudnn
torch.backends.cudnn.benchmark = True

# evaluation / splitting
SPLIT_MODE = "LOSO"  # leave-one-subject-out by Person_###
VAL_SUBJECTS_FRACTION = 0.0  # We use LOSO

CLASS_NAMES = ZONE_CLASS_NAMES
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# Checkpointing
SAVE_CHECKPOINTS = True
# Base path: repo root (one level above src)
_BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = _BASE_DIR / "output" / "checkpoints"

logger.info(f"Device: {DEVICE}")
logger.info(f"Classes: {CLASS_TO_IDX}")
logger.info(f"Window size: {WINDOW_SIZE} frames")

def extract_person_id(path: Path) -> str:
    # like .../<the_emotion>/Person_123/<file>.csv  => person id = "Person_123"
    parts = path.parts
    for i in range(len(parts)-1, 0, -1):
        if parts[i].startswith("Person_"):
            return parts[i]
    # fallback: try to detect "Person_<digits>" in path
    m = re.search(r"(Person_\d+)", str(path))
    return m.group(1) if m else "Person_UNKNOWN"


def infer_fps_from_name(name: str, default=TARGET_FPS) -> float:
    # example ..._25.0.csv -> capture trailing _<fps>.csv
    m = re.search(r"_([0-9]+(?:\.[0-9]+)?)\.csv$", name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return default


def find_csv_files(data_root: Path, emotion_dirs: List[str]) -> List[Dict]:
    items = []
    for emo in emotion_dirs:
        for csv_path in data_root.joinpath(emo).rglob("*.csv"):
            # ensure it's the OpenFace 2.x per-frame CSV (not e.g., _aus.csv if present)
            if csv_path.name.endswith(".csv"):
                person = extract_person_id(csv_path)
                fps = infer_fps_from_name(csv_path.name, TARGET_FPS)
                zone = EMOTION_TO_ZONE.get(emo)
                if zone is None:
                    # skip unknown emotions
                    continue
                items.append({
                    "csv": csv_path,
                    "emotion": emo,
                    "zone": zone,
                    "label": CLASS_TO_IDX[zone],
                    "person": person,
                    "fps": fps
                })
    return items

dataset_index = find_csv_files(DATA_ROOT, EMOTION_DIRS)
logger.info(f"Found {len(dataset_index)} CSV files.")

# peek few
for x in dataset_index[:5]:
    logger.info(f"{x['csv']} {x['emotion']} -> {x.get('zone')} {x['person']} {x['fps']}")

# Load a CSV + extract AU channels
AU_R_PATTERN = re.compile(r"^AU\d{2}_r$")
AU_C_PATTERN = re.compile(r"^AU\d{2}_c$")

# Load action units into memory
def load_au_timeseries(csv_path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = pd.read_csv(csv_path)
    # Find AU columns
    au_r_cols = [c for c in df.columns if AU_R_PATTERN.match(c)]
    au_c_cols = [c for c in df.columns if AU_C_PATTERN.match(c)]

    if not au_r_cols and not au_c_cols:
        raise ValueError(f"No AU columns found in {csv_path}")

    return df, au_r_cols, au_c_cols

df0, au_r0, au_c0 = load_au_timeseries(dataset_index[0]["csv"])
logger.info(f"Example AU_r columns: {au_r0[:10]}")
logger.info(f"Example AU_c columns: {au_c0[:10]}")

# Resampling to fight class imbalance
def resample_df(df: pd.DataFrame, src_fps: float, tgt_fps: float, cols: List[str]) -> pd.DataFrame:
    if abs(src_fps - tgt_fps) < 1e-6:
        return df[cols].reset_index(drop=True)

    # We can assume rows are uniformly spaced src_fps (CMED -> 25 fps). Then we create a time index and reindex at a target fps.
    t_src = np.arange(len(df)) / src_fps
    t_tgt = np.arange(0, t_src[-1] if len(t_src) else 0, 1.0 / tgt_fps)
    if len(t_tgt) == 0:
        return pd.DataFrame(columns=cols)

    out = {}
    for c in cols:
        out[c] = np.interp(t_tgt, t_src, df[c].values.astype(float))
    return pd.DataFrame(out)

# More features to enable more complex modeling and potentially better classification output
def add_deltas(features: np.ndarray, use_delta=True, use_delta2=True) -> np.ndarray:
    feats = [features]
    if use_delta:
        # Numpy does the heavy lifting for computations
        d1 = np.diff(features, axis=0, prepend=features[:1])
        feats.append(d1)
    if use_delta2:
        # Numpy does the heavy lifting for computations
        d2 = np.diff(features, n=2, axis=0, prepend=features[:1], append=features[-1:])
        feats.append(d2)
    return np.concatenate(feats, axis=1)

# standardize z-score (normalize standard deviation)
def standardize_per_sequence(x: np.ndarray, eps=1e-6) -> np.ndarray:
    # z-score per channel over time
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    # EPS (epsilon) to prevent /0
    return (x - mu) / (sd + eps)

def compute_success_ratio(df: pd.DataFrame) -> float:
    if "success" in df.columns:
        s = df["success"].values
        return float(np.mean(s > 0.5))
    return 1.0

# For our TCN, we can accept different lengths at inference, but for fast training we will make sure things are a uniform length
def sliding_windows(x: np.ndarray, win: int, stride: int) -> np.ndarray:
    T, C = x.shape
    if T < win:
        return np.empty((0, win, C), dtype=x.dtype)
    n = 1 + (T - win) // stride
    out = np.empty((n, win, C), dtype=x.dtype)
    idx = 0
    for start in range(0, T - win + 1, stride):
        out[idx] = x[start:start+win]
        idx += 1
    return out

# Dataset class
class AUWindowDataset(Dataset):
    def __init__(self, index: List[Dict], persons_keep: List[str], target_fps=TARGET_FPS,
                 window_size=WINDOW_SIZE, stride=WINDOW_STRIDE, use_delta=USE_DELTA, use_delta2=USE_DELTA2,
                 use_presence=USE_AU_PRESENCE, min_success_ratio=MIN_SUCCESS_RATIO, training=True,
                 show_progress=True):
        self.samples = []  # list of dicts
        self.labels = []
        self._build(index, persons_keep, target_fps, window_size, stride, use_delta, use_delta2, use_presence, min_success_ratio, training, show_progress)

    def _build(self, index, persons_keep, target_fps, window_size, stride, use_delta, use_delta2, use_presence, min_success_ratio, training, show_progress):
        desc = "Building dataset (train)" if training else "Building dataset (test)"
        iterable = tqdm(index, desc=desc, leave=False, disable=not show_progress)
        for item in iterable:
            if item["person"] not in persons_keep:
                continue
            df, au_r_cols, au_c_cols = load_au_timeseries(item["csv"])
            # success ratio in OpenFace Feature Extraction outputs
            if compute_success_ratio(df) < min_success_ratio:
                continue

            base_cols = au_r_cols.copy()
            if use_presence and au_c_cols:
                base_cols += au_c_cols

            if not base_cols:
                continue

            # resample
            df_rs = resample_df(df, item["fps"], target_fps, base_cols)
            if len(df_rs) < window_size:
                continue

            x = df_rs.values.astype(np.float32)  # [T, C]
            x = add_deltas(x, use_delta, use_delta2)  # [T, C']
            x = standardize_per_sequence(x)

            # windows
            windows = sliding_windows(x, window_size, stride)  # [N, W, C']
            if len(windows) == 0:
                continue

            # convert to [C, T] for Conv 1d input
            windows = np.transpose(windows, (0, 2, 1))

            y = int(item["label"])
            for w in windows:
                self.samples.append({"x": w, "y": y})
                self.labels.append(y)

            if show_progress:
                iterable.set_postfix({"samples": len(self.samples)})

        # class balance by undersampling/oversampling as needed
        if training and self.samples:
            # compute per-class counts
            counts = np.bincount(self.labels, minlength=len(CLASS_NAMES))
            max_count = counts.max() if counts.max() > 0 else 1
            # oversample minority classes
            by_class = {c: [] for c in range(len(CLASS_NAMES))}
            for s in self.samples:
                by_class[s["y"]].append(s)
            balanced = []
            for c in range(len(CLASS_NAMES)):
                if len(by_class[c]) == 0:
                    continue
                reps = int(math.ceil(max_count / len(by_class[c])))
                expanded = (by_class[c] * reps)[:max_count]
                balanced.extend(expanded)
            random.shuffle(balanced)
            self.samples = balanced
            self.labels = [s["y"] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = torch.tensor(s["x"], dtype=torch.float32)  # [C, T]
        y = torch.tensor(s["y"], dtype=torch.long)
        return x, y

# Leave-One-Subject-Out (LOSO) split utility function
def loso_splits(index: List[Dict]) -> Dict[str, Dict[str, List[str]]]:
    persons = sorted({item["person"] for item in index})
    splits = {}
    for test_person in persons:
        train_persons = [p for p in persons if p != test_person]
        splits[test_person] = {"train": train_persons, "test": [test_person]}
    return splits

# TCN blocks
class TCNBlock(nn.Module):
    def __init__(self, channels, k=5, d=1, p=0.2):
        super().__init__()
        pad = (k - 1) * d // 2  # make sure stuff stays the same length
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k, padding=pad, dilation=d)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, padding=pad, dilation=d)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        # ReLU activation
        y = self.drop(F.relu(self.bn1(self.conv1(x))))
        # Batch Normalization for the block
        y = self.drop(self.bn2(self.conv2(y)))
        return F.relu(x + y)

# Classifier Head
class AU_TCN(nn.Module):
    def __init__(self, in_ch, num_classes, widths=(64, 64, 128), ks=5, p=0.2, dilations=(1,2,4)):
        super().__init__()
        layers = []
        c = in_ch
        # TCN Operations
        for w, d in zip(widths, dilations):
            pad = (ks - 1) * d // 2
            layers += [
                nn.Conv1d(c, w, kernel_size=ks, padding=pad, dilation=d),
                nn.ReLU(),
                TCNBlock(w, k=ks, d=d, p=p),
            ]
            c = w
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):  # x: [B, C, T]
        z = self.tcn(x)
        z = self.pool(z).squeeze(-1)
        return self.head(z)

# Training Metrics: UF1, UAR
# UF1 to check for accuracy overall but accounting for different classes
# UAR is recall averaged from all classes
from sklearn.metrics import f1_score, recall_score, classification_report

def compute_metrics(y_true, y_pred, labels=None):
    if labels is None:
        labels = list(range(len(CLASS_NAMES)))
    uf1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    uar = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"UF1": uf1, "UAR": uar}

# Focal Loss for multi-class classification with imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # [B, C], target: [B]
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        # gather per-sample prob for target class
        logpt = logp.gather(1, target.view(-1, 1)).squeeze(1)
        pt = p.gather(1, target.view(-1, 1)).squeeze(1)
        if self.alpha is not None:
            at = self.alpha[target]
        else:
            at = 1.0
        loss = -at * (1.0 - pt).pow(self.gamma) * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# Training and evaluation loops
def train_one_epoch(model, loader, optim, criterion, device=DEVICE):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Train batches", leave=False)
    for x, y in pbar:
        x = x.to(device)  # [B, C, T]
        y = y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / max(1, len(loader.dataset))
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    return total_loss / max(1, len(loader.dataset))

@torch.no_grad()
def predict(model, loader, device=DEVICE):
    model.eval()
    ys, ps = [], []
    pbar = tqdm(loader, desc="Eval batches", leave=False)
    for x, y in pbar:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        ys.extend(y.numpy().tolist())
        ps.extend(pred)
    return np.array(ys), np.array(ps)

# Build one LOSO removal per and train
def _ensure_ckpt_dir():
    try:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.info(f"Could not create checkpoint directory {CHECKPOINT_DIR}: {e}")

# Save model checkpoints
def _save_checkpoint(model, optimizer, epoch, metrics, test_person, tag, extra_meta=None):
    if not SAVE_CHECKPOINTS:
        return None
    _ensure_ckpt_dir()
    ckpt_path = CHECKPOINT_DIR / f"{test_person}_{tag}.pt"
    try:
        state = {
            "epoch": int(epoch),
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
            "class_names": CLASS_NAMES,
        }
        if extra_meta:
            state["meta"] = extra_meta
        torch.save(state, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")
    except Exception as e:
        logger.info(f"Failed to save checkpoint to {ckpt_path}: {e}")
    return str(ckpt_path)


def run_loso_fold(index, test_person, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, wd=WEIGHT_DECAY,
                  widths=(64,64,128), ks=5, p=DROPOUT, dilations=(1,2,4), verbose=True):
    persons_all = sorted({it["person"] for it in index})
    train_persons = [p for p in persons_all if p != test_person]
    test_persons = [test_person]

    # Load datasets
    train_ds = AUWindowDataset(index, persons_keep=train_persons, training=True, show_progress=verbose)
    test_ds  = AUWindowDataset(index, persons_keep=test_persons,  training=False, show_progress=verbose)

    if len(train_ds) == 0 or len(test_ds) == 0:
        if verbose:
            logger.info(f"[{test_person}] Skipping: empty train or test set. Train={len(train_ds)} Test={len(test_ds)}")
        return None

    in_ch = train_ds[0][0].shape[0]  # channels
    num_classes = len(CLASS_NAMES)

    # DataLoader for training and testing
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader  = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # The model
    model = AU_TCN(in_ch=in_ch, num_classes=num_classes, widths=widths, ks=ks, p=p, dilations=dilations).to(DEVICE)
    # class weights to mitigate imbalance (computed on train labels)
    counts = np.bincount([s["y"] for s in train_ds.samples], minlength=num_classes)
    weights = 1.0 / np.maximum(1, counts)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction="mean")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Metrics and data for checkpoints etc.
    best_metrics = {"UF1": 0.0, "UAR": 0.0}
    best_state = None
    meta = {
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": wd,
        "widths": tuple(widths),
        "kernel_size": ks,
        "dropout": p,
        "dilations": tuple(dilations),
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "use_delta": USE_DELTA,
        "use_delta2": USE_DELTA2,
        "use_au_presence": USE_AU_PRESENCE,
        "target_fps": TARGET_FPS,
        "input_channels": in_ch,
        "num_classes": num_classes,
        "test_person": test_person,
        "device": DEVICE,
    }

    # Epoch loop
    epoch_iter = tqdm(range(1, epochs+1), desc=f"{test_person} epochs", leave=False, disable=not verbose)
    for epoch in epoch_iter:
        tr_loss = train_one_epoch(model, train_loader, optim, criterion)
        y_true, y_pred = predict(model, test_loader)
        metrics = compute_metrics(y_true, y_pred)
        # Helpful info
        if verbose:
            epoch_iter.set_postfix({"loss": f"{tr_loss:.4f}", "UF1": f"{metrics['UF1']:.3f}", "UAR": f"{metrics['UAR']:.3f}"})
            logger.info(f"[{test_person}] Epoch {epoch:02d} | loss={tr_loss:.4f} | UF1={metrics['UF1']:.3f} UAR={metrics['UAR']:.3f}")
        # early stopping on UF1 as early stopping metric
        if metrics["UF1"] >= best_metrics["UF1"]:
            best_metrics = metrics
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            # Save/improve best checkpoint for this section (meaning best performance on validation set with this person removed)
            _save_checkpoint(model, optim, epoch, metrics, test_person, tag="best", extra_meta=meta)

    # final report
    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = predict(model, test_loader)
    metrics = compute_metrics(y_true, y_pred)
    # Save final model
    _save_checkpoint(model, optim, epochs, metrics, test_person, tag="final", extra_meta=meta)
    if verbose:
        logger.info(f"[{test_person}] Best UF1={best_metrics['UF1']:.3f} UAR={best_metrics['UAR']:.3f}")
        logger.info(f"[{test_person}] Final  UF1={metrics['UF1']:.3f} UAR={metrics['UAR']:.3f}")
        logger.info("\n" + classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0,
        ))
    return metrics

# Run LOSO across all data
def run_full_loso(index):
    splits = loso_splits(index)
    per_fold = []
    fold_iter = tqdm(list(splits.keys()), desc="LOSO folds", leave=False)
    for test_person in fold_iter:
        logger.info(f"\n== LOSO test person: {test_person} ===")
        m = run_loso_fold(index, test_person)
        if m is not None:
            per_fold.append(m)

    # Logger data stuff
    if per_fold:
        uf1s = [m["UF1"] for m in per_fold]
        uars = [m["UAR"] for m in per_fold]
        logger.info("\n=== LOSO Summary ===")
        logger.info(f"UF1 mean={np.mean(uf1s):.3f} ± {np.std(uf1s):.3f}")
        logger.info(f"UAR mean={np.mean(uars):.3f} ± {np.std(uars):.3f}")
        # Save summary JSON
        if SAVE_CHECKPOINTS:
            _ensure_ckpt_dir()
            summary = {
                "UF1_mean": float(np.mean(uf1s)),
                "UF1_std": float(np.std(uf1s)),
                "UAR_mean": float(np.mean(uars)),
                "UAR_std": float(np.std(uars)),
                "per_fold": per_fold,
                "class_names": CLASS_NAMES,
            }
            try:
                with open(CHECKPOINT_DIR / "loso_summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Saved LOSO summary to {CHECKPOINT_DIR / 'loso_summary.json'}")
            except Exception as e:
                logger.info(f"Failed to write LOSO summary JSON: {e}")
    else:
        logger.info("No valid folds were run (check data paths).")

# Model training (no notebook style)
if __name__ == "__main__":
    logger.info("Building LOSO and training...")
    run_full_loso(dataset_index)
