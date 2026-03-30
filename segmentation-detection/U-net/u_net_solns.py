# %% [code]
# =========================================================
# IMPORT LIBRARIES
# =========================================================

import torch
import torch.nn as nn
import torchvision

import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import json
from tqdm import tqdm
from scipy import ndimage
from sklearn.metrics import roc_curve, auc, f1_score

from torch.utils.data import Dataset, DataLoader

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %% [code]
# =========================================================
# DATASET PATHS
# =========================================================

SEG_PATH = "/kaggle/input/datasets/harshitabansal307/task2-preprocessed-detetcion-segmentation/preprocessed"

val_pt            = os.path.join(SEG_PATH, "val_detection_samples_balanced.pt")
category_map_path = os.path.join(SEG_PATH, "category_map.pt")

# %% [code]
# =========================================================
# LOAD DATA
# =========================================================

val_data     = torch.load(val_pt)
category_map = torch.load(category_map_path)
NUM_CLASSES  = len(category_map)   # 5

print("Val samples:", len(val_data))
print("category_map:", category_map)
print("NUM_CLASSES:", NUM_CLASSES)

# %% [code]
# =========================================================
# CATEGORY MAP + SAVE JSON (REQUIRED)
# New preprocessing: labels 1-5, background=0
# =========================================================

CLASS_NAMES = [
    "__background__",
    "short sleeve top",
    "trousers",
    "shorts",
    "long sleeve top",
    "skirt"
]

label_map = {
    "short sleeve top": 1,
    "trousers":         2,
    "shorts":           3,
    "long sleeve top":  4,
    "skirt":            5
}

with open("/kaggle/working/label_map.json", "w") as f:
    json.dump(label_map, f, indent=4)

print("NUM_CLASSES:", NUM_CLASSES)
print("label_map.json saved")

# %% [code]
# =========================================================
# POLYGON -> MASK
# Labels are 1-5 from new preprocessing — use directly
# =========================================================

def polygons_to_mask(polygons, labels, h, w):

    mask = np.zeros((h, w), dtype=np.uint8)

    for poly_set, label in zip(polygons, labels):

        if int(label) not in category_map:
            continue

        for poly in poly_set:
            poly = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [poly], int(label))

    return mask

# %% [code]
# =========================================================
# DATASET CLASS
# =========================================================

class DeepFashionDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        img = cv2.imread(sample["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape

        mask = polygons_to_mask(sample["polygons"], sample["labels"], h, w)

        if np.random.rand() > 0.5:
            img  = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        if np.random.rand() > 0.5:
            img  = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        img  = cv2.resize(img,  (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        img  = torch.tensor(img).permute(2, 0, 1).float() / 255
        mask = torch.tensor(mask).long()

        return img, mask

# %% [code]
# =========================================================
# DATALOADERS
# =========================================================

val_dataset = DeepFashionDataset(val_data)
val_loader  = DataLoader(val_dataset, batch_size=8, shuffle=False)

# %% [code]
# =========================================================
# DOUBLE CONV BLOCK
# =========================================================

class DoubleConv(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

# %% [code]
# =========================================================
# UNET MODEL
# =========================================================

class UNet(nn.Module):

    def __init__(self, n_classes):

        super().__init__()

        self.enc1 = DoubleConv(3,   64)
        self.enc2 = DoubleConv(64,  128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, n_classes + 1, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)

# %% [code]
# =========================================================
# LOAD MODELS
# All three from notebook outputs — no separate dataset needed
# Scratch: U-NET_scratch notebook
# Transfer + Finetune: U-NET_finetune_transfer notebook
# =========================================================

# ---- Scratch ----
try:
    model_scratch = UNet(NUM_CLASSES)
    model_scratch.load_state_dict(
        torch.load("/kaggle/input/notebooks/arismitamukherjee/u-net-scratch/Scratch.pth", map_location=device),
        strict=False
    )
    model_scratch = model_scratch.to(device).eval()
    print("Scratch loaded OK")
except Exception as e:
    print(f"Scratch FAILED: {e}")

# ---- Transfer ----
# enc1.net[0] was replaced with ResNet conv1 (7x7, no bias, stride=1) during training
try:
    model_transfer = UNet(NUM_CLASSES)
    model_transfer.enc1.net[0] = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False).to(device)
    model_transfer.load_state_dict(
        torch.load("/kaggle/input/notebooks/arismitamukherjee/u-net-finetune-transfer/Transfer.pth", map_location=device),
        strict=False
    )
    model_transfer = model_transfer.to(device).eval()
    print("Transfer loaded OK")
except Exception as e:
    print(f"Transfer FAILED: {e}")

# ---- Finetune ----
# Finetune was initialized as standard UNet (no enc1 patch) — load directly
try:
    model_finetune = UNet(NUM_CLASSES)
    model_finetune.load_state_dict(
        torch.load("/kaggle/input/notebooks/arismitamukherjee/u-net-finetune-transfer/Finetune.pth", map_location=device),
        strict=False
    )
    model_finetune = model_finetune.to(device).eval()
    print("Finetune loaded OK")
except Exception as e:
    print(f"Finetune FAILED: {e}")

print("All models loaded")

# %% [code]
# =========================================================
# SEGMENTATION METRICS — per-class IoU + Dice (classes 1-5)
# =========================================================

def compute_metrics(preds, masks, num_classes):

    iou_per_class  = []
    dice_per_class = []

    for cls in range(1, num_classes + 1):   # 1-5, skip background 0

        pred_cls = (preds == cls)
        mask_cls = (masks == cls)

        intersection = np.logical_and(pred_cls, mask_cls).sum()
        union        = np.logical_or(pred_cls,  mask_cls).sum()

        iou  = np.nan if union == 0 else intersection / union

        denom = pred_cls.sum() + mask_cls.sum()
        dice  = np.nan if denom == 0 else (2 * intersection) / denom

        iou_per_class.append(iou)
        dice_per_class.append(dice)

    return (
        np.nanmean(iou_per_class),
        np.nanmean(dice_per_class),
        np.array(iou_per_class),
        np.array(dice_per_class)
    )

# %% [code]
# =========================================================
# DETECTION METRICS — mAP@0.5, ROC, AUC, F1
# =========================================================

def bbox_iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1    = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2    = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def extract_pred_boxes(mask, num_classes, min_area=50):
    boxes, labels, scores = [], [], []
    for cls in range(1, num_classes + 1):
        binary       = (mask == cls).astype(np.uint8)
        labeled, num = ndimage.label(binary)
        for i in range(1, num + 1):
            region = (labeled == i)
            area   = region.sum()
            if area < min_area:
                continue
            ys, xs = np.where(region)
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
            labels.append(cls)
            scores.append(float(area) / (256 * 256))
    return boxes, labels, scores


def compute_detection_metrics(all_preds, val_data_subset, num_classes, img_size=256):

    class_tp  = {c: [] for c in range(1, num_classes + 1)}
    class_fp  = {c: [] for c in range(1, num_classes + 1)}
    class_sc  = {c: [] for c in range(1, num_classes + 1)}
    class_ngt = {c: 0  for c in range(1, num_classes + 1)}

    gt_presence   = {c: [] for c in range(1, num_classes + 1)}
    pred_presence = {c: [] for c in range(1, num_classes + 1)}

    for pred_mask, sample in zip(all_preds, val_data_subset):

        gt_boxes  = sample["boxes"]
        gt_labels = sample["labels"]

        img = cv2.imread(sample["img_path"])
        if img is None:
            continue
        oh, ow = img.shape[:2]
        sx, sy = img_size / ow, img_size / oh

        gt_boxes_scaled = [
            [int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)]
            for b in gt_boxes
        ]

        pred_boxes, pred_labels, pred_scores = extract_pred_boxes(pred_mask, num_classes)
        matched = set()

        for c in range(1, num_classes + 1):
            gt_c   = [b for b, l in zip(gt_boxes_scaled, gt_labels) if l == c]
            pred_c = [(b, s) for b, l, s in zip(pred_boxes, pred_labels, pred_scores) if l == c]

            class_ngt[c] += len(gt_c)
            gt_presence[c].append(1 if len(gt_c) > 0 else 0)
            pred_presence[c].append(max([s for _, s in pred_c], default=0.0))

            for (pb, ps) in sorted(pred_c, key=lambda x: -x[1]):
                best_iou, best_j = 0, -1
                for j, gb in enumerate(gt_c):
                    iou = bbox_iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                class_sc[c].append(ps)
                if best_iou >= 0.5 and best_j not in matched:
                    class_tp[c].append(1); class_fp[c].append(0)
                    matched.add(best_j)
                else:
                    class_tp[c].append(0); class_fp[c].append(1)

    ap_per_class = {}
    for c in range(1, num_classes + 1):
        sc  = np.array(class_sc[c])
        tp  = np.array(class_tp[c])
        fp  = np.array(class_fp[c])
        ngt = class_ngt[c]

        if len(sc) == 0 or ngt == 0:
            ap_per_class[c] = 0.0
        else:
            order  = np.argsort(-sc)
            tp_cum = np.cumsum(tp[order])
            fp_cum = np.cumsum(fp[order])
            rec    = tp_cum / (ngt + 1e-8)
            prec   = tp_cum / (tp_cum + fp_cum + 1e-8)
            ap = 0
            for t in np.linspace(0, 1, 11):
                p = prec[rec >= t].max() if (rec >= t).any() else 0
                ap += p / 11
            ap_per_class[c] = ap

    mAP = np.mean(list(ap_per_class.values()))

    auc_per_class = {}
    roc_per_class = {}
    for c in range(1, num_classes + 1):
        gt_arr   = np.array(gt_presence[c])
        pred_arr = np.array(pred_presence[c])
        if gt_arr.sum() == 0 or (1 - gt_arr).sum() == 0:
            auc_per_class[c] = float('nan')
            roc_per_class[c] = None
        else:
            fpr, tpr, _ = roc_curve(gt_arr, pred_arr)
            auc_per_class[c] = auc(fpr, tpr)
            roc_per_class[c] = (fpr, tpr)

    f1_per_class = {}
    for c in range(1, num_classes + 1):
        gt_arr   = np.array(gt_presence[c])
        pred_bin = (np.array(pred_presence[c]) > 0).astype(int)
        f1_per_class[c] = float('nan') if gt_arr.sum() == 0 else f1_score(gt_arr, pred_bin, zero_division=0)

    macro_f1 = np.nanmean(list(f1_per_class.values()))

    return mAP, ap_per_class, auc_per_class, roc_per_class, f1_per_class, macro_f1

# %% [code]
# =========================================================
# EVALUATION
# =========================================================

results     = []
MAX_BATCHES = 125

def evaluate(model, name):

    all_preds, all_masks = [], []

    for i, (imgs, masks) in enumerate(tqdm(val_loader, total=MAX_BATCHES)):

        if i >= MAX_BATCHES:
            break

        imgs = imgs.to(device)

        with torch.no_grad():
            preds = model(imgs)

        preds = torch.argmax(preds, dim=1).cpu().numpy()
        masks = masks.numpy()

        all_preds.append(preds)
        all_masks.append(masks)

    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    # --- Segmentation metrics ---
    mIoU, Dice, iou_pc, dice_pc = compute_metrics(all_preds, all_masks, NUM_CLASSES)

    # --- Detection metrics ---
    val_subset = val_data[:len(all_preds)]
    mAP, ap_pc, auc_pc, roc_pc, f1_pc, macro_f1 = compute_detection_metrics(
        all_preds, val_subset, NUM_CLASSES
    )

    results.append({
        "name": name,
        "mIoU": mIoU,
        "Dice": Dice,
        "mAP":  mAP,
        "F1":   macro_f1
    })

    print(f"\n{'='*55}")
    print(f"Results: {name}")
    print(f"{'='*55}")

    print(f"\n--- Segmentation ---")
    print(f"mIoU (macro): {mIoU:.4f}")
    print(f"Dice (macro): {Dice:.4f}")
    print(f"\nPer-class Segmentation:")
    for cls_name, iou, dice in zip(CLASS_NAMES[1:], iou_pc, dice_pc):
        iou_str  = f"{iou:.4f}"  if not np.isnan(iou)  else "nan"
        dice_str = f"{dice:.4f}" if not np.isnan(dice) else "nan"
        print(f"  {cls_name:<20} IoU: {iou_str:>8}   Dice: {dice_str:>8}")

    print(f"\n--- Detection ---")
    print(f"mAP@0.5:    {mAP:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}")
    print(f"\nPer-class Detection:")
    for c in range(1, NUM_CLASSES + 1):
        auc_str = f"{auc_pc[c]:.4f}" if not np.isnan(auc_pc[c]) else "nan"
        f1_str  = f"{f1_pc[c]:.4f}"  if not np.isnan(f1_pc[c])  else "nan"
        print(f"  {CLASS_NAMES[c]:<20} AP: {ap_pc[c]:.4f}   AUC: {auc_str}   F1: {f1_str}")

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(1, NUM_CLASSES + 1):
        if roc_pc[c] is not None:
            fpr, tpr = roc_pc[c]
            ax.plot(fpr, tpr, label=f"{CLASS_NAMES[c]} (AUC={auc_pc[c]:.2f})")
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curves — {name}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/{name}_roc.png")
    plt.close()
    print(f"ROC curve saved: /kaggle/working/{name}_roc.png")

# %% [code]
# =========================================================
# RUN EVALUATION
# =========================================================

evaluate(model_scratch,  "Scratch")
evaluate(model_transfer, "Transfer")
evaluate(model_finetune, "Finetune")

# %% [code]
# =========================================================
# MODEL COMPARISON
# =========================================================

print(f"\n{'='*65}")
print(f"MODEL COMPARISON")
print(f"{'='*65}")
print(f"{'Model':<12} {'mIoU':>8} {'Dice':>8} {'mAP@0.5':>10} {'Macro F1':>10}")
print(f"{'='*65}")
for r in results:
    print(f"{r['name']:<12} {r['mIoU']:>8.4f} {r['Dice']:>8.4f} {r['mAP']:>10.4f} {r['F1']:>10.4f}")

# %% [code]
# =========================================================
# INSTANCE EXTRACTION (FILTER SMALL NOISE)
# =========================================================

def extract_instances(mask, min_area=100):

    instances = []

    for cls in np.unique(mask):

        if cls == 0:
            continue

        binary       = (mask == cls).astype(np.uint8)
        labeled, num = ndimage.label(binary)

        for i in range(1, num + 1):

            region = (labeled == i)
            area   = region.sum()

            if area < min_area:
                continue

            ys, xs = np.where(region)

            instances.append({
                "bbox":  [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "class": int(cls),
                "area":  int(area)
            })

    return instances

# %% [code]
# =========================================================
# VISUALIZATION (MULTIPLE SAMPLES)
# =========================================================

os.makedirs("/kaggle/working/visuals", exist_ok=True)

for idx in range(5):

    sample = val_data[idx]

    img = cv2.imread(sample["img_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inp = torch.tensor(cv2.resize(img, (256, 256))).permute(2, 0, 1).float() / 255
    inp = inp.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model_finetune(inp)

    mask = torch.argmax(pred, dim=1).cpu().numpy()[0]

    instances = extract_instances(mask)
    print(f"Sample {idx} instances:", instances)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, vmin=0, vmax=NUM_CLASSES)
    plt.title(f"Prediction (Finetune)")
    plt.colorbar()

    plt.savefig(f"/kaggle/working/visuals/sample_{idx}.png")
    plt.close()

print("Saved visual outputs")

# %% [code]
# =========================================================
# SHOW OUTPUT FILES
# =========================================================

print("Saved files:")
print(os.listdir("/kaggle/working"))