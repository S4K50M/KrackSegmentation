`import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
SEED = 42
BATCH_SIZE = 4
LR = 6e-5
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W, H = 476, 266 
DATA_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'
OUTPUT_DIR = 'stats'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 11

# Optimized Class Weights (Prioritizing Rare Classes)
CLASS_WEIGHTS = torch.tensor([
    1.0,  # Class 0
    2.0,  # Class 1
    10.0, # Class 2 (Rare - Boosted)
    3.0,  # Class 3
    5.0,  # Class 4 (Rare - Boosted)
    1.0,  # Class 5
    1.0,  # Class 6
    1.0,  # Class 7
    10.0, # Class 8 (Rare - Boosted)
    2.0,  # Class 9
    0.5   # Class 10 (Dominant - Suppressed)
]).to(DEVICE)

# Visualization Palette
PALETTE = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0]
]

# ==========================================
# 2. LOSS FUNCTIONS
# ==========================================
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels):
    if probas.numel() == 0: return probas * 0.
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0: continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        gt_sorted = fg[perm]
        grad = lovasz_grad(gt_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    return torch.stack(losses).mean() if losses else torch.tensor(0.).to(DEVICE)

class AdvancedCombinedLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        probas = F.softmax(inputs, dim=1)
        B, C, H, W = inputs.shape
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = targets.view(-1)
        valid = labels != 255
        l_loss = lovasz_softmax_flat(probas[valid], labels[valid]) if valid.sum() > 0 else 0.0
        return ce_loss + 0.6 * l_loss

# ==========================================
# 3. ARCHITECTURE (ENHANCED)
# ==========================================
class ResidualBlock(nn.Module):
    """Simple Residual Block to refine features"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True), nn.Dropout(0.3))
    def forward(self, x):
        feat0 = self.b0(x); feat1 = self.b1(x); feat2 = self.b2(x); feat3 = self.b3(x)
        feat4 = F.interpolate(self.b4(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.project(torch.cat([feat0, feat1, feat2, feat3, feat4], dim=1))

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + F.interpolate(x1, size=g1.shape[2:], mode='bilinear'))
        return x * self.psi(psi)

class DinoTerrainMax(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        dim = self.backbone.embed_dim # 768
        
        # 1. Feature Extraction
        self.aspp = ASPPModule(dim, 256)
        
        # 2. Attention Gates (Skip Connections from Backbone)
        self.ag3 = AttentionGate(256, dim, 128)
        self.ag2 = AttentionGate(256, dim, 128)
        self.ag1 = AttentionGate(128, dim, 64)
        
        # 3. New Low-Level Feature Extractor (From raw image)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True)
        )
        
        # 4. Decoder with Residual Blocks (Instead of plain Conv)
        self.up3 = ResidualBlock(256 + dim, 256)
        self.up2 = ResidualBlock(256 + dim, 128)
        self.up1 = ResidualBlock(128 + dim, 64)
        
        # 5. Final Fuse Block (Combines deep features + raw image details)
        self.final_fuse = nn.Sequential(
            nn.Conv2d(64 + 48, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        B, _, H_in, W_in = x.shape
        th, tw = H_in // 14, W_in // 14
        num_p = th * tw
        
        # --- Backbone Features ---
        feats = []
        out = self.backbone.prepare_tokens_with_masks(x)
        for i, block in enumerate(self.backbone.blocks):
            out = block(out)
            if i in [2, 5, 8, 11]:
                feats.append(out[:, -num_p:, :].reshape(B, th, tw, -1).permute(0, 3, 1, 2).contiguous())
        
        c1, c2, c3, c4 = feats # Multi-scale features
        
        # --- Deep Processing ---
        z4 = self.aspp(c4)
        
        # --- Decoding ---
        # Up 1
        z4_up = F.interpolate(z4, size=c3.shape[2:], mode='bilinear')
        z3 = self.up3(torch.cat([z4_up, self.ag3(z4, c3)], dim=1))
        
        # Up 2
        z3_up = F.interpolate(z3, size=c2.shape[2:], mode='bilinear')
        z2 = self.up2(torch.cat([z3_up, self.ag2(z3, c2)], dim=1))
        
        # Up 3
        z2_up = F.interpolate(z2, size=c1.shape[2:], mode='bilinear')
        z1 = self.up1(torch.cat([z2_up, self.ag1(z2, c1)], dim=1))
        
        # --- Final Fusion (Restoring Full Resolution) ---
        z1_up = F.interpolate(z1, size=(H_in, W_in), mode='bilinear', align_corners=False)
        raw_feat = self.low_level_conv(x) # [B, 48, H, W]
        
        # Concatenate Deep Features with Raw Low-Level Features
        final_input = torch.cat([z1_up, raw_feat], dim=1)
        
        return self.final_fuse(final_input)

# ==========================================
# 4. DATASET & HELPERS
# ==========================================
class MaskDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.img_dir = os.path.join(data_dir, 'Color_Images')
        self.msk_dir = os.path.join(data_dir, 'Segmentation')
        self.ids = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.mapping = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 600:6, 700:7, 800:8, 7100:9, 10000:10}
        self.is_train = is_train

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, id_)).convert("RGB").resize((W, H))
        msk_pil = Image.open(os.path.join(self.msk_dir, id_)).resize((W, H), Image.NEAREST)
        
        raw_msk = np.array(msk_pil)
        final_msk = np.zeros_like(raw_msk, dtype=np.uint8)
        for raw_val, class_idx in self.mapping.items():
            final_msk[raw_msk == raw_val] = class_idx

        img_t = TF.to_tensor(img)
        msk_t = torch.from_numpy(final_msk).long()
        
        if self.is_train and random.random() > 0.5:
            img_t = TF.hflip(img_t)
            msk_t = TF.hflip(msk_t)
        
        return self.norm(img_t), msk_t, id_

def calculate_iou_stats(pred, target, num_classes):
    inter_list, union_list = [], []
    pred, target = pred.view(-1), target.view(-1)
    for c in range(num_classes):
        p_inds, t_inds = (pred == c), (target == c)
        if t_inds.sum() > 0:
            inter = (p_inds & t_inds).sum().item()
            union = (p_inds | t_inds).sum().item()
            inter_list.append(inter)
            union_list.append(union)
        else:
            inter_list.append(0)
            union_list.append(0)
    return np.array(inter_list), np.array(union_list)

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        color_mask[mask == i] = color
    return color_mask

def save_comparison(img_tensor, pred_mask, gt_mask, filename):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    pred_color = mask_to_color(pred_mask.cpu().numpy())
    gt_color = mask_to_color(gt_mask.cpu().numpy())
    combined = np.hstack([img, gt_color, pred_color])
    Image.fromarray(combined).save(os.path.join(OUTPUT_DIR, filename))

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(SEED)
    
    # ---------------- TRAINING ----------------
    # Added drop_last=True to prevent Batch Norm crash on last batch
    train_loader = DataLoader(MaskDataset(DATA_DIR, is_train=True), batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(MaskDataset(VAL_DIR, is_train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = DinoTerrainMax(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = AdvancedCombinedLoss(weight=CLASS_WEIGHTS)
    
    # Optimizer now includes parameters from the new layers
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LR * 0.1},
        {'params': model.aspp.parameters(), 'lr': LR},
        {'params': model.low_level_conv.parameters(), 'lr': LR}, # New
        {'params': model.final_fuse.parameters(), 'lr': LR},     # New
        {'params': [p for n, p in model.named_parameters() if 'up' in n or 'ag' in n], 'lr': LR}
    ], weight_decay=0.05)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    scaler = GradScaler()
    
    best_iou = 0.0

    print(f"--- Starting Training on {DEVICE} ---")
    for epoch in range(EPOCHS):
        model.train()
        train_inter = np.zeros(NUM_CLASSES)
        train_union = np.zeros(NUM_CLASSES)
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, msks, _ in pbar:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(imgs)
                loss = criterion(logits, msks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                b_inter, b_union = calculate_iou_stats(preds, msks, NUM_CLASSES)
                train_inter += b_inter
                train_union += b_union
                valid = train_union > 0
                run_iou = np.mean(train_inter[valid]/train_union[valid]) if valid.any() else 0.0
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'mIoU': f"{run_iou:.4f}"})

        # --- VALIDATION ---
        model.eval()
        val_inter = np.zeros(NUM_CLASSES)
        val_union = np.zeros(NUM_CLASSES)
        with torch.no_grad():
            for imgs, msks, _ in val_loader:
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                out = model(imgs)
                pred = torch.argmax(out, dim=1)
                b_inter, b_union = calculate_iou_stats(pred, msks, NUM_CLASSES)
                val_inter += b_inter
                val_union += b_union

        present = val_union > 0
        val_iou = np.mean(val_inter[present] / val_union[present]) if present.sum() > 0 else 0.0
        print(f"Epoch {epoch+1} | Train mIoU: {run_iou:.4f} | Val mIoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(">>> Best Model Saved!")

    # ---------------- TESTING ----------------
    print("\n" + "="*30)
    print("   STARTING TEST EVALUATION")
    print("="*30)
    
    # Load Best Model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=DEVICE))
    model.eval()
    
    test_loader = DataLoader(MaskDataset(TEST_DIR, is_train=False), batch_size=1, shuffle=False)
    
    test_inter = np.zeros(NUM_CLASSES)
    test_union = np.zeros(NUM_CLASSES)
    
    with torch.no_grad():
        for i, (img, msk, img_id) in enumerate(tqdm.tqdm(test_loader, desc="Testing")):
            img, msk = img.to(DEVICE), msk.to(DEVICE)
            logits = model(img)
            pred = torch.argmax(logits, dim=1)
            
            # Metrics
            b_inter, b_union = calculate_iou_stats(pred, msk, NUM_CLASSES)
            test_inter += b_inter
            test_union += b_union
            
            # Save visual every 5th image
            if i % 5 == 0:
                save_comparison(img[0], pred[0], msk[0], f"test_result_{img_id[0]}")

    present = test_union > 0
    test_iou_per_class = test_inter / (test_union + 1e-6)
    test_mIoU = np.mean(test_iou_per_class[present])
    
    print("\nTest Results:")
    print(f"Mean IoU: {test_mIoU:.4f}")
    for c in range(NUM_CLASSES):
        if present[c]:
            print(f"Class {c}: {test_iou_per_class[c]:.4f}")
    
    print(f"\nVisual results saved to: {OUTPUT_DIR}")`
