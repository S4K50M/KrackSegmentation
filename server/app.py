import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, render_template, url_for
from PIL import Image
import torchvision.transforms as transforms

# ---------------------------------------------------------
# 1. MODEL ARCHITECTURE (Must match training exactly)
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
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
        dim = self.backbone.embed_dim 
        self.aspp = ASPPModule(dim, 256)
        self.ag3 = AttentionGate(256, dim, 128)
        self.ag2 = AttentionGate(256, dim, 128)
        self.ag1 = AttentionGate(128, dim, 64)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True)
        )
        self.up3 = ResidualBlock(256 + dim, 256)
        self.up2 = ResidualBlock(256 + dim, 128)
        self.up1 = ResidualBlock(128 + dim, 64)
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
        feats = []
        out = self.backbone.prepare_tokens_with_masks(x)
        for i, block in enumerate(self.backbone.blocks):
            out = block(out)
            if i in [2, 5, 8, 11]:
                feats.append(out[:, -num_p:, :].reshape(B, th, tw, -1).permute(0, 3, 1, 2).contiguous())
        c1, c2, c3, c4 = feats
        z4 = self.aspp(c4)
        z4_up = F.interpolate(z4, size=c3.shape[2:], mode='bilinear')
        z3 = self.up3(torch.cat([z4_up, self.ag3(z4, c3)], dim=1))
        z3_up = F.interpolate(z3, size=c2.shape[2:], mode='bilinear')
        z2 = self.up2(torch.cat([z3_up, self.ag2(z3, c2)], dim=1))
        z2_up = F.interpolate(z2, size=c1.shape[2:], mode='bilinear')
        z1 = self.up1(torch.cat([z2_up, self.ag1(z2, c1)], dim=1))
        z1_up = F.interpolate(z1, size=(H_in, W_in), mode='bilinear', align_corners=False)
        raw_feat = self.low_level_conv(x)
        return self.final_fuse(torch.cat([z1_up, raw_feat], dim=1))

# ---------------------------------------------------------
# 2. CONFIG & INITIALIZATION
# ---------------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, H = 476, 266

# Load Local Model
model = DinoTerrainMax(num_classes=11).to(DEVICE)
model.load_state_dict(torch.load("stats/best_model.pth", map_location=DEVICE))
model.eval()

PALETTE = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0]
]
CLASS_NAMES = {
    0: "Background", 1: "Trees", 2: "Lush Bushes", 3: "Dry Grass",
    4: "Dry Bushes", 5: "Ground Clutter", 6: "flowers", 7: "logs",
    8: "Rocks", 9: "Landscape", 10: "Sky"
}

MASK_MAPPING = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 600:6, 700:7, 800:8, 7100:9, 10000:10}

# ---------------------------------------------------------
# 3. UTILS
# ---------------------------------------------------------
def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        color_mask[mask == i] = color
    return Image.fromarray(color_mask)

def process_true_mask(mask_file):
    mask = Image.open(mask_file).resize((W, H), Image.NEAREST)
    mask_np = np.array(mask)
    clean_mask = np.zeros_like(mask_np, dtype=np.uint8)
    for raw_val, class_idx in MASK_MAPPING.items():
        clean_mask[mask_np == raw_val] = class_idx
    return clean_mask

def calculate_metrics(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum()
    pixel_acc = (correct / true_mask.size) * 100
    iou_list = []
    for c in range(11):
        p_inds, t_inds = (pred_mask == c), (true_mask == c)
        if t_inds.sum() == 0: continue
        inter, union = (p_inds & t_inds).sum(), (p_inds | t_inds).sum()
        if union > 0: iou_list.append(inter / union)
    miou = (sum(iou_list) / len(iou_list)) * 100 if iou_list else 0.0
    return round(miou, 2), round(pixel_acc, 2)

# ---------------------------------------------------------
# 4. ROUTES
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        mask_file = request.files.get("mask_file")
        
        if file:
            # 1. Process Input
            img = Image.open(file).convert("RGB").resize((W, H))
            img_t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img).unsqueeze(0).to(DEVICE)

            # 2. Local Inference
            start_time = time.time()
            with torch.no_grad():
                logits = model(img_t)
                pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
            duration_ms = round((time.time() - start_time) * 1000)

            # 3. Metrics and True Mask
            miou, pixel_acc, true_mask_src = None, None, None
            if mask_file and mask_file.filename != '':
                true_mask = process_true_mask(mask_file)
                miou, pixel_acc = calculate_metrics(pred_mask, true_mask)
                tm_name = f"true_{int(time.time())}.png"
                mask_to_color(true_mask).save(os.path.join(app.config['UPLOAD_FOLDER'], tm_name))
                true_mask_src = url_for('static', filename=tm_name)

            # 4. Save Prediction Visuals
            src_name = f"in_{int(time.time())}.jpg"
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], src_name))
            
            res_name = f"res_{int(time.time())}.png"
            mask_to_color(pred_mask).save(os.path.join(app.config['UPLOAD_FOLDER'], res_name))
            
            raw_name = f"raw_{int(time.time())}.png"
            Image.fromarray(pred_mask.astype(np.uint8)).save(os.path.join(app.config['UPLOAD_FOLDER'], raw_name))

            return render_template(
                'index.html',
                status_class="online", status_text="LOCAL_GPU" if DEVICE.type == 'cuda' else "LOCAL_CPU",
                src=url_for('static', filename=src_name),
                res=url_for('static', filename=res_name),
                raw_mask=url_for('static', filename=raw_name),
                true_mask_src=true_mask_src,
                miou=miou, pixel_acc=pixel_acc, time=duration_ms,
                classes=[{"name": CLASS_NAMES[cid], "pct": round((np.sum(pred_mask==cid)/pred_mask.size)*100, 1)} 
                         for cid in np.unique(pred_mask) if cid in CLASS_NAMES],
                class_map=CLASS_NAMES
            )
    return render_template('index.html', status_class="online", status_text="LOCAL_READY")

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=False)
