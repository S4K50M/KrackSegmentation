import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, render_template, url_for, flash, redirect
from PIL import Image
import torchvision.transforms as transforms

# ---------------------------------------------------------
# 1. MODEL ARCHITECTURE (Must match training exactly)
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(out_channels); self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels: self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
    def forward(self, x):
        res = self.shortcut(x); out = self.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out)); out += res
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
        f0 = self.b0(x); f1 = self.b1(x); f2 = self.b2(x); f3 = self.b3(x); f4 = F.interpolate(self.b4(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.project(torch.cat([f0, f1, f2, f3, f4], dim=1))

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()); self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g); x1 = self.W_x(x); psi = self.relu(g1 + F.interpolate(x1, size=g1.shape[2:], mode='bilinear'))
        return x * self.psi(psi)

class DinoTerrainMax(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg"); dim = self.backbone.embed_dim 
        self.aspp = ASPPModule(dim, 256); self.ag3 = AttentionGate(256, dim, 128); self.ag2 = AttentionGate(256, dim, 128); self.ag1 = AttentionGate(128, dim, 64)
        self.low_level_conv = nn.Sequential(nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(48), nn.ReLU(True))
        self.up3 = ResidualBlock(256 + dim, 256); self.up2 = ResidualBlock(256 + dim, 128); self.up1 = ResidualBlock(128 + dim, 64)
        self.final_fuse = nn.Sequential(nn.Conv2d(64 + 48, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, num_classes, 1))
    def forward(self, x):
        B, _, H_in, W_in = x.shape; th, tw = H_in // 14, W_in // 14; num_p = th * tw; feats = []
        out = self.backbone.prepare_tokens_with_masks(x)
        for i, block in enumerate(self.backbone.blocks):
            out = block(out)
            if i in [2, 5, 8, 11]: feats.append(out[:, -num_p:, :].reshape(B, th, tw, -1).permute(0, 3, 1, 2).contiguous())
        c1, c2, c3, c4 = feats; z4 = self.aspp(c4)
        z3 = self.up3(torch.cat([F.interpolate(z4, size=c3.shape[2:], mode='bilinear'), self.ag3(z4, c3)], dim=1))
        z2 = self.up2(torch.cat([F.interpolate(z3, size=c2.shape[2:], mode='bilinear'), self.ag2(z3, c2)], dim=1))
        z1 = self.up1(torch.cat([F.interpolate(z2, size=c1.shape[2:], mode='bilinear'), self.ag1(z2, c1)], dim=1))
        z1_up = F.interpolate(z1, size=(H_in, W_in), mode='bilinear', align_corners=False); rf = self.low_level_conv(x)
        return self.final_fuse(torch.cat([z1_up, rf], dim=1))

# ---------------------------------------------------------
# 2. CONFIG & INITIALIZATION
# ---------------------------------------------------------
app = Flask(__name__)
app.secret_key = "local_analysis_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, H = 476, 266

MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(BASE_DIR), "stats", "best_model.pth")

model = DinoTerrainMax(num_classes=11).to(DEVICE)
try:
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        model.eval()
        print(f"--- SUCCESS: Local Model Loaded ---")
except Exception as e:
    print(f"--- ERROR: {e} ---")

PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0]]
CLASS_NAMES = {0: "Background", 1: "Trees", 2: "Lush Bushes", 3: "Dry Grass", 4: "Dry Bushes", 5: "Ground Clutter", 6: "Flowers", 7: "Logs", 8: "Rocks", 9: "Landscape", 10: "Sky"}
MASK_MAPPING = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 600:6, 700:7, 800:8, 7100:9, 10000:10}

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def mask_to_color(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE): color_mask[mask == i] = color
    return Image.fromarray(color_mask)

def process_true_mask(mask_file, target_size):
    """Maps truth values based on MASK_MAPPING, leaving others unchanged."""
    with Image.open(mask_file) as img:
        if img.mode != 'L' and img.mode != 'I':
            img = img.convert('L')
        mask_res = img.resize(target_size, Image.NEAREST)
        mask_np = np.array(mask_res).astype(np.int32)
        
        clean_mask = mask_np.copy()
        for raw_val, class_idx in MASK_MAPPING.items():
            clean_mask[mask_np == raw_val] = class_idx
        return clean_mask.astype(np.uint8)

def calculate_metrics(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum()
    pixel_acc = (correct / true_mask.size) * 100
    iou_list = []
    for c in range(11):
        p_inds, t_inds = (pred_mask == c), (true_mask == c)
        if t_inds.sum() == 0: continue
        union = (p_inds | t_inds).sum()
        if union > 0: iou_list.append((p_inds & t_inds).sum() / union)
    miou = (sum(iou_list) / len(iou_list)) * 100 if iou_list else 0.0
    return round(miou, 2), round(pixel_acc, 2)

def get_class_dist(mask):
    total = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    dist = []
    for cid, count in zip(unique, counts):
        if cid in CLASS_NAMES and (count/total)*100 > 0.1:
            dist.append({"name": CLASS_NAMES[cid], "pct": round((count/total)*100, 1)})
    return sorted(dist, key=lambda x: x['pct'], reverse=True)

# ---------------------------------------------------------
# 4. ROUTES
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    status_text = "GPU_ACTIVE" if DEVICE.type == 'cuda' else "CPU_ONLY"
    status_class = "online"

    if request.method == "POST":
        try:
            file = request.files.get("file")
            mask_file = request.files.get("mask_file")

            if file:
                ts = int(time.time())
                filename = f"upload_{ts}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                img_pil = Image.open(filepath).convert("RGB").resize((W, H))
                img_t = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(img_pil).unsqueeze(0).to(DEVICE)

                # Inference Timing
                if torch.cuda.is_available(): torch.cuda.synchronize()
                start_time = time.perf_counter()
                with torch.no_grad():
                    logits = model(img_t)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    duration_ms = round((time.perf_counter() - start_time) * 1000)
                    pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

                # Metrics Logic
                miou, pixel_acc, true_mask_src = None, None, None
                if mask_file and mask_file.filename != '':
                    true_mask = process_true_mask(mask_file, (W, H))
                    miou, pixel_acc = calculate_metrics(pred_mask, true_mask)
                    
                    tm_name = f"true_{ts}.png"
                    mask_to_color(true_mask).save(os.path.join(app.config['UPLOAD_FOLDER'], tm_name))
                    true_mask_src = url_for('static', filename=tm_name)

                # Save Prediction Visuals
                res_name = f"res_{ts}.png"
                mask_to_color(pred_mask).save(os.path.join(app.config['UPLOAD_FOLDER'], res_name))
                
                raw_name = f"raw_{ts}.png"
                Image.fromarray(pred_mask.astype(np.uint8)).save(os.path.join(app.config['UPLOAD_FOLDER'], raw_name))

                return render_template(
                    'index.html',
                    status_class=status_class, status_text=status_text,
                    src=url_for('static', filename=filename),
                    res=url_for('static', filename=res_name),
                    raw_mask=url_for('static', filename=raw_name),
                    true_mask_src=true_mask_src,
                    miou=miou, pixel_acc=pixel_acc, time=duration_ms, 
                    classes=get_class_dist(pred_mask),
                    class_map=CLASS_NAMES
                )
        except Exception as e:
            return f"<h3>Local Backend Error: {e}</h3>"

    return render_template('index.html', status_class=status_class, status_text=status_text)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)
