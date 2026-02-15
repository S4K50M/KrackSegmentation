import io
import base64
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "../stats/best_model_test1.pth"
IMG_WIDTH = int(((960 / 2) // 14) * 14)
IMG_HEIGHT = int(((540 / 2) // 14) * 14)
NUM_CLASSES = 10

# --- Labels & Palette ---
LABELS = {
    0: "Background", 1: "Trees", 2: "Lush Bushes", 3: "Dry Grass",
    4: "Dry Bushes", 5: "Ground Clutter", 6: "Logs", 7: "Rocks",
    8: "Landscape", 9: "Sky"
}

PALETTE = np.array([
    [0, 0, 0],         # 0: Background
    [34, 139, 34],     # 1: Trees
    [50, 205, 50],     # 2: Lush Bushes
    [218, 165, 32],    # 3: Dry Grass
    [139, 69, 19],     # 4: Dry Bushes
    [128, 0, 0],       # 5: Ground Clutter
    [160, 82, 45],     # 6: Logs
    [112, 128, 144],   # 7: Rocks
    [72, 61, 139],     # 8: Landscape
    [135, 206, 235],   # 9: Sky
], dtype=np.uint8)

# --- Model Definitions ---
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, dilation=3, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        u = x.clone(); attn = self.conv0(x); attn = self.conv_spatial(attn); attn = self.conv1(attn)
        return u * attn

class DeepComplexDecoder(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_classes):
        super().__init__()
        self.refiners = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, embedding_dim, 1), nn.GroupNorm(32, embedding_dim), nn.ReLU())
            for _ in range(4) 
        ])
        self.lka = LKA(embedding_dim * 4)
        self.fusion = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, embedding_dim), nn.SiLU(), nn.Dropout2d(0.3)
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(embedding_dim // 2, num_classes, 1)
        )
        self.classifier = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, feature_list, train_mode=True):
        projected = [self.refiners[i](f) for i, f in enumerate(feature_list)]
        out = torch.cat(projected, dim=1)
        out = self.lka(out)
        out = self.fusion(out)
        if train_mode: return self.classifier(out), self.aux_head(projected[-2])
        return self.classifier(out)

class Dinov2AdvancedTerrainModel(nn.Module):
    def __init__(self, num_classes, img_size=(540, 960), backbone_size="base"):
        super().__init__()
        backbone_archs = {"small": "vits14", "base": "vitb14_reg", "large": "vitl14_reg"}
        print(f"Loading Backbone: {backbone_archs[backbone_size]}")
        self.backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_{backbone_archs[backbone_size]}")
        self.feature_indices = [2, 5, 8, 11] 
        self.token_h, self.token_w = img_size[0] // 14, img_size[1] // 14
        self.decoder = DeepComplexDecoder(self.backbone.embed_dim, 256, num_classes)

    def forward(self, x):
        feats = []
        out = self.backbone.prepare_tokens_with_masks(x)
        for i, block in enumerate(self.backbone.blocks):
            out = block(out)
            if i in self.feature_indices:
                t = out[:, -(self.token_h * self.token_w):, :] 
                t = t.reshape(-1, self.token_h, self.token_w, self.backbone.embed_dim).permute(0, 3, 1, 2)
                feats.append(t)
        decoder_output = self.decoder(feats, train_mode=False)
        return F.interpolate(decoder_output, size=x.shape[2:], mode='bilinear', align_corners=False)

# --- App Logic ---
app = Flask(__name__)

# NOTE: No upload folders needed anymore.

model = None
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def init_model():
    global model
    model = Dinov2AdvancedTerrainModel(NUM_CLASSES, (IMG_HEIGHT, IMG_WIDTH), backbone_size="base")
    try:
        print(f"Loading weights from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Could not load model weights ({e}). Initializing with random weights.")
    
    model.to(DEVICE).eval()
    # Warmup
    model(torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(DEVICE))
    print("System Ready.")

def colorize_mask(mask):
    """Converts (H,W) index mask to PIL Image."""
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        img[mask == i] = color
    return Image.fromarray(img)

def image_to_base64(pil_img):
    """Encodes PIL image to Base64 string for HTML display."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename: return 'Error', 400
        
        # 1. Read Image from RAM
        img_bytes = file.read()
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except:
            return "Invalid Image", 400

        # Create display version of source image
        # (Optional: resize large images to save bandwidth)
        display_img = img.resize((IMG_WIDTH, IMG_HEIGHT)) 
        src_b64 = image_to_base64(display_img)

        # 2. Inference
        t_start = time.time()
        with torch.no_grad():
            logits = model(transform(img).unsqueeze(0).to(DEVICE))
            duration = round((time.time() - t_start) * 1000, 1)
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
            
            # Metric: Confidence
            probs = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            avg_confidence = round(max_probs.mean().item() * 100, 2)

        
        # Metric: Distribution
        total_pixels = pred_mask.size
        unique_ids, counts = np.unique(pred_mask, return_counts=True)
        class_stats = []
        for uid, count in zip(unique_ids, counts):
            if uid == 0: continue 
            if uid in LABELS:
                pct = round((count / total_pixels) * 100, 1)
                class_stats.append({"name": LABELS[uid], "pct": pct, "id": int(uid)})
        class_stats.sort(key=lambda x: x['pct'], reverse=True)

        # 3. Generate Result Images in RAM
        res_pil = colorize_mask(pred_mask)
        res_b64 = image_to_base64(res_pil)

        raw_pil = Image.fromarray(pred_mask.astype(np.uint8))
        raw_b64 = image_to_base64(raw_pil)
        
        # Return Base64 strings instead of file paths
        return render_template('index.html', 
                               src=src_b64, 
                               res=res_b64, 
                               raw_mask=raw_b64,
                               time=duration, 
                               confidence=avg_confidence, 
                               classes=class_stats)
    
    return render_template('index.html')

if __name__ == '__main__':
    init_model()
    app.run(host='0.0.0.0', port=5000)