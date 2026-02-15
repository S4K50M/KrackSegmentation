import os
import time
import json
import boto3
import warnings
import numpy as np
from flask import Flask, request, render_template, url_for
from PIL import Image

# ---------------------------------------------------------
# 1. SUPPRESS WARNINGS & CONFIG
# ---------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning) 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# AWS CONFIGURATION
REGION = "ap-south-1"
ENDPOINT_NAME = "dino-terrain-endpoint" 

# AWS Clients
sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
sm_client = boto3.client("sagemaker", region_name=REGION)

# Color Palette (Matches training)
PALETTE = [
    [0, 0, 0],       # 0: Background
    [128, 0, 0],     # 1: Trees
    [0, 128, 0],     # 2: Lush Bushes
    [128, 128, 0],   # 3: Dry Grass
    [0, 0, 128],     # 4: Dry Bushes
    [128, 0, 128],   # 5: Ground Clutter
    [0, 128, 128],   # 6: Logs
    [128, 128, 128], # 7: Rocks
    [64, 0, 0],      # 8: Landscape
    [192, 0, 0],     # 9: Sky
    [64, 128, 0]     # 10: Other
]

CLASS_NAMES = {
    0: "Background", 1: "Trees", 2: "Lush Bushes", 3: "Dry Grass",
    4: "Dry Bushes", 5: "Ground Clutter", 6: "flowers", 7: "logs",
    8: "Rocks", 9: "Landscape", 10: "sky"
}

# Mapping for Ground Truth (Matches your Training Data Logic)
# We map raw pixel values (0, 100, 200...) to class IDs (0, 1, 2...)
MASK_MAPPING = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 600:6, 700:7, 800:8, 7100:9, 10000:10}

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def get_endpoint_status():
    """Checks SageMaker Endpoint Status."""
    try:
        response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = response['EndpointStatus']
        if status == 'InService':
            return "online", "ONLINE"
        elif status in ['Creating', 'Updating']:
            return "loading", status.upper()
        else:
            return "offline", status.upper()
    except Exception:
        return "offline", "UNREACHABLE"

def mask_to_color(mask):
    """Converts 2D mask to 3D Color Image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        color_mask[mask == i] = color
    return Image.fromarray(color_mask)

def process_true_mask(mask_file, target_size):
    """Loads and maps the ground truth mask to class IDs."""
    # 1. Load and Resize (Nearest Neighbor to preserve values)
    mask = Image.open(mask_file).resize(target_size, Image.NEAREST)
    mask_np = np.array(mask)
    
    # 2. Map raw values to 0-10
    clean_mask = np.zeros_like(mask_np, dtype=np.uint8)
    for raw_val, class_idx in MASK_MAPPING.items():
        clean_mask[mask_np == raw_val] = class_idx
        
    return clean_mask

def calculate_metrics(pred_mask, true_mask, num_classes=11):
    """Calculates mIoU and Pixel Accuracy."""
    # 1. Pixel Accuracy
    correct = (pred_mask == true_mask).sum()
    total = true_mask.size
    pixel_acc = (correct / total) * 100

    # 2. Mean IoU
    iou_list = []
    for c in range(num_classes):
        # Create boolean masks for current class
        p_inds = (pred_mask == c)
        t_inds = (true_mask == c)
        
        if t_inds.sum() == 0:
            continue # Skip classes not present in Ground Truth
            
        intersection = (p_inds & t_inds).sum()
        union = (p_inds | t_inds).sum()
        
        if union > 0:
            iou_list.append(intersection / union)
    
    miou = (sum(iou_list) / len(iou_list)) * 100 if iou_list else 0.0
    
    return round(miou, 2), round(pixel_acc, 2)

def get_class_dist(mask):
    """Calculates class percentage."""
    total = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    dist = []
    for cid, count in zip(unique, counts):
        if cid in CLASS_NAMES and (count/total)*100 > 0.1:
            dist.append({"name": CLASS_NAMES[cid], "pct": round((count/total)*100, 1)})
    return sorted(dist, key=lambda x: x['pct'], reverse=True)

# ---------------------------------------------------------
# 3. ROUTES
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    status_class, status_text = get_endpoint_status()

    if request.method == "POST":
        file = request.files.get("file")
        mask_file = request.files.get("mask_file") # Optional Ground Truth
        
        if file:
            filename = f"upload_{int(time.time())}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            with open(filepath, "rb") as f: payload = f.read()
            
            # --- START TIMER ---
            start_time = time.time()
            try:
                response = sm_runtime.invoke_endpoint(
                    EndpointName=ENDPOINT_NAME, 
                    ContentType="application/x-image", 
                    Body=payload
                )
                duration_ms = round((time.time() - start_time) * 1000)
                
                result = json.loads(response['Body'].read().decode())
                pred_mask = np.array(result).squeeze()

                # --- METRICS LOGIC ---
                miou, pixel_acc = None, None
                true_mask_filename = None
                
                # Check if user uploaded a true mask
                if mask_file and mask_file.filename != '':
                    # Target size must match prediction shape (Width, Height)
                    target_size = (pred_mask.shape[1], pred_mask.shape[0])
                    true_mask = process_true_mask(mask_file, target_size)
                    
                    # Calculate
                    miou, pixel_acc = calculate_metrics(pred_mask, true_mask)
                    
                    # Save True Mask Image
                    true_mask_filename = f"true_mask_{int(time.time())}.png"
                    mask_to_color(true_mask).save(os.path.join(app.config['UPLOAD_FOLDER'], true_mask_filename))

                # --- SAVE PREDICTION VISUALS ---
                res_name = f"res_{int(time.time())}.png"
                mask_to_color(pred_mask).save(os.path.join(app.config['UPLOAD_FOLDER'], res_name))
                
                raw_name = f"raw_{int(time.time())}.png"
                Image.fromarray(pred_mask.astype(np.uint8)).save(os.path.join(app.config['UPLOAD_FOLDER'], raw_name))

                return render_template(
                    'index.html',
                    status_class=status_class, status_text=status_text,
                    src=url_for('static', filename=filename),
                    res=url_for('static', filename=res_name),
                    raw_mask=url_for('static', filename=raw_name),
                    # Optional Data
                    true_mask_src=url_for('static', filename=true_mask_filename) if true_mask_filename else None,
                    miou=miou,
                    pixel_acc=pixel_acc,
                    time=duration_ms, 
                    classes=get_class_dist(pred_mask),
                    class_map=CLASS_NAMES
                )
            except Exception as e:
                return f"<h3>Error: {e}</h3><p>Make sure the Endpoint '{ENDPOINT_NAME}' is InService.</p>"

    return render_template('index.html', status_class=status_class, status_text=status_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
