import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import tqdm

# Import the model architecture from the train file
# Ensure train.py and evaluation.py are in the same directory
from train import DinoTerrainMax, calculate_iou_stats, MaskDataset

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, H = 476, 266 
TEST_DIR = 'data/test'
MODEL_PATH = 'stats/best_model.pth'
OUTPUT_DIR = 'stats/test_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 11

PALETTE = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0]
]

# ==========================================
# UTILS
# ==========================================
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
# MAIN EVALUATION
# ==========================================
if __name__ == "__main__":
    print(f"--- Starting Evaluation on {DEVICE} ---")
    
    # Load Model
    model = DinoTerrainMax(num_classes=NUM_CLASSES).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded weights from {MODEL_PATH}")
    else:
        print(f"ERROR: No model found at {MODEL_PATH}")
        exit()
        
    model.eval()
    
    # Dataset
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
    
    print("\n" + "="*30)
    print("      FINAL TEST RESULTS")
    print("="*30)
    print(f"Mean IoU: {test_mIoU:.4f}")
    
    class_names = [
        "Background", "Class 1", "Class 2", "Class 3", "Class 4", 
        "Class 5", "Class 6", "Class 7", "Class 8", "Class 9", "Class 10"
    ]
    
    for c in range(NUM_CLASSES):
        if present[c]:
            print(f"{class_names[c]:<15}: {test_iou_per_class[c]:.4f}")
    
    print(f"\nVisual results saved to: {OUTPUT_DIR}")