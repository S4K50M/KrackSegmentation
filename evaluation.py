import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as transforms

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU (scalar)."""
    # Use argmax to get class indices
    pred = torch.argmax(pred, dim=1)
    
    # Flatten for calculation
    pred = pred.view(-1)
    target = target.view(-1)

    iou_list = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        # Create boolean masks
        pred_inds = (pred == class_id)
        target_inds = (target == class_id)

        if target_inds.sum() == 0 and pred_inds.sum() == 0:
            # If class is not present in target AND not predicted, ignore it 
            # (or treat as 1.0 match, but typically ignored in mIoU)
            continue
            
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_list.append(0.0) # Avoid division by zero
        else:
            iou_list.append((intersection / union).item())

    # Return mean IoU as a simple float
    return np.mean(iou_list) if iou_list else 0.0

def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Mean Dice Score (scalar)."""
    pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    target = target.view(-1)

    dice_list = []
    for class_id in range(num_classes):
        pred_inds = (pred == class_id)
        target_inds = (target == class_id)
        
        # Skip class if not present in target to match mIoU logic (optional, but common)
        # For strict Dice, we often compute for all classes.
        
        intersection = (pred_inds & target_inds).sum().float()
        
        # Dice = 2*Inter / (Sum(Pred) + Sum(Target))
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        dice_list.append(dice_score.item())

    # Return only the mean scalar to fix the "+=" error
    return np.mean(dice_list)

def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy (scalar)."""
    pred_classes = torch.argmax(pred, dim=1)
    # Compare and convert to float, then take mean and get python scalar
    correct = (pred_classes == target).float()
    return correct.mean().item()

# Define Inverse Normalize Transform
# (x - mean) / std  ->  x * std + mean
inv_normalize = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])

def log_segmentation_sample(model, loader, epoch, device, writer):
    """
    Logs a comparison plot (Input, GT, Pred) to TensorBoard.
    """
    model.eval()
    with torch.no_grad():
        # Get one batch
        # Use try-except to handle potential iterator issues
        try:
            images, masks = next(iter(loader))
        except StopIteration:
            return

        images = images.to(device)
        masks = masks.to(device)
        
        # Inference
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1) 

        # --- Visualization Logic ---
        # 1. Prepare Input Image
        # Move to CPU, un-normalize, rearrange (C,H,W) -> (H,W,C)
        # Note: applying inv_normalize on CPU to avoid device errors if stats are strict
        img_tensor = images[0].cpu()
        img = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # 2. Prepare Masks
        gt_mask = masks[0].cpu().numpy()
        pred_mask = preds[0].cpu().numpy()

        # 3. Create Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # Use a consistent colormap range (vmin/vmax) so colors mean the same thing
        # Assuming classes 0-9
        axes[1].imshow(gt_mask, cmap='tab10', interpolation='nearest', vmin=0, vmax=9)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='tab10', interpolation='nearest', vmin=0, vmax=9)
        axes[2].set_title("Prediction")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 4. Log to TensorBoard
        writer.add_figure("Validation/Sample_Prediction", fig, epoch)
        
        # Close to free memory
        plt.close(fig)