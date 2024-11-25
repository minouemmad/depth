import os
import cv2
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2
# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEPTH_MODEL_CONFIG = {
    'encoder': 'vitl',
    'features': 256,
    'out_channels': [256, 512, 1024, 1024]
}
CHECKPOINT_PATH = 'checkpoints/depth_anything_v2_vitl.pth'
INPUT_DIR = '/scratch/ch3451/datasets/bdd100k-evals/images/10k/train'
OUTPUT_DIR = '/scratch/mae9855/Depth-Anything-V2/depth_maps'
THRESHOLD = 0.2  # Adjust based on depth shift sensitivity

def load_model():
    model = DepthAnythingV2(**DEPTH_MODEL_CONFIG)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    return model.to(DEVICE).eval()

def compute_depth_map(model, img_path):
    raw_img = cv2.imread(img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    depth = model.infer_image(raw_img)  # HxW depth map
    return depth

def identify_depth_shifts(depth_map, threshold):
    # Compute gradients to find depth shifts
    '''By identifying areas with depth discontinuities, 
    we can narrow down the regions of the image that are likely to contain critical features or objects, 
    which can guide subsequent processing steps.'''
    grad_x = np.abs(np.gradient(depth_map, axis=1))
    grad_y = np.abs(np.gradient(depth_map, axis=0))
    shift_mask = ((grad_x > threshold) | (grad_y > threshold)).astype(np.uint8)
    return shift_mask

def mask_high_activation_regions(feature_map, shift_mask):
    # Combine feature map with depth-based shift mask
    combined_mask = np.zeros_like(feature_map, dtype=np.uint8)
    combined_mask[shift_mask == 1] = 255  # Example: Highlight shifts as white
    return combined_mask

def save_output(depth_map, shift_mask, output_path, img_name):
    depth_output_path = os.path.join(output_path, f"{img_name}_depth.png")
    shift_output_path = os.path.join(output_path, f"{img_name}_shift.png")

    # Normalize depth map for visualization
    norm_depth_map = (255 * (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())).astype(np.uint8)
    
    # Apply a colormap to make the depth map colorful
    colorful_depth_map = cv2.applyColorMap(norm_depth_map, cv2.COLORMAP_JET)  # COLORMAP_JET gives a rainbow effect

    # Save the colorful depth map
    cv2.imwrite(depth_output_path, colorful_depth_map)

    #cv2.imwrite(depth_output_path, norm_depth_map)

    # Save shift mask
    cv2.imwrite(shift_output_path, shift_mask * 255)

def process_images(input_dir, output_dir, model, threshold, max_images=30):
    """
    Process up to `max_images` from the input directory, saving depth maps and shift masks.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0  # Initialize image counter
    for img_name in os.listdir(input_dir):
        if count >= max_images:
            break  # Stop after processing max_images
        img_path = os.path.join(input_dir, img_name)
        if os.path.isfile(img_path):
            print(f"Processing {img_name}... ({count + 1}/{max_images})")
            depth_map = compute_depth_map(model, img_path)
            shift_mask = identify_depth_shifts(depth_map, threshold)
            save_output(depth_map, shift_mask, output_dir, os.path.splitext(img_name)[0])
            count += 1  # Increment counter
    print(f"Processed {count} images.")

def main():
    print("Loading model...")
    model = load_model()
    print("Processing a subset of images...")
    process_images(INPUT_DIR, OUTPUT_DIR, model, THRESHOLD, max_images=10)
    print("Processing complete. Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()

