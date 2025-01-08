import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from matplotlib import cm

# Depth model import
from depth_anything_v2.dpt import DepthAnythingV2

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
CHECKPOINT_PATH = 'checkpoints/depth_anything_v2_vitl.pth'
INPUT_DIR = '/scratch/ch3451/datasets/bdd100k-evals/images/10k/train'
OUTPUT_DIR = '/scratch/mae9855/Depth-Anything-V2/combined_heatmaps'

# Load depth model
def load_depth_model():
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    return model.to(DEVICE).eval()

# Identify depth shifts
def identify_depth_shifts(depth_map, threshold=0.2):
    grad_x = np.abs(np.gradient(depth_map, axis=1))
    grad_y = np.abs(np.gradient(depth_map, axis=0))
    return ((grad_x > threshold) | (grad_y > threshold)).astype(np.uint8)

# Modified ResNet50
class DilatedResNet50(models.resnet.ResNet):
    def __init__(self, **kwargs):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], replace_stride_with_dilation=[False, True, True], **kwargs)

    def forward(self, x, return_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if return_feat:
            return x
        return self.avgpool(x).flatten(1)

    @staticmethod
    def load_dino_weights(model, weight_path):
        checkpoint = torch.load(weight_path, map_location=DEVICE)
        state_dict = checkpoint.get("student") or checkpoint.get("teacher") or checkpoint
        model.load_state_dict(state_dict, strict=False)
        return model

# Generate heatmap
def generate_heatmap(model, img_tensor):
    with torch.no_grad():
        features = model(img_tensor, return_feat=True)
    heatmap = features.mean(dim=1).squeeze()
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(720, 1280), mode='bilinear').squeeze(0).squeeze(0)
    heatmap = np.maximum(heatmap.cpu().numpy(), 0)
    return heatmap / (heatmap.max() + 1e-6)

# Apply color map to heatmap
def apply_colormap(heatmap):
    colormap = cm.jet(heatmap)[:, :, :3]  # Convert grayscale to RGB using 'jet' colormap
    return (colormap * 255).astype(np.uint8)

# Combine heatmap and depth shift mask
def combine_heatmap_with_depth(heatmap, shift_mask):
    shift_mask_normalized = shift_mask / 255.0
    combined = heatmap * shift_mask_normalized
    return combined

# Iterative masking of features
def iterative_masking(heatmap, iterations=3):
    masked_regions = []
    for _ in range(iterations):
        max_activation = heatmap.max()
        if max_activation < 0.1:  # Stop if heatmap is too faint
            break
        max_region = (heatmap == max_activation).astype(np.uint8)
        masked_regions.append(max_region)
        heatmap = np.where(max_region, 0, heatmap)  # Suppress the region
    return masked_regions

# Process a single image
def process_image(img_path, depth_model, heatmap_model, transform, threshold=0.2):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    depth_map = depth_model.infer_image(np.array(img))
    shift_mask = identify_depth_shifts(depth_map, threshold)

    heatmap = generate_heatmap(heatmap_model, img_tensor)
    combined = combine_heatmap_with_depth(heatmap, shift_mask)

    masked_regions = iterative_masking(heatmap)

    return combined, heatmap, shift_mask, masked_regions

# Main function
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    depth_model = load_depth_model()
    heatmap_model = DilatedResNet50()
    heatmap_model = DilatedResNet50.load_dino_weights(heatmap_model, '/scratch/ch3451/share/for-pathways/dino-resnet-bdd100k-highres-epoch100.pth')
    heatmap_model.to(DEVICE).eval()

    transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for img_name in os.listdir(INPUT_DIR):
        img_path = os.path.join(INPUT_DIR, img_name)
        if os.path.isfile(img_path):
            combined, heatmap, shift_mask, masked_regions = process_image(img_path, depth_model, heatmap_model, transform)

            combined_output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_combined.png")
            heatmap_output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_heatmap.png")
            shift_mask_output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_shift.png")

            cv2.imwrite(heatmap_output_path, apply_colormap(heatmap))
            cv2.imwrite(shift_mask_output_path, shift_mask * 255)
            cv2.imwrite(combined_output_path, apply_colormap(combined))

    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

