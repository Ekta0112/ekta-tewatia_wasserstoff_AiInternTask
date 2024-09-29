# Step 1: Image Segmentation
import cv2
import numpy as np
import os
import torch
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt

# Step 1: Upload the image from your local machine ( using static path of the image)
#from google.colab import files
#uploaded = files.upload()   

# Load pre-trained Mask R-CNN model
model = models.detection.maskrcnn_resnet50_fpn(weights=True)
model.eval()

# Define transform for image input
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to get prediction (masks) from the model
def get_prediction(img_path, model, threshold=0.5):
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    masks = prediction['masks'].cpu().numpy()  # Shape: [N, 1, H, W]
    scores = prediction['scores'].cpu().numpy()

    # Filter masks by score threshold
    filtered_masks = masks[scores >= threshold]

    # Convert masks to binary
    binary_masks = [mask[0] > 0.5 for mask in filtered_masks]

    return binary_masks, scores[scores >= threshold], prediction

# Function to visualize segmentation on the original image
def visualize_segmentation(img_path, masks, output_dir="/content/drive/MyDrive/Image_Segmentation/Project_root/data/output"):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #visualized_img = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/plot_output.png"
    #Create a copy for visualization
    visual_img = img.copy()
    for mask in masks:
        # Convert mask to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Color the mask and overlay on image (green)
        colored_mask = np.zeros_like(visual_img)
        colored_mask[mask_uint8 > 0] = [0, 255, 0]
        visual_img = cv2.addWeighted(visual_img, 1, colored_mask, 0.5, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(visual_img)
    plt.axis("off")
    plt.title("Visualized Segmented Objects")
    
    # Save the plot as an image file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "segmentation_visualization.png")
    plt.savefig(output_file, bbox_inches='tight')  # Save the image

    plt.show()
    #plt.savefig(visualized_img, bbox_inches='tight')  # Save the plot as an image file
    #return visualized_img
    
# Function to save the extracted objects
def save_extracted_objects(img_path, masks, output_dir="/content/drive/MyDrive/Image_Segmentation/Project_root/data/segmented_objects"):
    
    # Remove old data from output_dir if it exists
    if os.path.exists(output_dir):
        # Remove all files in the directory
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove directory (if empty)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
    else:
        os.makedirs(output_dir)  # Create directory if it doesn't exist
    
    img = cv2.imread(img_path)
    os.makedirs(output_dir, exist_ok=True)
    object_id = 1

    for mask in masks:
        # Convert mask to uint8 for OpenCV
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Extract object using the mask
        obj = cv2.bitwise_and(img, img, mask=mask_uint8)

        # Save the extracted object as an image
        cv2.imwrite(f"{output_dir}/object_{object_id}.png", obj)

        object_id += 1

    print(f"Extracted {object_id - 1} objects and saved them to {output_dir}.")

# Main function to perform object extraction and visualization
if __name__ == "__main__":
    #img_path = list(uploaded.keys())[0]  # Get the uploaded image file name
    img_path = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/input_images/img3.png"  # Input image path
    masks, _, _ = get_prediction(img_path, model)  # Get masks from the model

    # Visualize the segmented objects
    visualize_segmentation(img_path, masks)

    # Save each object as an image
    save_extracted_objects(img_path, masks)
