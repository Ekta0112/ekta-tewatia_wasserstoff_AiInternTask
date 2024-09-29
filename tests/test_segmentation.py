import sys

# Add the models directory to the Python path
sys.path.append('/content/drive/MyDrive/Image_Segmentation/Project_root/models')

# Now import the function
from segmentation_model import *

# Main function to perform object extraction and visualization
if __name__ == "__main__":
    #img_path = list(uploaded.keys())[0]  # Get the uploaded image file name
    img_path = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/input_images/img3.png"  # Input image path
    masks, _, _ = get_prediction(img_path, model)  # Get masks from the model

    # Visualize the segmented objects
    visualize_segmentation(img_path, masks)

    # Save each object as an image
    save_extracted_objects(img_path, masks)