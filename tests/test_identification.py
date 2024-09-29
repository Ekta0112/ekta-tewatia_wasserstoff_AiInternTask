import sys

# Add the models directory to the Python path
sys.path.append('/content/drive/MyDrive/Image_Segmentation/Project_root/models')

# Now import the function
from identification_model import *

# Main function to run object identification
if __name__ == "__main__":
    # Path to directory where extracted objects are saved
    extracted_objects_dir = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/segmented_objects"

    # Identify objects and generate descriptions
    identify_objects(extracted_objects_dir)