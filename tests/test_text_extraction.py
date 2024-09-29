import sys

# Add the models directory to the Python path
sys.path.append('/content/drive/MyDrive/Image_Segmentation/Project_root/models')

# Now import the function
from text_extraction_model import *

# Main function to run text extraction
if __name__ == "__main__":
    # Path to directory where extracted object images are saved
    extracted_objects_dir = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/segmented_objects"

    # Extract text using Tesseract OCR
    extract_text_tesseract(extracted_objects_dir)