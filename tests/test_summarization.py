import sys

# Add the models directory to the Python path
sys.path.append('/content/drive/MyDrive/Image_Segmentation/Project_root/models')

# Now import the function
from summarization_model import *

# Main function to run summarization
if __name__ == "__main__":
    # Path to the extracted text data file
    extracted_text_file = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/text_data_tesseract.csv"
    # Generate summaries for each object
    summarize_attributes(extracted_text_file)