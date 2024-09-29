import pytesseract
from PIL import Image
import os
import pandas as pd

# Function to extract text from images using Tesseract
def extract_text_tesseract(image_dir, output_file="/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/text_data_tesseract.csv"):
    extracted_text_data = []
    object_id = 1

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            print(f"Extracting text from {img_path}...")

            # Load the image using PIL
            img = Image.open(img_path)

            # Extract text using Tesseract
            text = pytesseract.image_to_string(img)

            # Print the extracted text
            print(f"Text from {filename}: {text.strip()}")

            # Store the extracted text data
            extracted_text_data.append({
                "Object_ID": f"object_{object_id}",
                "Image_File": filename,
                "Extracted_Text": text.strip()
            })
            object_id += 1

    # Save extracted text data to CSV
    df = pd.DataFrame(extracted_text_data)
    df.to_csv(output_file, index=False)
    print(f"Extracted text saved to {output_file}")

# Main function to run text extraction
if __name__ == "__main__":
    # Path to directory where extracted object images are saved
    extracted_objects_dir = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/segmented_objects"

    # Extract text using Tesseract OCR
    extract_text_tesseract(extracted_objects_dir)
