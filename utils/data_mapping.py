import pandas as pd
import json
import os

# Function to create a mapping of object attributes and their summaries
def create_data_mapping(original_image_path, extracted_text_file, summary_file, output_json_file="/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/data_mapping.json"):
    # Load the extracted text and summarized data
    text_df = pd.read_csv(extracted_text_file)
    summary_df = pd.read_csv(summary_file)

    # Create a mapping dictionary
    data_mapping = {
        "Master_Image": original_image_path,  # Change this to the actual path of your master image
        "Objects": []
    }

    # Iterate through each object and gather information
    for index, row in text_df.iterrows():
        object_id = row["Object_ID"]
        extracted_text = row["Extracted_Text"]
        
        # Check if extracted_text is a valid string; if not, assign a default message
        extracted_text = extracted_text.strip() if isinstance(extracted_text, str) else "No text extracted."

        # Get the summary for the object
        summary_row = summary_df[summary_df["Object_ID"] == object_id]
        summary = summary_row["Summary"].values[0] if not summary_row.empty else "No summary available."

        # Add the object details to the mapping
        data_mapping["Objects"].append({
            "Object_ID": object_id,
            "Image_File": row["Image_File"],
            "Extracted_Text": extracted_text,
            "Summary": summary
        })

    # Save the mapping to a JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(data_mapping, json_file, indent=4)
    
    print(f"Data mapping saved to {output_json_file}")

# Main function to run data mapping
if __name__ == "__main__":
    # Path to the extracted text data file and the summary file
    extracted_text_file = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/text_data_tesseract.csv"
    summary_file = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/summarized_attributes.csv"
    original_image_path = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/input_images/img1.png"
    # Create data mapping
    create_data_mapping(original_image_path, extracted_text_file, summary_file)
