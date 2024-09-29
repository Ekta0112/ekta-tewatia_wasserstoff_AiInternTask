import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import torch
import sys
import os
from io import BytesIO

# Add the models and utils directories to the Python path
sys.path.append('/content/drive/MyDrive/Image_Segmentation/Project_root/models')
sys.path.append('/content/drive/MyDrive/Image_Segmentation/Project_root/utils')

# Now import the functions
from segmentation_model import *
from identification_model import identify_objects
from text_extraction_model import extract_text_tesseract
from summarization_model import summarize_attributes
from data_mapping import create_data_mapping
from visualization import *

def main():
    st.title("AI Pipeline for Image Segmentation and Object Analysis")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Running pipeline...")

        # Save the uploaded image to a temporary file for further processing
        img_path = "/content/temp_uploaded_image.jpg"
        image.save(img_path)

        # Step 1: Get the segmented masks from the image
        masks, labels, boxes = get_prediction(img_path, model)
        
        # Step 2: Visualize the segmentation
        st.write("Running pipeline...visualize_segmentation start")
        #visualize_segmentation(img_path, masks)
        visualize_segmentation(img_path, masks)
        st.write("Running pipeline...visualize_segmentation end")
        # Step 3: Save the extracted objects
        st.write("Running pipeline...save_extracted_objects start: used to save the extracted objects from orignal image")
        save_extracted_objects(img_path, masks)
        st.write("Running pipeline...save_extracted_objects end")


        # Define paths to output directories and files
        #original_image_path = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/input_images/img1.png"
        extracted_objects_dir = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/segmented_objects"
        extracted_text_file = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/text_data_tesseract.csv"
        summary_file = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/summarized_attributes.csv"
        mapping_json_file = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/data_mapping.json"
        output_metadata_file="/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/object_metadata.csv"

        # Step 4: Identify objects (Optional if identification is needed)
        st.write("Running pipeline...identify_objects start")
        identify_objects(extracted_objects_dir)
        st.write("Running pipeline...identify_objects end")


        # Step 5: Extract text using Tesseract OCR
        st.write("Running pipeline...extract_text_tesseract start")
        extract_text_tesseract(extracted_objects_dir)
        st.write("Running pipeline...extract_text_tesseract end")

        # Step 6: Summarize the extracted text attributes
        st.write("Running pipeline...summarize_attributes start")
        summarize_attributes(extracted_text_file)
        st.write("Running pipeline...summarize_attributes end")

        # Step 7: Create data mapping (link extracted text, summaries to objects)
        st.write("Running pipeline...create_data_mapping start")
        create_data_mapping(img_path, extracted_text_file, summary_file)
        st.write("Running pipeline...create_data_mapping end")

        # Step 8: Generate final summary table
        st.write("Running pipeline...generate_annotated_image start")
        generate_annotated_image(img_path, mapping_json_file)
        st.write("Running pipeline...generate_annotated_image end")
        st.write("Running pipeline...generate_summary_table start")
        generate_summary_table(mapping_json_file)
        st.write("Running pipeline...generate_summary_table end")
        
        # Step 9: Display final outputs in the Streamlit UI
        st.image("/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/segmentation_visualization.png", 
          caption='Final Output Image with Segmentation', 
          use_column_width=True)
        
        #Displaying objects after identification
        st.write("Running pipeline...Displaying Identified Objects")
        st.dataframe(pd.read_csv(output_metadata_file))

        #Displaying extracted_text_file
        st.write("Running pipeline...Displaying extracted_text_file")
        st.dataframe(pd.read_csv(extracted_text_file))

        #Displaying after summarization model
        st.write("Running pipeline...Displaying summary_file after running summarization model ")
        st.dataframe(pd.read_csv(summary_file))

        #Displaying after mapping
        #Open and load the JSON file
        with open(mapping_json_file, 'r') as f:
            json_data = json.load(f)

        # Display the JSON data
        st.write("Running pipeline... Displaying mapping in JSON format:")
        st.json(json_data)

        st.image("/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/annotated_image.png", 
                 caption='Final Output Image with annotation', 
                 use_column_width=True)
        st.dataframe(pd.read_csv("/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/summary_table.csv"))

if __name__ == "__main__":
    main()
#streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py & npx localtunnel --port 8501
