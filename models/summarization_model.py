from transformers import pipeline
import pandas as pd
import os

# Function to summarize attributes of each object
def summarize_attributes(text_data_file, summary_file="/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/summarized_attributes.csv"):
    # Load the extracted text data from the CSV
    df = pd.read_csv(text_data_file)

    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    summarized_data = []

    # Iterate through each extracted text
    for index, row in df.iterrows():
        object_id = row["Object_ID"]
        text = row["Extracted_Text"]

        # Check if text is a valid string and not NaN
        if isinstance(text, str) and text.strip():  # Ensure it's a non-empty string
            print(f"Summarizing attributes for {object_id}...")
            summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
            summarized_data.append({
                "Object_ID": object_id,
                "Summary": summary[0]['summary_text']
            })
        else:
            summarized_data.append({
                "Object_ID": object_id,
                "Summary": "No text extracted."
            })

    # Save summarized data to CSV
    summary_df = pd.DataFrame(summarized_data)
    summary_df.to_csv(summary_file, index=False)
    print(f"Summarized attributes saved to {summary_file}")

# Main function to run summarization
if __name__ == "__main__":
    # Path to the extracted text data file
    extracted_text_file = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/text_data_tesseract.csv"
    # Generate summaries for each object
    summarize_attributes(extracted_text_file)
