import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd

# Function to draw text on the image
def draw_text_on_image(image, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# Function to generate the annotated image
def generate_annotated_image(original_image_path, mapping_json_file, output_image_path="/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/annotated_image.png"):
    # Load the original image
    img = cv2.imread(original_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    # Load the data mapping
    with open(mapping_json_file, 'r') as json_file:
        data_mapping = json.load(json_file)

    # Draw annotations on the image
    for obj in data_mapping["Objects"]:
        object_id = obj["Object_ID"]
        extracted_text = obj["Extracted_Text"]
        summary = obj["Summary"]

        # Example position for annotations; adjust as needed
        position = (10, 30 * (data_mapping["Objects"].index(obj) + 1))  # Adjust Y position based on object index

        # Create annotation text
        annotation_text = f"{object_id}: {extracted_text} | {summary}"

        # Draw the annotation on the image
        draw_text_on_image(img, annotation_text, position)

    # Display and save the annotated image
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Annotated Image with Segmented Objects")
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.show()
    print(f"Annotated image saved to {output_image_path}")

# Function to generate and save the summary table as CSV
def generate_summary_table(mapping_json_file, output_csv_file="/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/summary_table.csv"):
    # Load the data mapping
    with open(mapping_json_file, 'r') as json_file:
        data_mapping = json.load(json_file)

    # Create a DataFrame from the data mapping
    summary_df = pd.DataFrame(data_mapping["Objects"])

    # Save the summary DataFrame to CSV
    summary_df.to_csv(output_csv_file, index=False)
    print(f"Summary table saved to {output_csv_file}")

# Main function to run output generation
if __name__ == "__main__":
    original_image_path = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/input_images/img1.png"  # Change to your original image path
    mapping_json_file = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/data_mapping.json"

    # Generate and save the annotated image
    generate_annotated_image(original_image_path, mapping_json_file)
    
    # Generate and save the summary table
    generate_summary_table(mapping_json_file)
