import torch
import os
import cv2
import pandas as pd

# Load pre-trained YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.eval()

# Function to identify objects and generate descriptions
def identify_objects(image_dir, output_metadata_file="/content/drive/MyDrive/Image_Segmentation/Project_root/data/output/object_metadata.csv"):
    object_descriptions = []
    object_id = 1

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            print(f"Processing {img_path}...")

            # Load image using OpenCV
            img = cv2.imread(img_path)

            # Convert image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Perform object detection
            results = yolo_model(img_rgb)

            # Extract results (labels, confidence, and bounding boxes)
            labels = results.xyxyn[0][:, -1].numpy()  # Class labels (as indices)
            confidences = results.xyxyn[0][:, -2].numpy()  # Confidence scores
            bboxes = results.xyxyn[0][:, :-2].numpy()  # Bounding boxes

            # Convert labels from indices to actual class names
            detected_objects = [yolo_model.names[int(label)] for label in labels]

            # Store each detected object with description
            for obj, conf in zip(detected_objects, confidences):
                object_descriptions.append({
                    "Object_ID": f"object_{object_id}",
                    "Image_File": filename,
                    "Description": obj,
                    "Confidence": conf,
                })
                object_id += 1

    # Save metadata as CSV
    df = pd.DataFrame(object_descriptions)
    df.to_csv(output_metadata_file, index=False)
    print(f"Object descriptions saved to {output_metadata_file}")

# Main function to run object identification
if __name__ == "__main__":
    # Path to directory where extracted objects are saved
    extracted_objects_dir = "/content/drive/MyDrive/Image_Segmentation/Project_root/data/segmented_objects"

    # Identify objects and generate descriptions
    identify_objects(extracted_objects_dir)
