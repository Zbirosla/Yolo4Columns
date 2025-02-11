import os
import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("yolo_training_results/train/weights/best.pt")

# Path to the input image
prediction_path = "organized_dataset/test/images/grid_with_mask_0042.jpg"

# Get predictions from the model
results = model.predict(source=prediction_path, conf=0.4)

# Annotate the image with labels and bounding boxes or circles
class_colors = {
    "rect": (0, 255, 0),  # Green for rectangular objects
    "round": (255, 0, 0),  # Blue for round objects
}

# Load the original input image for annotation
image = cv2.imread(prediction_path)

# Annotate the image based on predictions
for result in results[0].boxes.data.cpu().numpy():
    x_min, y_min, x_max, y_max, confidence, class_id = result
    object_class = model.names[int(class_id)]  # Map class_id to class name
    color = class_colors.get(object_class, (0, 255, 255))  # Default to yellow if class not defined

    if object_class == "round":
        # Calculate the center and radius for the circle
        center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        radius = int(max(x_max - x_min, y_max - y_min) / 2)  # Radius is half the larger dimension
        cv2.circle(image, center, radius, color, 2)  # Draw a circle

        # Add the class label inside the circle
        label = f"{object_class}: {confidence:.2f}"
        label_position = (center[0] - radius, center[1] - radius - 10)  # Position above the circle
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        # For other objects, draw a rectangular bounding box
        top_left = (int(x_min), int(y_min))
        bottom_right = (int(x_max), int(y_max))
        cv2.rectangle(image, top_left, bottom_right, color, 2)

        # Add the class label
        label = f"{object_class}: {confidence:.2f}"
        label_position = (top_left[0], max(0, top_left[1] - 10))  # Avoid negative y-coordinates
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save only the annotated image
save_dir = "runs/detect/custom_predictions"  # Define custom save directory
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
annotated_output_path = os.path.join(save_dir, "annotated_prediction.jpg")
cv2.imwrite(annotated_output_path, image)

print(f"Annotated image with bounding boxes and labels saved as {annotated_output_path}")
