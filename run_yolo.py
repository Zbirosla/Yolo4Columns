import os
import json
import cv2
import pandas as pd
from ultralytics import YOLO


def load_ground_truth(json_path):
    """Load ground truth bounding boxes from the corresponding JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    gt_boxes = []
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        gt_boxes.append({
            "category": categories[ann["category_id"]],
            "bbox": [x, y, x + w, y + h]  # Convert to (x_min, y_min, x_max, y_max)
        })

    return gt_boxes


def iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def evaluate_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Compare predictions with ground truth using IoU.
    Returns precision, recall, and F1-score.
    """
    tp, fp, fn = 0, 0, 0  # True Positives, False Positives, False Negatives

    matched_gt = set()
    matched_pred = set()

    for i, pred in enumerate(pred_boxes):
        matched = False
        for j, gt in enumerate(gt_boxes):
            if pred["category"] == gt["category"] and iou(pred["bbox"], gt["bbox"]) > iou_threshold:
                matched = True
                matched_gt.add(j)
                matched_pred.add(i)
                break

        if matched:
            tp += 1  # True Positive
        else:
            fp += 1  # False Positive (wrongly detected)

    fn = len(gt_boxes) - len(matched_gt)  # False Negatives (missed detections)

    # Compute Precision, Recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score, tp, fp, fn


if __name__ == "__main__":
    # Load the trained YOLO model
    model = YOLO("weights_215epo_best.pt")

    # Define input and output directories
    input_folder = "data"
    output_folder = "runs/detect/correct_predictions"
    bad_prediction_folder = "runs/detect/bad_predictions"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(bad_prediction_folder, exist_ok=True)

    # Define class colors for annotation
    class_colors = {
        "rect": (0, 255, 0),  # Green for rectangular objects
        "round": (255, 0, 0),  # Blue for round objects
        "steel": (0, 0, 255)  # Red for steel objects
    }

    # Metric storage
    metrics_data = []

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_folder, filename)
            json_path = image_path.replace(".jpg", ".json").replace(".png", ".json").replace(".jpeg", ".json")

            if not os.path.exists(json_path):
                print(f"Missing JSON for {filename}, skipping.")
                continue

            # Get predictions from the model
            results = model.predict(source=image_path, conf=0.4)

            # Load ground truth
            gt_boxes = load_ground_truth(json_path)

            # Extract predicted boxes
            pred_boxes = []
            for result in results[0].boxes.data.cpu().numpy():
                x_min, y_min, x_max, y_max, confidence, class_id = result
                object_class = model.names[int(class_id)]
                pred_boxes.append({
                    "category": object_class,
                    "bbox": [x_min, y_min, x_max, y_max]
                })

            # Compute metrics
            precision, recall, f1_score, tp, fp, fn = evaluate_predictions(pred_boxes, gt_boxes)

            # Load the image
            image = cv2.imread(image_path)

            # Annotate predictions
            for pred in pred_boxes:
                x_min, y_min, x_max, y_max = map(int, pred["bbox"])
                color = class_colors.get(pred["category"], (0, 255, 255))
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                label = f"{pred['category']}: {confidence:.2f}"
                cv2.putText(image, label, (x_min, max(0, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save the image based on accuracy
            if f1_score > 0.7:  # Good detection
                output_path = os.path.join(output_folder, f"annotated_{filename}")
            else:
                output_path = os.path.join(bad_prediction_folder, f"bad_{filename}")

            cv2.imwrite(output_path, image)
            print(f"Saved: {output_path}")

            # Store metrics
            metrics_data.append({
                "image": filename,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn
            })

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv("runs/detect/metrics.csv", index=False)

    print("Processing completed for all images.")
    print("Metrics saved to runs/detect/metrics.csv")
