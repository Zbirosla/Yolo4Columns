import os
import json
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np


def load_ground_truth(json_local_path):
    with open(json_local_path, "r") as f:
        data = json.load(f)

    correct_boxes = []
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        correct_boxes.append({
            "category": categories[ann["category_id"]],
            "bbox": [x, y, x + w, y + h]  # Convert to (x_min, y_min, x_max, y_max)
        })

    print(f"Loaded {len(correct_boxes)} ground truth objects.")
    return correct_boxes


def intersection_over_union(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[0] + gt_box[2], pred_box[0] + pred_box[2]),
                              min(gt_box[1] + gt_box[3], pred_box[1] + pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection

    iou = intersection / union

    return iou


def evaluate_predictions(prediction_boxes, correct_boxes, iou_threshold=0.5):
    """
    Compare predictions with ground truth using IoU.
    Returns precision, recall, F1-score, and mAP.
    """
    true_positive, false_positive, false_negative = 0, 0, 0  # True Positives, False Positives, False Negatives

    matched_gt = set()
    matched_pred = set()

    for i, prediction in enumerate(prediction_boxes):
        matched = False
        for j, gt in enumerate(correct_boxes):
            if prediction["category"] == gt["category"] and intersection_over_union(prediction["bbox"], gt["bbox"]) > iou_threshold:
                matched = True
                matched_gt.add(j)
                matched_pred.add(i)
                break

        if matched:
            true_positive += 1  # True Positive
        else:
            false_positive += 1  # False Positive (wrongly detected)

    false_negative = len(correct_boxes) - len(matched_gt)  # False Negatives (missed detections)

    # Compute Precision, Recall, and F1-score
    Precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    Recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    F1_score = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

    # Compute Precision-Recall Curve
    precision_list = []
    recall_list = []

    sorted_preds = sorted(prediction_boxes, key=lambda x: x.get("confidence", 1.0), reverse=True)  # Sort by confidence if available

    tp, fp = 0, 0
    matched_gt = set()

    for pred in sorted_preds:
        matched = False
        for j, gt in enumerate(correct_boxes):
            if pred["category"] == gt["category"] and intersection_over_union(pred["bbox"], gt["bbox"]) > iou_threshold and j not in matched_gt:
                matched = True
                matched_gt.add(j)
                break

        if matched:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / len(correct_boxes)

        precision_list.append(precision)
        recall_list.append(recall)

    # Compute Average Precision (AP) using interpolation
    recall_array = np.array(recall_list)
    precision_array = np.array(precision_list)
    mAP = np.trapz(precision_array, recall_array) if len(recall_array) > 1 else 0  # Compute Area Under Curve

    return Precision, Recall, F1_score, mAP, true_positive, false_positive, false_negative


if __name__ == "__main__":
    # Load the trained YOLO model
    model = YOLO("weights_215epo_best.pt")

    # Define input and output directories
    input_folder = "data"
    correct_prediction_folder = "runs/detect/correct_predictions"
    bad_prediction_folder = "runs/detect/bad_predictions"
    os.makedirs(correct_prediction_folder, exist_ok=True)
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

            # Initialize lists for predicted boxes and masks
            pred_boxes = []
            pred_masks = []

            # Extract predicted boxes (and store confidence)
            for i, result in enumerate(results[0].boxes.data.cpu().numpy()):
                x_min, y_min, x_max, y_max, conf, class_id = result
                object_class = model.names[int(class_id)]
                pred_boxes.append({
                    "category": object_class,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": conf
                })

                # If segmentation masks are available, extract the corresponding mask
                if hasattr(results[0], "masks") and results[0].masks is not None:
                    # Assuming the masks align with the detection boxes
                    mask = results[0].masks.data[i].cpu().numpy()  # mask in float (e.g. probability map)
                    pred_masks.append(mask)

            # Compute metrics
            precision, recall, f1_score, mAP, tp, fp, fn = evaluate_predictions(pred_boxes, gt_boxes)

            # Load the image (once)
            image = cv2.imread(image_path)

            # Annotate predicted bounding boxes
            for pred in pred_boxes:
                x_min, y_min, x_max, y_max = map(int, pred["bbox"])
                color = class_colors.get(pred["category"], (0, 255, 255))
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                label = f"{pred['category']}: {pred['confidence']:.2f}"
                cv2.putText(image, label, (x_min, max(0, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # If segmentation masks were predicted, overlay them on the image
            if pred_masks:
                for mask, pred in zip(pred_masks, pred_boxes):
                    # Threshold the mask to create a binary mask
                    binary_mask = (mask > 0.5).astype(np.uint8) * 255
                    color = class_colors.get(pred["category"], (0, 255, 255))
                    # Create a colored mask image
                    colored_mask = np.zeros_like(image, dtype=np.uint8)
                    colored_mask[:, :] = color

                    # Blend the colored mask with the image where the mask is present
                    mask_indices = binary_mask.astype(bool)
                    image[mask_indices] = cv2.addWeighted(image[mask_indices], 0.5,
                                                          colored_mask[mask_indices], 0.5, 0)

            # Save the image based on accuracy
            if f1_score > 0.6:  # Good detection
                output_path = os.path.join(correct_prediction_folder, f"annotated_{filename}")
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
