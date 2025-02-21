import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from ultralytics import YOLO


def load_all_ground_truth(data_folder):
    """
    Recursively finds all .json files in `data_folder` and loads their COCO-format annotations.
    Returns:
      image_gt (dict): { 'path/to/image.jpg': set_of_class_ids }
      categories (dict): { category_id: category_name }
    """
    import json

    image_gt = {}
    categories = {}

    for file_name in os.listdir(data_folder):
        if file_name.lower().endswith('.json'):
            json_path = os.path.join(data_folder, file_name)
            with open(json_path, 'r') as f:
                gt_data = json.load(f)

            # Build a mapping from image id -> file name
            for img_info in gt_data["images"]:
                img_id = img_info["id"]
                img_fname = img_info["file_name"]

                # NEW: ensure we only take the base name, then join with data_folder
                img_fname = os.path.normpath(img_fname)
                img_fname = os.path.basename(img_fname)
                full_path = os.path.join(data_folder, img_fname)

                # Now store that path
                id_to_filename = {}
                id_to_filename[img_id] = full_path

            # Merge categories
            for cat in gt_data["categories"]:
                cat_id = cat["id"]
                cat_name = cat["name"]
                if cat_id not in categories:
                    categories[cat_id] = cat_name

            # Build ground-truth sets per image
            for ann in gt_data["annotations"]:
                img_id = ann["image_id"]
                cls_id = ann["category_id"]
                # If your JSON has multiple images, you need to store `id_to_filename` outside the loop
                # (shown here in a simplified way for a single-image JSON).

                if cls_id not in categories:
                    categories[cls_id] = f"unknown_{cls_id}"

                # Retrieve the image path from id_to_filename
                if img_id in id_to_filename:
                    img_path = id_to_filename[img_id]
                    if img_path not in image_gt:
                        image_gt[img_path] = set()
                    image_gt[img_path].add(cls_id)

    return image_gt, categories


if __name__ == '__main__':
    # -----------------------------
    # 1) Load all ground truth from a data folder
    # -----------------------------
    data_folder = "data"  # <-- update with your folder containing images and JSON files
    image_gt, categories = load_all_ground_truth(data_folder)
    print("Found categories:", categories)
    print(f"Found {len(image_gt)} images with annotations.")

    # -----------------------------
    # 2) Load YOLO model
    # -----------------------------
    model_path = "yolo_training_results/train300EPOCH_SEG/train/weights/best.pt"
    model = YOLO(model_path)

    # Confidence threshold for deciding a predicted class is present
    confidence_threshold = 0.5

    # Prepare data structure to store (gt, score, pred) per class
    eval_data = {cls_id: {"gt": [], "score": [], "pred": []} for cls_id in categories.keys()}

    # -----------------------------
    # 3) Inference over each image
    # -----------------------------
    for img_path, gt_classes in image_gt.items():
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image: {img_path}")
            continue

        # Run YOLO inference
        results = model(image)

        # Initialize predicted scores for each class
        pred_scores = {cls_id: 0.0 for cls_id in categories.keys()}

        # Gather detections from YOLO
        detections = results[0].boxes
        if len(detections) > 0:
            det_classes = detections.cls.cpu().numpy() if hasattr(detections.cls, 'cpu') else detections.cls.numpy()
            det_confs = detections.conf.cpu().numpy() if hasattr(detections.conf, 'cpu') else detections.conf.numpy()
            for c, conf in zip(det_classes, det_confs):
                c = int(c)
                if conf > pred_scores[c]:
                    pred_scores[c] = conf

        # -----------------------------
        # 4) Record metrics
        #    For each class:
        #      - gt = 1 if class is in the set of GT classes
        #      - score = max confidence
        #      - pred = 1 if score >= threshold
        # -----------------------------
        for cls_id in categories.keys():
            gt_val = 1 if cls_id in gt_classes else 0
            score = pred_scores[cls_id]
            pred_val = 1 if score >= confidence_threshold else 0

            eval_data[cls_id]["gt"].append(gt_val)
            eval_data[cls_id]["score"].append(score)
            eval_data[cls_id]["pred"].append(pred_val)

    # -----------------------------
    # 5) Generate Evaluation Plots
    # -----------------------------

    # Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for cls_id, data in eval_data.items():
        gt_arr = np.array(data["gt"])
        score_arr = np.array(data["score"])

        # If there are no positives for this class, skip to avoid warnings
        if sum(gt_arr) == 0:
            continue

        precision, recall, _ = precision_recall_curve(gt_arr, score_arr)
        plt.plot(recall, precision, label=f'{categories[cls_id]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves per Class')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ROC Curves
    plt.figure(figsize=(10, 8))
    for cls_id, data in eval_data.items():
        gt_arr = np.array(data["gt"])
        score_arr = np.array(data["score"])

        # If there are no positives for this class, skip
        if sum(gt_arr) == 0:
            continue

        fpr, tpr, _ = roc_curve(gt_arr, score_arr)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{categories[cls_id]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves per Class')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Confusion Matrices
    # Každá třída získá svou vlastní matici pro přítomnost/absenci
    for cls_id, data in eval_data.items():
        gt_arr = np.array(data["gt"])
        pred_arr = np.array(data["pred"])
        cm = confusion_matrix(gt_arr, pred_arr, labels=[0, 1])

        # Normalizace: Každý řádek vydělíme součtem hodnot v daném řádku
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(6, 5))
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Normalized Confusion Matrix: {categories[cls_id]}')
        plt.colorbar()

        tick_marks = np.arange(2)
        class_names = ['Negative', 'Positive']
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        thresh = cm_normalized.max() / 2.0
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                plt.text(j, i, format(cm_normalized[i, j], '.2f'),
                         horizontalalignment="center",
                         color="white" if cm_normalized[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()