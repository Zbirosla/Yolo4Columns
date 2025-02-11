import os
import json
import shutil
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# Paths
data_dir = 'data'  # Your data directory
output_dir = 'organized_dataset'  # Directory to store organized data
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Classes
classes = ['rect', 'round', 'steel']  # Replace with your actual class names

# Create directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

# Split data
train_files, val_test_files = train_test_split(image_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(val_test_files, test_size=0.3333, random_state=42)  # 0.3333 x 0.3 ≈ 0.1

splits = {
    'train': train_files,
    'val': val_files,
    'test': test_files
}

total_files = len(image_files)
train_percentage = (len(train_files) / total_files) * 100
val_percentage = (len(val_files) / total_files) * 100
test_percentage = (len(test_files) / total_files) * 100

def create_data_yaml(dataset_dir, class_names):

    data = {
        "names": class_names,
        "nc": len(class_names),
        "train": r"C:/Users/zbiro/PycharmProjects/ML_from_mask/organized_dataset/train/images",
        "val": r"C:/Users/zbiro/PycharmProjects/ML_from_mask/organized_dataset/val/images",
        "test": r"C:/Users/zbiro/PycharmProjects/ML_from_mask/organized_dataset/test/images",
    }

    # Remove the test entry if the test directory doesn't exist
    if data["test"] is None:
        del data["test"]

    # Save to YAML
    output_path = os.path.join(dataset_dir, "data.yaml")
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"data.yaml created at {output_path}")

# Function to convert LabelMe COCO to YOLO format
def convert_coco_to_yolo_segmentation(coco_annotations, yolo_output_path, categories, img_width, img_height):
    category_mapping = {cat['id']: cat['name'] for cat in categories}
    category_ids = {cat['name']: idx for idx, cat in enumerate(categories)}

    yolo_annotations = []
    for annotation in coco_annotations:
        category_id = annotation['category_id']
        if category_id not in category_mapping:
            continue

        yolo_class_id = category_ids[category_mapping[category_id]]
        segmentation = annotation.get('segmentation', [])

        if isinstance(segmentation, list):
            # Case 1: Single polygon represented as a flat list of coordinates (e.g., annotation ID 1)
            if all(isinstance(coord, float) for coord in segmentation):
                normalized_coords = []
                for i in range(0, len(segmentation), 2):
                    x = segmentation[i] / img_width
                    y = segmentation[i + 1] / img_height
                    normalized_coords.append(f"{x:.6f} {y:.6f}")
                yolo_annotations.append(f"{yolo_class_id} " + " ".join(normalized_coords))

            # Case 2: Multiple polygons (e.g., annotation ID 3)
            elif all(isinstance(seg, list) for seg in segmentation):
                for seg in segmentation:
                    normalized_coords = []
                    for i in range(0, len(seg), 2):
                        x = seg[i] / img_width
                        y = seg[i + 1] / img_height
                        normalized_coords.append(f"{x:.6f} {y:.6f}")
                    yolo_annotations.append(f"{yolo_class_id} " + " ".join(normalized_coords))

            else:
                print(f"Unsupported segmentation format for annotation ID {annotation['id']}. Skipping.")
        else:
            print(f"Unsupported segmentation format for annotation ID {annotation['id']}. Skipping.")

    # Save YOLO segmentation annotations to a file
    with open(yolo_output_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))




# Process and move files
for split, files in splits.items():
    for image_file in tqdm(files, desc=f'Processing {split} set'):
        image_path = os.path.join(data_dir, image_file)
        json_file = image_file.replace('.jpg', '.json')
        json_path = os.path.join(data_dir, json_file)

        # Verify that both image and annotation exist
        if not os.path.exists(json_path):
            print(f"Annotation {json_file} not found for image {image_file}. Skipping.")
            continue

        # Copy image
        dest_image_path = os.path.join(output_dir, split, 'images', image_file)
        shutil.copyfile(image_path, dest_image_path)

        # Get image dimensions
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size

        # Load COCO annotations from JSON
        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        # Find the corresponding image ID
        matching_images = [
            img for img in coco_data['images']
            if os.path.basename(img['file_name']) == image_file
        ]
        if not matching_images:
            print(f"No matching entry found for {image_file} in COCO annotations. Skipping.")
            continue
        image_id = matching_images[0]['id']

        # Get annotations for the image
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

        # Convert and save annotations
        dest_label_path = os.path.join(output_dir, split, 'labels', image_file.replace('.jpg', '.txt'))
        convert_coco_to_yolo_segmentation(image_annotations, dest_label_path, coco_data['categories'], width, height)


create_data_yaml(output_dir, classes)

print("Data organization complete.")
total_files = len(image_files)
train_percentage = (len(train_files) / total_files) * 100
val_percentage = (len(val_files) / total_files) * 100
test_percentage = (len(test_files) / total_files) * 100

print(f"Dataset was split into: \nTrain: {len(train_files)} files ({train_percentage:.2f}%)\nValidation: {len(val_files)} files ({val_percentage:.2f}%)\nTest: {len(test_files)} files ({test_percentage:.2f}%)")




