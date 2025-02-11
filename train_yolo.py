from ultralytics import YOLO
import argparse
import torch
import time

def train_yolo(data_yaml, model_checkpoint, epochs, img_size, batch_size, save_dir):

    # Load the pre-trained YOLO model
    model = YOLO(model_checkpoint)

    # Train the model in a single run for all epochs
    model.train(
        data=data_yaml,
        epochs=epochs,  # Train for all epochs at once
        imgsz=img_size,
        batch=batch_size,
        project=save_dir,
        device=0  # GPU
    )

if __name__ == "__main__":
    train_model = True
    evaluate_model = True

    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)

    if train_model:
        # Parse command-line arguments for flexibility
        parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
        parser.add_argument("--data", type=str, default="organized_dataset/data.yaml", help="Path to data.yaml file")
        parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="YOLO model checkpoint")
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
        parser.add_argument("--imgsz", type=int, default=1024, help="Image size (pixels)")
        parser.add_argument("--batch", type=int, default=16, help="Batch size")
        parser.add_argument("--save_dir", type=str, default="yolo_training_results", help="Directory to save training results")
        parser.add_argument("--device",type=int, default=0, help="GPU or CPU")

        args = parser.parse_args()

        # Train YOLO model
        train_yolo(
            data_yaml=args.data,
            model_checkpoint=args.model,
            epochs=args.epochs,
            img_size=args.imgsz,
            batch_size=args.batch,
            save_dir=args.save_dir
        )

    if evaluate_model:
        # Load the best trained model
        model = YOLO("yolo_training_results/train/weights/best.pt")  # Replace with your actual path if different

        # Run validation
        metrics = model.val(data="organized_dataset/data.yaml")
        print(metrics)  # Displays validation results, including mAP and loss