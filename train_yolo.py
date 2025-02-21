from ultralytics import YOLO
import argparse
import torch

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

    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)

    # Parse command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
    parser.add_argument("--data", type=str, default="organized_dataset/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="yolo_training_results")
    parser.add_argument("--device",type=int, default=0)  #0=cpu

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