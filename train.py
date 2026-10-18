from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="datasets/manhole_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        project="runs/urbanshield",
        name="manhole_detection",
        patience=10,
        workers=2,
        seed=42,
        plots=True,
        save=True
    )

if __name__ == "__main__":
    main()