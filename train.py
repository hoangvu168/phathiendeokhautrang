from ultralytics import YOLO

if __name__ == '__main__':
    # Khởi tạo mô hình YOLOv8 từ pre-trained model
    model = YOLO("yolov8s.pt")

    # Huấn luyện mô hình
    model.train(
        data="data.yaml",  # File cấu hình dữ liệu
        epochs=50,  # Số epoch huấn luyện
        batch=16,  # Batch size
        imgsz=640,  # Kích thước ảnh đầu vào
        device="cuda"  # Chạy trên GPU (hoặc "cpu" nếu không có GPU)
    )
