from ultralytics import YOLO

# Load mô hình đã huấn luyện
model = YOLO("best.pt")

# Dự đoán trên hình ảnh test
results = model.predict(source="test_images/", conf=0.3, save=True)

# Hiển thị kết quả
for r in results:
    r.show()
