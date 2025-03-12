import os
import json
import cv2
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("best.pt")

history = []
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except json.JSONDecodeError:
        history = []


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("file")
        if not files or files[0].filename.strip() == "":
            return "Không có file nào được chọn!", 400

        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            process_image(file_path, filename)

        return redirect(url_for("history_page"))

    return render_template("index.html")


def process_image(file_path, filename):
    img = cv2.imread(file_path)
    if img is None:
        return f"Lỗi khi đọc file {filename}!"

    results = model(file_path)
    mask_count, no_mask_count = 0, 0
    try:
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    mask_count += 1
                elif cls == 1:
                    no_mask_count += 1
    except Exception as e:
        return f"Lỗi khi nhận diện: {str(e)}"

    cv2.putText(img, f"Mask: {mask_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"No Mask: {no_mask_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    result_filename = f"result_{filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, img)

    history.append({
        "image": filename,
        "mask": mask_count,
        "no_mask": no_mask_count,
        "result_path": result_filename
    })
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


@app.route("/detect_folder", methods=["POST"])
def detect_folder():
    folder_path = request.form.get("folder_path", "").replace("\\", "/")
    if not os.path.exists(folder_path):
        return "Thư mục không tồn tại!", 400

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not images:
        return "Không tìm thấy ảnh hợp lệ trong thư mục!", 400

    for img_name in images:
        file_path = os.path.join(folder_path, img_name)
        process_image(file_path, img_name)

    return redirect(url_for("history_page"))


@app.route("/history")
def history_page():
    if not os.path.exists(HISTORY_FILE):
        return "Chưa có lịch sử nhận diện!", 400

    try:
        with open(HISTORY_FILE, "r") as f:
            history_data = json.load(f)
    except json.JSONDecodeError:
        history_data = []

    return render_template("history.html", history=history_data)


@app.route("/detect_camera")
def detect_camera():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame)
            mask_count, no_mask_count = 0, 0
            try:
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls == 0:
                            mask_count += 1
                        elif cls == 1:
                            no_mask_count += 1
            except Exception as e:
                print(f"Lỗi khi nhận diện: {str(e)}")
                continue

            cv2.putText(frame, f"Mask: {mask_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"No Mask: {no_mask_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
