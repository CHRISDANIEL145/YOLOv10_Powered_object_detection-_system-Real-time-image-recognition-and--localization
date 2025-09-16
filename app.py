from flask import Flask, render_template, request, Response, redirect, url_for
import os
import cv2
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Define upload and output directories
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv10n model
model = YOLO('yolov10n.pt')  # Ensure yolov10n.pt is in the project directory

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    output_image = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save input image
            filename = file.filename
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            # Perform object detection
            results = model.predict(source=input_path, save=False, conf=0.25)
            annotated_image = results[0].plot()

            # Save annotated image
            output_filename = f"annotated_{filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            cv2.imwrite(output_path, annotated_image)

            # Extract predictions (class names and confidence scores)
            for box in results[0].boxes.data:
                cls_id = int(box[5])  # Class ID
                conf = float(box[4])  # Confidence score
                class_name = model.names[cls_id]  # Class name
                predictions.append(f"{class_name}, Confidence: {conf:.2f}")

            output_image = f"outputs/{output_filename}"

    return render_template('index.html', output_image=output_image, predictions=predictions)


@app.route('/video_feed')
def video_feed():
    def generate_frames():
        cap = cv2.VideoCapture(0)  # Open webcam
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Perform object detection on live stream
            results = model(frame)
            annotated_frame = results[0].plot()

            # Encode the frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
