from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
from datetime import datetime

app = Flask(__name__)

model = YOLO("models/best.pt")  

OUTPUT_FOLDER = "static/outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Helper to count cars and people
def count_objects(results):
    cars = 0
    people = 0
    # Assuming YOLO class 0 = person, class 2 = car (COCO)
    # If your model uses different class indices, adjust accordingly!
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                people += 1
            elif cls == 2:
                cars += 1
    return cars, people

@app.route("/", methods=["GET", "POST"])
def index():
    filename = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filepath = os.path.join(OUTPUT_FOLDER, timestamp + "_" + file.filename)
            file.save(filepath)

            results = model.predict(filepath)
            result_img = results[0].plot()  
            save_path = os.path.join(OUTPUT_FOLDER, "det_" + timestamp + "_" + file.filename)
            cv2.imwrite(save_path, result_img)
            filename = "det_" + timestamp + "_" + file.filename 

    return render_template("index.html", filename=filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
