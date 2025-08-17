from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import shutil

app = Flask(__name__)

model = YOLO("models/best.pt")  


OUTPUT_FOLDER = "static/outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
    app.run(host="0.0.0.0", port=5000, debug=True)
