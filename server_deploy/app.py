from flask import Flask, render_template, request, redirect
import numpy as np
from PIL import Image
import cv2
import io
import os
import inference

app = Flask(__name__)

root_path = "/mnt/e/WorkSpace/CPlusPlus/2d_pose_estimation/server_deploy/"

def pose_estimation(img, root_path):
    inference.detectImg(img, root_path)
    return img

def main():
    img_path = "./family.jpg"
    img = cv2.imread(img_path)
    cv2.imwrite("family_out.jpg", pose_estimation(img, root_path))

@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if img is not None:
            pass

        imgFile = "static/img.jpg"
        cv2.imwrite(imgFile, pose_estimation(img, root_path))
        return redirect(imgFile)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=12345)













