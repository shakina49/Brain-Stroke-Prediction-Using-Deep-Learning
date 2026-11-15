# =========================================================
# Brain Stroke CT Scan Detection – FINAL PRODUCTION VERSION
# =========================================================

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from torchvision import transforms, models

# ===================== CONFIG =====================

UPLOAD_FOLDER = "static/uploads"
MODEL_DIR = "outputs_stroke_final"

IMG_SIZE = 256
CONF_THRESHOLD = 0.75
USE_TTA = True

CLASS_NAMES = ["normal", "stroke"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cpu")

# =========================================================
# ===================== CT VALIDATION =====================
# =========================================================

def is_ct_scan(image_path):

    """
    Strong Brain CT scan validation
    Rejects screenshots, documents, random images
    """

    try:

        img = cv2.imread(image_path)

        if img is None:
            return False

        h, w = img.shape[:2]

        # 1️⃣ Minimum size
        if h < 200 or w < 200:
            return False

        # 2️⃣ Aspect ratio check (CT scans nearly square)
        ratio = w / h
        if ratio < 0.8 or ratio > 1.2:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3️⃣ Check grayscale similarity
        b, g, r = cv2.split(img)

        color_diff = (
            np.mean(cv2.absdiff(r, g)) +
            np.mean(cv2.absdiff(r, b)) +
            np.mean(cv2.absdiff(g, b))
        ) / 3

        if color_diff > 8:
            return False

        # 4️⃣ Border darkness (CT scans have black background)

        border_pixels = np.concatenate([
            gray[0:20, :].flatten(),
            gray[-20:, :].flatten(),
            gray[:, 0:20].flatten(),
            gray[:, -20:].flatten()
        ])

        if np.mean(border_pixels) > 80:
            return False

        # 5️⃣ Brain texture variance

        center = gray[h//4:3*h//4, w//4:3*w//4]

        if np.var(center) < 500:
            return False

        # 6️⃣ Skull circle detection

        blurred = cv2.GaussianBlur(gray, (9,9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=80,
            maxRadius=min(h, w)//2
        )

        if circles is None:
            return False

        return True

    except:
        return False


# =========================================================
# ===================== PREPROCESS ========================
# =========================================================

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =========================================================
# ===================== MODEL BUILD =======================
# =========================================================

def build_resnet50():

    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2
    )

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 2)
    )

    return model


def build_effnet_b3():

    model = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
    )

    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        2
    )

    return model


# =========================================================
# ===================== LOAD MODELS =======================
# =========================================================

def load_models():

    models_list = []

    for ckpt in sorted(glob.glob(f"{MODEL_DIR}/*.pth")):

        if "resnet50" in ckpt:
            model = build_resnet50()

        elif "efficientnet_b3" in ckpt:
            model = build_effnet_b3()

        else:
            continue

        model.load_state_dict(
            torch.load(ckpt, map_location=device)
        )

        model.eval()
        model.to(device)

        models_list.append(model)

    if not models_list:
        raise RuntimeError("No trained models found!")

    return models_list


ENSEMBLE_MODELS = load_models()

# =========================================================
# ===================== PREDICTION ========================
# =========================================================

@torch.no_grad()
def predict_image(image_path):

    img = Image.open(image_path).convert("RGB")

    angles = [-5,0,5] if USE_TTA else [0]

    probs_all = []

    for angle in angles:

        img_rot = img.rotate(angle)

        x = val_tfms(img_rot).unsqueeze(0).to(device)

        probs = sum(
            torch.softmax(model(x), dim=1)
            for model in ENSEMBLE_MODELS
        ) / len(ENSEMBLE_MODELS)

        probs_all.append(probs.cpu().numpy())

    probs_final = np.mean(probs_all, axis=0)[0]

    pred_idx = int(np.argmax(probs_final))
    confidence = float(probs_final[pred_idx])

    if confidence < CONF_THRESHOLD:
        return "uncertain", confidence

    return CLASS_NAMES[pred_idx], confidence


# =========================================================
# ===================== FLASK ROUTE =======================
# =========================================================

@app.route("/", methods=["GET","POST"])

def index():

    result = None
    img_path = None

    if request.method == "POST":

        file = request.files.get("file")

        if file and file.filename != "":

            filename = secure_filename(file.filename)

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            # Step 1 — Validate CT Scan

            if not is_ct_scan(filepath):

                result = "Invalid image. Please upload a Brain CT Scan."

                return render_template(
                    "index.html",
                    result=result,
                    img_path=None
                )

            # Step 2 — Prediction

            try:

                pred, conf = predict_image(filepath)

                if pred == "uncertain":

                    result = "Model not confident. Please upload a clearer CT scan."

                else:

                    result = f"{pred.upper()} ({conf*100:.2f}%)"

                    img_path = url_for(
                        "static",
                        filename=f"uploads/{filename}"
                    )

            except Exception as e:

                result = f"Error: {str(e)}"

    return render_template(
        "index.html",
        result=result,
        img_path=img_path
    )


# =========================================================
# ===================== RUN APP ===========================
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)