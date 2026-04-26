import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from models.cnn.utils import load_model, predict_one
from opencv import findTumorContour  # (not used here but kept if you need later)

CASCADE_PATH = os.path.join("models", "haar", "cascade.xml")
WEIGHTS_PATH = os.path.join("models", "cnn", "base.pt")

app = Flask(__name__)
CORS(app)

# Load once
model = load_model(WEIGHTS_PATH)
cascade = cv2.CascadeClassifier(CASCADE_PATH)


def is_likely_mri(img_bgr) -> bool:
    """
    Heuristic check (FYP-friendly):
    MRI scans are usually grayscale/low-saturation.
    Reject highly colorful images (likely not MRI).
    """
    if img_bgr is None:
        return False

    h, w = img_bgr.shape[:2]
    if h < 64 or w < 64:
        return False

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))  # saturation channel mean

    # MRI usually very low saturation (0-20). Photos often higher.
    return sat_mean < 35


def conf_to_level(conf01: float) -> str:
    p = float(conf01) * 100.0
    if p < 30:
        return "Low"
    elif p < 45:
        return "Moderate"
    elif p < 70:
        return "Mid"
    else:
        return "High"


def draw_green_yolo_boxes(img_bgr, result, min_conf=0.0, show_level=True):
    out = img_bgr.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return out

    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c in zip(xyxy, confs):
        c = float(c)
        if c < float(min_conf):
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        level = conf_to_level(c) if show_level else ""
        txt = f"{c:.2f} ({c*100:.0f}%) {level}".strip()

        cv2.putText(
            out,
            txt,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    return out


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    """
    form-data:
      mri: image file (png/jpg/jpeg)
      conf: optional float default 0.25
      returnImage: "true"/"false" (optional)

    If non-MRI image (colorful) -> returns 400 INVALID_MRI
    """
    if "mri" not in request.files:
        return jsonify({"message": "mri file is required"}), 400

    conf = float(request.form.get("conf", 0.25))
    return_image = request.form.get("returnImage", "false").lower() == "true"

    f = request.files["mri"]
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"message": "Could not read image"}), 400

    # ✅ MRI validation
    if not is_likely_mri(img):
        return jsonify({
            "message": "Invalid image. Please upload a brain MRI scan.",
            "code": "INVALID_MRI"
        }), 400

    # YOLO predict
    yolo_result = predict_one(model, img, conf=conf)
    yolo_count = 0 if yolo_result.boxes is None else len(yolo_result.boxes)

    top_conf = 0.0
    if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
        top_conf = float(yolo_result.boxes.conf.max().cpu().numpy())

    tumor_yes = yolo_count > 0
    confidence_percent = float(top_conf * 100.0)
    severity = conf_to_level(top_conf) if tumor_yes else "N/A"

    payload = {
        "tumor": bool(tumor_yes),
        "prediction": "tumor" if tumor_yes else "no_tumor",
        "confidence": confidence_percent,
        "severity": severity,
        "yoloDetections": int(yolo_count),
    }

    if return_image:
        out = draw_green_yolo_boxes(img, yolo_result, min_conf=conf, show_level=True)
        ok, buffer = cv2.imencode(".jpg", out)
        if ok:
            payload["annotatedImageBase64"] = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return jsonify(payload), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)