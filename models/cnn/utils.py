from ultralytics import YOLO

def load_model(weights_path: str):
    return YOLO(weights_path)

def predict_one(model: YOLO, img_bgr, conf: float = 0.25):
    """
    Predict on a single OpenCV BGR image (numpy array).
    Returns Results[0].
    """
    results = model.predict(source=img_bgr, conf=conf, verbose=False)
    return results[0]