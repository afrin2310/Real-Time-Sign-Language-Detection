import numpy as np
from tensorflow.keras.models import load_model
import os

_model  = None
_labels = None

def _load():
    global _model, _labels
    if _model is None:
        _model  = load_model("model/sign_model.keras")
        _labels = np.load("model/labels.npy", allow_pickle=True)

def predict_sign(features: np.ndarray):
    """features: np.array (126,) — 2 hands × 21 pts × xyz"""
    _load()
    pred = _model.predict(features.reshape(1, -1), verbose=0)[0]
    idx  = int(np.argmax(pred))
    return _labels[idx], float(pred[idx])

def model_ready():
    return (os.path.exists("model/sign_model.keras") and
            os.path.exists("model/labels.npy"))