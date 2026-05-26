# test_model_load.py
from tensorflow.keras.models import load_model
import os
MODEL_PATH = "model/deepcrop_cnn.h5"
print("Exists?", os.path.exists(MODEL_PATH))
try:
    m = load_model(MODEL_PATH)
    print("Loaded model:", type(m))
except Exception as e:
    print("ERROR loading model:", e)
