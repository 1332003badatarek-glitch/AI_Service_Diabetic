import os
import json
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.applications.efficientnet import preprocess_input

app = FastAPI()

# المسارات
MODEL_PATH = "fundus_efficientnet_ultra.h5"
INDICES_PATH = "class_indices_ultra.json"

print("🔍 Loading Ultra High-Res Model (Pure Prediction Mode)...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Full Model Loaded Successfully!")
except Exception as e:
    print(f"⚠️ Reconstructing structure for compatibility... {e}")
    base_model = tf.keras.applications.EfficientNetB3(weights=None, include_top=False, input_shape=(450, 450, 3))
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.load_weights(MODEL_PATH)
    print("✅ Model Structure Rebuilt & Weights Loaded!")

# تحميل الكلاسات
with open(INDICES_PATH, 'r') as f:
    class_indices = json.load(f)
labels = {int(v): k for k, v in class_indices.items()}
class_names = [labels[i] for i in range(len(labels))]

def process_and_predict(img_bytes):
    # 1. المعالجة
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((450, 450), Image.NEAREST)
    
    x = np.array(img).astype('float32')
    x = np.expand_dims(x, axis=0)
    
    # 2. Preprocessing
    x = preprocess_input(x)

    # 3. التوقع
    probabilities = model.predict(x, verbose=0)[0]
    
    # تحويل النسب لشكل مفهوم ومقرب
    all_scores = {class_names[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(class_names))}
    
    # التعديل المطلوب: هنرجع الـ detailed_probabilities فقط
    return {
        "detailed_probabilities": all_scores
    }

@app.post("/predict")
async def predict(right_eye: UploadFile = File(None), left_eye: UploadFile = File(None)):
    results = {}
    if right_eye:
        results["RightEye"] = process_and_predict(await right_eye.read())
    if left_eye:
        results["LeftEye"] = process_and_predict(await left_eye.read())
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)