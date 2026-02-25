import os
import gdown
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io

app = FastAPI()

# روابط الموديلات من الدرايف بتاعك (الروابط اللي انتي بعتيها)
model_right_id = '1JqS1qpP7M6_1gFCUdQlYsIdNS_JsQKau'
model_left_id = '1dZ-eQXj0PXPjWs4OV7vHOR41Zcd2-1iP'

def download_and_load_models():
    # تحميل موديل العين اليمنى
    if not os.path.exists('model_right.h5'):
        print("Downloading Right Eye Model...")
        gdown.download(f'https://drive.google.com/uc?id={model_right_id}', 'model_right.h5', quiet=False)
    
    # تحميل موديل العين اليسرى
    if not os.path.exists('model_left.h5'):
        print("Downloading Left Eye Model...")
        gdown.download(f'https://drive.google.com/uc?id={model_left_id}', 'model_left.h5', quiet=False)
    
    print("Loading models into memory...")
    m1 = tf.keras.models.load_model('model_right.h5')
    m2 = tf.keras.models.load_model('model_left.h5')
    return m1, m2

# تحميل الموديلات عند بدء تشغيل السيرفر
model_right, model_left = download_and_load_models()

@app.get("/")
def home():
    return {"status": "Online", "message": "Diabetic AI Service is ready!"}

@app.post("/predict")
async def predict(right_eye: UploadFile = File(None), left_eye: UploadFile = File(None)):
    results = {}

    def process_image(file_contents, model):
        image = Image.open(io.BytesIO(file_contents)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
        prediction = model.predict(img_array)[0]
        return {
            "Diabetes": round(float(prediction[0]) * 100, 2),
            "Heart": round(float(prediction[1]) * 100, 2),
            "Hypertension": round(float(prediction[2]) * 100, 2),
            "Normal": round(float(prediction[3]) * 100, 2)
        }

    if right_eye:
        contents = await right_eye.read()
        results["right_eye"] = process_image(contents, model_right)
    
    if left_eye:
        contents = await left_eye.read()
        results["left_eye"] = process_image(contents, model_left)

    return results
