from ultralytics import YOLO
from enum import Enum
import cv2
from fastapi import FastAPI, UploadFile, Response
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

global image_output
model = None


class ModelTypes(str, Enum):
    nano = "yolov8n.pt"
    small = "yolov8s.pt"
    medium = "yolov8m.pt"
    large = "yolov8l.pt"
    xl = "yolov8x.pt"


@app.post("/load_model/{model_type}")
async def load_model(model_type: ModelTypes):
    global model
    model = YOLO(model_type.value)
    return {"message": "Model loaded successfully"}


@app.post("/predict")
async def predict(image: UploadFile):
    global model
    global image_output
    if model is None:
        return {"message": "Error: No model loaded"}

    image_array = image_to_numpy(image)
    results = model(image_array, save=True)

    outputs = os.listdir("/runs/detect")
    max_element = max(outputs, key=extract_number)
    image_output = cv2.imread(f"/runs/detect/{max_element}/image0.jpg")


@app.get("/image")
async def get_predicted_image():
    global image_output
    _, jpeg = cv2.imencode('.jpg', image_output)
    return Response(content=jpeg.tobytes(), media_type="image/JPEG")


def image_to_numpy(image: UploadFile):
    image_data = np.frombuffer(image.file.read(), np.uint8)
    image_array = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image_array


def extract_number(s):
    try:
        return int(s.split("predict")[1])
    except ValueError:
        return 0


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
