# TODO: CODE REFACTOR
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

from fastapi import FastAPI, File, UploadFile
import uvicorn
from typing import List

import json
import requests

from PIL import Image
import io

import numpy as np


# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()



LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

def preprocess(image: Image.Image) -> torch.Tensor:
    # Implement your preprocessing steps here
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor and specify dtype
    image = image.permute(2, 0, 1)  # Convert to C x H x W format
    return image

def predict(image: torch.Tensor) -> str:
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
    
    # Get the predicted class index
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()
    
    # Get the corresponding class label
    class_name = labels[predicted_class]
    
    return class_name
app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Read the image data directly into memory
    image_data = await file.read()

    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))

    # Preprocess the image
    processed_image = preprocess(image)

    # Call the predict function with the preprocessed image
    class_name = predict(processed_image) 
    
    return {"predicted_class": class_name}





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)