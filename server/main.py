from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
# import aiofiles
import numpy as np
import pickle
import json
from model import SketchClassifier

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

guess_model = pickle.load(open("./models/guess_model.pkl", "rb"))

def format_strokes(data):
    lines = data['lines'] # data from canvas
    strokes = [] # data for model
    for stroke in lines:
        x_coords, y_coords = [], []
        for point in stroke['points']:
            x_coords.append(point['x'])
            y_coords.append(point['y'])
        strokes.append([x_coords, y_coords])
    return f'{strokes}'

@app.post("/save")
async def save_image(request: Request):
    content = await request.json()
    stroke_str = format_strokes(json.loads(content["strokes"]))    
    result = guess_model.predict(stroke_str)
    return {"class": result[0], "probability": f'{result[1]}'}
    # return {"class": result[0], "probability": result[1]}

