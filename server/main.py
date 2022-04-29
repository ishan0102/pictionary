from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
# import aiofiles
import numpy as np
import pickle
import json
from model import SketchClassifier
from preprocess import simplify_drawings

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

@app.post("/save")
async def save_image(request: Request):
    content = await request.json()
    processed = simplify_drawings('user', json.loads(content["strokes"]))
    result = guess_model.predict(f'{processed["drawing"]}')
    return {"class": result[0], "probability": f'{result[1]}'}

