from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/open")
async def open_image():
    return {"message": "open image"}

@app.get("/save")
async def save_image():
    return {"message": "save image"}
