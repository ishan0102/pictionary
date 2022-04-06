from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/save")
async def save_image(request: Request):
    content = await request.json()
    async with aiofiles.open(f'./images/{content["uuid"][:8]}.json', 'wb') as out_file:
        await out_file.write(content["strokes"].encode())
    
    return {"message": "save image"}
