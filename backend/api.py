from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.face_detection import detect_faces, detect_faces_video, apply_bounding_box
from core.video_processing import generate_video, process_video
from PIL import Image
from typing import List
import shutil
import tempfile
import zipfile
import os
import cv2
import numpy as np
import io
import base64
from core.models.resnet_7c import Resnet_7C

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Resnet_7CModel = Resnet_7C()  # init trained model


@app.get("/")
async def root():
    return {"message": "Welcome to the age detection API!"}


@app.get("/models")
async def get_models():
    directory_path = "models"

    try:
        files = os.listdir(directory_path)
        if not files:
            raise HTTPException(status_code=404, detail="No models found")
        return JSONResponse(content={"files": files}, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Directory not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/detect-age/ws")
async def detect_age_single(websocket: WebSocket):
    await websocket.accept()

    frame_counter = 0
    detected_faces = None
    while True:
        data = await websocket.receive_text()
        decoded_data = base64.b64decode(data)
        frame = cv2.imdecode(np.frombuffer(decoded_data, dtype=np.uint8), 1)
        if frame_counter % 6 == 0:
            detected_faces = detect_faces_video(frame, Resnet_7CModel)
        frame = apply_bounding_box(frame, detected_faces)
        _, encoded_frame = cv2.imencode('.jpg', frame)
        image = base64.b64encode(encoded_frame.tobytes()).decode('utf-8')
        frame_counter += 1
        await websocket.send_text(image)


@app.post("/detect-age/multiple")
def detect_age_multiple(files: List[UploadFile] = File(...)):
    try:
        images = []
        for file in files:
            if not file.content_type.startswith("image/"):
                continue
            content = file.file.read()
            image_array = cv2.imdecode(np.frombuffer(content, np.uint8), -1)

            detected_faces = detect_faces(image_array, Resnet_7CModel)
            # if not detected_faces['faces'].any():
            #     raise HTTPException(status_code=400, detail="No faces detected in the image.")

            image_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            output_image = io.BytesIO()
            image_pil.save(output_image, format="PNG")
            output_image.seek(0)
            images.append({"name": file.filename, "content": output_image.getvalue()})

        if not images:
            raise HTTPException(status_code=400, detail="No valid image files found.")

        zip_file = io.BytesIO()
        with zipfile.ZipFile(zip_file, 'w') as zipf:
            for image in images:
                zipf.writestr(image["name"], image["content"])

        zip_file.seek(0)
        return StreamingResponse(iter([zip_file.getvalue()]), media_type="application/x-zip-compressed",
                                 headers={"Content-Disposition": f"attachment; filename=images.zip"})
    except Exception as e:
        return JSONResponse(content={"detail": f"An error occurred: {str(e)}"}, status_code=500)


@app.post("/detect-age/video")
async def detect_age_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video.")
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)

        with open(file_path, "wb") as video_file:
            shutil.copyfileobj(file.file, video_file)

        frames, frame_rate = process_video(file_path, detect_faces_video, apply_bounding_box, Resnet_7CModel)

        video_bytes = generate_video(frames, temp_dir, frame_rate)

        shutil.rmtree(temp_dir)

        return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
