from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import requests
import urllib.parse
import os

app = FastAPI(title="Pollinations Image Generator")

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.get("/generate")
async def generate_image(prompt: str):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Pollinations.ai endpoint
    safe_prompt = urllib.parse.quote(prompt.strip())
    image_url = f"https://pollinations.ai/p/{safe_prompt}"
    
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        
        # Stream the image directly (no saving to disk)
        return StreamingResponse(
            content=response.iter_content(chunk_size=8192),
            media_type=response.headers.get("content-type", "image/jpeg"),
            headers={"Content-Disposition": "inline; filename=generated.jpg"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
