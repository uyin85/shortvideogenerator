from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import httpx
import asyncio
import os
from pydantic import BaseModel

app = FastAPI(title="Stable Horde Generator")

# Get API key from environment (set in Render)
HORDE_API_KEY = os.getenv("HORDE_API_KEY", "0000000000")
HORDE_BASE = "https://stablehorde.net/api/v2"

class GenerateRequest(BaseModel):
    prompt: str

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    headers = {
        "Content-Type": "application/json",
        "apikey": HORDE_API_KEY
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Submit async job
        gen_payload = {
            "prompt": prompt,
            "models": ["SDXL Lightning"],
            "width": 512,
            "height": 512,
            "steps": 20,
            "sampler_name": "k_euler_a",
            "nsfw": False,
            "censor_nsfw": True,
            "r2": True
        }

        try:
            response = await client.post(f"{HORDE_BASE}/generate/async", json=gen_payload, headers=headers)
            if response.status_code != 202:
                raise HTTPException(status_code=500, detail=f"Horde API error: {response.text}")

            req_id = response.json()["id"]

            # Step 2: Poll until done (max 120 seconds)
            for _ in range(60):
                await asyncio.sleep(2)
                check_resp = await client.get(f"{HORDE_BASE}/generate/check/{req_id}")
                if check_resp.json().get("done"):
                    result_resp = await client.get(f"{HORDE_BASE}/generate/status/{req_id}")
                    result = result_resp.json()
                    if result.get("generations"):
                        img_url = result["generations"][0]["img"]
                        return {"image_url": img_url}
                    else:
                        raise HTTPException(status_code=500, detail="No image generated")
            
            raise HTTPException(status_code=408, detail="Timeout: Image generation took too long")

        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
