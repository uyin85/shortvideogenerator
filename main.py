from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
import requests
import urllib.parse
import uuid
import os
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip

app = FastAPI(title="AI Fact Short Video Generator")

PROMPTS = {
    "science": "Give me 5 short surprising science facts. One sentence each, under 15 words.",
    "successful_person": "Give me 5 short inspiring facts about successful people. One sentence each, under 15 words.",
    "unsolved_mystery": "Give me 5 short unsolved mysteries. One sentence each, under 15 words.",
    "history": "Give me 5 short memorable history facts. One sentence each, under 15 words.",
    "sports": "Give me 5 short legendary sports facts. One sentence each, under 15 words."
}

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.get("/facts")
async def get_facts(category: str):
    prompt = PROMPTS.get(category)
    if not prompt:
        return {"error": "Invalid category"}

    try:
        encoded = urllib.parse.quote(prompt)
        response = requests.get(f"https://text.pollinations.ai/{encoded}", timeout=15)
        if response.status_code != 200:
            return {"error": "AI failed to generate facts"}

        lines = response.text.strip().split('\n')
        facts = []
        for line in lines:
            cleaned = line.strip()
            # Remove leading bullets/numbers
            for char in "•-0123456789. ":
                if cleaned.startswith(char):
                    cleaned = cleaned.lstrip(char)
            if cleaned and len(cleaned) > 10:
                facts.append(cleaned)
            if len(facts) >= 5:
                break
        return {"facts": facts[:5]}
    except Exception as e:
        return {"error": f"Network error: {str(e)}"}

@app.get("/generate_video")
async def generate_video(fact: str):
    try:
        safe_fact = fact.strip()
        encoded = urllib.parse.quote(safe_fact)

        # 1. Generate image
        img_url = f"https://pollinations.ai/p/{encoded}?width=768&height=768&nologo=true"
        img_data = requests.get(img_url, timeout=15).content
        img_path = f"/tmp/{uuid.uuid4()}.jpg"
        with open(img_path, "wb") as f:
            f.write(img_data)

        # 2. Generate audio
        audio_url = f"https://text.pollinations.ai/{encoded}?model=openai-audio&voice=nova"
        audio_data = requests.get(audio_url, timeout=15).content
        audio_path = f"/tmp/{uuid.uuid4()}.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        # 3. Load and measure audio duration
        audio_clip = AudioFileClip(audio_path)
        duration = min(audio_clip.duration, 20)

        # 4. Create video elements
        image_clip = ImageClip(img_path).set_duration(duration)
        
        # Caption (centered, with background for readability)
        txt_clip = TextClip(
            safe_fact,
            fontsize=42,
            font="Arial-Bold",
            color="white",
            size=(768, 768),
            method="caption",
            align="center",
            stroke_color="black",
            stroke_width=1
        ).set_duration(duration)

        # 5. Combine
        final_video = CompositeVideoClip([image_clip, txt_clip]).set_audio(audio_clip.subclip(0, duration))

        # 6. Export MP4
        output_path = f"/tmp/{uuid.uuid4()}.mp4"
        final_video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=f"/tmp/{uuid.uuid4()}.m4a",
            remove_temp=True,
            logger=None,
            threads=4
        )

        # 7. Stream to user
        def iterfile():
            with open(output_path, "rb") as f:
                yield from f
        return StreamingResponse(iterfile(), media_type="video/mp4")

    except Exception as e:
        return {"error": str(e)}
