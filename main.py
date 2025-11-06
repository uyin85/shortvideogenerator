from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
import requests
import urllib.parse
import uuid
import os
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip
from gtts import gTTS

app = FastAPI(title="AI Fact Short Video Generator")

PROMPTS = {
    "science": "Give me 5 short surprising science facts. One sentence each, under 15 words.",
    "successful_person": "Give me 5 short inspiring facts about successful people. One sentence each, under 15 words.",
    "unsolved_mystery": "Give me 5 short unsolved mysteries. One sentence each, under 15 words.",
    "history": "Give me 5 short memorable history facts. One sentence each, under 15 words.",
    "sports": "Give me 5 short legendary sports facts. One sentence each, under 15 words."
}

def is_valid_mp3(data: bytes) -> bool:
    """Check if data starts with MP3 sync bytes or ID3 tag."""
    return data.startswith(b'\xff\xfb') or data.startswith(b'ID3')

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
            for prefix in "•-0123456789. ":
                cleaned = cleaned.lstrip(prefix)
            if 10 < len(cleaned) < 120:
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

        # 2. Try Pollinations TTS first
        audio_url = f"https://text.pollinations.ai/{encoded}?model=openai-audio&voice=nova"
        audio_resp = requests.get(audio_url, timeout=15)
        audio_path = f"/tmp/{uuid.uuid4()}.mp3"

        if is_valid_mp3(audio_resp.content):
            with open(audio_path, "wb") as f:
                f.write(audio_resp.content)
        else:
            # Fallback to gTTS
            tts = gTTS(text=safe_fact, lang='en', slow=False)
            tts.save(audio_path)

        # 3. Load audio and create video
        audio_clip = AudioFileClip(audio_path)
        duration = min(audio_clip.duration, 20)

        image_clip = ImageClip(img_path).set_duration(duration)
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

        final_video = CompositeVideoClip([image_clip, txt_clip]).set_audio(audio_clip.subclip(0, duration))
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

        def iterfile():
            with open(output_path, "rb") as f:
                yield from f
        return StreamingResponse(iterfile(), media_type="video/mp4")

    except Exception as e:
        return {"error": f"Video generation failed: {str(e)}"}
