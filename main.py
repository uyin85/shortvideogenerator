from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
import requests
import urllib.parse
import uuid
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
        response = requests.get(f"https://text.pollinations.ai/{encoded}", timeout=45)
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
        return {"error": f"Failed to fetch facts: {str(e)}"}

@app.get("/generate_video")
async def generate_video(fact: str):
    try:
        safe_fact = fact.strip()
        encoded = urllib.parse.quote(safe_fact)

        # Generate image from Pollinations
        img_url = f"https://pollinations.ai/p/{encoded}?width=768&height=768&nologo=true"
        img_data = requests.get(img_url, timeout=30).content
        img_path = f"/tmp/{uuid.uuid4()}.jpg"
        with open(img_path, "wb") as f:
            f.write(img_data)

        # Generate audio using gTTS (100% reliable)
        audio_path = f"/tmp/{uuid.uuid4()}.mp3"
        tts = gTTS(text=safe_fact, lang='en', slow=False)
        tts.save(audio_path)

        # Build video
        audio_clip = AudioFileClip(audio_path)
        duration = min(audio_clip.duration, 20)

        image_clip = ImageClip(img_path).set_duration(duration)
        txt_clip = TextClip(
            safe_fact,
            fontsize=40,
            font="Arial-Bold",
            color="white",
            size=image_clip.size,
            method="caption",
            align="center",
            stroke_color="black",
            stroke_width=1.5
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
