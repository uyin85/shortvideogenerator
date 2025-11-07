from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
import urllib.parse
import uuid
import os
import subprocess
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip
import aiofiles
import random
from PIL import Image, ImageDraw
from groq import Groq

app = FastAPI(title="AI Fact Short Video Generator")

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

PROMPTS = {
    "science": "Give me 5 short surprising science facts. One sentence each, under 15 words. Make them interesting and educational.",
    "successful_person": "Give me 5 short inspiring facts about successful people. One sentence each, under 15 words. Make them motivational.",
    "unsolved_mystery": "Give me 5 short unsolved mysteries. One sentence each, under 15 words. Make them intriguing.",
    "history": "Give me 5 short memorable history facts. One sentence each, under 15 words. Make them fascinating.",
    "sports": "Give me 5 short legendary sports facts. One sentence each, under 15 words. Make them exciting."
}

CATEGORY_COLORS = {
    "science": ["#4A90E2", "#50E3C2", "#9013FE", "#417505"],
    "successful_person": ["#F5A623", "#BD10E0", "#7ED321", "#B8E986"],
    "unsolved_mystery": ["#8B572A", "#4A4A4A", "#9B9B9B", "#D0021B"],
    "history": ["#8B4513", "#CD853F", "#D2691E", "#A0522D"],
    "sports": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
}

def generate_facts_with_groq(category: str):
    try:
        prompt = PROMPTS.get(category, PROMPTS["science"])
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates interesting, concise facts. Always return exactly 5 facts, one per line, without numbers or bullet points."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.8
        )
        facts_text = response.choices[0].message.content.strip()
        lines = facts_text.split('\n')
        facts = []
        for line in lines:
            cleaned = line.strip()
            for prefix in ["•", "-", "—", "–", "1.", "2.", "3.", "4.", "5."]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            cleaned = cleaned.strip('"').strip("'").strip()
            if 10 < len(cleaned) < 120 and cleaned:
                facts.append(cleaned)
            if len(facts) >= 5:
                break
        return facts[:5]
    except Exception as e:
        print(f"Groq facts error: {e}")
        return None

def generate_audio_with_groq(text: str, audio_path: str):
    try:
        response = groq_client.audio.speech.create(
            model="playai-tts",
            voice="Fritz-PlayAI",
            input=text,
            response_format="wav"
        )
        wav_path = audio_path.replace('.mp3', '.wav')
        response.write_to_file(wav_path)
        if os.path.exists(wav_path):
            subprocess.run([
                "ffmpeg", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", audio_path, "-y", "-loglevel", "error"
            ], capture_output=True, timeout=30)
            os.unlink(wav_path)
            return os.path.getsize(audio_path) > 1000
        return False
    except Exception as e:
        print(f"Groq TTS error: {e}")
        return False

def create_karaoke_subtitles(text: str, duration: float, screen_size):
    words = text.split()
    if not words:
        return TextClip("", size=(1, 1)).set_duration(0)
    
    word_duration = duration / len(words)
    w, h = screen_size
    y_pos = h - 120

    clips = []
    for i in range(len(words)):
        start = i * word_duration
        end = (i + 1) * word_duration
        spoken = " ".join(words[:i+1])
        upcoming = " ".join(words[i+1:])

        # Background
        try:
            bg = TextClip(
                ' ' * 100, fontsize=50, color='white',
                bg_color='black', size=(w - 40, 80),
                method='caption', align='center'
            ).set_opacity(0.6).set_position(('center', y_pos)).set_start(start).set_duration(word_duration)
            clips.append(bg)
        except:
            pass

        # Spoken text (gold)
        full_text = spoken + (" " + upcoming if upcoming else "")
        try:
            if upcoming:
                # Show full sentence, but we can't highlight part — so show spoken in gold
                txt = TextClip(
                    spoken, fontsize=46, color='#FFD700',
                    font='DejaVu-Sans-Bold',
                    stroke_color='black', stroke_width=2,
                    method='caption', align='center'
                )
                # Center it
                txt = txt.set_position(('center', y_pos)).set_start(start).set_duration(word_duration)
                clips.append(txt)
            else:
                txt = TextClip(
                    full_text, fontsize=46, color='#FFD700',
                    font='DejaVu-Sans-Bold',
                    stroke_color='black', stroke_width=2,
                    size=(w * 0.9, None), method='caption', align='center'
                ).set_position(('center', y_pos)).set_start(start).set_duration(word_duration)
                clips.append(txt)
        except Exception as e:
            print(f"⚠️ Subtitle render error: {e}")
            fallback = TextClip(
                text, fontsize=46, color='white',
                stroke_color='black', stroke_width=1,
                size=(w * 0.9, None), method='caption', align='center'
            ).set_position(('center', y_pos)).set_duration(duration)
            return fallback

    if clips:
        return CompositeVideoClip(clips)
    else:
        return TextClip(
            text, fontsize=46, color='white',
            stroke_color='black', stroke_width=1,
            size=(w * 0.9, None), method='caption', align='center'
        ).set_position(('center', y_pos)).set_duration(duration)

def generate_gradient_background(width=768, height=768, colors=None):
    if colors is None:
        colors = ["#4A90E2", "#50E3C2"]
    image = Image.new('RGB', (width, height), colors[0])
    draw = ImageDraw.Draw(image)
    for _ in range(5):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(50, 200)
        c = random.choice(colors[1:]) if len(colors) > 1 else colors[0]
        draw.ellipse([x-r, y-r, x+r, y+r], fill=c)
    return image

def generate_image_pollinations(text: str, img_path: str):
    try:
        encoded = urllib.parse.quote(text)
        url = f"https://pollinations.ai/p/{encoded}?width=768&height=768&nologo=true"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        print(f"Pollinations error: {e}")
    return False

def generate_image_placeholder(text: str, img_path: str, category="science"):
    try:
        img = generate_gradient_background(768, 768, CATEGORY_COLORS.get(category, CATEGORY_COLORS["science"]))
        draw = ImageDraw.Draw(img)
        cx, cy = 384, 384
        draw.ellipse([cx-200, cy-200, cx+200, cy+200], outline="white", width=5)
        img.save(img_path, "JPEG", quality=85)
        return True
    except Exception as e:
        print(f"Placeholder image error: {e}")
        return False

def generate_silent_audio(duration: int, audio_path: str):
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(duration), "-q:a", "2", "-y", audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.exists(audio_path)

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.get("/facts")
async def get_facts(category: str):
    if category not in PROMPTS:
        raise HTTPException(status_code=400, detail="Invalid category")
    facts = generate_facts_with_groq(category)
    if not facts:
        raise HTTPException(status_code=500, detail="Failed to generate facts")
    return {"facts": facts[:5]}

@app.get("/generate_video")
async def generate_video(fact: str, category: str = "science"):
    safe_fact = fact.strip()[:300]
    if not safe_fact:
        raise HTTPException(status_code=400, detail="Fact is empty")

    # Generate image
    img_path = f"/tmp/{uuid.uuid4()}.jpg"
    if not (generate_image_pollinations(safe_fact, img_path) or generate_image_placeholder(safe_fact, img_path, category)):
        raise HTTPException(status_code=500, detail="Image generation failed")

    # Generate audio
    audio_path = f"/tmp/{uuid.uuid4()}.mp3"
    audio_ok = generate_audio_with_groq(safe_fact, audio_path)

    # Determine duration
    duration = 5.0
    if audio_ok:
        try:
            ac = AudioFileClip(audio_path)
            duration = min(ac.duration, 15.0)
        except:
            audio_ok = False
            duration = max(len(safe_fact.split()) * 0.45, 5.0)
    else:
        duration = max(len(safe_fact.split()) * 0.45, 5.0)
        generate_silent_audio(int(duration), audio_path)

    # Build video
    image_clip = ImageClip(img_path).set_duration(duration)
    screen_size = (image_clip.w, image_clip.h)

    try:
        subtitle_clip = create_karaoke_subtitles(safe_fact, duration, screen_size)
    except Exception as e:
        print(f"Final subtitle fallback: {e}")
        subtitle_clip = TextClip(
            safe_fact, fontsize=46, color='white',
            stroke_color='black', stroke_width=1,
            size=(screen_size[0]*0.9, None), method='caption', align='center'
        ).set_position(('center', screen_size[1]-120)).set_duration(duration)

    # Combine
    if audio_ok and os.path.exists(audio_path):
        audio_clip = AudioFileClip(audio_path).set_duration(duration)
        final = CompositeVideoClip([image_clip, subtitle_clip]).set_audio(audio_clip)
    else:
        final = CompositeVideoClip([image_clip, subtitle_clip])

    # Export
    out_path = f"/tmp/{uuid.uuid4()}.mp4"
    final.write_videofile(
        out_path,
        fps=24,
        codec="libx264",
        audio_codec="aac" if audio_ok else None,
        remove_temp=True,
        logger=None,
        verbose=False,
        ffmpeg_params=["-preset", "fast", "-crf", "28"]
    )

    # Cleanup
    for f in [img_path, audio_path]:
        if os.path.exists(f):
            os.unlink(f)

    def iterfile():
        with open(out_path, "rb") as f:
            yield from f
        if os.path.exists(out_path):
            os.unlink(out_path)

    return StreamingResponse(iterfile(), media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
