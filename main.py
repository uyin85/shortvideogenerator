from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
import urllib.parse
import uuid
import os
import subprocess
import random
from PIL import Image, ImageDraw
import pyttsx3

# --- CONFIGURATION ---
app = FastAPI(title="AI Fact Short Video Generator")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: serve static files (e.g., index.html)
if os.path.exists("index.html"):
    app.mount("/static", StaticFiles(directory="."), name="static")

# Groq for fact generation (voice is offline via pyttsx3)
from groq import Groq
groq_client = None
if os.getenv("GROQ_API_KEY"):
    try:
        groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    except Exception as e:
        print(f"Groq init warning: {e}")

# Prompts
PROMPTS = {
    "science": "Give me 5 short surprising science facts. One sentence each, under 15 words.",
    "successful_person": "Give me 5 short inspiring facts about successful people. One sentence each, under 15 words.",
    "unsolved_mystery": "Give me 5 short unsolved mysteries. One sentence each, under 15 words.",
    "history": "Give me 5 short memorable history facts. One sentence each, under 15 words.",
    "sports": "Give me 5 short legendary sports facts. One sentence each, under 15 words."
}

CATEGORY_COLORS = {
    "science": ["#4A90E2", "#50E3C2"],
    "successful_person": ["#F5A623", "#BD10E0"],
    "unsolved_mystery": ["#8B572A", "#4A4A4A"],
    "history": ["#8B4513", "#CD853F"],
    "sports": ["#FF6B6B", "#4ECDC4"]
}

# --- HELPER FUNCTIONS ---

def wrap_text_for_ffmpeg(text: str, max_chars_per_line: int = 30) -> str:
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(word) > max_chars_per_line:
            if current_line:
                lines.append(current_line)
                current_line = ""
            for i in range(0, len(word), max_chars_per_line):
                lines.append(word[i:i+max_chars_per_line])
        elif len(current_line) + len(word) + (1 if current_line else 0) <= max_chars_per_line:
            current_line = f"{current_line} {word}".strip()
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\\n".join(lines).replace("'", "'\\\\\\''").replace(":", "\\\\:")  # FFmpeg-safe


def generate_facts_with_groq(category: str):
    if not groq_client:
        return None
    try:
        prompt = PROMPTS.get(category, PROMPTS["science"])
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return exactly 5 facts, one per line, no bullets, no numbers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.8
        )
        lines = response.choices[0].message.content.strip().split("\n")
        facts = []
        for line in lines:
            cleaned = line.strip().strip("\"'•-—12345.")
            if 10 < len(cleaned) < 120:
                facts.append(cleaned)
            if len(facts) >= 5:
                break
        return facts[:5] or None
    except Exception as e:
        print(f"Groq fact gen failed: {e}")
        return None


def generate_facts_fallback(category: str):
    defaults = {
        "science": [
            "Bananas are naturally radioactive.",
            "Octopuses have three hearts.",
            "Honey never spoils.",
            "Venus rotates backward.",
            "Your stomach acid can dissolve metal."
        ]
    }
    return defaults.get(category, defaults["science"])


def generate_audio_with_pyttsx3(text: str, audio_path: str):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)

        wav_path = audio_path.replace(".mp3", ".wav")
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        engine.stop()

        if os.path.exists(wav_path):
            subprocess.run([
                "ffmpeg", "-i", wav_path,
                "-acodec", "libmp3lame", "-b:a", "128k",
                audio_path, "-y", "-loglevel", "error"
            ], timeout=30)
            os.unlink(wav_path)
            return os.path.getsize(audio_path) > 1000
    except Exception as e:
        print(f"TTS error: {e}")
    return False


def create_video(image_path, audio_path, text, output_path, duration):
    font_part = ""
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.exists(font_path):
        font_part = f"fontfile={font_path}:"

    filter_str = (
        f"[0:v]scale=768:768:force_original_aspect_ratio=increase,crop=768:768,setsar=1/1[bg];"
        f"[bg]drawtext=text='{text}':fontcolor=white:fontsize=28:{font_part}"
        f"box=1:boxcolor=black@0.6:boxborderw=12:x=(w-text_w)/2:y=h-text_h-80[outv]"
    )

    cmd = [
        "ffmpeg", "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-filter_complex", filter_str,
        "-map", "[outv]", "-map", "1:a",
        "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p", "-shortest", "-y", "-loglevel", "error",
        output_path
    ]
    return subprocess.run(cmd, timeout=60).returncode == 0


def generate_image_pollinations(prompt, path):
    try:
        url = f"https://pollinations.ai/p/{urllib.parse.quote(prompt)}?width=768&height=768&nologo=true"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            with open(path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        print(f"Pollinations failed: {e}")
    return False


def generate_image_placeholder(prompt, path, category="science"):
    width, height = 768, 768
    colors = CATEGORY_COLORS.get(category, ["#4A90E2", "#50E3C2"])
    img = Image.new("RGB", (width, height), colors[0])
    draw = ImageDraw.Draw(img)
    for _ in range(5):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(50, 200)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=random.choice(colors[1:]), width=0)
    img.save(path, "JPEG", quality=85)
    return True


# --- ENDPOINTS ---

@app.get("/")
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "AI Fact Video Generator - Ready"}


@app.get("/facts")
def get_facts(category: str):
    if category not in PROMPTS:
        raise HTTPException(400, "Invalid category")
    facts = generate_facts_with_groq(category) or generate_facts_fallback(category)
    return {"facts": facts}


@app.get("/generate_video")
def generate_video(fact: str, category: str = "science"):
    safe_fact = fact.strip()[:300]
    img_path = f"/tmp/{uuid.uuid4()}.jpg"
    audio_path = f"/tmp/{uuid.uuid4()}.mp3"
    output_path = f"/tmp/{uuid.uuid4()}.mp4"

    # Image
    if not (generate_image_pollinations(safe_fact, img_path) or generate_image_placeholder(safe_fact, img_path, category)):
        raise HTTPException(500, "Image generation failed")

    # Audio (offline TTS)
    duration = max(len(safe_fact.split()) * 0.4, 4.0)
    if not generate_audio_with_pyttsx3(safe_fact, audio_path):
        # Fallback to silent
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
            "-t", str(int(duration)), "-q:a", "0", "-c:a", "libmp3lame",
            audio_path, "-y", "-loglevel", "error"
        ], timeout=30)

    # Video
    wrapped = wrap_text_for_ffmpeg(safe_fact, 30)
    if not create_video(img_path, audio_path, wrapped, output_path, duration):
        raise HTTPException(500, "Video creation failed")

    # Cleanup & stream
    for p in [img_path, audio_path]:
        if os.path.exists(p):
            os.unlink(p)

    def iterfile():
        with open(output_path, "rb") as f:
            yield from f
        if os.path.exists(output_path):
            os.unlink(output_path)

    return StreamingResponse(iterfile(), media_type="video/mp4")


# --- Render.com requires this ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
