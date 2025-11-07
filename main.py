from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
import urllib.parse
import uuid
import os
import subprocess
import aiofiles
import random
from PIL import Image, ImageDraw
import time
from groq import Groq

# --- CONFIGURATION ---

app = FastAPI(title="AI Fact Short Video Generator")

try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"Warning: Groq client initialization failed. Check GROQ_API_KEY. Error: {e}")
    groq_client = None

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
            current_line = (current_line + " " + word).strip()
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    # Escape for FFmpeg
    escaped = '\\n'.join(lines).replace("'", "'\\\\\\''").replace(":", "\\\\:")
    return escaped


def generate_facts_with_groq(category: str):
    if not groq_client:
        print("Groq client not initialized.")
        return None
    try:
        print(f"Generating facts for {category} using Groq...")
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
        print(f"Groq facts generation failed: {e}")
        return None


def generate_audio_with_groq(text: str, audio_path: str):
    if not groq_client:
        return False
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
            try:
                os.unlink(wav_path)
            except:
                pass
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                return True
        return False
    except Exception as e:
        print(f"Groq TTS failed: {e}")
        return False


def create_video_with_simple_subtitles(image_path: str, audio_path: str, text: str, output_path: str, duration: float):
    try:
        # Build filter without hardcoded font (safer across systems)
        fontfile_part = ""
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if os.path.exists(font_path):
            fontfile_part = f"fontfile={font_path}:"

        filter_complex_str = (
            f"[0:v]scale=768:768:force_original_aspect_ratio=increase,crop=768:768,setsar=1/1[bg];"
            f"[bg]drawtext=text='{text}':"
            f"fontcolor=white:fontsize=28:"
            f"{fontfile_part}"
            f"box=1:boxcolor=black@0.6:boxborderw=12:"
            f"x=(w-text_w)/2:y=h-text_h-80[outv]"
        )

        cmd = [
            'ffmpeg',
            '-loop', '1',
            '-i', image_path,
            '-i', audio_path,
            '-filter_complex', filter_complex_str,
            '-map', '[outv]',
            '-map', '1:a',
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            '-y',
            '-loglevel', 'error',
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0

    except Exception as e:
        print(f"Simple subtitle video creation failed: {e}")
        return False


def generate_gradient_background(width=768, height=768, colors=None):
    if colors is None:
        colors = ["#4A90E2", "#50E3C2"]
    bg_color = colors[0]
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    for _ in range(5):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(50, 200)
        color = random.choice(colors[1:]) if len(colors) > 1 else colors[0]
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, width=0)
    return image


def generate_image_pollinations(text: str, img_path: str):
    try:
        encoded = urllib.parse.quote(text)
        img_url = f"https://pollinations.ai/p/{encoded}?width=768&height=768&nologo=true"
        response = requests.get(img_url, timeout=30)
        if response.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        print(f"Pollinations.ai error: {e}")
        return False


def generate_image_placeholder(text: str, img_path: str, category="science"):
    try:
        width, height = 768, 768
        colors = CATEGORY_COLORS.get(category, CATEGORY_COLORS["science"])
        image = generate_gradient_background(width, height, colors)
        draw = ImageDraw.Draw(image)
        center_x, center_y = width // 2, height // 2
        circle_radius = 200
        draw.ellipse([
            center_x - circle_radius, center_y - circle_radius,
            center_x + circle_radius, center_y + circle_radius
        ], outline="white", width=5)
        image.save(img_path, "JPEG", quality=85)
        return True
    except Exception as e:
        print(f"Placeholder image failed: {e}")
        return False


def generate_silent_audio(duration: int, audio_path: str):
    try:
        result = subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", str(duration), "-c:a", "libmp3lame", audio_path, "-y", "-loglevel", "error"
        ], capture_output=True, timeout=30)
        return os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
    except Exception as e:
        print(f"Silent audio failed: {e}")
        return False


# --- API ENDPOINTS ---

@app.get("/")
async def home():
    return FileResponse("index.html")


@app.get("/facts")
async def get_facts(category: str):
    if category not in PROMPTS:
        raise HTTPException(status_code=400, detail="Invalid category")
    if not groq_client:
        raise HTTPException(status_code=503, detail="AI service unavailable (GROQ_API_KEY missing/invalid).")
    facts = generate_facts_with_groq(category)
    if not facts:
        raise HTTPException(status_code=500, detail="Failed to generate facts with AI")
    return {"facts": facts[:5]}


@app.get("/generate_video")
async def generate_video(fact: str, category: str = "science"):
    safe_fact = fact.strip()[:300]
    print(f"🎬 Generating video for: '{safe_fact}'")

    img_path = f"/tmp/{uuid.uuid4()}.jpg"
    if not (generate_image_pollinations(safe_fact, img_path) or generate_image_placeholder(safe_fact, img_path, category)):
        raise HTTPException(status_code=500, detail="All image generation methods failed")

    audio_path = f"/tmp/{uuid.uuid4()}.mp3"
    duration = 5
    audio_generated = False

    if groq_client:
        if generate_audio_with_groq(safe_fact, audio_path):
            try:
                result = subprocess.run([
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
                ], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    duration = min(float(result.stdout.strip()), 15)
                    audio_generated = True
            except:
                pass

    if not audio_generated:
        duration = max(len(safe_fact.split()) / 2, 5)
        generate_silent_audio(int(duration), audio_path)

    # ✅ Critical: Wrap text BEFORE passing to video function
    wrapped_text = wrap_text_for_ffmpeg(safe_fact, max_chars_per_line=30)
    output_path = f"/tmp/{uuid.uuid4()}.mp4"

    if not create_video_with_simple_subtitles(img_path, audio_path, wrapped_text, output_path, duration):
        raise HTTPException(status_code=500, detail="Video creation failed")

    # Cleanup & stream
    for temp in [img_path, audio_path]:
        if os.path.exists(temp):
            os.unlink(temp)

    def iterfile():
        with open(output_path, "rb") as f:
            yield from f
        if os.path.exists(output_path):
            os.unlink(output_path)

    return StreamingResponse(iterfile(), media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    if not os.path.exists("/tmp"):
        os.makedirs("/tmp")
    uvicorn.run(app, host="0.0.0.0", port=8000)
