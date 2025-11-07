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
import time
from groq import Groq
import numpy as np

app = FastAPI(title="AI Fact Short Video Generator")

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (optional)
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
        print(f"Raw Groq response: {facts_text}")
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
        print(f"Parsed {len(facts)} facts: {facts}")
        return facts[:5]
    except Exception as e:
        print(f"Groq facts generation failed: {e}")
        return None

def generate_audio_with_groq(text: str, audio_path: str):
    try:
        print(f"Generating audio with Groq TTS: '{text}'")
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
                print(f"✓ Groq TTS audio generated: {os.path.getsize(audio_path)} bytes")
                return True
        print("❌ Groq TTS file creation failed")
        return False
    except Exception as e:
        print(f"❌ Groq TTS failed: {e}")
        return False

def create_karaoke_subtitles(text: str, duration: float, screen_size):
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return TextClip("", size=(1, 1)).set_duration(0)

    word_duration = duration / word_count
    screen_width, screen_height = screen_size
    text_y_position = screen_height - 120

    clips = []

    for i in range(word_count):
        start_time = i * word_duration
        end_time = (i + 1) * word_duration
        spoken_words = " ".join(words[:i+1])
        upcoming_words = " ".join(words[i+1:]) if i + 1 < word_count else ""

        # Background for readability
        try:
            bg_clip = TextClip(
                txt=' ' * 100,
                fontsize=50,
                color='white',
                bg_color='black',
                size=(screen_width - 40, 80),
                method='caption',
                align='center'
            ).set_opacity(0.6).set_position(('center', text_y_position)).set_duration(word_duration).set_start(start_time)
            clips.append(bg_clip)
        except Exception as e:
            print(f"Warning: Background subtitle failed: {e}")

        try:
            if spoken_words and upcoming_words:
                # Spoken (gold)
                highlight_clip = TextClip(
                    txt=spoken_words,
                    fontsize=46,
                    color='#FFD700',
                    stroke_color='black',
                    stroke_width=3,
                    method='caption',
                    align='center'
                ).set_position(('center', text_y_position)).set_duration(word_duration).set_start(start_time)
                clips.append(highlight_clip)

                # Upcoming (white) - render full line and mask? Simpler: just show full line with partial color not supported.
                # So fallback: show full line in white, then overwrite spoken part in gold (already done above).
                # For simplicity, skip word-by-word alignment and just show full sentence with progressive highlight in center.
            else:
                full_text = spoken_words + (" " + upcoming_words if upcoming_words else "")
                final_clip = TextClip(
                    txt=full_text,
                    fontsize=46,
                    color='#FFD700',
                    stroke_color='black',
                    stroke_width=3,
                    size=(screen_width * 0.9, None),
                    method='caption',
                    align='center'
                ).set_position(('center', text_y_position)).set_duration(word_duration).set_start(start_time)
                clips.append(final_clip)
        except Exception as e:
            print(f"Warning: Main subtitle failed: {e}")
            # Fallback to simple white text
            fallback = TextClip(
                txt=text,
                fontsize=46,
                color='white',
                stroke_color='black',
                stroke_width=2,
                size=(screen_width * 0.9, None),
                method='caption',
                align='center'
            ).set_position(('center', text_y_position)).set_duration(duration)
            return fallback

    if clips:
        return CompositeVideoClip(clips)
    else:
        return TextClip(
            txt=text,
            fontsize=46,
            color='white',
            stroke_color='black',
            stroke_width=2,
            size=(screen_width * 0.9, None),
            method='caption',
            align='center'
        ).set_position(('center', text_y_position)).set_duration(duration)

def generate_gradient_background(width=768, height=768, colors=None):
    if colors is None:
        colors = ["#4A90E2", "#50E3C2"]
    image = Image.new('RGB', (width, height), colors[0])
    draw = ImageDraw.Draw(image)
    for _ in range(5):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(50, 200)
        color = random.choice(colors[1:]) if len(colors) > 1 else colors[0]
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    return image

def generate_image_pollinations(text: str, img_path: str):
    try:
        print("Trying Pollinations.ai...")
        encoded = urllib.parse.quote(text)
        img_url = f"https://pollinations.ai/p/{encoded}?width=768&height=768&nologo=true"
        response = requests.get(img_url, timeout=30)
        if response.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(response.content)
            print("✓ Pollinations.ai image generated")
            return True
        else:
            print(f"✗ Pollinations.ai failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Pollinations.ai error: {e}")
        return False

def generate_image_placeholder(text: str, img_path: str, category="science"):
    try:
        print("Generating placeholder image...")
        width, height = 768, 768
        colors = CATEGORY_COLORS.get(category, CATEGORY_COLORS["science"])
        image = generate_gradient_background(width, height, colors)
        draw = ImageDraw.Draw(image)
        center_x, center_y = width // 2, height // 2
        draw.ellipse([center_x - 200, center_y - 200, center_x + 200, center_y + 200], outline="white", width=5)
        image.save(img_path, "JPEG", quality=85)
        print("✓ Placeholder image generated")
        return True
    except Exception as e:
        print(f"✗ Placeholder image failed: {e}")
        return False

def generate_silent_audio(duration: int, audio_path: str):
    try:
        print(f"Generating silent audio ({duration}s)...")
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", str(duration), "-c:a", "libmp3lame", audio_path, "-y", "-loglevel", "error"
        ], capture_output=True, timeout=30)
        return os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
    except Exception as e:
        print(f"Silent audio failed: {e}")
        return False

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.get("/facts")
async def get_facts(category: str):
    if category not in PROMPTS:
        return {"error": "Invalid category"}
    try:
        facts = generate_facts_with_groq(category)
        if not facts:
            return {"error": "Failed to generate facts"}
        return {"facts": facts[:5]}
    except Exception as e:
        print(f"Facts error: {e}")
        return {"error": str(e)}

@app.get("/generate_video")
async def generate_video(fact: str, category: str = "science"):
    safe_fact = fact.strip()[:300]
    print(f"🎬 Generating video for: '{safe_fact}'")

    img_path = f"/tmp/{uuid.uuid4()}.jpg"
    if not (generate_image_pollinations(safe_fact, img_path) or generate_image_placeholder(safe_fact, img_path, category)):
        raise HTTPException(status_code=500, detail="Image generation failed")

    audio_path = f"/tmp/{uuid.uuid4()}.mp3"
    audio_generated = generate_audio_with_groq(safe_fact, audio_path)

    duration = 5
    if audio_generated:
        try:
            audio_clip = AudioFileClip(audio_path)
            duration = min(audio_clip.duration, 15)
        except:
            audio_generated = False
            duration = max(len(safe_fact.split()) * 0.4, 5)
    else:
        duration = max(len(safe_fact.split()) * 0.4, 5)
        generate_silent_audio(int(duration), audio_path)

    image_clip = ImageClip(img_path).set_duration(duration)
    screen_size = (image_clip.w, image_clip.h)

    try:
        karaoke_subtitles = create_karaoke_subtitles(safe_fact, duration, screen_size)
    except Exception as e:
        print(f"⚠️ Subtitle crash: {e}")
        karaoke_subtitles = TextClip(
            safe_fact, fontsize=46, color='white', stroke_color='black', stroke_width=2,
            size=(screen_size[0]*0.9, None), method='caption', align='center'
        ).set_position(('center', screen_size[1]-120)).set_duration(duration)

    final_video = image_clip
    if audio_generated and os.path.exists(audio_path):
        try:
            audio_clip = AudioFileClip(audio_path).set_duration(duration)
            final_video = CompositeVideoClip([image_clip, karaoke_subtitles]).set_audio(audio_clip)
        except:
            final_video = CompositeVideoClip([image_clip, karaoke_subtitles])
    else:
        final_video = CompositeVideoClip([image_clip, karaoke_subtitles])

    output_path = f"/tmp/{uuid.uuid4()}.mp4"
    final_video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac" if audio_generated else None,
        remove_temp=True,
        logger=None,
        verbose=False,
        ffmpeg_params=['-preset', 'fast', '-crf', '28']
    )

    # Cleanup
    for f in [img_path, audio_path]:
        if os.path.exists(f):
            os.unlink(f)

    def iterfile():
        with open(output_path, "rb") as f:
            yield from f
        if os.path.exists(output_path):
            os.unlink(output_path)

    return StreamingResponse(iterfile(), media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
