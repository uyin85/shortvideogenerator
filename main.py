from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
import urllib.parse
import uuid
import os
import subprocess
import random
import json
import re
from PIL import Image, ImageDraw

# --- CONFIGURATION ---
app = FastAPI(title="AI Fact Short Video Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("index.html"):
    app.mount("/static", StaticFiles(directory="."), name="static")

# Groq for fact generation
from groq import Groq
groq_client = None
if os.getenv("GROQ_API_KEY"):
    try:
        groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    except Exception as e:
        print(f"Groq init warning: {e}")

# ElevenLabs for TTS (using requests - simpler and more reliable)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

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

# ElevenLabs voice IDs (free tier)
VOICE_IDS = {
    "science": "21m00Tcm4TlvDq8ikWAM",      # Rachel - Clear, professional female
    "successful_person": "pNInz6obpgDQGcFmaJgB",  # Adam - Confident male
    "unsolved_mystery": "AZnzlk1XvdvUeBnXmlld",   # Domi - Mysterious female
    "history": "ErXwobaYiN019PkySvjV",      # Antoni - Deep narrative male
    "sports": "TxGEqnHWrfWFTfGW9XjX"          # Josh - Energetic male
}

# --- HELPER FUNCTIONS ---

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
        "science": ["Bananas are naturally radioactive.", "Octopuses have three hearts.", "Honey never spoils.", "Venus rotates backward.", "Your stomach acid can dissolve metal."],
        "successful_person": ["Oprah was fired from her first TV job.", "Steve Jobs was adopted.", "JK Rowling was rejected 12 times.", "Colonel Sanders started KFC at 65.", "Walt Disney was fired for lacking imagination."],
        "unsolved_mystery": ["The Voynich manuscript remains undeciphered.", "DB Cooper vanished after hijacking.", "The Bermuda Triangle mystery continues.", "Zodiac Killer was never caught.", "Oak Island money pit unsolved."],
        "history": ["Cleopatra lived closer to iPhone than pyramids.", "Oxford University predates Aztec Empire.", "The Great Wall visible from space myth.", "Napoleon was actually average height.", "Vikings discovered America before Columbus."],
        "sports": ["Michael Jordan was cut from high school team.", "Usain Bolt has scoliosis.", "Serena Williams holds 23 Grand Slams.", "Muhammad Ali won Olympic gold medal.", "Pele scored 1283 career goals."]
    }
    return defaults.get(category, defaults["science"])


def generate_audio_with_elevenlabs(text: str, audio_path: str, category: str = "science"):
    """Generate audio using ElevenLabs TTS API via REST"""
    if not ELEVENLABS_API_KEY:
        print("ElevenLabs API key not set, using fallback")
        return generate_audio_fallback(text, audio_path)
    
    try:
        # Select voice based on category
        voice_id = VOICE_IDS.get(category, VOICE_IDS["science"])
        
        print(f"Generating audio with ElevenLabs voice ID: {voice_id}")
        
        # Make API request
        url = f"{ELEVENLABS_API_URL}/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"ElevenLabs API error: {response.status_code} - {response.text}")
            return generate_audio_fallback(text, audio_path)
        
        # Save audio
        temp_mp3 = audio_path.replace(".mp3", "_temp.mp3")
        with open(temp_mp3, 'wb') as f:
            f.write(response.content)
        
        # Get audio duration using ffprobe
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                temp_mp3
            ], capture_output=True, text=True, timeout=10)
            duration = float(result.stdout.strip())
        except:
            # Estimate duration if ffprobe fails
            duration = len(text.split()) * 0.5 + 1.0
        
        # Convert to standard format with FFmpeg
        subprocess.run([
            "ffmpeg", "-i", temp_mp3,
            "-acodec", "libmp3lame", "-b:a", "128k",
            "-ar", "22050",
            audio_path, "-y", "-loglevel", "error"
        ], timeout=30)
        
        # Cleanup temp file
        if os.path.exists(temp_mp3):
            os.unlink(temp_mp3)
        
        success = os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000
        print(f"Audio generation {'succeeded' if success else 'failed'}, duration: {duration}s")
        return success, duration
        
    except Exception as e:
        print(f"ElevenLabs TTS error: {e}")
        return generate_audio_fallback(text, audio_path)


def generate_audio_fallback(text: str, audio_path: str):
    """Generate silent audio as fallback"""
    try:
        words = text.split()
        duration = len(words) * 0.5 + 1.0  # ~0.5s per word
        
        # Create silent audio
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
            "-t", str(duration), "-acodec", "libmp3lame", "-b:a", "128k",
            audio_path, "-y", "-loglevel", "error"
        ], timeout=30)
        
        return os.path.exists(audio_path), duration
    except Exception as e:
        print(f"Audio fallback error: {e}")
        return False, 0.0


def generate_word_timings(text: str, duration: float):
    """Generate word timings for karaoke effect"""
    words = text.split()
    if not words:
        return []
    
    # Simple equal distribution
    time_per_word = duration / len(words)
    timings = []
    
    for i, word in enumerate(words):
        start_time = i * time_per_word
        end_time = (i + 1) * time_per_word
        timings.append({
            "word": word,
            "start": start_time,
            "end": end_time
        })
    
    return timings


def create_karaoke_subtitles(word_timings, subtitle_path, effect="karaoke"):
    """Create ASS subtitle file with karaoke or other effects"""
    
    # ASS file header with styling
    ass_content = """[Script Info]
Title: AI Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 768
PlayResY: 768
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000088EF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,10,10,60,1
Style: Highlight,Arial,48,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,10,10,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    if effect == "karaoke":
        # Karaoke effect: word-by-word yellow highlight
        for timing in word_timings:
            start = format_time_ass(timing["start"])
            end = format_time_ass(timing["end"])
            
            # Build text with highlighted word
            highlighted_text = ""
            for t in word_timings:
                if t["word"] == timing["word"] and t["start"] == timing["start"]:
                    # Current word - yellow highlight
                    highlighted_text += "{\\c&H00FFFF&\\b1}" + t["word"] + "{\\c&HFFFFFF&\\b0} "
                else:
                    highlighted_text += t["word"] + " "
            
            ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{highlighted_text.strip()}\n"
    
    elif effect == "fade":
        # Fade in effect
        full_text = " ".join([w["word"] for w in word_timings])
        start = format_time_ass(word_timings[0]["start"])
        end = format_time_ass(word_timings[-1]["end"])
        ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\fad(800,500)}}{full_text}\n"
    
    elif effect == "typewriter":
        # Typewriter: reveal text progressively
        for i, timing in enumerate(word_timings):
            start = format_time_ass(timing["start"])
            end = format_time_ass(word_timings[-1]["end"])
            text_so_far = " ".join([w["word"] for w in word_timings[:i+1]])
            ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text_so_far}\n"
    
    elif effect == "bouncing":
        # Bouncing text animation
        full_text = " ".join([w["word"] for w in word_timings])
        start = format_time_ass(word_timings[0]["start"])
        end = format_time_ass(word_timings[-1]["end"])
        # Bounce animation using move and scale
        ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\move(384,800,384,680,0,500)\\t(0,300,\\fscx120\\fscy120)\\t(300,500,\\fscx100\\fscy100)}}{full_text}\n"
    
    else:  # static
        # Static text at bottom
        full_text = " ".join([w["word"] for w in word_timings])
        start = format_time_ass(word_timings[0]["start"])
        end = format_time_ass(word_timings[-1]["end"])
        ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{full_text}\n"
    
    with open(subtitle_path, "w", encoding="utf-8") as f:
        f.write(ass_content)


def format_time_ass(seconds):
    """Convert seconds to ASS timestamp format (0:00:00.00)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def create_video_with_subtitles(image_path, audio_path, subtitle_path, output_path, duration):
    """Create video with image, audio, and ASS subtitles"""
    
    # FFmpeg command with ASS subtitle overlay
    cmd = [
        "ffmpeg",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", f"scale=768:768:force_original_aspect_ratio=increase,crop=768:768,setsar=1,subtitles={subtitle_path}:force_style='Alignment=2,MarginV=50'",
        "-c:v", "libx264",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "128k",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-shortest",
        "-y",
        "-loglevel", "error",
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, timeout=90, capture_output=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr.decode()}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        print(f"Video creation error: {e}")
        return False


def generate_image_pollinations(prompt, path):
    try:
        # Enhanced prompt for better visuals
        enhanced_prompt = f"high quality cinematic image: {prompt}, 4k, detailed, vibrant colors"
        url = f"https://pollinations.ai/p/{urllib.parse.quote(enhanced_prompt)}?width=768&height=768&nologo=true&enhance=true"
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200 and len(resp.content) > 1000:
            with open(path, "wb") as f:
                f.write(resp.content)
            print(f"Image generated successfully from Pollinations")
            return True
    except Exception as e:
        print(f"Pollinations failed: {e}")
    return False


def generate_image_placeholder(prompt, path, category="science"):
    width, height = 768, 768
    colors = CATEGORY_COLORS.get(category, ["#4A90E2", "#50E3C2"])
    
    # Create gradient background
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    
    # Create gradient
    for y in range(height):
        r1, g1, b1 = tuple(int(colors[0].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        if len(colors) > 1:
            r2, g2, b2 = tuple(int(colors[1].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        else:
            r2, g2, b2 = r1, g1, b1
        
        ratio = y / height
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        
        for x in range(width):
            pixels[x, y] = (r, g, b)
    
    draw = ImageDraw.Draw(img)
    
    # Add decorative circles
    for _ in range(10):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(40, 180)
        alpha = random.randint(30, 80)
        color = colors[1] if len(colors) > 1 else colors[0]
        color_rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color_rgb)
    
    img.save(path, "JPEG", quality=90)
    print(f"Placeholder image generated")
    return True


# --- ENDPOINTS ---

@app.get("/")
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "AI Fact Video Generator with ElevenLabs Voice & Karaoke - Ready"}


@app.get("/facts")
def get_facts(category: str):
    if category not in PROMPTS:
        raise HTTPException(400, "Invalid category")
    facts = generate_facts_with_groq(category) or generate_facts_fallback(category)
    return {"facts": facts}


@app.get("/generate_video")
def generate_video(fact: str, category: str = "science", effect: str = "karaoke"):
    """Generate video with ElevenLabs voice and karaoke-style subtitles"""
    
    safe_fact = fact.strip()[:300]
    if not safe_fact:
        raise HTTPException(400, "Fact text is required")
    
    print(f"\n=== Starting video generation ===")
    print(f"Fact: {safe_fact}")
    print(f"Category: {category}")
    print(f"Effect: {effect}")
    
    # Temporary file paths
    img_path = f"/tmp/{uuid.uuid4()}.jpg"
    audio_path = f"/tmp/{uuid.uuid4()}.mp3"
    subtitle_path = f"/tmp/{uuid.uuid4()}.ass"
    output_path = f"/tmp/{uuid.uuid4()}.mp4"
    
    try:
        # Step 1: Generate image
        print("Step 1: Generating image...")
        image_prompt = f"{category} theme: {safe_fact[:100]}"
        if not (generate_image_pollinations(image_prompt, img_path) or 
                generate_image_placeholder(safe_fact, img_path, category)):
            raise HTTPException(500, "Image generation failed")
        
        # Step 2: Generate audio with ElevenLabs
        print("Step 2: Generating voice with ElevenLabs...")
        audio_success, duration = generate_audio_with_elevenlabs(safe_fact, audio_path, category)
        if not audio_success:
            raise HTTPException(500, "Audio generation failed")
        
        duration = max(duration, 3.0)  # Minimum 3 seconds
        print(f"Audio duration: {duration}s")
        
        # Step 3: Generate word timings for karaoke
        print("Step 3: Creating word timings...")
        word_timings = generate_word_timings(safe_fact, duration)
        print(f"Generated {len(word_timings)} word timings")
        
        # Step 4: Create subtitle file with selected effect
        print(f"Step 4: Creating {effect} subtitles...")
        create_karaoke_subtitles(word_timings, subtitle_path, effect)
        
        # Step 5: Create final video with subtitles
        print("Step 5: Composing final video...")
        if not create_video_with_subtitles(img_path, audio_path, subtitle_path, output_path, duration):
            raise HTTPException(500, "Video composition failed")
        
        print(f"Video created successfully: {os.path.getsize(output_path)} bytes")
        
        # Cleanup temp files
        for temp_file in [img_path, audio_path, subtitle_path]:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        # Stream video response
        def iterfile():
            try:
                with open(output_path, "rb") as f:
                    yield from f
            finally:
                if os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except:
                        pass
        
        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename=video_{effect}_{category}.mp4"}
        )
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        # Cleanup on error
        for temp_file in [img_path, audio_path, subtitle_path, output_path]:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
        raise HTTPException(500, f"Video generation error: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "groq_available": groq_client is not None,
        "elevenlabs_available": ELEVENLABS_API_KEY is not None
    }


# --- Run server ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
