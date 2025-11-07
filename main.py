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

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

PROMPTS = {
    "science": "Give me 5 short surprising science facts. One sentence each, under 15 words. Make them interesting and educational.",
    "successful_person": "Give me 5 short inspiring facts about successful people. One sentence each, under 15 words. Make them motivational.",
    "unsolved_mystery": "Give me 5 short unsolved mysteries. One sentence each, under 15 words. Make them intriguing.",
    "history": "Give me 5 short memorable history facts. One sentence each, under 15 words. Make them fascinating.",
    "sports": "Give me 5 short legendary sports facts. One sentence each, under 15 words. Make them exciting."
}

# Background colors
CATEGORY_COLORS = {
    "science": ["#4A90E2", "#50E3C2", "#9013FE", "#417505"],
    "successful_person": ["#F5A623", "#BD10E0", "#7ED321", "#B8E986"],
    "unsolved_mystery": ["#8B572A", "#4A4A4A", "#9B9B9B", "#D0021B"],
    "history": ["#8B4513", "#CD853F", "#D2691E", "#A0522D"],
    "sports": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
}

def generate_facts_with_groq(category: str):
    """Generate facts using Groq API with llama-3.1-8b-instant"""
    try:
        print(f"Generating facts for {category} using Groq...")
        
        prompt = PROMPTS.get(category, PROMPTS["science"])
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates interesting, concise facts. Always return exactly 5 facts, one per line, without numbers or bullet points."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0.8
        )
        
        facts_text = response.choices[0].message.content.strip()
        print(f"Raw Groq response: {facts_text}")
        
        # Parse the facts - split by newlines and clean
        lines = facts_text.split('\n')
        facts = []
        
        for line in lines:
            cleaned = line.strip()
            # Remove numbers, bullets, and other prefixes
            for prefix in ["•", "-", "—", "–", "—", "1.", "2.", "3.", "4.", "5."]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            # Remove quotes and other common formatting
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
    """Generate audio using Groq TTS API"""
    try:
        print(f"Generating audio with Groq TTS: '{text}'")
        
        response = groq_client.audio.speech.create(
            model="playai-tts",
            voice="Fritz-PlayAI",
            input=text,
            response_format="wav"
        )
        
        # Save as WAV first
        wav_path = audio_path.replace('.mp3', '.wav')
        response.write_to_file(wav_path)
        
        # Convert WAV to MP3 using ffmpeg
        if os.path.exists(wav_path):
            subprocess.run([
                "ffmpeg",
                "-i", wav_path,
                "-codec:a", "libmp3lame",
                "-qscale:a", "2",
                audio_path,
                "-y",
                "-loglevel", "error"
            ], capture_output=True, timeout=30)
            
            # Clean up WAV file
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

def format_time(seconds: float) -> str:
    """Convert seconds to ASS time format: H:MM:SS.cc"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"

def create_karaoke_ass_subtitle(text: str, duration: float, ass_path: str):
    """Create ASS subtitle with word-by-word karaoke effect"""
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return create_static_ass_subtitle(text, duration, ass_path)
    
    word_duration = duration / word_count
    
    ass_content = """[Script Info]
Title: Karaoke Subtitles
ScriptType: v4.00+
PlayResX: 768
PlayResY: 768

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Spoken,Arial,48,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,2,10,10,50,1
Style: Upcoming,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    current_time = 0
    for i in range(word_count):
        start_time = format_time(current_time)
        end_time = format_time(current_time + word_duration)
        
        # Words spoken so far (yellow)
        spoken_words = " ".join(words[:i+1])
        # Words upcoming (white)
        upcoming_words = " ".join(words[i+1:]) if i + 1 < word_count else ""
        
        if upcoming_words:
            dialogue = f"Dialogue: 0,{start_time},{end_time},Spoken,,0,0,0,,{spoken_words}\\N{{\\rUpcoming}}{upcoming_words}"
        else:
            dialogue = f"Dialogue: 0,{start_time},{end_time},Spoken,,0,0,0,,{spoken_words}"
        
        ass_content += dialogue + "\n"
        current_time += word_duration
    
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    print(f"✓ Karaoke subtitle created: {text}")

def create_fade_in_ass_subtitle(text: str, duration: float, ass_path: str):
    """Create ASS subtitle with fade-in effect"""
    fade_duration = min(2.0, duration / 3)  # Fade over 2 seconds or 1/3 of duration
    
    ass_content = f"""[Script Info]
Title: Fade-in Subtitles
ScriptType: v4.00+
PlayResX: 768
PlayResY: 768

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: FadeStyle,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
# Fade in
Dialogue: 0,0:00:00.00,{fade_duration:.2f},FadeStyle,,0,0,0,fade(0,255),{text}
# Stay visible
Dialogue: 0,{fade_duration:.2f},{duration:.2f},FadeStyle,,0,0,0,,{text}
"""
    
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    print(f"✓ Fade-in subtitle created: {text}")

def create_typewriter_ass_subtitle(text: str, duration: float, ass_path: str):
    """Create ASS subtitle with typewriter effect"""
    char_duration = duration / max(len(text), 1)
    
    ass_content = """[Script Info]
Title: Typewriter Subtitles
ScriptType: v4.00+
PlayResX: 768
PlayResY: 768

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TypeStyle,Arial,48,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    current_time = 0
    for i in range(1, len(text) + 1):
        start_time = format_time(current_time)
        end_time = format_time(min(current_time + char_duration, duration))
        
        visible_text = text[:i]
        if visible_text.strip():
            dialogue = f"Dialogue: 0,{start_time},{end_time},TypeStyle,,0,0,0,,{visible_text}"
            ass_content += dialogue + "\n"
        
        current_time += char_duration
        if current_time >= duration:
            break
    
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    print(f"✓ Typewriter subtitle created: {text}")

def create_bouncing_ass_subtitle(text: str, duration: float, ass_path: str):
    """Create ASS subtitle with bouncing animation"""
    bounce_duration = min(2.0, duration / 3)
    
    ass_content = f"""[Script Info]
Title: Bouncing Subtitles
ScriptType: v4.00+
PlayResX: 768
PlayResY: 768

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: BounceStyle,Arial,48,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
# Bounce in from bottom
Dialogue: 0,0:00:00.00,{bounce_duration:.2f},BounceStyle,,0,0,0,move(384,800,384,650),{text}
# Stay with slight movement
Dialogue: 0,{bounce_duration:.2f},{duration:.2f},BounceStyle,,0,0,0,,{text}
"""
    
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    print(f"✓ Bouncing subtitle created: {text}")

def create_static_ass_subtitle(text: str, duration: float, ass_path: str):
    """Create static ASS subtitle"""
    ass_content = f"""[Script Info]
Title: Static Subtitles
ScriptType: v4.00+
PlayResX: 768
PlayResY: 768

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: StaticStyle,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,{duration:.2f},StaticStyle,,0,0,0,,{text}
"""
    
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    print(f"✓ Static subtitle created: {text}")

def create_video_with_subtitles_ffmpeg(image_path: str, audio_path: str, subtitle_path: str, output_path: str, duration: float):
    """Create video with burned-in subtitles using FFmpeg"""
    
    cmd = [
        'ffmpeg',
        '-loop', '1',              # Loop the single image
        '-i', image_path,          # Input image
        '-i', audio_path,          # Input audio
        '-vf', f"ass={subtitle_path}",  # Subtitle filter
        '-t', str(duration),       # Duration
        '-c:v', 'libx264',         # Video codec
        '-c:a', 'aac',             # Audio codec
        '-pix_fmt', 'yuv420p',     # Pixel format
        '-shortest',               # End when audio ends
        '-y',                      # Overwrite output
        '-loglevel', 'error',      # Only show errors
        output_path
    ]
    
    print(f"Running FFmpeg: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ Video with subtitles created successfully")
            return True
        else:
            print(f"❌ FFmpeg failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg timeout")
        return False
    except Exception as e:
        print(f"❌ FFmpeg error: {e}")
        return False

def generate_gradient_background(width=768, height=768, colors=None):
    """Generate a beautiful background"""
    if colors is None:
        colors = ["#4A90E2", "#50E3C2"]
    
    bg_color = colors[0]
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    # Add visual elements
    for _ in range(5):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(50, 200)
        color = random.choice(colors[1:]) if len(colors) > 1 else colors[0]
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, width=0)
    
    return image

def generate_image_pollinations(text: str, img_path: str):
    """Try Pollinations.ai for image generation"""
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
            print(f"✗ Pollinations.ai failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Pollinations.ai error: {e}")
        return False

def generate_image_placeholder(text: str, img_path: str, category="science"):
    """Generate a beautiful placeholder image"""
    try:
        print("Generating placeholder image...")
        width, height = 768, 768
        colors = CATEGORY_COLORS.get(category, CATEGORY_COLORS["science"])
        
        image = generate_gradient_background(width, height, colors)
        draw = ImageDraw.Draw(image)
        
        # Add a central circle
        center_x, center_y = width // 2, height // 2
        circle_radius = 200
        draw.ellipse([
            center_x - circle_radius, center_y - circle_radius,
            center_x + circle_radius, center_y + circle_radius
        ], outline="white", width=5)
        
        image.save(img_path, "JPEG", quality=85)
        print("✓ Placeholder image generated")
        return True
        
    except Exception as e:
        print(f"✗ Placeholder image failed: {e}")
        return False

def generate_silent_audio(duration: int, audio_path: str):
    """Generate silent audio as last resort"""
    try:
        print(f"Generating silent audio ({duration}s)...")
        result = subprocess.run([
            "ffmpeg",
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", str(duration),
            "-c:a", "libmp3lame",
            audio_path,
            "-y",
            "-loglevel", "error"
        ], capture_output=True, timeout=30)
        
        success = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
        if success:
            print("✓ Silent audio generated")
        return success
        
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
        # Use Groq API to generate facts
        facts = generate_facts_with_groq(category)
        
        if not facts or len(facts) == 0:
            return {"error": "Failed to generate facts with AI"}
        
        return {"facts": facts[:5]}
        
    except Exception as e:
        print(f"Facts generation error: {e}")
        return {"error": f"Failed to fetch facts: {str(e)}"}

@app.get("/generate_video")
async def generate_video(fact: str, category: str = "science", effect: str = "karaoke"):
    try:
        safe_fact = fact.strip()
        if len(safe_fact) > 300:
            safe_fact = safe_fact[:300]

        print(f"🎬 Generating video for: '{safe_fact}' with {effect} effect")

        # Generate image using multiple fallbacks
        print("🖼️  Step 1: Generating image...")
        img_path = f"/tmp/{uuid.uuid4()}.jpg"
        image_generated = False
        
        # Try multiple image sources
        if generate_image_pollinations(safe_fact, img_path):
            image_generated = True
        elif generate_image_placeholder(safe_fact, img_path, category):
            image_generated = True
        
        if not image_generated:
            raise HTTPException(status_code=500, detail="All image generation methods failed")

        # Generate audio with Groq TTS
        print("🔊 Step 2: Generating VOICE-OVER audio with Groq...")
        audio_path = f"/tmp/{uuid.uuid4()}.mp3"
        audio_generated = generate_audio_with_groq(safe_fact, audio_path)
        
        print(f"🎯 Groq TTS result: {audio_generated}")

        # Calculate duration
        duration = 5  # Default duration
        
        if audio_generated:
            try:
                # Get duration from audio file using ffprobe
                result = subprocess.run([
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    duration = min(float(result.stdout.strip()), 15)  # Max 15 seconds
                else:
                    duration = max(len(safe_fact.split()) / 2, 5)
                    
                print(f"⏱️  Audio duration: {duration:.2f} seconds")
            except Exception as e:
                print(f"Error getting audio duration: {e}")
                audio_generated = False
                duration = max(len(safe_fact.split()) / 2, 5)
        else:
            duration = max(len(safe_fact.split()) / 2, 5)
            print(f"⏱️  Estimated duration: {duration} seconds (NO VOICE-OVER)")
            generate_silent_audio(duration, audio_path)

        # Create subtitle with selected effect
        print("📝 Step 3: Creating animated subtitles...")
        subtitle_path = f"/tmp/{uuid.uuid4()}.ass"
        
        if effect == "karaoke":
            create_karaoke_ass_subtitle(safe_fact, duration, subtitle_path)
        elif effect == "fade":
            create_fade_in_ass_subtitle(safe_fact, duration, subtitle_path)
        elif effect == "typewriter":
            create_typewriter_ass_subtitle(safe_fact, duration, subtitle_path)
        elif effect == "bouncing":
            create_bouncing_ass_subtitle(safe_fact, duration, subtitle_path)
        elif effect == "static":
            create_static_ass_subtitle(safe_fact, duration, subtitle_path)
        else:
            create_karaoke_ass_subtitle(safe_fact, duration, subtitle_path)  # Default

        # Create video with FFmpeg and burned-in subtitles
        print("🎥 Step 4: Creating video with FFmpeg subtitles...")
        output_path = f"/tmp/{uuid.uuid4()}.mp4"
        
        video_created = create_video_with_subtitles_ffmpeg(
            img_path, 
            audio_path, 
            subtitle_path, 
            output_path, 
            duration
        )
        
        if not video_created:
            # Fallback: try without subtitles
            print("🔄 Fallback: Trying without subtitles...")
            cmd = [
                'ffmpeg',
                '-loop', '1',
                '-i', img_path,
                '-i', audio_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-pix_fmt', 'yuv420p',
                '-shortest',
                '-y',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            video_created = result.returncode == 0

        if not video_created:
            raise HTTPException(status_code=500, detail="Video creation failed")

        print("✅ Video with ANIMATED SUBTITLES created successfully!")

        # Cleanup temporary files
        for temp_file in [img_path, audio_path, subtitle_path]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

        def iterfile():
            with open(output_path, "rb") as f:
                yield from f
            # Clean up video file after streaming
            try:
                os.unlink(output_path)
            except:
                pass
        
        print(f"🎉 Video with {effect.upper()} SUBTITLES ready for streaming!")
        return StreamingResponse(iterfile(), media_type="video/mp4")

    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        return {"error": f"Video generation failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
