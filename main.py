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
    """Convert seconds to SRT time format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

def create_simple_srt_subtitle(text: str, duration: float, srt_path: str):
    """Create simple SRT subtitle that displays throughout the video"""
    srt_content = f"""1
{format_time(0)} --> {format_time(duration)}
{text}
"""
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    print(f"✓ SRT subtitle created: {text}")

def create_video_with_subtitles_simple(image_path: str, audio_path: str, text: str, output_path: str, duration: float):
    """Create video with burned-in subtitles using simple FFmpeg approach"""
    
    # Create temporary SRT file
    srt_path = f"/tmp/{uuid.uuid4()}.srt"
    create_simple_srt_subtitle(text, duration, srt_path)
    
    try:
        # Use FFmpeg with drawtext filter (more reliable than subtitles filter)
        cmd = [
            'ffmpeg',
            '-loop', '1',
            '-i', image_path,
            '-i', audio_path,
            '-vf', f"drawtext=text='{text}':fontcolor=white:fontsize=48:box=1:boxcolor=black@0.8:boxborderw=5:x=(w-text_w)/2:y=h-th-100",
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            '-y',
            '-loglevel', 'info',  # Change to info to see errors
            output_path
        ]
        
        print(f"Running FFmpeg: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Clean up SRT file
        try:
            os.unlink(srt_path)
        except:
            pass
            
        if result.returncode == 0:
            print("✓ Video with subtitles created successfully")
            return True
        else:
            print(f"❌ FFmpeg failed: {result.stderr}")
            # Fallback: try without subtitles
            return create_video_without_subtitles(image_path, audio_path, output_path, duration)
        
    except Exception as e:
        print(f"❌ FFmpeg subtitle failed: {e}")
        # Clean up SRT file
        try:
            os.unlink(srt_path)
        except:
            pass
        return create_video_without_subtitles(image_path, audio_path, output_path, duration)

def create_video_without_subtitles(image_path: str, audio_path: str, output_path: str, duration: float):
    """Fallback: Create video without subtitles"""
    try:
        cmd = [
            'ffmpeg',
            '-loop', '1',
            '-i', image_path,
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
        return result.returncode == 0
    except Exception as e:
        print(f"Fallback video creation failed: {e}")
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
async def generate_video(fact: str, category: str = "science"):
    try:
        safe_fact = fact.strip()
        if len(safe_fact) > 300:
            safe_fact = safe_fact[:300]

        print(f"🎬 Generating video for: '{safe_fact}'")

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

        # Create video with FFmpeg and burned-in subtitles
        print("🎥 Step 3: Creating video with SUBTITLES...")
        output_path = f"/tmp/{uuid.uuid4()}.mp4"
        
        video_created = create_video_with_subtitles_simple(
            img_path, 
            audio_path, 
            safe_fact, 
            output_path, 
            duration
        )
        
        if not video_created:
            raise HTTPException(status_code=500, detail="Video creation failed")

        print("✅ Video with SUBTITLES created successfully!")

        # Cleanup temporary files
        for temp_file in [img_path, audio_path]:
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
        
        print("🎉 Video with SUBTITLES ready for streaming!")
        return StreamingResponse(iterfile(), media_type="video/mp4")

    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        return {"error": f"Video generation failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
