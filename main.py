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

# IMPORTANT: Ensure you set the GROQ_API_KEY environment variable.

app = FastAPI(title="AI Fact Short Video Generator")

# Initialize Groq client
# This requires GROQ_API_KEY environment variable to be set
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"Warning: Groq client initialization failed. Check GROQ_API_KEY. Error: {e}")
    groq_client = None

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

# --- HELPER FUNCTIONS ---

def wrap_text_for_ffmpeg(text: str, max_chars_per_line: int = 30) -> str:
    """
    Splits long text into lines using '\\n' for FFmpeg's drawtext filter
    to prevent horizontal overflow in a short video format.
    """
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        # Check if adding the word exceeds the max length
        if len(current_line) + len(word) + 1 > max_chars_per_line and current_line:
            lines.append(current_line)
            current_line = word
        else:
            # Append word to current line
            current_line = (current_line + " " + word).strip()
            
    lines.append(current_line) # Add the last line
    
    # FFmpeg requires two backslashes for a literal newline
    # Also escape single quotes and colons for FFmpeg
    escaped_text = '\\n'.join(lines).replace("'", "'\\\\\\''").replace(":", "\\\\:")
    
    return escaped_text


def generate_facts_with_groq(category: str):
    """Generate facts using Groq API with llama-3.1-8b-instant"""
    if not groq_client:
        print("Groq client not initialized.")
        return None
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
    if not groq_client:
        print("Groq client not initialized. Cannot generate audio.")
        return False
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

def create_video_with_simple_subtitles(image_path: str, audio_path: str, text: str, output_path: str, duration: float):
    """
    Creates video with simple centered subtitles (static text).
    This function replaces the original complex and failing Karaoke logic.
    """
    try:
        # Text is already escaped and wrapped by the calling function.
        escaped_wrapped_text = text 
        
        # We will use a smaller font size (30) to better accommodate two lines of text.
        # The y position 'h-th-100' dynamically places the text 100 pixels from the bottom.
        
        # Using filter_complex for better control over the video stream (768x768 vertical video)
        filter_complex_str = (
            f"[0:v]scale=768:768:force_original_aspect_ratio=increase,crop=768:768,setsar=1/1[bg];"
            f"[bg]drawtext=text='{escaped_wrapped_text}':"
            f"fontcolor=white:fontsize=30:"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:" # Use a common font path if available, or remove if system default is fine
            f"box=1:boxcolor=black@0.7:boxborderw=10:"
            f"x=(w-text_w)/2:y=(h*0.8)-text_h/2[outv]" # Place text roughly at 80% of height
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
        
        print(f"Running FFmpeg to create simple video...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Simple video with subtitles created successfully")
            return True
        else:
            print(f"❌ FFmpeg video creation failed: {result.stderr}")
            return False
        
    except Exception as e:
        print(f"Simple subtitle video creation failed: {e}")
        return False


# The original create_video_with_subtitles_simple is replaced by the simpler version above
# to fix the karaoke issue, but we keep the name for compatibility.
def create_video_with_subtitles_simple(image_path: str, audio_path: str, text: str, output_path: str, duration: float):
    # Wrap and escape the text first to fix the "text too big" issue
    wrapped_text = wrap_text_for_ffmpeg(text, max_chars_per_line=30)
    
    # Call the simplified function (which was the old 'fallback' but is now the primary)
    return create_video_with_simple_subtitles(image_path, audio_path, wrapped_text, output_path, duration)


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
        # Using 768x768 for square image suitable for short video aspect ratio
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

# --- API ENDPOINTS ---

@app.get("/")
async def home():
    # Assumes an index.html file exists in the same directory
    return FileResponse("index.html")

@app.get("/facts")
async def get_facts(category: str):
    if category not in PROMPTS:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    if not groq_client:
        raise HTTPException(status_code=503, detail="AI service unavailable (GROQ_API_KEY missing/invalid).")

    try:
        # Use Groq API to generate facts
        facts = generate_facts_with_groq(category)
        
        if not facts or len(facts) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate facts with AI")
        
        return {"facts": facts[:5]}
        
    except Exception as e:
        print(f"Facts generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch facts: {str(e)}")

@app.get("/generate_video")
async def generate_video(fact: str, category: str = "science"):
    if not groq_client:
        # Allow running with placeholder image/silent audio if key is missing
        print("Warning: Groq client not initialized. Proceeding with silent audio and placeholder images.")
    
    try:
        safe_fact = fact.strip()
        if len(safe_fact) > 300:
            safe_fact = safe_fact[:300]

        print(f"🎬 Generating video for: '{safe_fact}'")

        # --- Step 1: Generate Image ---
        print("🖼️ Step 1: Generating image...")
        img_path = f"/tmp/{uuid.uuid4()}.jpg"
        
        image_generated = False
        # Try multiple image sources
        if generate_image_pollinations(safe_fact, img_path):
            image_generated = True
        # Always fallback to placeholder
        if not image_generated and generate_image_placeholder(safe_fact, img_path, category):
            image_generated = True
        
        if not image_generated:
            raise HTTPException(status_code=500, detail="All image generation methods failed")

        # --- Step 2: Generate Audio ---
        print("🔊 Step 2: Generating VOICE-OVER audio...")
        audio_path = f"/tmp/{uuid.uuid4()}.mp3"
        audio_generated = False
        duration = 5 # Default duration
        
        if groq_client:
            audio_generated = generate_audio_with_groq(safe_fact, audio_path)
            
            if audio_generated:
                try:
                    # Get duration from audio file using ffprobe
                    result = subprocess.run([
                        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        duration = min(float(result.stdout.strip()), 15) # Max 15 seconds
                    else:
                        duration = max(len(safe_fact.split()) / 2, 5)
                        
                    print(f"⏱️ Audio duration: {duration:.2f} seconds")
                except Exception as e:
                    print(f"Error getting audio duration: {e}. Falling back to silent audio.")
                    audio_generated = False # Force silent fallback if duration fails
        
        # Fallback to silent audio if TTS failed or was skipped
        if not audio_generated:
            duration = max(len(safe_fact.split()) / 2, 5) # Estimate duration
            print(f"⏱️ Estimated duration: {duration:.2f} seconds (NO VOICE-OVER)")
            generate_silent_audio(int(duration), audio_path)


        # --- Step 3: Create Video ---
        print("🎥 Step 3: Creating video with SUBTITLES...")
        output_path = f"/tmp/{uuid.uuid4()}.mp4"
        
        # This call uses the corrected function which includes text wrapping and simpler drawtext
        video_created = create_video_with_subtitles_simple(
            img_path, 
            audio_path, 
            safe_fact, # Pass original fact, wrapping happens inside the function
            output_path, 
            duration
        )
        
        if not video_created:
            raise HTTPException(status_code=500, detail="Video creation failed")

        print("✅ Video with SUBTITLES created successfully!")

        # --- Step 4: Cleanup and Stream ---
        
        # Cleanup temporary files (image and audio)
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
        # Attempt to clean up any files that might have been created before the final video
        for path in [img_path, audio_path, output_path]:
            if 'path' in locals() and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Make sure the /tmp directory exists or is writable
    if not os.path.exists("/tmp"):
        os.makedirs("/tmp")
        
    uvicorn.run(app, host="0.0.0.0", port=8000)
