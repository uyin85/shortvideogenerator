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

app = FastAPI(title="AI Fact Short Video Generator")

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
    "science": "Give me 5 short surprising science facts. One sentence each, under 15 words.",
    "successful_person": "Give me 5 short inspiring facts about successful people. One sentence each, under 15 words.",
    "unsolved_mystery": "Give me 5 short unsolved mysteries. One sentence each, under 15 words.",
    "history": "Give me 5 short memorable history facts. One sentence each, under 15 words.",
    "sports": "Give me 5 short legendary sports facts. One sentence each, under 15 words."
}

# Background colors
CATEGORY_COLORS = {
    "science": ["#4A90E2", "#50E3C2", "#9013FE", "#417505"],
    "successful_person": ["#F5A623", "#BD10E0", "#7ED321", "#B8E986"],
    "unsolved_mystery": ["#8B572A", "#4A4A4A", "#9B9B9B", "#D0021B"],
    "history": ["#8B4513", "#CD853F", "#D2691E", "#A0522D"],
    "sports": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
}

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

def generate_audio_gtts_reliable(text: str, audio_path: str):
    """Use gTTS with multiple retries and better error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to generate audio with gTTS...")
            from gtts import gTTS
            
            # Create gTTS with specific parameters
            tts = gTTS(
                text=text,
                lang='en',
                slow=False,
                lang_check=False
            )
            
            # Save the audio file
            tts.save(audio_path)
            
            # Verify the file was created and has content
            if os.path.exists(audio_path):
                file_size = os.path.getsize(audio_path)
                print(f"✓ gTTS audio generated: {file_size} bytes")
                
                if file_size > 1000:  # File should be at least 1KB
                    return True
                else:
                    print("File too small, retrying...")
            else:
                print("Audio file not created, retrying...")
                
            # Wait before retry
            time.sleep(2)
            
        except Exception as e:
            print(f"gTTS attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            continue
    
    print("❌ All gTTS attempts failed")
    return False

def generate_audio_elevenlabs_free(text: str, audio_path: str):
    """Try ElevenLabs free tier (no API key required for some voices)"""
    try:
        print("Trying ElevenLabs free TTS...")
        
        # ElevenLabs has some free voices available without API key
        url = "https://api.elevenlabs.io/v1/text-to-speech/piTKgcLEGmPE4e6mEKli"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            with open(audio_path, "wb") as f:
                f.write(response.content)
            
            file_size = os.path.getsize(audio_path)
            print(f"✓ ElevenLabs audio generated: {file_size} bytes")
            return file_size > 1000
        else:
            print(f"ElevenLabs failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ElevenLabs error: {e}")
        return False

def generate_audio_voicerss(text: str, audio_path: str):
    """Try VoiceRSS free TTS service"""
    try:
        print("Trying VoiceRSS TTS...")
        
        # VoiceRSS has a free tier (need to register for API key, but there's a demo key)
        api_key = "YOUR_VOICERSS_API_KEY"  # Get free from voicerss.org
        if api_key == "YOUR_VOICERSS_API_KEY":
            # Use demo mode (limited)
            api_key = "demo"
        
        url = "http://api.voicerss.org/"
        params = {
            "key": api_key,
            "hl": "en-us",
            "src": text,
            "c": "MP3",
            "f": "44khz_16bit_stereo"
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200 and len(response.content) > 1000:
            with open(audio_path, "wb") as f:
                f.write(response.content)
            print("✓ VoiceRSS audio generated")
            return True
        else:
            print("VoiceRSS failed or returned empty audio")
            return False
            
    except Exception as e:
        print(f"VoiceRSS error: {e}")
        return False

def generate_audio_pyttsx3_fallback(text: str, audio_path: str):
    """Try pyttsx3 as final fallback"""
    try:
        print("Trying pyttsx3 fallback...")
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Try to configure the engine
        try:
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 0.9)
        except:
            pass  # Some properties might not be available
        
        engine.save_to_file(text, audio_path)
        engine.runAndWait()
        
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
            print("✓ pyttsx3 audio generated")
            return True
        return False
        
    except Exception as e:
        print(f"pyttsx3 failed: {e}")
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

        # Generate audio with VOICE-OVER
        print("🔊 Step 2: Generating VOICE-OVER audio...")
        audio_path = f"/tmp/{uuid.uuid4()}.mp3"
        audio_generated = False
        audio_method = "None"
        
        # Try multiple TTS services in order of quality
        audio_methods = [
            ("ElevenLabs", lambda: generate_audio_elevenlabs_free(safe_fact, audio_path)),
            ("gTTS", lambda: generate_audio_gtts_reliable(safe_fact, audio_path)),
            ("VoiceRSS", lambda: generate_audio_voicerss(safe_fact, audio_path)),
            ("pyttsx3", lambda: generate_audio_pyttsx3_fallback(safe_fact, audio_path))
        ]
        
        for method_name, method_func in audio_methods:
            if not audio_generated:
                print(f"  🔊 Trying {method_name}...")
                if method_func():
                    audio_generated = True
                    audio_method = method_name
                    print(f"  ✅ {method_name} SUCCESS - Voice-over will be in video!")
                    break
                else:
                    print(f"  ❌ {method_name} failed")
        
        print(f"🎯 Audio generation result: {audio_generated} (method: {audio_method})")

        # Calculate duration
        if audio_generated:
            try:
                audio_clip = AudioFileClip(audio_path)
                duration = min(audio_clip.duration, 15)  # Max 15 seconds
                print(f"⏱️  Audio duration: {duration:.2f} seconds")
            except Exception as e:
                print(f"Error loading audio: {e}")
                audio_generated = False
                duration = max(len(safe_fact.split()) / 2, 5)
        else:
            duration = max(len(safe_fact.split()) / 2, 5)
            print(f"⏱️  Estimated duration: {duration} seconds (NO VOICE-OVER)")
            generate_silent_audio(duration, audio_path)

        # Create video
        print("🎥 Step 3: Creating video with text and audio...")
        image_clip = ImageClip(img_path).set_duration(duration)
        
        # Create text caption (the fact text)
        txt_clip = TextClip(
            safe_fact,
            fontsize=32,
            font="Arial-Bold",
            color="white",
            size=(image_clip.w * 0.8, None),
            method="caption",
            align="center",
            stroke_color="black",
            stroke_width=3
        ).set_position('center').set_duration(duration)

        # Load audio for voice-over
        audio_clip = None
        if audio_generated and os.path.exists(audio_path):
            try:
                audio_clip = AudioFileClip(audio_path).set_duration(duration)
                print("✅ Audio loaded successfully for voice-over")
            except Exception as e:
                print(f"Error loading audio clip: {e}")
                audio_generated = False

        # Combine video with VOICE-OVER audio
        if audio_generated and audio_clip:
            final_video = CompositeVideoClip([image_clip, txt_clip]).set_audio(audio_clip)
            print("✅ Video created WITH VOICE-OVER")
        else:
            final_video = CompositeVideoClip([image_clip, txt_clip])
            print("⚠️  Video created WITHOUT voice-over (silent)")

        # Export video
        output_path = f"/tmp/{uuid.uuid4()}.mp4"
        print("💾 Step 4: Exporting video...")
        
        final_video.write_videofile(
            output_path,
            fps=12,
            codec="libx264",
            audio_codec="aac" if audio_generated else None,
            remove_temp=True,
            logger=None,
            verbose=False,
            ffmpeg_params=['-preset', 'fast', '-crf', '28']
        )
        print("✅ Video exported successfully")

        # Cleanup
        for temp_file in [img_path, audio_path]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

        def iterfile():
            with open(output_path, "rb") as f:
                yield from f
            try:
                os.unlink(output_path)
            except:
                pass
        
        print("🎉 Video with voice-over ready for streaming!")
        return StreamingResponse(iterfile(), media_type="video/mp4")

    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        return {"error": f"Video generation failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
