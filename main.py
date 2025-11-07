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

def generate_audio_pyttsx3(text: str, audio_path: str):
    """Use pyttsx3 - pure Python TTS that works on Render"""
    try:
        print("Trying pyttsx3 TTS...")
        import pyttsx3
        
        # Initialize the TTS engine
        engine = pyttsx3.init()
        
        # Configure voice settings for better quality
        engine.setProperty('rate', 180)    # Speaking speed
        engine.setProperty('volume', 0.9)  # Volume level
        engine.setProperty('voice', 'english')  # Force English voice
        
        # Get available voices for debugging
        voices = engine.getProperty('voices')
        print(f"Available voices: {[v.id for v in voices]}")
        
        # Try to use a female voice if available
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        
        # Save to file
        engine.save_to_file(text, audio_path)
        engine.runAndWait()
        
        # Check if file was created successfully
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            print(f"Audio file created: {audio_path} ({file_size} bytes)")
            return file_size > 1000  # File should be at least 1KB
        
        print("Audio file was not created")
        return False
        
    except Exception as e:
        print(f"pyttsx3 failed: {e}")
        return False

def generate_audio_gtts(text: str, audio_path: str):
    """Use gTTS as backup"""
    try:
        print("Trying gTTS...")
        from gtts import gTTS
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(audio_path)
        
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            print(f"gTTS audio created: {file_size} bytes")
            return file_size > 1000
        return False
        
    except Exception as e:
        print(f"gTTS failed: {e}")
        return False

def generate_silent_audio(duration: int, audio_path: str):
    """Generate silent audio as last resort"""
    try:
        print(f"Generating silent audio for {duration} seconds...")
        # Use ffmpeg to generate silent audio
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
        print(f"Silent audio generated: {success}")
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
async def generate_video(fact: str):
    try:
        safe_fact = fact.strip()
        if len(safe_fact) > 300:
            safe_fact = safe_fact[:300]

        print(f"Generating video for fact: {safe_fact}")

        # Generate image from Pollinations
        print("Step 1: Generating image...")
        img_url = f"https://pollinations.ai/p/{urllib.parse.quote(safe_fact)}?width=768&height=768&nologo=true"
        img_response = requests.get(img_url, timeout=30)
        if img_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to generate image")
        
        img_path = f"/tmp/{uuid.uuid4()}.jpg"
        with open(img_path, "wb") as f:
            f.write(img_response.content)
        print("✓ Image generated")

        # Generate audio
        print("Step 2: Generating audio...")
        audio_path = f"/tmp/{uuid.uuid4()}.mp3"
        audio_generated = False
        audio_method = "None"
        
        # Try pyttsx3 first
        if generate_audio_pyttsx3(safe_fact, audio_path):
            audio_generated = True
            audio_method = "pyttsx3"
        # Then try gTTS
        elif generate_audio_gtts(safe_fact, audio_path):
            audio_generated = True
            audio_method = "gTTS"
        
        print(f"Audio generation: {audio_generated} (method: {audio_method})")

        # Calculate duration
        if audio_generated:
            try:
                audio_clip = AudioFileClip(audio_path)
                duration = min(audio_clip.duration, 10)
                print(f"Audio duration: {duration:.2f} seconds")
            except Exception as e:
                print(f"Error loading audio: {e}")
                audio_generated = False
                duration = 5
        else:
            # Estimate duration based on text length
            duration = max(len(safe_fact.split()) / 2, 3)
            print(f"Estimated duration: {duration} seconds (no audio)")
            
            # Generate silent audio so video has an audio track
            generate_silent_audio(duration, audio_path)

        # Create video
        print("Step 3: Creating video...")
        image_clip = ImageClip(img_path).set_duration(duration)
        
        # Create text clip
        txt_clip = TextClip(
            safe_fact,
            fontsize=28,
            font="Arial-Bold",
            color="white",
            size=(image_clip.w * 0.85, None),
            method="caption",
            align="center",
            stroke_color="black",
            stroke_width=2
        ).set_position('center').set_duration(duration)

        # Load audio if available
        audio_clip = None
        if audio_generated and os.path.exists(audio_path):
            try:
                audio_clip = AudioFileClip(audio_path).set_duration(duration)
            except Exception as e:
                print(f"Error loading audio clip: {e}")
                audio_generated = False

        # Combine everything
        if audio_generated and audio_clip:
            final_video = CompositeVideoClip([image_clip, txt_clip]).set_audio(audio_clip)
            print("✓ Video with audio created")
        else:
            final_video = CompositeVideoClip([image_clip, txt_clip])
            print("✓ Video without audio created")

        # Export video
        output_path = f"/tmp/{uuid.uuid4()}.mp4"
        print("Step 4: Exporting video...")
        
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
        print("✓ Video exported")

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
        
        print("✓ Video streaming ready")
        return StreamingResponse(iterfile(), media_type="video/mp4")

    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        return {"error": f"Video generation failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
