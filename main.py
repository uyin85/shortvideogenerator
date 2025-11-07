from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import requests
import urllib.parse
import uuid
import os
import subprocess
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip
import aiofiles

app = FastAPI(title="AI Fact Short Video Generator")

PROMPTS = {
    "science": "Give me 5 short surprising science facts. One sentence each, under 15 words.",
    "successful_person": "Give me 5 short inspiring facts about successful people. One sentence each, under 15 words.",
    "unsolved_mystery": "Give me 5 short unsolved mysteries. One sentence each, under 15 words.",
    "history": "Give me 5 short memorable history facts. One sentence each, under 15 words.",
    "sports": "Give me 5 short legendary sports facts. One sentence each, under 15 words."
}

async def generate_audio_espeak(text: str, audio_path: str):
    """Use eSpeak TTS - completely free and works on Render"""
    try:
        # eSpeak is pre-installed on Render's Ubuntu environment
        # First create a WAV file
        wav_path = f"/tmp/{uuid.uuid4()}.wav"
        
        # Use eSpeak to generate WAV audio
        result = subprocess.run([
            "espeak", 
            "-w", wav_path,      # Output WAV file
            "-s", "150",         # Speed (words per minute)
            "-p", "50",          # Pitch
            "-a", "200",         # Amplitude
            "-v", "en+f3",       # Voice (English female 3)
            text
        ], capture_output=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(wav_path):
            # Convert WAV to MP3 using ffmpeg
            subprocess.run([
                "ffmpeg", 
                "-i", wav_path, 
                "-codec:a", "libmp3lame", 
                "-qscale:a", "4",  # Good quality
                audio_path,
                "-y"  # Overwrite output
            ], capture_output=True, timeout=30)
            
            # Clean up WAV file
            try:
                os.unlink(wav_path)
            except:
                pass
            
            return os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000
        return False
    except Exception as e:
        print(f"eSpeak TTS failed: {e}")
        return False

async def generate_audio_festival(text: str, audio_path: str):
    """Use Festival TTS - free alternative"""
    try:
        # Create temporary text file
        text_path = f"/tmp/{uuid.uuid4()}.txt"
        with open(text_path, "w") as f:
            f.write(text)
        
        wav_path = f"/tmp/{uuid.uuid4()}.wav"
        
        # Use Festival to generate speech
        result = subprocess.run([
            "text2wave", 
            text_path, 
            "-o", wav_path,
            "-scale", "2.0"  # Increase volume
        ], capture_output=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(wav_path):
            # Convert to MP3
            subprocess.run([
                "ffmpeg", 
                "-i", wav_path, 
                "-codec:a", "libmp3lame", 
                audio_path,
                "-y"
            ], capture_output=True, timeout=30)
            
            # Clean up
            for temp_file in [text_path, wav_path]:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            return os.path.exists(audio_path)
        return False
    except Exception as e:
        print(f"Festival TTS failed: {e}")
        return False

async def generate_silent_audio(duration: int, audio_path: str):
    """Generate silent audio as fallback"""
    try:
        subprocess.run([
            "ffmpeg", 
            "-f", "lavfi", 
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", str(duration), 
            "-c:a", "libmp3lame", 
            audio_path, 
            "-y"
        ], capture_output=True, timeout=30)
        return os.path.exists(audio_path)
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
        if len(safe_fact) > 300:  # Shorter for TTS
            safe_fact = safe_fact[:300]

        # Generate image from Pollinations
        img_url = f"https://pollinations.ai/p/{urllib.parse.quote(safe_fact)}?width=768&height=768&nologo=true"
        img_response = requests.get(img_url, timeout=30)
        if img_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to generate image")
        
        img_path = f"/tmp/{uuid.uuid4()}.jpg"
        async with aiofiles.open(img_path, "wb") as f:
            await f.write(img_response.content)

        # Generate audio using FREE TTS methods
        audio_path = f"/tmp/{uuid.uuid4()}.mp3"
        audio_generated = False
        
        # Try eSpeak first (most reliable free option)
        audio_generated = await generate_audio_espeak(safe_fact, audio_path)
        
        # Try Festival as backup
        if not audio_generated:
            audio_generated = await generate_audio_festival(safe_fact, audio_path)
        
        # Calculate duration based on text length
        estimated_duration = max(len(safe_fact) / 12, 3)  # Words per second
        
        # If no TTS worked, create silent audio
        if not audio_generated:
            silent_generated = await generate_silent_audio(estimated_duration, audio_path)
            if silent_generated:
                audio_clip = AudioFileClip(audio_path)
                duration = audio_clip.duration
            else:
                duration = estimated_duration
                audio_clip = None
        else:
            audio_clip = AudioFileClip(audio_path)
            duration = min(audio_clip.duration, 15)

        # Create video
        image_clip = ImageClip(img_path).set_duration(duration)
        
        # Create text clip with better formatting
        txt_clip = TextClip(
            safe_fact,
            fontsize=24,
            font="Arial-Bold",
            color="white",
            size=(image_clip.w * 0.85, None),
            method="caption",
            align="center",
            stroke_color="black",
            stroke_width=2
        ).set_position('center').set_duration(duration)

        if audio_clip:
            final_video = CompositeVideoClip([image_clip, txt_clip]).set_audio(audio_clip)
        else:
            final_video = CompositeVideoClip([image_clip, txt_clip])

        output_path = f"/tmp/{uuid.uuid4()}.mp4"
        
        # Write video with optimized settings
        final_video.write_videofile(
            output_path,
            fps=10,  # Lower FPS for faster rendering
            codec="libx264",
            audio_codec="aac" if audio_clip else None,
            remove_temp=True,
            logger=None,
            verbose=False,
            ffmpeg_params=[
                '-preset', 'fast',
                '-crf', '28'  # Higher CRF = smaller file
            ]
        )

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
        
        return StreamingResponse(iterfile(), media_type="video/mp4")

    except Exception as e:
        return {"error": f"Video generation failed: {str(e)}"}
