from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import requests
import urllib.parse
import uuid
import os
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip
import tempfile
import asyncio
import aiofiles

app = FastAPI(title="AI Fact Short Video Generator")

PROMPTS = {
    "science": "Give me 5 short surprising science facts. One sentence each, under 15 words.",
    "successful_person": "Give me 5 short inspiring facts about successful people. One sentence each, under 15 words.",
    "unsolved_mystery": "Give me 5 short unsolved mysteries. One sentence each, under 15 words.",
    "history": "Give me 5 short memorable history facts. One sentence each, under 15 words.",
    "sports": "Give me 5 short legendary sports facts. One sentence each, under 15 words."
}

# Alternative TTS service using ElevenLabs (free tier available)
async def generate_audio_elevenlabs(text: str, output_path: str):
    """Use ElevenLabs TTS API"""
    # You'll need to sign up for ElevenLabs and get an API key
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "your-api-key-here")
    
    if ELEVENLABS_API_KEY == "your-api-key-here":
        # Fallback to system TTS or other service
        return await generate_audio_system(text, output_path)
    
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
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
            "similarity_boost": 0.5
        }
    }
    
    response = requests.post(url, json=data, headers=headers, timeout=30)
    if response.status_code == 200:
        async with aiofiles.open(output_path, "wb") as f:
            await f.write(response.content)
        return True
    else:
        return False

async def generate_audio_system(text: str, output_path: str):
    """Use system TTS as fallback (requires festival on Linux)"""
    try:
        # This works on Render's Linux environment
        import subprocess
        # Create a temporary text file
        text_path = f"/tmp/{uuid.uuid4()}.txt"
        async with aiofiles.open(text_path, "w") as f:
            await f.write(text)
        
        # Use festival TTS (available on Ubuntu)
        subprocess.run([
            "text2wave", text_path, "-o", output_path,
            "-scale", "2.0", "-F", "24000"
        ], check=True, timeout=30)
        
        # If festival is not available, try espeak
        if not os.path.exists(output_path):
            subprocess.run([
                "espeak", "-w", output_path, "-s", "150", text
            ], check=True, timeout=30)
            
        return os.path.exists(output_path)
    except Exception:
        return False

async def generate_audio_polly(text: str, output_path: str):
    """AWS Polly TTS (free tier available)"""
    try:
        # This requires boto3 and AWS credentials
        import boto3
        polly = boto3.client('polly',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name='us-east-1'
        )
        
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId='Joanna'
        )
        
        if "AudioStream" in response:
            async with aiofiles.open(output_path, "wb") as f:
                audio_data = response['AudioStream'].read()
                await f.write(audio_data)
            return True
    except Exception:
        pass
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
        if len(safe_fact) > 500:  # Limit fact length
            safe_fact = safe_fact[:500] + "..."

        # Generate image from Pollinations
        img_url = f"https://pollinations.ai/p/{urllib.parse.quote(safe_fact)}?width=768&height=768&nologo=true"
        img_response = requests.get(img_url, timeout=30)
        if img_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to generate image")
        
        img_path = f"/tmp/{uuid.uuid4()}.jpg"
        async with aiofiles.open(img_path, "wb") as f:
            await f.write(img_response.content)

        # Generate audio using fallback method
        audio_path = f"/tmp/{uuid.uuid4()}.mp3"
        
        # Try multiple TTS services in order of preference
        audio_generated = False
        
        # Method 1: Try ElevenLabs first
        if not audio_generated:
            audio_generated = await generate_audio_elevenlabs(safe_fact, audio_path)
        
        # Method 2: Try AWS Polly
        if not audio_generated and os.getenv('AWS_ACCESS_KEY_ID'):
            audio_generated = await generate_audio_polly(safe_fact, audio_path)
        
        # Method 3: Try system TTS
        if not audio_generated:
            audio_generated = await generate_audio_system(safe_fact, audio_path)
        
        if not audio_generated:
            raise HTTPException(status_code=500, detail="All TTS services failed")

        # Build video
        audio_clip = AudioFileClip(audio_path)
        duration = min(audio_clip.duration, 20)

        image_clip = ImageClip(img_path).set_duration(duration)
        txt_clip = TextClip(
            safe_fact,
            fontsize=40,
            font="Arial-Bold",
            color="white",
            size=image_clip.size,
            method="caption",
            align="center",
            stroke_color="black",
            stroke_width=1.5
        ).set_duration(duration)

        final_video = CompositeVideoClip([image_clip, txt_clip]).set_audio(audio_clip.subclip(0, duration))
        output_path = f"/tmp/{uuid.uuid4()}.mp4"
        
        # Use lower quality for faster rendering
        final_video.write_videofile(
            output_path,
            fps=15,  # Reduced from 24
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=f"/tmp/{uuid.uuid4()}.m4a",
            remove_temp=True,
            logger=None,
            threads=2,  # Reduced threads
            bitrate="1000k"  # Lower bitrate
        )

        # Clean up temporary files
        for temp_file in [img_path, audio_path]:
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
        
        return StreamingResponse(iterfile(), media_type="video/mp4")

    except Exception as e:
        return {"error": f"Video generation failed: {str(e)}"}
