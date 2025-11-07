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
import json
import math

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

def estimate_word_timings(text: str, total_duration: float):
    """Estimate when each word should be highlighted (simple version)"""
    words = text.split()
    word_count = len(words)
    
    # Simple estimation: each word gets equal time
    word_duration = total_duration / word_count
    
    timings = []
    current_time = 0
    
    for i, word in enumerate(words):
        # Start time for this word
        start = current_time
        # End time for this word (slightly overlap with next word)
        end = start + word_duration * 1.1
        
        timings.append({
            'word': word,
            'start': start,
            'end': end,
            'index': i
        })
        
        current_time += word_duration
    
    return timings

def create_karaoke_text_clip(text: str, duration: float, screen_size, word_timings):
    """Create a text clip with karaoke-style word highlighting"""
    
    def make_frame(t):
        """Create a frame with highlighted words up to current time"""
        # Create a PIL image for this frame
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        width, height = screen_size
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Try to use a font (fallback to default)
        try:
            font = ImageFont.truetype("Arial Bold", 36)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            except:
                font = ImageFont.load_default()
        
        # Split text into words
        words = text.split()
        
        # Calculate text positioning
        padding = 50
        max_width = width - 2 * padding
        line_height = 50
        current_line = []
        current_line_width = 0
        lines = []
        
        # Break text into lines
        for word in words:
            # Estimate word width (rough approximation)
            word_width = len(word) * 20
            
            if current_line_width + word_width > max_width and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_line_width = word_width
            else:
                current_line.append(word)
                current_line_width += word_width + 10  # +10 for space
        
        if current_line:
            lines.append(" ".join(current_line))
        
        # Calculate starting y position to center text
        total_text_height = len(lines) * line_height
        start_y = (height - total_text_height) // 2
        
        # Draw each line
        for line_idx, line in enumerate(lines):
            line_y = start_y + line_idx * line_height
            
            # Split line into words for highlighting
            line_words = line.split()
            current_x = padding
            
            for word_idx, word in enumerate(line_words):
                # Check if this word should be highlighted
                should_highlight = False
                for timing in word_timings:
                    if timing['word'] == word and timing['start'] <= t <= timing['end']:
                        should_highlight = True
                        break
                
                # Choose color
                color = "#FFD700" if should_highlight else "#FFFFFF"  # Gold for highlighted, white for others
                
                # Draw word
                draw.text((current_x, line_y), word, fill=color, font=font)
                
                # Move x position for next word
                current_x += len(word) * 20 + 10  # Rough approximation
            
        return np.array(img)
    
    # Create VideoClip from the frame function
    from moviepy.video.VideoClip import VideoClip
    clip = VideoClip(make_frame, duration=duration)
    return clip

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

        # Calculate duration and estimate word timings
        duration = 5  # Default duration
        word_timings = []
        
        if audio_generated:
            try:
                audio_clip = AudioFileClip(audio_path)
                duration = min(audio_clip.duration, 15)  # Max 15 seconds
                print(f"⏱️  Audio duration: {duration:.2f} seconds")
                
                # Estimate word timings for karaoke effect
                word_timings = estimate_word_timings(safe_fact, duration)
                print(f"📝 Word timings estimated for {len(word_timings)} words")
                
            except Exception as e:
                print(f"Error loading audio: {e}")
                audio_generated = False
                duration = max(len(safe_fact.split()) / 2, 5)
        else:
            duration = max(len(safe_fact.split()) / 2, 5)
            print(f"⏱️  Estimated duration: {duration} seconds (NO VOICE-OVER)")
            generate_silent_audio(duration, audio_path)

        # Create video with karaoke text
        print("🎥 Step 3: Creating video with KARAOKE TEXT and audio...")
        image_clip = ImageClip(img_path).set_duration(duration)
        
        # Create karaoke-style text clip
        screen_size = (image_clip.w, image_clip.h)
        karaoke_text_clip = create_karaoke_text_clip(safe_fact, duration, screen_size, word_timings)

        # Load audio for voice-over
        audio_clip = None
        if audio_generated and os.path.exists(audio_path):
            try:
                audio_clip = AudioFileClip(audio_path).set_duration(duration)
                print("✅ Groq TTS audio loaded successfully for voice-over")
            except Exception as e:
                print(f"Error loading audio clip: {e}")
                audio_generated = False

        # Combine everything
        if audio_generated and audio_clip:
            final_video = CompositeVideoClip([image_clip, karaoke_text_clip]).set_audio(audio_clip)
            print("✅ Video created WITH KARAOKE TEXT & GROQ VOICE-OVER")
        else:
            final_video = CompositeVideoClip([image_clip, karaoke_text_clip])
            print("⚠️  Video created WITHOUT voice-over (silent)")

        # Export video
        output_path = f"/tmp/{uuid.uuid4()}.mp4"
        print("💾 Step 4: Exporting video...")
        
        final_video.write_videofile(
            output_path,
            fps=24,  # Higher FPS for smoother text animation
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
        
        print("🎉 Video with KARAOKE TEXT & Groq voice-over ready for streaming!")
        return StreamingResponse(iterfile(), media_type="video/mp4")

    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        return {"error": f"Video generation failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
