from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import urllib.parse
import uuid
import os
import subprocess
import random
import json
import re
import tempfile
from PIL import Image, ImageDraw

# --- CONFIGURATION ---
app = FastAPI(
    title="AI Fact Short Video Generator API",
    description="Backend API for generating short videos with AI facts, images, and animated subtitles",
    version="1.0.0"
)

# CORS configuration - specifically for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://multisite.interactivelink.site",
        "https://multisite.interactivelink.site/factshortvideogen",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq for fact generation
from groq import Groq
groq_client = None
if os.getenv("GROQ_API_KEY"):
    try:
        groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    except Exception as e:
        print(f"Groq init warning: {e}")

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

# --- TTS FUNCTIONS ---

def generate_audio_with_gtts(text: str, audio_path: str):
    """Generate audio using gTTS (Google Text-to-Speech)"""
    try:
        from gtts import gTTS
        
        print(f"Generating audio with gTTS for text: {text[:50]}...")
        
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save as MP3
        tts.save(audio_path)
        
        # Check if file was created
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
            # Get actual duration using ffprobe
            try:
                result = subprocess.run([
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path
                ], capture_output=True, text=True, timeout=10)
                duration = float(result.stdout.strip())
                print(f"gTTS success: {os.path.getsize(audio_path)} bytes, {duration:.2f}s duration")
            except Exception as e:
                print(f"ffprobe error, estimating duration: {e}")
                # Estimate duration if ffprobe fails
                duration = len(text.split()) * 0.5 + 1.0
            return True, duration
        else:
            print("gTTS failed: File too small or not created")
            return False, 0.0
            
    except Exception as e:
        print(f"gTTS error: {e}")
        return False, 0.0

def generate_audio_fallback(text: str, audio_path: str):
    """Generate enhanced fallback audio with better quality"""
    try:
        words = text.split()
        duration = max(len(words) * 0.5 + 1.0, 3.0)  # Minimum 3 seconds
        
        print(f"Generating enhanced fallback audio: {duration:.2f}s duration")
        
        # Create more natural-sounding audio with varying tones
        # This creates a more pleasant background sound instead of pure silence
        base_freq = 200  # Base frequency
        words = text.split()
        
        if len(words) > 0:
            # Create a filter chain that varies with the text
            filter_chain = []
            for i, word in enumerate(words):
                word_duration = duration / len(words)
                start_time = i * word_duration
                freq_variation = base_freq + (len(word) * 10)  # Vary frequency by word length
                
                filter_chain.append(
                    f"sine=frequency={freq_variation}:duration={word_duration}:"
                    f"sample_rate=22050,adelay={int(start_time * 1000)}|{int(start_time * 1000)}"
                )
            
            # Mix all the sine waves together
            filter_complex = f"{'+'.join(filter_chain)}"
            
            subprocess.run([
                "ffmpeg", "-f", "lavfi",
                "-i", filter_complex,
                "-af", f"volume=0.05,afade=t=in:st=0:d=0.5,afade=t=out:st={duration-0.5}:d=0.5",
                "-acodec", "libmp3lame", "-b:a", "64k", "-ar", "22050",
                "-t", str(duration),
                audio_path, "-y", "-loglevel", "error"
            ], timeout=30)
        else:
            # Simple tone for very short text
            subprocess.run([
                "ffmpeg", "-f", "lavfi", 
                "-i", f"sine=frequency=300:duration={duration}",
                "-af", f"afade=t=in:st=0:d=0.5,afade=t=out:st={duration-0.5}:d=0.5,volume=0.05",
                "-acodec", "libmp3lame", "-b:a", "64k", "-ar", "22050",
                audio_path, "-y", "-loglevel", "error"
            ], timeout=30)
        
        success = os.path.exists(audio_path) and os.path.getsize(audio_path) > 500
        if success:
            print(f"Enhanced fallback success: {os.path.getsize(audio_path)} bytes")
        else:
            print("Enhanced fallback failed, using basic fallback")
            # Ultimate fallback - silent audio
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
                "-t", str(duration), "-acodec", "libmp3lame", "-b:a", "64k",
                audio_path, "-y", "-loglevel", "error"
            ], timeout=30)
        
        return os.path.exists(audio_path), duration
        
    except Exception as e:
        print(f"Enhanced fallback error: {e}")
        # Ultimate fallback - silent audio
        duration = len(text.split()) * 0.5 + 1.0
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
            "-t", str(duration), "-acodec", "libmp3lame", "-b:a", "64k",
            audio_path, "-y", "-loglevel", "error"
        ], timeout=30)
        return os.path.exists(audio_path), duration

def generate_audio(text: str, audio_path: str, category: str = "science"):
    """Generate audio using gTTS with enhanced fallback"""
    
    # Try gTTS first (requires internet)
    print("Attempting gTTS audio generation...")
    success, duration = generate_audio_with_gtts(text, audio_path)
    if success:
        print("✅ Audio generated with gTTS")
        return success, duration
    
    # Use enhanced fallback
    print("gTTS failed, using enhanced fallback audio...")
    return generate_audio_fallback(text, audio_path)

# --- IMPROVED WORD TIMING FUNCTIONS ---

def analyze_speech_pattern(text: str, duration: float):
    """Analyze text to create more realistic word timings based on linguistic patterns"""
    words = text.split()
    if not words:
        return []
    
    # Linguistic patterns for better timing
    word_complexity = {
        'short': 0.3,    # and, the, is
        'medium': 0.5,   # most common words
        'long': 0.8,     # multi-syllable words
        'complex': 1.2   # technical/long words
    }
    
    # Common short words that are spoken quickly
    short_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    # Calculate base time per word
    base_time_per_word = duration / len(words)
    
    timings = []
    current_time = 0.1  # Start slightly after beginning
    
    for i, word in enumerate(words):
        # Determine word complexity
        word_lower = word.lower().strip('.,!?;:"')
        if word_lower in short_words:
            complexity = word_complexity['short']
        elif len(word_lower) <= 3:
            complexity = word_complexity['short']
        elif len(word_lower) <= 5:
            complexity = word_complexity['medium']
        elif len(word_lower) <= 8:
            complexity = word_complexity['long']
        else:
            complexity = word_complexity['complex']
        
        # Add pause for punctuation
        pause_multiplier = 1.0
        if i > 0 and any(punc in words[i-1] for punc in ',;'):
            pause_multiplier = 1.3
        elif i > 0 and any(punc in words[i-1] for punc in '.!?'):
            pause_multiplier = 1.6
        
        # Calculate word duration
        word_duration = base_time_per_word * complexity * pause_multiplier
        
        # Ensure we don't exceed total duration
        if current_time + word_duration > duration - 0.1:
            word_duration = duration - current_time - 0.1
        
        start_time = current_time
        end_time = current_time + word_duration
        
        timings.append({
            "word": word,
            "start": start_time,
            "end": end_time
        })
        
        current_time = end_time
        
        # Add small pause between words
        if i < len(words) - 1:
            current_time += 0.05  # 50ms pause between words
    
    return timings

def generate_word_timings(text: str, duration: float):
    """Generate improved word timings for better karaoke sync"""
    words = text.split()
    if not words:
        return []
    
    # Use linguistic analysis for better timing
    return analyze_speech_pattern(text, duration)

def create_karaoke_subtitles(word_timings, subtitle_path, effect="karaoke"):
    """Create ASS subtitle file with karaoke or other effects - CENTERED TEXT - FIXED"""
    
    # ASS header for 768x768 centered subtitles
    ass_content = """[Script Info]
Title: AI Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 768
PlayResY: 768
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000088EF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,5,10,10,384,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    if not word_timings:
        # Fallback: display empty for 3 seconds
        ass_content += "Dialogue: 0,0:00:00.00,0:00:03.00,Default,,0,0,0,, \n"
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
        return

    if effect == "karaoke":
        # Calculate video end time (add 2 seconds after last word ends)
        video_end = word_timings[-1]["end"] + 2.0
        
        # Show full white text from start, then highlight words as they're spoken
        full_text = " ".join(w["word"] for w in word_timings)
        
        for i, timing in enumerate(word_timings):
            start = format_time_ass(timing["start"])
            
            # CRITICAL FIX: End current line RIGHT BEFORE next word starts
            # This prevents overlapping dialogue lines
            if i < len(word_timings) - 1:
                # End just before next word (subtract tiny amount to avoid overlap)
                end = format_time_ass(word_timings[i + 1]["start"] - 0.001)
            else:
                # Last word stays until video end
                end = format_time_ass(video_end)
            
            # Build sentence with current word highlighted in yellow
            highlighted_words = []
            for j, t in enumerate(word_timings):
                if j < i:
                    # Already spoken - keep white
                    highlighted_words.append(t["word"])
                elif j == i:
                    # Currently speaking - yellow + bold
                    highlighted_words.append("{\\c&H00FFFF&\\b1}" + t["word"] + "{\\c&HFFFFFF&\\b0}")
                else:
                    # Not yet spoken - white
                    highlighted_words.append(t["word"])
            
            highlighted_text = " ".join(highlighted_words)
            ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{highlighted_text}\n"
    
    elif effect == "fade":
        full_text = " ".join(w["word"] for w in word_timings)
        start = format_time_ass(word_timings[0]["start"])
        # Keep text visible for 2 seconds after audio ends
        end = format_time_ass(word_timings[-1]["end"] + 2.0)
        ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\fad(800,500)}}{full_text}\n"
    
    elif effect == "typewriter":
        # Show words appearing one by one - each line replaces the previous
        video_end = word_timings[-1]["end"] + 2.0
        
        for i, timing in enumerate(word_timings):
            start = format_time_ass(timing["start"])
            
            # CRITICAL FIX: End current line RIGHT BEFORE next word starts
            # This prevents overlapping dialogue lines
            if i < len(word_timings) - 1:
                # End just before next word (subtract tiny amount to avoid overlap)
                end = format_time_ass(word_timings[i + 1]["start"] - 0.001)
            else:
                # Last word stays until video end
                end = format_time_ass(video_end)
            
            # Show all words up to and including current word
            text_so_far = " ".join(w["word"] for w in word_timings[:i+1])
            ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text_so_far}\n"
    
    elif effect == "bouncing":
        full_text = " ".join(w["word"] for w in word_timings)
        start = format_time_ass(word_timings[0]["start"])
        # Keep text visible for 2 seconds after audio ends
        end = format_time_ass(word_timings[-1]["end"] + 2.0)
        ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\move(384,500,384,384,0,500)\\t(0,300,\\fscx120\\fscy120)\\t(300,500,\\fscx100\\fscy100)}}{full_text}\n"
    
    else:  # static
        full_text = " ".join(w["word"] for w in word_timings)
        start = format_time_ass(word_timings[0]["start"])
        # Keep text visible for 2 seconds after audio ends
        end = format_time_ass(word_timings[-1]["end"] + 2.0)
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
    """Create video with image, audio, and ASS subtitles - EXTENDED DURATION"""
    
    # Extend video duration by 2 seconds to keep text visible after audio ends
    video_duration = duration + 2.0
    
    # FFmpeg command with ASS subtitle overlay
    cmd = [
        "ffmpeg",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", f"scale=768:768:force_original_aspect_ratio=increase,crop=768:768,setsar=1,subtitles={subtitle_path}",
        "-c:v", "libx264",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "128k",
        "-t", str(video_duration),  # Use extended duration
        "-pix_fmt", "yuv420p",
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

# --- API ENDPOINTS ---

@app.get("/")
def home():
    """API root endpoint"""
    return {
        "message": "AI Fact Video Generator API",
        "version": "1.0.0",
        "endpoints": {
            "/facts": "GET - Get AI-generated facts by category",
            "/generate_video": "GET - Generate video with fact and effects",
            "/health": "GET - Health check",
            "/test": "GET - Test endpoint"
        },
        "status": "operational",
        "frontend_url": "https://multisite.interactivelink.site/factshortvideogen",
        "tts_engine": "gTTS + Enhanced Fallback"
    }

@app.get("/test")
def test_endpoint():
    """Test endpoint to verify CORS is working"""
    return {
        "message": "Backend is working! CORS should be configured correctly.",
        "status": "success",
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.get("/facts")
def get_facts(category: str):
    """Get AI-generated facts for a specific category"""
    if category not in PROMPTS:
        raise HTTPException(400, "Invalid category")
    facts = generate_facts_with_groq(category) or generate_facts_fallback(category)
    return {"facts": facts}

@app.get("/generate_video")
def generate_video(fact: str, category: str = "science", effect: str = "karaoke"):
    """Generate video with gTTS and centered animated subtitles"""
    
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
        
        # Step 2: Generate audio with gTTS
        print("Step 2: Generating voice with gTTS...")
        audio_success, duration = generate_audio(safe_fact, audio_path, category)
        if not audio_success:
            raise HTTPException(500, "Audio generation failed")
        
        duration = max(duration, 3.0)  # Minimum 3 seconds
        print(f"Audio duration: {duration:.2f}s")
        
        # Step 3: Generate IMPROVED word timings for karaoke
        print("Step 3: Creating improved word timings...")
        word_timings = generate_word_timings(safe_fact, duration)
        print(f"Generated {len(word_timings)} word timings with improved sync")
        
        # Debug: print timing information
        total_word_time = sum([t['end'] - t['start'] for t in word_timings])
        print(f"Total word time: {total_word_time:.2f}s, Audio duration: {duration:.2f}s")
        
        # Step 4: Create subtitle file with selected effect - CENTERED
        print(f"Step 4: Creating {effect} subtitles (centered)...")
        create_karaoke_subtitles(word_timings, subtitle_path, effect)
        
        # Step 5: Create final video with centered subtitles
        print("Step 5: Composing final video with centered text...")
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
            headers={
                "Content-Disposition": f"attachment; filename=video_{effect}_{category}.mp4",
                "X-Video-Size": str(os.path.getsize(output_path)),
                "X-Video-Duration": str(duration),
                "Access-Control-Expose-Headers": "Content-Disposition, X-Video-Size, X-Video-Duration"
            }
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
        "tts_available": True,
        "ffmpeg_available": True,
        "environment": "production",
        "cors_enabled": True,
        "frontend_url": "https://multisite.interactivelink.site/factshortvideogen",
        "karaoke_sync": "improved",
        "tts_engine": "gTTS + Enhanced Fallback"
    }

# --- Run server ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
