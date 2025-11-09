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
import concurrent.futures # Added for parallel image generation

# --- CONFIGURATION ---
app = FastAPI(
    title="AI Fact Short Video Generator API",
    description="Backend API for generating short videos with AI facts, images, and animated subtitles",
    version="1.0.0"
)

# CORS configuration
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

# UPDATED PROMPTS - Now requesting 2 sentences per fact
PROMPTS = {
    "science": "Give me 5 surprising science facts. Each fact should have EXACTLY 2 sentences. First sentence: the main fact (under 15 words). Second sentence: an interesting detail or explanation (under 15 words). Format: Fact 1 sentence. Detail sentence. (newline) Fact 2 sentence. Detail sentence. etc.",
    "successful_person": "Give me 5 inspiring facts about successful people. Each fact should have EXACTLY 2 sentences. First sentence: the main achievement (under 15 words). Second sentence: an interesting detail about their journey (under 15 words).",
    "unsolved_mystery": "Give me 5 unsolved mysteries. Each fact should have EXACTLY 2 sentences. First sentence: the mystery (under 15 words). Second sentence: an intriguing detail (under 15 words).",
    "history": "Give me 5 memorable history facts. Each fact should have EXACTLY 2 sentences. First sentence: the main historical fact (under 15 words). Second sentence: a surprising detail (under 15 words).",
    "sports": "Give me 5 legendary sports facts. Each fact should have EXACTLY 2 sentences. First sentence: the achievement (under 15 words). Second sentence: an interesting context (under 15 words)."
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
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(audio_path)
        
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
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
        duration = max(len(words) * 0.5 + 1.0, 3.0)
        
        print(f"Generating enhanced fallback audio: {duration:.2f}s duration")
        
        base_freq = 200
        words = text.split()
        
        if len(words) > 0:
            filter_chain = []
            for i, word in enumerate(words):
                word_duration = duration / len(words)
                start_time = i * word_duration
                freq_variation = base_freq + (len(word) * 10)
                
                filter_chain.append(
                    f"sine=frequency={freq_variation}:duration={word_duration}:"
                    f"sample_rate=22050,adelay={int(start_time * 1000)}|{int(start_time * 1000)}"
                )
            
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
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
                "-t", str(duration), "-acodec", "libmp3lame", "-b:a", "64k",
                audio_path, "-y", "-loglevel", "error"
            ], timeout=30)
        
        return os.path.exists(audio_path), duration
        
    except Exception as e:
        print(f"Enhanced fallback error: {e}")
        duration = len(text.split()) * 0.5 + 1.0
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
            "-t", str(duration), "-acodec", "libmp3lame", "-b:a", "64k",
            audio_path, "-y", "-loglevel", "error"
        ], timeout=30)
        return os.path.exists(audio_path), duration

def generate_audio(text: str, audio_path: str, category: str = "science"):
    """Generate audio using gTTS with enhanced fallback"""
    
    print("Attempting gTTS audio generation...")
    success, duration = generate_audio_with_gtts(text, audio_path)
    if success:
        print("✅ Audio generated with gTTS")
        return success, duration
    
    print("gTTS failed, using enhanced fallback audio...")
    return generate_audio_fallback(text, audio_path)

# --- WORD TIMING FUNCTIONS ---

def analyze_speech_pattern(text: str, duration: float):
    """Analyze text to create more realistic word timings based on linguistic patterns"""
    words = text.split()
    if not words:
        return []
    
    word_complexity = {
        'short': 0.3,
        'medium': 0.5,
        'long': 0.8,
        'complex': 1.2
    }
    
    short_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    base_time_per_word = duration / len(words)
    
    timings = []
    current_time = 0.1
    
    for i, word in enumerate(words):
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
        
        pause_multiplier = 1.0
        if i > 0 and any(punc in words[i-1] for punc in ',;'):
            pause_multiplier = 1.3
        elif i > 0 and any(punc in words[i-1] for punc in '.!?'):
            pause_multiplier = 1.6
        
        word_duration = base_time_per_word * complexity * pause_multiplier
        
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
        
        if i < len(words) - 1:
            current_time += 0.05
    
    return timings

def generate_word_timings(text: str, duration: float):
    """Generate improved word timings for better karaoke sync"""
    words = text.split()
    if not words:
        return []
    
    return analyze_speech_pattern(text, duration)

def split_into_sentences(text: str):
    """Split text into sentences"""
    # Split by period, exclamation, or question mark followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def format_time_ass(seconds):
    """Convert seconds to ASS timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"

def create_karaoke_subtitles(word_timings, subtitle_path, effect="karaoke", sentence_word_groups=None):
    """
    Create ASS subtitle file with karaoke or other effects - CENTERED TEXT - SENTENCE BY SENTENCE
    Uses sentence_word_groups to show each sentence on its own timing segment.
    Falls back to original behavior if sentence_word_groups is not provided or empty.
    """
    
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
"""

    if not word_timings:
        ass_content += "Dialogue: 0,0:00:00.00,0:00:03.00,Default,,0,0,0,, \n"
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
        return

    # Use sentence word groups if provided and valid
    if sentence_word_groups and len(sentence_word_groups) > 0:
        print(f"DEBUG: Using {len(sentence_word_groups)} sentence word groups for subtitles")
        video_end = word_timings[-1]["end"] + 2.0 # Or determine from sentence_timings if passed separately

        # Determine the end time for each sentence group
        sentence_end_times = []
        for i in range(len(sentence_word_groups)):
            if i < len(sentence_word_groups) - 1:
                # Sentence ends when the next sentence starts
                next_sentence_start = sentence_word_groups[i + 1][0]["start"]
                sentence_end_times.append(next_sentence_start)
            else:
                # Last sentence ends at video_end
                sentence_end_times.append(video_end)

        # Process each sentence group based on the effect
        for sent_idx, (sentence_words, sentence_end_time) in enumerate(zip(sentence_word_groups, sentence_end_times)):
            if not sentence_words:
                continue

            sentence_start_time = sentence_words[0]["start"]
            sentence_text = " ".join(w["word"] for w in sentence_words)

            start_ass = format_time_ass(sentence_start_time)
            end_ass = format_time_ass(sentence_end_time)

            if effect == "karaoke":
                # Karaoke effect *within* the sentence's time window
                for i, timing in enumerate(sentence_words):
                    word_start_ass = format_time_ass(timing["start"])

                    # Determine word end time
                    if i < len(sentence_words) - 1:
                        # End just before the next word in *this* sentence starts
                        word_end_ass = format_time_ass(sentence_words[i + 1]["start"] - 0.001)
                    else:
                        # Last word of the sentence, ends when the sentence window ends
                        word_end_ass = end_ass

                    # Build text string for this frame: previous words normal, current word highlighted, rest normal
                    highlighted_words = []
                    for j, w_timing in enumerate(sentence_words):
                        if j < i:
                            highlighted_words.append(w_timing["word"])
                        elif j == i:
                            highlighted_words.append("{\\c&H00FFFF&\\b1}" + w_timing["word"] + "{\\c&HFFFFFF&\\b0}")
                        else:
                            highlighted_words.append(w_timing["word"])

                    current_sentence_text = " ".join(highlighted_words)
                    ass_content += f"Dialogue: 0,{word_start_ass},{word_end_ass},Default,,0,0,0,,{current_sentence_text}\n"

            elif effect == "typewriter":
                # Typewriter effect *within* the sentence's time window
                for i, timing in enumerate(sentence_words):
                    word_start_ass = format_time_ass(timing["start"])

                    if i < len(sentence_words) - 1:
                        word_end_ass = format_time_ass(sentence_words[i + 1]["start"] - 0.001)
                    else:
                        word_end_ass = end_ass

                    # Text appears progressively
                    text_so_far = " ".join(w["word"] for w in sentence_words[:i+1])
                    ass_content += f"Dialogue: 0,{word_start_ass},{word_end_ass},Default,,0,0,0,,{text_so_far}\n"

            else: # 'static', 'fade', 'bouncing', or any other effect
                # Show the entire sentence text for its duration window
                if effect == "fade":
                    # Apply fade in/out effect
                    ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{{\\fad(300,300)}}{sentence_text}\n"
                elif effect == "bouncing":
                    # Apply a simple movement effect
                    ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{{\\move(384,550,384,450,0,500)\\t(0,250,\\fscx110\\fscy110)\\t(250,500,\\fscx100\\fscy100)}}{sentence_text}\n"
                else: # static or default
                    ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{sentence_text}\n"

    else:
        # --- OLD FALLBACK BEHAVIOR ---
        print("DEBUG: sentence_word_groups not provided or empty, using fallback.")
        video_end = word_timings[-1]["end"] + 2.0
        full_text = " ".join(w["word"] for w in word_timings)
        start_ass = format_time_ass(word_timings[0]["start"])
        end_ass = format_time_ass(video_end)

        if effect == "karaoke":
            for i, timing in enumerate(word_timings):
                word_start_ass = format_time_ass(timing["start"])
                
                if i < len(word_timings) - 1:
                    word_end_ass = format_time_ass(word_timings[i + 1]["start"] - 0.001)
                else:
                    word_end_ass = end_ass
                
                highlighted_words = []
                for j, t in enumerate(word_timings):
                    if j < i:
                        highlighted_words.append(t["word"])
                    elif j == i:
                        highlighted_words.append("{\\c&H00FFFF&\\b1}" + t["word"] + "{\\c&HFFFFFF&\\b0}")
                    else:
                        highlighted_words.append(t["word"])
                
                highlighted_text = " ".join(highlighted_words)
                ass_content += f"Dialogue: 0,{word_start_ass},{word_end_ass},Default,,0,0,0,,{highlighted_text}\n"
        
        elif effect == "fade":
            ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{{\\fad(800,500)}}{full_text}\n"
        
        elif effect == "typewriter":
            for i, timing in enumerate(word_timings):
                word_start_ass = format_time_ass(timing["start"])
                
                if i < len(word_timings) - 1:
                    word_end_ass = format_time_ass(word_timings[i + 1]["start"] - 0.001)
                else:
                    word_end_ass = end_ass
                
                text_so_far = " ".join(w["word"] for w in word_timings[:i+1])
                ass_content += f"Dialogue: 0,{word_start_ass},{word_end_ass},Default,,0,0,0,,{text_so_far}\n"
        
        elif effect == "bouncing":
            ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{{\\move(384,500,384,384,0,500)\\t(0,300,\\fscx120\\fscy120)\\t(300,500,\\fscx100\\fscy100)}}{full_text}\n"
        
        else: # static or default
            ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{full_text}\n"

    # Write the final ASS content to the file
    ass_content += "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    with open(subtitle_path, "w", encoding="utf-8") as f:
        f.write(ass_content)


def create_video_with_images_and_subtitles(image_paths, audio_path, subtitle_path, output_path, duration, sentence_timings):
    """Create video with multiple images transitioning based on sentences - OPTIMIZED FOR SPEED"""
    
    video_duration = duration + 2.0
    
    # Create filter complex for image transitions
    if len(image_paths) == 1:
        # Single image - simple case with FAST settings
        cmd = [
            "ffmpeg",
            "-loop", "1", "-i", image_paths[0],
            "-i", audio_path,
            "-vf", f"scale=768:768:force_original_aspect_ratio=increase,crop=768:768,setsar=1,subtitles={subtitle_path}",
            "-c:v", "libx264",
            "-preset", "ultrafast",  # Changed from "medium" to "ultrafast"
            "-crf", "28",  # Added CRF for faster encoding (lower quality but much faster)
            "-c:a", "aac",
            "-b:a", "96k",  # Reduced from 128k to 96k
            "-t", str(video_duration),
            "-pix_fmt", "yuv420p",
            "-threads", "0",  # Use all available CPU threads
            "-y",
            "-loglevel", "error",
            output_path
        ]
    else:
        # Multiple images - create transitions with FAST settings
        inputs = []
        for img in image_paths:
            inputs.extend(["-loop", "1", "-i", img])
        inputs.extend(["-i", audio_path])
        
        # Calculate transition point based on sentence timings
        transition_time = sentence_timings[1] if len(sentence_timings) > 1 else duration / 2
        
        # Build filter for image transition with crossfade
        filter_parts = []
        for i in range(len(image_paths)):
            filter_parts.append(f"[{i}:v]scale=768:768:force_original_aspect_ratio=increase,crop=768:768,setsar=1,setpts=PTS-STARTPTS[v{i}]")
        
        # Create smooth crossfade between images
        if len(image_paths) == 2:
            crossfade_duration = 0.3  # Reduced from 0.5 to 0.3 seconds
            filter_parts.append(f"[v0][v1]xfade=transition=fade:duration={crossfade_duration}:offset={transition_time-crossfade_duration/2}[vout]")
            filter_parts.append(f"[vout]subtitles={subtitle_path}[final]")
        else:
            # If more than 2 images, chain xfades
            prev = "v0"
            for i in range(1, len(image_paths)):
                t_time = sentence_timings[i] if i < len(sentence_timings) else (duration * i / len(image_paths))
                crossfade_duration = 0.3  # Reduced from 0.5 to 0.3
                next_label = f"vout{i}" if i < len(image_paths) - 1 else "vout"
                filter_parts.append(f"[{prev}][v{i}]xfade=transition=fade:duration={crossfade_duration}:offset={t_time-crossfade_duration/2}[{next_label}]")
                prev = next_label
            filter_parts.append(f"[vout]subtitles={subtitle_path}[final]")
        
        filter_complex = ";".join(filter_parts)
        
        cmd = [
            "ffmpeg"
        ] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[final]",
            "-map", f"{len(image_paths)}:a",
            "-c:v", "libx264",
            "-preset", "ultrafast",  # Changed from "medium" to "ultrafast"
            "-crf", "28",  # Added CRF for faster encoding
            "-c:a", "aac",
            "-b:a", "96k",  # Reduced from 128k
            "-t", str(video_duration),
            "-pix_fmt", "yuv420p",
            "-threads", "0",  # Use all CPU threads
            "-y",
            "-loglevel", "error",
            output_path
        ]
    
    try:
        result = subprocess.run(cmd, timeout=60, capture_output=True)  # Reduced timeout from 120 to 60
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr.decode()}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        print(f"Video creation error: {e}")
        return False

def generate_image_pollinations(prompt, path):
    """Generate image with faster timeout"""
    try:
        enhanced_prompt = f"high quality cinematic image: {prompt}, 4k, detailed, vibrant colors"
        url = f"https://pollinations.ai/p/{urllib.parse.quote(enhanced_prompt)}?width=768&height=768&nologo=true&enhance=true"
        resp = requests.get(url, timeout=15)  # Reduced from 20 to 15 seconds
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
    
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    
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
    """Generate facts with higher temperature for more variety"""
    if not groq_client:
        return None
    try:
        prompt = PROMPTS.get(category, PROMPTS["science"])
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return exactly 5 facts, each with 2 sentences. No bullets, no numbers. Format: Sentence1. Sentence2. (blank line) Next fact..."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=1.2  # INCREASED temperature for more variety
        )
        
        content = response.choices[0].message.content.strip()
        
        # Split by double newlines or paragraph breaks
        fact_blocks = re.split(r'\n\s*\n', content)
        
        facts = []
        for block in fact_blocks:
            # Clean the block
            cleaned = block.strip().strip("\"'•-—12345.")
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', cleaned)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) >= 2:
                # Take first 2 sentences
                fact_text = f"{sentences[0]} {sentences[1]}"
                if 20 < len(fact_text) < 200:
                    facts.append(fact_text)
            elif len(sentences) == 1 and 20 < len(sentences[0]) < 120:
                # Single sentence - add it anyway
                facts.append(sentences[0])
            
            if len(facts) >= 5:
                break
        
        return facts[:5] if len(facts) >= 5 else None
        
    except Exception as e:
        print(f"Groq fact gen failed: {e}")
        return None

def generate_facts_fallback(category: str):
    """Fallback facts - now with 2 sentences each"""
    defaults = {
        "science": [
            "Bananas are naturally radioactive. They contain potassium-40, a radioactive isotope.",
            "Octopuses have three hearts. Two pump blood to gills, one to organs.",
            "Honey never spoils. Archaeologists found edible honey in ancient tombs.",
            "Venus rotates backward. It's the only planet spinning clockwise.",
            "Your stomach acid can dissolve metal. It's incredibly corrosive hydrochloric acid."
        ],
        "successful_person": [
            "Oprah was fired from her first TV job. They said she was unfit for television.",
            "Steve Jobs was adopted. His biological parents gave him up.",
            "JK Rowling was rejected 12 times. Publishers didn't believe in Harry Potter.",
            "Colonel Sanders started KFC at 65. He was broke and living off checks.",
            "Walt Disney was fired for lacking imagination. His editor didn't see potential."
        ],
        "unsolved_mystery": [
            "The Voynich manuscript remains undeciphered. No one can read its strange text.",
            "DB Cooper vanished after hijacking. He parachuted with ransom money, never found.",
            "The Bermuda Triangle mystery continues. Ships and planes disappear mysteriously.",
            "Zodiac Killer was never caught. He taunted police with cryptic letters.",
            "Oak Island money pit unsolved. Treasure hunters have searched for centuries."
        ],
        "history": [
            "Cleopatra lived closer to iPhone than pyramids. Time perspective is mind-blowing.",
            "Oxford University predates Aztec Empire. It's incredibly ancient institution.",
            "The Great Wall visible from space myth. Astronauts can't actually see it.",
            "Napoleon was actually average height. British propaganda made him seem short.",
            "Vikings discovered America before Columbus. They reached Newfoundland centuries earlier."
        ],
        "sports": [
            "Michael Jordan was cut from high school team. Rejection fueled his legendary career.",
            "Usain Bolt has scoliosis. His spine curves despite being fastest man.",
            "Serena Williams holds 23 Grand Slams. She's one of tennis's greatest.",
            "Muhammad Ali won Olympic gold medal. He later threw it in river protesting racism.",
            "Pele scored 1283 career goals. He's football's most prolific scorer."
        ]
    }
    # Randomize the order to add variety
    facts = defaults.get(category, defaults["science"])
    return random.sample(facts, min(5, len(facts)))

# --- API ENDPOINTS ---

@app.get("/")
def home():
    return {
        "message": "AI Fact Video Generator API",
        "version": "2.0.0",
        "endpoints": {
            "/facts": "GET - Get AI-generated facts by category",
            "/generate_video": "GET - Generate video with fact and effects",
            "/health": "GET - Health check",
            "/test": "GET - Test endpoint"
        },
        "status": "operational",
        "frontend_url": "https://multisite.interactivelink.site/factshortvideogen  ",
        "tts_engine": "gTTS + Enhanced Fallback",
        "features": ["2 sentences per fact", "2 images per video", "Higher variety"]
    }

@app.get("/test")
def test_endpoint():
    return {
        "message": "Backend is working! CORS should be configured correctly.",
        "status": "success",
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.get("/facts")
def get_facts(category: str):
    """Get AI-generated facts with 2 sentences each - more variety"""
    if category not in PROMPTS:
        raise HTTPException(400, "Invalid category")
    
    # Try AI generation with higher temperature
    facts = generate_facts_with_groq(category)
    
    # Fallback with randomization
    if not facts:
        facts = generate_facts_fallback(category)
    
    return {"facts": facts}

@app.get("/generate_video")
def generate_video(fact: str, category: str = "science", effect: str = "karaoke"):
    """Generate video with 2 images (one per sentence) and centered animated subtitles"""
    
    safe_fact = fact.strip()[:300]
    if not safe_fact:
        raise HTTPException(400, "Fact text is required")
    
    print(f"\n=== Starting video generation ===")
    print(f"Fact: {safe_fact}")
    print(f"Category: {category}")
    print(f"Effect: {effect}")
    
    # Split fact into sentences
    sentences = split_into_sentences(safe_fact)
    print(f"Found {len(sentences)} sentences: {sentences}")
    
    # Temporary file paths
    image_paths = [f"/tmp/{uuid.uuid4()}.jpg" for _ in range(len(sentences))]
    audio_path = f"/tmp/{uuid.uuid4()}.mp3"
    subtitle_path = f"/tmp/{uuid.uuid4()}.ass"
    output_path = f"/tmp/{uuid.uuid4()}.mp4"
    
    try:
        # Step 1: Generate images for each sentence (PARALLEL PROCESSING for speed)
        print(f"Step 1: Generating {len(sentences)} images in parallel...")
        print(f"DEBUG: Sentences to process: {sentences}")
        
        def generate_single_image(args):
            i, sentence, img_path = args
            print(f"DEBUG: Generating image {i+1} for sentence: '{sentence}'")
            image_prompt = f"{category} theme: {sentence}"
            
            # Try Pollinations first
            success = generate_image_pollinations(image_prompt, img_path)
            if not success:
                print(f"DEBUG: Pollinations failed for image {i+1}, using placeholder")
                success = generate_image_placeholder(sentence, img_path, category)
            
            if success:
                print(f"DEBUG: Image {i+1} saved to {img_path}, size: {os.path.getsize(img_path)} bytes")
            else:
                print(f"DEBUG: Image {i+1} FAILED completely")
            
            return i, success
        
        # Generate images in parallel threads for 2-3x speed boost
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sentences), 3)) as executor:
            image_args = [(i, sentence, image_paths[i]) for i, sentence in enumerate(sentences)]
            results = list(executor.map(generate_single_image, image_args))
        
        # Check all images generated successfully
        for i, success in results:
            if not success:
                raise HTTPException(500, f"Image {i+1} generation failed")
            print(f"✅ Image {i+1}/{len(sentences)} generated successfully")
        
        # Step 2: Generate audio with gTTS for full fact
        print("Step 2: Generating voice with gTTS...")
        audio_success, duration = generate_audio(safe_fact, audio_path, category)
        if not audio_success:
            raise HTTPException(500, "Audio generation failed")
        
        duration = max(duration, 3.0)
        print(f"Audio duration: {duration:.2f}s")
        
        # Step 3: Generate word timings for karaoke
        print("Step 3: Creating improved word timings...")
        word_timings = generate_word_timings(safe_fact, duration)
        print(f"Generated {len(word_timings)} word timings")
        
        # Calculate sentence timing boundaries and word groups
        sentence_timings = []
        sentence_word_groups = []  # NEW: Track which words belong to which sentence
        word_index = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            if word_index + sentence_word_count <= len(word_timings):
                # Get timing of first word in this sentence
                sentence_start = word_timings[word_index]["start"]
                sentence_timings.append(sentence_start)
                
                # Get word timings for this sentence
                sentence_word_timings = word_timings[word_index:word_index + sentence_word_count]
                sentence_word_groups.append(sentence_word_timings)
                
                word_index += sentence_word_count
            else:
                # Fallback: distribute evenly
                sentence_timings.append(duration * len(sentence_timings) / len(sentences))
                sentence_word_groups.append([])
        
        print(f"Sentence timings: {sentence_timings}")
        print(f"Sentence word groups: {len(sentence_word_groups)} groups")
        
        # Step 4: Create subtitle file with selected effect
        print(f"Step 4: Creating {effect} subtitles (centered)...")
        create_karaoke_subtitles(word_timings, subtitle_path, effect, sentence_word_groups=sentence_word_groups)
        
        # Step 5: Create final video with multiple images and transitions
        print(f"Step 5: Composing final video with {len(image_paths)} images...")
        if not create_video_with_images_and_subtitles(
            image_paths, audio_path, subtitle_path, output_path, duration, sentence_timings
        ):
            raise HTTPException(500, "Video composition failed")
        
        print(f"Video created successfully: {os.path.getsize(output_path)} bytes")
        
        # Cleanup temp files
        for temp_file in image_paths + [audio_path, subtitle_path]:
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
                "X-Image-Count": str(len(image_paths)),
                "X-Sentence-Count": str(len(sentences)),
                "Access-Control-Expose-Headers": "Content-Disposition, X-Video-Size, X-Video-Duration, X-Image-Count, X-Sentence-Count"
            }
        )
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        # Cleanup on error
        for temp_file in image_paths + [audio_path, subtitle_path, output_path]:
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
        "frontend_url": "https://multisite.interactivelink.site/factshortvideogen  ",
        "karaoke_sync": "improved",
        "tts_engine": "gTTS + Enhanced Fallback",
        "features": {
            "sentences_per_fact": 2,
            "images_per_video": 2,
            "variety": "high_temperature",
            "transitions": "crossfade"
        }
    }

# --- Run server ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
