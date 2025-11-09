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
        "https://multisite.interactivelink.site  ",
        "https://multisite.interactivelink.site/factshortvideogen  ",
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
                    ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{{\\move(384,55
