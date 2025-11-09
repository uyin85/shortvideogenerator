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
import concurrent.futures

# --- CONFIGURATION ---
app = FastAPI(
    title="AI Fact Short Video Generator API",
    description="Backend API for generating short videos with AI facts, images, and animated subtitles",
    version="2.1.0"
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

# UPDATED PROMPTS - 2 sentences per fact
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
                print(f"gTTS success: {os.path.getsize(audio_path)} bytes, {duration:.2f}s")
            except:
                duration = len(text.split()) * 0.5 + 1.0
            return True, duration
        else:
            return False, 0.0
    except Exception as e:
        print(f"gTTS error: {e}")
        return False, 0.0

def generate_audio_fallback(text: str, audio_path: str):
    try:
        words = text.split()
        duration = max(len(words) * 0.5 + 1.0, 3.0)
        print(f"Generating enhanced fallback audio: {duration:.2f}s")
        base_freq = 200
        if len(words) > 0:
            filter_chain = []
            for i, word in enumerate(words):
                word_duration = duration / len(words)
                start_time = i * word_duration
                freq_variation = base_freq + (len(word) * 10)
                filter_chain.append(
                    f"sine=frequency={freq_variation}:duration={word_duration}:sample_rate=22050,"
                    f"adelay={int(start_time * 1000)}|{int(start_time * 1000)}"
                )
            filter_complex = f"{'+'.join(filter_chain)}"
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", filter_complex,
                "-af", f"volume=0.05,afade=t=in:st=0:d=0.5,afade=t=out:st={duration-0.5}:d=0.5",
                "-acodec", "libmp3lame", "-b:a", "64k", "-ar", "22050",
                "-t", str(duration), audio_path, "-y", "-loglevel", "error"
            ], timeout=30)
        else:
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", f"sine=frequency=300:duration={duration}",
                "-af", f"afade=t=in:st=0:d=0.5,afade=t=out:st={duration-0.5}:d=0.5,volume=0.05",
                "-acodec", "libmp3lame", "-b:a", "64k", "-ar", "22050",
                audio_path, "-y", "-loglevel", "error"
            ], timeout=30)
        success = os.path.exists(audio_path) and os.path.getsize(audio_path) > 500
        if not success:
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
    print("Attempting gTTS audio generation...")
    success, duration = generate_audio_with_gtts(text, audio_path)
    if success:
        print("Audio generated with gTTS")
        return success, duration
    print("gTTS failed, using enhanced fallback audio...")
    return generate_audio_fallback(text, audio_path)

# --- WORD TIMING FUNCTIONS ---
def analyze_speech_pattern(text: str, duration: float):
    words = text.split()
    if not words:
        return []
    word_complexity = {'short': 0.3, 'medium': 0.5, 'long': 0.8, 'complex': 1.2}
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
        timings.append({"word": word, "start": start_time, "end": end_time})
        current_time = end_time
        if i < len(words) - 1:
            current_time += 0.05
    return timings

def generate_word_timings(text: str, duration: float):
    words = text.split()
    if not words:
        return []
    return analyze_speech_pattern(text, duration)

def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def format_time_ass(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"

# --- NEW: Sentence timing helper ---
def build_sentence_timings(sentence_word_groups, total_duration):
    timings = []
    end_times = []
    for i, group in enumerate(sentence_word_groups):
        if group:
            start = group[0]["start"]
        else:
            start = total_duration * i / max(len(sentence_word_groups), 1)
        timings.append(start)
        if i < len(sentence_word_groups) - 1 and sentence_word_groups[i + 1]:
            end = sentence_word_groups[i + 1][0]["start"]
        else:
            end = total_duration + 2.0
        end_times.append(end)
    return timings, end_times

# --- FIXED: create_karaoke_subtitles ---
def create_karaoke_subtitles(
    word_timings,
    subtitle_path,
    effect="karaoke",
    sentence_word_groups=None,
    total_duration=0.0,
):
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

    if sentence_word_groups and len(sentence_word_groups) > 0:
        timings, end_times = build_sentence_timings(sentence_word_groups, total_duration)

        for sent_idx, (words, sent_end) in enumerate(zip(sentence_word_groups, end_times)):
            if not words:
                continue
            sent_start = words[0]["start"]
            start_ass = format_time_ass(sent_start)
            end_ass   = format_time_ass(sent_end)

            if effect == "karaoke":
                for local_i, w in enumerate(words):
                    w_start = format_time_ass(w["start"])
                    w_end = format_time_ass(
                        words[local_i + 1]["start"] - 0.001 if local_i < len(words) - 1 else sent_end
                    )
                    parts = []
                    for j, ww in enumerate(words):
                        if j < local_i:
                            parts.append(ww["word"])
                        elif j == local_i:
                            parts.append("{\\c&H00FFFF&\\b1}" + ww["word"] + "{\\c&HFFFFFF&\\b0}")
                        else:
                            parts.append(ww["word"])
                    line = " ".join(parts)
                    ass_content += f"Dialogue: 0,{w_start},{w_end},Default,,0,0,0,,{line}\n"

            elif effect == "typewriter":
                for local_i, w in enumerate(words):
                    w_start = format_time_ass(w["start"])
                    w_end   = format_time_ass(
                        words[local_i + 1]["start"] - 0.001 if local_i < len(words) - 1 else sent_end
                    )
                    line = " ".join(ww["word"] for ww in words[: local_i + 1])
                    ass_content += f"Dialogue: 0,{w_start},{w_end},Default,,0,0,0,,{line}\n"

            else:
                sentence_text = " ".join(w["word"] for w in words)
                if effect == "fade":
                    ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{{\\fad(300,300)}}{sentence_text}\n"
                elif effect == "bouncing":
                    ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{{\\move(384,550,384,450,0,500)\\t(0,250,\\fscx110\\fscy110)\\t(250,500,\\fscx100\\fscy100)}}{sentence_text}\n"
                else:
                    ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{sentence_text}\n"

    else:
        video_end = word_timings[-1]["end"] + 2.0
        full_text = " ".join(w["word"] for w in word_timings)
        start_ass = format_time_ass(word_timings[0]["start"])
        end_ass   = format_time_ass(video_end)

        if effect == "karaoke":
            for i, t in enumerate(word_timings):
                w_start = format_time_ass(t["start"])
                w_end   = format_time_ass(word_timings[i + 1]["start"] - 0.001 if i + 1 < len(word_timings) else video_end)
                parts = []
                for j, tt in enumerate(word_timings):
                    if j < i:
                        parts.append(tt["word"])
                    elif j == i:
                        parts.append("{\\c&H00FFFF&\\b1}" + tt["word"] + "{\\c&HFFFFFF&\\b0}")
                    else:
                        parts.append(tt["word"])
                line = " ".join(parts)
                ass_content += f"Dialogue: 0,{w_start},{w_end},Default,,0,0,0,,{line}\n"

        elif effect == "typewriter":
            for i, t in enumerate(word_timings):
                w_start = format_time_ass(t["start"])
                w_end   = format_time_ass(word_timings[i + 1]["start"] - 0.001 if i + 1 < len(word_timings) else video_end)
                line = " ".join(w["word"] for w in word_timings[: i + 1])
                ass_content += f"Dialogue: 0,{w_start},{w_end},Default,,0,0,0,,{line}\n"

        elif effect == "fade":
            ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{{\\fad(800,500)}}{full_text}\n"

        elif effect == "bouncing":
            ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{{\\move(384,500,384,384,0,500)\\t(0,300,\\fscx120\\fscy120)\\t(300,500,\\fscx100\\fscy100)}}{full_text}\n"
        else:
            ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{full_text}\n"

    ass_content += "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    with open(subtitle_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

# --- VIDEO COMPOSITION ---
def create_video_with_images_and_subtitles(image_paths, audio_path, subtitle_path, output_path, duration, sentence_timings):
    video_duration = duration + 2.0
    if len(image_paths) == 1:
        cmd = [
            "ffmpeg", "-loop", "1", "-i", image_paths[0],
            "-i", audio_path,
            "-vf", f"scale=768:768:force_original_aspect_ratio=increase,crop=768:768,setsar=1,subtitles={subtitle_path}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "96k", "-t", str(video_duration),
            "-pix_fmt", "yuv420p", "-threads", "0", "-y", "-loglevel", "error",
            output_path
        ]
    else:
        inputs = []
        for img in image_paths:
            inputs.extend(["-loop", "1", "-i", img])
        inputs.extend(["-i", audio_path])
        transition_time = sentence_timings[1] if len(sentence_timings) > 1 else duration / 2
        filter_parts = []
        for i in range(len(image_paths)):
            filter_parts.append(f"[{i}:v]scale=768:768:force_original_aspect_ratio=increase,crop=768:768,setsar=1,setpts=PTS-STARTPTS[v{i}]")
        if len(image_paths) == 2:
            crossfade_duration = 0.3
            filter_parts.append(f"[v0][v1]xfade=transition=fade:duration={crossfade_duration}:offset={transition_time-crossfade_duration/2}[vout]")
            filter_parts.append(f"[vout]subtitles={subtitle_path}[final]")
        else:
            prev = "v0"
            for i in range(1, len(image_paths)):
                t_time = sentence_timings[i] if i < len(sentence_timings) else (duration * i / len(image_paths))
                crossfade_duration = 0.3
                next_label = f"vout{i}" if i < len(image_paths) - 1 else "vout"
                filter_parts.append(f"[{prev}][v{i}]xfade=transition=fade:duration={crossfade_duration}:offset={t_time-crossfade_duration/2}[{next_label}]")
                prev = next_label
            filter_parts.append(f"[vout]subtitles={subtitle_path}[final]")
        filter_complex = ";".join(filter_parts)
        cmd = [
            "ffmpeg"
        ] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[final]", "-map", f"{len(image_paths)}:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "96k", "-t", str(video_duration),
            "-pix_fmt", "yuv420p", "-threads", "0", "-y", "-loglevel", "error",
            output_path
        ]
    try:
        result = subprocess.run(cmd, timeout=60, capture_output=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr.decode()}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        print(f"Video creation error: {e}")
        return False

# --- IMAGE GENERATION ---
def generate_image_pollinations(prompt, path):
    try:
        enhanced_prompt = f"high quality cinematic image: {prompt}, 4k, detailed, vibrant colors"
        url = f"https://pollinations.ai/p/{urllib.parse.quote(enhanced_prompt)}?width=768&height=768&nologo=true&enhance=true"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200 and len(resp.content) > 1000:
            with open(path, "wb") as f:
                f.write(resp.content)
            print(f"Image generated from Pollinations")
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
        r2, g2, b2 = tuple(int(colors[1].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) if len(colors) > 1 else (r1, g1, b1)
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
        color = colors[1] if len(colors) > 1 else colors[0]
        color_rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color_rgb)
    img.save(path, "JPEG", quality=90)
    print(f"Placeholder image generated")
    return True

# --- FACT GENERATION ---
def generate_facts_with_groq(category: str):
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
            temperature=1.2
        )
        content = response.choices[0].message.content.strip()
        fact_blocks = re.split(r'\n\s*\n', content)
        facts = []
        for block in fact_blocks:
            cleaned = block.strip().strip("\"'•-—12345.")
            sentences = re.split(r'(?<=[.!?])\s+', cleaned)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            if len(sentences) >= 2:
                fact_text = f"{sentences[0]} {sentences[1]}"
                if 20 < len(fact_text) < 200:
                    facts.append(fact_text)
            elif len(sentences) == 1 and 20 < len(sentences[0]) < 120:
                facts.append(sentences[0])
            if len(facts) >= 5:
                break
        return facts[:5] if len(facts) >= 5 else None
    except Exception as e:
        print(f"Groq fact gen failed: {e}")
        return None

def generate_facts_fallback(category: str):
    defaults = {
        "science": [
            "Bananas are naturally radioactive. They contain potassium-40, a radioactive isotope.",
            "Octopuses have three hearts. Two pump blood to gills, one to organs.",
            "Honey never spoils. Archaeologists found edible honey in ancient tombs.",
            "Venus rotates backward. It's the only planet spinning clockwise.",
            "Your stomach acid can dissolve metal. It's incredibly corrosive hydrochloric acid."
        ],
        # ... (rest unchanged)
    }
    facts = defaults.get(category, defaults["science"])
    return random.sample(facts, min(5, len(facts)))

# --- API ENDPOINTS ---
@app.get("/")
def home():
    return {
        "message": "AI Fact Video Generator API",
        "version": "2.1.0",
        "status": "2nd scene karaoke FIXED"
    }

@app.get("/generate_video")
def generate_video(fact: str, category: str = "science", effect: str = "karaoke"):
    safe_fact = fact.strip()[:300]
    if not safe_fact:
        raise HTTPException(400, "Fact text is required")

    print(f"\n=== Generating Video ===")
    print(f"Fact: {safe_fact}")

    sentences = split_into_sentences(safe_fact)
    print(f"Sentences: {sentences}")

    image_paths = [f"/tmp/{uuid.uuid4()}.jpg" for _ in range(len(sentences))]
    audio_path = f"/tmp/{uuid.uuid4()}.mp3"
    subtitle_path = f"/tmp/{uuid.uuid4()}.ass"
    output_path = f"/tmp/{uuid.uuid4()}.mp4"

    try:
        # 1. Generate images
        def gen_img(i, s, p):
            prompt = f"{category} theme: {s}"
            success = generate_image_pollinations(prompt, p) or generate_image_placeholder(s, p, category)
            return i, success
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(gen_img, range(len(sentences)), sentences, image_paths))
        for i, success in results:
            if not success:
                raise HTTPException(500, f"Image {i+1} failed")

        # 2. Generate audio
        audio_success, duration = generate_audio(safe_fact, audio_path, category)
        if not audio_success:
            raise HTTPException(500, "Audio failed")
        duration = max(duration, 3.0)

        # 3. Word timings
        word_timings = generate_word_timings(safe_fact, duration)

        # 4. Group words by sentence
        sentence_word_groups = []
        word_index = 0
        for sentence in sentences:
            sentence_words = sentence.split()
            count = len(sentence_words)
            if word_index + count <= len(word_timings):
                group = word_timings[word_index:word_index + count]
                sentence_word_groups.append(group)
                word_index += count
            else:
                sentence_word_groups.append([])

        # 5. Build timings
        sentence_timings, _ = build_sentence_timings(sentence_word_groups, duration)

        # 6. Subtitles
        create_karaoke_subtitles(
            word_timings, subtitle_path, effect,
            sentence_word_groups=sentence_word_groups,
            total_duration=duration
        )

        # 7. Compose video
        if not create_video_with_images_and_subtitles(
            image_paths, audio_path, subtitle_path, output_path, duration, sentence_timings
        ):
            raise HTTPException(500, "Video composition failed")

        # 8. Stream
        def iterfile():
            try:
                with open(output_path, "rb") as f:
                    yield from f
            finally:
                if os.path.exists(output_path):
                    try: os.unlink(output_path)
                    except: pass

        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename=video_{effect}.mp4"}
        )

    except Exception as e:
        print(f"ERROR: {e}")
        for f in image_paths + [audio_path, subtitle_path, output_path]:
            if os.path.exists(f):
                try: os.unlink(f)
                except: pass
        raise HTTPException(500, f"Video generation error: {str(e)}")

# --- Run ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
