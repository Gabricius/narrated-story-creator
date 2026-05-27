from PIL import Image, ImageDraw, ImageFont
import os
import random
import string
from kokoro import KPipeline
import soundfile as sf
from typing import List, Dict
import subprocess
import numpy as np
import re

CUDA = os.environ.get('CUDA', '0') == '1'

# ── Subtitle layout constants ──
SUBTITLE_POS_Y_V2 = int(1080 * 0.675)   # 729px — vertical center of v2 karaoke text
SUBTITLE_STRIP_TOP = 0.75               # fraction — subtitle strip starts at 75% height (v3/v4)
CHARACTER_SCALE_FACTOR = 0.8            # v4 — encolhe o personagem para dar respiro acima (base permanece ancorada em strip_y)

# ── V4 Subtitle Color Presets ──
# ASS color format: &HAABBGGRR (alpha, blue, green, red — hex)
# &H00 prefix = fully opaque
COLOR_PRESETS = [
    {"name": "yellow",  "current": "&H0000FFFF", "unspoken": "&H00FFFFFF", "spoken": "&H00808080"},
    {"name": "cyan",    "current": "&H00FFFF00", "unspoken": "&H00FFFFFF", "spoken": "&H00808080"},
    {"name": "orange",  "current": "&H000080FF", "unspoken": "&H00FFFFFF", "spoken": "&H00808080"},
    {"name": "magenta", "current": "&H00FF00FF", "unspoken": "&H00FFFFFF", "spoken": "&H00808080"},
    {"name": "red",     "current": "&H000000FF", "unspoken": "&H00FFFFFF", "spoken": "&H00808080"},
    {"name": "lime",    "current": "&H0000FF00", "unspoken": "&H00FFFFFF", "spoken": "&H00808080"},
    {"name": "pink",    "current": "&H00FF69B4", "unspoken": "&H00FFFFFF", "spoken": "&H00808080"},
    {"name": "gold",    "current": "&H0000D7FF", "unspoken": "&H00FFFFFF", "spoken": "&H00808080"},
]

def _render_dynamic_red_box(
    overlay,  # The main Image object to paste onto
    draw,  # The ImageDraw object for the overlay
    overlay_total_width, # Total width of the main overlay image
    box_y_start,  # Y-coordinate for the top of the box
    box_height_abs,  # Absolute height of the box
    display_name,  # Text to display in the box
    font_path,  # Path to the .ttf font file
    volume_icon_path,  # Path to the volume icon image
    base_color_rgb  # Tuple (R, G, B) for the box background color
):
    """
    Renders a dynamically sized box with a solid part, a gradient part,
    an icon, and text.
    """
    # A) Text dimensions for the display name in the box
    font_size = int(box_height_abs * 0.6)
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = font.getbbox(display_name)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1] # Full height of text
    text_y_offset = text_bbox[1] # Y offset for precise vertical centering
    
    # B) Volume icon dimensions for the box
    vol_icon_image = Image.open(volume_icon_path).convert("RGBA")
    
    icon_height = int(box_height_abs * 0.7) # Icon height 70% of box
    _icon_aspect_ratio = vol_icon_image.width / vol_icon_image.height
    icon_width = int(icon_height * _icon_aspect_ratio)
    vol_icon_resized = vol_icon_image.resize((icon_width, icon_height))

    # C) Define the "5% padding segment" width
    _padding_reference_width = int(overlay_total_width * 0.25) 
    padding_segment_amount = int(_padding_reference_width * 0.08)

    # D) Calculate the new dynamic box_width
    # Layout: [P] Icon [P] Text [P][P][P]  (P = padding_segment_amount)
    dynamic_box_width = (padding_segment_amount * 2) + icon_width + text_width + (padding_segment_amount * 7)
    dynamic_box_width = max(dynamic_box_width, padding_segment_amount * 5) 

    # Create the solid part (e.g., 60% of the box width for a 40% gradient)
    solid_width = int(dynamic_box_width * 0.7)
    draw.rectangle(
        [(0, box_y_start), (solid_width, box_y_start + box_height_abs)],
        fill=(*base_color_rgb, 255)  # Solid color with full opacity
    )
    
    # Create the gradient part (now 40% of the box width)
    gradient_width = dynamic_box_width - solid_width
    
    if gradient_width > 0:
        gradient_img = Image.new("RGBA", (gradient_width, box_height_abs+1), (0, 0, 0, 0))
        gradient_draw = ImageDraw.Draw(gradient_img)
        for x_grad in range(gradient_width):
            alpha = int(255 * (1 - (x_grad / gradient_width)))
            gradient_draw.line([(x_grad, 0), (x_grad, box_height_abs+100)], fill=(*base_color_rgb, alpha))
        overlay.paste(gradient_img, (solid_width, box_y_start), gradient_img)

    # Position volume icon and text within the dynamically-sized box
    icon_x_position = padding_segment_amount
    icon_y_position = int(box_y_start + (box_height_abs - icon_height) / 2)
    overlay.paste(vol_icon_resized, (icon_x_position, icon_y_position), vol_icon_resized)

    name_x_position = icon_x_position + icon_width + padding_segment_amount
    name_y_position = int(box_y_start + (box_height_abs - text_height) / 2 - text_y_offset)
    draw.text((name_x_position, name_y_position), display_name, font=font, fill="white")
    
    return solid_width

def create_overlay(
    person_image_path,
    volume_icon_path,
    display_name,
    output_path,
    base_color_rgb=(232, 14, 64),  # Original red color
    subtitle_background_color=(0, 0, 0, 99),  # Semi-transparent black
    font_path="assets/noto.ttf",  # Make sure to have a suitable .ttf font file available
    overlay_size=(1920, 1080)  # Set to 1920x1080 as per requirements
):
    # Base overlay image (transparent)
    overlay = Image.new("RGBA", overlay_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Calculate sizes and positions based on percentages
    width, height = overlay_size
    
    # Define layout percentages
    top_margin_percent = 0.10
    person_image_height_percent = 0.40
    red_box_height_percent = 0.10
    black_box_height_percent = 0.30
    # Bottom margin is implicitly 1.0 - sum of above = 0.10

    # Calculate absolute pixel values for heights
    top_margin_abs = int(height * top_margin_percent)
    person_image_height_abs = int(height * person_image_height_percent)
    red_box_height_abs = int(height * red_box_height_percent)
    black_box_height_abs = int(height * black_box_height_percent)

    # Calculate Y positions
    person_y_start = top_margin_abs
    red_box_y_start = person_y_start + person_image_height_abs
    box_y_start = red_box_y_start + red_box_height_abs # This is the semi-transparent black box
    
    # 1. Semi-transparent black box - NOW USING THE PARAMETER VALUE
    draw.rectangle(
        [(0, box_y_start), (width, box_y_start + black_box_height_abs)],
        fill=subtitle_background_color  # Using the parameter directly
    )
    
    # Call the new helper function to render the red box
    try:
        solid_width = _render_dynamic_red_box(
            overlay=overlay,
            draw=draw,
            overlay_total_width=width,
            box_y_start=red_box_y_start,
            box_height_abs=red_box_height_abs,
            display_name=display_name,
            font_path=font_path,
            volume_icon_path=volume_icon_path,
            base_color_rgb=base_color_rgb
        )
    except FileNotFoundError as e:
        print(f"ERROR: Asset file not found - {e}")
        print(f"Font path: {font_path}")
        print(f"Volume icon path: {volume_icon_path}")
        # List available files
        import os
        if os.path.exists("assets"):
            print(f"Available assets: {os.listdir('assets')}")
        raise Exception(f"Cannot open resource: {e}")
    except Exception as e:
        print(f"ERROR in _render_dynamic_red_box: {e}")
        raise

    # 3. Position the person image
    try:
        person_img = Image.open(person_image_path).convert("RGBA")
    except FileNotFoundError as e:
        print(f"ERROR: Person image not found at {person_image_path}")
        raise Exception(f"Cannot open person image: {e}")
    except Exception as e:
        print(f"ERROR opening person image: {e}")
        raise Exception(f"Cannot open person image resource: {e}")
    
    person_img_original_width, person_img_original_height = person_img.size
    person_width_abs = int(person_image_height_abs * person_img_original_width / person_img_original_height)
    
    person_x_start = 15
    
    # Resize and paste person image
    person_img_resized = person_img.resize((person_width_abs, person_image_height_abs))
    overlay.paste(person_img_resized, (person_x_start, person_y_start), person_img_resized)
    
    # Save the overlay
    overlay.save(output_path)
    print(f"Overlay saved to {output_path}")

def create_overlay_v3(
    person_image_path,
    output_path,
    subtitle_background_color=(0, 0, 0, 200),
    overlay_size=(1920, 1080)
):
    """V3 overlay: character image + subtitle black bar only. No colored band, no name, no icon."""
    overlay = Image.new("RGBA", overlay_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = overlay_size

    person_image_height_percent = 0.40
    black_box_height_percent = 0.30
    bottom_margin_percent = 0.10

    person_image_height_abs = int(height * person_image_height_percent)
    black_box_height_abs = int(height * black_box_height_percent)
    bottom_margin_abs = int(height * bottom_margin_percent)

    # Black bar at the same bottom position as v2
    box_y_start = height - bottom_margin_abs - black_box_height_abs
    draw.rectangle(
        [(0, box_y_start), (width, box_y_start + black_box_height_abs)],
        fill=subtitle_background_color
    )

    # Person image: bottom edge aligned with the top of the black bar
    person_y_start = box_y_start - person_image_height_abs

    try:
        person_img = Image.open(person_image_path).convert("RGBA")
    except FileNotFoundError as e:
        raise Exception(f"Cannot open person image: {e}")
    except Exception as e:
        raise Exception(f"Cannot open person image resource: {e}")

    person_img_original_width, person_img_original_height = person_img.size
    person_width_abs = int(person_image_height_abs * person_img_original_width / person_img_original_height)

    person_img_resized = person_img.resize((person_width_abs, person_image_height_abs))
    overlay.paste(person_img_resized, (15, person_y_start), person_img_resized)

    overlay.save(output_path)
    print(f"V3 overlay saved to {output_path}")

def create_overlay_v4(
    person_image_path,
    position,
    output_path,
    overlay_size=(1920, 1080),
    subtitle_background_color=(0, 0, 0, 180),
):
    """V4 overlay: background-removed character + subtitle strip (bottom 25%).

    position: 'left' | 'center' | 'right'
      - left:   character's leftmost pixel at canvas x=0
      - center: character horizontally centered
      - right:  character's rightmost pixel at canvas right edge

    Vertical: character base anchored at y=strip_y (top of subtitle strip).
    Subtitle strip occupies bottom 25% of canvas (y=810 to 1080 for 1080p).
    """
    width, height = overlay_size
    overlay = Image.new("RGBA", overlay_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Subtitle strip: bottom 25%
    strip_y = int(height * SUBTITLE_STRIP_TOP)  # 810px for 1080p

    draw.rectangle([(0, strip_y), (width, height)], fill=subtitle_background_color)

    try:
        person_img = Image.open(person_image_path).convert("RGBA")
    except FileNotFoundError as e:
        raise Exception(f"Cannot open person image: {e}")
    except Exception as e:
        raise Exception(f"Cannot open person image resource: {e}")

    orig_w, orig_h = person_img.size

    # Scale character to fill available height above strip
    max_char_height = strip_y
    char_height = int(min(orig_h, max_char_height) * CHARACTER_SCALE_FACTOR)
    char_width = int(char_height * orig_w / orig_h)

    person_img_resized = person_img.resize((char_width, char_height), Image.LANCZOS)

    # Horizontal position
    if position == "left":
        char_x = 0
    elif position == "right":
        char_x = width - char_width
    else:  # center
        char_x = (width - char_width) // 2

    # Vertical: base of character touches top of subtitle strip
    char_y = strip_y - char_height

    overlay.paste(person_img_resized, (char_x, char_y), person_img_resized)
    overlay.save(output_path)
    print(f"V4 overlay saved | position={position} | char={char_width}×{char_height} | x={char_x}")


def create_tts_english(text, output_path, lang_code, voice):
    pipeline = KPipeline(
        lang_code=lang_code,
    )

    generator = pipeline(text, voice=voice)

    captions = []
    audio_data = []
    full_audio_length = 0
    for i, result in enumerate(generator):
        data = result.audio
        audio_length = len(data) / 24000
        audio_data.append(data)
        if result.tokens:
            tokens = result.tokens
            for t in tokens:
                if t.start_ts is None or t.end_ts is None:
                    if captions:
                        captions[-1]["text"] += t.text
                        captions[-1]["end_ts"] = full_audio_length + audio_length
                    continue
                try:
                    captions.append({
                        "text": t.text,
                        "start_ts": full_audio_length + t.start_ts,
                        "end_ts": full_audio_length + t.end_ts
                    })
                except Exception as e:
                    print(f"Processing error in Kokoro, there's a character that is not supported. text: {t.text}")
                    raise ValueError(f"Error processing token: {t}, Error: {e}")
        full_audio_length += audio_length
    
    audio_data = np.concatenate(audio_data)
    sf.write(output_path, audio_data, 24000)
    return captions, full_audio_length

def break_text_into_sentences(text, lang_code) -> List[str]:
    """
    Advanced sentence splitting with better handling of abbreviations and edge cases.
    """
    if not text or not text.strip():
        return []
    
    # Language-specific sentence boundary patterns
    patterns = {
        'a': r'(?<=[.!?])\s+(?=[A-Z])',     # English
        'e': r'(?<=[.!?¿¡])\s+(?=[A-ZÁÉÍÓÚÑÜ])',  # Spanish
        'f': r'(?<=[.!?])\s+(?=[A-ZÁÀÂÄÇÉÈÊËÏÎÔÖÙÛÜŸ])',  # French
        'h': r'(?<=[।!?])\s+',           # Hindi: Split after devanagari danda
        'i': r'(?<=[.!?])\s+(?=[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß])',  # Italian
        'p': r'(?<=[.!?])\s+(?=[A-ZÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ])',  # Portuguese
        'z': r'(?<=[。！？])',            # Chinese: Split after Chinese punctuation
    }
    
    # Common abbreviations that shouldn't trigger sentence breaks
    abbreviations = {
        'e': {'Sr.', 'Sra.', 'Dr.', 'Dra.', 'Prof.', 'etc.', 'pág.', 'art.', 'núm.', 'cap.', 'vol.'},  # Spanish
        'f': {'M.', 'Mme.', 'Dr.', 'Prof.', 'etc.', 'art.', 'p.', 'vol.', 'ch.', 'fig.', 'n°'},  # French
        'h': {'श्री', 'श्रीमती', 'डॉ.', 'प्रो.', 'etc.', 'पृ.', 'अध.'},  # Hindi
        'i': {'Sig.', 'Sig.ra', 'Dr.', 'Prof.', 'ecc.', 'pag.', 'art.', 'n.', 'vol.', 'cap.', 'fig.'},  # Italian
        'p': {'Sr.', 'Sra.', 'Dr.', 'Dra.', 'Prof.', 'etc.', 'pág.', 'art.', 'n.º', 'vol.', 'cap.'},  # Portuguese
        'z': {'先生', '女士', '博士', '教授', '等等', '第', '页', '章'}  # Chinese
    }
    
    abbrevs = abbreviations.get(lang_code, set())
    
    # Protect abbreviations by temporarily replacing them
    protected_text = text
    replacements = {}
    for i, abbrev in enumerate(abbrevs):
        placeholder = f"__ABBREV_{i}__"
        protected_text = protected_text.replace(abbrev, placeholder)
        replacements[placeholder] = abbrev
    
    # Apply the regex splitting
    pattern = patterns.get(lang_code, patterns['a'])
    sentences = re.split(pattern, protected_text.strip())
    
    # Restore abbreviations and clean up
    restored_sentences = []
    for sentence in sentences:
        for placeholder, original in replacements.items():
            sentence = sentence.replace(placeholder, original)
        sentence = sentence.strip()
        if sentence:
            restored_sentences.append(sentence)
    
    return restored_sentences if restored_sentences else [text.strip()]

def create_tts_international(text, output_path, lang_code, voice):
    sentences = break_text_into_sentences(text, lang_code)

    # generate the audio for each sentence
    audio_data = []
    captions = []
    full_audio_length = 0
    pipeline = KPipeline(
        lang_code=lang_code,
    )
    for sentence in sentences:
        generator = pipeline(sentence, voice=voice)
        
        for i, result in enumerate(generator):
            data = result.audio
            audio_length = len(data) / 24000
            audio_data.append(data)
            # since there are no tokens, we can just use the sentence as the text
            captions.append({
                "text": sentence,
                "start_ts": full_audio_length,
                "end_ts": full_audio_length + audio_length
            })
            full_audio_length += audio_length
    audio_data = np.concatenate(audio_data)
    sf.write(output_path, audio_data, 24000)
    return captions, full_audio_length

def is_punctuation(text):
    return text in string.punctuation

def create_subtitle_segments_english(captions: List[Dict], max_length=80, lines=2):
    """
    Breaks up the captions into segments of max_length characters
    on two lines and merge punctuation with the last word
    """
    
    if not captions:
        return []
    
    segments = []
    current_segment_texts = ["" for _ in range(lines)]
    current_line = 0
    segment_start_ts = captions[0]["start_ts"]
    segment_end_ts = captions[0]["end_ts"]
    
    for caption in captions:
        text = caption["text"]
        start_ts = caption["start_ts"]
        end_ts = caption["end_ts"]
        
        # Update the segment end timestamp
        segment_end_ts = end_ts
        
        # If the caption is a punctuation, merge it with the current line
        if is_punctuation(text):
            if current_line < lines and current_segment_texts[current_line]:
                current_segment_texts[current_line] += text
            continue
        
        # If the line is too long, move to the next one
        if current_line < lines and len(current_segment_texts[current_line] + text) > max_length:
            current_line += 1
        
        # If we've filled all lines, save the current segment and start a new one
        if current_line >= lines:
            segments.append({
                "text": current_segment_texts,
                "start_ts": segment_start_ts,
                "end_ts": segment_end_ts
            })
            
            # Reset for next segment
            current_segment_texts = ["" for _ in range(lines)]
            current_line = 0
            # Add a small gap (0.05s) between segments to prevent overlap
            segment_start_ts = start_ts + 0.05
        
        # Add the text to the current segment
        if current_line < lines:
            current_segment_texts[current_line] += " " if current_segment_texts[current_line] else ""
            current_segment_texts[current_line] += text
    
    # Add the last segment if there's any content
    if any(current_segment_texts):
        segments.append({
            "text": current_segment_texts,
            "start_ts": segment_start_ts,
            "end_ts": segment_end_ts
        })
    
    # Post-processing: make subtitles persistent (no gaps between segments)
    for i in range(len(segments) - 1):
        next_start = segments[i+1]["start_ts"]
        # Extend this segment's end to the next segment's start (fills gap)
        if next_start > segments[i]["end_ts"]:
            segments[i]["end_ts"] = next_start
        # Fix overlaps
        elif segments[i]["end_ts"] >= next_start:
            segments[i]["end_ts"] = next_start - 0.01
    
    return segments 

def create_subtitle_segments_international(captions: List[Dict], max_length=80, lines=2):
    """
    Breaks up international captions (full sentences) into smaller segments that fit
    within max_length characters per line, with proper timing distribution.
    
    Handles both space-delimited languages like English and character-based languages like Chinese.
    
    Args:
        captions: List of caption dictionaries with text, start_ts, and end_ts
        max_length: Maximum number of characters per line
        lines: Number of lines per segment
        
    Returns:
        List of subtitle segments
    """
    if not captions:
        return []
    
    segments = []
    
    for caption in captions:
        text = caption["text"].strip()
        start_ts = caption["start_ts"]
        end_ts = caption["end_ts"]
        duration = end_ts - start_ts
        
        # Check if text is using Chinese/Japanese/Korean characters (CJK)
        # For CJK, we'll split by characters rather than words
        is_cjk = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        parts = []
        if is_cjk:
            # For CJK languages, process character by character
            current_part = ""
            for char in text:
                if len(current_part + char) > max_length:
                    parts.append(current_part)
                    current_part = char
                else:
                    current_part += char
            
            # Add the last part if not empty
            if current_part:
                parts.append(current_part)
        else:
            # Original word-based splitting for languages with spaces
            words = text.split()
            current_part = ""
            
            for word in words:
                # If adding this word would exceed max_length, start a new part
                if len(current_part + " " + word) > max_length and current_part:
                    parts.append(current_part.strip())
                    current_part = word
                else:
                    # Add space if not the first word in the part
                    if current_part:
                        current_part += " "
                    current_part += word
            
            # Add the last part if not empty
            if current_part:
                parts.append(current_part.strip())
        
        # Group parts into segments with 'lines' number of lines per segment
        segment_parts = []
        for i in range(0, len(parts), lines):
            segment_parts.append(parts[i:i+lines])
        
        # Calculate time proportionally based on segment text length
        total_chars = sum(len(''.join(part_group)) for part_group in segment_parts)
        
        current_time = start_ts
        for i, part_group in enumerate(segment_parts):
            # Get character count for this segment group
            segment_chars = len(''.join(part_group))
            
            # Calculate time proportionally, but ensure at least a minimum duration
            if total_chars > 0:
                segment_duration = (segment_chars / total_chars) * duration
                segment_duration = max(segment_duration, 0.5)  # Ensure minimum duration of 0.5s
            else:
                segment_duration = duration / len(segment_parts)
            
            segment_start = current_time
            segment_end = segment_start + segment_duration
            
            # Move current time forward for next segment
            current_time = segment_end
            
            # Create segment with proper text array format for the subtitle renderer
            segment_text = part_group + [""] * (lines - len(part_group))
            
            segments.append({
                "text": segment_text,
                "start_ts": segment_start,
                "end_ts": segment_end
            })
    
    # Ensure no overlaps between segments by adjusting end times if needed
    for i in range(len(segments) - 1):
        if segments[i]["end_ts"] >= segments[i+1]["start_ts"]:
            segments[i]["end_ts"] = segments[i+1]["start_ts"] - 0.05
    
    return segments

def create_subtitle(segments, output_path, font_size=24, font_color="&H00FFFFFF"):
    # Create the .ass subtitle file with headers
    ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},{font_color},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,0,0,8,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(font_size=font_size, font_color=font_color)
    
    # Position at 67.5% from the top is achieved using alignment 8 (center) and \pos tag
    pos_y = SUBTITLE_POS_Y_V2
    
    # Process each segment and add to the subtitle file
    for segment in segments:
        start_time = format_time(segment["start_ts"])
        end_time = format_time(segment["end_ts"])
        
        # Create text with line breaks
        text_lines = segment["text"]
        formatted_text = ""
        for i, line in enumerate(text_lines):
            if line:  # Only add non-empty lines
                if i > 0:  # Add line break if not the first line
                    formatted_text += "\\N"
                formatted_text += line
        
        # Position at 65% from top, horizontally centered
        formatted_text = f"{{\\pos(960,{pos_y})}}" + formatted_text
        
        # Add the dialogue line to the subtitle content
        ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{formatted_text}\n"
    
    # Write the subtitle file
    output_file = f"{output_path}"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    print(f"Subtitle file created: {output_file}")
    return output_file

def _create_karaoke_ass(
    word_captions: list,
    output_path: str,
    colors: dict,
    pos_y: int,
    alignment: int,
    font_size: int = 80,
    max_chars_per_line: int = 40,
    max_lines: int = 2,
    outline: int = 0,
    back_colour: str = "&H00000000",
) -> str:
    """Core karaoke ASS builder shared by v2 and v4. Returns output_path.

    colors: dict with keys 'unspoken', 'current', 'spoken' (ASS &HAABBGGRR hex strings).
    """
    COLOR_UNSPOKEN = colors["unspoken"]
    COLOR_CURRENT  = colors["current"]
    COLOR_SPOKEN   = colors["spoken"]

    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},{COLOR_UNSPOKEN},&H000000FF,&H00000000,{back_colour},-1,0,0,0,100,100,0,0,1,{outline},0,{alignment},20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # ── Normalize caption keys ──
    normalized_captions = []
    for cap in word_captions:
        if isinstance(cap, dict):
            s = cap.get("start_ts", cap.get("start", cap.get("s", 0)))
            e = cap.get("end_ts",   cap.get("end",   cap.get("e", 0)))
            t = cap.get("text",     cap.get("word",  cap.get("w", "")))
            normalized_captions.append({"start_ts": float(s), "end_ts": float(e), "text": str(t)})
        elif hasattr(cap, "start_ts"):
            normalized_captions.append({"start_ts": float(cap.start_ts), "end_ts": float(cap.end_ts),
                                         "text": str(getattr(cap, "text", getattr(cap, "word", "")))})
        elif hasattr(cap, "start"):
            normalized_captions.append({"start_ts": float(cap.start), "end_ts": float(cap.end),
                                         "text": str(getattr(cap, "text", getattr(cap, "word", "")))})
        else:
            print(f"[KARAOKE] WARNING: Unknown caption format: {type(cap).__name__} = {repr(cap)[:100]}")
    word_captions = normalized_captions

    # ── Merge punctuation with previous word ──
    merged = []
    for cap in word_captions:
        if cap["text"] in string.punctuation and merged:
            merged[-1]["text"] += cap["text"]
            merged[-1]["end_ts"] = cap["end_ts"]
        else:
            merged.append(cap.copy())

    # ── Group into segments (max_lines × max_chars_per_line) ──
    segments = []
    cur_seg = []
    cur_line_words = []
    cur_line_len = 0
    cur_line_num = 0

    for cap in merged:
        word = cap["text"]
        potential = cur_line_len + (1 if cur_line_words else 0) + len(word)
        if potential > max_chars_per_line and cur_line_words:
            cur_seg.extend(cur_line_words)
            cur_line_words = []
            cur_line_len = 0
            cur_line_num += 1
            if cur_line_num >= max_lines:
                segments.append(cur_seg)
                cur_seg = []
                cur_line_num = 0
        cur_line_words.append(cap)
        cur_line_len += (1 if cur_line_len > 0 else 0) + len(word)

    if cur_line_words:
        cur_seg.extend(cur_line_words)
    if cur_seg:
        segments.append(cur_seg)

    # ── Extend word end times to eliminate gaps (persistent subtitles) ──
    for seg_idx, seg_words in enumerate(segments):
        if not seg_words:
            continue
        for w_idx in range(len(seg_words) - 1):
            nxt = seg_words[w_idx + 1]["start_ts"]
            if nxt > seg_words[w_idx]["end_ts"]:
                seg_words[w_idx]["end_ts"] = nxt
        if seg_idx < len(segments) - 1:
            nxt_seg = segments[seg_idx + 1]
            if nxt_seg:
                nxt_start = nxt_seg[0]["start_ts"]
                last = seg_words[-1]
                if nxt_start > last["end_ts"]:
                    last["end_ts"] = nxt_start

    segment_end_times = []

    for seg_idx, seg_words in enumerate(segments):
        if not seg_words:
            continue

        seg_start = seg_words[0]["start_ts"]
        seg_end   = seg_words[-1]["end_ts"]

        if seg_idx > 0 and segment_end_times:
            prev_end = segment_end_times[-1]
            if seg_start < prev_end:
                shift = prev_end - seg_words[0]["start_ts"]
                for wd in seg_words:
                    wd["start_ts"] += shift
                    wd["end_ts"]   += shift
                seg_start = seg_words[0]["start_ts"]
                seg_end   = seg_words[-1]["end_ts"]

        # Break into display lines
        lines = []
        cur_line = []
        cur_len  = 0
        for wd in seg_words:
            pot = cur_len + (1 if cur_line else 0) + len(wd["text"])
            if pot > max_chars_per_line and cur_line:
                lines.append(cur_line)
                cur_line = []
                cur_len  = 0
            cur_line.append(wd)
            cur_len += (1 if cur_len > 0 else 0) + len(wd["text"])
        if cur_line:
            lines.append(cur_line)
        lines = lines[:max_lines]

        # O(n) position lookup: avoids O(n²) list.index() inside nested loops
        word_pos = {id(wd): i for i, wd in enumerate(seg_words)}

        for word_idx, cur_word in enumerate(seg_words):
            ev_start = cur_word["start_ts"]
            ev_end   = seg_words[word_idx + 1]["start_ts"] if word_idx < len(seg_words) - 1 else seg_end
            if ev_end <= ev_start:
                ev_end = ev_start + 0.05

            colored_lines = []
            for line_words in lines:
                colored_line = ""
                for wd in line_words:
                    idx_in_seg = word_pos[id(wd)]
                    if idx_in_seg < word_idx:
                        colored_line += f"{{\\c{COLOR_SPOKEN}}}{wd['text']}{{\\c}}"
                    elif idx_in_seg == word_idx:
                        colored_line += f"{{\\c{COLOR_CURRENT}}}{wd['text']}{{\\c}}"
                    else:
                        colored_line += f"{{\\c{COLOR_UNSPOKEN}}}{wd['text']}{{\\c}}"
                    if wd != line_words[-1]:
                        colored_line += " "
                colored_lines.append(colored_line)

            formatted = "\\N".join(colored_lines)
            formatted  = f"{{\\pos(960,{pos_y})}}" + formatted
            ass_content += (f"Dialogue: 0,{format_time(ev_start)},{format_time(ev_end)},"
                            f"Default,,0,0,0,,{formatted}\n")

        segment_end_times.append(seg_end)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    return output_path


def create_subtitle_v2_karaoke(word_captions, output_path, font_size=24, max_chars_per_line=40, max_lines=2):
    """Karaoke subtitle for v2: green current word, white unspoken, gray spoken."""
    colors = {"unspoken": "&H00FFFFFF", "current": "&H0000FF00", "spoken": "&H00808080"}
    result = _create_karaoke_ass(
        word_captions, output_path, colors,
        pos_y=SUBTITLE_POS_Y_V2, alignment=8,
        font_size=font_size, max_chars_per_line=max_chars_per_line, max_lines=max_lines,
        outline=2, back_colour="&H80000000",
    )
    print(f"V2 Karaoke subtitle file created: {output_path}")
    return result


def create_subtitle_v4_karaoke(word_captions, output_path, color_preset, font_size=80,
                                max_chars_per_line=40, max_lines=2):
    """V4 karaoke subtitle: colored word-by-word highlight over the bottom-25% subtitle strip.

    color_preset: dict from COLOR_PRESETS, e.g.:
        {"name": "yellow", "current": "&H0000FFFF", "unspoken": "&H00FFFFFF", "spoken": "&H00808080"}
    """
    colors = {
        "unspoken": color_preset["unspoken"],
        "current":  color_preset["current"],
        "spoken":   color_preset["spoken"],
    }
    result = _create_karaoke_ass(
        word_captions, output_path, colors,
        pos_y=945, alignment=5,
        font_size=font_size, max_chars_per_line=max_chars_per_line, max_lines=max_lines,
        outline=0, back_colour="&H00000000",
    )
    print(f"V4 karaoke subtitle created: {output_path} | preset={color_preset['name']}")
    return result


def format_time(seconds):
    """
    Convert seconds to ASS time format (H:MM:SS.cc)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

def render_video(sound_path, subtitle_path, overlay_path, audio_length, bg_video_path, output_path,
                 subscribe_overlay_path=None, subscribe_first_at=30, subscribe_interval=180,
                 progress_callback=None):
    """
    Renders a video with the given sound, subtitle, overlay image, and a background video.
    Uses concat demuxer for seamless background looping (no stream_loop stutter).
    
    Optional subscribe_overlay_path: a green-screen video overlaid periodically.
    First appearance at subscribe_first_at seconds, then every subscribe_interval seconds.
    Green is removed via chromakey.
    """
    
    try:
        # ── Pre-loop background using concat demuxer ──
        # This avoids stream_loop's known freeze/stutter at loop boundaries
        bg_duration = 0
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", bg_video_path],
                capture_output=True, text=True, timeout=10
            )
            if probe.returncode == 0:
                bg_duration = float(probe.stdout.strip())
        except:
            bg_duration = 30  # fallback assumption
        
        if bg_duration <= 0:
            bg_duration = 30
        
        # Calculate loops needed (add 1 extra for safety)
        loops_needed = int(audio_length / bg_duration) + 2
        
        # Create concat list file
        concat_list_path = os.path.join(os.path.dirname(output_path), "bg_concat_list.txt")
        if not os.path.dirname(output_path):
            concat_list_path = os.path.join(os.path.dirname(bg_video_path), "bg_concat_list.txt")
        
        with open(concat_list_path, 'w') as f:
            for _ in range(loops_needed):
                # Use absolute path and escape single quotes
                safe_path = os.path.abspath(bg_video_path).replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")
        
        print(f"[BG] Loop: {bg_duration:.1f}s × {loops_needed} = {bg_duration * loops_needed:.0f}s (need {audio_length:.0f}s)")
        
        # ── Build FFmpeg command ──
        cmd = ['ffmpeg', '-y']
        
        if CUDA:
            cmd.extend(['-hwaccel', 'cuda'])
        
        # Input 0: background video (looped via concat)
        cmd.extend([
            '-f', 'concat', '-safe', '0',
            '-t', str(audio_length),
            '-i', concat_list_path,
        ])
        
        # Input 1: person/channel overlay image
        cmd.extend(['-i', overlay_path])
        
        # Input 2: audio
        cmd.extend(['-i', sound_path])
        
        # Input 3 (optional): subscribe overlay video with green screen
        has_sub_overlay = subscribe_overlay_path and os.path.exists(subscribe_overlay_path)
        if has_sub_overlay:
            # Probe subscribe overlay duration
            sub_dur = 5.0  # fallback
            try:
                probe_sub = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", subscribe_overlay_path],
                    capture_output=True, text=True, timeout=10
                )
                if probe_sub.returncode == 0:
                    sub_dur = float(probe_sub.stdout.strip())
            except:
                pass
            
            print(f"[SUB-OVERLAY] Duration: {sub_dur:.1f}s | First at: {subscribe_first_at}s | Every: {subscribe_interval}s")
            
            # Align interval to nearest multiple of sub_dur so overlay restarts at frame 0
            # e.g. sub_dur=5, interval=180: 180%5=0 ✓
            # e.g. sub_dur=7, interval=180: 180%7=5 → adjust to 182 (26×7)
            if sub_dur > 0 and (subscribe_interval % sub_dur) > 0.01:
                adjusted = round(subscribe_interval / sub_dur) * sub_dur
                if adjusted < sub_dur:
                    adjusted = sub_dur
                print(f"[SUB-OVERLAY] Adjusted interval {subscribe_interval}→{adjusted:.0f}s for frame alignment")
                subscribe_interval = adjusted
            
            # itsoffset = first_at: overlay's frame 0 lands at main video t=first_at
            # At t=first_at → overlay time 0 → frame 0 ✓
            # At t=first_at+interval → overlay time=interval → interval%sub_dur=0 → frame 0 ✓
            itsoffset = subscribe_first_at
            
            total_sub_needed = audio_length - subscribe_first_at + sub_dur
            sub_loops = max(1, int(total_sub_needed / sub_dur) + 2)
            
            print(f"[SUB-OVERLAY] itsoffset={itsoffset}s, loops={sub_loops}, interval={subscribe_interval}s")
            
            cmd.extend([
                '-itsoffset', str(itsoffset),
                '-stream_loop', str(sub_loops),
                '-i', subscribe_overlay_path,
            ])
            
            # ── Filter with subscribe overlay ──
            # Chromakey: remove green (#00FF00), similarity=0.3, blend=0.15
            # Enable: show at subscribe_first_at, then every subscribe_interval for sub_dur seconds
            esc = "\\,"  # escaped comma for FFmpeg expression
            enable_expr = f"gte(t{esc}{subscribe_first_at})*lt(mod(t-{subscribe_first_at}{esc}{subscribe_interval}){esc}{sub_dur})"
            
            filter_complex = (
                f"[0:v]scale=1920:1080,gblur=sigma=2[scaled];"
                f"[scaled][1:v]overlay=format=auto[with_person];"
                f"[3:v]colorkey=0x00FF00:0.3:0.15,format=rgba[subkey];"
                f"[with_person][subkey]overlay=(W-w)/2:(H-h)/2:format=auto:enable='{enable_expr}'[with_sub];"
                f"[with_sub]subtitles={subtitle_path}[v]"
            )
            audio_map = '2:a'
        else:
            # ── Filter without subscribe overlay ──
            filter_complex = f"[0:v]scale=1920:1080,gblur=sigma=2[scaled];[scaled][1:v]overlay=format=auto[overlaid];[overlaid]subtitles={subtitle_path}[v]"
            audio_map = '2:a'
        
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[v]',
            '-map', audio_map,
            '-c:v', 'h264_nvenc' if CUDA else 'libx264',
            '-preset', 'fast' if CUDA else 'ultrafast',
            '-crf', '25',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-t', str(audio_length),
            output_path
        ])
        
        # Use Popen instead of run to capture output in real-time
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            text=True
        )
        
        # Process the output line by line as it becomes available
        last_reported_pct = -1
        for line in process.stderr:
            # Extract time information for progress tracking
            if "time=" in line and "speed=" in line:
                try:
                    # Extract the time information
                    time_str = line.split("time=")[1].split(" ")[0]
                    # Convert HH:MM:SS.MS format to seconds
                    h, m, s = time_str.split(":")
                    seconds = float(h) * 3600 + float(m) * 60 + float(s)

                    # Calculate progress percentage
                    progress = min(100, (seconds / audio_length) * 100)
                    print(f"Processing: {progress:.2f}% complete (Time: {time_str} / Total: {format_time(audio_length)})")
                    pct_int = int(progress)
                    if progress_callback and pct_int != last_reported_pct:
                        last_reported_pct = pct_int
                        try:
                            progress_callback(pct_int)
                        except Exception:
                            pass
                except (ValueError, IndexError) as e:
                    # If parsing fails, just print the line
                    print(f"ffmpeg: {line.strip()}")
            elif any(keyword in line for keyword in [
                # Skip initialization information
                "ffmpeg version", "built with", "configuration:", "libav",
                "Input #", "Metadata:", "Duration:", "Stream #",
                "Press [q]", "Output #", "Stream mapping:",
                
                # Skip processing details
                "frame=", "fps=", 
                "[libx264", "kb/s:", "Qavg:", 
                "video:", "audio:", "subtitle:", 
                "frame I:", "frame P:", "mb I", "mb P",
                "coded y,", "i16 v,h,dc,p:", "i8c dc,h,v,p:",
                "compatible_brands:", "encoder", "Side data:",
                "libswscale", "libswresample", "libpostproc",
                
                # Fix the missing commas in this line
                "ffmpeg: libswscale", "ffmpeg: libswresample", "ffmpeg: libpostproc"
            ]):
                # Skip all technical output lines
                pass
            else:
                # Only print important messages (like errors and warnings)
                # that don't match any of the filtered patterns
                if not line.strip() or line.strip().startswith('['):
                    continue
                
                # Skip header lines that describe inputs
                if ":" in line and any(header in line for header in 
                                      ["major_brand", "minor_version", "creation_time", 
                                       "handler_name", "vendor_id", "Duration", "bitrate"]):
                    continue
                
                print(f"ffmpeg: {line.strip()}")
        
        # Wait for the process to complete and check the return code
        return_code = process.wait()
        
        # Cleanup concat list
        try:
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)
        except:
            pass
        
        if return_code != 0:
            raise RuntimeError(f"FFmpeg exited with code {return_code}")

        return True

    except Exception as e:
        print(f"Error during video rendering: {e}")
        # Cleanup on error too
        try:
            if 'concat_list_path' in dir() and os.path.exists(concat_list_path):
                os.remove(concat_list_path)
        except:
            pass
        raise


def render_video_v4(bg_paths, overlay_path, sound_path, subtitle_path, output_path,
                    audio_length, effect_paths=None, effect_layers_resolved=None,
                    progress_callback=None,
                    intro_video_paths=None, overlay_during_intro=True,
                    subtitle_during_intro=True):
    """V4 video render: multiple background clips + optional particle effect layers.

    bg_paths:              List of local .mp4 paths used as background (concatenated and cycled).
    effect_layers_resolved: List of dicts: {local_path, opacity, blend_mode, label}.
                           Supported blend_mode values: colorkey_black, screen, chromakey_green, add.
    effect_paths:          Legacy fallback — list of local .mp4 paths (applied as screen blend).

    intro_video_paths:     (2026-05-20 — Intro Cinematográfico IA) Lista opcional de paths
                           de clips MP4 gerados pelo Flow (Veo 3.1) para servir como abertura
                           do vídeo. São prepended à lista de bg (cobrem os primeiros segundos)
                           e o TTS continua narrando desde t=0 normalmente.
    overlay_during_intro:  Se False, o overlay do personagem (PNG bg-removed) só aparece
                           após o término do intro (útil quando o personagem já está visível
                           NO vídeo IA — evita duplicar).
    subtitle_during_intro: Se False, a legenda karaoke fica oculta durante o intro. NOTA:
                           feature ainda não implementada via filter_complex — fase 1.5.
                           Hoje o subtitle continua visível independente desse flag (loga warn).
    """
    # Normalise to effect_layers_resolved (prefer new, fall back to legacy)
    if effect_layers_resolved is None and effect_paths:
        effect_layers_resolved = [
            {'local_path': p, 'opacity': round(random.uniform(0.40, 0.70), 2),
             'blend_mode': 'screen', 'label': f'legacy-{i}'}
            for i, p in enumerate(effect_paths)
        ]
    effect_layers_resolved = effect_layers_resolved or []
    intro_video_paths = intro_video_paths or []
    concat_list_path = None
    effect_concat_paths = []

    # ── Pre-flight: calcular duração total do intro IA (se houver) ──
    intro_total_duration_s = 0.0
    for ip in intro_video_paths:
        try:
            probe_i = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", ip],
                capture_output=True, text=True, timeout=10,
            )
            d = float(probe_i.stdout.strip()) if probe_i.returncode == 0 else 8.0
        except Exception:
            d = 8.0
        intro_total_duration_s += max(d, 1.0)
    if intro_video_paths:
        print(f"[INTRO-V4] {len(intro_video_paths)} intro clip(s) totalling "
              f"{intro_total_duration_s:.1f}s (overlay_during_intro={overlay_during_intro}, "
              f"subtitle_during_intro={subtitle_during_intro})")
        if not subtitle_during_intro:
            print("[INTRO-V4] WARN: subtitle_during_intro=False not yet enforced — "
                  "subtitle remains visible. Will be implemented in phase 1.5.")

    try:
        # ── 1. Build background concat list (all clips, cycled to cover audio_length) ──
        total_bg_dur = 0.0
        for p in bg_paths:
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", p],
                    capture_output=True, text=True, timeout=10,
                )
                dur = float(probe.stdout.strip()) if probe.returncode == 0 else 30.0
            except Exception:
                dur = 30.0
            total_bg_dur += max(dur, 1.0)

        if total_bg_dur <= 0:
            total_bg_dur = 30.0

        # Os intros ocupam os primeiros N segundos; o bg precisa cobrir o restante.
        bg_duration_needed = max(audio_length - intro_total_duration_s, 1.0)
        cycles_needed = int(bg_duration_needed / total_bg_dur) + 2
        concat_list_path = os.path.join(os.path.dirname(output_path), "bg_v4_concat.txt")

        with open(concat_list_path, "w") as f:
            # 2026-05-20: intros IA vêm PRIMEIRO (ordem preservada). Depois os clips de bg
            # ciclam normalmente. concat demuxer aceita codec/resolução iguais (1920x1080 H.264);
            # Veo 3.1 entrega exatamente esse formato, então não é necessário re-encode.
            for ip in intro_video_paths:
                safe_intro = os.path.abspath(ip).replace("'", "'\\''")
                f.write(f"file '{safe_intro}'\n")
            for _ in range(cycles_needed):
                for p in bg_paths:
                    safe = os.path.abspath(p).replace("'", "'\\''")
                    f.write(f"file '{safe}'\n")

        print(f"[BG-V4] {len(bg_paths)} clip(s) × {cycles_needed} cycles = "
              f"{total_bg_dur * cycles_needed:.0f}s + {intro_total_duration_s:.0f}s intro "
              f"(need {audio_length:.0f}s)")

        # ── 2. Build FFmpeg inputs ──
        cmd = ["ffmpeg", "-y"]
        if CUDA:
            cmd.extend(["-hwaccel", "cuda"])

        # Input 0: background (looped concat)
        cmd.extend(["-f", "concat", "-safe", "0",
                    "-t", str(audio_length), "-i", concat_list_path])

        # Input 1: overlay PNG — loop para manter visivel durante todo o video
        cmd.extend(["-loop", "1", "-i", overlay_path])

        # Input 2: audio
        cmd.extend(["-i", sound_path])

        # Inputs 3..N: effect layer videos (each individually looped)
        for i, layer in enumerate(effect_layers_resolved):
            ep = layer['local_path']
            try:
                probe_e = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", ep],
                    capture_output=True, text=True, timeout=10,
                )
                e_dur = float(probe_e.stdout.strip()) if probe_e.returncode == 0 else 10.0
            except Exception:
                e_dur = 10.0
            e_dur = max(e_dur, 1.0)
            e_loops = int(audio_length / e_dur) + 2

            e_concat = os.path.join(os.path.dirname(output_path), f"effect_{i}_concat.txt")
            effect_concat_paths.append(e_concat)

            with open(e_concat, "w") as f:
                for _ in range(e_loops):
                    safe_ep = os.path.abspath(ep).replace("'", "'\\''")
                    f.write(f"file '{safe_ep}'\n")

            cmd.extend(["-f", "concat", "-safe", "0",
                        "-t", str(audio_length), "-i", e_concat])
            print(f"[EFFECT-V4] layer {i} '{layer.get('label', '')}': {e_dur:.1f}s × {e_loops} loops | blend={layer['blend_mode']} opacity={layer['opacity']}")

        # ── 3. Build filter_complex ──
        filter_parts = []
        # 2026-05-22: Overlay REC durante a intro IA — esconde a marca d'água do VEO.
        # Replica o padrão do Rede Z; enable='lt(t,N)' garante remoção exata ao
        # término da intro, sem flag de configuração (sempre ativo quando há intro).
        if intro_total_duration_s > 0:
            DUR = f"{intro_total_duration_s:.2f}"
            FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            # Aspas simples em volta da expressão isolam a vírgula do parser
            # de filtros do FFmpeg — sem necessidade de backslash escape.
            ENABLE = f"enable='lt(t,{DUR})'"
            rec_chain = (
                f",drawbox=x=iw-200:y=ih-90:w=180:h=70:color=black@1.0:t=fill:{ENABLE}"
                f",drawtext=fontfile={FONT}:text='●':fontcolor=red:fontsize=44"
                f":x=main_w-185:y=main_h-75:{ENABLE}"
                f",drawtext=fontfile={FONT}:text='REC':fontcolor=white:fontsize=36"
                f":x=main_w-130:y=main_h-71:{ENABLE}"
            )
        else:
            rec_chain = ""
        filter_parts.append(f"[0:v]scale=1920:1080,gblur=sigma=2{rec_chain}[scaled]")
        # Overlay PNG do personagem: opcionalmente desabilitado durante o intro IA
        # (quando a personagem já está visível no clip IA, evita "personagem dupla").
        if intro_total_duration_s > 0 and not overlay_during_intro:
            # `enable=` no overlay filter aceita expressão de tempo (em segundos)
            overlay_enable = f":enable='gte(t,{intro_total_duration_s:.2f})'"
        else:
            overlay_enable = ""
        filter_parts.append(f"[scaled][1:v]overlay=format=auto{overlay_enable}[with_person]")

        current_label = "with_person"
        for i, layer in enumerate(effect_layers_resolved):
            input_idx = 3 + i
            blend = layer['blend_mode']
            opacity = layer['opacity']
            out_label = f"fx{i}"

            if blend == 'colorkey_black':
                filter_parts.append(
                    f"[{input_idx}:v]colorkey=0x000000:0.35:0.10,format=yuva420p[fxck{i}];"
                    f"[{current_label}][fxck{i}]overlay=0:0:shortest=1[{out_label}]"
                )
            elif blend == 'screen':
                boosted = f"fx_boosted_{i}"
                filter_parts.append(f"[{input_idx}:v]eq=contrast=8.1[{boosted}]")
                filter_parts.append(
                    f"[{current_label}][{boosted}]blend=all_mode=screen:"
                    f"c0_opacity={opacity}:c1_opacity=0:c2_opacity=0[{out_label}]"
                )
            elif blend == 'chromakey_green':
                filter_parts.append(
                    f"[{input_idx}:v]chromakey=green:0.3:0.0,format=yuva420p[fxcg{i}];"
                    f"[{current_label}][fxcg{i}]overlay=0:0:shortest=1[{out_label}]"
                )
            elif blend == 'add':
                filter_parts.append(
                    f"[{input_idx}:v]format=yuv420p[fxadd{i}];"
                    f"[{current_label}][fxadd{i}]blend=all_mode=addition:"
                    f"c0_opacity={opacity}:c1_opacity=0:c2_opacity=0[{out_label}]"
                )
            else:
                # Unknown blend — pass through as screen
                boosted = f"fx_boosted_{i}"
                filter_parts.append(f"[{input_idx}:v]eq=contrast=8.1[{boosted}]")
                filter_parts.append(
                    f"[{current_label}][{boosted}]blend=all_mode=screen:"
                    f"c0_opacity={opacity}:c1_opacity=0:c2_opacity=0[{out_label}]"
                )

            current_label = out_label

        # Escape subtitle path for FFmpeg (colons must be escaped on Linux)
        safe_sub = subtitle_path.replace("\\", "/").replace(":", "\\:")
        filter_parts.append(f"[{current_label}]subtitles={safe_sub}[v]")

        filter_complex = ";".join(filter_parts)

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "2:a",
            "-c:v", "h264_nvenc" if CUDA else "libx264",
            "-preset", "fast" if CUDA else "ultrafast",
            "-crf", "25",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-t", str(audio_length),
            output_path,
        ])

        print(f"[RENDER-V4] Starting FFmpeg...")

        # ── 4. Execute with progress tracking ──
        process = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, universal_newlines=True, text=True
        )

        last_pct = -1
        for line in process.stderr:
            if "time=" in line and "speed=" in line:
                try:
                    time_str = line.split("time=")[1].split(" ")[0]
                    h, m, s = time_str.split(":")
                    seconds = float(h) * 3600 + float(m) * 60 + float(s)
                    pct = min(100, int((seconds / audio_length) * 100))
                    print(f"[RENDER-V4] {pct}% ({time_str})")
                    if progress_callback and pct != last_pct:
                        last_pct = pct
                        try:
                            progress_callback(pct)
                        except Exception:
                            pass
                except (ValueError, IndexError):
                    pass
            elif any(kw in line for kw in [
                "ffmpeg version", "built with", "configuration:", "libav",
                "Input #", "Metadata:", "Duration:", "Stream #",
                "Press [q]", "Output #", "Stream mapping:",
                "frame=", "fps=", "[libx264", "kb/s:",
            ]):
                pass
            else:
                stripped = line.strip()
                if stripped and not stripped.startswith("["):
                    print(f"ffmpeg: {stripped}")

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(f"[RENDER-V4] FFmpeg exited with code {return_code}")

        return True

    except Exception as e:
        print(f"[RENDER-V4] Error: {e}")
        raise

    finally:
        for cp in ([concat_list_path] + effect_concat_paths):
            if cp:
                try:
                    if os.path.exists(cp):
                        os.remove(cp)
                except Exception:
                    pass
