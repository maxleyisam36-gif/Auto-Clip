#!/usr/bin/env python3
"""
autoclip.py — Turn one long YouTube video into several short, captioned clips.

Pipeline:
  1. Download your video (yt-dlp)
  2. Transcribe it (faster-whisper, runs locally, no API needed)
  3. Pick the most interesting segments
       - if ANTHROPIC_API_KEY is set: Claude reads the transcript and picks
         the best moments (much better quality)
       - otherwise: a simple keyword/pace heuristic picks them
  4. Cut each segment into its own clip, with a random length inside the
     range you choose (e.g. 15-60s, or 60-300s)
  5. Burn in captions on each clip, ready to upload

Usage:
    python autoclip.py "https://youtube.com/watch?v=XXXX" \
        --num-clips 5 --min-sec 20 --max-sec 60 --out clips/

Requirements: see requirements.txt (pip install -r requirements.txt)
Also requires ffmpeg installed on your system (https://ffmpeg.org/download.html)
"""

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class ClipPlan:
    start: float
    end: float
    reason: str = ""
    words: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Step 1: Download
# --------------------------------------------------------------------------

def download_video(url: str, workdir: Path) -> Path:
    """Download the video with yt-dlp, return the local file path."""
    out_template = str(workdir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        url,
    ]
    print(f"[1/5] Downloading: {url}")
    subprocess.run(cmd, check=True)

    # yt-dlp resolves the actual filename; find the mp4 it produced
    candidates = list(workdir.glob("source.*"))
    for c in candidates:
        if c.suffix == ".mp4":
            return c
    if candidates:
        return candidates[0]
    raise FileNotFoundError("yt-dlp did not produce an output file")


# --------------------------------------------------------------------------
# Step 2: Transcribe
# --------------------------------------------------------------------------

def transcribe_video(video_path: Path, model_size: str = "base") -> list[TranscriptSegment]:
    """Transcribe locally with faster-whisper. Returns word-free segments."""
    from faster_whisper import WhisperModel

    print(f"[2/5] Transcribing ({model_size} model)... this can take a while for long videos")
    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(str(video_path), vad_filter=True)

    result = []
    for seg in segments:
        result.append(TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip()))
    print(f"      got {len(result)} transcript segments")
    return result


# --------------------------------------------------------------------------
# Step 3: Pick highlight segments
# --------------------------------------------------------------------------

def pick_highlights_with_claude(
    transcript: list[TranscriptSegment],
    num_clips: int,
    min_sec: float,
    max_sec: float,
) -> list[ClipPlan]:
    """Ask Claude to pick the best moments. Requires ANTHROPIC_API_KEY."""
    import anthropic

    client = anthropic.Anthropic()

    # Build a compact, timestamped transcript for the prompt
    lines = [f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in transcript]
    transcript_text = "\n".join(lines)

    prompt = f"""You are picking highlight clips from a video transcript for short-form
social media (like YouTube Shorts / TikTok / Reels).

Pick {num_clips} distinct, non-overlapping moments that would make the most
engaging, self-contained short clips. Prefer moments with a clear point,
a strong hook, a surprising fact, or useful advice. Each clip must be
between {min_sec} and {max_sec} seconds long, and start/end times must
fall on natural sentence boundaries from the transcript below.

Respond ONLY with a JSON array, no other text, in this exact format:
[{{"start": 12.3, "end": 45.6, "reason": "short reason why this moment works"}}, ...]

Transcript (format is [start-end] text):
{transcript_text}
"""

    print("[3/5] Asking Claude to pick the best moments...")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    picks = json.loads(raw)

    plans = [ClipPlan(start=p["start"], end=p["end"], reason=p.get("reason", "")) for p in picks]
    return plans


def pick_highlights_heuristic(
    transcript: list[TranscriptSegment],
    num_clips: int,
    min_sec: float,
    max_sec: float,
) -> list[ClipPlan]:
    """Fallback with no API key: score segments by keyword density and pick
    spread-out windows of a random length within [min_sec, max_sec]."""
    print("[3/5] No ANTHROPIC_API_KEY found — using simple heuristic instead")

    KEYWORDS = (
        "important", "key", "secret", "mistake", "never", "always", "best",
        "worst", "surprising", "actually", "truth", "learned", "tip", "rule",
        "biggest", "huge", "amazing", "warning", "avoid", "must", "why",
    )

    def score(text: str) -> int:
        t = text.lower()
        return sum(1 for k in KEYWORDS if k in t) + (1 if "?" in t or "!" in t else 0)

    if not transcript:
        return []

    total_duration = transcript[-1].end
    plans: list[ClipPlan] = []
    used_ranges: list[tuple[float, float]] = []

    # score each transcript segment, sort best first
    scored = sorted(transcript, key=lambda s: score(s.text), reverse=True)

    for seg in scored:
        if len(plans) >= num_clips:
            break
        length = random.uniform(min_sec, max_sec)
        start = max(0.0, seg.start - length * 0.2)
        end = min(total_duration, start + length)

        # skip if it overlaps a clip we already picked
        if any(not (end <= u0 or start >= u1) for u0, u1 in used_ranges):
            continue

        plans.append(ClipPlan(start=start, end=end, reason="keyword match"))
        used_ranges.append((start, end))

    # if heuristic didn't find enough, fill in evenly spaced random clips
    while len(plans) < num_clips and total_duration > min_sec:
        length = random.uniform(min_sec, max_sec)
        start = random.uniform(0, max(0.0, total_duration - length))
        end = start + length
        if any(not (end <= u0 or start >= u1) for u0, u1 in used_ranges):
            continue
        plans.append(ClipPlan(start=start, end=end, reason="filler"))
        used_ranges.append((start, end))

    plans.sort(key=lambda p: p.start)
    return plans


def pick_highlights(transcript, num_clips, min_sec, max_sec) -> list[ClipPlan]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return pick_highlights_with_claude(transcript, num_clips, min_sec, max_sec)
        except Exception as e:
            print(f"      Claude selection failed ({e}), falling back to heuristic")
    return pick_highlights_heuristic(transcript, num_clips, min_sec, max_sec)


# --------------------------------------------------------------------------
# Step 4 + 5: Cut clip and burn captions
# --------------------------------------------------------------------------

def format_srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt_for_clip(transcript: list[TranscriptSegment], clip: ClipPlan, srt_path: Path):
    """Write an SRT file with timestamps re-based to the clip's own start=0."""
    lines = []
    idx = 1
    for seg in transcript:
        # keep any transcript segment that overlaps this clip
        if seg.end <= clip.start or seg.start >= clip.end:
            continue
        rel_start = max(0.0, seg.start - clip.start)
        rel_end = min(clip.end - clip.start, seg.end - clip.start)
        if rel_end <= rel_start or not seg.text:
            continue
        lines.append(str(idx))
        lines.append(f"{format_srt_timestamp(rel_start)} --> {format_srt_timestamp(rel_end)}")
        lines.append(seg.text)
        lines.append("")
        idx += 1
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def cut_and_caption_clip(
    source_video: Path,
    transcript: list[TranscriptSegment],
    clip: ClipPlan,
    out_path: Path,
):
    """Cut [clip.start, clip.end] out of source_video and burn in captions."""
    duration = clip.end - clip.start
    srt_path = out_path.with_suffix(".srt")
    write_srt_for_clip(transcript, clip, srt_path)

    # Style: bold white text, black outline, positioned near the bottom —
    # legible over any footage, sized for phone screens.
    style = (
        "FontName=Arial,FontSize=14,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,"
        "Alignment=2,MarginV=60"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip.start),
        "-i", str(source_video),
        "-t", str(duration),
        "-vf", f"subtitles={srt_path}:force_style='{style}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    srt_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(url: str, num_clips: int, min_sec: float, max_sec: float,
        out_dir: str, whisper_model: str, keep_source: bool,
        progress_cb=None, workdir: Path | None = None):
    """Run the full pipeline. progress_cb(status: str, detail: str) is called
    at each major step so a caller (e.g. a web app) can show live status."""

    def report(status, detail=""):
        print(f"[{status}] {detail}")
        if progress_cb:
            progress_cb(status, detail)

    workdir = workdir or Path("autoclip_work")
    workdir.mkdir(exist_ok=True, parents=True)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report("downloading", url)
    source_video = download_video(url, workdir)

    report("transcribing", f"model={whisper_model}")
    transcript = transcribe_video(source_video, model_size=whisper_model)

    if not transcript:
        report("error", "No speech detected — can't pick highlights automatically.")
        return []

    report("selecting", "choosing the best moments")
    plans = pick_highlights(transcript, num_clips, min_sec, max_sec)
    if not plans:
        report("error", "Could not find enough distinct segments for the requested clip count/length.")
        return []

    report("cutting", f"{len(plans)} clip(s)")
    manifest = []
    for i, clip in enumerate(plans, start=1):
        clip_len = clip.end - clip.start
        out_file = out_path / f"clip_{i:02d}_{int(clip_len)}s.mp4"
        report("cutting", f"clip {i}/{len(plans)}: {clip.start:.1f}s-{clip.end:.1f}s — {clip.reason}")
        cut_and_caption_clip(source_video, transcript, clip, out_file)
        manifest.append({
            "file": out_file.name,
            "start": clip.start,
            "end": clip.end,
            "length_sec": round(clip_len, 1),
            "reason": clip.reason,
        })

    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if not keep_source:
        source_video.unlink(missing_ok=True)

    report("done", f"{len(plans)} captioned clip(s) in: {out_path.resolve()}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Auto-clip your own long video into captioned shorts.")
    parser.add_argument("url", help="YouTube URL of your own video")
    parser.add_argument("--num-clips", type=int, default=5, help="how many clips to produce")
    parser.add_argument("--min-sec", type=float, default=20, help="minimum clip length in seconds")
    parser.add_argument("--max-sec", type=float, default=60, help="maximum clip length in seconds")
    parser.add_argument("--out", default="clips", help="output folder")
    parser.add_argument("--whisper-model", default="base",
                         choices=["tiny", "base", "small", "medium", "large-v3"],
                         help="bigger = more accurate transcription, but slower")
    parser.add_argument("--keep-source", action="store_true",
                         help="keep the downloaded source video instead of deleting it")
    args = parser.parse_args()

    if args.min_sec > args.max_sec:
        print("--min-sec cannot be greater than --max-sec")
        sys.exit(1)

    run(args.url, args.num_clips, args.min_sec, args.max_sec,
        args.out, args.whisper_model, args.keep_source)


if __name__ == "__main__":
    main()
