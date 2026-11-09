"""Synthesize the Step 3 clips, transcribe them with the production ASR, write data/asr_demo.csv.

    uv run --no-project --with google-genai --with numpy python tools/make_audio.py [clip_01 ...]

Needs ffmpeg and gcloud application-default credentials (Vertex AI). Every API
result is cached under audio/tmp/, so re-runs are free; delete a file to redo it.

Transcription replicates profile `gemini37-flash-v1` from
evaluation-fidelity-monitoring (the model behind FINDINGS' pooled chrF2 0.8716):
gemini-3.7-flash, its language-lock prompt, thinking budget 1024, the segments_v1
schema, mono 16 kHz 64 kbps mp3. Production's gemini-2.5-flash is not served on
Vertex for this project. Clips are under 600 s, so there is no chunking. Audio
goes inline rather than through the Files API, which Vertex does not have.
Each clip is transcribed twice, to measure the model's self-consistency.
"""
import csv, io, json, os, subprocess, sys, wave
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS, TMP, OUT = ROOT / "audio/scripts", ROOT / "audio/tmp", ROOT / "audio"
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "shamiri-digital-hub")
TTS_MODEL = os.environ.get("TTS_MODEL", "gemini-2.5-flash-preview-tts")
ASR_MODEL = "gemini-3.7-flash"
RATE = 24000  # Gemini TTS returns 16-bit mono PCM at 24 kHz

# One voice per role. Students rotate across clips so each clip sounds like a different group.
FELLOW_VOICES = ["Charon", "Kore"]
STUDENT_VOICES = ["Leda", "Puck", "Aoede", "Fenrir", "Zephyr", "Orus"]
STYLE = ("Read the quoted line aloud exactly once, word for word, adding and changing nothing. "
         "Voice: a {who} in a Nairobi secondary school, casual and warm, Kenyan accent, "
         "Swahili and Sheng words pronounced like a native speaker. Line: ")

ASR_PROMPT = """
    Transcribe this audio. It is a group session recorded in Kenya, spoken in a mixture of English, Kiswahili and Sheng; speakers switch between the three freely, often within a single sentence. Write every word in the language it was actually spoken in. Never translate a word or a phrase from one of these languages into another, in either direction: leave Kiswahili and Sheng words in Kiswahili and Sheng, and leave English words in English.
    Return a JSON list where every item has:
    - "text": The spoken phrase.
    - "start": Start time in seconds (float).
    - "end": End time in seconds (float).
    Do not identify speakers.
    """
ASR_SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "text": {"type": "STRING"}, "start": {"type": "NUMBER"}, "end": {"type": "NUMBER"}},
    "required": ["text", "start", "end"]}}

client = genai.Client(vertexai=True, project=PROJECT, location="us-central1")  # TTS
asr_client = genai.Client(vertexai=True, project=PROJECT, location="global")


def voice(clip_no, speaker):
    if speaker == "fellow":
        return FELLOW_VOICES[clip_no % len(FELLOW_VOICES)]
    k = int(speaker.split("_")[1]) - 1
    return STUDENT_VOICES[(clip_no + k) % len(STUDENT_VOICES)]


def wav_bytes(pcm_bytes):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm_bytes)
    return buf.getvalue()


def tts(text, voice_name, who, path):
    """Synthesize one turn, rejecting takes where the TTS spoke extra words.

    Gemini TTS sometimes repeats a line or improvises one ("who can tell me what
    shukuru means"), and then the script is no longer what was said. A quick
    transcript of the turn catches that by length alone: only takes with gross
    extra or missing speech are re-drawn, so spelling and recognition errors
    are untouched and still count against the model in the real scoring.
    """
    check = path.with_suffix(".check.txt")  # accepted take's transcript, kept as evidence
    for _ in range(5):
        if not path.exists():
            r = client.models.generate_content(
                model=TTS_MODEL, contents=STYLE.format(who=who) + f'"{text}"',
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)))))
            path.write_bytes(r.candidates[0].content.parts[0].inline_data.data)
            check.unlink(missing_ok=True)
        if not check.exists():
            heard = asr_client.models.generate_content(
                model=ASR_MODEL, contents=[types.Part.from_bytes(data=wav_bytes(path.read_bytes()), mime_type="audio/wav"),
                                           "Transcribe this audio verbatim. Return only the words spoken."]).text or ""
            n, m = len(text.split()), len(heard.split())
            if not (0.7 * n - 1 <= m <= 1.3 * n + 1):
                print(f"  re-drawing {path.name}: script {n} words, heard {m}: {heard.strip()!r}")
                path.unlink(); continue
            check.write_text(heard)
        return np.frombuffer(path.read_bytes(), dtype=np.int16).astype(np.float32)
    raise RuntimeError(f"{path.name}: TTS would not read the line as written; shorten or reword it")


def synth(clip_id, turns):
    clip_no = int(clip_id.split("_")[1])
    rng = np.random.default_rng(2026 + clip_no)
    parts = []
    for i, t in enumerate(turns):
        who = "young university-student facilitator" if t["speaker"] == "fellow" else "teenage student"
        parts.append(tts(t["text"], voice(clip_no, t["speaker"]), who, TMP / f"{clip_id}_{i:02d}.pcm"))
        parts.append(np.zeros(int(RATE * rng.uniform(0.25, 0.9))))  # varied gap between turns
    audio = np.concatenate(parts)
    audio += rng.normal(0, 120, audio.size)  # ponytail: white-noise "room"; swap for recorded room tone if it sounds thin
    wav = TMP / f"{clip_id}.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(np.clip(audio, -32768, 32767).astype(np.int16).tobytes())
    mp3 = OUT / f"{clip_id}.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                    "-ac", "1", "-ar", "16000", "-b:a", "64k", str(mp3)], check=True)
    return mp3


def transcribe(clip_id, mp3, run):
    path = TMP / f"{clip_id}_asr{run}.json"
    if not path.exists():
        r = asr_client.models.generate_content(
            model=ASR_MODEL,
            contents=[types.Part.from_bytes(data=mp3.read_bytes(), mime_type="audio/mp3"), ASR_PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               response_schema=ASR_SCHEMA, max_output_tokens=65536,
                                               thinking_config=types.ThinkingConfig(thinking_budget=1024)))
        path.write_text(r.text)
    return " ".join(s["text"] for s in json.loads(path.read_text()))


def main(only):
    TMP.mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
    out = ROOT / "data/asr_demo.csv"
    rows = {r["clip_id"]: r for r in csv.DictReader(out.open())} if out.exists() else {}
    for script in sorted(SCRIPTS.glob("clip_*.json")):
        clip_id = script.stem
        if only and clip_id not in only:
            continue
        turns = json.loads(script.read_text())
        mp3 = synth(clip_id, turns)
        prev = rows.get(clip_id, {})
        rows[clip_id] = {"clip_id": clip_id, "verbatim": " ".join(t["text"] for t in turns),
                         "human_a": prev.get("human_a", ""), "human_b": prev.get("human_b", ""),
                         "ai_model": transcribe(clip_id, mp3, 1), "ai_model_run2": transcribe(clip_id, mp3, 2)}
        print(f"{clip_id}: {mp3.relative_to(ROOT)}\n  said: {rows[clip_id]['verbatim']}\n  asr:  {rows[clip_id]['ai_model']}")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, ["clip_id", "verbatim", "human_a", "human_b", "ai_model", "ai_model_run2"])
        w.writeheader(); w.writerows(rows[k] for k in sorted(rows))


if __name__ == "__main__":
    main(set(sys.argv[1:]))
