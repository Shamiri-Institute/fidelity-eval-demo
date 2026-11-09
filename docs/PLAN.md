# Plan: synthetic audio for Step 3

## Why

Step 3 of the notebook compares four transcripts of one clip: what was said, two human transcribers, and the AI model. Today all four are hand-written, including the "AI model" one (the verbatim text with a few spelling slips added). That illustrates the method but isn't a real result.

Goal: generate a handful of short, fully synthetic session recordings, run them through a real transcription model, and let Step 3 show real model output against a script we know is exactly what was "said".

A nice property: the script **is** the ground-truth verbatim, which we never have for real sessions.

## Scope

Small. This is a demo, not a dataset.

- **4 to 6 clips, 60 to 120 seconds each.** One would be enough for Step 3; a few let the notebook show an average and a spread.
- **Speakers:** one Fellow and three or four students per clip, each with a distinct voice.
- **Language:** code-switched Kenyan English, Swahili and Sheng, like real sessions. This is the hard part for both TTS and ASR, and the point of the demo.
- **Content:** ordinary, warm session moments (gratitude, growth mindset, values, check-ins). No disclosures, distress or risk content of any kind, not even fictional. Step 8 covers safety without audio.
- **No real names**, schools or places that identify anyone.

## Pipeline

1. **Write scripts** as structured files, one per clip, e.g. `audio/scripts/clip_01.json`:
   `[{"speaker": "fellow", "text": "..."}, {"speaker": "student_1", "text": "..."}, ...]`.
   Keep fillers ("eh", "um") and Sheng spellings in, since they are what makes CER vs WER interesting. The existing clip in `tools/build_nb.py` (the `verbatim` column) is a good seed for clip 1.
2. **Synthesize each turn** with text-to-speech, one voice per speaker, then join turns with short, slightly varied pauses. Optionally add light room noise, and occasional overlaps, so it sounds like a classroom rather than a podcast.
3. **Transcribe** each clip with the real pipeline (see "Prior work" below). Save the raw output.
4. **Get human transcripts** (optional but best): two colleagues transcribe the clips without seeing the script. This reproduces the human-vs-human comparison for real. If there's no time, keep the two hand-written human versions, clearly labelled.
5. **Store results** as `data/asr_demo.csv` with columns `clip_id, line, verbatim, human_a, human_b, ai_model`. Commit the scripts and CSV. Decide whether audio files go in git (small MP3s are fine) or a release asset.
6. **Update the notebook**: Step 3 loads `data/asr_demo.csv` from the repo's raw GitHub URL when available, otherwise falls back to the embedded text. Update the Step 3 markdown so it says plainly which parts are synthetic speech, real model output, and human transcription. Rebuild with `tools/build_nb.py` and re-run with `tools/run_nb.py`.
7. **Optional stretch:** embed one clip with `IPython.display.Audio` so the room hears it before seeing the transcripts. Strong moment for the live demo.

## TTS options to evaluate

Check each one's current Swahili and Kenyan English support before committing. Code-switching mid-sentence is the real test.

- **Gemini TTS on Vertex AI.** Shamiri already uses Vertex (see `workshop_transcript_generation`), and it supports multi-speaker dialogue in one call.
- **Azure Neural TTS.** Has had Kenyan voices for Swahili and English; may need per-language voice switching inside a turn.
- **Google Cloud Text-to-Speech.** Check for sw-KE / en-KE voices.
- **Meta MMS-TTS** (open model, has Swahili). Runs locally, no API cost, robotic.
- **Fallback:** colleagues read the scripts aloud. Five minutes per clip, most realistic, no API at all.

Pick by listening: it has to sound plausible to a room of Kenyans, or it undermines the demo.

## Prior work in ~/Workdir/institute

Check these before building anything new:

- `evaluation-fidelity-monitoring/` — the real evaluation. `FINDINGS.md` has the results the demo mirrors; `evals/asr/` has the ASR evaluation method (CER vs WER, normalization, the strict-vs-masked comparison). Reuse its normalization so the demo matches.
- `shamiri_whisper/` — fine-tuned Whisper checkpoint and an eval notebook. A candidate "AI model" for step 3.
- `workshop_transcript_generation/` — transcription with Gemini on Vertex AI, including notes on content-safety filters for mental-health audio.
- `audio-annotation-platform/` — how human transcriptions are collected (two independent transcriptions per segment).

Use whichever transcription model is the current production one, so the demo reflects what Shamiri actually runs.

## Definition of done

- [ ] 4–6 synthetic clips with scripts, audio and real model transcripts in the repo
- [ ] Step 3 uses real model output, with labels that say what is synthetic and what is real
- [ ] Notebook rebuilt and re-run; the CER comparison still makes its point. If the real model doesn't beat human-vs-human disagreement on this audio, **report that honestly** rather than tuning the clips until it does
- [ ] Tested once in Google Colab from a clean browser
- [ ] Repo (or a public copy of `notebooks/` + `data/`) published, Colab link working

Deadline: the session is Friday, November 13, at 11:30am.
