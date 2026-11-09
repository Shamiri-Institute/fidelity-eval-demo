# fidelity-eval-demo

Live demo for the Shamiri Summit 2026 Tech track session **"Developing AI Evaluation Systems You Can Trust"** (Friday, November 13, 11:30am to 1:00pm). The audience follows a Colab notebook that walks through how Shamiri evaluated its AI fidelity-rating pipeline, step by step, on synthetic data.

Nothing in this repo comes from real sessions, students, Fellows or schools. Keep it that way: it will be public.

## What's here

| Path | What it is |
|---|---|
| `notebooks/fidelity_eval_walkthrough.ipynb` | Audience copy, no outputs. Runs in Colab with no installs and no data files. |
| `notebooks/fidelity_eval_walkthrough_with_outputs.ipynb` | Presenter backup with every output saved, for when the Wi-Fi fails. |
| `tools/build_nb.py` | **Source of truth.** Writes the notebook from Python. Edit this, not the `.ipynb`. |
| `tools/make_audio.py` | Synthesizes the Step 3 clips (Gemini TTS), transcribes them, writes `data/asr_demo.csv`. Cached in `audio/tmp/`. |
| `tools/run_nb.py` | Minimal executor that runs every cell in order and saves outputs. Use it if Jupyter isn't installed. |
| `docs/PLAN.md` | Next piece of work: synthetic audio for Step 3. |
| `audio/` | Step 3 clip scripts (`scripts/`) and the synthesized MP3s. |

## Build and run

```bash
uv venv && uv pip install numpy pandas scipy scikit-learn matplotlib
uv run python tools/build_nb.py notebooks/fidelity_eval_walkthrough.ipynb
uv run python tools/run_nb.py notebooks/fidelity_eval_walkthrough.ipynb notebooks/fidelity_eval_walkthrough_with_outputs.ipynb
```

`run_nb.py` exits non-zero if any cell fails. With Jupyter available, `jupyter nbconvert --to notebook --execute` works too.

## The eight steps

1. Audit the reference data (duplicates, missing scores, bad totals, rater workload)
2. Measure human agreement first, and use it as the ceiling (ICC, Spearman-Brown)
3. Check the transcript stage (CER vs WER on code-switched speech)
4. Score the judge against humans, per dimension (finds the collapsed protocol-adherence item)
5. Two traps: the trivial baseline, and re-draw noise on identical input
6. Compare prompt versions with a paired test and a pre-registered floor
7. Report a rank and a "listen to this one" flag instead of a score
8. Safety flags on a seeded test set, with an asymmetric release gate

## Status (November 9)

- All cells run end to end with `tools/run_nb.py`. **Not yet tested in real Colab.**
- Step 3 uses five synthetic clips (`audio/`), transcribed twice each by `gemini-3.7-flash` (`data/asr_demo.csv`, built by `tools/make_audio.py`). The two human transcripts of clip 1 are still hand-written. Production's `gemini-2.5-flash` is not served on Vertex for our project, so the eval's candidate model stands in.
- Public at https://github.com/Shamiri-Institute/fidelity-eval-demo. `REPO_RAW` in `tools/build_nb.py` points at it, so the notebook loads the CSV from GitHub (falling back to its embedded copy) and plays clip 1 in Step 3.
- Colab link (for the QR code on the holding slide): https://colab.research.google.com/github/Shamiri-Institute/fidelity-eval-demo/blob/main/notebooks/fidelity_eval_walkthrough.ipynb
