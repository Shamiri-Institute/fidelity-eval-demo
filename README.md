# fidelity-eval-demo

Live demo for the Shamiri Summit 2026 Tech track session **"Developing AI Evaluation Systems You Can Trust"** (Friday, November 13, 11:30am to 1:00pm). The audience follows a Colab notebook that walks through how Shamiri evaluated its AI fidelity-rating pipeline, step by step, on synthetic data.

Nothing in this repo comes from real sessions, students, Fellows or schools. Keep it that way: it will be public.

## What's here

| Path | What it is |
|---|---|
| `notebooks/fidelity_eval_walkthrough.ipynb` | Audience copy, no outputs. Runs in Colab with no installs and no data files. |
| `notebooks/fidelity_eval_walkthrough_with_outputs.ipynb` | Presenter backup with every output saved, for when the Wi-Fi fails. |
| `tools/build_nb.py` | **Source of truth.** Writes the notebook from Python. Edit this, not the `.ipynb`. |
| `tools/run_nb.py` | Minimal executor that runs every cell in order and saves outputs. Use it if Jupyter isn't installed. |
| `docs/PLAN.md` | Next piece of work: synthetic audio for Step 3. |
| `audio/` | Where generated session scripts and audio will live. |

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
- Step 3's "AI model" transcript is **hand-written**, not real ASR output. See `docs/PLAN.md`.
- Before the session: publish the audience notebook somewhere public, make the Colab link, put it behind the QR code on the holding slide.
