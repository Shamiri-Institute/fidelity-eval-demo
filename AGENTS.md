# fidelity-eval-demo

Live-demo notebook for Shadrack's Shamiri Summit 2026 session, "Developing AI Evaluation Systems You Can Trust" (Friday, November 13, 11:30am). Read `README.md` for what's here and `docs/PLAN.md` for the current task: synthetic audio for Step 3.

## Rules

- **Synthetic only.** Never copy real recordings, transcripts, ratings, names or schools from the sibling repos into this one. It will be public. Read the sibling repos for methods, not data.
- **No risk or distress content** in any script or audio, even fictional.
- **Edit `tools/build_nb.py`, never the `.ipynb` by hand.** Rebuild both notebooks after any change:
  `uv run python tools/build_nb.py notebooks/fidelity_eval_walkthrough.ipynb && uv run python tools/run_nb.py notebooks/fidelity_eval_walkthrough.ipynb notebooks/fidelity_eval_walkthrough_with_outputs.ipynb`
- **The audience copy must stay Colab-ready:** standard Colab libraries only, no local files. Anything new (CSV, audio) is loaded from a public URL with a fallback to embedded data.
- **The data generator uses a fixed seed (2026).** The markdown text quotes results that depend on it. If you change the generator or seed, re-read every markdown cell against the new outputs.
- **Keep the notebook honest.** Label anything simulated as simulated. If real model output doesn't support a claim in the markdown, change the claim, not the data.

## Context

- The synthetic data in Setup is tuned to mirror the patterns in `../evaluation-fidelity-monitoring/FINDINGS.md` (human ICC about 0.2, a harsh judge with one collapsed item, re-draw noise, a trivial baseline that beats offset correction). Real figures are quoted separately by the presenter, not in the notebook.
- The presenter's run of show (a Word document, outside this repo) times each step to about 5 to 8 minutes. Keep cells short and outputs readable at a large zoom on a projector.
