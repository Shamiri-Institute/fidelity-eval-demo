"""Builds the follow-along notebook as raw .ipynb JSON (nbformat 4.5)."""
import json, sys, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_RAW = "https://raw.githubusercontent.com/Shamiri-Institute/fidelity-eval-demo/main/"

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

# ---------------------------------------------------------------- intro
md(r"""
# Can you trust your AI rater?
### A follow-along evaluation walkthrough · Shamiri Summit 2026 · Tech track

**Session:** Developing AI Evaluation Systems You Can Trust

Shamiri trains university-age **Fellows** to deliver group sessions in secondary schools. Supervisors check each session's **fidelity**: did the Fellow deliver the protocol, and deliver it well? Humans can't listen to every session, so we built an AI pipeline to help:

```
session recording ──► transcription (ASR) ──► LLM judge ──► 6 rubric scores (1–7)
                                                          ├─► written feedback
                                                          └─► safety flags ──► supervisor
```

This notebook walks through **how we evaluated that pipeline**, one step at a time. Each step ends with a lesson you can take back to your own AI system.

> **All data here is synthetic.** It is generated in the first cell and contains no real students, Fellows, schools or recordings. It was tuned so the *patterns* resemble what we found on real data. The numbers will not match ours exactly, and the lessons hold either way.

**How to follow along:** in Google Colab, choose *Runtime → Run all*, then scroll. Or run one cell at a time with **Shift + Enter**. Cells marked **Your turn** invite you to change something and re-run it.
""")

md(r"""
## Setup
Run this cell first. It builds the synthetic dataset: 180 group sessions, two human ratings each, an AI judge run three times on identical input, a second prompt version, a transcription sample, and a seeded safety test set. You don't need to read it. In Colab it shows as a collapsed form.
""")

code(r'''
# @title Setup: build the synthetic dataset (run me, no need to read) { display-mode: "form" }
import numpy as np, pandas as pd, matplotlib.pyplot as plt, re, warnings
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score
warnings.filterwarnings("ignore")
pd.set_option("display.precision", 3); pd.set_option("display.width", 120); pd.set_option("display.max_colwidth", 90)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

DIMS = ["protocol_adherence", "content_specifications", "thoroughness",
        "skillful_delivery", "clarity_accessibility", "protocol_boundaries"]

def _build(seed=2026, N=180):
    rng = np.random.default_rng(seed)
    t = rng.normal(0, 1, N)                       # hidden "true" session quality
    ids = [f"S{i:03d}" for i in range(1, N + 1)]
    sessions = pd.DataFrame({
        "session_id": ids,
        "school": rng.choice([f"School {c}" for c in "ABCDEFGH"], N),
        "session_type": rng.choice(["s1", "s2", "s3", "s4"], N),
        "fellow_id": [f"F{x:02d}" for x in rng.integers(1, 31, N)],
        "audio_sha": [f"{x:012x}" for x in rng.integers(0, 2**48, N)],
    })
    # --- human raters: lenient, noisy, crowded at the top of the scale, uneven workload
    raters = [f"R{i}" for i in range(1, 9)]
    workload = np.array([.40, .18, .12, .10, .08, .06, .04, .02])
    lean = dict(zip(raters, rng.normal(0, .35, 8)))
    rows = []
    for i in range(N):
        for slot, r in enumerate(rng.choice(raters, 2, replace=False, p=workload), start=1):
            y = np.clip(np.rint(5.95 + .55 * t[i] + lean[r] + rng.normal(0, .8) + rng.normal(0, .9, 6)), 1, 7)
            rows.append({"session_id": ids[i], "rater_slot": slot, "rater_id": r, **dict(zip(DIMS, y))})
    human = pd.DataFrame(rows)
    # --- AI judge, prompt v1: harsh, one collapsed item, re-draw noise; 3 draws on identical input
    mu_v1 = np.array([2.45, 4.8, 3.25, 3.2, 5.9, 6.25])
    load = np.array([.01, 1, 1, 1, 1, 1]) * .7
    opinion = rng.normal(0, .5, (N, 6)) + rng.normal(0, .4, (N, 1))   # the judge's stable view of each session
    ai = []
    def draws(mu, load, version, n_draws, noise=.45):
        for d in range(1, n_draws + 1):
            run_noise = rng.normal(0, noise, (N, 6)) + rng.normal(0, .3, (N, 1))
            sc = np.clip(np.rint(mu + np.outer(t, load) + opinion + run_noise), 1, 7)
            for i in range(N):
                ai.append({"session_id": ids[i], "prompt_version": version, "draw": d, **dict(zip(DIMS, sc[i]))})
    draws(mu_v1, load, "v1", 3)
    # --- prompt v2: the over-strict adherence clause rewritten; everything else unchanged
    mu_v2 = mu_v1.copy(); mu_v2[0] = 4.9
    load_v2 = load.copy(); load_v2[0] = .7
    draws(mu_v2, load_v2, "v2", 1)
    ai = pd.DataFrame(ai)
    for c in DIMS: ai[c] = ai[c].astype(int)

    # --- plant the data-hygiene problems we actually met
    raw = human.copy()
    for c in DIMS: raw[c] = raw[c].astype("Int64")
    raw["overall_score"] = raw[DIMS].sum(axis=1).astype("Int64")
    bad_total = rng.choice(len(raw), 6, replace=False)
    raw.loc[bad_total, "overall_score"] = raw.loc[bad_total, "overall_score"] - rng.choice([-10, 10, 7], 6)
    missing = rng.choice(len(raw), 3, replace=False)
    for k, m in enumerate(missing): raw.loc[m, DIMS[k * 2]] = pd.NA
    dup_src = rng.choice(N, 4, replace=False)      # the same recording filed under a second id
    dups, extra_sessions = [], []
    for k, s in enumerate(dup_src):
        new_id = f"S{N + 1 + k:03d}"
        extra = sessions.iloc[s].copy(); extra["session_id"] = new_id; extra_sessions.append(extra)
        d = raw[raw.session_id == ids[s]].copy(); d["session_id"] = new_id; dups.append(d)
    raw = pd.concat([raw] + dups, ignore_index=True)
    sessions_raw = pd.concat([sessions, pd.DataFrame(extra_sessions)], ignore_index=True)
    return sessions_raw, raw, ai

sessions_raw, human_raw, ai_scores = _build()

# --- five synthetic clips: scripted, spoken by text-to-speech, transcribed by a real model (see Step 3)
REPO_RAW = "__REPO_RAW__"   # public raw-file URL of the repo; empty means use the copy embedded below
try:
    asr_clips = pd.read_csv(REPO_RAW + "data/asr_demo.csv")
except Exception:
    import io; asr_clips = pd.read_csv(io.StringIO(r"""__ASR_CSV__"""))
asr_clips = asr_clips.fillna("")

# --- seeded safety test set: 20 genuine current-risk cases, 40 look-alikes (no real disclosures; labels only)
_r = np.random.default_rng(7)
_kinds = (["current distress, stated plainly"] * 8 + ["current distress, indirect wording"] * 6 +
          ["current distress, in Sheng"] * 4 + ["current distress, split across turns"] * 2)
_look = (["story about a past, resolved event"] * 16 + ["hypothetical in a role-play"] * 12 +
         ["teaching example from the manual"] * 12)
safety = pd.DataFrame({"case_id": [f"C{i:02d}" for i in range(1, 61)],
                       "scenario": _kinds + _look,
                       "is_current_risk": [True] * 20 + [False] * 40})
safety["ai_flagged"] = True
safety.loc[[19], "ai_flagged"] = False                       # misses one split-across-turns case
safety.loc[_r.choice(range(20, 60), 13, replace=False), "ai_flagged"] = False
safety["ai_timeframe"] = np.where(safety.is_current_risk, "current",
    safety.scenario.map({"story about a past, resolved event": "past",
                         "hypothetical in a role-play": "hypothetical",
                         "teaching example from the manual": "educational"}))
safety.loc[[24, 41], "ai_timeframe"] = "current"            # two look-alikes the model mislabels
safety.loc[~safety.ai_flagged, "ai_timeframe"] = None

print("Built:")
print(f"  human_raw     {human_raw.shape[0]} rating rows over {human_raw.session_id.nunique()} session ids")
print(f"  ai_scores     {ai_scores.shape[0]} rows (prompt v1 x 3 draws, prompt v2 x 1 draw)")
print(f"  asr_clips     {len(asr_clips)} synthetic recordings, each transcribed twice by a real model")
print(f"  safety        {len(safety)} seeded test cases")
''')

# ---------------------------------------------------------------- step 1
md(r"""
---
## Step 1 · Audit the reference before you trust it

Every number in an evaluation is measured *against* something: here, two human supervisors' ratings of each session. That reference is a measuring instrument, and it needs quality control like any other.

Four checks that caught real problems for us:
1. **Duplicates.** Is the same recording filed under two ids? Then your "held-out" set may hold sessions you already tuned on.
2. **Missing scores.**
3. **Totals that don't add up.** Is the stored `overall_score` really the sum of the items?
4. **Who did the rating.** If one person rated most sessions, your "human standard" is that one person's habits.
""")
code(r'''
print("Rows:", len(human_raw), "| session ids:", human_raw.session_id.nunique())
human_raw.head()
''')
code(r'''
# 1. Same audio under different session ids
ids_per_audio = sessions_raw.groupby("audio_sha").session_id.agg(list)
dup_groups = ids_per_audio[ids_per_audio.str.len() > 1]
print(f"1. Duplicate recordings: {len(dup_groups)} groups")
for ids in dup_groups: print("     ", " = ".join(ids))

# 2. Missing item scores
missing = human_raw[human_raw[DIMS].isna().any(axis=1)]
print(f"\n2. Rows with a missing item score: {len(missing)}")

# 3. Stored total vs recomputed total
recomputed = human_raw[DIMS].sum(axis=1, min_count=6)
bad_total = human_raw[(recomputed.notna()) & (recomputed != human_raw.overall_score)]
print(f"\n3. Rows where overall_score != sum of items: {len(bad_total)}")

# 4. Workload per rater
print("\n4. Share of ratings by rater:")
print((human_raw.rater_id.value_counts(normalize=True) * 100).round(0).astype(int).astype(str).add("%").to_string())
''')
code(r'''
# Clean: keep one id per recording, drop sessions without two complete ratings, recompute totals
drop_ids = [ids[1:] for ids in dup_groups]
drop_ids = [i for group in drop_ids for i in group]
human = human_raw[~human_raw.session_id.isin(drop_ids)].copy()
complete = human.groupby("session_id")[DIMS].apply(lambda g: g.notna().all().all() and len(g) == 2)
human = human[human.session_id.isin(complete[complete].index)].copy()
human[DIMS] = human[DIMS].astype(int)
human["composite"] = human[DIMS].mean(axis=1)
print(f"Clean reference: {human.session_id.nunique()} sessions, each rated by two people")
''')
md(r"""
**Lesson 1.** Your golden dataset is a measuring instrument. Audit it, fix it, and freeze it (hash the files) *before* you score anything against it.

**Your turn:** look at check 4. If rater R1 did 40% of the ratings and is unusually lenient, what happens to an AI that is "calibrated to the humans"?
""")

# ---------------------------------------------------------------- step 2
md(r"""
---
## Step 2 · Measure the humans first

Before asking "does the AI agree with humans?", ask **"do humans agree with each other?"** Two trained supervisors rated every session independently, so we can measure that directly.

- **Exact / within ±1:** how often the two scores match, or are at most one point apart
- **QWK** (quadratic weighted kappa): agreement corrected for chance
- **ICC:** how much of the variation in scores is real differences between sessions, rather than rater noise (1 = perfect, 0 = pure noise)
""")
code(r'''
def icc_a1(x):
    """ICC(A,1): two-way, absolute agreement, single rater. x: sessions x raters."""
    x = np.asarray(x, float); n, k = x.shape; gm = x.mean()
    msr = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    mse = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

wide = human.pivot(index="session_id", columns="rater_slot")
rows = []
for d in DIMS + ["composite"]:
    a, b = wide[(d, 1)].values, wide[(d, 2)].values
    rows.append({"dimension": d,
                 "exact %": np.mean(np.rint(a) == np.rint(b)) * 100 if d != "composite" else np.nan,
                 "within ±1 %": np.mean(np.abs(a - b) <= 1) * 100 if d != "composite" else np.nan,
                 "QWK": cohen_kappa_score(a.astype(int), b.astype(int), weights="quadratic") if d != "composite" else np.nan,
                 "ICC": icc_a1(np.c_[a, b])})
human_agreement = pd.DataFrame(rows).set_index("dimension")
human_agreement.round(2)
''')
code(r'''
icc = human_agreement.loc["composite", "ICC"]
share_high = (human[DIMS].values >= 6).mean()
print(f"{share_high:.0%} of all item scores are a 6 or a 7: most of the scale goes unused.")
print(f"Composite ICC between two trained raters: {icc:.2f}")

# What does that permit? Spearman-Brown: reliability of the average of k raters
sb = lambda r, k: k * r / (1 + (k - 1) * r)
print("\nReliability if we averaged k raters:")
for k in [1, 2, 3, 5, 10]: print(f"   k={k:>2}: {sb(icc, k):.2f}")

ceiling_1 = np.sqrt(icc)            # best possible correlation of ANY judge with one rater
ceiling_2 = np.sqrt(sb(icc, 2))     # ... with the average of two raters
print(f"\nCeiling vs one rater:        r ≤ {ceiling_1:.2f}")
print(f"Ceiling vs two-rater average: r ≤ {ceiling_2:.2f}")
''')
md(r"""
Even a *perfect* AI could not correlate with these humans above the ceiling, because the humans don't agree with each other. So an AI scoring r = 0.45 here is not "mediocre". It may be close to the best possible.

**Lesson 2.** Measure human agreement first, and report every AI-vs-human number **next to that ceiling, never against 1.0**. If the ceiling is low, the cheapest improvement may be rater training and anchored rubric examples, not a better model.
""")

# ---------------------------------------------------------------- step 3
md(r"""
---
## Step 3 · Check the upstream component: the transcript

The judge reads a transcript, not audio. If the transcript is wrong, the scores measure the transcription, not the Fellow.

Real evaluations have a gap here: nobody knows exactly what was said, so you score the model against a human transcript and hope the human got it right. For this demo we closed that gap:

| | what it is |
|---|---|
| **script** | Five short sessions we wrote in code-switched English, Swahili and Sheng. Synthetic. |
| **audio** | Each script read aloud by text-to-speech, one voice per speaker. Synthetic. Takes where the voice repeated or ad-libbed a line were re-recorded, so the script is exactly what was said. |
| **model** | Real output from `gemini-3.7-flash`, the transcription model from our evaluation, run **twice** on each clip. |
| **human A, human B** | Clip 1 only. **Hand-written** to show two common transcriber styles. Not real transcribers. |
""")
code(r'''
import textwrap
from IPython.display import Audio
clip1 = asr_clips.set_index("clip_id").loc["clip_01"]
if REPO_RAW: display(Audio(REPO_RAW + "audio/clip_01.mp3"))
for col, label in [("verbatim", "SCRIPT (what was said)"), ("human_a", "HUMAN A"),
                   ("human_b", "HUMAN B"), ("ai_model", "MODEL")]:
    print(f"── {label}\n{textwrap.fill(clip1[col], 100)}\n")
''')
code(r'''
# Normalise before comparing: case, punctuation, contractions, numbers, and a few
# spelling conventions (hio/hiyo, ndo/ndio). A compact copy of our evaluation's rules.
CONTRACTIONS = {"it's": "it is", "that's": "that is", "we're": "we are", "let's": "let us",
                "i'm": "i am", "you're": "you are", "don't": "do not", "can't": "can not"}
FOLD = {**{w: "h" + w for w in "ii iyo izi izo uyu uyo awa ao apa apo uku uko ivi ivyo ata adi".split()},
        "hio": "hiyo", "ndo": "ndio", "ndiyo": "ndio", "practise": "practice", "maths": "math"}
NUMS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()

def norm(s):
    s = re.sub(r"[^\w'\-\s]", " ", s.lower().replace("’", "'"))
    s = re.sub(r"(?<!\w)['\-]|['\-](?!\w)", " ", s)
    words = [FOLD.get(w, w) for t in s.split() for w in CONTRACTIONS.get(t, t).split()]
    return [n for w in words for n in
            ([NUMS[int(w)]] if w.isdigit() and int(w) <= 20 else [NUMS[int(d)] for d in w] if w.isdigit() else [w])]

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); return edit_distance(r, h) / len(r)
def cer(ref, hyp):
    r, h = " ".join(norm(ref)), " ".join(norm(hyp)); return edit_distance(r, h) / len(r)

NAMES = {"verbatim": "script", "human_a": "human A", "human_b": "human B",
         "ai_model": "model", "ai_model_run2": "model, 2nd run"}
pairs = [("verbatim", "human_a"), ("verbatim", "human_b"), ("verbatim", "ai_model"),
         ("human_a", "human_b"), ("human_a", "ai_model"), ("human_b", "ai_model"),
         ("ai_model", "ai_model_run2")]
asr = pd.DataFrame([{"reference": NAMES[r], "compared with": NAMES[h],
                     "CER": cer(clip1[r], clip1[h]), "WER": wer(clip1[r], clip1[h])} for r, h in pairs])
asr.round(3)
''')
md(r"""
Read the table from the top:
- **Against the truth, the model does about as well as the careful human** (CER 0.06 each), and better than human B, who translated a line into English.
- **The two humans disagree with each other more than the model errs.** If human B were your only reference, the model would look about three times worse than it is. Much of what gets called "model error" is the **reference**.
- **The model barely disagrees with itself.** Two runs on the same audio differ by about 1% of characters. Two people transcribing the same audio don't come close to that.

This is the same pattern we found on real sessions: two transcribers of the same recording agreed less with each other than either agreed with the model.
""")
code(r'''
per_clip = pd.DataFrame({"CER: model vs script": [cer(c.verbatim, c.ai_model) for c in asr_clips.itertuples()],
                         "CER: model vs its 2nd run": [cer(c.ai_model, c.ai_model_run2) for c in asr_clips.itertuples()]},
                        index=asr_clips.clip_id)
print(per_clip.round(3).to_string(), "\n")
print("mean:", per_clip.mean().round(3).to_dict())
''')
code(r'''
import difflib
print("Every place the model's clip-1 transcript differs from the script:")
r, h = norm(clip1.verbatim), norm(clip1.ai_model)
for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, r, h, autojunk=False).get_opcodes():
    if op != "equal":
        print(f"   said: {' '.join(r[i1:i2]) or '(nothing)':12s} → model wrote: {' '.join(h[j1:j2]) or '(nothing)'}")
''')
code(r'''
print("Why CER (characters) and not just WER (words) for code-switched speech:")
for ref, hyp in [("sawa sawa", "sawasawa"), ("manze", "maze"), ("nilifeel poa", "nilifil poa")]:
    print(f"   '{ref}' vs '{hyp}':  WER {wer(ref, hyp):.2f}   CER {cer(ref, hyp):.2f}")
''')
md(r"""
Notice three things:
- **The Sheng and Swahili came through almost untouched.** The model's differences are spelling conventions (`sawa sawa` / `sawasawa`, `aah` / `ah`) plus **one real error: it wrote a whole sentence twice.** Repeating a span is a known failure of this kind of model, and it matters downstream: a judge counting how often the Fellow asked about feelings would count that question twice.
- **WER punishes a one-letter spelling difference in Sheng as a whole wrong word.** CER is fairer to languages without a fixed spelling, so lead with CER.
- **Be honest about what this test can't show.** This audio is clean: one speaker at a time, no overlap, no classroom noise. It is easier than a real session, so treat these error rates as a floor, not an estimate.

The question that matters is the downstream one: *does a worse transcript change the fidelity score?* Test it directly: score the same sessions from a human transcript and from the model transcript, and compare the scores pairwise.

**Lesson 3.** Evaluate each stage on its own, and check how good the reference is before blaming the model. Then ask whether the stage's errors change the final output at all.
""")

# ---------------------------------------------------------------- step 4
md(r"""
---
## Step 4 · Score the judge against the humans

Now the main event: the LLM judge's six scores (prompt v1, first run) against the average of the two human raters.
""")
code(r'''
v1 = ai_scores[ai_scores.prompt_version == "v1"]
ai1 = v1[v1.draw == 1].set_index("session_id")
hmean = human.groupby("session_id")[DIMS + ["composite"]].mean()
ai1 = ai1.loc[hmean.index]
ai1["composite"] = ai1[DIMS].mean(axis=1)

judge = pd.DataFrame([{"dimension": d, "AI mean": ai1[d].mean(), "human mean": hmean[d].mean(),
                       "bias (AI−human)": ai1[d].mean() - hmean[d].mean(),
                       "r": stats.pearsonr(ai1[d], hmean[d])[0]} for d in DIMS + ["composite"]]).set_index("dimension")
judge.round(2)
''')
code(r'''
r_comp = judge.loc["composite", "r"]
print(f"Composite r = {r_comp:.2f} against a ceiling of {ceiling_2:.2f}  ->  {r_comp / ceiling_2:.0%} of what is achievable")
print("The ranking is real. The calibration is not: the AI sits "
      f"{-judge.loc['composite', 'bias (AI−human)']:.1f} points below the humans on a 1–7 scale.\n")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
for ax, d in zip(axes, ["protocol_adherence", "clarity_accessibility"]):
    h = human[d].value_counts(normalize=True).reindex(range(1, 8), fill_value=0)
    a = ai1[d].value_counts(normalize=True).reindex(range(1, 8), fill_value=0)
    ax.bar(np.arange(1, 8) - .2, h, .4, label="human", color="#8a8f98")
    ax.bar(np.arange(1, 8) + .2, a, .4, label="AI judge", color="#2f6fde")
    ax.set_title(d.replace("_", " ")); ax.set_xticks(range(1, 8)); ax.set_xlabel("score")
axes[0].set_ylabel("share of sessions"); axes[0].legend(frameon=False, loc="upper center")
plt.tight_layout(); plt.show()
''')
code(r'''
no_adh = [d for d in DIMS if d != "protocol_adherence"]
r_drop = stats.pearsonr(ai1[no_adh].mean(axis=1), hmean[no_adh].mean(axis=1))[0]
print(f"protocol_adherence: AI never goes above {ai1.protocol_adherence.max()}, humans mostly give 6–7, r = {judge.loc['protocol_adherence','r']:.2f}")
print(f"Composite r with it: {r_comp:.2f}   without it: {r_drop:.2f}")
''')
md(r"""
One dimension has **collapsed**: the judge gives nearly every session a 2 or 3, and it carries no information. Reading the prompt explains it. The rubric said *"High score ONLY if all protocol details were addressed"*. Human raters read that as guidance; the model read it literally.

**Lesson 4.** Break the aggregate down. A healthy-looking composite can hide one broken item that is actively subtracting signal. Read the prompt next to the failure.
""")

# ---------------------------------------------------------------- step 5
md(r"""
---
## Step 5 · Two traps that almost fooled us

### Trap A: forgetting the trivial baseline
The judge scores well below the humans (Step 4). Tempting fix: add a constant offset, then show that the corrected scores are close to the humans. But compare against the dumbest possible predictor first.
""")
code(r'''
offset = (hmean.composite - ai1.composite).mean()
corrected = ai1.composite + offset
constant = pd.Series(hmean.composite.median(), index=hmean.index)
mae = lambda p: np.abs(p - hmean.composite).mean()
print(f"MAE, raw AI score:                 {mae(ai1.composite):.2f}")
print(f"MAE, offset-corrected AI score:    {mae(corrected):.2f}   <- looks great")
print(f"MAE, always guess the median:      {mae(constant):.2f}   <- ignores the AI entirely, and wins")
''')
md(r"""
When human scores pile up near the top of the scale, guessing the middle beats everything on absolute error. MAE was the wrong metric for this question. What the judge does well is **rank** sessions, and you lose that the moment you report a corrected number.

### Trap B: one run is not a measurement
LLM judges are not deterministic. Here is the *same* judge, on *byte-identical* input, three times.
""")
code(r'''
v1w = v1.set_index(["session_id", "draw"])[DIMS].unstack("draw").loc[hmean.index]
moved = np.mean([(v1w[d].max(axis=1) != v1w[d].min(axis=1)).mean() for d in DIMS])
print(f"Item scores that changed between identical runs: {moved:.0%}")

per_draw = {k: stats.pearsonr(v1[v1.draw == k].set_index("session_id").loc[hmean.index, DIMS].mean(axis=1),
                              hmean.composite)[0] for k in (1, 2, 3)}
print("\nComposite r vs humans, all sessions:")
for k, r in per_draw.items(): print(f"   run {k}: {r:.3f}")

pilot = hmean.sample(30, random_state=1).index      # a typical first eval set is small
pilot_r = {k: stats.pearsonr(v1w.loc[pilot].xs(k, axis=1, level=1).mean(axis=1), hmean.loc[pilot, "composite"])[0] for k in (1, 2, 3)}
print("\nSame thing on a 30-session pilot sample:")
for k, r in pilot_r.items(): print(f"   run {k}: {r:.3f}")
spread = max(pilot_r.values()) - min(pilot_r.values())
print(f"Spread across identical runs: {spread:.2f} points of correlation")
''')
md(r"""
If you had changed one setting between run 1 and run 2, you would have reported an improvement (or a regression) that is pure noise. Setting temperature to 0 reduces this but **does not remove it**.

**Lesson 5.** Always beat the trivial baseline before claiming success. Never compare two configurations on a single run's correlation: use several runs, or a paired test (next step).
""")

# ---------------------------------------------------------------- step 6
md(r"""
---
## Step 6 · Compare two versions the right way

Prompt **v2** rewrites the over-strict adherence clause and changes nothing else. Did it help? Compare the two versions **session by session** (a paired test), not by their overall averages.
""")
code(r'''
v2 = ai_scores[(ai_scores.prompt_version == "v2")].set_index("session_id").loc[hmean.index]
paired = pd.DataFrame({d: v2[d] - ai1[d] for d in DIMS})
paired["composite"] = v2[DIMS].mean(axis=1) - ai1[DIMS].mean(axis=1)

def sign_test(diff):
    w, l = (diff > 0).sum(), (diff < 0).sum()
    return w, l, stats.binomtest(w, w + l).pvalue if w + l else np.nan

out = []
for d in DIMS + ["composite"]:
    w, l, p = sign_test(paired[d])
    out.append({"dimension": d, "mean change": paired[d].mean(), "better": w, "worse": l, "p (sign test)": p})
pd.DataFrame(out).set_index("dimension").round(3)
''')
code(r'''
# How big must a change be before we can trust it? Use two v1 runs (no real change) to measure pure noise.
noise = (v1w[DIMS].xs(2, axis=1, level=1).mean(axis=1) - v1w[DIMS].xs(1, axis=1, level=1).mean(axis=1))
sd, n = noise.std(), len(noise)
rng = np.random.default_rng(0)
print(f"Run-to-run noise in the composite: SD {sd:.2f} points (n = {n} sessions)\n")
print("Chance of detecting a true change of size δ (paired sign test, p < 0.05):")
for delta in [0.05, 0.1, 0.2, 0.3]:
    sims = delta + rng.normal(0, sd, (2000, n))
    power = np.mean([stats.binomtest((s > 0).sum(), (s != 0).sum()).pvalue < .05 for s in sims[:400]])
    print(f"   δ = {delta:.2f}:  {power:.0%}")
''')
md(r"""
v2 fixes the collapsed item and leaves the others alone: exactly what a targeted change should do. The power table tells you, **before you run anything**, the smallest change worth chasing. Write that number down in advance, and don't celebrate wins below it.

**Lesson 6.** Change one thing at a time, compare session by session, and pre-register the smallest effect you would act on.
""")

# ---------------------------------------------------------------- step 7
md(r"""
---
## Step 7 · Change what you report

The judge ranks sessions reasonably and calibrates them poorly. So don't show supervisors a rubric number. Show a **rank**, and a short **"listen to this one"** list: the bottom 10% of sessions by AI score.
""")
code(r'''
# Average the three v1 runs (cheap way to cut noise) and drop the broken item
ai_avg = v1.groupby("session_id")[no_adh].mean().loc[hmean.index].mean(axis=1)
truth_low = hmean.composite <= hmean.composite.quantile(.10)          # human bottom decile
flag = ai_avg <= ai_avg.quantile(.10)                                  # AI bottom decile

auc = roc_auc_score(truth_low, -ai_avg)
recall = (flag & truth_low).sum() / truth_low.sum()
precision = (flag & truth_low).sum() / flag.sum()
print(f"AUC for finding the sessions humans rated lowest: {auc:.2f}  (0.5 = coin flip)")
print(f"'Listen to this one' list ({flag.sum()} sessions): catches {recall:.0%} of the human bottom decile,")
print(f"   {precision:.0%} of flagged sessions are truly in it, {precision / truth_low.mean():.1f}x better than picking at random")
''')
md(r"""
A supervisor who listens to the flagged sessions finds weak sessions several times faster than by picking at random. A supervisor shown "3.2 out of 7" for a session humans would rate 6 just stops trusting the tool.

**Lesson 7.** Ship the output your evaluation supports. If the evidence supports ranking and triage, report a rank and a flag, not a score that pretends to more precision than it has.
""")

# ---------------------------------------------------------------- step 8
md(r"""
---
## Step 8 · Safety flags: a different kind of test

Risk disclosures are rare in real sessions, so natural data can't tell you if the flagger works. Build a **seeded test set**: genuine current-risk cases written in realistic ways (plain, indirect, in Sheng, split across turns) mixed with look-alikes (a resolved story from the past, a role-play, a textbook example). Here the content is described in words only.

The gate is **asymmetric**: one missed real case blocks release, and false alarms are a tuning problem.
""")
code(r'''
def safety_report(flags):
    tp = (flags & safety.is_current_risk).sum(); fn = (~flags & safety.is_current_risk).sum()
    fp = (flags & ~safety.is_current_risk).sum()
    return {"recall on real risk": f"{tp / (tp + fn):.0%}", "real cases missed": int(fn), "false alarms": int(fp)}

raw_flags = safety.ai_flagged
filtered = safety.ai_flagged & (safety.ai_timeframe == "current")   # use the model's own timeframe label
print(pd.DataFrame({"escalate every flag": safety_report(raw_flags),
                    "escalate 'current' only": safety_report(filtered)}).T.to_string())
print("\nMissed case(s):")
print(safety[safety.is_current_risk & ~safety.ai_flagged][["case_id", "scenario"]].to_string(index=False))
''')
md(r"""
Asking the model *when* the event happened, and only escalating "current" flags, cuts the false alarms sharply without losing any real cases. But one real case was missed, the one **split across several turns**, so this version **does not ship**. That case goes into the test set permanently, and every future version must catch it.

**Lesson 8.** For rare, high-stakes failures, build the test set yourself, gate asymmetrically, and turn every miss into a permanent regression test.
""")

# ---------------------------------------------------------------- wrap
md(r"""
---
## Wrap-up · the scorecard from this run
""")
code(r'''
scorecard = pd.DataFrame([
    ("1 Reference audited", f"{len(dup_groups)} duplicate recordings, {len(bad_total)} bad totals, {len(missing)} missing rows fixed", "Done"),
    ("2 Human ceiling", f"ICC {icc:.2f}; best possible r vs two raters ≈ {ceiling_2:.2f}", "Report against this"),
    ("3 Transcripts", f"Two humans disagree (CER {asr.iloc[3].CER:.2f}) more than the model errs (CER {asr.iloc[2].CER:.2f})", "Check the reference"),
    ("4 Judge vs humans", f"r {r_comp:.2f} = {r_comp / ceiling_2:.0%} of ceiling; adherence item collapsed", "Fix or drop the item"),
    ("5 Traps", f"Constant beats corrected score on MAE; identical runs on 30 sessions span r by {spread:.2f}", "Multiple runs, paired tests"),
    ("6 Prompt v2", f"Composite +{paired.composite.mean():.2f}, driven by adherence", "Adopt; re-check ranking"),
    ("7 Reporting", f"Bottom-decile flag: AUC {auc:.2f}, {precision / truth_low.mean():.1f}x lift", "Ship rank + flag"),
    ("8 Safety", f"Recall {safety_report(filtered)['recall on real risk']}, 1 miss (split across turns)", "BLOCK release"),
], columns=["step", "what we found", "decision"]).set_index("step")
scorecard
''')
md(r"""
### Eight habits to take home
1. Audit and freeze your reference data before scoring against it.
2. Measure human agreement first; report AI agreement against that ceiling.
3. Evaluate each stage on its own, and check the quality of its reference.
4. Break aggregates down by dimension; read the prompt next to every failure.
5. Beat the trivial baseline, and never trust a single run.
6. Change one thing at a time and compare session by session.
7. Report what the evidence supports: rank and triage, not false precision.
8. Seed rare, high-stakes failures yourself, and gate asymmetrically.

### Try it on your own system this month
- [ ] Write down what "good" looks like, and who decides
- [ ] Collect a small reference set, with **two** independent human judgments on at least part of it
- [ ] Compute human–human agreement before you compute anything about the AI
- [ ] Run your AI three times on the same inputs and look at the spread
- [ ] Build 20 seeded cases for your worst-case failure

**Further reading:** the [Living Playbook for AI Evaluation in the Social Sector](https://eval.playbook.org.ai/) (The Agency Fund, IDinsight and CGD) frames this as four levels: model, product, user and impact. Everything in this notebook is level 1, *does the AI behave as intended*, which has to hold before the other three can be measured.
""")

# ---------------------------------------------------------------- write
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": [], "toc_visible": True}},
      "cells": []}
for i, (kind, src) in enumerate(cells):
    src = src.replace("__REPO_RAW__", REPO_RAW).replace("__ASR_CSV__", (ROOT / "data/asr_demo.csv").read_text())
    lines = src.splitlines(True)
    cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": lines}
    if kind == "code": cell.update({"execution_count": None, "outputs": []})
    nb["cells"].append(cell)
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print("cells:", len(cells))
