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

**How to follow along:** in Google Colab, choose *Runtime → Run all*, then scroll. Or run one cell at a time with **Shift + Enter**. Cells marked **Your turn** invite you to change something and re-run it. You don't need to read the code: the text around each cell says what it does and what to look for.

### Words you'll see

| term | what it means here |
|---|---|
| **Fellow** | the young facilitator who leads a group session |
| **fidelity** | how closely, and how well, a session follows the intervention's protocol |
| **rubric** | the checklist supervisors score: 6 items, each from 1 (poor) to 7 (excellent) |
| **composite** | the average of the 6 rubric items: one overall score per session |
| **transcription (ASR)** | turning the audio into text. ASR stands for automatic speech recognition |
| **LLM judge** | a large language model that reads the transcript and scores it against the rubric |
| **reference** | the human scores (or transcripts) we compare the AI against |
| **correlation (r)** | from −1 to 1: do sessions the humans score higher also get higher AI scores? 1 is perfect agreement on order, 0 is none |
""")

md(r"""
## Setup
Run this cell first. It builds the synthetic dataset: 180 group sessions, two human ratings each, an AI judge run three times on identical input, a second prompt version, five transcribed audio clips, and a seeded safety test set. You don't need to read it. In Colab it shows as a collapsed form.
""")

code(r'''
# @title Setup: build the synthetic dataset (run me, no need to read) { display-mode: "form" }
import numpy as np, pandas as pd, matplotlib.pyplot as plt, re, warnings
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score
warnings.filterwarnings("ignore")
pd.set_option("display.precision", 3); pd.set_option("display.width", 120); pd.set_option("display.max_colwidth", 90); pd.set_option("display.max_columns", None)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

RUBRIC_ITEMS = ["protocol_adherence", "content_specifications", "thoroughness",
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
            rows.append({"session_id": ids[i], "rater_slot": slot, "rater_id": r, **dict(zip(RUBRIC_ITEMS, y))})
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
                ai.append({"session_id": ids[i], "prompt_version": version, "run": d, **dict(zip(RUBRIC_ITEMS, sc[i]))})
    draws(mu_v1, load, "v1", 3)
    # --- prompt v2: the over-strict adherence clause rewritten; everything else unchanged
    mu_v2 = mu_v1.copy(); mu_v2[0] = 4.9
    load_v2 = load.copy(); load_v2[0] = .7
    draws(mu_v2, load_v2, "v2", 1)
    ai = pd.DataFrame(ai)
    for c in RUBRIC_ITEMS: ai[c] = ai[c].astype(int)

    # --- plant the data-hygiene problems we actually met
    raw = human.copy()
    for c in RUBRIC_ITEMS: raw[c] = raw[c].astype("Int64")
    raw["overall_score"] = raw[RUBRIC_ITEMS].sum(axis=1).astype("Int64")
    bad_total = rng.choice(len(raw), 6, replace=False)
    raw.loc[bad_total, "overall_score"] = raw.loc[bad_total, "overall_score"] - rng.choice([-10, 10, 7], 6)
    missing = rng.choice(len(raw), 3, replace=False)
    for k, m in enumerate(missing): raw.loc[m, RUBRIC_ITEMS[k * 2]] = pd.NA
    dup_src = rng.choice(N, 4, replace=False)      # the same recording filed under a second id
    dups, extra_sessions = [], []
    for k, s in enumerate(dup_src):
        new_id = f"S{N + 1 + k:03d}"
        extra = sessions.iloc[s].copy(); extra["session_id"] = new_id; extra_sessions.append(extra)
        d = raw[raw.session_id == ids[s]].copy(); d["session_id"] = new_id; dups.append(d)
    raw = pd.concat([raw] + dups, ignore_index=True)
    sessions_raw = pd.concat([sessions, pd.DataFrame(extra_sessions)], ignore_index=True)
    return sessions_raw, raw, ai

sessions_raw, human_ratings_raw, ai_judge_scores = _build()

# --- five synthetic clips: scripted, spoken by text-to-speech, transcribed by a real model (see Step 3)
REPO_FILES_URL = "__REPO_RAW__"   # public raw-file URL of the repo; empty means use the copy embedded below
try:
    transcription_clips = pd.read_csv(REPO_FILES_URL + "data/asr_demo.csv")
except Exception:
    import io; transcription_clips = pd.read_csv(io.StringIO(r"""__ASR_CSV__"""))
transcription_clips = transcription_clips.fillna("")

# --- seeded safety test set: 20 genuine current-risk cases, 40 look-alikes (no real disclosures; labels only)
_r = np.random.default_rng(7)
_kinds = (["current distress, stated plainly"] * 8 + ["current distress, indirect wording"] * 6 +
          ["current distress, in Sheng"] * 4 + ["current distress, split across turns"] * 2)
_look = (["story about a past, resolved event"] * 16 + ["hypothetical in a role-play"] * 12 +
         ["teaching example from the manual"] * 12)
safety_test_cases = pd.DataFrame({"case_id": [f"C{i:02d}" for i in range(1, 61)],
                                  "scenario": _kinds + _look,
                                  "is_current_risk": [True] * 20 + [False] * 40})
safety_test_cases["ai_flagged"] = True
safety_test_cases.loc[[19], "ai_flagged"] = False                       # misses one split-across-turns case
safety_test_cases.loc[_r.choice(range(20, 60), 13, replace=False), "ai_flagged"] = False
safety_test_cases["ai_timeframe"] = np.where(safety_test_cases.is_current_risk, "current",
    safety_test_cases.scenario.map({"story about a past, resolved event": "past",
                                    "hypothetical in a role-play": "hypothetical",
                                    "teaching example from the manual": "educational"}))
safety_test_cases.loc[[24, 41], "ai_timeframe"] = "current"            # two look-alikes the model mislabels
safety_test_cases.loc[~safety_test_cases.ai_flagged, "ai_timeframe"] = None

print("Built these tables:")
print(f"  human_ratings_raw     {human_ratings_raw.shape[0]} human ratings over {human_ratings_raw.session_id.nunique()} session ids")
print(f"  ai_judge_scores       {ai_judge_scores.shape[0]} AI scores (prompt v1 run 3 times, prompt v2 run once)")
print(f"  transcription_clips   {len(transcription_clips)} synthetic recordings, each transcribed twice by a real model")
print(f"  safety_test_cases     {len(safety_test_cases)} seeded test cases")
''')

# ---------------------------------------------------------------- step 1
md(r"""
---
## Step 1 · Audit the reference before you trust it

Every number in an evaluation is measured *against* something: here, two human supervisors' ratings of each session. That reference is a measuring instrument, and it needs quality control like any other.

Four checks that caught real problems for us:
1. **Duplicates.** Is the same recording filed under two session ids? Then your "held-out" test set may contain sessions you already tuned on.
2. **Missing scores.** Ratings with a blank rubric item.
3. **Totals that don't add up.** Is the stored `overall_score` really the sum of the six items?
4. **Who did the rating.** If one person rated most sessions, your "human standard" is that one person's habits.
""")
code(r'''
print("Rating rows:", len(human_ratings_raw), "| session ids:", human_ratings_raw.session_id.nunique())
human_ratings_raw.head()
''')
code(r'''
# 1. The same recording filed under more than one session id
session_ids_per_recording = sessions_raw.groupby("audio_sha").session_id.agg(list)
duplicate_recordings = session_ids_per_recording[session_ids_per_recording.str.len() > 1]
print(f"1. Duplicate recordings: {len(duplicate_recordings)} groups")
for session_ids in duplicate_recordings: print("     ", " = ".join(session_ids))

# 2. Ratings with a blank rubric item
rows_missing_a_score = human_ratings_raw[human_ratings_raw[RUBRIC_ITEMS].isna().any(axis=1)]
print(f"\n2. Rows with a missing item score: {len(rows_missing_a_score)}")

# 3. Stored total vs the total we recompute from the six items
recomputed_total = human_ratings_raw[RUBRIC_ITEMS].sum(axis=1, min_count=6)
rows_with_wrong_total = human_ratings_raw[(recomputed_total.notna()) & (recomputed_total != human_ratings_raw.overall_score)]
print(f"\n3. Rows where overall_score != sum of items: {len(rows_with_wrong_total)}")

# 4. What share of all ratings each rater did
print("\n4. Share of ratings by rater:")
print((human_ratings_raw.rater_id.value_counts(normalize=True) * 100).round(0).astype(int).astype(str).add("%").to_string())
''')
code(r'''
# Clean: keep one id per recording, drop sessions without two complete ratings, recompute totals
duplicate_ids_to_drop = [session_id for session_ids in duplicate_recordings for session_id in session_ids[1:]]
human_ratings = human_ratings_raw[~human_ratings_raw.session_id.isin(duplicate_ids_to_drop)].copy()
has_two_complete_ratings = human_ratings.groupby("session_id")[RUBRIC_ITEMS].apply(
    lambda ratings: ratings.notna().all().all() and len(ratings) == 2)
human_ratings = human_ratings[human_ratings.session_id.isin(has_two_complete_ratings[has_two_complete_ratings].index)].copy()
human_ratings[RUBRIC_ITEMS] = human_ratings[RUBRIC_ITEMS].astype(int)
human_ratings["composite"] = human_ratings[RUBRIC_ITEMS].mean(axis=1)
print(f"Clean reference: {human_ratings.session_id.nunique()} sessions, each rated by two people")
''')
md(r"""
**Lesson 1.** Your golden dataset (the reference you trust most) is a measuring instrument. Audit it, fix it, and freeze it so it can't change silently *before* you score anything against it.

**Your turn:** look at check 4. If rater R1 did 40% of the ratings and is unusually lenient, what happens to an AI that is "calibrated to the humans"?
""")

# ---------------------------------------------------------------- step 2
md(r"""
---
## Step 2 · Measure the humans first

Before asking "does the AI agree with humans?", ask **"do humans agree with each other?"** Two trained supervisors rated every session independently, so we can measure that directly.

How to read the table below:
- **same score % / within 1 point %:** how often the two supervisors gave exactly the same score, or scores at most one point apart
- **QWK** (quadratic weighted kappa): agreement after removing what you'd get by luck. 1 is perfect, 0 is no better than chance
- **ICC** (intraclass correlation): how much of the spread in scores comes from real differences between sessions rather than from which rater happened to score them. 1 is all real, 0 is all rater noise
""")
code(r'''
def intraclass_correlation(scores):
    """ICC(A,1): two-way, absolute agreement, single rater. scores: one row per session, one column per rater."""
    x = np.asarray(scores, float); n, k = x.shape; gm = x.mean()
    msr = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    mse = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

ratings_side_by_side = human_ratings.pivot(index="session_id", columns="rater_slot")
agreement_rows = []
for item in RUBRIC_ITEMS + ["composite"]:
    rater_1, rater_2 = ratings_side_by_side[(item, 1)].values, ratings_side_by_side[(item, 2)].values
    is_item = item != "composite"
    agreement_rows.append({
        "rubric item": item,
        "same score %": np.mean(np.rint(rater_1) == np.rint(rater_2)) * 100 if is_item else np.nan,
        "within 1 point %": np.mean(np.abs(rater_1 - rater_2) <= 1) * 100 if is_item else np.nan,
        "QWK": cohen_kappa_score(rater_1.astype(int), rater_2.astype(int), weights="quadratic") if is_item else np.nan,
        "ICC": intraclass_correlation(np.c_[rater_1, rater_2])})
human_agreement = pd.DataFrame(agreement_rows).set_index("rubric item")
human_agreement.round(2)
''')
code(r'''
human_icc = human_agreement.loc["composite", "ICC"]
share_scored_6_or_7 = (human_ratings[RUBRIC_ITEMS].values >= 6).mean()
print(f"{share_scored_6_or_7:.0%} of all item scores are a 6 or a 7: most of the scale goes unused.")
print(f"Composite ICC between two trained raters: {human_icc:.2f}")

# Spearman-Brown: how reliable the AVERAGE of several raters would be
spearman_brown = lambda reliability, n_raters: n_raters * reliability / (1 + (n_raters - 1) * reliability)
print("\nReliability if we averaged several raters:")
for n_raters in [1, 2, 3, 5, 10]: print(f"   {n_raters:>2} raters: {spearman_brown(human_icc, n_raters):.2f}")

ceiling_vs_one_rater = np.sqrt(human_icc)                          # best possible correlation of ANY judge with one rater
ceiling_vs_two_raters = np.sqrt(spearman_brown(human_icc, 2))      # ... with the average of two raters
print(f"\nBest possible correlation with one rater:          r ≤ {ceiling_vs_one_rater:.2f}")
print(f"Best possible correlation with two raters' average: r ≤ {ceiling_vs_two_raters:.2f}")
''')
md(r"""
This is the **ceiling**. Even a *perfect* AI could not correlate with these humans above it, because the humans don't agree with each other. So an AI scoring r = 0.45 here is not "mediocre". It may be close to the best possible.

**Lesson 2.** Measure human agreement first, and report every AI-vs-human number **next to that ceiling, never against a perfect 1.0**. If the ceiling is low, the cheapest improvement may be rater training and worked examples in the rubric, not a better model.
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

Listen to clip 1, then compare the four versions of it.
""")
code(r'''
import textwrap
from IPython.display import Audio
clip_1 = transcription_clips.set_index("clip_id").loc["clip_01"]
if REPO_FILES_URL: display(Audio(REPO_FILES_URL + "audio/clip_01.mp3"))
for column, label in [("verbatim", "SCRIPT (what was said)"), ("human_a", "HUMAN A"),
                      ("human_b", "HUMAN B"), ("ai_model", "MODEL")]:
    print(f"── {label}\n{textwrap.fill(clip_1[column], 100)}\n")
''')
md(r"""
To compare two transcripts we count how much you'd have to change to turn one into the other:
- **CER** (character error rate): the share of *letters* that differ. 0 means identical, 0.10 means about 10% of letters differ.
- **WER** (word error rate): the same, counted in *whole words*.

Both are measured after tidying away differences that don't matter: capital letters, punctuation, "we're" vs "we are", "3" vs "three", and a few spelling variants such as *hio* / *hiyo*.
""")
code(r'''
# Tidy away differences that don't matter. A compact copy of our evaluation's rules.
CONTRACTIONS = {"it's": "it is", "that's": "that is", "we're": "we are", "let's": "let us",
                "i'm": "i am", "you're": "you are", "don't": "do not", "can't": "can not"}
SPELLING_VARIANTS = {**{w: "h" + w for w in "ii iyo izi izo uyu uyo awa ao apa apo uku uko ivi ivyo ata adi".split()},
                     "hio": "hiyo", "ndo": "ndio", "ndiyo": "ndio", "practise": "practice", "maths": "math"}
NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()

def tidy_words(text):
    text = re.sub(r"[^\w'\-\s]", " ", text.lower().replace("’", "'"))
    text = re.sub(r"(?<!\w)['\-]|['\-](?!\w)", " ", text)
    words = [SPELLING_VARIANTS.get(w, w) for t in text.split() for w in CONTRACTIONS.get(t, t).split()]
    return [n for w in words for n in
            ([NUMBER_WORDS[int(w)]] if w.isdigit() and int(w) <= 20
             else [NUMBER_WORDS[int(d)] for d in w] if w.isdigit() else [w])]

def edit_distance(a, b):
    """The fewest insertions, deletions and substitutions that turn a into b."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def word_error_rate(reference, other):
    ref, hyp = tidy_words(reference), tidy_words(other)
    return edit_distance(ref, hyp) / len(ref)

def character_error_rate(reference, other):
    ref, hyp = " ".join(tidy_words(reference)), " ".join(tidy_words(other))
    return edit_distance(ref, hyp) / len(ref)

SOURCE_NAMES = {"verbatim": "script", "human_a": "human A", "human_b": "human B",
                "ai_model": "model", "ai_model_run2": "model, 2nd run"}
comparisons = [("verbatim", "human_a"), ("verbatim", "human_b"), ("verbatim", "ai_model"),
               ("human_a", "human_b"), ("human_a", "ai_model"), ("human_b", "ai_model"),
               ("ai_model", "ai_model_run2")]
transcript_agreement = pd.DataFrame([
    {"reference": SOURCE_NAMES[ref], "compared with": SOURCE_NAMES[other],
     "CER": character_error_rate(clip_1[ref], clip_1[other]),
     "WER": word_error_rate(clip_1[ref], clip_1[other])} for ref, other in comparisons])
transcript_agreement.round(3)
''')
md(r"""
Read the table from the top:
- **Against the truth, the model does about as well as the careful human** (CER 0.06 each), and better than human B, who translated a line into English.
- **The two humans disagree with each other more than the model errs.** If human B were your only reference, the model would look about three times worse than it is. Much of what gets called "model error" is the **reference**.
- **The model barely disagrees with itself.** Two runs on the same audio differ by about 1% of characters. Two people transcribing the same audio don't come close to that.

This is the same pattern we found on real sessions: two transcribers of the same recording agreed less with each other than either agreed with the model.

The same check on all five clips:
""")
code(r'''
error_rate_per_clip = pd.DataFrame({
    "CER: model vs script": [character_error_rate(c.verbatim, c.ai_model) for c in transcription_clips.itertuples()],
    "CER: model vs its 2nd run": [character_error_rate(c.ai_model, c.ai_model_run2) for c in transcription_clips.itertuples()]},
    index=transcription_clips.clip_id)
print(error_rate_per_clip.round(3).to_string(), "\n")
print("average:", error_rate_per_clip.mean().round(3).to_dict())
''')
code(r'''
import difflib
print("Every place the model's clip-1 transcript differs from the script:")
said_words, model_words = tidy_words(clip_1.verbatim), tidy_words(clip_1.ai_model)
for change, s1, s2, m1, m2 in difflib.SequenceMatcher(None, said_words, model_words, autojunk=False).get_opcodes():
    if change != "equal":
        print(f"   said: {' '.join(said_words[s1:s2]) or '(nothing)':12s} → model wrote: {' '.join(model_words[m1:m2]) or '(nothing)'}")
''')
code(r'''
print("Why CER (letters) and not just WER (words) for code-switched speech:")
for said, written in [("sawa sawa", "sawasawa"), ("manze", "maze"), ("nilifeel poa", "nilifil poa")]:
    print(f"   '{said}' vs '{written}':  WER {word_error_rate(said, written):.2f}   CER {character_error_rate(said, written):.2f}")
''')
md(r"""
Notice three things:
- **The Sheng and Swahili came through almost untouched.** The model's differences are spelling conventions (`sawa sawa` / `sawasawa`, `aah` / `ah`) plus **one real error: it wrote a whole sentence twice.** Repeating a span is a known failure of this kind of model, and it matters downstream: a judge counting how often the Fellow asked about feelings would count that question twice.
- **WER punishes a one-letter spelling difference in Sheng as a whole wrong word.** CER is fairer to languages without a fixed spelling, so lead with CER.
- **Be honest about what this test can't show.** This audio is clean: one speaker at a time, no overlap, no classroom noise. It is easier than a real session, so treat these error rates as a floor, not an estimate.

The question that matters is the downstream one: *does a worse transcript change the fidelity score?* Test it directly: score the same sessions from a human transcript and from the model transcript, and compare the scores session by session.

**Lesson 3.** Evaluate each stage on its own, and check how good the reference is before blaming the model. Then ask whether the stage's errors change the final output at all.
""")

# ---------------------------------------------------------------- step 4
md(r"""
---
## Step 4 · Score the judge against the humans

Now the main event: the LLM judge's six scores (prompt v1, first run) against the average of the two human raters.

In the table: **bias** is how far the AI's average sits from the humans' average (negative means the AI scores lower), and **correlation (r)** asks whether the AI puts sessions in the same order as the humans.
""")
code(r'''
judge_v1_all_runs = ai_judge_scores[ai_judge_scores.prompt_version == "v1"]
human_average = human_ratings.groupby("session_id")[RUBRIC_ITEMS + ["composite"]].mean()
judge_v1_run1 = judge_v1_all_runs[judge_v1_all_runs.run == 1].set_index("session_id").loc[human_average.index]
judge_v1_run1["composite"] = judge_v1_run1[RUBRIC_ITEMS].mean(axis=1)

judge_vs_humans = pd.DataFrame([{
    "rubric item": item,
    "AI average": judge_v1_run1[item].mean(),
    "human average": human_average[item].mean(),
    "bias (AI − human)": judge_v1_run1[item].mean() - human_average[item].mean(),
    "correlation (r)": stats.pearsonr(judge_v1_run1[item], human_average[item])[0],
} for item in RUBRIC_ITEMS + ["composite"]]).set_index("rubric item")
judge_vs_humans.round(2)
''')
code(r'''
composite_correlation = judge_vs_humans.loc["composite", "correlation (r)"]
print(f"Composite r = {composite_correlation:.2f} against a ceiling of {ceiling_vs_two_raters:.2f}"
      f"  ->  {composite_correlation / ceiling_vs_two_raters:.0%} of what is achievable")
print("The ranking is real. The calibration is not: the AI sits "
      f"{-judge_vs_humans.loc['composite', 'bias (AI − human)']:.1f} points below the humans on a 1–7 scale.\n")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
for ax, item in zip(axes, ["protocol_adherence", "clarity_accessibility"]):
    human_share = human_ratings[item].value_counts(normalize=True).reindex(range(1, 8), fill_value=0)
    ai_share = judge_v1_run1[item].value_counts(normalize=True).reindex(range(1, 8), fill_value=0)
    ax.bar(np.arange(1, 8) - .2, human_share, .4, label="human", color="#8a8f98")
    ax.bar(np.arange(1, 8) + .2, ai_share, .4, label="AI judge", color="#2f6fde")
    ax.set_title(item.replace("_", " ")); ax.set_xticks(range(1, 8)); ax.set_xlabel("score")
axes[0].set_ylabel("share of sessions"); axes[0].legend(frameon=False, loc="upper center")
plt.tight_layout(); plt.show()
''')
code(r'''
items_without_adherence = [item for item in RUBRIC_ITEMS if item != "protocol_adherence"]
correlation_without_adherence = stats.pearsonr(judge_v1_run1[items_without_adherence].mean(axis=1),
                                               human_average[items_without_adherence].mean(axis=1))[0]
print(f"protocol_adherence: AI never goes above {judge_v1_run1.protocol_adherence.max()}, humans mostly give 6–7, "
      f"r = {judge_vs_humans.loc['protocol_adherence', 'correlation (r)']:.2f}")
print(f"Composite r with it: {composite_correlation:.2f}   without it: {correlation_without_adherence:.2f}")
''')
md(r"""
One rubric item has **collapsed**: the judge gives nearly every session a 2 or 3, so that score tells you nothing about the session. Reading the prompt explains it. The rubric said *"High score ONLY if all protocol details were addressed"*. Human raters read that as guidance; the model read it literally.

**Lesson 4.** Break the overall number down. A healthy-looking composite can hide one broken item that is actively dragging it down. Read the prompt next to the failure.
""")

# ---------------------------------------------------------------- step 5
md(r"""
---
## Step 5 · Two traps that almost fooled us

### Trap A: forgetting the trivial baseline
The judge scores well below the humans (Step 4). Tempting fix: add a constant to every AI score, then show that the corrected scores are close to the humans. But first compare against the dumbest possible predictor.

**MAE** (mean absolute error) is the average distance between a prediction and the human score, in rubric points. Lower is better.
""")
code(r'''
average_gap = (human_average.composite - judge_v1_run1.composite).mean()
ai_score_corrected = judge_v1_run1.composite + average_gap
always_guess_median = pd.Series(human_average.composite.median(), index=human_average.index)
mean_absolute_error = lambda prediction: np.abs(prediction - human_average.composite).mean()
print(f"MAE, raw AI score:                 {mean_absolute_error(judge_v1_run1.composite):.2f}")
print(f"MAE, AI score + constant:          {mean_absolute_error(ai_score_corrected):.2f}   <- looks great")
print(f"MAE, always guess the median:      {mean_absolute_error(always_guess_median):.2f}   <- ignores the AI entirely, and wins")
''')
md(r"""
When human scores pile up near the top of the scale, guessing the middle beats everything on average distance. MAE was the wrong measure for this question. What the judge does well is **rank** sessions, and you lose that the moment you report a corrected number.

### Trap B: one run is not a measurement
LLM judges don't give the same answer every time. Here is the *same* judge, on *identical* input, run three times.
""")
code(r'''
judge_v1_runs_side_by_side = (judge_v1_all_runs.set_index(["session_id", "run"])[RUBRIC_ITEMS]
                              .unstack("run").loc[human_average.index])
share_of_scores_that_changed = np.mean([
    (judge_v1_runs_side_by_side[item].max(axis=1) != judge_v1_runs_side_by_side[item].min(axis=1)).mean()
    for item in RUBRIC_ITEMS])
print(f"Item scores that changed between identical runs: {share_of_scores_that_changed:.0%}")

def composite_for_run(run, sessions):
    return judge_v1_runs_side_by_side.loc[sessions].xs(run, axis=1, level=1).mean(axis=1)

correlation_per_run = {run: stats.pearsonr(composite_for_run(run, human_average.index), human_average.composite)[0]
                       for run in (1, 2, 3)}
print("\nComposite r vs humans, all sessions:")
for run, r in correlation_per_run.items(): print(f"   run {run}: {r:.3f}")

pilot_sessions = human_average.sample(30, random_state=1).index      # a typical first evaluation set is small
pilot_correlation_per_run = {run: stats.pearsonr(composite_for_run(run, pilot_sessions),
                                                 human_average.loc[pilot_sessions, "composite"])[0] for run in (1, 2, 3)}
print("\nSame thing on a 30-session pilot sample:")
for run, r in pilot_correlation_per_run.items(): print(f"   run {run}: {r:.3f}")
run_to_run_spread = max(pilot_correlation_per_run.values()) - min(pilot_correlation_per_run.values())
print(f"Spread across identical runs: {run_to_run_spread:.2f} points of correlation")
''')
md(r"""
If you had changed one setting between run 1 and run 2, you would have reported an improvement (or a regression) that is pure noise. Setting the model's temperature (its randomness) to 0 reduces this but **does not remove it**.

**Lesson 5.** Always beat the trivial baseline before claiming success. Never compare two versions on a single run: use several runs, or compare session by session (next step).
""")

# ---------------------------------------------------------------- step 6
md(r"""
---
## Step 6 · Compare two versions the right way

Prompt **v2** rewrites the over-strict adherence clause and changes nothing else. Did it help? Compare the two versions **session by session**, not by their overall averages.

For each rubric item the table counts how many sessions scored **better** and **worse** under v2. The **p-value** asks: if v2 made no real difference, how likely is a split this lopsided by luck alone? Below 0.05 is the usual bar for "probably real".
""")
code(r'''
judge_v2 = ai_judge_scores[ai_judge_scores.prompt_version == "v2"].set_index("session_id").loc[human_average.index]
change_v1_to_v2 = pd.DataFrame({item: judge_v2[item] - judge_v1_run1[item] for item in RUBRIC_ITEMS})
change_v1_to_v2["composite"] = judge_v2[RUBRIC_ITEMS].mean(axis=1) - judge_v1_run1[RUBRIC_ITEMS].mean(axis=1)

def sign_test(changes):
    """Count sessions that went up and down, and how likely that split is if nothing really changed."""
    better, worse = (changes > 0).sum(), (changes < 0).sum()
    return better, worse, stats.binomtest(better, better + worse).pvalue if better + worse else np.nan

comparison_rows = []
for item in RUBRIC_ITEMS + ["composite"]:
    better, worse, p_value = sign_test(change_v1_to_v2[item])
    comparison_rows.append({"rubric item": item, "average change": change_v1_to_v2[item].mean(),
                            "sessions better": better, "sessions worse": worse, "p-value": p_value})
pd.DataFrame(comparison_rows).set_index("rubric item").round(3)
''')
md(r"""
How big must a change be before you could even see it? Two v1 runs contain no real change at all, so the difference between them is pure noise. From that noise we can work out the chance of spotting a real improvement of a given size (**δ**, in rubric points).
""")
code(r'''
run_to_run_noise = (judge_v1_runs_side_by_side[RUBRIC_ITEMS].xs(2, axis=1, level=1).mean(axis=1)
                    - judge_v1_runs_side_by_side[RUBRIC_ITEMS].xs(1, axis=1, level=1).mean(axis=1))
noise_spread, n_sessions = run_to_run_noise.std(), len(run_to_run_noise)
simulation_rng = np.random.default_rng(0)
print(f"Run-to-run noise in the composite: standard deviation {noise_spread:.2f} points ({n_sessions} sessions)\n")
print("Chance of detecting a true change of size δ (session-by-session sign test, p < 0.05):")
for true_change in [0.05, 0.1, 0.2, 0.3]:
    simulated_changes = true_change + simulation_rng.normal(0, noise_spread, (2000, n_sessions))
    detection_chance = np.mean([stats.binomtest((s > 0).sum(), (s != 0).sum()).pvalue < .05
                                for s in simulated_changes[:400]])
    print(f"   δ = {true_change:.2f}:  {detection_chance:.0%}")
''')
md(r"""
v2 fixes the collapsed item and leaves the others alone: exactly what a targeted change should do. The table above tells you, **before you run anything**, the smallest change worth chasing. Write that number down in advance, and don't celebrate wins below it.

**Lesson 6.** Change one thing at a time, compare session by session, and decide in advance the smallest improvement you would act on.
""")

# ---------------------------------------------------------------- step 7
md(r"""
---
## Step 7 · Change what you report

The judge ranks sessions reasonably and calibrates them poorly. So don't show supervisors a rubric number. Show a **rank**, and a short **"listen to this one"** list: the bottom 10% of sessions by AI score.

How we measure the list:
- **AUC:** pick one session humans rated in their bottom 10% and one they didn't. How often does the AI score the weak one lower? 0.5 is a coin flip, 1.0 is always.
- **caught:** the share of the humans' bottom 10% that made it onto the AI's list.
- **hit rate:** the share of the AI's list that really is in the humans' bottom 10%.
""")
code(r'''
# Average the three v1 runs (a cheap way to cut noise) and leave out the broken item
ai_average_score = (judge_v1_all_runs.groupby("session_id")[items_without_adherence].mean()
                    .loc[human_average.index].mean(axis=1))
human_bottom_10pct = human_average.composite <= human_average.composite.quantile(.10)
ai_listen_list = ai_average_score <= ai_average_score.quantile(.10)

auc = roc_auc_score(human_bottom_10pct, -ai_average_score)
caught = (ai_listen_list & human_bottom_10pct).sum() / human_bottom_10pct.sum()
hit_rate = (ai_listen_list & human_bottom_10pct).sum() / ai_listen_list.sum()
times_better_than_random = hit_rate / human_bottom_10pct.mean()
print(f"AUC for finding the sessions humans rated lowest: {auc:.2f}  (0.5 = coin flip)")
print(f"'Listen to this one' list ({ai_listen_list.sum()} sessions): catches {caught:.0%} of the human bottom 10%,")
print(f"   {hit_rate:.0%} of listed sessions are truly in it, {times_better_than_random:.1f}x better than picking at random")
''')
md(r"""
A supervisor who listens to the listed sessions finds weak sessions several times faster than by picking at random. A supervisor shown "3.2 out of 7" for a session humans would rate 6 just stops trusting the tool.

**Lesson 7.** Ship the output your evaluation supports. If the evidence supports ranking and triage, report a rank and a flag, not a score that pretends to more precision than it has.
""")

# ---------------------------------------------------------------- step 8
md(r"""
---
## Step 8 · Safety flags: a different kind of test

Risk disclosures are rare in real sessions, so natural data can't tell you if the flagger works. Build a **seeded test set**: genuine current-risk cases written in realistic ways (plain, indirect, in Sheng, split across turns) mixed with look-alikes (a resolved story from the past, a role-play, a textbook example). Here the content is described in words only.

The release rule is **lopsided on purpose**: one missed real case blocks release, while false alarms (look-alikes wrongly flagged) are a tuning problem.
""")
code(r'''
def safety_report(flagged):
    real_risk = safety_test_cases.is_current_risk
    caught = (flagged & real_risk).sum()
    missed = (~flagged & real_risk).sum()
    false_alarms = (flagged & ~real_risk).sum()
    return {"real cases caught": f"{caught / (caught + missed):.0%}", "real cases missed": int(missed),
            "false alarms": int(false_alarms)}

every_flag = safety_test_cases.ai_flagged
current_only_flags = safety_test_cases.ai_flagged & (safety_test_cases.ai_timeframe == "current")   # use the model's own timeframe label
print(pd.DataFrame({"escalate every flag": safety_report(every_flag),
                    "escalate 'current' only": safety_report(current_only_flags)}).T.to_string())
print("\nMissed case(s):")
missed_cases = safety_test_cases[safety_test_cases.is_current_risk & ~safety_test_cases.ai_flagged]
print(missed_cases[["case_id", "scenario"]].to_string(index=False))
''')
md(r"""
Asking the model *when* the event happened, and only escalating "current" flags, cuts the false alarms sharply without losing any real cases. But one real case was missed, the one **split across several turns**, so this version **does not ship**. That case goes into the test set permanently, and every future version must catch it.

**Lesson 8.** For rare, high-stakes failures, build the test set yourself, make the release rule lopsided, and turn every miss into a permanent test case.
""")

# ---------------------------------------------------------------- wrap
md(r"""
---
## Wrap-up · the scorecard from this run
""")
code(r'''
scorecard = pd.DataFrame([
    ("1 Reference audited", f"{len(duplicate_recordings)} duplicate recordings, {len(rows_with_wrong_total)} bad totals, "
                           f"{len(rows_missing_a_score)} missing rows fixed", "Done"),
    ("2 Human ceiling", f"ICC {human_icc:.2f}; best possible r vs two raters ≈ {ceiling_vs_two_raters:.2f}", "Report against this"),
    ("3 Transcripts", f"Two humans disagree (CER {transcript_agreement.iloc[3].CER:.2f}) "
                      f"more than the model errs (CER {transcript_agreement.iloc[2].CER:.2f})", "Check the reference"),
    ("4 Judge vs humans", f"r {composite_correlation:.2f} = {composite_correlation / ceiling_vs_two_raters:.0%} of ceiling; "
                          f"adherence item collapsed", "Fix or drop the item"),
    ("5 Traps", f"Constant beats corrected score on MAE; identical runs on 30 sessions span r by {run_to_run_spread:.2f}",
                "Multiple runs, session-by-session tests"),
    ("6 Prompt v2", f"Composite +{change_v1_to_v2.composite.mean():.2f}, driven by adherence", "Adopt; re-check ranking"),
    ("7 Reporting", f"Bottom-10% list: AUC {auc:.2f}, {times_better_than_random:.1f}x better than random", "Ship rank + list"),
    ("8 Safety", f"Caught {safety_report(current_only_flags)['real cases caught']} of real cases, "
                 f"1 miss (split across turns)", "BLOCK release"),
], columns=["step", "what we found", "decision"]).set_index("step")
scorecard
''')
md(r"""
### Eight habits to take home
1. Audit and freeze your reference data before scoring against it.
2. Measure human agreement first; report AI agreement against that ceiling.
3. Evaluate each stage on its own, and check the quality of its reference.
4. Break the overall number down by item; read the prompt next to every failure.
5. Beat the trivial baseline, and never trust a single run.
6. Change one thing at a time and compare session by session.
7. Report what the evidence supports: rank and triage, not false precision.
8. Seed rare, high-stakes failures yourself, and make the release rule lopsided.

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
