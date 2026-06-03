# MindForm Research Review

**Question under review:** *How should experiences change a persistent personality over time?*

**Stance:** lead researcher, not maintainer. Existing design decisions are not
sacred. The goal is the quality of the long-term personality-formation
architecture, not the preservation of any component (Pandora included).

This document states the findings; the accompanying code changes implement them.

---

## 1. What MindForm is today (ground truth, not the README's claims)

```
text -> MiniLM embedding -> appraisal vector (8-d) -> signed push (matrix M)
     -> two-timescale update (fast mood `state` + slow disposition `trait`)
     -> persist + memory (habituation / chronicity / retrieval)
```

* **Personality** = five Big Five axes in [-1, 1], each carried on two timescales
  (`state` = fast mood that decays; `trait` = slow disposition that consolidates
  *sustained* mood and relaxes toward a set-point).
* **Experience** = an 8-d appraisal vector (valence, intensity, novelty, agency,
  social, outcome, self_relevance, threat_challenge).
* **Pandora** = a `text -> OCEAN` MLP (frozen MiniLM -> 128 -> 64 -> 5, sigmoid),
  trained on the `jingjietan/pandora-big5` Reddit dataset. The README says it is a
  "read-only readout."

Two facts establish the baseline:

1. **The formation core works and is Pandora-free.** `acceptance_test.py` passes
   all 19 stylized-fact checks (mood reacts fast / disposition slow; bounded
   S-curve accumulation; diminishing returns; set-point relaxation; the maturity
   principle — sustained mastery raises C and lowers N; habituation; chronicity;
   retrieval blending) using only `appraisal -> impact -> updater`.
2. **Pandora is not in the loop.** A dependency scan shows `trait_model` /
   `readout` are imported only by `runnn.py` (a 4-line demo) and by the unused
   `readout.py`. Nothing in the live pipeline
   (`simulation.py` / `interactive.py` / `acceptance_test.py`) calls it.

---

## 2. The six questions

### Q1 — Is Pandora actually useful? **No. Remove it from the architecture.**

Five independent lines of evidence, any one of which is disqualifying:

1. **It is dead weight.** It is imported by nothing in the formation path
   (§1.2). It already contributes zero to how experiences change personality.
2. **Wrong training signal for any job MindForm has.** PANDORA attaches a Reddit
   author's *self-reported Big Five percentiles* to all of that author's
   comments. `text -> OCEAN` therefore learns **author attribution / stylometry**
   ("what kind of person tends to write like this"), which is neither *the
   appraised meaning of an experience* nor — emphatically — *how an experience
   changes a person*.
3. **Epistemically backwards.** MindForm is a simulator: the trait vector is a
   **state variable we set and observe**. Estimating it from text is solving a
   problem we do not have. You do not measure a thermostat's set-point by
   photographing the room.
4. **Circular in the current loop.** The only text available to "read" is
   `response.py`'s five canned sentences, themselves emitted from the trait vector
   by a trivial threshold rule. A readout would noisily recover what we just
   wrote — zero information.
5. **No validity evidence, ever.** `train_trait_model.py` optimizes and reports
   **MSE only** — never correlation or rank accuracy. On imbalanced percentile
   labels, MSE is minimized by predicting the marginal mean. An empirical probe of
   the shipped `trait_model.pth` (20k unit-norm embeddings; see
   `docs/` discussion) shows outputs pinned near trait-specific **marginal
   offsets** (C ≈ 0.62, A ≈ 0.35) with a **compressed std ≈ 0.10**; on real
   (anisotropic, concentrated) sentence embeddings the conditional range is
   narrower still. That is the signature of a marginal-dominated, weakly
   input-conditional map — not a validated personality instrument.

It is useful as neither a formation force (correctly already barred) nor a
measurement (backwards, circular, unvalidated). **Removed.**

### Q2 — Is the dataset fundamentally flawed? **The central one cannot answer the question at all.**

There are two datasets:

* **PANDORA big5 (Pandora's training data).** Cross-sectional, between-person,
  author-attribution. It contains **no experiences and no within-person change.**
  The project's question is **longitudinal and causal**; PANDORA has zero
  longitudinal/causal content. It is not "noisy but usable" for formation — it is
  *categorically unable* to supervise it. (It could supervise a between-person
  text→trait probe, but Q1/Q3 explain why that probe should not exist in the core,
  and its labels — self-selected Redditors posting test screenshots — are weak
  even for that.)
* **The appraisal bootstrap (EmoBank VAD + heuristic weak labels).** The *right
  kind* of data. EmoBank's Valence/Arousal/Dominance are validated; Dominance →
  agency is a defensible proxy. **But only 3 of 8 appraisal dims get real
  supervision;** the other 5 are the heuristic lexicon copied in as "weak labels,"
  so training the head on them **launders the lexicon into a "learned" model** (no
  new information, false confidence). Direction right; realization under-supervises
  5/8 dims. Fix: bring in appraisal-annotated event corpora (enVENT, ISEAR-style)
  to supervise outcome / novelty / threat properly.

**The deepest finding:** the question is causal-longitudinal and **no dataset in
or near the project is.** `ARCHITECTURE.md` concedes this. That — not the model —
is the real bottleneck, and it is why the `appraisal -> trait` map `M` must remain
theory-authored for now rather than learned.

### Q3 — Should the observation/measurement model exist at all? **Split the question — one yes, one no.**

"Observation model" conflates two different things:

* **Appraisal extractor (`text -> appraisal`): YES — keep and invest.** It is the
  perception front-end: how an experience acquires *meaning*. It is the causal
  mediator and the single most important *learnable* component (experiences change
  personality only through how they are appraised).
* **Trait estimator (`text -> Big Five`, i.e. Pandora): NO — not in the core.**
  In a simulator the trait vector is observed (Q1.3). The only legitimate niche is
  *external validity for a real generative agent* — and that is an evaluation
  concern outside the formation architecture, **better served by administering a
  validated inventory to the generator** (and scoring with its published key) than
  by a noisy author-attribution MLP.

### Q4 — Should Big Five remain? **Yes — but reframed as a coordinate/readout basis, not the mechanism; grounded via Whole Trait Theory.**

Against Big Five *as the mechanistic state of a dynamical process:* it is a
**descriptive, factor-analytic summary of between-person variation in adults**,
never a generative model of within-person change. Treating emergent correlational
factors as mechanistic state variables is, strictly, a category error
(atheoretical/lexical origins; WEIRD-biased; factors not truly orthogonal; does
not carve the causal joints of change).

For keeping it: the personality-*change* literature — **set-point theory**
(Costa & McCrae), the **maturity principle** and **corresponsive principle**
(Roberts) — is overwhelmingly framed in Big Five. Keeping the basis lets MindForm
connect to that literature and be **checked against its stylized facts** (the
acceptance test already verifies the maturity principle). `config` already makes
the basis swappable; `M` is the only basis-specific piece.

**Resolution:** keep Big Five as the *interpretable coordinate system we project
appraisals onto and read trajectories out of*, but stop treating the five scalars
as the mechanism. Ground the state in **Whole Trait Theory (Fleeson):** a trait
*is* a density distribution over momentary states. MindForm's `state`/`trait`
split is already a crude Whole-Trait model (`state` = momentary expression;
`trait` = central tendency). Make it explicit and **complete it** (Q5).
**Cybernetic Big Five Theory (DeYoung)** is the documented route to a mechanistic
reading of the same five axes (goals; the Stability/Plasticity metatraits), and
its metatrait covariance is the principled next dynamical step (Q5).

### Q5 — What representation should be used for personality formation?

Most of the right representation already exists; one real upgrade is added here.

* **Experience = appraisal vector (8-d).** Keep. Causal ingredients, not trait
  readings. The right place to invest learning (Q3).
* **Personality = per-axis *density distribution* on two timescales:** slow central
  tendency `trait` + fast `state` + **dispersion `trait_sd`** (the Whole-Trait
  completion, *added in this PR*). A scalar mean cannot distinguish a steady
  extravert from a volatile one with the same average; the dispersion can.
  Implemented as a slow EWMA of `(state - trait)²` — within-person variability is
  itself a stable individual difference.
* **Dynamics = two-timescale, memory-modulated.** Mood decays; disposition
  consolidates *sustained* mood with diminishing returns and relaxes to a
  set-point; recurrence damps mood (habituation) but amplifies consolidation
  (chronicity); retrieval tints appraisal ("this reminds me of…"). Keep — it is
  well-grounded and passes the stylized-facts ruler.
* **Coordinate basis = Big Five.** Keep (Q4).
* **Removed:** the `text -> Big Five` estimator (Pandora).
* **Next step, documented not yet built — trait covariance.** Appraisals should
  move *correlated bundles* of traits (the metatrait structure; the maturity
  principle moves C↑/A↑/N↓ together), not five independent axes. Today `M` and the
  update treat axes independently. This is the highest-value remaining mechanistic
  improvement and is specified in `evaluation.py`/this doc for a follow-up.

### Q6 — Concrete changes (this PR)

**Implemented:**

1. **Removed Pandora from the architecture** — deleted `trait_model.py`,
   `train_trait_model.py`, `readout.py`, `runnn.py`, `trait_model.pth`
   (rationale: Q1–Q3).
2. **Added `evaluation.py`** — the correct instrument for a *formation* model:
   reports the **observed** formed traits + density summary (no estimation) and
   **scores a trajectory against developmental stylized facts** (construct
   validity). Includes a documented `ExpressionProbe` protocol describing the
   *right* external-validity method (administer a validated inventory to a real
   generator) and why the PANDORA-MLP was unfit for it.
3. **Completed the representation** — added per-axis dispersion `trait_sd` so a
   personality is a **density distribution** (Whole Trait Theory), not a point.
   Additive; the stylized-facts test is extended (volatile life ⇒ wider state
   density than a consistent one) and stays green.
4. **Docs** — `ARCHITECTURE.md` rewritten to match; `requirements.txt` note
   (`datasets` is now only for the appraisal-head bootstrap).

**Deliberately *not* done (with reasons):**

* Did **not** rip out Big Five — it is the literature bridge and the stylized-facts
  ruler (Q4).
* Did **not** build a real generator — out of scope; but it is the precondition for
  any expressed-trait probe to mean anything.
* Did **not** implement trait covariance — needs tuning and ideally data; specified
  as the next step (Q5).
* Did **not** train the appraisal head here — Hugging Face is network-blocked in
  this environment; the recipe is unchanged and the under-supervision of 5/8 dims
  (Q2) is documented for when it is run.

---

## 3. The answer to the core question

> Experiences change a persistent personality only through their **appraised
> meaning**, not their surface text. That meaning lands first as **fast mood** and
> graduates into **slow disposition** only when it is **sustained or recurrent**.
> Dispositions are **distributions** (a mean *and* a variability), they accumulate
> with **diminishing returns** and stay **bounded**, they **relax toward a
> set-point** when unreinforced, and **recurrence makes a pattern feel weaker while
> shaping character more strongly**. Big Five is the *coordinate system* we read
> this out in; **appraisal is the causal mediator we should learn**; and the
> personality itself is *observed*, so there is nothing for a `text -> trait`
> estimator (Pandora) to do.
