## Summary

GAOKAO-Eval is a comprehensive LLM evaluation benchmark built on China's annual National College Entrance Examination, featuring temporal isolation (evaluating only pre-exam models), expert grading of subjective responses by 54 teachers, annual updates, and multi-modal coverage. The paper's central analytical argument uses the Rasch Item Response Theory (IRT) model to characterize how LLM scoring patterns deviate from theoretical human performance curves, identifying two key phenomena: semi-difficulty-invariant scoring (weak difficulty–score correlation) and high performance variance on similarly-difficult questions. Additional findings include grading inconsistencies among human teachers and an exploratory result showing o1's reasoning tokens improve Rasch model fit from R²=−0.22 to R²=0.10.

---

## Strengths

- **Temporally-isolated benchmark with subjective grading**: The use of annual Gaokao exams (sealed creation process, ~490 new questions per year) combined with pre-exam model cutoffs and 54 human teachers grading subjective responses without knowing they were evaluating AI outputs is a concrete and meaningful advance over static, MCQ-only benchmarks. This design directly addresses two persistent criticisms of the evaluation literature (leakage and format poverty).

- **Item-level score–difficulty analysis**: The per-subject, per-question-type correlation heatmaps in Figure 8, decomposed across multiple models, go beyond simple aggregate score reporting and identify an empirically real phenomenon: LLMs do not exhibit monotonically decreasing performance as human-rated item difficulty increases. Even if the theoretical framing using Rasch is contestable, the weak correlations observed across subjects and models constitute a genuine and reproducible empirical signal.

- **Concrete qualitative error taxonomy**: Figure 10's four examples (illogical geometric deduction, correct answer via flawed steps, fabricated poem, verbatim copying instead of summarization) are specific, verifiable, and illustrate why raw correctness can decouple from human-like reasoning. These are not cherry-picked oddities — they correspond to documented LLM failure modes and make the quantitative findings tangible.

---

## Weaknesses

### Fatal
None that fully nullify the benchmark contribution, but the central analytical argument has serious structural problems that substantially weaken the paper's main claims.

---

### Major

**1. The core claim — that Rasch-curve deviation demonstrates "high scores fail to reflect human-aligned capabilities" — rests on an unvalidated equivalence.**  
The paper (Sec. 3.1) states: *"we directly use this equation as the basis for evaluation"* and interprets R²=−0.22 as evidence of a "significant mismatch between the LLMs' capabilities and the expected human-aligned ability." But the Rasch model is a psychometric model for *human* examinees; it was not validated against actual human performance data on these same 2024 Gaokao items. The paper never shows what a human student's score-vs-difficulty scatterplot looks like on these questions. "LLMs deviate from a Rasch curve" and "LLMs lack human-aligned capability" are not the same claim. The former is about the shape of the IRT curve; the latter is a broader capability judgment. Without human baseline data on the same items, the paper cannot establish whether the deviation is specific to LLMs, is a property of the Gaokao item pool, or results from noisy difficulty estimation. This is not a writing fix — it is a structural gap in the argument that the paper's headline claim depends on.

**2. WQX improvement claim is contradicted by the paper's own figure.**  
Section 2.1 states: *"The improvements observed in Math, MMLU, CMMLU, and C-Eval benchmarks further affirm the WQX model's comprehensive natural language understanding."* Figure 3b, however, shows WQX ≈ InternLM2-20b-base on every benchmark (MMLU ~65, CMMLU ~65, C-Eval ~65, GaokaoBench ~65, Math ~35, with ~0 improvement throughout). If Figure 3b is accurate, the textual claim of improvements is false. This is a direct contradiction within the paper and undermines the credibility of the results presentation.

**3. The ISR metric is miscalibrated against trivial baselines.**  
Equation 4 defines ISR as the fraction of scores where |s − μ| > σ. For any approximately normal distribution, approximately 32% of values naturally fall outside ±1σ by definition. The paper declares it a finding that ISR "exceeds 32%," but this merely indicates a slightly heavier-tailed distribution than Gaussian — not an abnormally high level of grader disagreement. More importantly, there is no comparison to ISR computed for human student answers graded by the same teachers under the same protocol. Without that baseline, the claim that LLM answers are *unusually* hard for graders to evaluate consistently is not established.

**4. The difficulty estimation methodology is insufficient to support central analyses.**  
The paper's key findings (Figures 6, 8) are only as valid as the difficulty labels, yet Sec. 3.1 provides minimal detail: a "hybrid approach combining manual annotations with an Elo rating system... incorporating both human expertise and LLM-based judgments" with a claimed "internal correlation of up to 0.94." Figure 5 shows distributional similarity, not correlation. Critical details are absent: number of annotators, their protocol, the pairwise comparison setup, how human and Elo ratings were integrated, and human-to-human inter-annotator agreement. Using GPT-4o-mini and GPT-4o to help assign difficulty ratings to questions on which these same model families are then evaluated introduces circularity.

---

### Minor

**5. Figure 4 heatmap contains anomalous data.**  
The performance heatmap in Figure 4 shows identical values across all subjects for each model (e.g., Mistral 8x22B = 70.7 for all 12 subjects; GPT-4o = 76.7 across all subjects). This may be a PDF extraction artifact, but if the actual figure shows genuine subject-level differentiation, the paper needs to make this unambiguous, because subject-level performance variation is central to the paper's analytical claims.

**6. The o1 "mitigation" framing is overstated.**  
The paper claims in the abstract that "o1's reasoning-as-difficulties can mitigate the mismatch." R² improving from −0.22 to 0.1019 means that reasoning tokens explain ~10% of variance in scoring rates. This is still an extremely poor fit. The paper presents this as a substantive mitigation, but both fits are essentially uninformative. The discussion (Sec. 4) appropriately hedges with "promising to explore," but the abstract and contribution bullets do not. This should be recast as a preliminary exploratory finding on a single model.

**7. Four qualitative examples cannot bear explanatory weight for "recurring patterns."**  
Section 3.3 is titled "LLMs' Unique Scoring Patterns" and uses four examples (Fig. 10) to explain why "high scores obtained by the model only indicate accuracy according to specific grading rules." Four examples establish the existence of these failure types, not their prevalence. No frequency analysis, error taxonomy with counts, or relation to subject/difficulty/model is provided.

---

### Trivial

None beyond the major and minor above.

---

## Nice-to-Haves

- Obtaining aggregate human student score-per-question data (even historical) on the same exam would transform the paper: a side-by-side human vs. LLM score-vs-difficulty plot would be immediately compelling and would replace the theoretical Rasch assumption with direct empirical grounding.
- Applying standard inter-rater reliability metrics (ICC, Krippendorff's α) alongside ISR would make the grading inconsistency finding much more credible and comparable to prior work.
- Testing whether the semi-invariant scoring pattern replicates across 2–3 years of Gaokao would significantly strengthen the paper's robustness claims.
- Trying 2PL or 3PL IRT models, which allow varying discrimination and guessing parameters, would either strengthen the claim (mismatch persists) or reveal that the 1PL Rasch model's simplicity is driving the apparent misfit.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **"Temporal isolation does not fully guarantee absence of data contamination"** (Human Finder, W3): The hard rule applies — the paper cites this as its design choice, and criticizing the existence or effectiveness of this strategy as if it can be independently verified is out of scope. The approach is a reasonable mitigation; the paper does not claim it is mathematically perfect.

- **"The novelty of applying IRT/psychometrics to LLM evaluation is limited"** (Human Finder, W4): This cites other papers' reviews to argue prior work exists. Under the no-missing-related-works rule, this cannot be verified and should not be included.

- **"Single-dataset and cultural bias limits generalizability"**: Weakened/removed. A paper about X (Gaokao-based evaluation) is evaluated on whether it does X well, not on whether it also covers Brazil, India, and the US. Gaokao's scope is explicitly its scope.

- **Generic strengths** from the neutral reviewer ("principled analytical framework from psychometrics" is reasonable to keep, but "important empirical finding on scoring patterns" is borderline generic): Removed the more generic framings.

---

## Novel Insights

The observation that LLM performance on exam questions follows a *semi-difficulty-invariant* pattern — that aggregate correlations between human-rated difficulty and LLM scoring rate are weak across multiple subjects and multiple models — is a real and underappreciated empirical regularity. This is distinct from merely observing that "LLMs can solve hard problems"; it suggests that LLMs do not share the *relative ordering* of difficulty that human examinees exhibit, and may reflect fundamentally different strategies (pattern matching at scale rather than compositional reasoning). The exploratory o1 result — that reasoning tokens correlate better with scoring rate than human-rated difficulty does — is a suggestive hint that difficulty for LLMs may be primarily a function of inference-time computation demand rather than conceptual depth. If validated, this reframes benchmark difficulty design from a human-centered to a compute-centered paradigm.

---

## Evaluation on Key Axes

- **Novelty**: Moderate. The benchmark design (annual update, temporal isolation, subjective grading) is a clear and useful advance over static Gaokao benchmarks. The application of IRT to LLM evaluation is not new, but the specific analysis of semi-invariant scoring across a comprehensive exam is a fresh angle. The o1 inference-compute framing is novel but underdeveloped.
- **Technical soundness**: Below average. The Rasch-as-normative-target assumption is unjustified, the ISR metric is poorly calibrated, the difficulty estimation lacks sufficient rigor, and the WQX claim is contradicted by its own figure. The benchmark engineering is sound; the analytical framework is not.
- **Empirical support**: Weak for the headline claims, moderate for secondary observations. The weak difficulty–score correlations in Figure 8 are credible. The Rasch fit (R²=−0.22), ISR interpretation, and WQX improvements are all unreliable or contradicted.
- **Significance**: Moderate if restricted to the benchmark contribution. The research question (does high score = high capability?) is genuinely important. But as presented, the analytical contribution oversells what the evidence can support.
- **Clarity**: Mixed. The benchmark design sections are clear. The psychometric analysis sections are underspecified and overstate conclusions. The WQX figure-text contradiction is a serious clarity failure.

---

## Score and Decision

**Calibration against past reviews:**

- *gAEEjGv5Oa.md* (debate training / scalable oversight): 6.5 — real empirical contribution, validated finding on the central claim, honest about limitations.
- *AAZ3vwyQ4X.md* (MSPL benchmark/representation learning): 4.0 — headline metric has structural problem, evaluation unclear.

This paper is **worse** than the debate training paper: the central analytical claim is structurally unsupported (no human baseline data), the WQX figure contradicts the text, and the ISR metric is trivially defined. It is **comparable to or slightly above** the MSPL paper: the benchmark engineering is more clearly useful, and the research question is more important — but the analytical framework that forms the paper's main contribution is almost as unreliable as MSPL's problematic metric.

Placement: ~4.5 — above a clear reject, but the paper's main analytical argument does not hold in its current form, and the WQX contradiction is a concrete factual issue. The benchmark alone is worth something, but ICLR expects the analytical claims to be substantiated.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>