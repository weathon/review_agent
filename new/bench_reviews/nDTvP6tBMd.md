Now I have enough information to write the final review. Let me analyze the key findings from reviewing the paper carefully against reviewer claims and calibration papers.

**Key verification of reviewer claims:**

1. **LLM-based procedural grading validation** - Paper says "We manually verify a subset of grading responses and found that LLM-based grading is closely aligned with human grading." No quantitative agreement stats given. This is a valid concern.

2. **Cross-benchmark comparison** - Paper compares to externally-reported numbers from other papers, not re-run under matched conditions. Valid concern.

3. **Word problem evaluation** - Different prompts (no hints), only 40 problems, only one model. The interpretation that "context makes it harder" is confounded by multiple variables. Valid concern.

4. **10% threshold** - The paper provides a reasonable explanation for why 10% is chosen but doesn't justify it formally. Moderate concern.

5. **Human baseline** - Absent. Valid concern for calibrating difficulty.

6. **Narrow scope** - The paper's scope is explicitly asymptotic methods, so critiquing the title as "applied mathematics" but focusing on asymptotics is somewhat legitimate but also partly scope creep.

7. **Template diversity** - Valid concern about structural homogeneity.

**Calibration comparison:**
- ARB (advanced reasoning benchmark): Scores 6,6,5,5 → Reject. Small dataset (~34 math problems), similar LLM-grading concerns
- UGMathBench: Scores 6,8,6,6 → Accept. Much larger (5062 problems), multiple subjects, dynamic benchmark
- U-MATH: Scores 5,5,6,5 → Reject. 1100 problems, LLM grading concern, limited validation
- Omni-MATH: Scores 8,8,6,5 → Accept. 4428 problems, rigorous annotation, developed verifier with metrics

HARDMATH is between ARB (rejected, 6,6,5,5) and UGMathBench (accepted, 6,8,6,6). The key differences from UGMathBench: smaller evaluation set (366 vs 5062), narrower scope (one topic vs 16 subjects), LLM grader not validated with metrics. But HARDMATH has the unique angle of asymptotic reasoning + algorithmic generation + numerical validation. Comparing to ARB (which covers advanced math and was rejected), HARDMATH has larger dataset, better validation, but similar LLM-grading issues.

I'll rate this at approximately 5.0 - borderline reject territory.

---

## Summary
HARDMATH introduces a dataset of 1,466 algorithmically generated problems targeting asymptotic/approximation methods in applied mathematics (dominant balance, nondimensionalization, polynomial root-finding, ODEs, integrals). Problems are generated via SymPy/SciPy templates and filtered against numerical ground truths within 10% error. A 366-problem evaluation subset (HARDMATH-MINI) and 40 word problems are used to benchmark leading LLMs, showing substantially lower accuracy (GPT-4: 43.8%, o1-mini: 62.3%) than on standard math benchmarks, with detailed error mode analysis.

## Strengths

- **Genuinely novel benchmark niche with a coherent mathematical focus**: HARDMATH specifically targets asymptotic/approximation reasoning—dominant balance, Laplace's method, regime-based solutions—which is absent from all existing LLM math benchmarks. This is not simply "harder math" but a qualitatively different type of reasoning (multi-regime, approximate, judgment-requiring) that is directly relevant to scientific and engineering research contexts, and no comparable dataset exists.

- **Algorithmic generation with numerical validity verification enables scalability**: The SymPy/SciPy-based generation pipeline produces arbitrarily many problems with automatic 10% error filtering against numerical ground truths and visual verification for HARDMATH-MINI. This directly addresses the primary limitation (scale and copyright) of all comparable graduate-level datasets (ARB: 34 problems, GHOSTS GRAD-TEXT: 130 problems). The code is public, making it genuinely reproducible and extensible.

- **Error mode analysis reveals interpretable, actionable failure patterns**: The fine-grained breakdown showing that 5-shot CoT shifts GPT-4's dominant error from "incorrect dominant balance terms" (66.1%) to "missing cases" (50.8%) is specific, concrete, and informative about what reasoning capability is being improved by CoT. This goes beyond accuracy tables and identifies the nature of the reasoning challenge.

- **Hybrid evaluation protocol is well-suited to the task**: For problems requiring approximate answers with inherent flexibility (multiple valid asymptotic regimes), combining exact/symbolic final-answer matching with GPT-4o procedural grading is a principled response to an evaluation design challenge that most exact-answer benchmarks sidestep.

## Weaknesses

### Major

- **LLM-based procedural grading lacks quantitative validation, undermining all reported accuracy numbers.** The paper relies on GPT-4o as a procedural grader for Roots, ODEs, and Integrals—three of four problem categories—but validates it only by stating "We manually verify a subset of grading responses and found that LLM-based grading is closely aligned with human grading." No sample size, no Cohen's kappa or similar agreement statistic, no failure-mode examples from the grader, and no evidence of stability across model/problem-type combinations are provided. Since partial credit depends on this grader and reported accuracies in Table 2 are shaped by it, the paper cannot fully defend its headline numbers. The ARB paper (a comparable benchmark, rejected, scores 6/6/5/5) was criticized on the same grounds but at least reported human-LLM correlation. Additionally, GPT-4o grading GPT-4 responses introduces a potential in-family circularity that is not discussed.

- **Cross-benchmark comparison is not apples-to-apples.** The paper's central empirical framing—that models perform "significantly lower" on HARDMATH than on existing benchmarks—rests on comparing internally-run HARDMATH numbers against externally-reported numbers from other papers using different prompting schemes, shot counts, model versions, and scoring procedures. For example, "GPT-4 achieves 72.2% on MATH with 0-shot CoT" is cited from the GPT-4 technical report, while "GPT-4 achieves 43.8% on HARDMATH-MINI with 5-shot CoT" is the paper's own number—these are not directly comparable. The paper should either re-run at least one comparator benchmark under matched conditions or explicitly weaken this claim.

- **No human performance baseline makes it impossible to calibrate difficulty.** HARDMATH does not report accuracy from graduate students or domain experts on any subset of HARDMATH-MINI. Without this, the claim that these are "graduate-level" problems is asserted but unvalidated, and the interpretation of o1-mini's 62.3% accuracy—whether impressive or poor—is unanchored. Comparable benchmarks such as Omni-MATH (accepted, 8/8/6/5) and UGMathBench (accepted, 6/8/6/6) both include or estimate human performance baselines.

### Minor

- **Word-problem evaluation is uncontrolled and its interpretation is unsupported.** Section 4.3.1 compares GPT-4 at 28.1% on 40 word problems versus 43.8% on HARDMATH-MINI, but three variables change simultaneously: (a) only 40 problems, (b) problem-specific hints are omitted, and (c) the problem set composition differs. The lower score cannot be attributed to "realistic context" making the task harder; it could be entirely due to removing hints. A paired comparison on the same problems with and without context under identical prompts would resolve this.

- **The 10% numerical validation threshold is stated but not justified.** The paper does not explain why 10% is appropriate for asymptotic approximations, nor does it report what fraction of candidate problems fail this threshold (which would indicate how often the generation procedure produces poor solutions). Sensitivity analysis at stricter thresholds (e.g., 5%, 1%) would strengthen confidence in dataset quality.

- **Error analysis covers only one problem type for one model.** The detailed error-mode analysis (Fig. 4) is limited to GPT-4 on Roots. The abstract claims "detailed error analysis" broadly, but error modes for ODEs and Integrals—where LLM performance is lowest and most informative—are absent. This is a missed opportunity given that these are the hardest and most interesting failure cases.

- **Model coverage omits recent math-specialized models.** The evaluation includes only GPT-3.5, GPT-4, o1-mini, Llama3-8b, and CodeLlama-13b. Math-specialized models (Qwen2.5-Math, DeepSeekMath) are relevant comparators and their omission limits the benchmark's characterization of where the field stands.

### Trivial

- Table 2 caption says "HARDMATH evaluation set" while the body refers to "HARDMATH-MINI"; terminology should be consistent.
- The framing of o1-mini as "confirming its optimized ability for STEM reasoning" slightly over-interprets benchmark performance as a mechanistic capability claim.

## Nice-to-Haves

- Evaluate at least one tool-augmented model (e.g., GPT-4 with Python/SymPy code interpreter), since the paper explicitly argues these problems require computational tool use. This would directly test the paper's own characterization of what makes HARDMATH hard.
- A fine-tuning experiment on the 1,060-problem training split to show utility for model development—this is repeatedly claimed but never demonstrated.
- Per-regime accuracy breakdown (small ε, large ε, very large ε) would test whether LLMs fail at regime identification specifically, which is the core claimed challenge.
- Dataset contamination check for the evaluated models.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Graduate-level" is not proven by content labeling alone**: While the harsh reviewer raises a valid point that no human study calibrates the difficulty, this is partially addressed by the paper's anchoring to a named graduate-level course and the complexity of the techniques (dominant balance, Laplace's method, correction terms). The difficulty claim is defensible; what's missing is a human baseline, not a fundamental flaw in the difficulty assertion. Moved to Nice-to-Have.

- **Template diversity / structural homogeneity concern**: The harsh reviewer suggests that fine-tuning leakage and template overlap are a concern. While real, the paper is not primarily a fine-tuning paper (fine-tuning is stated as future work), and evaluation performance on HARDMATH-MINI is valid even if templates are restricted. This is a valid concern for future model development use but does not undermine the evaluation paper's core contribution. Weakened to minor/nice-to-have.

- **Narrow scope / title mismatch**: Calling the scope narrow relative to "applied mathematics" is partially scope creep—the paper explicitly states the dataset is "inspired by a graduate course on asymptotic methods." The problem is real (the title slightly overpromises) but trivial to fix with revised framing. Removed as a substantive weakness; the scope is coherent.

- **Claim that o1-mini "confirms optimized STEM reasoning"**: Over-interpretation but a writing fix, not a methodological flaw.

## Novel Insights

The most novel contribution is the observation that asymptotic/approximation reasoning constitutes a qualitatively distinct type of mathematical problem that exposes a specific LLM failure mode: while CoT prompting dramatically reduces errors in fundamental setup (wrong dominant balance terms), it shifts errors to structural completeness failures (missing cases, dropping complex roots). This is a substantive finding about how prompting changes error type rather than uniformly improving capability, and it suggests that the reasoning gap for approximation problems is not primarily one of calculation but of case enumeration and self-consistency checking—something that richer few-shot demonstrations address only partially.

## Suggestions

1. Report inter-rater agreement (e.g., Cohen's kappa) between GPT-4o grader and human graders on a stratified sample of at least 50–100 problems across all problem types. This single addition would substantially de-risk the central evaluation methodology concern.
2. Explicitly weaken or qualify the cross-benchmark difficulty comparison; state that protocols differ and the comparison is directional rather than controlled.
3. Add at least a small human baseline (even 3–5 graduate students on 30–50 HARDMATH-MINI problems across types) to anchor difficulty claims.
4. Expand error-mode analysis to ODEs and Integrals for at least one model—these are the hardest categories and deserve the same treatment as Roots.
5. Clarify exactly how automatic and procedural grading are combined into the final accuracy number in Table 2 (is it a threshold on partial credit? average? binary after LLM override?).

## Score and Decision

**Calibration references:**
- **Omni-MATH** (8,8,6,5 → Accept): 4,428 problems, rigorous annotation, dedicated verifier with 86% human consistency, broad coverage. HARDMATH is weaker on scale, coverage, and grader validation.
- **UGMathBench** (6,8,6,6 → Accept): 5,062 problems, 16 subjects, dynamic benchmark. Stronger on scale and coverage but HARDMATH has a more distinctive niche.
- **U-MATH** (5,5,6,5 → Reject): 1,100 university problems, LLM grading used, limited grader validation. Similar methodological concern to HARDMATH, but U-MATH is larger and more broadly scoped. HARDMATH's niche is more distinctive but its evaluation is similarly weakly validated.
- **ARB** (6,6,5,5 → Reject): Advanced reasoning benchmark, LLM grader with human comparison. Small dataset, rubric grading. HARDMATH has larger dataset and more systematic generation but similar grader concerns and similar final verdict from reviewers.

HARDMATH lands between ARB/U-MATH (rejected, ~5) and UGMathBench (accepted, ~6.5). The unique niche (asymptotic reasoning), algorithmic generation, and numerical validation pipeline are genuine strengths. However, the grader validation gap is a real problem for a benchmark paper, the cross-benchmark comparison overclaims, and the word-problem analysis is uncontrolled. The evaluation set (366 problems) is also on the smaller side for a benchmark whose primary contribution is evaluation. The paper has clear value and the niche is real, but in its current form it falls just below the acceptance bar due to the grader reliability gap and inadequate controls.

**Score: 5.0 — Borderline Reject**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>