Now let me check the calibration papers for scoring:Now I have enough information to write the consolidated final review.

---

## Summary
HARDMATH introduces a benchmark of 1,466 graduate-level applied mathematics problems centered on asymptotic approximation methods (dominant balance, Laplace's method, Taylor corrections, etc.), algorithmically generated using SymPy/SciPy and verified with a 10% numerical error threshold. A curated subset of 366 problems (HARDMATH-MINI) is used to evaluate LLMs; even the best model (o1-mini) achieves only 62.3% accuracy with 5-shot CoT, substantially lower than published scores on existing math benchmarks.

---

## Strengths

- **Algorithmic generation with numerical self-verification addresses a real scalability bottleneck.** Unlike every comparable benchmark (ARB at 34 problems, GHOSTS-GRAD-TEXT at 130), HARDMATH's pipeline produces arbitrarily many problems without manual curation or copyright exposure. The combination of SymPy-based symbolic solving, SciPy numerical verification, and a 10% error gating is a concrete technical contribution that cannot be replicated by collecting exam problems.

- **Targets a genuinely underrepresented reasoning modality.** As the paper demonstrates (Box 1, Sec 3.1), the Method of Dominant Balance requires identifying the dominant terms in different parameter regimes, making self-consistency checks, and producing regime-conditional approximations—none of which appears in existing large-scale math benchmarks. The motivation is principled, not opportunistic.

- **Fine-grained error analysis reveals non-trivial insight.** Figure 4 shows that 5-shot CoT shifts GPT-4 errors on Roots from "incorrect dominant balance terms" (66.1% → 9.5%) to "missing dominant balance cases" (27.4% → 50.8%), demonstrating that CoT teaches the technique but not exhaustiveness. This is more actionable than a simple accuracy leaderboard.

- **Clear within-benchmark finding on difficulty variation.** ODEs are consistently the hardest category across all models (GPT-4: 21.7%, o1-mini: 30.6%), while Nondim is the easiest. This maps onto mathematical intuition and is well-supported by Table 2 and Figure 3.

---

## Weaknesses

### Fatal
*(None. The core claim—LLMs struggle with asymptotic applied math—is credibly supported within the benchmark, and the dataset genuinely exists and is released.)*

### Major

- **The procedural LLM grader is under-validated, directly undermining the headline accuracy numbers.** The paper acknowledges that GPT-4o is used as a procedural grader for Roots, ODEs, and Integrals, and that "we manually verify a subset of grading responses and found that LLM-based grading is closely aligned with human grading" (Sec 4.1). No sample size, Cohen's kappa, Pearson correlation, or disagreement rate is given. Because approximate solutions admit multiple valid forms, exact-answer matching is inherently incomplete for this benchmark, making the procedural grader's validity *central* rather than ancillary. Additionally, GPT-4o grades outputs from GPT-4 and GPT-3.5, which introduces potential leniency bias toward stylistically similar outputs. Without quantitative grader validation, the reported accuracy figures carry unquantified error that could materially affect relative model rankings.

- **The benchmark's actual scope is far narrower than the title and framing suggest.** Section 3.1 explicitly states: "One key commonality between all HARDMATH problems is the use of the *Method of Dominant Balance* in calculating solutions." All seven problem types—nondimensionalization, polynomial roots, root corrections, ODEs, traditional integrals, Laplace integrals—apply this single technique family from a single graduate course (Bender & Orszag, 2013). Absent are boundary layer theory, WKB approximation, regular/singular perturbation series, variational methods, matched asymptotics, and other staples of advanced applied mathematics. This is not inherently disqualifying, but calling the paper "HARDMATH: A Benchmark Dataset for Challenging Problems in Applied Mathematics" and claiming it evaluates "diverse mathematical approaches" significantly overstates the scope. The dataset tests one technique family, not applied mathematics broadly.

- **No data contamination analysis.** The problems are algorithmically generated from the Bender & Orszag textbook framework, which is widely cited and whose problem-solving procedure likely appears verbatim in LLM training corpora. The absence of any contamination investigation—even a simple textbook overlap analysis or canary test—leaves open whether reported model performance reflects genuine asymptotic reasoning or in-context recall of familiar templates.

### Minor

- **No human performance baseline.** The paper states o1-mini achieves 62.3% and GPT-4 achieves 43.8%, but provides no reference point for what a human graduate student or instructor achieves. Without a ceiling, it is impossible to interpret whether the LLM gap is large or modest, or whether the benchmark is at the right difficulty calibration.

- **Template-based generation raises template-overfitting concerns that are not analyzed.** Polynomial root problems all follow the form εx^{n₁} ± x^{n₂} ± 1; traditional integrals all follow I(ε) = ∫₀ᵃ 1/(ε + P(x)) dx. A model that learns the structural template for one problem instance could solve many others without understanding the underlying asymptotics. The paper does not analyze whether performance generalizes beyond these templated forms, nor does it test any retrieval or shallow pattern-matching baseline.

- **Limited and partially dated model evaluation.** Only five models are tested, including GPT-3.5 and CodeLlama-13b, which are not frontier models at time of submission. No math-specialized models (Qwen2.5-Math, DeepSeek-Math, Mathstral) are included despite their strong performance on existing math benchmarks. With only 5 models, the benchmark's power to discriminate among current frontier systems is limited.

- **Cross-benchmark comparisons are methodologically confounded.** The paper compares HARDMATH-MINI results to published scores on MATH, GSM8K, and GHOSTS under different prompting schemes (0-shot vs 5-shot vs 8-shot), different model versions, and different scoring rules. The comparison is suggestive that HARDMATH is harder, but cannot be treated as a rigorous cross-benchmark ranking.

### Trivial

- **The word-problem evaluation (40 problems, 1 model, different hints) cannot isolate the effect of contextualization.** The comparison between HARDMATH-MINI (43.8%) and word problems (28.1%) changes both the dataset and the prompting protocol simultaneously (hints removed for word problems), so neither the magnitude nor the cause of the drop can be interpreted cleanly. The result is descriptive only.

---

## Nice-to-Haves

- **Tool-augmented evaluation** (e.g., GPT-4 with Python/SymPy code interpreter). The paper's own motivation explicitly highlights tool use as a key skill these problems require, yet all experiments test pure text-in/text-out models. Demonstrating that tool-augmented agents improve (or not) would directly validate the stated motivation.

- **Out-of-template generalization test.** Adding a small held-out set of problems with different functional forms (e.g., integrands not of the 1/(ε+P(x)) type) would let the authors demonstrate that models learn the method rather than the template.

- **Fine-tuning experiments on the larger HARDMATH.** The paper claims HARDMATH is useful for "model developments like fine-tuning" but presents no evidence. A fine-tuning ablation would substantiate this claim.

- **Quantitative evaluation of automated context generation.** Section 3.5 uses a verifier threshold of ">0.5 plausibility" with no calibration or human check. Human plausibility ratings on a small sample would ground this preliminary pipeline.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic] Claim that the "largest graduate-level dataset" framing is not established** — The paper explicitly footnotes the filtering methodology in Table 1 ("we report the number of relevant problems at a comparable difficulty"). The comparison is disclosed, not hidden. The claim is qualified and defensible given the disclosed comparison.

- **[Harsh Critic] Claim that HARDMATH-MINI lacks statistical justification for the 366-sample size** — Requesting formal bootstrap confidence intervals or sample-size power calculations is not standard practice for benchmark evaluation papers at ICLR. Moved to nice-to-have territory rather than a weakness.

- **[Harsh Critic] Claim about regime-boundary verification and transition behavior being unvalidated** — The paper describes visual verification by plotting analytical solutions against numerical ground truths across a range of values in each regime (Sec 3.2). This is a legitimate validation step; the criticism that transition behavior should be separately verified is beyond what benchmark papers in this area typically provide.

- **[Harsh Critic] Framing HARDMATH as unsuitable for tool-use benchmarking because no experiments are provided** — This is a positioning claim in the paper's motivation section, not an empirical claim requiring proof. The observation that formal tools like Lean cannot handle approximate analytical solutions is simply correct; no experiment is needed to validate it.

- **[Neutral] Suggestion to rename the benchmark** — Pure editorial/style feedback; removed per hard rules.

---

## Novel Insights

The most technically interesting finding in this paper—confirmed by Figure 4—is that few-shot CoT does not merely improve accuracy uniformly: it shifts GPT-4's errors from *incorrect technique application* (wrong dominant balance terms) to *incomplete case enumeration* (missing some dominant balance cases). This means CoT successfully teaches the qualitative method but not the systematic exhaustiveness that asymptotic analysis requires. This failure-mode transition could be informative for broader research into how structured prompting changes reasoning strategies versus coverage completeness.

---

## Evaluation Against Quality Axes

- **Novelty:** Moderate. The algorithmic generation + asymptotic methods focus combination is genuinely novel among math benchmarks. The actual mathematical content (dominant balance) is mature and narrowly scoped.
- **Technical soundness:** Weak. Grader validation is inadequate; template diversity is unanalyzed; no contamination check.
- **Empirical support:** Moderate for within-benchmark claims; weak for cross-benchmark difficulty comparisons.
- **Significance:** Moderate. Asymptotic applied math is underrepresented and the gap is real. But the narrow scope limits transferability.
- **Clarity:** Adequate. The paper clearly explains the generation pipeline and problem types.

---

## Score and Decision

**Calibration papers:**

| Paper | Decision | Scores |
|---|---|---|
| U-MATH (university benchmark, ~1100 problems) | Reject | 5,5,6,5 |
| ARB (advanced reasoning, 34 problems) | Reject | 6,6,5,5 |
| MathCAMPS (template-based generation) | Reject | 6,6,6,5 |
| Omni-MATH (Olympiad, 4428 rigorous problems) | Accept | 8,8,6,5 |

HARDMATH is most comparable to ARB and U-MATH, both rejected. Its key differentiator over ARB is the algorithmic generation pipeline (a real contribution), and over U-MATH is the tighter domain focus. However, relative to ARB, HARDMATH has *weaker* grader validation (ARB reports GPT-4 rubric agreement statistics; HARDMATH does not) and narrower scope. Relative to MathCAMPS, HARDMATH targets harder problems but has fewer models, no contamination analysis, and the same template-diversity concern. Omni-MATH (accepted) has 3x more problems, rigorous human annotation, 33 sub-domains, and validated inter-annotator agreement—HARDMATH falls well short on all these axes.

The algorithmic generation contribution prevents this from being a low-score paper, but the combination of under-validated grading, narrow scope overclaimed as "applied mathematics," no human baseline, no contamination analysis, and limited model coverage keeps this below the acceptance bar for ICLR.

**Final score: 4.5**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>