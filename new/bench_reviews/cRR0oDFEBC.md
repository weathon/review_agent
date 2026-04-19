Now let me do calibration searches to anchor the scoring.Now I have enough information to write the final consolidated review. Let me synthesize everything.

---

## Summary

AUTOIF introduces a method for automatically generating instruction-following training data by converting instruction correctness verification into code execution: LLMs generate instructions, write Python verification functions and unit tests, and execution feedback is used for rejection sampling to construct SFT and DPO training pairs. A multi-stage quality pipeline (cross-verification of generated code, NLI-based back-translation consistency checks, and LLM-based query-instruction compatibility scoring) filters the generated data. The paper demonstrates improvements across three training strategies (SFT, Offline DPO, Online DPO), two model families (Qwen2, LLaMA3), and two alignment settings (self-alignment and strong-to-weak distillation), achieving a reported 90.4% loose instruction accuracy on IFEval with LLaMA3-70B.

---

## Strengths

- **Novel transformation of instruction verification into code execution** (Section 3.2, Figure 1): The core insight that verifiable instruction constraints (length, keyword, format) can be checked by auto-generated Python functions, eliminating the need for human annotation or proprietary LLM judges for correctness signals, is technically clean and well-motivated. Using cross-validation between independently generated verification functions and unit tests to filter quality is a principled design.

- **State-of-the-art results on IFEval and FollowBench** (Table 1): LLaMA3-70B achieves 90.4% Instruction (L) and Qwen2-72B achieves 88.0% on IFEval, surpassing GPT-4's 85.4%. The gains hold across multiple model sizes and training regimes. FollowBench (evaluated by GPT-4 judge on non-verifiable instructions) also shows consistent +3–5% improvements for large models, providing evidence that gains are not confined to programmatically verifiable constraints.

- **Principled ablation study** (Table 4): The ablation on specific quality components shows Cross Verification contributes the most (−3.0 Ins. L), followed by Query Quality Verification (−2.4) and Back-translation (−1.7), with removing all components yielding a −3.8 drop. This systematically validates each design choice and is informative about the relative importance of components.

- **Cross-domain generalization to non-verifiable instructions** (Table 2): AUTOIF-trained Qwen2-7B shows gains on InfoBench (+3.52), MT-Bench (+0.19), and Arena Hard (+6.71 winrate) with Online DPO — benchmarks that evaluate general instruction quality, not programmatically verifiable constraints. This is the most important evidence that training on verifiable instructions generalizes.

- **Data efficiency** (Figure 4 right, Table 5): Even 1/64 of DPO data produces ~55% IFEval Prompt (L) (an 11.4-point gain), indicating high per-sample quality. Table 5's correlation between supervision model code ability (MRPP), pass rate, and final IF performance is an informative analysis of the data synthesis pipeline's quality drivers.

- **Open code and data release**: The paper commits to releasing SFT and DPO datasets and pipeline code, increasing reproducibility for a subfield where data generation pipelines are often opaque.

---

## Weaknesses

### Fatal
None.

### Major

- **Cross-domain validation is only reported for Qwen2-7B, not for the headline large models**: Table 2 (InfoBench, MT-Bench, Arena Hard) is only evaluated for Qwen2-7B. The headline result — 90.4% on IFEval with LLaMA3-70B — is not accompanied by any cross-domain validation showing that this large model also generalizes to non-verifiable instruction benchmarks. The distribution alignment concern (see below) is partially but not fully mitigated by the 7B cross-domain results, since the model sizes and training configurations differ substantially.

- **Type-level distribution alignment between training data and IFEval is unaddressed**: AUTOIF specifically trains on *verifiable* instructions (those checkable by code), and IFEval evaluates *verifiable* instructions (format, length, keyword constraints, which the paper itself describes as "25 types of verifiable instructions"). The contamination analysis in Figure 6 only checks n-gram overlap and rephrasing detection — it cannot rule out type-level structural alignment. Two instructions can share zero n-grams while measuring identical constraint classes. The "first to surpass 90%" claim cannot be confidently attributed to general instruction-following improvement rather than to distributional alignment within the verifiable instruction category. The FollowBench results partially address this (since FollowBench uses GPT-4-as-judge), but FollowBench results for the 70B models are modest (~3–10% improvement) and the cross-domain evidence for the headline model is absent.

### Minor

- **LLM-as-judge step in query quality verification contradicts the paper's core framing**: Section 3.3 employs an LLM to score instruction-query compatibility on a 1–10 scale, with samples below 8 filtered out. This is LLM-as-judge for quality control, the exact paradigm the introduction criticizes ("even advanced LLMs can make mistakes, and the reliability of the distilled data cannot be guaranteed"). The paper offers a reasonable justification (semantic incompatibility is hard to detect with simple NLI), but this step receives no validation (inter-rater agreement, error analysis) and the threshold of 8 is not ablated. The paper should more explicitly acknowledge this as a pragmatic compromise rather than treating code execution as the sole quality signal.

- **LLaMA3-8B base model's pathologically low FollowBench baseline inflates reported gains**: Table 1 shows LLaMA3-8B base achieves 10.0–14.3 on FollowBench levels, while LLaMA3-8B(ShareGPT) achieves 33–44 on the same levels. The "+36.6" gain for AUTOIF (LLaMA3-8B) at Level 1 is largely attributable to learning the output format expected by FollowBench via instruction tuning, not specifically to AUTOIF's verifiable instruction mechanism. The comparison should use LLaMA3-8B-Instruct (or at minimum LLaMA3-8B with ShareGPT) as the primary baseline for strong-to-weak experiments.

- **Missing Online DPO results for Qwen2-72B and missing "+SFT" for LLaMA3-70B self-alignment**: Table 1 shows blank ("−") entries for "+Online DPO" under Qwen2-72B self-alignment and "+SFT" under LLaMA3-70B self-alignment. Given that Section 4.1 explicitly states "On-policy Learning is More Effective," the absence of Online DPO for the largest models is a gap. The paper reports the best known configuration in the main "AUTOIF (X)" row without clearly explaining why other configurations are unavailable.

- **GPT-4 supervision substantially outperforms self-supervision, qualifying the self-alignment framing**: Table 3 shows GPT-4 as supervision model achieves 59.5 Prompt (L) vs. 46.6 for Qwen2-7B self-supervision — a 13-point gap. The paper frames AUTOIF as a "scalable and reliable method" without relying on proprietary models, but the strongest results for 7B-scale models depend heavily on GPT-4. The large-model (72B, 70B) self-alignment results are more defensible, but the 7B setting — where practitioners most need scalable open alternatives — shows a large proprietary model dependency.

- **ShareGPT integration's role in capability preservation is asserted but not ablated**: Section 4.1 states "we attribute this preservation largely to incorporating ShareGPT data during data synthesis," but there is no ablation comparing AUTOIF with vs. without ShareGPT data on capability metrics. The claim is plausible but unsupported by evidence within the paper.

### Trivial

- **Table 3 subscript notation is ambiguous**: The subscript "±X.X" in Table 3 (e.g., "+SFT: 44.5 ±0.9") denotes gain over baseline, but ± conventionally means standard deviation or error range. Table 1 uses "−X.X" with subscripts correctly for gains. The inconsistency will mislead readers who have not read the table caption carefully.

---

## Nice-to-Haves

- A coverage analysis of which constraint categories AUTOIF generates, cross-referenced with IFEval and FollowBench categories, would sharpen the interpretation of gains.
- A manually audited sample of verification functions (even 50–100) would establish what fraction are actually semantically correct rather than syntactically passing, addressing the reliability claim more directly.
- Reporting FollowBench and at least one cross-domain benchmark for LLaMA3-70B after AUTOIF would be the single most impactful addition to the current paper.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"First" framing critique**: The harsh reviewer criticizes AUTOIF's claim of being "first," pointing to Conifer and WizardLM. The paper explicitly cites Conifer in Section 2 and explains the differentiation (not relying on proprietary LLMs for quality signals). The "first" claim is scoped to "scalable and reliable method for automatically generating IF training data" without proprietary LLM quality judgment. This is a reasonable framing distinction, not a misrepresentation.

- **Scaling Figure 5 mixes model families**: The critic argues that Figure 5 mixes model families (Qwen1.5 to Vicuna) and therefore the trend line is invalid. The paper's modest claim — "AUTOIF delivers substantial and stable benefits across different base model parameter sizes" — is supported by the figure even across heterogeneous families. The figure is not intended to establish a single scaling law but to show robustness of gains.

- **Missing related works**: Per policy, not included as these cannot be verified.

- **Reproducibility concerns about hyperparameters, NLI model choice**: Critiques about the specific NLI model used for back-translation or undisclosed hyperparameters are removed as standard reproducibility nitpicks.

---

## Novel Insights

The most important insight surfacing from reviewer synthesis beyond the paper's own claims: the paper's genuinely novel methodological contribution (code-based verification) is conflated in the headline evaluation with a structurally favorable benchmark choice (IFEval = verifiable instructions). The paper's *actual* generalization claim is better supported by FollowBench and the three cross-domain benchmarks — but these are underemphasized relative to the IFEval headline. If the paper reframed its main result around cross-domain improvements and used IFEval as a supporting verification, the contribution would be both more defensible and easier to evaluate. The gap between GPT-4 supervision performance and self-supervision performance at the 7B scale also suggests the method's practical value for open-source practitioners is currently strongest at the 70B scale, where self-alignment is feasible.

---

## Suggestions

1. Add FollowBench and at least one cross-domain evaluation (InfoBench or Arena Hard) for LLaMA3-70B in the self-alignment setting — this is the most urgent gap.
2. Clarify Table 1's row hierarchy with explicit training labels for self-alignment rows; explain why certain configurations have blank entries.
3. Include a brief ablation or justification for the threshold-8 LLM query quality filter.
4. Acknowledge type-level distribution alignment with IFEval explicitly, and frame IFEval results accordingly.
5. Use LLaMA3-8B-Instruct (or LLaMA3-8B with ShareGPT SFT) as the primary strong-to-weak baseline to avoid inflated FollowBench delta claims.

---

## Score and Decision

**Calibration anchors:**

- **SALMON** (self-alignment with instructable reward model, comparable scope): Human scores 6, 6, 6, 8 → ~6.5, Accept. AUTOIF has stronger and more systematic empirical results than SALMON, with clearer ablations and a more mechanistically grounded approach.
- **LMSYS-Chat-1M** (large-scale instruction following dataset): Human scores 6, 8, 8, 8 → ~7.5, Accept spotlight. That paper is primarily a dataset contribution; AUTOIF is also a methodological contribution with broader evaluation.
- **RAIN** (self-alignment via self-evaluation): Human score 8, Accept poster. RAIN's mechanism is fundamentally different (inference-time); AUTOIF's training-time contribution is less elegant but more practically impactful.
- **Rejected alignment/instruction-tuning papers** (scores 3–5): These had weaker empirical evidence, less clear methodology, or fewer ablations than AUTOIF.

AUTOIF sits clearly above the 5-range rejected papers and is competitive with SALMON (6.5 average). The genuine major concerns — missing cross-domain for large models, type-level distribution alignment with IFEval — prevent a rating comparable to RAIN (8). The paper is methodologically sound, well-ablated, and reports strong results that generalize beyond verifiable instructions (at 7B scale). The weaknesses are real and worth the authors addressing but are not fatal to the paper's core contribution. I place this at **6.5**, reflecting a solid borderline-accept with clear actionable gaps.

**Evaluation dimensions:**
- *Originality*: Good — code-based verification for IF data generation is a clean and practical idea
- *Importance of research question*: High — scalable IF training data generation is a real bottleneck
- *Claims vs. evidence*: Partially overclaimed for the headline IFEval result; better supported for FollowBench
- *Soundness of experiments*: Mostly sound, with the noted gaps in cross-domain coverage for large models
- *Clarity of writing*: Adequate; Table 1 structure and some notation need improvement
- *Value to community*: High — open dataset, solid methodology, practical improvements

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>