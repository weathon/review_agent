Now I have a thorough understanding of the paper. Let me produce the consolidated review.

---

## Summary

EFFI-CODE introduces a dataset construction pipeline and resulting fine-tuning dataset aimed at improving *both* the efficiency and correctness of LLM-generated code. The authors collect ~780K candidate tasks from eight HuggingFace datasets, apply a multi-step filtering and cleaning process (including GPT-3.5-based filtering for risky operations and non-algorithmic tasks), generate and validate unit tests, and then apply SOAP (self-optimization based on overhead profiling) using DeepSeek-Coder-V2-Lite as a teacher model. The resulting 9,451-task dataset is used to fine-tune multiple LLMs (DeepSeek-Coder, Qwen2.5-Coder, CodeLlama), yielding substantial reported gains on HumanEval and EffiBench in both pass@1 and execution-time metrics.

---

## Strengths

- **Explicit disentanglement of efficiency-specific supervision from generic code SFT (Table 6).** The paper directly compares fine-tuning on EFFI-CODE's optimized solutions vs. fine-tuning on the same tasks with their original canonical (unoptimized) solutions. For DeepSeek-Coder-6.7B-base, canonical-solution SFT yields 15.2% pass@1 and a slight *increase* in ET, while EFFI-CODE yields 59.8% pass@1 and a 41% ET reduction. This is the key experiment for the paper's core claim, and it is executed and included — most papers of this type omit it entirely.

- **Demonstration that an open-source teacher matches or exceeds proprietary teachers for downstream pass@1 (Table 5).** DeepSeek-Coder-V2-Lite as teacher achieves 59.8% downstream pass@1 for the base model, compared to 39.0% for GPT-4o and 29.9% for Claude-3.5-Sonnet. This is a non-obvious and practically valuable finding: proprietary teachers produce more efficient code locally but yield worse generalisation after fine-tuning, suggesting a distribution-alignment effect that the paper exposes but does not fully explain.

- **Scalability across model sizes (Table 4) and training objectives (Table 7).** The dataset is evaluated across 1.3B, 6.7B, and 33B models and under SFT, DPO, and ORPO — providing breadth that goes beyond most dataset papers. All sizes show consistent efficiency and correctness improvements, strengthening the generalizability claim within the Python/DeepSeek family.

- **Well-characterised efficiency gains within the training data (Figure 1).** The paper shows execution time distributions for both original (mean 1.14s) and SOAP-optimised (mean 0.31s) solutions, demonstrating that SOAP reliably optimises the selected tasks, not just a few outliers.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Decontamination against HumanEval and EffiBench is not performed and the justification is inadequate.** Footnote 2 states: "Data decontamination was not included in the filtering process as most of the tasks we collected have been decontaminated, such as OSS-Instruct." However, several source datasets (e.g., APPS, Alpaca, Glaive, Dolphin) have no such decontamination guarantee, and the claim is not backed by any verification. With pass@1 for DeepSeek-Coder-6.7B-Instruct jumping from 43.3% to 76.8% — a 33.5-point gain from a 9k-task dataset — the risk of near-duplicate overlap with HumanEval's 164 problems cannot be dismissed without explicit verification. This is not a niche concern: it is a standard prerequisite for ICLR evaluation, and its absence leaves the most headline-grabbing results untrustworthy.

- **Efficiency evaluation is fragile when the overlap set is very small.** The paper correctly conditions efficiency comparisons on the set of tasks solved by *both* the original and fine-tuned model. However, in several important rows, this overlap is tiny: DeepSeek-Coder-6.7B-base on HumanEval has 7.3% overlap; DeepSeek-Coder-6.7B-Instruct on EffiBench has only 1.0% overlap. When fine-tuning raises pass@1 from 1.3% to 51.6% on EffiBench, the 1.0% overlap set is an extreme outlier of difficulty. Any efficiency gains reported for that model on EffiBench (7.1% ET reduction) are based on a handful of tasks and cannot be taken as representative. The paper reports these numbers in Table 2 without flagging the fragility.

- **Near-zero memory improvements contradict broad efficiency claims.** Across virtually all experiments in Tables 2–7, MU and NMU show 0.0% change. The paper's abstract and framing claim to improve "efficiency" in terms of both execution time and memory. The training data (Figure 1) does show memory improvements (26.50 → 6.03 MB), suggesting the SOAP process optimises memory in the dataset, but this is not transmitted to inference behaviour. The paper does not explain this asymmetry, nor does it appropriately caveat the claim that EFFI-CODE improves "efficiency" broadly.

### Minor

- **The teacher model tradeoff in Table 5 is substantive but unexplained.** GPT-4o as teacher produces more time-efficient code (ET = 0.20s for instruct student vs. 0.21s for V2-Lite student) but catastrophically lower pass@1 (9.8% vs. 76.8% for the instruct model). This is a core empirical mystery: a better teacher produces worse generalisation. The paper notes the phenomenon but offers no hypothesis. Possible explanations include distribution mismatch (GPT-4o uses more advanced library calls or code structure unfamiliar to the student), shorter solutions that sacrifice algorithmic clarity, or simply that proprietary-teacher code is harder for small models to imitate faithfully. Without analysis, this finding actually undermines the paper's framing — it suggests the gains are as much about stylistic alignment as about learning efficiency.

- **Test case validity relies on a circular assumption.** Step 3 generates test cases with GPT-3.5-turbo and accepts them if they pass the initial solution. As the paper acknowledges, initial solutions are "usually correct," but a non-trivial fraction from synthetic datasets may contain bugs. A test case that passes a subtly buggy solution may validate a wrong specification. This is particularly concerning because the entire SOAP optimisation loop then evaluates correctness against these same tests, so a flawed test-solution pair can be consistently "correct" throughout the pipeline and propagate into fine-tuning data. The paper quantifies neither the initial-solution error rate nor the test-case false-acceptance rate; even a manual audit of a small sample would substantially increase confidence.

- **Non-monotonic ablation in Table 3 is unexplained.** For the base model, the NET (normalised execution time) improvement is 7.1% at 25% data and only 5.5% at 75% data, then jumps sharply to 46.4% at 100%. The pass@1 curve is flat between 50% and 75% (54.3% both) and then jumps 5.5 points. The paper attributes this simply to "more data helps" but the shape suggests a threshold effect or a particularly valuable stratum in the final quartile of data. This deserves analysis — understanding what types of tasks drive the final gain would strengthen both the ablation interpretation and the dataset design.

- **Efficiency metrics for already well-aligned instruct models on EffiBench are marginal.** Qwen2.5-Coder-7B-Instruct on EffiBench shows only 2.3% ET and 0.7% TMU reduction despite a dramatic pass@1 increase (3.3% → 61.0%). The paper does not discuss why already-aligned instruct models show weaker efficiency gains on this benchmark. This may reflect ceiling effects, benchmark-specific distributions, or that EFFI-CODE's efficiency signal degrades once the model already has strong instruction following.

- **SOAP prompt and selection strategy underspecified.** Key details needed for replication are absent from the main text and presumably absent from the appendix (given it was removed in the review copy): the exact optimization prompt given to DeepSeek-Coder-V2-Lite, whether the five iterations are chained (output of iteration *i* is input to *i+1*) or independent, the selection criterion across iterations (best ET? best ET among correct solutions?), and the hardware/environment used for profiling. Since execution-time measurements are hardware-sensitive, omitting profiling environment details limits reproducibility.

### Tiny

- **TMU units are inconsistent.** Table 2 headers show "Mb/s" while Tables 5–7 show "Mb*s." The two units represent entirely different quantities. The correct unit should be clarified uniformly.

- **"Inefficient" terminology overstates the original solutions' flaw.** The paper calls dataset-provided canonical solutions "inefficient" throughout, but Step 6 explicitly acknowledges that some may already be efficient and are merely removed because SOAP cannot improve them. Using "original" or "initial" rather than "inefficient" would be more accurate.

---

## Nice-to-Haves

- **Controlled comparison against a general code SFT baseline (e.g., Code Alpaca / WizardCoder at the same dataset size) across all models.** Table 6 performs the critical comparison for DeepSeek-Coder-6.7B only. Extending this to Qwen and CodeLlama would provide stronger evidence that the efficiency-specific component of EFFI-CODE generalises beyond DeepSeek-style models.

- **Evaluation of efficiency on newly-solved (non-overlap) tasks.** When pass@1 nearly triples, the fine-tuned model solves many tasks that the original model could not. The paper never reports whether these new solutions are themselves efficient, which would reveal whether the model is learning to produce efficient code generally or only on tasks it was already "nearly solving."

- **Out-of-distribution evaluation (e.g., MBPP, LiveCodeBench).** All evaluation is on HumanEval and EffiBench, which may partially overlap with the training source distribution. An OOD benchmark would strengthen the transferability claim.

- **Qualitative analysis of what types of optimisations the model learns.** The single case study (Figure 3) is insufficient to distinguish algorithm-level changes (e.g., O(n²) → O(n log n)) from micro-optimisations (e.g., caching, avoiding redundant calls). A systematic categorisation of a sample of optimisations would clarify whether the model learns generalizable efficiency reasoning.

- **Reporting the computational cost of dataset construction** (GPU-hours for SOAP, API costs for GPT-3.5-turbo filtering) to help others assess whether the pipeline is reproducible outside well-resourced settings.

- **A scatter plot of per-task efficiency before vs. after fine-tuning** on the overlap set would reveal whether efficiency gains are broad (most tasks improve slightly) or concentrated (a few tasks improve dramatically while others degrade), which would materially affect interpretation of averaged metrics.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **"Cross-language evaluation is missing" (Harsh Critic).** The paper's contribution is explicitly a Python instruction-tuning dataset. The framework is described as adaptable to other languages, but no language-extension claim is made empirically. Criticising the absence of cross-language evaluation is scope creep. The core Python results are evaluable on their own merits.

- **"Comparison with PIE is unfair because PIE is C++" (implicit in Harsh Critic framing).** Per review guidelines, this should be removed. PIE is trained on C++ data evaluated on C++ problems; EFFI-CODE is trained on Python but must use CodeLlama for comparison because PIE's release is CodeLlama-family only. The asymmetry (PIE has domain alignment, EFFI-CODE does not) actually *favours* PIE, making EFFI-CODE's superiority a stronger result, not a weaker one.

- **"The paper needs confidence intervals and significance tests for pass@1 on HumanEval" (Harsh Critic).** Single-run pass@1 with greedy decoding is the norm in this community and in the benchmarks cited. Table 8 additionally demonstrates execution-time robustness with five repeated runs. Demanding significance testing for large-scale code benchmarks goes beyond community standards.

- **"Requesting theoretical proofs for efficiency claims" (implicit in Harsh Critic).** This is an empirical systems paper; theoretical analysis of when SOAP-style optimisation generalises is not expected or standard.

- **"The dataset is too small compared to WizardCoder's 100k+"** (Reviewer 2). The dataset is filtered to 9,451 high-quality efficiency-annotated tasks, which is appropriate for the filtering depth applied. Demanding a larger dataset for its own sake is a generic weakness that does not target the specific quality tradeoffs made here.

- **"The 'orthogonality' claim needs experimental proof"** (Harsh Critic). The paper states that its method is "orthogonal to existing synthetic techniques," meaning combining them is plausible, not that the combination has been demonstrated. This is a reasonable framing claim, not a testable empirical assertion, and demanding a compositional experiment exceeds the paper's stated scope.

---

## Novel Insights

The most genuinely novel finding in this paper — underemphasised in the text — is the **inverse relationship between teacher model quality and student model generalisation** seen in Table 5. GPT-4o and Claude-3.5-Sonnet produce locally more efficient code than DeepSeek-Coder-V2-Lite, yet fine-tuning on their outputs yields drastically worse downstream pass@1 (9.8% and 11.0% respectively vs. 76.8% for the open-source teacher). This suggests a sharp distribution-alignment bottleneck: the efficiency patterns of frontier proprietary models may be too syntactically or structurally distant from the target model's hypothesis space to be imitable. If confirmed and studied, this would have broad implications for knowledge distillation in code generation beyond efficiency tuning.

---

## Suggestions

1. **Run a decontamination check against HumanEval and EffiBench before the camera-ready version.** Even a simple exact-match and edit-distance filter over problem descriptions and canonical solutions would substantially increase confidence in the pass@1 numbers. This is not optional for ICLR.

2. **In Table 2, flag rows with overlap < 5% as "low-confidence efficiency estimates"** and move the efficiency discussion for those rows to appendix or caveat text. The main efficiency claim should rest on rows where the overlap is large enough to be representative (e.g., Qwen2.5-Coder-7B-Instruct on HumanEval at 51.2%).

3. **Add a brief analysis to the teacher model section (Section 4.2, Table 5)** hypothesising why open-source teacher alignment outperforms proprietary teacher quality. Even a qualitative code comparison of one task across teacher types would be illuminating.

4. **Report efficiency metrics for the newly-correct (non-overlap) tasks** alongside the overlap metrics. This can be done post-hoc without new fine-tuning and would directly answer whether the model generates efficient code on tasks it newly learns to solve.

5. **Add a limitations section** explicitly acknowledging: (a) the circular validation of test cases against initial solutions; (b) the selection bias toward tasks with optimisation headroom (Step 6 filter); (c) Python-only evaluation; and (d) near-zero memory improvements despite dataset-level memory gains.

6. **Fix the TMU unit inconsistency** across tables (Mb/s vs. Mb*s) and add precise metric formulas to the main text for NET, NMU, and NTMU.

7. **Include the SOAP optimisation prompt and iteration selection strategy** in the appendix for full reproducibility.

---

**Axis evaluations:**

- *Novelty:* Moderate. The specific contribution — an efficiency-focused instruction-tuning dataset for Python LLMs — is a genuine first in this form, but the core optimisation method (SOAP) is adopted from prior work, and the dataset construction is primarily an engineering pipeline with no new algorithmic ideas.
- *Technical soundness:* Moderate. The pipeline is reasonable and the ablation studies are thoughtful, but the test-case validation is circular, the filtering process lacks quality audits, and the contamination concern is unaddressed.
- *Empirical support:* Moderate-to-good. Breadth across models, sizes, and training objectives is a clear strength. The critical Table 6 experiment is present. However, the most striking headline numbers (33+ point pass@1 gains) lack contamination validation, and efficiency claims on small-overlap rows are fragile.
- *Significance:* Moderate-to-high. Code efficiency is underexplored and the open-source release of data, weights, and code could meaningfully advance the field if the contamination concern is resolved.
- *Clarity:* Adequate but imperfect. The evaluation protocol (overlap definition, per-metric computation) needs to be more prominent and consistently described; metric unit inconsistencies create confusion.