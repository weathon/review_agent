## Summary

This paper systematically analyzes the GRPO loss function to identify which components are essential for improving mathematical reasoning in LLMs. Through controlled ablations on three small-scale instruction-tuned models (0.5B–1.5B), the authors find that (1) negative feedback is indispensable (positive-only training collapses), (2) group-relative advantage estimation is crucial (vanilla REINFORCE with raw rewards also collapses), and (3) PPO-style clipping is unnecessary. They propose RGRA, which removes clipping and policy ratios while retaining group-relative advantages, and report it outperforms GRPO in 17 of 27 benchmark comparisons across 9 mathematical and STEM tasks.

## Strengths

- **Clean, well-motivated ablation design**: The paper decomposes GRPO into clearly delineated variants (positive-only advantages, direct rewards, RGRA with clipping removed), each isolating a specific design choice. This systematic approach—testing what happens when each component is removed—is more informative than the typical "propose a variant and beat the baseline" paradigm and directly addresses a genuine question in the community about whether GRPO's complexity is justified.

- **Compelling demonstration that negative feedback prevents collapse**: The training dynamics in Figure 1 clearly show that positive-only and RAFT-trained 0.5B models suffer reward/response-length collapse within ~20 steps. This is a sharp, actionable finding: practitioners who might be tempted to train only on successful completions receive a clear warning that doing so destabilizes learning, particularly for smaller models.

- **Broad multilingual benchmark coverage**: Evaluation across 9 benchmarks spanning English math, Chinese math, and STEM tasks provides a more thorough assessment than typical single-benchmark evaluations in this space. The inclusion of Chinese-language benchmarks (CMATH, CN-Middle-School, Gaokao2024) is a meaningful strength that tests cross-lingual generalization from English-only training data.

## Weaknesses

### Major:

- **Unanalyzed failure cases undermine the generality of the "clipping is unnecessary" claim**: RGRA underperforms GRPO in 10 of 27 comparisons, and some gaps are substantial. Most notably, on Gaokao2024-STEM for Llama3.2-1B, GRPO achieves 17.2 while RGRA drops to 11.4—a 34% relative degradation. Similarly, on MATH for Qwen2.5-1.5B, GRPO achieves 30.4 vs. RGRA's 29.1. The paper counts wins (17/27) but provides no analysis of *when or why* RGRA fails. Without understanding the conditions under which clipping helps, the practical guidance the paper aims to provide is incomplete. The claim that PPO-style constraints are "not required" is overstated given these counterexamples.

- **Incomplete ablation study for a paper claiming to identify "essential" components**: The paper ablates clipping and positive filtering, but does not test (a) removing KL regularization from RGRA, or (b) varying the group size *G*. Since RGRA retains KL regularization, it is possible that KL—not advantage estimation alone—is doing the stabilization work previously attributed to the combination of advantage estimation and clipping. Without an "RGRA without KL" condition, the decomposition of essential vs. inessential components remains unfinished. Group size *G* similarly goes untested, despite being a core hyperparameter of the group-relative advantage mechanism.

### Minor:

- **Experiments limited to small models (0.5B–1.5B)**: The authors acknowledge this limitation, noting hardware constraints. However, the paper's title and claims address "teaching LLMs to reason" broadly, and the dynamics of policy updates may differ substantially at 7B+ scales where PPO clipping was originally motivated. The findings should be scoped more carefully to small models until larger-scale validation is available.

- **Efficiency claims lack empirical support**: The abstract and conclusion describe RGRA as a "more transparent and efficient alternative," but no wall-clock time, memory usage, or FLOPs comparisons are provided. While removing clipping arguably simplifies implementation, the actual computational savings are not demonstrated, and "efficient" in the RL context typically refers to sample or compute efficiency, not just code simplicity.

- **No statistical significance testing or multiple-seed evaluation**: All results appear to be from single runs. While single-run evaluation is common in recent large-scale RL-for-LLM work, the margins between RGRA and GRPO are often small (1–3 percentage points), making it difficult to determine whether observed differences reflect genuine improvements or run-to-run variance. This is particularly concerning given the mixed results noted above.

- **Domain restricted to mathematical reasoning**: The paper claims implications for "reasoning-focused post-training" broadly but evaluates only on math and STEM tasks. Whether the findings transfer to logical reasoning, code generation, or other reasoning domains remains unknown.

### Trivial:

- The abbreviation "ft" (fine-tuning) in Tables 1–3 is not defined in the table captions.

## Nice-to-Haves

- Comparison with other recent GRPO variants (DAPO, S-GRPO, CPPO) mentioned in the introduction, to contextualize RGRA's improvements against specialized alternatives rather than only against vanilla GRPO.
- Quantitative analysis of reasoning trace emergence (e.g., percentage of outputs containing reasoning steps, correlation between trace length and correctness) rather than the single qualitative example in Figure 2.
- Ablation of RGRA without KL regularization to complete the decomposition of essential components.
- At least one experiment at 7B+ scale to test generalizability of the core findings.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Formatting/typo complaints** (e.g., "prefernces" typo, equation formatting in Section 2.2): Per hard rules, pure formatting/style nitpicks are removed. The parser note explicitly states formatting artifacts are extraction issues, not paper problems.

- **Demand for human evaluation**: For mathematical reasoning with verifiable answers, automated accuracy evaluation is the standard in this area. Requesting human evaluation is scope creep for a purely algorithmic contribution.

- **Broader impact / negative societal impact discussion**: Not required by ICLR and not relevant to assessing the technical contribution.

- **Demand for comparison with DPO variants**: DPO operates under a different paradigm (offline preference optimization) and is outside the paper's stated scope of analyzing GRPO-family RL objectives.

- **Demand for theoretical proofs of why clipping is unnecessary**: The paper is positioned as an empirical analysis. Requesting theoretical gradient-variance analysis is a nice-to-have, not a core requirement for this type of contribution.

- **Reproducibility concerns about GRPO implementation fidelity**: Per hard rules, nitpicks about reproducibility of implementation details are removed. The paper provides code and hyperparameters.

- **RAFT collapse vs. non-trivial test scores "contradiction"**: The paper states collapse occurs "particularly in the 0.5B model" and that larger models show "reward stagnation and gradual shortening" rather than immediate collapse. The Llama3.2-1B RAFT results are consistent with this description. This is not a genuine contradiction.

## Novel Insights

The most interesting observation emerging from the reviews is the tension between RGRA's overall win record and its specific failure modes. RGRA tends to outperform GRPO on the benchmarks where absolute accuracy is lower (harder tasks like MATH, OlympiadBench), but can underperform on higher-accuracy or differently-structured tasks (Gaokao2024-STEM for Llama3.2). This pattern hints that removing clipping might allow more exploratory updates that benefit harder problems but occasionally overshoot on easier ones—a hypothesis the authors do not explore but that could meaningfully advance understanding of when simplicity helps vs. hurts in RL for LLMs.

## Suggestions

- **Analyze the 10/27 cases where RGRA underperforms GRPO**: Identify common properties of these benchmarks/models (e.g., language, difficulty, reward distribution) to characterize when clipping provides value. This would transform the paper from "clipping is unnecessary (mostly)" to "clipping is unnecessary under conditions X, Y, Z," which is far more actionable.
- **Add an RGRA-without-KL ablation**: This is the single most important missing experiment. If RGRA without KL remains stable, the "essential components" story becomes much cleaner. If it collapses, KL regularization—not just advantage estimation—deserves credit for stability, which changes the paper's narrative.
- **Calibrate claims to match evidence**: Replace "PPO-style constraints are not required" with "PPO-style constraints are not required in the small-model math-reasoning settings we tested," and discuss the conditions under which they may still be beneficial.

---

**Axis Assessments:**

- **Novelty**: Moderate. The systematic ablation approach is valuable, but RGRA itself is a straightforward combination of existing ideas (REINFORCE + group-relative advantages from GRPO). The contribution is primarily empirical and analytical rather than algorithmic.

- **Technical soundness**: Partial. The experimental design is clean, but the incomplete ablation (no KL removal, no group-size variation) and unanalyzed failure cases leave core claims insufficiently supported.

- **Empirical support**: Mixed. The broad benchmark coverage and clear collapse demonstrations are strong, but the lack of statistical testing, the 10/27 underperformance cases, and the missing ablations weaken the empirical case for the paper's stronger claims.

- **Significance**: Moderate. If validated at scale and the failure conditions are characterized, this could meaningfully simplify post-training pipelines. Currently, the impact is limited by the small-model-only evaluation and unanalyzed failure modes.

- **Clarity**: Good. The paper is well-structured and the ablation variants are clearly defined. Minor issues with undefined abbreviations and claim calibration do not substantially impede understanding.