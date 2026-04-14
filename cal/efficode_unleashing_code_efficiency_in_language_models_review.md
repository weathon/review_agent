=== CALIBRATION EXAMPLE 13 ===

# Final Consolidated Review
## Summary
EFFI-CODE introduces a dataset and pipeline for fine-tuning LLMs to generate code that is both correct and computationally efficient. Starting from ~780k code tasks aggregated across 8 HuggingFace datasets, the authors apply multi-stage filtering and an iterative self-optimization technique (SOAP, adapted from Huang et al. 2024a) to produce 9,451 efficiency-optimized training samples. Fine-tuning several open-source LLMs (DeepSeek-Coder, Qwen2.5-Coder) on this dataset yields substantial gains in both pass@1 correctness and execution time reduction on HumanEval and EffiBench, with the approach validated across SFT, DPO, and ORPO fine-tuning methods.

---

## Strengths

- **Simultaneous correctness and efficiency improvement via fine-tuning, grounded in evidence:** The paper shows empirically that a 9k-sample efficiency-focused dataset can substantially boost both pass@1 and runtime across multiple model families and sizes (Table 2, 4). This simultaneous gain, rather than the typical correctness–efficiency trade-off, is a concrete and specific empirical finding this area lacks.

- **Transparent ablation over dataset size, model size, teacher model, and fine-tuning method:** Rather than reporting a single best result, the paper systematically studies how performance changes with dataset proportion (Table 3), model scale from 1.3B to 33B (Table 4), choice of teacher (Table 5), and fine-tuning algorithm (Table 7). The consistent benefit across these axes strengthens confidence in the dataset's utility.

- **Comparison against canonical-solution SFT (Table 6) isolates the efficiency optimization's contribution:** Fine-tuning on the same tasks with only the original (unoptimized) canonical solutions improves pass@1 but fails to improve or even worsens efficiency (e.g., ET increases from 0.39s to 0.42s for DeepSeek-Coder-6.7B-base), directly demonstrating that the SOAP-optimized solutions are the key ingredient. This is a well-designed ablation most resource papers omit.

- **Robustness check over five runs (Table 8):** The standard deviation of efficiency metrics across five separate evaluations is negligibly small, ruling out hardware noise as an explanation for the improvements.

---

## Weaknesses

### Fatal
None identified that individually invalidates all results, but the combination of C2 and C3 below is serious enough to substantially undermine confidence in the headline correctness claims.

### Major

- **Unexplained and disproportionate pass@1 gains cast doubt on the efficiency-learning narrative (C2).** The most striking result in the paper is the massive pass@1 increase from efficiency-focused fine-tuning: DeepSeek-Coder-6.7B-instruct goes from 43.3% → 76.8%, and the base model from 7.3% → 59.8%. Standard SFT on similar-scale code datasets typically yields far more modest correctness gains. The paper frames these as a beneficial side-effect of efficiency training, but provides no mechanism or analysis. Plausible confounders include incidental topic coverage of HumanEval-like problems in the training set, or improved instruction-following from the fine-tuning format—neither of which would be attributable to efficiency learning. Without an explanation, the reader cannot disentangle whether the efficiency gains (measured on an overlap set whose composition depends heavily on which tasks the improved model now solves) are real or an artifact.

- **Data contamination is not rigorously addressed (C3).** The paper dismisses contamination concerns in a single footnote (#2): "Data decontamination was not included as most of the tasks we collected have been decontaminated, such as OSS-Instruct." Only one of the eight source datasets is cited for decontamination. Several others—Magicoder-Evol-Instruct-110K, CodeFeedback, Alpaca-143k—do not have documented decontamination against HumanEval. Given the exceptional pass@1 improvements, a rigorous contamination check (e.g., n-gram overlap between training prompts and HumanEval problem descriptions) is not optional at ICLR — it is a prerequisite for credibility of the correctness claims.

- **Counterintuitive teacher model results are unexplained and raise a methodological concern (C8).** Table 5 shows that fine-tuning on DeepSeek-Coder-V2-Lite-generated data yields 59.8% pass@1, while GPT-4o and Claude-3.5-Sonnet-generated data yield only 39.0% and 29.9% respectively — despite GPT-4o and Claude-3.5-Sonnet being substantially more capable models (and GPT-4o actually generating *faster* code, ET 0.20s vs. 0.21s). This inversion may be explained by differences in dataset size after filtering (GPT-4o/Claude solutions may fail test cases at higher rates, producing fewer surviving training samples), or by distribution mismatch between GPT-4o code style and the student model's pretraining. The paper acknowledges the result but provides no analysis. This is not merely a curiosity — it directly bears on whether the benefits come from *efficiency patterns in the training data* or from other data distribution properties that happen to be correlated with the teacher choice.

### Minor

- **Efficiency evaluation over small overlap sets weakens quantitative claims for low-pass@1 baselines.** Efficiency metrics are computed only on the intersection of tasks solved correctly by both the baseline and the fine-tuned model. For DeepSeek-Coder-6.7B-base with 7.3% pass@1 on HumanEval, this overlap is approximately 12 out of 164 problems. For DeepSeek-Coder-6.7b-instruct on EffiBench, the overlap is 1.0%. Efficiency statistics computed over such tiny sets are not statistically meaningful. The paper does not report the absolute number of tasks in each overlap set. Claims from models with ≥35% overlap (e.g., DeepSeek-Coder-6.7B-instruct on HumanEval) are far more credible than those from base models with 7.3% overlap; these should be distinguished clearly rather than presented uniformly.

- **Nonmonotonic efficiency improvement with dataset size is unexplained (C10).** In Table 3, for the base model, NET is roughly flat from 25% to 75% (1.96 → 2.03 → 1.93) and then drops sharply at 100% (1.13). For the instruct model, the same cliff appears (2.02 → 1.94 → 1.96 → 1.09). The paper asserts "a consistent trend of improvement" but the data shows no meaningful efficiency gain until the full dataset is used. This nonlinearity is unexplained and suggests the efficiency effect may be driven by a specific subset of examples rather than general learning.

- **Memory usage improvements on HumanEval are consistently zero for max memory metrics, contrary to the paper's broad efficiency claims.** Across all HumanEval experiments in Table 2, MU and NMU improvements are 0.0% for nearly every model. This contradicts the paper's framing of "improvements across all metrics." The partial improvements on EffiBench (e.g., 11.6% MU for DeepSeek-Coder-6.7b-base) are noted but the systematic null result on HumanEval MU/NMU is neither acknowledged nor explained.

- **Inconsistent baseline ET values across tables are not explained.** DeepSeek-Coder-6.7B-base shows ET=0.89s (Table 2), 0.99s (Table 3), and 0.39s (Table 6). The instruct model shows ET=0.59s (Table 2), 0.42s (Table 3/5), and 0.44s (Table 6). These differences are likely attributable to different overlap sets (different subsets of tasks), but the paper never explains this. Readers comparing numbers across tables will be confused. The overlap set sizes and compositions should be stated clearly per table.

- **Figure 1 presents a positively-selected efficiency distribution without noting the selection.** Step 6 of the pipeline retains only tasks where SOAP produced measurable efficiency improvement. Figure 1 then compares the "inefficient" vs. "efficient" distributions on this filtered subset. The large gap (mean ET 1.14s → 0.31s) is guaranteed by construction — it characterizes only tasks where optimization succeeded, not all algorithmic tasks. This should be stated explicitly in the caption.

### Tiny

- **Table 3 section description is confusing.** The text describes "individual" and "all" perspectives but Table 3 appears to show only one perspective. The reader cannot determine which view is presented.

- **GPT-3.5-turbo is used for risk filtering and test case generation**, introducing cost and reproducibility barriers for independently constructing the dataset. The paper does not discuss open-source alternatives or whether GPT-3.5 is a replaceable component.

---

## Nice-to-Haves

- **Evaluate on an out-of-distribution benchmark (e.g., LiveCodeBench)** to show efficiency gains generalize to unseen problems, rather than being potentially tied to training-distribution overlap with HumanEval.

- **Input-size scaling experiments** (e.g., plot ET vs. input size on log-log scale) would distinguish genuine algorithmic complexity improvements (O(n²) → O(n log n)) from constant-factor speedups or test-case-specific optimizations.

- **Code diff case studies** showing what optimizations the fine-tuned model learns (e.g., replacing nested loops with hash maps, switching algorithmic strategies) would elevate the paper from a resource contribution to a mechanistic study of efficiency learning.

- **Quantify dataset size per teacher model** in Table 5 to explain whether the GPT-4o/Claude pass@1 gap is driven by fewer surviving training samples after filtering.

- **Analysis of what fraction of test cases or tasks were rejected at each filtering step**, particularly Steps 3 and 6, to characterize pipeline yield and understand where SOAP fails.

- **Inference-time efficiency prompting as a baseline** (asking the model to generate efficient code with an explicit prompt, without fine-tuning) would clarify how much of the gain requires fine-tuning vs. instruction-following.

---

## Removed Points
*These points were flagged for removal; treat them with caution.*

- **C11 – PIE comparison is "unfair" (Harsh Critic):** The paper explicitly fine-tunes CodeLlama-7b-hf on both PIE and EFFI-CODE for a direct backbone-matched comparison (Table 9). PIE's training data is in C++ (competitive programming), which could be viewed as a disadvantage for Python evaluation — meaning the comparison already favors PIE's generalization. EFFI-CODE still outperforms. This is an intentionally asymmetric comparison that benefits the baseline, per the policy of removing such complaints.

- **"Narrow benchmark set" limitation (Harsh Critic, framed as missing):** Evaluating additional benchmarks like MBPP or LiveCodeBench would strengthen the paper, but calling the absence of these a *weakness* is scope creep given that HumanEval and EffiBench are standard benchmarks for this exact problem setting.

- **Requesting confidence intervals or multiple-run statistics for large-scale benchmarks:** The paper already addresses randomness in Table 8 by running 5 evaluations for the main model and reporting negligible variance. Demanding confidence intervals across all table entries imposes a non-standard requirement for this community.

- **Societal impact of code efficiency optimization (negative):** Criticism that efficient code could enable high-frequency trading or surveillance is highly speculative and not a scientific weakness of the paper.

---

## Novel Insights
The most genuinely striking observation — not adequately discussed by any of the three reviewers — is the **inversion of teacher model capability vs. student pass@1**: GPT-4o generates objectively faster code than DeepSeek-Coder-V2-Lite in Table 5 (ET 0.20s vs. 0.21s), yet the student fine-tuned on GPT-4o-generated data achieves only 39% pass@1 while the DeepSeek-V2-Lite-trained student achieves 59.8%. This suggests that *efficiency of the teacher's code* and *quality of the resulting fine-tuning distribution for a student model* may be orthogonal properties — and that distributional alignment between teacher and student may matter more than absolute teacher quality. If this holds, it has implications beyond this paper for data selection in knowledge distillation for code. This insight deserves explicit investigation and could be a more impactful contribution than the dataset itself.

---

## Suggestions

1. **Run a decontamination check** between all 8 source datasets and HumanEval/EffiBench test prompts using n-gram or embedding similarity. Report exact overlap rates. If contamination is found, re-run pass@1 evaluations on decontaminated training splits. This is the single most important action to restore credibility to the correctness claims.

2. **Investigate and explain the source of pass@1 gains.** Conduct a controlled experiment: fine-tune on an efficiency-matched dataset of equal size but drawn from tasks with no conceivable HumanEval overlap, and measure pass@1 on HumanEval. If pass@1 still improves dramatically, the effect is real; if it doesn't, the gains are likely contamination-driven.

3. **Report absolute overlap set sizes** (not just percentages) in all tables, and add a sentence in each table caption noting that efficiency metrics are computed over these N problems. This directly addresses reproducibility concerns and helps readers weight results appropriately.

4. **Explain the teacher model pass@1 discrepancy in Table 5** by reporting: (a) how many tasks survived Step 6 filtering per teacher, and (b) the distribution of task types and difficulties in each teacher's surviving set. This will clarify whether the effect is data-quantity-driven or distribution-driven.

5. **Address the nonmonotonic Table 3 behavior** by reporting which specific tasks move in/out of the overlap set at each dataset proportion, or alternatively by reporting efficiency on a fixed held-out overlap set defined by the 100% model. This will reveal whether the cliff is a measurement artifact or a genuine learning phenomenon.

---

**Evaluation Summary:**
- **Novelty:** Moderate. The efficiency-focused fine-tuning direction is underexplored and the dataset construction pipeline is thoughtful, but the core optimization method (SOAP) is adopted wholesale from prior work. The contribution is primarily empirical and resource-oriented.
- **Technical soundness:** Below average for ICLR. The contamination gap, small overlap-set reliability issue, and unexplained correctness anomalies collectively undermine confidence in the key claims.
- **Empirical support:** Mixed. The experimental coverage (models, sizes, fine-tuning methods) is commendable, but the reliability of the most prominent results (base model efficiency gains, EffiBench efficiency with 1% overlap) is questionable.
- **Significance:** Moderate-to-high as a problem. Generating efficient code is a real gap. Whether this paper credibly closes that gap depends on resolving the contamination and mechanism questions.
- **Clarity:** Adequate overall, with specific lapses in Table 3 interpretation and unexplained baseline inconsistencies across tables.

# Actual Human Scores
Individual reviewer scores: [1.0, 6.0, 3.0, 6.0]
Average score: 4.0
Binary outcome: Reject
