## Summary
EFFI-CODE introduces a dataset construction pipeline that uses Self-Optimization based on Overhead Profiling (SOAP) to generate efficiency-optimized Python code solutions for fine-tuning LLMs. The pipeline aggregates eight open-source datasets, filters to algorithmic tasks, iteratively refines solutions using DeepSeek-Coder-V2-Lite as a teacher, and produces 9,451 training tasks with verified efficiency improvements. Fine-tuning on this dataset yields gains in both pass@1 and execution time (e.g., DeepSeek-Coder-6.7B-Instruct pass@1 rises from 43.3% to 76.8% on HumanEval with 30.5% ET reduction), and the authors will open-source data, code, and model weights.

---

## Strengths

- **Canonical-solution ablation demonstrates efficiency-specific training signal (Table 6).** Fine-tuning on the original (non-optimized) canonical solutions improves pass@1 but *increases or barely changes* execution time, while EFFI-CODE fine-tuning reduces ET by ~40%. This controlled comparison directly validates that the optimized targets—not just additional supervised coding data—drive the efficiency gains, making this the strongest piece of evidence for the paper's central claim.

- **Novel application of profile-guided optimization to dataset construction.** Using line-level `line_profiler`/`memory_profiler` traces to condition teacher-model rewrites, and using those rewrites as fine-tuning targets, is a creative and mechanistically motivated approach to closing the efficiency gap in code LLMs—an axis largely ignored by prior instruction-tuning datasets.

- **Breadth of empirical evaluation relative to paper scope.** The paper tests model sizes from 1.3B to 33B within the DeepSeek family, includes a second architecture family (Qwen2.5-Coder), evaluates SFT, DPO, and ORPO fine-tuning methods, and assesses on two distinct Python code benchmarks (HumanEval and EffiBench). For a dataset/framework paper, this coverage is solid.

- **Teacher model comparison reveals an informative tradeoff (Table 5).** GPT-4o and Claude-3.5-Sonnet as teachers produce marginally better raw efficiency on overlap tasks, but the open-source DeepSeek-Coder-V2-Lite teacher yields dramatically higher pass@1 after fine-tuning. This finding—that distribution match between teacher and student matters more than absolute teacher quality for SFT—is a practically significant and non-obvious result.

---

## Weaknesses

### Fatal
*No fatal flaws that fully invalidate the contribution, but the two "Major" issues below, if unresolved, would substantially undermine the quantitative claims.*

### Major

- **No decontamination against evaluation benchmarks.** Footnote 2 states: *"Data decontamination was not included in the filtering process as most of the tasks we collected have been decontaminated, such as OSS-Instruct."* This is inadequate for the aggregated final dataset evaluated on HumanEval. Sources such as CodeFeedback, Alpaca, and APPS are known to contain HumanEval-adjacent problems. The pass@1 jump from 43.3% to 76.8% on HumanEval—a 33-point absolute gain—cannot be confidently attributed to generalization without ruling out benchmark contamination. This is the most critical validity concern: if the gain is partially explained by memorization of benchmark problems, the core claim that EFFI-CODE teaches efficient *generalizable* code generation is weakened. A rigorous n-gram or AST-based decontamination check against HumanEval and EffiBench problem statements must be reported.

- **Efficiency is evaluated only on the overlap subset, which can be vanishingly small.** The "Overlap" column in Table 2 reveals that in the worst cases—e.g., DeepSeek-Coder-6.7B-Instruct on EffiBench at 1.0% overlap—efficiency is measured on roughly one task. For Qwen2.5-Coder-7B-Instruct on EffiBench (3.2% overlap), similarly few tasks contribute. Efficiency metrics computed over such tiny, non-random subsets (tasks solved correctly by *both* the original and the fine-tuned model) are statistically unreliable and potentially subject to task-composition bias: the overlap set might consist of easy, fast-to-execute tasks regardless of fine-tuning. The paper does not report efficiency separately for (a) tasks only the fine-tuned model solves or (b) all tasks, making it impossible to judge whether ET improvements on overlap tasks reflect genuine speed-up or selection artifact. This concern is most acute for EffiBench instruct-model rows, where the unusually low baselines (1.3% and 3.3% pass@1) and tiny overlap undermine interpretability.

- **Unusually low baselines for instruct models on EffiBench point to prompt template mismatch.** The paper states (Section 4): *"For all experiments, we use the inference prompt provided by DeepSeek-Coder for both fine-tuning and inference."* DeepSeek-Coder-6.7B-Instruct achieves only 1.3% pass@1 on EffiBench before fine-tuning—an implausibly poor score for a modern 6.7B code model. Similarly, Qwen2.5-Coder-7B-Instruct achieves only 3.3%. These numbers likely reflect a mismatch between the DeepSeek-Coder chat template and the expected input format for Qwen models. Since the fine-tuned models learn to use the DeepSeek-Coder template, they gain a prompt-format advantage not present in the baseline. This inflates the apparent correctness improvements for non-DeepSeek models on EffiBench. Evaluations for each model family should use their native prompt template, or a template-matched evaluation should be used uniformly.

### Minor

- **Memory improvements are negligible throughout.** Across all tables, MU and NMU changes are consistently 0.0%–0.1%, and NMU never improves meaningfully (e.g., NMU stays at 0.99–1.00 in Table 2 HumanEval rows). The paper claims to improve "both execution time and memory usage," but the evidence supports only execution time improvement. The paper should honestly discuss why SOAP-trained models do not generalize to memory efficiency, and either temper memory claims or propose a mechanism.

- **Dataset size ablation shows anomalous efficiency behavior.** In Table 3, for DeepSeek-Coder-6.7B-Base, NET barely changes from 25%–75% (2.11→1.96→2.03→1.93) and then drops sharply at 100% (1.13). Similarly for the Instruct model, ET is flat from 25%–75% (0.43→0.41→0.42) then drops to 0.24 at 100%. This cliff-edge pattern suggests efficiency learning is threshold-dependent in an unexplained way, or that an outlier in the final 25% of training data dominates. The paper should characterize this non-monotonicity rather than treat the 100% result as confirming a smooth scaling trend.

- **"Open-source only" claim is misleading as applied to the actual pipeline.** The contribution bullet states the framework "can be implemented only using open-sourced LLMs," explicitly contrasting with GPT-4. However, the actual pipeline uses GPT-3.5-turbo for three critical steps: risk filtering (Step 2), test case generation (Step 3), and algorithmic/non-algorithmic classification (Step 4)—together affecting all ~780K source tasks. The claim should be reworded to clarify that the *optimization teacher* can be open-source, while the preprocessing steps currently rely on a closed model that could in principle be replaced.

- **Survivorship bias from Step 6 filtering is acknowledged but not analyzed.** The paper removes tasks where SOAP fails to improve efficiency—correctly noting that "the initial code is already efficient" would cause false-positive removal. This selects training data toward teacher-success cases and may overrepresent easy-to-optimize patterns. The paper explicitly acknowledges this but offers no characterization of how many removed tasks had already-optimal initial solutions, nor any sensitivity analysis. The current argument—"which proved to still perform very well in our evaluation"—does not address the bias; it just reports aggregate outcomes.

- **Normalized metrics (NET, NMU, NTMU) lack formal definitions.** Section 3.5 describes these as "measuring how efficient/inefficient LLM-generated code is compared with the human-written canonical solution" but provides no equation. It is unclear whether they are arithmetic means of per-task ratios, ratios of totals, or something else. These are the primary evaluation metrics and require a precise definition.

- **"Generalizable" claim overstated relative to demonstrated scope.** The abstract calls EFFI-CODE "scalable and generalizable," and the contributions mention "different programming languages and domains." However, the pipeline exclusively targets Python (Step 1 filters to Python), profiling relies on Python-specific tools, and all benchmarks are Python-only. The cross-language generalizability is entirely aspirational. The abstract and contributions should scope this claim to Python, noting extension as future work.

### Tiny

- **Hardware specifications absent from efficiency evaluation.** Execution time and memory metrics are hardware-dependent. No CPU model, RAM configuration, OS, or Python version is reported, impeding reproduction of absolute efficiency numbers.

- **Table 5 teacher comparison lacks dataset-size controls.** Different teachers produce different numbers of post-SOAP surviving tasks (due to differing success rates at Step 5/6). The table does not report the size of each teacher's resulting dataset. If GPT-4o generates a smaller dataset than DeepSeek-V2-Lite after filtering, then the pass@1 comparison is confounded with data volume.

---

## Nice-to-Haves

- **Comparison with a standard general code SFT baseline (e.g., WizardCoder data or Code Alpaca).** Table 6 shows EFFI-CODE beats canonical solutions for the same tasks. But it does not show EFFI-CODE vs fine-tuning on a different general code dataset of similar size. Such a comparison would isolate how much of the improvement comes from the *efficiency orientation* of the data versus simply from having more supervised code fine-tuning.

- **Qualitative analysis of optimization types performed by SOAP.** Showing concrete examples of algorithmic complexity improvements (e.g., O(n²)→O(n log n) via dynamic programming, loop hoisting, data-structure substitution) would validate the causal story that SOAP teaches *efficiency* and not just cleaner or shorter code. The paper mentions Figure 3 exists, but a broader categorization of transformation types would strengthen the methodological case.

- **Per-task scatter plots for efficiency on overlap tasks.** Aggregate ET reductions could be driven by a few outlier tasks. A scatter plot of per-task ET (original vs. fine-tuned model, overlap tasks only) would show whether improvements are consistent or concentrated.

- **Broader model family coverage.** All size-scaling experiments (Table 4) use DeepSeek-Coder only. Including Qwen2.5-Coder at multiple sizes, or CodeLlama at multiple sizes, would better support the generalization claim.

- **Discussion of test case quality.** GPT-3.5-turbo-generated test cases are validated only against the (potentially noisy) initial solution from each source dataset, creating a circular validity check. Reporting test coverage statistics or a manual sample evaluation of test case quality on a subset of tasks would address this concern.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **Critique of PIE comparison as unfair or inconclusive** *(Harsh Critic, §4.2)*: PIE is a C++/competitive-programming edit dataset evaluated on Python benchmarks. The unfairness in this comparison is favorable to the baseline (PIE is penalized for the C++→Python domain shift), not to EFFI-CODE. This actually makes EFFI-CODE's outperformance a *stronger* demonstration, not a weaker one. The comparison is informative and should be kept; the criticism that it is "not fully fair or conclusive" is reversed in direction.

- **Request for formal proof that EFFI-CODE is orthogonal to synthetic techniques** *(Harsh Critic, §2)*: This is a reasonable forward-looking observation in the paper. Demanding empirical proof of orthogonality goes beyond the paper's stated scope. No published ICLR empirical paper is required to verify every claimed orthogonality.

- **Demand for license/redistribution analysis for source datasets** *(Harsh Critic, §3.1)*: Legitimate data-release hygiene, but a licensing audit is not an experimental or scientific weakness. This is an editorial/legal matter outside the scope of a technical review.

- **Criticism of not using per-example deduplication across the 8 source datasets** *(Harsh Critic, §3.1)*: The paper aggregates 8 well-known public datasets. While near-duplicate analysis across them would be thorough, the current dataset scale (9,451 tasks from 780k initial) and diversity of sources mean this concern—without evidence of actual duplication—is a speculative concern, not a demonstrated flaw.

- **Demanding detailed prompt text, temperature settings, and full decoding parameters for SOAP iterations** *(Harsh Critic, §3.3)*: These are reasonable reproduction details, but absent an appendix in the provided text, it is not confirmed these are missing—the paper references an Appendix A. This criticism may be unfounded.

- **Critique that DPO/ORPO don't isolate preference learning for efficiency** *(Harsh Critic, §4.2)*: Correct, but the paper frames these as auxiliary fine-tuning method comparisons, not as evidence for learning efficiency via preferences. The point is that EFFI-CODE data is also useful in preference setups—this is a reasonable and modest claim.

---

## Novel Insights

The most non-obvious insight in this paper is the teacher-model tradeoff revealed in Table 5: GPT-4o and Claude-3.5-Sonnet produce more efficient raw code than the open-source DeepSeek-Coder-V2-Lite, yet fine-tuning *student* models on data generated by the open-source teacher yields dramatically higher pass@1 (59.8% vs. 39.0% for base model; 76.8% vs. 9.8% for instruct model). This suggests that the pedagogical value of fine-tuning data depends critically on stylistic or structural alignment between teacher and student, not just on the quality of the teacher's outputs—a finding that is both practically important for dataset construction and theoretically interesting for distillation research. However, the paper treats this as a secondary observation rather than analyzing it mechanistically.

---

## Suggestions

1. **Run and report benchmark decontamination.** Apply n-gram deduplication (e.g., 13-gram overlap) between the full 9,451-task training set and both HumanEval and EffiBench problem statements. Report the contamination rate and, if tasks are removed, re-run the main evaluation to show whether the large pass@1 gains survive. This is the single most important action to take.

2. **Report efficiency on all tasks, not just overlap.** Add columns or a companion table reporting ET for (a) overlap tasks (current), (b) tasks solved only by the fine-tuned model, and (c) all tasks with confidence intervals. For cases where overlap is <5%, deprioritize or flag efficiency numbers as insufficiently reliable.

3. **Use model-native prompt templates for baselines.** Re-run Qwen2.5-Coder baselines using the Qwen-native chat template. If the 1–3% EffiBench baseline was indeed a prompt-format artifact, the corrected numbers will more accurately reflect the actual improvement from fine-tuning.

4. **Characterize and quantify the survivorship bias in Step 6.** Randomly sample 100–200 tasks removed at Step 6 and manually assess what fraction had already-efficient initial solutions vs. genuinely unoptimizable tasks. Report this fraction and discuss its implication for dataset representativeness.

5. **Add formal equations for normalized metrics.** Provide a precise mathematical definition of NET, NMU, and NTMU (including the averaging scheme and normalization baseline) in Section 3.5 or an appendix.

6. **Explain or investigate the dataset-size cliff.** Run the ablation at finer data-fraction granularity (e.g., 80%, 85%, 90%, 95%, 100%) for the instruct model and identify which tasks or task types enter at the 100% mark to explain the non-monotonic efficiency scaling.

---

## Evaluation Summary

| Axis | Assessment |
|---|---|
| **Originality** | Moderate. Applying SOAP to dataset construction for LLM fine-tuning is a genuinely new contribution. SOAP itself is prior work; the core novelty is the pipeline and the released dataset. |
| **Importance of research question** | High. Code efficiency is underexplored relative to correctness, and the sustainability/resource arguments are well-motivated. |
| **Claims well supported** | Partially. The canonical-vs-EFFI-CODE ablation is strong. The large pass@1 gains are unverified without decontamination, and efficiency improvements on tiny overlap sets are statistically fragile. |
| **Soundness of experiments** | Moderate. Good breadth across model sizes and fine-tuning methods; marred by the prompt-template confound for non-DeepSeek baselines and the overlap-measurement issue. |
| **Clarity of writing** | Adequate. The pipeline is readable, but key metrics lack formal definitions and the evaluation protocol requires the reader to infer critical details. |
| **Value to the research community** | Potentially high if dataset and models are clean and released as promised; the contamination question must be resolved first. |
| **Contextualization relative to prior work** | Adequate on instruction-tuning literature; thin on code optimization and execution-guided synthesis literature. |