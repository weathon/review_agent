=== CALIBRATION EXAMPLE 55 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title is descriptive and the term "quasi-dynamic" is catchy, though upon reading the paper it becomes clear the method is still entirely *static* at inference time — no actual program execution occurs. The "dynamic" aspect is that optimization passes are *applied* to obtain new IRs whose static features are then measured; everything is deterministic and involves no runtime profiling. The title's characterization as "quasi-dynamic" risks misleading readers who associate "dynamic" with actual execution. The abstract's claim that the framework "transcends the trade-off" between static and dynamic representations somewhat oversells the contribution given this subtlety.

A more accurate framing would be: the method computes a static representation that is *richer* than conventional static representations by capturing optimization sensitivity, but still avoids actual execution.

---

### Introduction & Motivation

The static/dynamic dilemma is well-articulated and the motivation is genuine. The gap the paper identifies — static representations describe program *structure* but not *optimization sensitivity* — is real and compelling. The three stated contributions are clear and mostly honest about scope.

One concern: the introduction frames the approach as generating representations that "offer profound insights into a program's true performance bottlenecks," which is stronger than what the experiments actually demonstrate (two specific compiler optimization proxy tasks with simplified metrics). This is not fatal, but the claims should be calibrated.

---

### Method

**2.1.1 — Probe Construction:**
The probe-generation procedure is described at a high level but critically omits the value of **P** (the number of probes/clusters). This is the single most important hyperparameter of the entire framework and is never disclosed in the main paper. The final Behavioral Spectrum is a matrix **S**_p ∈ R^{P×56}$; without knowing P, it is impossible to assess the dimensionality, computational cost, or redundancy of the representation. The paper must specify P and include ablations over it.

Additionally, probe construction uses a genetic algorithm/greedy strategy whose stochastic nature raises reproducibility concerns. How many independent runs were conducted? What variance is there in the resulting probes? Are the probes fixed across all experiments or re-computed per trial?

**2.1.2 — Scale-invariant Reaction Quantification:**
Equation 1 defines the reaction as:

```
d_{i,j} = log(1 + max(0, h_opt,i,j)) - log(1 + max(0, h_orig,i,j))
```

The use of `max(0, ·)` discards negative feature values entirely. The paper offers no explanation for why Autophase features — which are counts of program elements (instructions, basic blocks, etc.) — could ever be negative, nor what the appropriate handling of such edge cases should be. If negative values arise from "feature extraction artifacts," they suggest data quality issues that deserve discussion rather than silent clamping.

More fundamentally, this formula is a difference of log-transformed values, not a true log-ratio of changes: it equals log((1+h_opt)/(1+h_orig)) only when both are non-negative. For a feature that was zero before optimization and becomes non-zero after (a common case), the reaction is log(1+h_opt), which is purely a function of the new value, not a *change*. The scale-invariance argument is valid in spirit but the formula does not strictly deliver it in all cases.

**2.2 — Product Quantization:**
The PQ configuration (M=8 sub-vectors, k*=256 centroids) is presented as a fixed choice with no ablation. The virtual vocabulary of size 256^8 is astronomically large — far exceeding the pre-training corpus of 220,000 programs — implying extreme sparsity in practice. The paper should quantify the codebook utilization rate and the average reconstruction error of the PQ encoding. Without this, the claim that PQ "retains fine-grained structural information with minimal loss" is unsupported.

**Deeper conceptual concern — circularity with Autophase:**
The Behavioral Spectrum measures *changes in Autophase features* under optimization probes. Since Autophase is a 56-dimensional count-based summary of IR structure, the entire representation is fundamentally bounded by what Autophase can express. The approach is, at its core, learning a mapping from Autophase-under-transformations to downstream tasks. Any semantic information not captured by Autophase — including control-flow structure, data dependencies, or memory access patterns — is invisible to the method. The authors acknowledge this briefly in the Limitations section, but its significance is undersold. Compared to ProGraML (which operates on full program graphs) or IR2Vec (which embeds IR operations with type information), the input information content of the Behavioral Spectrum is quite limited even if its representation is novel.

---

### Experiments & Results

**Validation-Test Generalization Gap:**
The gap between validation and test performance is large and concerning: for Best Pass Top-5, 99.08% validation vs. 89.55% test; for -Oz Prediction, 2.22% validation MAE vs. 8.19% test MAE. While the paper explains this as expected in-distribution vs. out-of-distribution behavior, a 3.7× gap in MAE for the regression task warrants deeper investigation. Does the probe set — constructed from the pre-training corpus — generalize to the structural diversity of curated benchmarks? The test set programs from cbench, mibench, chstone, and npb represent very different computational patterns than competitive programming solutions, and this domain shift deserves explicit analysis.

**Test Set Size and Statistical Significance:**
The total test set is 184 programs (cbench=11, mibench=40, chstone=12, npb=121). At this scale, a difference of 4–5 programs in predictions corresponds to roughly 2–3 percentage points in Top-1 accuracy. No error bars, confidence intervals, or statistical significance tests are reported anywhere in the paper. Given the small test sets, it is impossible to assess whether the improvements over baselines (particularly those near 4–5 percentage points, such as the K-NN accuracy of 79.70% vs. InstCount's 75.82%) are statistically meaningful. This is a serious methodological gap for an ICLR submission.

**Ablation Results Contradict Core Claims (Table 2):**
The ablation study produces a result that undermines one of the paper's key technical contributions. For Best Pass Top-5 accuracy on the *test set*, the **No-Relative** variant achieves **94.33%**, while the full Behavioral-PQ model achieves only **89.55%** — a 4.78 percentage point *disadvantage* for the full model. The paper states "our full model remains competitive on the test set" but it is actually worse. The explanation offered ("coarser-grained signals can provide reasonably strong predictive power") does not resolve why the full model is *inferior*. This is a failure of the ablation study to support the paper's narrative, and no additional analysis is provided to understand why scale-invariant quantification hurts this task.

**Missing Baselines:**
The baselines are predominantly from 2018–2021. Given that the paper is submitted to ICLR 2026, several relevant recent approaches are absent:
- LLM-based compiler optimization (cited in related work: Cummins et al., 2023) is not included as a baseline.
- The paper cites CompilerDream (Deng et al., 2025) in the introduction but does not compare against it.
- There is no comparison with end-to-end learned optimization approaches that jointly represent and optimize.

**inst2vec Treatment:**
inst2vec uses an LSTM encoder while all other baselines use a two-layer MLP. The paper acknowledges this inconsistency but justifies it as necessary because inst2vec provides instruction-level rather than program-level embeddings. However, this means inst2vec is disadvantaged in the MLP setting (where it needs to aggregate before the downstream head) and advantaged in the LSTM setting (where the LSTM can integrate information across the sequence). The comparison is not controlled.

**ProGraML Results:**
ProGraML is included as a baseline in the K-NN table (Table 3, 70.75%) but does not appear to be prominently shown in Figure 3 or the main text results for the two primary tasks. Its performance should be shown consistently.

---

### Writing & Clarity

Section 4.2 contains apparent parser artifacts ("methodscale-invariant", "methoddeep contextual reasoning...method") that appear to be formatting issues as noted in the instructions. Setting these aside, the paper is generally clearly written. However, two substantive clarity issues:

1. **P is never defined.** This is not a formatting artifact — P is the most important design parameter of the method and is simply absent from the paper.
2. The claim in Section 2.2.2 that the original vector can be reconstructed as $\hat{d} = [c_{1,c_1}, c_{2,c_2}, ..., c_{8,c_8}]$ uses confusing notation (subscript conflicts between cluster index and centroid lookup).

---

### Limitations & Broader Impact

The limitations section honestly acknowledges probe diversity and preprocessing overhead. However, two important failure modes are not addressed:

1. **Programs with degenerate Autophase profiles**: Programs where all Autophase features are near-zero or near-identical across probes will produce near-zero behavioral spectra, making them effectively indistinguishable from each other. How common are such degenerate cases in the test set?

2. **Sensitivity to probe quality**: If the genetic algorithm converges to poor or redundant sequences for a cluster, the corresponding probe carries no discriminative information. There is no quality filter or validation of the probes' discriminative power.

The broader impact statement is adequate for the paper's scope.

---

### Overall Assessment

This paper addresses a genuine problem — program representations that capture optimization sensitivity rather than just structural features — and the core idea of probing programs with optimization sequences and measuring Autophase changes is creative and technically novel. However, the paper has several significant weaknesses that must be addressed before acceptance. The most critical issues are: (1) P, the fundamental hyperparameter of the method, is never stated; (2) the ablation study (Table 2) directly contradicts the paper's narrative — the full model is worse than the No-Relative ablation on the primary task's test set with no satisfying explanation; (3) the test sets are too small (184 programs total) for the reported differences to be statistically meaningful without significance testing; and (4) the representation is fundamentally bounded by Autophase's 56 features, a limitation the paper downplays. The validation-test generalization gap (especially 2.22% vs. 8.19% MAE) also warrants deeper investigation. The paper has the seeds of a good contribution but requires significant methodological clarification, additional ablations (particularly over P, M, and probe generation), and statistical rigor before it meets the ICLR acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a **quasi-dynamic program representation** framework called Behavioral-PQ, which captures a program's optimization sensitivity by probing its IR with diverse optimization sequences to generate a "Behavioral Spectrum." The authors employ **Product Quantization** to discretize these behavior vectors and pre-train a multi-task **Transformer (PQ-BERT)** to learn their contextual grammar. Extensive experiments on CompilerGym benchmarks demonstrate that this method significantly outperforms state-of-the-art static and dynamic baselines on compiler optimization prediction tasks.

### Strengths
1.  **Superior Empirical Performance:** The method achieves state-of-the-art results on key downstream tasks. Specifically, it reaches **64.48% Top-1 accuracy** on Best Pass Prediction (surpassing the strong baseline `inst2vec` by ~25 percentage points) and achieves the lowest error (**8.19% MAE**) on -Oz Benefit Prediction compared to all baselines (Figure 3, Section 4.1).
2.  **Comprehensive Evaluation Strategy:** The authors rigorously assess out-of-domain generalization by using uncurated benchmarks for training and curated suites for testing (Section 3). The evaluation extends beyond instruction counts to include runtime prediction, cycle reduction (Appendix C.1), and comparison against hardware profiling baselines (Table 9), validating the utility of the "quasi-dynamic" approach.
3.  **Reproducibility and Open Science:** The paper provides a public repository with code, pre-trained models, and detailed configuration scripts (Reproducibility section, Section 5.1). The inclusion of Appendix D regarding LLM usage also adheres to transparency standards in modern ML submissions.

### Weaknesses
1.  **Inconsistency in Classification Performance:** The ablation study reveals a potential over-engineering of the full model for classification tasks. As noted in Section 4.2 and Table 2, the `No-Relative` variant (without scale-invariant quantification) achieves higher Test Set Top-5 accuracy for Best Pass Prediction **(94.33%)** compared to the full PQ-BERT model **(89.55%)**, yet the authors highlight the full model as the primary contribution. This suggests the complexity of PQ and the Transformer may not always yield gains for classification tasks.
2.  **Unclear Cost-Benefit Analysis of "Quasi-Dynamic" Probing:** While the authors claim preprocessing overhead is low (~0.2s per program in Section 5.1), they do not provide a concrete cost-benefit analysis comparing this static probing effort against the runtime overhead of the dynamic HPC baseline (Table 9). Quantifying the "prohibitive overhead" of dynamic profiling relative to this fixed preprocessing cost would better justify the efficiency claims.
3.  **Domain Gap in Pre-training vs. Downstream Tasks:** Pre-training is conducted on 220k OJ programming contest solutions (Section 3), whereas downstream evaluation is on CompilerGym workloads (e.g., `linux-v0`, `mibench`). While the authors argue this ensures strict out-of-domain evaluation, the paper lacks a deep analysis of whether the "optimization sensitivity" learned from OJ data transfers effectively to embedded or general-purpose benchmark suites compared to models pre-trained on IR-based codebases.

### Novelty & Significance
*   **Novelty:** The core novelty lies in the **quasi-dynamic** paradigm that bridges static efficiency and dynamic profiling by encoding program reactions to optimization probes. The application of **Product Quantization** to construct a behavioral vocabulary for Transformer pre-training is a unique contribution to program representation learning.
*   **Significance:** The work addresses a fundamental bottleneck in machine learning-assisted compiler optimization: the lack of representations that capture "optimization sensitivity." By demonstrating significant improvements in pass selection and benefit prediction, it offers a practical pathway to more effective auto-tuning systems.
*   **Clarity:** Despite minor OCR artifacts in the text (e.g., broken equation delimiters), the paper's logical flow and methodology description are clear and well-structured.
*   **Reproducibility:** The public release of code and adherence to a clear experimental protocol (including anonymous repository for review) support high reproducibility standards.

### Suggestions for Improvement
1.  **Clarify Method Selection Rationale:** The authors should explicitly discuss why the full PQ-BERT model is preferred despite the `No-Relative` variant outperforming it on the Best Pass Test set (Table 2). A discussion on the trade-off between model complexity and the specific requirements of the -Oz regression task would strengthen the design justification.
2.  **Quantify Overhead Comparison:** To substantiate the claim that dynamic profiling is "impractical," include a comparative table comparing the wall-clock time and energy consumption of the proposed probing method versus the dynamic HPC baseline execution, not just the accuracy comparison.
3.  **Analyze Pre-training Data Bias:** Discuss in Section 5.1 why OJ data (algorithmic, competitive programming) represents the general optimization behavior of embedded/compiler benchmarks (e.g., MIBench). A small ablation or sensitivity analysis on the composition of the pre-training data would strengthen claims of generalization.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Dynamic Baselines on Main Tasks:** Compare hardware performance counter (HPC) features directly on the Best Pass and -Oz tasks, not just Runtime Prediction (Appendix). Without this, the claim that quasi-dynamic methods surpass the static/dynamic trade-off for optimization prediction is unsupported.
2. **Probe Objective Decoupling:** Evaluate using probes optimized for runtime on the instruction count task (and vice versa). Current probes are generated using instruction count reduction, matching the evaluation metric, which risks target leakage rather than learning general behavior.
3. **Inference Latency vs. Static Baselines:** Report embedding generation time across varying program sizes compared to static feature extraction (e.g., Autophase). The claimed 0.2s overhead must be justified against static methods (typically <10ms) to validate the "practicality" contribution.

### Deeper Analysis Needed (top 3-5 only)
1. **Full Model vs. Ablation Generalization:** Explain why the Full model underperforms the `No-Relative` and `KMeans` variants on the Best Pass Test set (89.55% vs 94.33%/93.43%). This suggests the proposed complexity may harm generalization on classification tasks, undermining the necessity of PQ-BERT.
2. **Probe Coverage and Sparsity:** Quantify the percentage of test programs that exhibit negligible reaction to all $P$ probes. If a significant portion of the spectrum is zero/near-zero, the behavioral signal is sparse, questioning the embedding's richness for those cases.
3. **Total Compute Cost Quantification:** Provide total GPU-hours and compilation-hours for pre-training the 220k corpus. The environmental impact statement is vague; ICLR requires precise compute reporting to assess the sustainability of running $P$ optimizations per program.

### Visualizations & Case Studies
1. **Failure Case Analysis:** Provide a case study where Behavioral-PQ predicts incorrectly despite strong probe reactions. The current Appendix only shows a success case, hiding failure modes where the spectrum misleads the model.
2. **Probe Redundancy Heatmap:** Visualize the correlation matrix between the $P$ probe reaction vectors. High correlation would indicate wasted computation and reduced spectral diversity, weakening the "diverse set of optimization sequences" claim.
3. **Reaction Magnitude Distribution:** Plot the distribution of log-relative differences across the dataset. If most values cluster near zero, the "behavioral" signal is weak, suggesting the method relies on static features remaining after log-transformation.

### Obvious Next Steps
1. **Adaptive Probe Selection:** Implement a mechanism to select a subset of probes per program rather than running all $P$. This is necessary to reduce the inference overhead identified in the experiments for real-world deployment.
2. **Cross-Metric Generalization:** Validate embeddings on tasks unrelated to instruction count (e.g., power consumption, binary size). This would prove the spectrum captures general behavior rather than just sensitivity to the probe generation metric.
3. **Hardware-Specific Probing:** Extend probes to include target-architecture specific flags (e.g., `-march=native`). Current LLVM IR probes are hardware-agnostic, limiting relevance for final machine code performance prediction.

# Final Consolidated Review
## Summary

This paper introduces Behavioral-PQ, a "quasi-dynamic" program representation for compiler optimization tasks. Rather than using purely static features or costly dynamic profiling, the authors generate a Behavioral Spectrum by applying P optimization probe sequences to each program's LLVM IR and measuring the resulting changes in Autophase features using a scale-invariant logarithmic relative difference. Product Quantization discretizes these continuous reaction vectors into compositional sub-words, and a multi-task Transformer (PQ-BERT) is pre-trained to learn contextual relationships. Experiments on Best Pass Prediction (124-class classification) and -Oz Benefit Prediction (regression) demonstrate substantial improvements over static baselines (25+ percentage point Top-1 accuracy gain vs. inst2vec).

## Strengths

- **Novel methodological contribution**: The quasi-dynamic paradigm—probing programs with optimization sequences and measuring reactions—genuinely addresses the gap between static representations (efficient but myopic) and dynamic profiling (insightful but costly). This is a creative alternative to both extremes.
- **Strong empirical results with substantial margins**: Behavioral-PQ achieves 64.48% Top-1 accuracy on Best Pass Prediction versus 39.27% for inst2vec (+25 points) and 8.19% MAE on -Oz Benefit Prediction versus 16.23% for inst2vec (≈2× better). The improvements are large enough to be practically meaningful for compiler auto-tuning.
- **Comprehensive evaluation across multiple dimensions**: Beyond the main tasks, the paper evaluates runtime prediction (Table 7), cycle reduction (Table 8), device mapping (Table 10), and comparison against dynamic HPC features (Table 9). The consistent superiority across tasks strengthens the claim that behavioral embeddings capture generalizable program semantics.
- **Transparent methodology with public code**: The reproducibility statement and public repository support independent verification. The comparison with zero-shot LLMs (Table 6) contextualizes the contribution against modern alternatives, showing specialized models still substantially outperform general-purpose LLMs.

## Weaknesses

- **Ablation contradicts primary claims for classification**: Table 2 shows that for Best Pass Top-5 accuracy on the test set, the No-Relative variant achieves 94.33% while the full Behavioral-PQ achieves only 89.55%. The full model is *worse* by nearly 5 percentage points. For -Oz regression, the full model is indeed superior (8.19% vs. 10.96%), but the paper's narrative that the full architecture is uniformly better is contradicted by the classification results. The authors must explain this discrepancy—currently they state the full model "remains competitive" when it is actually inferior on this key metric.
- **Fundamental hyperparameter P is not specified**: The number of probes P is the most important design parameter (determining representation dimensionality and preprocessing cost) but is never stated in the methodology. Only by reading Appendix A can one infer P=100. This omission makes the method's computational requirements opaque and prevents readers from understanding the representation's complexity.
- **Representation bounded by Autophase's expressive capacity**: The Behavioral Spectrum measures changes in 56 Autophase features. While the transformation through probes and PQ-BERT encoding is non-trivial, the method fundamentally cannot capture information that Autophase misses—e.g., detailed control-flow structure, data dependencies, or memory access patterns. Compared to graph-based methods like ProGraML that operate on full program graphs, this is an inherent limitation that the paper acknowledges but undersells.
- **Probe optimization objective matches evaluation metric**: Probes are constructed by maximizing instruction count reduction, and the evaluation tasks measure instruction count changes (Best Pass for reduction, -Oz benefit). This risks target leakage rather than demonstrating that the approach learns general optimization sensitivity. The method may simply be memorizing that certain probe reactions correlate with instruction reduction rather than capturing transferable behavioral patterns.
- **Validation-test generalization gap warrants investigation**: The 3.7× MAE gap for -Oz prediction (2.22% validation vs. 8.19% test) is substantial. While the authors attribute this to out-of-domain evaluation, understanding *which* test programs perform poorly and *why* would strengthen claims about generalization. The probe set constructed from competitive programming solutions may not transfer well to embedded/systems benchmarks.

## Nice-to-Haves

- **Cost-benefit analysis vs. dynamic profiling**: The paper claims 0.2s preprocessing overhead per program. A direct comparison of wall-clock time (including compilation overhead) against the 28-dimensional HPC baseline would strengthen practicality claims. Static Autophase extraction typically takes <10ms; understanding when the quasi-dynamic cost is justified matters for deployment.
- **Confidence intervals or statistical significance tests**: With 184 test programs and ~5 percentage point differences between methods in some comparisons, reporting uncertainty would help readers assess result robustness. This is increasingly expected in ML for systems work.
- **Cross-metric probe evaluation**: Evaluating probes optimized for one metric (instruction count) on unrelated tasks (runtime, power, binary size) would test whether the representation captures general optimization sensitivity or just metric-specific patterns.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Quasi-dynamic" framing is misleading**: The paper clearly explains (Section 1) that no runtime execution occurs—the "dynamic" aspect is that optimization passes are applied to IR. The term captures this appropriately; the criticism is overly pedantic about terminology.
- **LLMs not compared as baselines**: The paper *does* compare against GPT-5-mini, GLM-4.5, DeepSeek-V3, and others in Appendix B (Table 6), showing substantial advantages for Behavioral-PQ. The comparison is provided.
- **ProGraML missing from main results**: ProGraML appears in K-NN results (Table 3) and is included throughout the evaluation. While it could be more prominently featured, it is not omitted.
- **inst2vec uses different architecture**: The authors acknowledge this and explain their rationale. While imperfect, they provide a reasonable justification (instruction-level vs. program-level embeddings). This is noted but not a methodological flaw warranting rejection.

## Novel Insights

The key insight that behavioral reactions to optimization probes encode richer information than static structure alone is genuinely valuable. The paper demonstrates that *how a program changes* under transformations carries predictive signal that *what a program is* misses. The Product Quantization approach to discretizing continuous reaction vectors into compositional codes is an elegant solution to the vocabulary construction problem—allowing a virtual vocabulary of 256^8 types from only 8×256 learned centroids. However, the finding that simpler variants (No-Relative, KMeans) sometimes outperform the full model on classification suggests the PQ-BERT complexity may not always be necessary, pointing toward future work on task-adaptive representations.

## Suggestions

- **Add a sentence in Section 2.1.1**: "In our experiments, we use P=100 probes generated through the clustering and optimization procedure described above."
- **Revise Section 4.2 discussion**: Explain why scale-invariant quantification helps regression but harms classification generalization. One hypothesis: absolute differences preserve magnitude information useful for distinguishing instruction reduction levels, while the log-ratio normalization removes this signal.
- **Add a "Limitations" paragraph on Autophase bounds**: Explicitly discuss what information Autophase cannot capture and acknowledge that graph-based methods like ProGraML have access to structural information that Behavioral-PQ fundamentally cannot see.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 4.0]
Average score: 4.7
Binary outcome: Accept
