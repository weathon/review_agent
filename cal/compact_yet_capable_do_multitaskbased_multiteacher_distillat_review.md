=== CALIBRATION EXAMPLE 25 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is overly long and reads like a summary of the method rather than a concise, memorable title. While descriptive, it is not ideal for a conference paper. The abstract clearly states the problem, approach, and claimed results. However, the claims are extremely strong (e.g., "near-perfect accuracy," "superior accuracy–efficiency trade-off") and set a high bar for the evidence required in the paper.

### Introduction & Motivation
The problem is well-motivated: efficient multitask NLU for low-resource Indic languages. The three contributions are stated clearly. However, the introduction lacks a clear statement of the key research question or hypothesis being tested (e.g., "Does task-specific dynamic PTQ outperform static PTQ when combined with multi-teacher KD?"). It also does not explicitly position the novelty against prior mixed-precision quantization works.

### Literature Survey
The survey covers relevant areas but is somewhat superficial. It correctly identifies gaps: prior work often treats weight and activation quantization separately, lacks per-task-head control, and misses a unified runtime policy. However, the claim that prior mixed-precision methods (e.g., HAQ, HAWQ) operate only at layer/block level and not at per-head granularity is accurate and sets up the claimed novelty. The connection to multi-teacher KD for multilingual tasks is appropriate.

### Methodology
This section has significant issues that affect reproducibility and soundness.

**Section 4.1 (Static Quantization):** The description is standard. However, the choice of N=100 calibration batches is justified by citing "seminal PTQ work," but no citation is provided. More critically, Equation (2) is garbled (parser artifact), but the surrounding text explains affine quantization adequately.

**Section 4.2 (Multi-teacher KD):** The loss function (Eq. 4) is presented, but the definitions of the individual loss terms (e.g., \(L_{KD}^{ID}\), \(L_{CRD}^{SF}\)) are not provided. What is contrastive learning used for? The attention-based fusion mechanism is mentioned but not described. How are the teacher outputs combined? The student model architecture is not specified—is it the same as the baseline XLM-R? If so, distillation is not reducing model size, only improving performance, which should be clarified.

**Section 4.2.3 (Precision-Controlled PTQ):** This is the core novel contribution but is **highly unclear and potentially contradictory**.
*   **Training vs. PTQ:** The method is described as a "post-training quantization" technique, yet the description involves a "learned precision controller" with "trainable logits" (Eq. 10) and Algorithm 1 shows backpropagation to update the controller and student model. This sounds like Quantization-Aware Training (QAT) or a neural architecture search, not PTQ. The fundamental premise of applying a *learned* controller *after* training is confusing.
*   **Controller Mechanism:** How is the controller implemented? Is it a small neural network? How is it trained? What is the objective (e.g., a latency-accuracy reward)? Algorithm 1 is too vague (e.g., "Compute sensitivity score based on weight and activation variance") and does not align with the Gumbel-softmax formulation in Eq. 10. The process for "freezing" bit-widths is mentioned but not detailed.
*   **Unified Policy:** The idea of a single policy governing both weight and activation precision is a claimed advantage, but the mechanics are not explained. At runtime, are activations quantized to the same bit-width as the corresponding weights? How is this implemented efficiently?
*   **Equation Issues:** Equations 7-9 are standard quantization formulas. However, Eq. 10 is presented as a method for choosing bit-widths but uses a temperature \(\tau\) and Gumbel noise \(g\), which suggests a training-time sampling strategy. This conflicts with the PTQ claim.

**Overall:** The methodology section fails to provide a clear, reproducible description of the proposed precision-controlled PTQ. The conflation of PTQ and QAT is a major conceptual flaw that must be resolved.

### Experiments & Results
The results are striking but raise serious questions.

**Experimental Setup:**
*   The baseline and student models appear to be the same XLM-R Base architecture. Therefore, "Only KD" has the same model size (1064 MB) as the baseline, which is not a compressed student. This contradicts the typical goal of KD to create a smaller model. The distillation seems aimed only at performance improvement.
*   Hyperparameters for distillation (e.g., loss weights \(\alpha, \beta, \gamma\)) are not provided.
*   The custom dataset is derived from MASSIVE but no details are given on how it was constructed for *multi-intent* tasks, as MASSIVE is a single-intent dataset. This is a critical omission for reproducibility.

**Results (Table 2):**
*   **Anomalous Accuracy Improvements:** Static PTQ on the baseline model *improves* Intent Accuracy from 0.9481 to 0.9947 and Slot F1 from 0.9674 to 0.9994. This is highly unusual; quantization typically causes a minor drop in accuracy. No explanation is given (e.g., quantization as regularization). This anomaly undermines confidence in the experimental setup or reporting.
*   **Model Size Inconsistencies:** The model size for "Baseline + Dynamic PTQ" (310 MB) is *larger* than for "Baseline + Static PTQ" (279 MB). Dynamic PTQ (often weight-only) should typically result in a larger model than static (weight+activation) if activations are stored in float32. However, the description in 4.2.2 suggests weight-only quantization, which should be ~270 MB (1/4 of 1064 MB). The reported 310 MB is unexplained. Similarly, the proposed method's size (428 MB) is larger than a full INT8 model (expected ~270 MB), suggesting the mixed-precision model is less compact than uniform INT8, which contradicts the efficiency claim. These numbers need verification and clarification.
*   **Near-Perfect Performance:** The final model achieves accuracy scores >0.999 on Intent and Slot tasks. This is suspiciously high and suggests the test set may not be challenging or there may be data contamination. The per-language results (Table 3) also show near-perfect scores, which is unexpected for low-resource languages.
*   **Statistical Significance (Table 4):** The p-values are provided, but it's unclear how many runs were used to compute the mean and standard deviation. The differences, while statistically significant for some tasks, are extremely small in absolute terms (e.g., Intent Accuracy change from 0.9947 to 0.9991).

**Figures:** References to figures (Fig. 4, 5, 6) are made, but without the figures, the analysis of trade-offs is hard to evaluate.

### Error Analysis
This section is good, providing meaningful insights into task sensitivity and language-specific challenges. It strengthens the paper by showing the authors have analyzed failure modes.

### Limitations
The stated limitation (no QAT) is minor. More critical limitations are not addressed: the unclear controller training, the potential overfitting or easiness of the dataset, the unexplained accuracy boost from quantization, and the fact that distillation does not yield a smaller architecture.

### Writing & Clarity
Leaving aside parser artifacts, the writing is generally clear. However, the methodological description of the precision controller is critically confusing, as noted. The paper would benefit from a clearer high-level flowchart of the entire pipeline.

## Overall Assessment
The paper addresses an important and timely problem: efficient multilingual multitask NLU. The idea of combining multi-teacher distillation with task-specific dynamic quantization is appealing. However, the current presentation has **severe flaws** that prevent acceptance at ICLR. The core contribution (precision-controlled PTQ) is not clearly described and seems to conflate PTQ with QAT. The experimental results contain unexplained anomalies, including accuracy *improvements* from static PTQ and inconsistent model size reporting. The near-perfect scores raise doubts about the dataset difficulty or experimental integrity. While the error analysis and per-language evaluation are positive aspects, the fundamental technical clarity and soundness of the proposed method are not demonstrated. Major revisions are required to clarify the methodology, justify the experimental results, and provide full reproducibility details before this work could be considered for publication.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses efficient deployment of multitask NLU (Intent Detection, Domain Classification, Slot Filling) for six low-resource Indic languages. The core contribution is a two-stage pipeline: first, a multi-teacher knowledge distillation framework trains a compact student model using three specialized teachers; second, a novel precision-controlled, task-specific dynamic post-training quantization (PTQ) scheme is applied. This PTQ method uses a learned controller to assign mixed bit-widths (4, 8, 16) independently to encoder components and task heads, unifying weight and activation quantization under a single policy. Experiments show significant reductions in model size and inference latency while maintaining high accuracy across tasks and languages.

### Strengths
1. **Comprehensive and Rigorous Evaluation**: The paper systematically compares seven distinct model configurations (baseline, baseline+static/dynamic PTQ, distilled, distilled+static/dynamic PTQ, and the proposed method). Results are reported for all three NLU tasks (accuracy/F1), model size, and inference time, providing a clear picture of the accuracy-efficiency trade-offs.
2. **Focus on Under-Studied Languages**: The work targets six low-resource Indic languages (Bengali, Hindi, Tamil, Telugu, Kannada, Malayalam), addressing a practical and socially valuable problem of making efficient NLU accessible in multilingual settings.
3. **Novel Integration of Techniques**: The combination of multi-teacher distillation (with adaptive attention fusion) and a precision-controlled, task-specific dynamic PTQ scheme is novel. The idea of a controller assigning per-component bit-widths for both weights and activations in a multitask model is a clear advance over uniform or layer-wise-only quantization.
4. **Strong Empirical Results**: The proposed method (KD + Precision-Controlled PTQ) achieves compelling efficiency gains: 59.8% reduction in model size and 67.1% reduction in inference time versus the FP32 baseline, while achieving near-perfect accuracy on Intent Detection (99.91%) and Slot Filling (99.72%), and a significant boost in Domain Classification (90.15% vs. 86.68% baseline). Statistical significance tests confirm improvements.
5. **Detailed Methodological Description**: The paper provides clear algorithmic descriptions and equations for static PTQ, dynamic PTQ, and the proposed precision-controlled PTQ, including the distillation loss formulation and the controller's Gumbel-softmax sampling (Eq. 10).

### Weaknesses
1. **Insufficient Detail on Controller Training and Objective**: While Algorithm 1 outlines the precision-controlled PTQ process, critical details are missing. The paper does not specify the controller's architecture, how the sensitivity score \(s_l\) is computed, the exact form of \(P(q|s_l, \alpha)\), or the optimization objective (e.g., a combined loss of task performance and bit-width cost). This lack of detail hinders reproducibility and understanding.
2. **Limited Comparison to State-of-the-Art Quantization Methods**: The baselines are standard static and dynamic PTQ. The paper does not compare against recent advanced PTQ methods like GPTQ, SmoothQuant, or mixed-precision techniques (e.g., HAWQ) which are highly relevant for transformer models. This omission makes it difficult to assess the true novelty and advantage of the proposed method within the current research landscape.
3. **Reproducibility Concerns**: The custom dataset, though based on MASSIVE, is not publicly available, and the exact construction process (e.g., how multi-intent examples were created) is not described. Code is not provided. Key implementation details are vague: e.g., the architecture of the attention-based fusion module for teachers, the specific linear layers quantized in "dynamic PTQ," and whether a CRF is used for slot filling in the distilled models.
4. **Superficial Analysis of Precision Assignments**: The paper presents the final efficiency results but does not analyze or discuss the actual bit-width assignments learned by the controller (e.g., which task heads or encoder blocks received 4-bit vs. 8-bit). This analysis is crucial for interpreting the method's behavior and validating its design.
5. **Incomplete Error Analysis and Limitations**: The error analysis is qualitative and brief. It identifies that Slot Filling is sensitive and Dravidian languages have more errors but lacks quantitative breakdowns (e.g., per-language error rates for each task under quantization). The stated limitation (no exploration of QAT or pruning) is cursory; a deeper discussion of the method's boundaries (e.g., scalability to more tasks/languages, sensitivity to controller hyperparameters) is needed.

### Novelty & Significance
The paper's novelty lies in the tailored integration of multi-teacher distillation for multitask NLU with a precision-controlled dynamic PTQ scheme that operates at the granularity of encoder blocks and individual task heads. Unifying weight and activation precision under a single learned policy is a distinct contribution compared to prior work that often focuses on weights only or uses layer-wise heuristics. The significance is high for real-world deployment of multilingual NLU on resource-constrained devices, as demonstrated by substantial reductions in memory footprint and latency while preserving accuracy across six low-resource languages. However, the significance is tempered by the lack of comparison to contemporary SOTA quantization methods.

### Suggestions for Improvement
1. **Detail the Controller Design and Training**: Add a subsection or appendix explicitly describing the controller's neural network architecture (if any), how the probability distribution \(P(q|s_l, \alpha)\) is parameterized, the exact loss function used for training (e.g., incorporating a bit-width penalty), and the optimization procedure (e.g., straight-through Gumbel-softmax).
2. **Benchmark Against Advanced PTQ Methods**: Include comparisons with at least two recent strong baselines such as GPTQ (weight-only) and SmoothQuant (weight-activation) on the same multitask model to position the proposed method's performance relative to the state of the art.
3. **Enhance Reproducibility**: Release code and detailed dataset preparation scripts. In the paper, provide an appendix with full hyperparameters, model architectures (for teachers, student, fusion module), and clarify the use of CRF across all models.
4. **Analyze Learned Precision Policies**: Present a table or figure showing the final bit-width assignment (e.g., for attention layers, MLP blocks, ID/DC/SF heads) for the proposed model. Discuss any observed patterns and their correlation with task sensitivity or model components.
5. **Deepen the Analysis**: Expand the error analysis with quantitative results (e.g., per-language, per-task error rates for the proposed model versus baselines). Discuss limitations more thoroughly, such as the computational overhead of training the controller, potential failure cases, and the method's applicability beyond the six Indic languages studied.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison with modern, relevant baselines.** The paper only compares static PTQ and a simple dynamic PTQ. For ICLR, it is essential to compare against state-of-the-art quantization methods (e.g., GPTQ, SmoothQuant, AWQ) and distillation-quantization co-design methods (e.g., from Liu et al. 2024, Ranjan & Savakis 2024). Without this, the claim of outperforming "static quantization" is weak and does not demonstrate novelty against the current field.
2. **Ablation study on the multi-teacher distillation framework.** The contribution of using three complementary teacher pairs (ID-DC, etc.) is asserted but not validated. An ablation removing individual teachers or using a single multi-task teacher is necessary to prove this design is critical to the final performance.
3. **Ablation study on the precision controller.** The paper does not isolate the gain from the proposed "precision-controlled, task-specific" policy. An experiment comparing it to (a) a uniform low-bit (e.g., all-INT8) version of the distilled student, and (b) a random or heuristic mixed-precision assignment, is needed to show the controller's value.
4. **Quantization sensitivity analysis per component.** The method assigns bits to encoder attention, MLP, and task heads. There is no analysis showing *why* these granularities were chosen or what the accuracy/sensitivity trade-off curve looks like for each component type. This is required to trust the design.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of why KD helps quantization robustness.** The paper states KD "preserves the integrity of all losses" and helps, but provides no mechanistic analysis. A study comparing the Hessian spectrum, weight/activation distributions, or task embedding spaces of the baseline vs. distilled model pre-quantization would explain why distillation is a necessary precursor for their PTQ.
2. **Root-cause analysis for per-language performance differences.** Table 3 shows slight variations, and the error analysis vaguely attributes issues to "richer morphology." A deeper linguistic analysis (e.g., correlation of performance drop with morphological complexity scores, token length, or training data size per language) is needed to move beyond speculation.
3. **Analysis of the precision controller's decisions.** What bit-widths did the controller actually choose for attention vs. MLP vs. each task head? Presenting and interpreting this final policy (e.g., "SF head requires 8-bit, DC head can use 4-bit") would provide crucial insight into task-specific sensitivity and validate the "task-specific" claim.
4. **Statistical significance across all key comparisons.** Table 4 only compares Baseline+Static PTQ vs. the proposed method. Significance testing must be extended to all critical comparisons in Table 2 (e.g., KD+Dynamic PTQ vs. Proposed) to confirm improvements are not due to chance.

### Visualizations & Case Studies
1. **t-SNE/UMAP plots of hidden representations** for the baseline, distilled, and quantized models on sample utterances. This would visually demonstrate whether the distillation and quantization preserve the semantic structure of the embedding space, especially for the confused intents/domains mentioned in the error analysis.
2. **Case studies of failure modes for each language.** The error analysis mentions over-prediction in Malayalam/Bengali and confusion in Dravidian languages. Concrete examples of input sentences, model predictions, and comparisons across the different model variants (baseline, distilled, quantized) would make these errors tangible and show where the method helps or fails.
3. **Latency/throughput vs. accuracy curves for the precision controller.** A plot showing the Pareto frontier of accuracy vs. latency/model size as the controller's trade-off parameter (α) varies would clearly demonstrate the efficiency-accuracy trade-off achieved.

### Obvious Next Steps
1. **Incorporate Quantization-Aware Training (QAT).** The paper correctly identifies the lack of QAT as a limitation, but for a paper on efficient deployment, this is a major missing step. A simple QAT finetuning of the distilled student should have been run and compared to the PTQ results, as it often yields better accuracy at low bits.
2. **Test on a public, established multilingual NLU benchmark.** Relying solely on a custom split of MASSIVE limits reproducibility and comparability. Results on the full, standard MASSIVE benchmark or another public dataset (e.g., MultiATIS++) are necessary to validate generalizability.
3. **Report hardware-aware metrics beyond CPU inference time.** For deployment claims, metrics like energy consumption, memory bandwidth, and latency on edge devices (e.g., ARM CPUs, mobile NPUs) are more relevant than generic CPU seconds. Profiling with tools like `torch.profiler` should be included.
4. **Clarify the calibration process for the proposed method.** The algorithm and text are contradictory. Algorithm 1 suggests an online, loss-driven training of the controller, while the text describes it as a post-distillation PTQ step. This needs a clear, reproducible description. A comparison to standard calibration-based mixed-precision search (e.g., using Hessian information) is also warranted.

# Final Consolidated Review
## Summary
This paper proposes a pipeline for efficient multilingual multitask NLU (Intent Detection, Domain Classification, Slot Filling) in six low-resource Indic languages. The method first employs multi-teacher knowledge distillation to train a robust student model, then applies a novel precision-controlled, task-specific dynamic post-training quantization (PTQ) scheme. This PTQ uses a learned controller to assign mixed bit-widths (4, 8, 16) independently to encoder components and task heads, unifying weight and activation policies. Experiments report significant reductions in model size and latency while maintaining high task accuracy.

## Strengths
- **Targets a socially valuable and under-studied setting.** The work focuses on six low-resource Indic languages, addressing a practical challenge for equitable and efficient NLU deployment.
- **Comprehensive empirical evaluation.** The paper systematically compares multiple configurations (baseline, distilled, and various PTQ variants) across three NLU tasks, reporting accuracy, F1, model size, and inference time, providing a clear view of trade-offs.
- **Novel integration of techniques.** The combination of a multi-teacher distillation framework (with adaptive fusion) and a precision-controlled, task-specific dynamic PTQ scheme that operates at the granularity of encoder blocks and individual task heads is a distinct contribution.

## Weaknesses
- **Critical lack of clarity in the core methodology.** The description of the proposed "precision-controlled PTQ" is contradictory and unclear. It is presented as a post-training method, yet the description involves a "learned precision controller" with "trainable logits," Gumbel-softmax sampling (Eq. 10), and Algorithm 1 shows backpropagation to update the model. This conflates PTQ with Quantization-Aware Training (QAT) or neural architecture search, making the core contribution difficult to understand, reproduce, or evaluate. The controller's architecture, training objective (e.g., balancing accuracy and bit-cost), and the process for freezing the final policy are not specified.
- **Unexplained and anomalous experimental results.** Key results undermine confidence in the experimental setup. Most notably, applying static PTQ to the *baseline* model reportedly *improves* Intent Accuracy (0.9481 to 0.9947) and Slot F1 (0.9674 to 0.9994), which is highly unusual for quantization and is not explained (e.g., as a regularization effect). Furthermore, the reported model sizes are inconsistent: "Baseline + Dynamic PTQ" (310 MB) is larger than "Baseline + Static PTQ" (279 MB), and the proposed mixed-precision model (428 MB) is larger than a uniform INT8 model should be (~270 MB), contradicting the efficiency narrative. These discrepancies require verification and clarification.
- **Insufficient comparison to relevant state-of-the-art.** The baselines are standard static and dynamic PTQ. The paper does not compare against recent, strong PTQ methods specifically designed for transformers (e.g., GPTQ, SmoothQuant) or mixed-precision techniques (e.g., HAWQ). Without these comparisons, the claimed advancement over "static quantization" is weak and the method's position within the current research landscape is unclear.

## Nice-to-Haves
- **Analysis of the learned precision policy.** Presenting the final bit-width assignments (which components received 4, 8, or 16 bits) would provide crucial insight into task/component sensitivity and validate the "task-specific" claim.
- **Deeper error and linguistic analysis.** Extending the qualitative error analysis with quantitative breakdowns (e.g., per-language error rates for each task under quantization) and correlating performance with linguistic features (e.g., morphological complexity) would strengthen the findings.
- **Reproducibility details.** Releasing the custom dataset construction code and full model/controller implementation details would greatly benefit the community.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Weakness: "The title is overly long."** (Formatting/style nitpick)
- **Weakness: "Distillation does not reduce model size, contradicting the typical goal of KD."** (The paper's goal is to use KD for performance improvement prior to quantization, not architectural compression. This is a valid design choice, not a flaw.)
- **Weakness: "Demands for comparison to pruning or QAT."** (The paper explicitly scopes its contribution to PTQ; suggesting QAT or pruning is a "nice-to-have" for future work, not a core weakness.)
- **Weakness: "Requests for hardware-aware metrics like energy consumption."** (While interesting, CPU inference time is a standard efficiency metric for this type of work.)
- **Strength: "The paper is well-written."** (Generic strength)

## Novel Insights
The paper's key novel insight is that for multitask models, quantization sensitivity is task-dependent, and a unified policy that jointly controls weight and activation precision at the granularity of encoder components *and individual task heads* can achieve a better efficiency-accuracy trade-off than uniform or layer-wise-only quantization. This insight is supported by the results showing maintained performance on sensitive tasks (Slot Filling) alongside gains on others (Domain Classification) under aggressive compression.

## Suggestions
- **Clarify the methodology.** Rewrite Section 4.2.3 to unambiguously describe whether the precision controller is trained *before* quantization (making it part of a QAT pipeline) or derived via a post-hoc search. Detail the controller's design, training objective, and how the final bit-width policy is frozen and deployed.
- **Explain anomalous results and verify metrics.** Provide a plausible explanation for the accuracy improvement from baseline static PTQ (e.g., quantization as regularization, or details of the calibration process). Double-check and justify all model size calculations to resolve inconsistencies.
- **Add comparisons to strong baselines.** Include results for at least one modern PTQ method (e.g., GPTQ or SmoothQuant) applied to the distilled student model to properly contextualize the performance of the proposed method.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
