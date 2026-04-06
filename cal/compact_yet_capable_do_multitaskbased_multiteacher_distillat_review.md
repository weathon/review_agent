=== CALIBRATION EXAMPLE 19 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title is long but accurately reflects the core comparison. The abstract clearly states the problem, approach, and claimed results. However, it makes strong claims of "superior accuracy–efficiency trade-off" and "significantly reducing inference latency... while preserving accuracy" that set high expectations. The abstract mentions a "custom multilingual Indic dataset" but does not disclose its size or composition, making initial assessment difficult.

**Introduction & Motivation**
The problem is well-motivated: efficient multitask NLU for low-resource Indic languages. The three contributions are clearly listed. A significant weakness is that the contributions are presented as a list of *what* was done rather than a clear statement of *novelty*. For ICLR, it is crucial to distinguish what is new versus what is an application of known techniques (e.g., static/dynamic PTQ, multi-teacher KD). The claimed novelty of a "precision-controller-driven task specific dynamic PTQ scheme" and a "unified weight-activation policy" needs to be sharply defined against prior mixed-precision and dynamic quantization work cited in the survey.

**Literature Survey**
The survey adequately covers relevant areas (multitask NLU, KD, PTQ). It correctly identifies gaps: prior work often lacks per-task-head control and a unified policy for weights *and* activations. This sets up the paper's claimed contribution. However, the critique that prior methods are "weight-only or activation-only in practice" is overstated; many PTQ methods quantize both. The gap is better framed as the lack of *task-conditioned, controller-based* joint precision assignment.

**Dataset**
The description is minimal. Citing MASSIVE is good, but stating a "custom" dataset was prepared from it requires clarification. How was it customized? Is it a subset or a new annotation? The provided statistics (163k train, 40k test utterances, 540 intents, etc.) are helpful, but the split rationale (train/test) and potential data leakage issues are not discussed. For reproducibility, a more detailed description or a public release plan is needed.

**Methodology**
This is the core section and contains several serious issues that undermine the paper's claims and reproducibility.
1.  **Unclear Baseline and Distillation Framework:** The description of the multi-teacher distillation is confusing. Equation 4 shows a total loss with components for ID, DC, and SF, but the teachers are pairs (ID+DC, etc.). How are the KD losses \(L_{KD}^{ID}, L_{KD}^{DC}, L_{KD}^{SF}\) computed from these paired teachers? The "attention-based fusion" is mentioned but not described mathematically. What is \(L_{CRD}^{SF}\)? It's introduced without definition.
2.  **Precision Controller Vagueness:** The proposed precision controller is the key novelty but is poorly explained. Equation 10 shows a Gumbel-Softmax sampling over logits \(\theta_L\), but it's unclear what these logits are a function of. What inputs does the controller use (task ID, layer statistics)? How is it trained? Algorithm 1 mentions a "sensitivity score \(s_l\)" and updating the controller via backpropagation, but this contradicts the earlier claim of applying PTQ *after* distillation "without calibration" and "without retraining." The text and algorithm describe what seems like a quantization-aware training (QAT) or learned mixed-precision search, not a pure PTQ method. This is a major logical inconsistency.
3.  **Mathematical Errors and Undefined Variables:** Equation 2 uses \(H^{[i]}\) which is not defined. Equation 5's denominator uses "127", which is correct for symmetric int8, but the variable \(p(b|L)\) is never used again. The quantization formulas switch between per-tensor (Eq. 2, 8) and per-channel (implied by \(s_W\) in Eq. 5) schemes without justification.
4.  **Static PTQ on Distilled Model (4.2.1):** The statement that static PTQ "degraded performance due to disrupted KD signals" is an observation, not a methodological explanation. It hints that the distilled model's representations are less quantization-robust, which is interesting but not analyzed.

**Experiments & Results**
The results are dramatic, showing near-perfect accuracy (99.9+% on ID/SF) with large compression. This raises skepticism.
1.  **Extremely High Baselines:** The baseline model (FP32 XLM-R) achieves 94.8% ID Acc and 97.8% Slot Acc. After simple static PTQ, these jump to **99.5% and 99.9%** (Table 2). This is highly unusual; PTQ typically causes a minor drop, not a large *increase*. This suggests either (a) the baseline was severely under-trained, (b) the evaluation metric or data split is problematic, or (c) there is an error in the PTQ implementation (e.g., it's not actually quantized at runtime). This invalidates the primary comparison.
2.  **Unfair Comparison?** The proposed method (KD + Precision PTQ) is compared against a **non-distilled** baseline with static PTQ. The more relevant and challenging comparison would be against **KD + standard Dynamic PTQ** (row 6 of Table 2). Compared to that, the gains of the precision controller are marginal (e.g., Domain Acc 87.7% vs. 90.2%) while being more complex. The paper's main claim is not strongly supported by this comparison.
3.  **Missing Ablations and Details:** How was the precision controller trained? What was its architecture? What was the search space (bit-widths per component)? How long did the search take? The algorithm suggests an iterative process, but the "Experiments" section (Section 5) does not describe this training loop for the controller, again pointing to a major methodological gap.
4.  **Efficiency Metrics:** Reporting inference time in seconds is not standard without specifying the exact hardware (CPU type, cores), batch size, and whether times are averaged over multiple runs. "Model Size" is clear, but latency/Bandwidth trade-offs need proper measurement.
5.  **Statistical Significance (Table 4):** Reporting p-values is good, but the comparisons are between the *non-distilled* static PTQ baseline and the final proposed model. The significant gains likely stem from adding KD, not necessarily from the precision controller. The comparison should isolate the controller's effect.

**Error Analysis**
This section is superficial. It lists common error types (over-prediction of domains, confusion between similar intents) but does not provide quantitative analysis (e.g., % of errors of each type) or link them back to the design of the precision controller. The claim that "the dynamically quantized model demonstrates fewer such errors" is not supported with evidence in the paper.

**Limitations & Conclusion**
The limitation section is weak. It mentions not using QAT as a limitation but fails to acknowledge the critical limitations exposed in this review: the potentially flawed baseline comparison, the lack of clarity and reproducibility in the controller design, and the extremely high accuracies that challenge believability. The conclusion overstates the findings, claiming the approach "sustaining near-perfect accuracy," which is not credible for this problem domain.

**Writing & Clarity**
The writing is generally clear at a high level, but the technical details in the methodology are confusing and incomplete, as noted. Figure and table references are broken (e.g., "Fig. 1", "Table 1" appear in the text but the figures themselves are not included in the provided content, which is a parser issue). The flow from problem to method to results is logical.

### Overall Assessment
The paper addresses an important and timely problem: efficient multilingual multitask NLU. The idea of a precision controller for task-specific dynamic PTQ is potentially novel. However, the work is severely undermined by critical methodological omissions and highly suspicious experimental results. The massive accuracy *gains* from applying basic PTQ to the baseline are a major red flag, suggesting fundamental issues with the experimental setup or evaluation. The core algorithm for the precision controller is not reproducible, with conflicting descriptions between the text, equations, and pseudo-code. For ICLR, where novelty, rigor, and reproducibility are paramount, the paper in its current form does not meet the bar. The contribution is currently unclear and unsupported by the evidence presented. Significant revisions—including a credible baseline, a complete and correct description of the proposed method, and thorough ablation studies—are required.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a method to improve efficiency and accuracy for low-resource, multilingual, multitask NLU (covering Intent Detection, Domain Classification, and Slot Filling). The core contribution is a pipeline that first applies multi-teacher knowledge distillation (KD) to create a capable student model, and then applies a novel precision-controlled, task-specific dynamic post-training quantization (PTQ) scheme. This scheme uses a controller to assign mixed bit-widths (4, 8, 16) to different model components (e.g., encoder layers, task heads) under a unified weight-activation policy. Experiments on a custom dataset of six Indic languages show this combined approach outperforms static PTQ and standard dynamic PTQ in terms of model size, inference latency, and task accuracy.

### Strengths
1. **Relevant and Practical Problem**: The work tackles a significant challenge: deploying efficient, accurate multitask NLU models for low-resource Indic languages. The focus on efficiency (via KD and quantization) for real-world, constrained hardware is highly applicable.
2. **Comprehensive Experimental Comparison**: The paper systematically evaluates multiple configurations: baseline, baseline+PTQ, KD-only, KD+Static PTQ, KD+Dynamic PTQ, and the proposed KD+Precision-Controlled PTQ. This provides clear evidence for the incremental benefits of each component (Table 2).
3. **Strong Empirical Results**: The proposed method achieves compelling results: a 59.8% reduction in model size and 67.1% faster inference versus the baseline, while reporting near-perfect scores for Intent and Slot tasks and a solid improvement in Domain Classification (Table 2). The per-language analysis (Table 3) demonstrates consistent performance across diverse languages.
4. **Incorporation of Statistical Analysis**: The inclusion of p-values from paired t-tests (Table 4) to validate the significance of improvements (especially for Intent and Domain tasks) adds rigor to the claims.

### Weaknesses
1. **Lack of Clarity in Critical Methodological Details**: The paper is vague on how the central component—the "precision controller"—is actually implemented and trained. Algorithm 1 is high-level and does not specify the controller's architecture, how the sensitivity score \(s_l\) is computed, or how the controller is optimized jointly with the quantization process (lines 12-13). The description of the Gumbel-based selection (Eq. 10) is not integrated into the algorithm narrative.
2. **Insufficient Dataset and Experimental Setup Details**: While the custom dataset is based on MASSIVE, crucial details are missing: the specific preprocessing steps, the train/validation/test split proportions, and how the "multi-intent, cross-domain" annotations were derived or verified. This hampers reproducibility.
3. **Weak Baseline Comparisons and Missing SOTA Context**: The comparisons are primarily against the authors' own ablations (different PTQ methods on their models). There is no comparison with other state-of-the-art efficient NLU methods, quantization-aware training (QAT) techniques, or other distillation schemes for multilingual settings, making it difficult to assess the true novelty and relative performance.
4. **Overstated Claims and Unexplained High Performance**: The reported accuracies (e.g., 99.91% for Intent) are exceptionally high, nearing perfection, which is unusual for complex, low-resource, multitask NLU. This raises questions about potential data leakage, an overly simple test set, or the specific evaluation metrics used. The paper does not sufficiently discuss or justify these near-ceiling results.
5. **Presentation and Figure Issues**: Several figures are referenced (Figs. 1, 2, 3, 4, 5, 6) but their content is described only in captions within the text; the actual visual data is not provided in the submitted text file. This makes it impossible to evaluate the diagrams illustrating the architecture and results. The writing also contains minor grammatical errors and inconsistent formatting (e.g., "~~q~~ uery").

### Novelty & Significance
**Novelty**: The idea of a *task-specific* dynamic PTQ controller applied to a *multitask* model distilled from *multiple teachers* is a novel combination of existing techniques (KD, mixed-precision quantization). The explicit assignment of precision per task head (ID, DC, SF) is a distinctive design choice compared to typical layer-wise mixed-precision methods.
**Significance**: If reproducible and generalizable, the method offers a tangible path to deploy accurate NLU services on edge devices in linguistically diverse regions. The work highlights the potential of tightly coupling architectural distillation with granular, policy-driven quantization.

### Suggestions for Improvement
1. **Clarify the Precision Controller**: Add a dedicated subsection detailing the controller's neural network architecture (if any), the exact training procedure (how the Gumbel-Softmax/temperature scheduling works, the loss function for the controller), and how it interacts with the quantization process. Explain how the final "frozen" policy is derived from the stochastic training phase.
2. **Provide Comprehensive Dataset and Reproducibility Details**: Explicitly describe the dataset construction process: how utterances were selected from MASSIVE, the annotation protocol for multi-intent scenarios, and the exact splits. Release the dataset or provide a detailed recipe for its creation. Publish code for the full pipeline.
3. **Expand Comparisons and Justify High Scores**: Compare against strong external baselines, such as a quantized version of a state-of-the-art multilingual model (e.g., Quantized mT5) or other KD+QAT methods. Perform a deeper error analysis to understand the sources of the remarkably high accuracy and discuss potential limitations or simplifying assumptions in the evaluation setup.
4. **Address the Limitation of PTQ-only Approach**: The "Limitations" section correctly notes the absence of QAT. The discussion should be expanded to hypothesize how much further gain QAT might provide and why it was not explored, perhaps due to the low-resource setting making QAT challenging.
5. **Improve Presentation**: Ensure all critical figures are included or their key findings are described fully in the text. Meticulously proofread the manuscript to correct grammatical errors and improve clarity. The title, while descriptive, is extremely long and could be shortened for impact.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Benchmark against established PTQ methods.** The paper lacks comparisons to state-of-the-art PTQ baselines like GPTQ, SmoothQuant, or ZeroQuant for the backbone model (XLM-R). Without these, the claim that the proposed method "outperforms static quantization" is weak and unconvincing for ICLR.
2. **Ablation study on the precision controller.** The contribution of the "precision controller" is not isolated. An ablation must show the performance of the distilled model with (a) uniform 8-bit, (b) uniform 4-bit, and (c) the controller's mixed precision, to prove the controller adds value beyond simple lower-bit quantization.
3. **Quantitative analysis of distillation necessity.** The paper claims multi-teacher distillation is crucial, but there's no experiment showing a quantized version of the non-distilled multitask baseline (or a single-task model) under dynamic PTQ. This gap undermines the claim that the full KD pipeline is necessary for the achieved efficiency.
4. **Proper dataset split and reproducibility details.** The dataset is "custom" from MASSIVE, but no details are given on the exact split creation, potential data leakage, or how multi-intent examples were handled. This makes the near-perfect results (e.g., 99.9% accuracy) highly suspect and unreproducible.

### Deeper Analysis Needed (top 3-5 only)
1. **Explain the precision controller's decisions.** The paper does not analyze *what* the controller learned. Which layers/tasks get which bit-widths, and does this align with known sensitivity? A breakdown is needed to trust that it's not a random assignment.
2. **Statistical significance of "near-perfect" scores.** Reporting 99.91% accuracy with a standard deviation of 0.0006 is extraordinary for low-resource languages. A deeper analysis must investigate if this is due to an overly simple or flawed evaluation setup (e.g., majority class prediction, test set contamination).
3. **Failure mode analysis linked to quantization.** The error analysis is generic and not tied to the quantization method. A critical analysis must show: for samples where the quantized model fails but the FP32 teacher succeeds, is the error correlated with low precision assignments or activation outliers?

### Visualizations & Case Studies
1. **Visualize the precision assignment map.** A heatmap showing the assigned bit-width (4,8,16) for each encoder layer and task head would instantly reveal if the controller's policy is structured and interpretable, or seemingly arbitrary.
2. **Case studies of quantization-induced errors.** Provide specific utterance examples (in original language and translation) where the proposed model fails, alongside the predictions of the FP32 teacher and a standard 8-bit PTQ model. This would concretely demonstrate the method's advantages and failure boundaries.

### Obvious Next Steps
1. **Incorporate Quantization-Aware Training (QAT).** The paper dismisses QAT as future work, but given the focus on low-bit (4-bit) quantization, QAT is a standard and necessary step to stabilize performance. Its absence is a major methodological shortcoming.
2. **Justify the multi-teacher architecture.** The choice of three teacher pairs (ID+DC, etc.) is not motivated. An obvious step is to compare against a simpler, more standard setup: a single, larger teacher trained on all three tasks, or an ensemble of single-task teachers.
3. **Report hardware-aware metrics.** For an efficiency paper, only model size and CPU inference time are reported. To prove real-world utility, standard metrics like energy consumption, latency on edge devices (e.g., Raspberry Pi), or operations count (e.g., BitOps) are essential.

# Final Consolidated Review
## Summary
This paper proposes a pipeline for efficient multilingual multitask NLU (Intent Detection, Domain Classification, Slot Filling) for low-resource Indic languages. It combines multi-teacher knowledge distillation with a novel precision-controlled, task-specific dynamic post-training quantization (PTQ) scheme, where a controller assigns mixed bit-widths (4, 8, 16) to different model components under a unified weight-activation policy. Experiments on a custom dataset derived from MASSIVE claim significant reductions in model size and latency while maintaining high accuracy.

## Strengths
- **Addresses a Relevant and Practical Problem**: The work tackles the significant challenge of deploying accurate and efficient multitask NLU models on constrained hardware for low-resource Indic languages, which is a valuable direction.
- **Comprehensive Ablative Experimental Framework**: The paper systematically evaluates a progression of methods (baseline, +PTQ, KD-only, KD+PTQ variants), providing clear evidence for the incremental benefits of distillation and dynamic quantization over static approaches (Table 2).

## Weaknesses
- **Critical Lack of Methodological Clarity and Reproducibility**: The core novel component—the precision controller—is described inconsistently and insufficiently. The text (Sec. 4.2.3) describes a post-training application "without calibration," yet Algorithm 1 and associated equations (e.g., Eq. 10) depict a training process involving sensitivity scores, Gumbel-Softmax sampling, and controller updates via backpropagation. This contradiction makes the proposed method impossible to understand or reproduce. Furthermore, key distillation details (e.g., the formulation of the attention-based fusion and the `L_CRD^SF` loss) are missing.
- **Highly Suspicious and Unexplained Experimental Results**: The reported accuracies are extraordinarily high (e.g., 99.91% for Intent Accuracy) and defy expectations for complex, low-resource, multitask NLU. More critically, applying simple static PTQ to the non-distilled baseline *improves* accuracy dramatically (e.g., Slot Accuracy from 97.82% to 99.94%), which is a major red flag. This anomalous result suggests potential issues with the evaluation setup, data leakage, or implementation, severely undermining the validity of all comparative claims.
- **Insufficient and Unfocused Comparative Analysis**: The paper's central claim is that precision-controlled PTQ outperforms standard methods. However, the most relevant comparison—against **KD + Dynamic PTQ** (row 6, Table 2)—shows only marginal gains (e.g., Domain Accuracy 87.7% vs. 90.2%) from the far more complex controller. The paper fails to convincingly isolate and demonstrate the value of its key novelty beyond the benefits already provided by distillation and standard dynamic quantization.

## Nice-to-Haves
- **Visualization of the Controller's Policy**: A figure showing the learned bit-width assignment across encoder layers and task heads would help interpret the controller's behavior.
- **Case Studies of Quantization-Sensitive Errors**: Providing concrete examples where the proposed model fails, compared to the full-precision teacher, could better illustrate the method's limitations and advantages.

## Novel Insights
None beyond the paper's own contributions. The proposed combination of multi-teacher distillation with a task-head-aware precision controller is a stated design, but its novel insights are obscured by the methodological and experimental issues described above.

## Suggestions
- **Clarify the Precision Controller Method**: Revise the methodology section to provide a single, clear, and consistent description of the precision controller. Specify whether it involves a search/training phase or is a rule-based post-hoc assignment, detail its inputs and architecture (if any), and explain exactly how the final frozen policy is derived.
- **Thoroughly Investigate and Explain the Experimental Anomalies**: The authors must rigorously audit their experimental pipeline to identify the cause of the implausible accuracy jumps from PTQ and the near-perfect scores. This includes verifying the data splits for leakage, re-running evaluations, and providing a credible explanation or revising the results. The comparison must be refocused to fairly isolate the contribution of the precision controller against the strongest baseline (KD+Dynamic PTQ).

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
