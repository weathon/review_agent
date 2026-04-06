=== CALIBRATION EXAMPLE 16 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title is appropriate and reflects the core idea. The abstract clearly states the problem (latency of LLM classification), the solution (single-token outputs via atomic labels), and the results (accuracy and speed improvements). However, the reference to "GPT-5" is highly non-standard and confusing, as no such model is publicly known or documented. This significantly undermines credibility. If this is a placeholder for an unreleased model or an internal codename (e.g., "o1"), it must be explicitly clarified and justified. The claim of outperforming "GPT-5" cannot be evaluated without knowing what it is. The speed comparisons are compelling if the baselines are correct.

**Introduction & Motivation:** Well-written and effectively frames the latency-efficiency gap between prompting/decoding methods and encoder models. The three-pillar contribution (accuracy, latency, generality) is clearly stated. The motivation for real-time applications is solid. A minor gap: it doesn't explicitly mention prior attempts at single-token classification (e.g., via logit lens or direct projection), which could better position the novelty of the atomic token design.

**Related Work:** Covers prompt-based and encoder-based methods adequately. A notable omission is discussion of **Direct Preference Optimization (DPO)** or **Rejection Sampling** techniques that also aim to shape LLM outputs, which could provide context for the "constrained generation" formulation. The contrast with constrained decoding is clear. The section could be strengthened by mentioning concurrent work on efficient LLM classification to establish a more complete landscape.

**Methodology:** The core idea is sound and well-explained. Key strengths: the atomic token design, randomized label assignments to prevent memorization, and the clear single-step inference rule. However, several important implementation details are **missing or ambiguous**, affecting reproducibility:
1.  **Special Token Initialization:** How are the embeddings for the new control tokens `[o_k]` initialized? Randomly? As the mean of certain existing tokens? This can significantly impact training stability and final performance.
2.  **Tokenizer Modification:** The process of adding 500 new tokens to a pretrained tokenizer and resizing the embedding matrix is non-trivial. Are the new tokens added as "special" or "regular" tokens? How does this interact with the model's causal masking and generation logic? A brief description or reference to standard practices is needed.
3.  **Training Data Construction:** While the unified JSON schema is mentioned, the process of "reformulating" datasets like A-OKVQA into a classification task is not described. What were the input formats and label definitions for these vision-language datasets? This is crucial for understanding the model's multimodal training signal.
4.  **Loss Masking:** The description of setting other positions to `-100` is clear, but it should be confirmed that this applies to the entire input prompt (including system message and label descriptions), not just the "assistant" prefix.

**Experiments & Results:**
*   **Baseline Comparisons (Major Issue):** The text classification comparisons are **not fair**. The encoder baselines (BERT, RoBERTa) are fine-tuned with only **16 examples per class**, while the LaaC models are fine-tuned on a large, mixed corpus of 28k examples. This is an extreme low-shot vs. substantial fine-tuning comparison. To claim "zero-shot generalization," LaaC should be compared to a true zero-shot baseline (e.g., the base LLM with a carefully designed prompt) or to encoder models fine-tuned on comparable data. The current setup unfairly disadvantages the encoder models and overstates LaaC's generality.
*   **GPT-5 Ambiguity (Major Issue):** As noted, the "GPT-5" baseline is undefined and invalidates a key result. This must be corrected.
*   **Multimodal Results:** The MIntRec 2.0 results are strong. The comparison to encoder models (MAG-BERT, MulT) is reasonable as they are task-specifically fine-tuned, and LaaC's competitive performance is a valid point for its flexibility. The latency breakdown in Appendix A.5 is excellent and honest.
*   **Scaling Analysis:** The trends are clear and support the method's value for larger models. The label-set size analysis is good but limited to a maximum of 14 classes (DBpedia). For a method claiming scalability, testing on a dataset with >100 classes (like Banking77) for both accuracy and latency would be more convincing; Banking77 is in the appendix but not analyzed in the context of label-set scaling.
*   **Statistical Significance:** Accuracy is reported as single percentages. For the text benchmarks evaluated on only 200 samples, confidence intervals or p-values should be provided, especially when differences are small (e.g., 95.0% vs. 95.5%).
*   **Ablations Missing:** While the effect of mixed training data is shown, key ablations are absent: 1) The impact of **randomized label assignment** versus fixed mapping. 2) The performance of using **existing single tokens** (e.g., numbers, letters) as labels instead of new special tokens. 3) A comparison to a **simple linear probe** on the LLM's last hidden state (a strong, fast baseline), which would help isolate the benefit of the fine-tuning and token-based decoding.

**Writing & Clarity:** Overall, the paper is well-structured and readable. Some technical passages (e.g., Section 3.3) are dense but precise. The figures and tables are helpful. The main impediments to understanding are the missing methodological details noted above and the confusing "GPT-5" reference.

**Limitations & Broader Impact:** The limitations section is good, covering modality extension, calibration, and multilingual generalization. It could be strengthened by acknowledging: 1) The **finite label limit** (500 tokens) and the need for retraining if exceeded, contrasting with the infinite flexibility of prompting. 2) The **cost of fine-tuning** itself, even with LoRA, which is absent in pure prompting methods. 3) Potential negative societal impacts are not discussed (e.g., making highly efficient classifiers could lower the barrier for surveillance or automated content moderation systems); a brief statement would be appropriate.

### Overall Assessment
The paper proposes a clever and practically valuable method for making decoder LLMs efficient classifiers. The core idea—atomic label tokens with randomized assignment—is novel and well-executed. The empirical demonstration of latency reduction is convincing and important. However, **two critical flaws significantly weaken the current submission**: 1) The use of an undefined "GPT-5" baseline renders a major claim unverifiable. 2) The text classification comparisons are unfair, undermining the claims of strong zero-shot generality. Furthermore, the paper lacks necessary implementation details for reproducibility and key ablations to validate design choices. If the authors can replace the GPT-5 baseline with a standard model, conduct fair comparisons with properly trained encoder baselines, and add the missing methodological details and ablations, the contribution would be strong and likely suitable for ICLR. In its current form, these issues are too substantial for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes LaaC (LLM as a Classifier), a framework that reformulates classification as a constrained generation task with single-token outputs. By augmenting decoder-style LLMs/VLMs with atomic special tokens for each class and using parameter-efficient fine-tuning (LoRA), the method reduces classification to a deterministic, one-step decoding process. This yields significant inference speedups while maintaining competitive accuracy, positioning decoder LLMs as practical classifiers for latency-sensitive, multimodal applications.

### Strengths
1. **Substantial and Well-Demonstrated Latency Improvement**: The core contribution—single-token constrained generation—leads to order-of-magnitude reductions in median and tail latency compared to prompting-based LLMs (e.g., 8x faster than GPT-4o on text tasks) and consistent gains over base models. The paper provides thorough latency analysis across batch sizes (Appendix A.7), demonstrating the robustness of the efficiency gains.
2. **Strong Multimodal Performance**: The fine-tuned Gemma-3-27B model achieves 62.7% accuracy on the challenging MIntRec 2.0 benchmark, outperforming GPT-4o (43.7%) and GPT-5 (51.8%) and matching or surpassing specialized encoder-based models (MAG-BERT, MulT). This is a compelling result that demonstrates the effectiveness of the approach on complex, real-world tasks.
3. **Rigorous Experimental Design**: The evaluation is comprehensive, covering both multimodal (MIntRec 2.0) and diverse text-only benchmarks (SST-2, Amazon, AG News, DBpedia). The paper includes sensible baselines (proprietary APIs, encoder models, few-shot methods), scaling analyses, and ablation studies (e.g., effect of training data mixture, zero-shot token permutation tests in Appendix A.6).

### Weaknesses
1. **Limited Discussion of Calibration and Robustness**: While accuracy and latency are well-evaluated, critical aspects for real-world deployment—such as model calibration, confidence scores, and robustness to distribution shift or adversarial inputs—are not addressed. For ICLR, a deeper analysis of these properties would strengthen the claims about "practical" and "scalable" classifiers.
2. **Superficial Comparison to Encoder-Based Baselines**: The encoder baselines (MAG-BERT, MulT) are presented as task-specific and non-generalizable, but their latency is shown to be competitive or better (P50 0.23s vs. Gemma-3-4B's 0.37s). A more rigorous comparison, perhaps on a per-parameter or per-FLOP basis, and a discussion of the trade-off between specialization and generality would provide better context.
3. **Novelty is Incremental**: The core ideas—using special tokens for classification and constrained single-token generation—are natural extensions of existing concepts like verbalizers, grammar-constrained decoding, and prompt tuning. The paper's primary novelty lies in the cohesive integration for latency-sensitive multimodal applications, but the conceptual leap is modest.

### Novelty & Significance
**Novelty**: Moderate. The formulation of classification as single-token constrained generation is a clear and clever engineering solution, but it builds directly upon established paradigms (constrained decoding, prompt/prefix tuning, special tokens). The randomized label assignment during training is a nice technical detail to prevent memorization.
**Significance**: High for the systems and efficiency community. The paper convincingly demonstrates that decoder LLMs can be adapted to match the latency of encoder-based classifiers while retaining their generative flexibility and achieving strong accuracy. This addresses a real pain point in deploying LLMs for real-time classification tasks.

### Suggestions for Improvement
1. **Deepen the Analysis of Reliability**: Include experiments on model calibration (e.g., Expected Calibration Error), out-of-distribution detection, and sensitivity to prompt phrasing. This would substantiate the claim of being "practical for real-world deployment."
2. **Provide a More Equitable Efficiency Comparison**: Compare against encoder baselines not just on end-to-end latency but also on metrics like throughput (examples/sec) under identical hardware budgets and discuss the total cost of training (LoRA + data curation vs. full fine-tuning).
3. **Clarify the Limitations and Scope**: The framework currently handles up to 500 classes via reserved tokens. Discuss the implications of scaling to extremely large label spaces (e.g., thousands of classes). Also, explicitly state that the method requires fine-tuning and is not a zero-shot technique for entirely new tasks (zero-shot here refers to new label-to-token mappings, not new tasks).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on the randomized label assignment vs. fixed mapping is missing.** The paper claims randomization prevents memorization, but there is no quantitative comparison to a fixed token-class mapping. Without this, it’s unclear if randomization is necessary or beneficial for generalization.
2. **No fair comparison with encoder models fine-tuned on the same multimodal corpus.** The encoder baselines (MAG-BERT, MulT) are fine-tuned on the target dataset, while LaaC is fine-tuned on a mixed corpus. To claim superiority, encoder models must be trained on the same data and evaluated end-to-end with their full feature extraction pipelines.
3. **Lack of controlled ablation on training data composition.** The paper shows adding text data helps on a proprietary dataset, but there is no systematic study on MIntRec 2.0. An ablation training only on multimodal vs. only on text data is needed to justify the mixed corpus.
4. **No experiment scaling to truly large label spaces (e.g., hundreds of classes).** The paper claims support for up to 500 classes but only evaluates up to 77 (Banking77). Testing on a dataset with several hundred classes is necessary to validate scalability claims.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of why LaaC underperforms the base model on Banking77.** Table 7 shows a ~4% drop for Mistral-3-24B (LaaC) vs. base. Understanding this regression—whether due to training data, token randomization, or loss of generative flexibility—is critical to assess limitations.
2. **Calibration analysis is absent.** Single-token classification provides a confidence score via softmax; reporting expected calibration error (ECE) would show whether these probabilities are reliable for deployment.
3. **Sensitivity analysis of prompt wording for zero-shot tasks.** The paper uses fixed prompt templates but does not test how variations in label descriptions affect accuracy. This is key to trusting the claimed generality.
4. **Modality contribution analysis for multimodal tasks.** For MIntRec 2.0, there is no analysis ablating vision or text inputs to show the model actually uses both modalities, rather than relying on one.

### Visualizations & Case Studies
1. **Case studies of failures on MIntRec 2.0 vs. GPT-4o.** Showing concrete examples where LaaC misclassifies but GPT-4o gets it right (and vice versa) would reveal whether the speed-accuracy trade-off stems from reasoning gaps or modality misunderstanding.
2. **Visualization of attention patterns for the single-token decision.** Highlighting which input tokens or image regions the model attends to when outputting the control token would verify it’s making decisions based on relevant content.
3. **t-SNE/PCA plot of learned control token embeddings.** Visualizing whether semantically similar classes cluster in embedding space would indicate if the model learns meaningful label representations beyond random assignments.

### Obvious Next Steps
1. **Compare latency with optimized encoder models (quantized, distilled).** The encoder latency comparison uses standard models; for a realistic efficiency claim, compare against highly optimized encoders (e.g., quantized BERT) that are common in production.
2. **Preliminary out-of-distribution (OOD) detection experiment.** Since real-world classifiers must handle OOD inputs, a simple test on CLINC’s out-of-scope examples would strengthen practical relevance.
3. **Extend evaluation to standard image classification (e.g., ImageNet with text prompts).** Relying only on MIntRec 2.0 for multimodal evaluation is narrow; adding a standard vision dataset would demonstrate broader applicability.
4. **Include cost analysis (dollars per inference) alongside latency.** For deployment, cost matters. Comparing LaaC’s self-hosted cost vs. GPT-4o API costs would provide a more complete practical argument.

# Final Consolidated Review
## Summary
This paper proposes LaaC, a framework that adapts decoder-style LLMs for classification by introducing atomic special tokens for each class and using parameter-efficient fine-tuning. This reformulates classification as a single-token generation task, achieving substantial latency reductions while maintaining competitive accuracy on text and multimodal benchmarks.

## Strengths
- **Substantial and well-demonstrated latency improvement:** The single-token constrained generation leads to order-of-magnitude lower latency compared to prompting-based LLMs (e.g., 8× faster than GPT-4o on text tasks) and consistent gains over base models, with thorough analysis across batch sizes (Appendix A.7).
- **Strong multimodal performance:** Fine-tuned Gemma-3-27B attains 62.7% accuracy on the challenging MIntRec 2.0 benchmark, outperforming GPT-4o (43.7%) and matching or surpassing specialized encoder-based models (MAG-BERT, MulT), demonstrating effectiveness on complex tasks.
- **Clever design choices:** The use of randomized label assignments during training prevents memorization and enables zero-shot adaptation to new label mappings, as shown in Appendix A.6.

## Weaknesses
- **Undefined baseline:** The paper compares against "GPT-5" and "GPT-5-NANO", which are not publicly known or documented models. This makes a key empirical claim (outperforming GPT-5) unverifiable and undermines the credibility of the results.
- **Unfair evaluation of generality:** The text classification comparisons involve encoder baselines fine-tuned on only 16 examples per class, while LaaC is pre-fine-tuned on a large mixed corpus of 28k examples. This asymmetric setup does not fairly assess zero-shot generalization and overstates LaaC's advantages. A more appropriate comparison would involve encoders trained on similar data or true zero-shot LLM baselines.
- **Insufficient analysis of design choices:** The paper lacks ablations on critical components such as the randomized token assignment (vs. fixed) and the use of special tokens (vs. existing single tokens). Without these, it is unclear whether the reported gains are due to the core design or other factors.

## Nice-to-Haves
- Analysis of model calibration and robustness (e.g., expected calibration error, sensitivity to prompt variations) to strengthen claims about practical deployment.
- Experiments scaling to very large label spaces (e.g., hundreds of classes) to validate the scalability claim beyond 77 classes.
- Deeper investigation into the modality contributions for multimodal tasks (e.g., ablating vision or text inputs on MIntRec 2.0).

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Replace the "GPT-5" baseline with a publicly available and well-documented model, or provide a clear description and justification for what it represents.
- Re-evaluate the text classification experiments with a fairer comparison: either fine-tune encoder models on the same mixed corpus used for LaaC or compare to true zero-shot LLM prompting (without any fine-tuning).
- Include ablations on the randomized token assignment and the special token design to validate their necessity and impact.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
