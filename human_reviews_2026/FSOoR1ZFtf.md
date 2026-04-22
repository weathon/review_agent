# Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
Recent progress in large language models (LLMs) has led to systems capable of producing text with remarkable fluency. However, these models are still prone to factual inaccuracies, often referred to as \``hallucinations''. One strategy to alleviate this issue is uncertainty quantification (UQ), but most existing approaches are computationally intensive or require supervision. In this work, we propose Recurrent Attention-based Uncertainty Quantification (RAUQ), an unsupervised and efficient framework for identifying hallucinations. The method leverages an observation about transformer attention behavior: when incorrect information is generated, certain \``uncertainty-aware'' attention heads, tend to reduce their focus on preceding tokens. RAUQ automatically detects these attention heads and combines their activation patterns with token-level confidence measures in a recurrent scheme, producing a sequence-level uncertainty estimate in just a single forward pass. Through experiments on twelve tasks spanning question answering, summarization, and translation across four different LLMs, we show that RAUQ consistently outperforms state-of-the-art UQ baselines. Importantly, it does so with minimal cost, less than 1\% additional computation. Since it requires neither labeled data nor extensive parameter tuning, RAUQ serves as a lightweight, plug-and-play solution for real-time hallucination detection in white-box LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes RAUQ (Recurrent Attention-based Uncertainty Quantification), an unsupervised white-box method for hallucination detection. The key insight is that certain "uncertainty-aware" attention heads reduce focus on preceding tokens when models generate incorrect information. RAUQ identifies these heads per layer, computes token-level confidence scores recurrently combining attention weights and token probabilities, and aggregates to sequence-level uncertainty using maximum across layers. Evaluated on 4 LLMs across 12 diverse tasks (QA, summarization, translation), RAUQ achieves state-of-the-art performance with <1% computational overhead.

### Strengths
The empirical results are impressive and consistent. RAUQ substantially outperforms 15 baselines across diverse tasks and models, with particularly strong gains on translation tasks (mean PRR 0.384 vs. 0.326 for the next-best method). The mechanistic insights are compelling: Figures 1-4 clearly show how certain attention heads reduce focus on preceding tokens precisely at hallucination points, providing genuine intuition for why the method works.

The practical efficiency is exceptional—less than 1% computational overhead compared to 400-800% for sampling-based competitors. A single forward pass requirement makes deployment genuinely feasible. The experimental evaluation is thorough and comprehensive, encompassing 4 LLM families across 12 diverse datasets with detailed ablation studies examining aggregation functions, hyperparameters, and layer selection. Findings generalize across tasks without task-specific tuning, and middle layers consistently emerge as most informative. The method is reproducible with promised code release and detailed experimental specifications.

The work demonstrates robustness: performance holds across different model families and diverse generation tasks without modification. The analysis shows that unsupervised uncertainty emerges naturally in model internals, providing theoretical encouragement that these patterns reflect genuine uncertainty rather than artifacts.

### Weaknesses
The theoretical foundation is shallow. The paper shows that specific attention heads drop focus at hallucination tokens but doesn't deeply explain the underlying mechanism. Why does maximum aggregation across layers emerge as optimal? Why does combining head selection, recurrence, and aggregation work together synergistically? Limited ablations on component interactions make it hard to understand which pieces are essential. The connection between attention patterns and hallucinations relies on empirical observation rather than principled explanation.

Several methodological choices appear ad-hoc. Head selection based on average attention to previous tokens lacks justification—alternative metrics aren't considered. The fixed decision center (top-right) is used across all models and tasks despite the paper acknowledging optimal centers vary by context. The hyperparameter α=0.2 is selected via validation on Llama-2 but robustness across all evaluated models isn't verified. Layer selection (first to second third) also seems arbitrary, especially since Table 8 suggests individual layers sometimes outperform the aggregated version.

The evaluation uses inconsistent quality metrics across tasks (accuracy for QA, AlignScore for summarization, COMET for machine translation), making unified interpretation difficult. There's no evaluation on instruction-following, dialog, or diverse generation types. Critically, selected heads differ substantially across datasets (Tables 9-10), which is concerning for a method claiming to be truly unsupervised. The paper lacks analysis of which error types (factual, reasoning, formatting) are detected or why answer length significantly affects performance across tasks.

Generalization is questionable. Only open-source models are evaluated; whether closed-source models (GPT-4, Claude) exhibit the same attention patterns remains unknown. There's no evaluation on models trained with different objectives (DPO, preference learning) or on a wider size range (no models <3B or >70B). Tables 12-13 show supervised methods substantially outperform RAUQ in-domain, suggesting a real opportunity cost to the unsupervised approach.

White-box access is required, limiting applicability since most users access models through APIs. The method's reliance on specific attention patterns may not generalize to future architectures with different attention mechanisms. The core assumption that hallucinations cause detectable attention drops hasn't been validated on adversarial inputs where models hallucinate confidently.

### Questions
1. Why specifically does attention to previous token drop at hallucinations? Have you verified this pattern holds for other attention patterns (e.g., attention to question)?

2. Can you provide theoretical justification for the recurrent confidence formula (Equation 2)? Why is this the right way to propagate uncertainty?

3. How does performance vary if you adapt α per-model using validation data?

4. Why do selected heads differ substantially across datasets if the method is fully unsupervised?

5. How does the method perform on adversarial inputs where models hallucinate confidently?

6. Can you test on models with non-standard attention mechanisms (e.g., sparse attention, linear attention)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes RAUQ (Recurrent Attention-based Uncertainty Quantification), a lightweight and unsupervised hallucination detection approach for large language models (LLMs).RAUQ identifies “uncertainty-aware attention heads” that display attention degradation when models produce hallucinated content. The method recursively propagates token-level uncertainty based on attention and token probabilities using only a single forward pass, incurring less than 1% extra computation. Extensive experiments across 4 LLMs (e.g., Llama-3.1, Gemma-2) and 12 tasks demonstrate consistent performance improvements over more than 15 baseline methods.

### Strengths
1.	RAUQ Leverages “uncertainty-aware” attention heads for hallucination detection is a creative and intuitive direction that remains computationally efficient and easy to implement in white-box settings. The authors evaluate multiple models and tasks, showing solid gains in PRR and selective generation metrics, with thorough ablation studies on α, layer ranges, and head selection.
2.	RAUQ operates in a single forward pass and introduces negligible computational overhead (<1%), which makes it suitable for real-time or large-scale deployment.
3.	The work provides both quantitative and qualitative insights into why certain attention heads correlate with uncertainty, offering interpretability beyond pure calibration metrics.

### Weaknesses
1. While empirical findings are convincing, the paper lacks a rigorous theoretical explanation of why specific heads consistently reflect uncertainty. A deeper causal or functional analysis would enhance scientific understanding.
2. RAUQ requires access to internal attention weights, limiting its applicability in black-box LLMs such as API-based commercial models. Discussion on possible extensions or alternative signals for black-box use would strengthen the paper.

### Questions
1.	How are attention weights normalized and extracted from different architectures (e.g., Llama vs Gemma) to ensure consistent computation of RAUQ? How stable the identified “uncertainty-aware heads” remain across different prompt styles or lengths?
2.	Lines 270–289 — the pseudocode formatting in Algorithm 1 should be adjusted. The current line numbers exceed the right boundary, and the table layout contains excessive white space, which negatively affects readability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work propose Recurrent Attention-based Uncertainty Quantification (RAUQ), an unsupervised and efficient framework for identifying hallucinations. RAUQ leverages the behavior of “uncertainty-aware” attention heads to further improve the Focus approach by Zhang et al., 2023. Experiments on question answering, summarization, and translation tasks show that RAUQ consistently outperforms SOTA baselines. The low resource cost make RAUQ plug-and-play for real-time hallucination detection in white-box LLMs.

### Strengths
1. The analysis of uncertainty-aware attention heads is both interesting and sound, lending interpretability to the subsequent improvements made to the Focus approach.

2. Experiments on 12 relevant benchmarks demonstrate the consistent superiority of the proposed approach. Further analysis of the framework modules and computational costs reinforces the strength of this work.

### Weaknesses
1. The experiments only conduct on UQ methods, other hallucination detection works based on external information can also be introduced.

2. This work is based on small-scale LLMs with approximately 8B parameters. Future experiments on larger models (30B or 70B parameters) are expected to yield improved soundness.

### Questions
1. What does "Simple Focus" mean? There is no description about this term. Do you mean a simplified version of the original Focus Approach by Zhang et al., 2023?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose Recurrent Attention-based Uncertainty Quantification (RAUQ), a method that detects "uncertainty-aware" attention heads to identify hallucinations in NLG.

### Strengths
- The paper addresses a well-defined problem with a clear mechanistic insight: the link between attention head behavior and uncertainty.
- It is clearly written, well-structured, and the visualizations effectively support the main findings.
- RAUQ is fully unsupervised and adds less than 1% computational overhead, making it practical for real-time use when model internals are accessible.
- The experimental evaluation is comprehensive, covering 12 datasets, 4 models, and 3 generation tasks. RAUQ consistently outperforms strong baselines, and the ablation studies are detailed and informative.

### Weaknesses
- The intuition behind “uncertainty-aware” attention heads is entirely empirical and lacks theoretical grounding. There is no evidence that these attention patterns cause hallucinations rather than simply correlate with them.
- The main conceptual issue is inconsistency between motivation and method. The paper claims certain heads are consistently linked to factuality, yet the algorithm reselects new heads for each sequence based on current attention magnitudes. This per-sequence selection contradicts the idea of stable “uncertainty-aware” heads and weakens the mechanistic motivation.
- Although the authors claim that RAUQ requires no hyperparameter tuning, the balancing coefficient $\alpha$ is tuned on a held-out set, and Figure 5 shows that performance can significantly change with its value. Hence, this claim is overstated.
- Comparisons focus mainly on attention-based or sampling baselines. Recent approaches using mechanistic interpretability or calibration-based uncertainty estimation are not included.
- Minor: The method requires access not only to token probabilities but also to internal attention weights, restricting its broader applicability.

### Questions
- How stable are the selected attention heads across different sequences or inputs? Do the same heads tend to be selected consistently, or does the selection vary substantially? How do the authors justify selecting different heads per input sequence?
- Which sampling temperature are the output sequences generated with? Table 3 indicates a temperature of 1.0. Howerver, Aichberger et al. [1] show that the greedy output sequence should be used for computing the MSP to obtain the highest performance. It would be interesting to compare against this baseline as well.

---

[1] Lukas Aichberger, Kajetan Schweighofer, and Sepp Hochreiter. Rethinking uncertainty estimation in natural language generation. arXiv preprint arXiv:2412.15176, 2024.

### Soundness
3

### Presentation
3

### Contribution
3
