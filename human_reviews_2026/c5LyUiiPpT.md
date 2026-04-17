# Understanding and Leveraging Expert-level Monosemanticity in Finetuning MoE LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2

## Abstract
Large language models (LLMs) with Mixture-of-Experts (MoE) architectures have emerged as a promising approach for enhancing scalability and efficiency, with minimal performance degradation across diverse downstream tasks. However, the interpretability of experts and efficient post-training methods of domain experts remain understudied. In this paper, we first analyze the expert-level monosemanticity of MoE based on the sparse autoencoder (SAE), thereby facilitating a deeper understanding of domain experts' roles. Additionally, leveraging the enhanced monosemanticity induced by the sparse activations of MoE LLMs, we propose a new fine-tuning strategy that freezes domain-agnostic experts in specific layers. Unlike dense LLMs, the sparsity of MoE enables experts to exhibit stronger expert-level monosemantic behavior, allowing us to identify experts responsible for particular downstream tasks and freeze those unrelated during post-training. By only updating domain-relevant experts, our method mitigates the risk of catastrophic forgetting in other domains and reduces computational costs. Empirically, we apply this strategy to supervised fine-tuning of MoE models on tool-use data. Results show that monosemanticity-guided tuning achieves performance comparable to fully-tuned models on tool-use tasks, while preserving better performance in other domains. Our study provides an interpretability-guided strategy for understanding and finetuning MoE LLMs while alleviating performance degradation across domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a framework to evaluate expert-level monosemanticity in Mixture-of-Experts (MoE) language models, inspired by SAE. It defines new metrics (EMS, EDMS, LDS, EDMRS) to quantify monosemanticity and uses them to guide selective fine-tuning of domain-relevant experts, aiming to preserve cross-domain generalization. Experiments on tool-use data suggest that monosemanticity-guided tuning achieves similar in-domain performance as full fine-tuning while reducing degradation on other domains.

### Strengths
1. The paper tackles an important and timely topic, interpretability and efficient adaptation of MoE LLMs.

2. The focus on explainability and expert-level interpretability is valuable and well-motivated, as it attempts to bridge the gap between activation analysis and human-understandable reasoning about experts.

### Weaknesses
1. The proposed metrics (EMS and EDMS) do not convincingly measure monosemanticity. They evaluate how accurately a separate LLM can predict expert activations from textual explanations, which captures predictability or consistency rather than semantic purity. Therefore, these scores cannot directly indicate whether an expert encodes a single coherent concept. All subsequent analyses and the EDMRS-based fine-tuning strategy are built upon this assumption, raising concerns about theoretical soundness.

2. The paper conflates semantic monosemanticity with functional capability. The concept of monosemanticity concerns the disentanglement of semantic features or conceptual meanings, but the chosen evaluation domain—tool use—reflects a procedural ability, not a semantic category. As a result, the study does not genuinely assess semantic interpretability, and further experiments across semantic or linguistic dimensions are necessary to substantiate the claim.

3. The reasoning in Insight 1 about sparsity and monosemanticity is weak. The authors claim that MoE’s sparse routing inherently increases monosemanticity compared to dense models, but they provide neither a controlled comparison with dense LLMs nor evidence of causality. The inference that “higher sparsity leads to stronger monosemanticity” is speculative and lacks rigorous support.

4. The methodological contribution is incremental. The framework remains fundamentally activation-based; it interprets standard activation statistics with natural-language summaries from an external LLM. While this adds a layer of explainability, it does not introduce new interpretability mechanisms or learning principles. The novelty lies primarily in presentation rather than substance.

5. The experiments are limited in scope and impact. The reported improvement (0.8%) on out-of-domain benchmarks is marginal, and the experiments cover only one task domain without ablation studies or statistical variance analysis. The results, while suggestive, are insufficient to establish the method’s effectiveness or general applicability.

### Questions
See weaknesses.

### Soundness
1

### Presentation
2

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
This paper studies interpretability and fine-tuning in Mixture-of-Experts (MoE) large language models through expert-level monosemanticity, where each expert represents a distinct concept. Inspired by sparse autoencoders, the authors propose quantitative metrics (EMS, EDMS) to measure monosemanticity and introduce a selective fine-tuning strategy based on the Expert Domain Monosemantic Responsibility Score (EDMRS). The method fine-tunes only domain-relevant experts, reducing interference across domains. Experiments on tool-use tasks show comparable in-domain results to full tuning while improving cross-domain generalization, offering an interpretable and efficient framework for MoE adaptation.

### Strengths
1. This paper presents a clear theoretical framework with formal derivations.

2. This paper demonstrates practical improvements in maintaining task performance and enhancing cross-domain generalization.

### Weaknesses
1. Limited experimental scope: Evaluation is restricted to a single tool-use task, lacking validation on more challenging language understanding or reasoning tasks.

2. Dependence on external LLM (GPT-4o) for explanation generation introduces potential evaluation bias and high computational cost.

3. Strong theoretical assumptions: The hypothesized linear relationship between EDMS and performance improvement is not empirically substantiated.

4. Missing comparisons with mainstream fine-tuning methods such as LoRA, Adapter, or Router-tuning.

5. Small performance gain (≈0.8%), which may not convincingly demonstrate general superiority.

### Questions
1. How stable are EMS and EDMS results when using different external LLMs?

2. Have the authors considered validating EDMRS effectiveness in multi-task or multi-domain fine-tuning scenarios?

3. Could a lighter-weight interpretability proxy model replace GPT-4o for explanation generation?

4. Could EDMRS be combined with parameter-efficient fine-tuning methods (e.g., LoRA or Adapter) to enhance generality?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the interpretability of experts (such as specialization for different domains) in the Mixture-of-Expert (MoE) architecture. To this end, authors analyse monosemantic representations in MoE experts. They find that increased sparsity in MoEs improves expert-level monosemantic representations, hence the specialization across downstream domains. Based on their analysis, the paper proposes a specialized supervised finetuning method for MoEs where "selected" experts are updated during finetuning. To select these experts for a domain, they propose using monosemanticity scores combined with activation frequency. Their results on a tool-use dataset show that while this method achieves comparable performance with full model finetuning, it maintains the performance in the other domains.

### Strengths
1. Interpretability analysis for the MoE expert is a highly important topic. Although recent LLMs are adapting the MoE architecture, expert specialization and exploration on this is commonly underexplored.

2. The paper proposes an evaluation framework for MoE models to examine monosemanticity. They use the output of the MoE gate, similar to the hidden representation in SAE.

### Weaknesses
1. Gaps in monosemanticity analysis:

a. It is not very clear how the monosemanticity evaluation framework is fundamentally different than analysing expert activation frequency on sampled datasets. 

b. The MoE models that were experimented with have been trained with a certain number of activated experts during their pretraining. Without having control over this, varying the activated expert during the test time may lead incomplete conclusion. For this, LLMs with different pretraining sparsity needs to be explored.
 
2. Experiments include only one dataset (BFCL v3, tool-use) as the ownstream domain. From this experiment, it is not possible say that the findings are generalizable across different domains or an inherent property of this particular domain. 

2. Results compared to the baselines are within 1%. Although this does not necessarily invalidate the results, statistical analysis and multiple runs with different seeds are required to assess the conclusion.

### Questions
1. Why do the finetuning experiments only include a single layer? 

2. The selected model is trained with 8 expert activated per token; any fine-tuning changing this strategy may give suboptimal results. Why were 16 experts selected to update?

### Soundness
2

### Presentation
2

### Contribution
2
