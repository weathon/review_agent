# Attributing Response to Context: A Jensen–Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 8

## Abstract
Retrieval-Augmented Generation (RAG) leverages large language models (LLMs) combined with external contexts to enhance the accuracy and reliability of generated responses. However, reliably attributing generated content to specific context segments, context attribution, remains challenging due to the computationally intensive nature of current methods, which often require extensive fine-tuning or human annotation. In this work, we introduce a novel Jensen–Shannon Divergence driven method to Attribute Response to Context (ARC-JSD), enabling efficient and accurate identification of essential context sentences without additional fine-tuning, gradient-calculation or surrogate modelling. Evaluations on a wide range of RAG benchmarks, such as TyDi QA, Hotpot QA, and Musique, using instruction-tuned LLMs in different scales demonstrate superior accuracy and significant computational efficiency improvements compared to the previous baselines. Furthermore, our mechanistic analysis reveals specific attention heads and multilayer perceptron (MLP) layers responsible for context attribution, providing valuable insights into the internal workings of RAG models and how they affect RAG behaviours.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes ARC-JSD for RAG, which quantifies the Jensen–Shannon Divergence between response distributions generated with full and sentence-ablated contexts. This divergence serves as a reliable signal for identifying which context sentences most influence model outputs. The authors demonstrate that ARC-JSD achieves around 10% higher attribution accuracy and up to threefold computational efficiency compared to existing baselines.

### Strengths
1.The writing is fluent and the proposed method is easy to understand.

2.The theoretical analysis is well-developed, effectively explaining why JSD can serve as a valid metric for attribution analysis.

3.The experimental evaluation is comprehensive, demonstrating both the efficiency and effectiveness of the proposed approach.

### Weaknesses
1.Lack of clear definition. The paper does not appear to provide a formalized definition of the entire response distribution; while the token-level distribution is intuitive, the implementation details for computing the distribution over the full generated response should be clarified.

2.Limited model diversity: Although the evaluation is comprehensive, the models used are primarily from the Qwen and Gemma families at relatively small scales. It is recommended to include other representative open-source architectures such as LLaMA, and to analyze larger-scale models (e.g., 70B parameters) for more generalizable insights.

### Questions
1.How does this approach address real-world complex queries where accurate responses typically require the integration of multiple context segments?

2.How is the granularity of context segmentation determined, and what impact does this granularity have on the attribution performance?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the critical challenge of context attribution in RAG, which requires identifying specific context segments that ground generated responses.  The authors propose ARC-JSD, a novel inference-time method driven by Jensen–Shannon Divergence that avoids additional fine-tuning, gradient computations, or surrogate models.  Evaluations on different datasets using instruction-tuned LLMs demonstrate an effecticve average accuracy improvement and up to 3-fold speedup over baselines.  The work contributes a lightweight, integrable attribution method, insights into RAG’s internal workings, and a practical hallucination mitigation strategy.

### Strengths
1. ARC-JSD directly addresses a core limitation of existing context attribution methods by eliminating the need for fine-tuning, surrogate modeling, or gradient computations, thereby enabling robust and lightweight RAG context attribution.

2. The authors validate ARC-JSD across three diverse QA benchmarks (single-hop, multi-hop, long-context) and four LLM scales, demonstrating consistent improvements in accuracy and computational efficiency. The compute-accuracy trade-off analysis (Pareto optimality) and GPT-4.1 validation of response semantic equivalence (99.3% accuracy) strengthen the results’ credibility.

3. The method’s ability to reduce hallucination by 39% while maintaining factual F1 makes it immediately applicable to real-world RAG systems.  Its modular design allows seamless integration into existing pipelines, enhancing transparency and trustworthiness.

### Weaknesses
1. It is recommended that the authors further clarify the core differences between the proposed method and conventional Jensen–Shannon Divergence–based approaches, in order to highlight the novelty of their method.

2. It is recommended that the authors further clarify whether the proposed method is applicable to scenarios involving longer contexts. Does the mechanism of attribution differ when dealing with long-context settings?

3. The authors conducted experiments on several multi-hop QA datasets; however, they did not analyze whether the attribution mechanisms in multi-hop tasks differ from those in single-hop tasks.

4. The authors primarily conducted experiments on general-domain datasets. In domains with higher demands for interpretability and attribution, such as medical or legal tasks, where a large number of domain-specific terms are present, is the proposed method still applicable?

### Questions
1. Based on the experimental results, the proposed method achieves better performance on models with larger parameter scales. Is model size an explicit factor influencing the RAG attribution mechanism?

2. How does the proposed method perform in attribution when the retrieved documents contain conflicting or distracting information, or when relevant content is not retrieved at all? Could the authors provide corresponding experiments or analytical studies to address this scenario?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes ARC-JSD, a JSD-based, training-free attribution method for RAG, plus a logit-lens mechanistic analysis. Appendix materials add examples, heatmaps, and metric comparisons. Despite a neat idea, the evidence remains brittle and in places self-contradictory.

### Strengths
1. ARC-JSD operates purely through forward inference without additional fine-tuning or gradients, making it computationally simple and easy to integrate into existing RAG setups.
2. The paper evaluates on multiple public QA datasets (TyDi QA, Hotpot QA, MuSiQue), showing consistency across different model sizes and architectures.
3. Combining JSD-based attribution with logit-lens analysis reflects an effort to connect interpretability metrics to internal LLM components like attention heads and MLPs.

### Weaknesses
1. Reported results contain inconsistencies (e.g., invalid correlation values) and lack human or causal validation, which undermines confidence in the findings.
2. Weak Baselines and Limited Scope: Comparisons exclude modern, strong RAG baselines (hybrid retrieval, reranking, RePlug, HyDE, Self-RAG, etc.), and experiments use only small models, limiting generalizability.
3. Factual and Attribution Errors: Appendix examples reveal incorrect reasoning and noisy top-1 attributions (e.g., false album claim for Bieber example), suggesting that ARC-JSD may highlight irrelevant or misleading evidence.

### Questions
1. Include human evaluations of attribution correctness and apply causal interventions (e.g., activation patching or ablation) to verify that high JSD layers or sentences truly influence model outputs.
2. Compare against stronger modern RAG systems (hybrid retrieval, reranking, HyDE, RePlug, trained Self-RAG) and test on larger models and diverse domains to demonstrate robustness and real-world applicability.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a training-free method for context attribution. It measures how much the model’s output distribution changes when a context sentence is removed, using Jensen–Shannon Divergence (JSD) as a symmetric, bounded signal of attribution strength. The proposed method is computationally efficient compared to previous methods. Moreover, the paper performs a mechanistic analysis combining JSD and Logit Lens probing to identify attention heads and MLP layers responsible for context grounding.

### Strengths
- Paper is well organized, easy to follow. 
- The proposed method is computationally efficient and performs well. 
- Mechanistic interpretibility brings insight into the problem.
- The choice of JSD is justified experimentally, and a discussion on other metrics is included in the appendix.

### Weaknesses
- Multiple sentences might be required to generate a response. A multiple-sentence outputting scenario would require a thresholding mechanism, which would be done using a calibration set. An evaluation of the robustness of threshold selection (calibration set size, distribution shift, output set coverage) is important for real-life scenarios. 
- The mechanistic study is not causal. It shows insight; however, does not imply causality. Causal ablation studies would strengthen the claims.

### Questions
- Currently, while calculating the JSD score, all response tokens are treated equally. However, the importance of the tokens might be different. For example, we would not care about the distribution change of the token "the", but the token "two" in the mosquito question is more essential. How would weighting the tokens affect the method's performance? Or plotting the distribution of the JSD scores of each token would also provide insight into how the method works and its effectiveness.

### Soundness
3

### Presentation
4

### Contribution
3
