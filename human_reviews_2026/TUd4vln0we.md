# PerturbFormer: Adversarial Graph Transformers for Scalable and Resilient Representation Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 8, 2, 4

## Abstract
We introduce PerturbFormer, a unified framework for node-level representation learning that addresses three persistent limitations in modern graph models: transformer attention degradation under low homophily, vulnerability to structural perturbations, and the high cost of large-scale inference. PerturbFormer integrates multi-scale structural synthesis with contrastive pretraining to produce geometry-aware embeddings, a heterophily-adaptive transformer backbone guided by learned structural cues, and an end-to-end adversarial propagation module where a generator proposes plausible edge modifications while a discriminator maintains semantic consistency. A node-confidence-weighted residual correction further adjusts propagation strength at fine granularity and enables practical contractivity controls for stable iterative refinement. The combined design enhances robustness and predictive quality on both homophilous and heterophilous benchmarks while keeping parameter and runtime costs competitive. Practical guidelines and implementation details are included to support effective application of the framework.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes PerturbFormer, a transformer-based framework for graph learning that integrates multi-scale structural embeddings, contrastive pretraining, and adversarial propagation. By combining degree-normalized attention with generative perturbations, it achieves robust, efficient, and state-of-the-art performance across diverse graph tasks.

### Strengths
1. This paper proposes PerturbFormer, a new graph learning framework capable of handling multiple tasks, demonstrating good applicability across various domains.
2. The paper provides comprehensive experimental analyses, including evaluations on different tasks, ablation studies, efficiency assessments, and insightful visualizations.

### Weaknesses
1. The first major weakness of this paper lies in its writing quality. The manuscript is poorly written, and I suspect that the use of LLMs goes beyond typos and grammar (a review request has been submitted). 
   (1) The Introduction fails to clearly convey the motivation and logical foundation of the proposed methods, while it lists up to six “limitations” the paper aims to address—many of which are exaggerated, such as the distillation limitation that is never discussed later.  (2) The Related Work section enumerates six points, but each is overly simplistic and resembles LLM-generated text. (3) In the Methodology section, the authors merely stack several modules without explaining their interconnections, and even use terms that sound machine-generated, such as “residual refinement” vs. “residual correction” or “feature extraction” vs. “feature synthesis,” which severely hampers readability. (4) Figure 1 provides little informative content and looks like an LLM generated it.

2. The PerturbFormer framework appears to be a combination of multiple existing modules and techniques (e.g., multi-resolution representation, contrastive alignment, adversarial propagation), yet their relationships are unclear. The approach PerturbFormer lacks novelty.

3. The experimental setup is confusing. For instance, PCQM4Mv2 is a graph regression dataset, but the paper incorrectly describes it as a node classification task. Table 3 lists datasets without any description or references.

4. The reported results are hard to believe: PerturbFormer achieves the best performance on all tasks and datasets, yet no standard deviation over multiple runs is provided, and no code for reproducibility.

5. Given that PerturbFormer integrates multiple modules, an analysis of its computational complexity is essential. However, no theoretical discussion is offered, and the results in Table 8 are unclear—comparing runtime across different methods and datasets is not meaningful.

Overall, the paper is logically inconsistent and poorly written, the proposed method lacks novelty, and there is concern that the use of LLMs in the writing and content generation exceeds acceptable limits, without any disclosure.

### Questions
Please respond to the Weaknesses part.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses key challenges in graph-based semi-supervised learning, including heterophily, robustness to structural noise, and scalability. It proposes PerturbFormer, an integrated framework that synthesizes multi-scale feature augmentation, contrastive self-supervised pretraining, generative adversarial networks, and optimized transformer architectures. The methodology involves enhancing node representations through multi-hop structural embeddings, adversarial contrastive pretraining, and a Graphormer with degree-normalized attention. Core contributions include a novel adversarial propagation mechanism for dynamic perturbation synthesis and confidence-weighted residual refinement. Experimental evaluation on benchmarks demonstrates improved classification accuracy and robustness compared to state-of-the-art baselines.

### Strengths
1.	The framework is novel as it combines feature synthesis, contrastive alignment, and GAN-based perturbations to enhance robustness.
2.	Convergence guarantees for residual propagation are provided.
3.	The paper is easy to follow.

### Weaknesses
1.	Claims of outperforming prior work in low-homophily settings are not rigorously justified with comparative analysis; evidence is limited to benchmark results in Table 2 without discussing fundamental gaps.
2.	The GAN-based perturbation lacks analysis of critical issues like model training stability. There is no experimental curves or metrics on adversarial optimization progress.

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors summarize six problems in the graph learning domain. To address these issues, they propose PerturbFormer, which integrates multiple techniques (including contrastive learning, attention mechanisms, and adversarial machine learning). In the experiments, the authors report that it demonstrates strong performance.

### Strengths
- The field of graph learning is a valuable area.
- The paper utilizes large-scale graph datasets for evaluation.

### Weaknesses
- The writing should be substantially improved, as it is very challenging to follow. The subsequent weaknesses stem from this aspect.
- The motivation presented in the paper appears somewhat disorganized. The authors introduce SIX problems in Section 1, but these lack clear organization and detailed explanations.
- The proposed method seems to simply combine existing techniques without introducing significant novelty.
- The writing and clarity of formulas in Section 3 should be enhanced. Additionally, I strongly recommend that the authors provide an overview diagram of the method.

### Questions
Please see the weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PerturbFormer, a graph representation learning framework that combines many components, such as multi-scale structural embeddings, contrastive pretraining, a heterophily-aware Graphormer and an adversarial propagation mechanism. The core idea is to jointly train a generator that perturbs the graph topology and a discriminator that enforces semantic consistency in node representations. An adaptive residual correction module further refines predictions using node-level confidence scores. The authors evaluate PerturbFormer on a wide range of  datasets and tasks. The results are powerful.

### Strengths
1. The paper effectively integrates many components into a single end-to-end framework.
2. This paper is well-organized and easy to follow.
3. The results show strong performance.

### Weaknesses
1. The motivation is mentioned briefly but remains underdeveloped. Neither the Abstract nor the Introduction sufficiently clarifies the problem’s significance or the specific gap, and the narrative leans toward describing components.
2. It remains unclear how much each component contributes to the reported gains. The current ablation (Table 4) removes modules singly and therefore does not assess interaction effects.
3. While the method is described in detail, key hyper-parameters are not reported.

### Questions
1. In Table 5, PerturbFormer shows remarkable robustness. Is this primarily due to the GAN regularizer, the confidence mechanism, or their combination?
2. The contraction condition (Eq. 26) assumes $\lVert \tilde{A}_{\tau} \rVert_{2} < 1$, but Appendix C shows this often fails in heterophilous graphs. How to address this in practice? Do you apply spectral normalization, and if so, does it degrade performance on homophilous graphs?

### Soundness
2

### Presentation
2

### Contribution
3
