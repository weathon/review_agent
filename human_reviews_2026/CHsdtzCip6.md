# Energy-Regularized Sequential Model Editing on Hyperspheres

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 8

## Abstract
Large language models (LLMs) require constant updates to remain aligned with evolving real-world knowledge. Model editing offers a lightweight alternative to retraining, but sequential editing that updates the LLM knowledge through multiple successive edits often destabilizes representations and induces catastrophic forgetting. In this work, we seek to better understand and mitigate performance degradation caused by sequential editing. We hypothesize that hyperspherical uniformity, a property that maintains uniform distribution of neuron weights on a hypersphere, helps the model remain stable, retain prior knowledge, while still accommodate new updates. We use Hyperspherical Energy (HE) to quantify neuron uniformity during editing, and examine its correlation with editing performance. Empirical studies across widely used editing methods reveals a strong correlation between HE dynamics and editing performance, with editing failures consistently coinciding with uncontrolled HE fluctuations. We further theoretically prove that HE dynamics impose a lower bound on the degradation of pretrained knowledge, highlighting why HE stability is crucial for knowledge retention. Motivated by these insights, we propose SPHERE (Sparse Projection for Hyperspherical Energy-Regularized Editing), an HE-driven regularization strategy that stabilizes neuron weight distributions, ultimately preserving prior knowledge while enabling reliable sequential updates. Specifically, SPHERE identifies a sparse space complementary to the principal hyperspherical directions of the pretrained weight matrices and projects new knowledge onto it, attenuating perturbations on the principal directions. Extensive experiments on LLaMA3 (8B) and Qwen2.5 (7B) show that SPHERE outperforms the best baseline in editing capability by an average of 16.41%, while most faithfully preserving general model performance, thereby offering a principled path toward reliable large-scale knowledge editing.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses catastrophic forgetting in Large Language Models (LLMs) during sequential editing. The authors propose a core hypothesis: hyperspherical uniformity of model weights is key to stability and knowledge retention. To validate it, they first use Hyperspherical Energy (HE) to quantify this uniformity, and show strong correlation between HE fluctuations and editing failures. They theoretically prove that changes in HE impose a lower bound on output perturbation, providing theoretical support for HE stability. Based on these findings, they propose SPHERE (Sparse Projection for Hyperspherical Energy-Regularized Editing). SPHERE identifies principal hyperspherical directions in pretrained weights and projects new knowledge updates into a complementary sparse subspace, aiming to inject new information without disturbing the model's core knowledge structure.

### Strengths
Strength 1: The main contribution of this paper lies in providing a novel geometric perspective—the disruption of "hyperspherical uniformity"—for understanding model editing failures (catastrophic forgetting), offering a plausible explanation for why models collapse.

Strength 2: The proposed method demonstrates clear advancement over existing mainstream approaches, showing strong scalability and promising potential.

### Weaknesses
Weakness 1: Since the method involves principal component analysis, applying it to larger models will incur significant computational overhead, yet the paper lacks relevant discussion and analysis on this issue.

Weakness 2: The paper lacks analysis and experiments regarding the settings of hyperparameters.

### Questions
Q1: Related to Weakness2, In particular, I noticed that different parameter choices are used for AlphaEdit and other editing methods, why?

Q2: Have the authors considered leveraging the theoretical lower bound of HE change to proactively estimate a model's potential editing capacity, or in other words, its editing limit?

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
### Summary

This paper addresses performance degradation in sequential model editing, where successive updates to a large language model often lead to catastrophic forgetting. The authors hypothesize that this failure is linked to the disruption of "hyperspherical uniformity", the geometric distribution of neuron weights, which they quantify using Hyperspherical Energy. 
They provide empirical evidence of a strong correlation between HE fluctuations and editing failures and offer a theoretical proof that HE changes establish a lower bound on knowledge degradation. Based on this, they propose SPHERE, a regularization strategy that preserves stability by identifying the principal directions of pretrained weights and projecting new knowledge updates onto a complementary sparse space. Experiments demonstrate that SPHERE significantly improves editing capability over strong baselines while better preserving the models' general abilities.



### Advantages

* The paper provides a novel and intuitive analysis connecting the geometric concept of hyperspherical uniformity with the practical problem of catastrophic forgetting in sequential editing.
* The authors support their empirical observations with a theoretical proof establishing that changes in Hyperspherical Energy create a lower bound for the degradation of pretrained knowledge.
* The proposed SPHERE method not only outperforms existing baselines in large-scale sequential editing but also functions as a plug-and-play enhancement that significantly boosts the performance of other editing techniques.




### Drawbacks and Questions

* The proposed method relies on estimating the principal hyperspherical directions of the weight matrices, which involves an eigendecomposition that may introduce computational overhead not fully analyzed in the paper.
In this case, would it be possible to conduct an experiment analyzing the computational cost (e.g., latency or FLOPS) of applying the SPHERE projection compared to the base editing methods, particularly as the model size or the number of edits increases?



* The effectiveness of the sparse space projection appears dependent on two key hyperparameters, the cumulative ratio $\eta$ and the suppression strength $\alpha$, but the paper provides limited ablation studies on how sensitive the model's performance is to different values for these parameters.
Thus, could the authors provide an additional ablation study showing how editing performance (Efficacy and Specificity) varies across a wider range of $\eta$ and $\alpha$ values to better understand this sensitivity?



* The theoretical analysis linking HE changes to output perturbation relies on simplifying assumptions, such as "orthonormal inputs" and "small perturbations," which may not fully capture the dynamics of real-world editing scenarios.
Thus, could an empirical experiment be designed to measure the orthonormality of the actual input key vectors ($K_0$) or the magnitude of perturbations ($\Delta W$) in practice, to assess how well these assumptions hold during sequential editing?

### Strengths
Please see above

### Weaknesses
Please see above

### Questions
Please see above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper focuses on catastrophic forgetting in sequential knowledge editing for LLMs. The authors argue that maintaining the hyperspherical uniformity of the model’s internal weights is key to balancing the preservation of pretrained knowledge with the integration of new edits. They employ hyperspherical energy to quantify weight uniformity and use this metric to develop HE-driven regularization strategies that stabilize the editing process and mitigate forgetting.

### Strengths
- The paper provides both empirical and theoretical analyses showing how the hyperspherical uniformity of an LLM’s internal weights affects sequential knowledge editing—a novel and valuable perspective for this problem.
- The paper introduces SPHERE, an HE-driven regularization strategy that stabilizes neuron weight distributions, thereby preserving prior knowledge while enabling reliable sequential updates.

### Weaknesses
- The AlphaEdit baseline appears much lower than its reported results; please reconcile this via implementation details, protocol alignment, multi-seed statistics, and a brief reproduction-gap analysis (ideally with official code).
- Please clearly articulate how SPHERE differs from AlphaEdit conceptually, algorithmically.

### Questions
Please see weaknesses

### Soundness
3

### Presentation
4

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
This paper tackles the problem of performance degradation in sequential model editing. The authors propose that editing failures like catastrophic forgetting are linked to the disruption of the hyperspherical uniformity of neuron weights. They use Hyperspherical Energy or HE to quantify this uniformity and provide both empirical and theoretical evidence linking HE instability to knowledge degradation. Based on this they introduce SPHERE a regularization strategy. SPHERE works by identifying the principal directions of the pretrained weight matrix and then projecting the edit updates onto the orthogonal complement of this space. This approach aims to integrate new knowledge while minimizing disruption to the models original weight geometry. Experiments on LLaMA3-8B and Qwen2.5-7B show that SPHERE outperforms existing methods in long-sequence editing particularly in preserving the models general abilities.

### Strengths
1. The core idea of linking sequential editing stability to hyperspherical uniformity is a novel and insightful contribution. It provides a new and intuitive explanation for why models collapse during repeated edits.
    
2. The paper does a very good job of supporting this central hypothesis. Its not just an idea it's backed by strong empirical correlations and a formal theoretical analysis that bounds knowledge degradation by HE dynamics.
    
3. The SPHERE method itself is elegant. The strategy of projecting updates into a sparse space complementary to the principal weight directions is a clean and direct implementation of the papers main hypothesis.
    
4. The experimental results are impressive. SPHERE shows a clear advantage in large-scale sequential editing and very importantly in preserving the models general abilities which is a common failure point for other methods. The fact that it also works as a plug-and-play enhancement is a significant practical strength.

### Weaknesses
1. The link between the HE motivation and the method feels loose. HE is defined by pairwise angular relationships but the method identifies its space using a variance approach in Eq 10 that looks just like PCA. It's not obvious why high-variance directions are the most important ones for preserving HE.
    
2. The paper needs to better distinguish its novelty from AlphaEdit. AlphaEdit also uses null-space projection. The conceptual advantage of projecting away from weight directions versus previous knowledge directions isn't fully explored.
    
3. I'm concerned about the SVD scalability. The paper doesn't discuss the cost. Is it a one-time computation? If so does that original principal space remain the correct one to avoid after thousands of edits?
    
4. The method introduces $\eta$ and $\alpha$ but lacks a sensitivity analysis. The values seem set arbitrarily for example $\alpha$ is 0.5 for AlphaEdit but 0.8 for others. This makes it hard to judge robustness.

### Questions
1. Can you elaborate on the SVD cost? Is it a one-time cost? I'm wondering if the original models principal directions are still the most important ones to preserve after 10000 edits.
    
2. Why is preserving the principal directions of the weight matrix the right way to stabilize HE? I'm trying to square this variance-based method with the angular-distance-based HE metric.
    
3. How did you select the $\alpha$ and $\eta$ values? The performance seems like it would be sensitive to these. The choice of $\alpha$ is different for different baselines when you use SPHERE as a plug-in. Could you provide an ablation on this?

### Soundness
3

### Presentation
4

### Contribution
3
