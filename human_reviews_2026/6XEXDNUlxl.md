# Layer-Aware Influence for Online Data Valuation Estimation

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Data-centric learning emphasizes curating high-quality training samples to boost performance rather than designing new architectures. A central problem is to estimate the influence of training sample efficiently. Prior studies largely focus on static influence measured on a converged model, overlooking how data valuation dynamically changes during optimization. This omission neglects the dynamic nature of sample influence during optimization, especially in deep models. To address the computational burden of frequent influence estimation, we develop a layer-aware online estimator that requires only loss-to-output gradients. This design avoids parameter-level and full-network gradients while preserving ranking fidelity. Extensive experiments across LLM pretraining, fine-tuning and image classification demonstrate that our method improves accuracy with substantially lower time and memory cost in both text and image datasets, making dynamic data curation both efficient and scalable in practice.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Layer-Aware Influence (LAI), an online, Hessian-free method for estimating per-sample influence during training. LAI approximates influence by leveraging multi-layer embeddings but only requires gradients at the output layer, thereby avoiding costly full-network backpropagation. The method is designed to enable dynamic data curation (e.g., filtering out detrimental samples) within a single training pass, integrated naturally with SGD-style optimizers. The authors evaluate LAI across diverse settings and demonstrate improvements in accuracy, training efficiency, and influence fidelity compared to baselines like Ghost Influence and static influence estimators.

### Strengths
1. LAI significantly reduces computational and memory overhead compared to prior online influence methods (e.g., Ghost Influence), making dynamic data valuation feasible for large models like LLMs.

2. The experiments are extensive and cover multiple modalities (text, vision) and training regimes (pretraining, fine-tuning). The fidelity analysis against Monte Carlo Shapley values is particularly compelling.

3. The insight to replace per-layer gradient feedback with a single output-layer signal while preserving multi-layer embedding information is clever and empirically effective.

### Weaknesses
1. Overreliance on Implicit Convexity Assumptions: Despite claiming to operate in deep non-convex settings, the theoretical grounding of LAI (and its predecessor, Ghost Influence) still stems from second-order influence frameworks originally derived under local convexity or smoothness assumptions (e.g., Koh & Liang, 2017). The paper does not adequately address how these assumptions break down in highly non-convex, high-dimensional landscapes typical of deep learning. This limits the theoretical justification for why the inner-product-based influence proxy remains meaningful throughout training. 

2. Moreover, during training, the key assumption in Koh et al.’s work, namely, the existence of a local stationary point. This is generally not satisfied. Consequently, their methodology is theoretically inapplicable within this framework. 


3. While Appendix A provides a noise propagation analysis, it assumes structured perturbations (e.g., linearized backprop operators, bounded activations). In practice, modern architectures (with LayerNorm, residual connections, attention) violate these assumptions, and the claim that LAI “reduces variance” lacks rigorous non-asymptotic guarantees in realistic settings. 

4. The experiment was too small in scale and lacked persuasiveness. Please provide the verification results under the Non-trivial experimental Settings of the large model training scenarios. 

5. Missing **Key Related Works** on Non-Convex / Optimization-Aware Influence: 

[a] Data Cleansing for Models Trained with SGD

[b] Data Pruning via Moving-one-Sample-out

[c].  Z0-Inf: Zeroth Order Approximation for Data Influence

### Questions
See Weakness. 

I will raise my rate if my concerns are well addressed.

### Soundness
2

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
This submission focuses on data-centric learning methods where the focus is on careful selection of training data samples instead of improving the architecture. The major challenge in such methods is the efficiency with which such methods can be designed. 
In this work, authors propose a layer-aware online estimator that avoids need for parameter-lever or full-network gradients. 
The method mainly focuses on online dynamic per-sample influence estimation, which is noticeably suffers from inverse Hessian computation. To that end, this work introduces layer-aware influence (LAI) estimator that uses a single, stable feedback channel.

### Strengths
Strengths:
- Unlike previous methods, this approach focuses on online data valuation 
- Computes online per-sample inflated during training and integrates it naturally with SGD-style updates
- Identifies the issues where use on inverse of Hessian matrix and oversight due to no-longer-be-deterimental samples for the trained model
- Focuses on beneficial-or-not approach instead of easy-to-hard (curriculum) learning approach
- Experiments are very carefully done for image and text classification

### Weaknesses
Weaknesses:
- It is not clear how existing approaches are different from importance-driven methods like Salaun et al. 2025a,b for data pruning or careful data selection
- The overall theoretical contribution of the work seems quite limited. However, the simple adjustments done writ Wang et al. 2024c  and making it layer-aware seems to bring noticeable improvements in both image and text classifications. This needs to be carefully explained.

### Questions
Questions: 
- Are influence functions different from importance functions (weights) used by existing literature?
- How to handle cases where multiple influence functions are applicable? Does it resembles Multiple importance Sampling methods from the probability statistics (Salaun et al. 2025b)?
- Salaun et al. 2025a work on online importance sampling using weights from the last layer to design importance weighting function. Are there any benefits such methods bring to sample influence estimation?

Online Importance Sampling for Stochastic Gradient Optimization, Salaun et al. 2025a
Multiple Importance Sampling for Stochastic Gradient Estimation, Salaun et al. 2025b

### Soundness
4

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
This paper proposes Layer-Aware Influence (LAI), a Hessian-free, online data valuation method that dynamically estimates per-sample influence during training without requiring full backpropagation or retraining. LAI leverages multi-layer embeddings but only computes gradients up to the output layer, significantly reducing computational and memory overhead compared to prior influence estimation methods like Ghost Influence. The approach is evaluated across diverse settings, including LLM pre-training, LLM fine-tuning, and image/text classification with noisy labels, and demonstrates consistent improvements in accuracy, validation loss, and training efficiency while maintaining high fidelity to Monte Carlo Shapley values.

### Strengths
Novelty and Practicality: LAI introduces a lightweight yet effective approximation to online influence estimation by avoiding per-layer per-sample gradients. This design choice addresses a critical scalability bottleneck in dynamic data curation, making it feasible for large models like LLMs.

Strong Empirical Validation: The method is rigorously evaluated across multiple modalities (text and vision), training regimes (pre-training and fine-tuning), and noise settings. Results consistently show that LAI outperforms both static influence methods and recent online baselines (e.g., Ghost Influence) in both performance and efficiency.

Theoretical Justification: The paper provides a clear bias-variance analysis showing that LAI reduces noise accumulation compared to Ghost Influence, explaining its superior empirical stability and fidelity. The derivation in the appendix further supports LAI as a principled approximation.

### Weaknesses
Validation Set Dependency: The method relies on a validation set for influence estimation, which may not always be available (e.g., in unsupervised pre-training). While the paper uses self-influence as a workaround, this approach’s robustness and generalizability remain underexplored. 


---

Lack of Comparison to Non-Convex, Non-Influence-Based Data Curation Methods: The paper does not compare against recent dynamic sample selection strategies that do not rely on influence functions or convex assumptions (e.g., JEST, ACID, or RHO-Loss variants), which are relevant baselines given the paper’s focus on online curation.

---

Missing Key Related Works on Non-Convex / Optimization-Aware Influence:

[a] Data Cleansing for Models Trained with SGD

[b] Data Pruning via Moving-one-Sample-out

[c]. Z0-Inf: Zeroth Order Approximation for Data Influence

---

This method inherits the formulation from classical Koh's influence functions, which rely on assumptions such as local convexity, smoothness, and the existence of a well-defined stationary point. This represents a major theoretical flaw in the paper, because such a condition rarely holds in deep learning.

The paper does not rigorously analyze how or whether these assumptions hold during training, nor does it provide alternative theoretical grounding for why the inner-product-based influence proxy remains valid in such settings.

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2
