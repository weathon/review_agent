# MASS: MoErging through Adaptive Subspace Selection

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Model merging has recently emerged as a lightweight alternative to ensembling, combining multiple fine-tuned models into a single set of parameters with no additional training overhead. Yet, existing merging methods fall short of matching the full accuracy of separately fine-tuned endpoints. We present MASS (MoErging through Adaptive Subspace Selection), a new approach that closes this gap by unifying multiple fine-tuned models while retaining near state-of-the-art performance across tasks. Building on the low-rank decomposition of per-task updates, MASS stores only the most salient singular components for each task and merges them into a shared model. At inference time, a non-parametric, data-free router identifies which subspace (or combination thereof) best explains an input's intermediate features and activates the corresponding task-specific block. This procedure is fully training-free and introduces only a two-pass inference overhead plus a ~2 storage factor compared to a single pretrained model, irrespective of the number of tasks. We evaluate MASS on CLIP-based image classification using ViT-B-16, ViT-B-32 and ViT-L-14 for benchmarks of 8, 14 and 20 tasks respectively, establishing a new state-of-the-art. Most notably, MASS recovers up to ~98% of the average accuracy of individual fine-tuned models, making it a practical alternative to ensembling at a fraction of the storage cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MASS (MoErging through Adaptive Subspace Selection), a novel model merging approach, and its core premise is that in practical applications, the task identity is often unknown at inference time. To tackle this more challenging and realistic scenario, MASS introduces an adaptive routing mechanism. This mechanism, which is free of additional training and data, dynamically selects the most relevant task subspace for each input sample by calculating the projection residuals of the input's features onto different task subspaces. Experimental results demonstrate that MASS achieves state-of-the-art performance on several benchmarks.

### Strengths
1.	A Critical and Practical Problem Setting: The paper astutely identifies and challenges an unrealistic assumption common in existing model merging literature: that task identity is known at inference time. By proposing a more realistic scenario where task identity is unknown, the authors elevate the practical relevance of their research and open up a more meaningful direction for the community. MASS provides a complete and effective end-to-end solution for this setting.
2.	A Novel and Efficient Training-Free Routing Mechanism: A major highlight of this work is its projection-based router. This process is entirely free of additional labeled data and training overhead, which aligns perfectly with the lightweight and training-free philosophy of model merging. And it performs on par with MLP-based routers that require additional data and training.

### Weaknesses
1.	Limited Contribution: The paper's core framework is heavily built upon prior work, particularly Task Singular Vectors (TSV). Consequently, the core contribution of this paper can be viewed as the addition of an efficient, dynamic "selector" module on top of the TSV framework. While this selector is well-designed and effective, it functions more as a task scheduling or selection component rather than a fundamentally paradigm for “adaptive”  model merging. This makes the overall contribution of the paper feel somewhat incremental.
2.	Insufficient Experimental Scope: The paper's experiments are primarily focused on ViT-based vision classification tasks and NLU tasks. However, a more challenging and highly relevant application for model merging is on large language models (LLMs) performing difficult generative tasks (e.g., code generation, long-form text generation). For such tasks, task vectors often exhibit more complex, high-rank structures. Previous research[1, 2, 3] has questioned that simply relying on low-rank decomposition may not perform optimally when dealing with these high-rank task vectors. Given that MASS's performance is highly dependent on the effectiveness of TSV, its performance on LLMs and generative tasks remains a critical, unverified question. The absence of these experiments raises questions about the generalizability of MASS and its effectiveness on cutting-edge applications.
[1] Delta-CoMe: Training-Free Delta-Compression with Mixed-Precision for Large Language Models
[2] Knowledge Grafting of Large Language Models
[3] Delta Decompression for MoE-based LLMs Compression

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes applying model merging in a more challenging scenario, where the optimal encoder subspace and classification head for the current task are unknown. Under this setting, the authors introduce MASS, which predicts by selecting the most relevant series of encoders within the subspace. The method’s superiority over baselines is validated across tasks of varying scales.

### Strengths
- The authors provide detailed experimental results for analysis.
- The MASS method is simple yet effective.

### Weaknesses
- The description in Section 3.2.1 is somewhat unclear. Since minimizing $r_i$ yields the optimal $L_2$ projector, the proj in L191–192 should not be referred to as “the optimal $L_2$ projector.” Additionally, in line 195, what does “the residual” refer to—is it $r_i$?
- My main concern lies in the reasonableness of the paper’s setting. 
    - Given that models have already been fine-tuned on each task, why is it necessary to assume that the task-optimal encoder and classification head are unknown? It is explicitly known which task each model has been fine-tuned on. 
    - The authors should impose stricter constraints on the models involved in merging to make the setting reasonable.
    - Otherwise, the authors should clarify this point; otherwise, the foundation of the paper is difficult to justify.
- Line 406 mentions “prior work suggests that mid-layer embeddings in CLIP-like models...”; a citation should be added here.

### Questions
Please revise or clarify the description in Section 3.2.1.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a training‑ and data‑free way to merge multiple fine‑tuned models into a single network that selects relevant encoder subspaces per input and also selects the appropriate classification head when the task is unknown at test time. Concretely, each per‑task weight update $\Delta_i$ is SVD‑decomposed and truncated; at inference, an input’s mid‑layer features $z_\ell$ are projected onto each task’s right‑singular subspace and the residual $r_i=|z_\ell - V_i V_i^\top z_\ell|_2^2$ scores task relevance. The router keeps top‑K tasks above a softmax‑based threshold and adaptively merges the corresponding low‑rank updates; a second pass runs the selected heads and takes the max‑logit across them.

### Strengths
1. It Introduces a projection‑based, training‑free router in weight space that operates on TSV‑organized task subspaces and extends routing to head selection when the task is unknown. The MAP view provides a statistical rationale. 
2. Algorithms and figures make the method easy to follow. 
3. It Provides constant 2× storage independent of task count.

### Weaknesses
1. Routing layer and thresholds are selected with labeled validation accuracy as Appendix C. This partially contradicts the “data‑free” framing and may inflate reported gains versus truly label‑free procedures. 
2. Alg. 1 uses $w=\mathrm{softmax}(-r)$ only for selection; the merge $\sum_{i\in\Omega}U_i\Sigma_iV_i^\top$ is uniform, which miss a straightforward weighted variant $\sum w_i\Delta_i$. An ablation could show whether weighting reduces interference or improves tail tasks. 
3. The fixed merge (Alg. 2) uses orthogonalized bases $U_\perp,V_\perp$. It is unclear whether the adaptive merge applies the same orthogonalization restricted to $\Omega$. Clarify and ablate. 
4. MoE baselines are adapted with a majority‑vote head heuristic. Provide companion results with oracle heads (as in their original setting) and/or head‑logit calibration (e.g., temperature scaling) to separate routing vs. head‑selection effects. 
5. Euclidean residual assumes isotropy; unlabeled whitening/Mahalanobis distances on $z_\ell$ could be stronger and still data‑free, which relaxes Proposition 3.1’s assumptions. So it would be better to report sensitivity. 
6. Cosine similarity of vec($\Delta$) ignores subspace geometry. Principal angles between spans $(V_i)$ would be more principled. Provide ablation. 
7. No seeds/variance or CIs are given; wall‑clock overhead of two passes and multi‑head evaluation is not quantified; GLUE per‑task wins/losses are not detailed.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces **MASS**, a novel approach for model merging. It targets the problem of merging multiple fine-tuned models into a single, shared model while retaining performance comparable to that of individually fine-tuned models, without the need for additional training or data. MASS leverages low-rank decomposition and adaptive routing mechanisms to select the most relevant task-specific subspaces for each input, making it an efficient solution for multitask learning scenarios. Experimental results show that MASS achieves near-state-of-the-art performance on various benchmarks while reducing storage requirements compared to traditional ensemble methods.

### Strengths
1. **Innovative Concept:** The adaptive routing mechanism for task-specific subspace selection and the use of low-rank decomposition is an interesting approach to reducing storage requirements in model merging.

2. **Practical Application:** The method’s ability to merge models without additional data or retraining is a valuable contribution, especially for scenarios involving pre-trained models from public repositories.

3. **Strong Experimental Results:** MASS performs well across multiple vision and language benchmarks, surpassing existing methods in accuracy and storage efficiency.

4. **Efficiency:**  MASS performs well with a moderate increase in storage compared to a single model, providing a storage-efficient alternative to traditional ensembling methods.

### Weaknesses
1. **Generalization to Unseen Tasks:** The approach seems to focus on tasks seen during fine-tuning, but its ability to generalize to unseen tasks or tasks that require more nuanced adaptations is unclear. The task subspaces may not adequately cover unseen task distributions, affecting its robustness in dynamic or changing environments.

2. **Scalability to Larger Task Sets:**  While MASS performs well on 8–20 tasks, its scalability to a larger number of tasks is unclear. As the task set increases, the complexity of the routing mechanism may grow exponentially, making the method less feasible for large-scale applications.

3. **Inference Overhead:** The two-pass inference process introduces additional overhead, which could be problematic in real-time applications where latency is critical.

### Questions
1. How does the two-pass inference process impact latency, especially in time-critical applications?
2. How can the method be generalized to handle tasks that were not part of the fine-tuning phase?
3. As the number of tasks increases, how does the method maintain performance and efficiency?

### Soundness
3

### Presentation
3

### Contribution
1
