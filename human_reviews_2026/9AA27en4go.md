# Difference-Aware Retrieval Policies for Imitation Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Parametric imitation learning via behavior cloning can suffer from poor generalization to out-of-distribution states due to compounding errors during deployment. We show that reusing the training data during inference via a semi-parametric retrieval-based imitation learning approach can alleviate this challenge. We present Difference-Aware Retrieval Policies for Imitation Learning (DARP), a semi-parametric retrieval-based imitation learning approach that addresses this limitation by reparameterizing the imitation learning problem in terms of local neighborhood structure rather than direct state-to-action mappings. Instead of learning a global policy, DARP trains a model to predict actions based on k-nearest neighbors from expert demonstrations, their corresponding actions, and the relative distance vectors between neighbor states and query states. DARP requires no additional assumptions beyond those made for standard behavior cloning -- it does not require additional data collection, online expert feedback, or task-specific knowledge. We demonstrate consistent performance improvements of 15-46% over standard behavior cloning across diverse domains, including continuous control and robotic manipulation, and across different representations, including high-dimensional visual features.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Authors propose a way to combine retrieval based IL with standard BC that will turns out to be equivalent to manifold smoothed BC. Authors add both theoretical analysis and experimental validation in terms of standard MuJoCo IL experiments and ablation studies.

### Strengths
- Interesting, theoretically motivated and practical offline IL method. 
- MuJoCo and robosuite experimental results are quite good. 
- Narrative is well written and easy to follow.

### Weaknesses
- Even though narrative is well written, the mathematical exposition leaves room for improvement. For example looking at page 3, we see a lot of symbols that have not been defined: s^*, \pi(s_t), L, ... 
- Even though I understand that focus is in offline IL, it would be good obtain results from IL methods that have access to environment, such as adversarial methods (GAIL, AIRL etc). Can the proposed method bridge the gap between BC and GAIL? 
- It appears that expert demos have been obtained from an optimal RL policy, or have I been mistaken? In the "What Matters for Adversarial Imitation Learning?" -paper authors note that there is a difference when using expert demos from optimal policy and expert demos from human expert. It would be good to test with such demos, one option is to use ALE environment and with Atari-HEAD expert demo dataset.

### Questions
- How would you contrast the proposed method to the retrieval IL method presented in: Federico Malato, Ville Hautamäki, "Online Adaptation for Enhancing Imitation Learning Policies", CoG 2024?
- In Section 2.4.2 you show an interesting way to go beyond simple averaging, but in experiments I see no improvement in using it. Why is that? I would suggest not to bold DARP Set Transformer numbers in Table 1 as all of those numbers are worse than plain DARP.

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
5

### Summary
This paper proposes DARP, a new semi-parametric BC method that integrates non-parametric retrieval with parametric prediction. DARP retrieves a set of nearest-neighbor expert demonstrations for a given query state and conditions the policy’s action prediction on both the neighbor states and their difference vectors relative to the query. The retrieved neighbors are aggregated in a permutation-invariant manner, enabling the model to enforce implicit local smoothness without explicit regularization. The authors show theoretically that DARP approximates a Laplacian smoothing operation over the expert k-NN graph, promoting manifold-consistent policies. Empirical results demonstrate the improved stability, generalization, and robustness across various continuous-control tasks.

### Strengths
1. The core idea of reparameterizing the policy in terms of relative differences to known expert state-actions is highly intuitive. It anchors the policy's predictions to the ground-truth data manifold, providing a strong and data-centric form of 'regularization'.

1. The authors provide a strong theoretical motivation for their architectural choice. The connection drawn between neighbor aggregation (iMRIL) and explicit manifold regularization (MRIL) via spectral graph theory (Theorem 2) offers a compelling, first-principles explanation for why this method should reduce variance and improve stability, linking it to low-pass filtering on the graph Laplacian.

1. The experimental results are good, and the ablation is impressive and sufficient.

### Weaknesses
Overall, the proposed method is inspiring and technically sound, and the paper is easy to read. I only have concerns on the inference cost and the distance metric:

1. The method could suffer from large computational/storage overhead at inference time. For every single decision step, the policy must perform a k-NN search over the entire $N$-point expert dataset and then perform $k$ network forward passes. Also, the method requires storing a large expert dataset or its embeddings, which increases memory and storage requirements by orders of magnitude compared to a standalone policy network. 

1. The paper assumes that Euclidean distance (or a simple temporal extension with decay) suffices to identify meaningful neighbors, which strong in high-dimensional or structured state spaces, where Euclidean proximity does not necessarily imply functional similarity.

### Questions
1.  Have the authors analyzed the trade-off between inference latency and performance?

1. In the proof of Theorem 1 (iii), what does $r$ represent? There seems a loss of definition.

1. Can the method be extended to other distance metrics?

### Soundness
3

### Presentation
4

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
This paper presents DARP, a method to reduce variances in behavior cloning methods. It conditions the action generation using the retrieved nearest neighbors and aggregates the predictions from individual neighbors to output the final action. The experiments show the improvements over prior methods based on retrieved neighbors and BC on MuJoCo and robosuite tasks.

### Strengths
- The paper bridges the non-parametric retrieval-based method and the parametric BC to improve the policy performance.
- The paper provides good analysis, both theoretical and experimental, to show the effect of DARP.

### Weaknesses
- The main experiment result is based on MuJoCo tasks, which have a lot more neighbors than manipulation, so it is easier to show the improvements. For manipulation tasks, it is unclear which BC architecture it is employing. If it is an MLP, it is not a well-performing architecture for manipulation; it will need to compare with other BC architectures, like diffusion policy, to show the real improved performance.
- As shown in Appendix A2.1, the performance is significantly affected by the distance measure, e.g.,  the amount of look-back. This can show that the retrieval mechanism outweighs the architectural changes proposed in the paper. 
- The motivation of the paper is about improving generalization. However, none of the experiments evaluate the generalization performance of DARP.

### Questions
- The step of divergence analysis can be considered as slight evidence that DARP handles OOD better, but how often can we detect the divergence? Also, it seems like this analysis is only possible with deterministic policies. How can we apply the same analysis to the push-T task, where DARP uses a GMM head?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DARP, a retrieval-based imitation learning method that addresses covariate shift in behavior cloning by reparameterizing the problem using local neighborhood structure. The key innovation is conditioning action predictions on k-nearest neighbors along with their actions and crucially, the difference vectors between neighbor states and query states. The authors provide theoretical analysis showing DARP implicitly performs Laplacian smoothing and demonstrate 15-46% performance improvements across continuous control and robotic manipulation tasks.

### Strengths
- The use of difference vectors (s*_i - s_q) rather than just neighbor states is creative and well-motivated. The ablation in Figure 5 convincingly shows this is crucial for performance.
- The connection to Laplacian smoothing provides intuition for why the method works and bridges local and global learning paradigms effectively.
- The experiments span MuJoCo locomotion, robotic manipulation, visual observations, and even deliberately discontinuous environments, showing broad applicability.

### Weaknesses
-  The paper doesn't discuss the computational cost of k-NN retrieval at every training and inference step. For k=500 (as suggested in Figure 8), this could be prohibitive for large datasets or real-time applications.
- While the paper compares to BC and some retrieval methods, it lacks comparison with other smoothness-inducing approaches mentioned in related work (L2C2, CCIL). The MRIL baseline helps but isn't a published method.
- The jump from iMRIL (which uses simple averaging) to DARP (with difference vectors and learned aggregation) isn't theoretically justified. Does the Laplacian smoothing interpretation still hold?

### Questions
- What is the time complexity for training and inference? How does this scale with dataset size and k?
- How sensitive is DARP to the choice of distance metric, especially in high-dimensional spaces? Have you experimented with learned metrics?
-  Does the Laplacian smoothing analysis extend to the full DARP algorithm with difference vectors and parametric aggregation functions?

### Soundness
3

### Presentation
3

### Contribution
3
