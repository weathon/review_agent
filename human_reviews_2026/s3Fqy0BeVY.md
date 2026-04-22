# Bridging Interpretability and Optimization: Provably Attribution-Weighted Actor–Critic in Reproducing-Kernel Hilbert Spaces

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Actor--critic (AC) methods are a cornerstone of reinforcement learning (RL) but offer limited interpretability. Current explainable RL methods seldom use *state attributions* to assist training. Rather, they treat all state features equally, thereby neglecting the heterogeneous impacts of individual state dimensions on the reward. We propose *RKHS--SHAP-based Advanced Actor--Critic (RSA2C)*, an attribution-aware, kernelized, two–timescale AC, including Actor, Value Critic, and Advantage Critic. The Actor is instantiated in a vector-valued reproducing kernel Hilbert space (RKHS) with a Mahalanobis-weighted operator-valued kernel, while the Value Critic and Advantage Critic reside in scalar RKHSs. These RKHS-enhanced components use sparsified dictionaries: the Value Critic maintains its own dictionary, while the Actor and Advantage Critic share one. State attributions, computed from the Value Critic via RKHS--SHAP (kernel mean embedding for on-manifold expectations and conditional mean embedding for off-manifold expectations), are converted into Mahalanobis-gated weights that modulate Actor gradients and Advantage Critic targets. Theoretically, we derive a global, non-asymptotic convergence bound under *state perturbations*, showing stability through the perturbation-error term and efficiency through the convergence-error term. Empirical results on three standard continuous-control environments show that RSA2C achieves efficiency, stability, and interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents RSA2C, a novel Actor-Critic (AC) algorithm that operates within Reproducing Kernel Hilbert Spaces (RKHSs). The core and significant contribution is the integration of "in-the-loop" interpretability: the algorithm computes state feature attributions from the Value Critic using RKHS-SHAP (with both on-manifold KME and off-manifold CME variants) and directly injects this information into the Actor's policy updates. This is achieved by converting attributions into a Mahalanobis-weighted Operator-Valued Kernel (OVK), effectively making the policy gradient "aware" of feature importance. The authors provide a strong theoretical analysis, including a global, non-asymptotic convergence guarantee under state perturbations.

### Strengths
The paper's primary strength is its conceptual leap. It successfully "bridges interpretability and optimization" by transforming explainable RL (XRL) from a passive, post-hoc analysis tool into an active, "in-the-loop" component that directly informs and guides the optimization process. This is a valuable and novel direction for the field.

### Weaknesses
While this is a strong theoretical paper, its impact could be significantly improved by addressing the following key points:

(1) The theoretical analysis in Section 4 (Theorems 4.8, 4.9, 4.10) rests on a substantial set of assumptions (e.g., Assumptions 4.3, 4.5, 4.6, 4.7). While such assumptions are often necessary for convergence proofs, the paper provides minimal discussion on their practical implications or justification. The paper would be greatly strengthened by a dedicated discussion on the reasonableness of these assumptions. For instance, under what practical conditions can we expect the $C_\nu$-Lipschitz continuity of the stationary distribution (Ass. 4.7) to hold? 

(2) A primary contribution is the non-asymptotic convergence guarantee (e.g., the $\mathcal{O}(\log^2 T / T^{1/4})$ rate from Theorem 4.9). However, the empirical validation in Section 5 focuses exclusively on final rewards, stability, and the values of the SHAP attributions. There is a missed opportunity to empirically validate the core theoretical claims.

(3) The paper is technically dense and written for a specialized audience. The introduction and contributions section (1.1) could be improved to more clearly articulate the significance of this work to the general RL community. The paper immediately dives into the how (RKHS, OVKs, SHAP) without sufficiently establishing the why for a non-expert. Why should a general RL researcher, accustomed to deep learning, care about this highly complex, non-parametric framework? The introduction should be refined to better motivate the work, perhaps by highlighting specific problem classes where this provable, attribution-aware approach is uniquely suited.

### Questions
Minor Issues and Clarifications

(1) The paper states a diagonal weight matrix $W$ is used for "robustness and efficiency" (Section 3.1), which explicitly ignores feature interactions. However, the key benefit of the CME variant is precisely that it models feature correlations (Section 5, Table 2 analysis). This seems contradictory. Please discuss this tension. 

(2)  In Section 3.1, the paper avoids applying SHAP-weighting to the Value Critic's kernel "to avoid circularity." This is an important design choice. Please expand this point slightly to be more explicit (e.g., "to prevent unstable feedback loops where attributions define the value function from which they are simultaneously derived").

(3) Typo in Algorithm 1: Line 7 (Optimize Value Critic) incorrectly cites the gradient as $\nabla_h J(h, \Sigma)$, which is the Actor's gradient. This should be $\nabla_{w_V} J(w_V)$ as defined in Equation (8).

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
4

### Summary
This paper presents the RSA2C (RKHS-SHAP-based Advanced Actor-Critic) algorithm, which integrates kernel-based Shapley attribution (RKHS-SHAP) into an Actor-Critic framework for reinforcement learning. The key innovation of the paper lies in combining state attribution with kernel methods in order to improve both the interpretability and efficiency of the Actor-Critic (AC) algorithm. RSA2C incorporates state feature importance into policy updates via a Mahalanobis-weighted operator-valued kernel. The authors claim that this method provides a stable, interpretable, and efficient solution for continuous control problems, with theoretical guarantees and empirical validation on standard benchmarks.

### Strengths
* Novelty: The proposed RSA2C algorithm introduces a promising combination of kernel methods and Shapley attribution in the context of reinforcement learning. The integration of state-level SHAP values into an Actor-Critic setup is a novel approach to enhance both interpretability and stability during policy optimization.
* Mathematical Rigor: The paper provides a solid theoretical foundation, including global convergence bounds under state perturbations. The mathematical formulation is generally well-presented and supports the proposed method's stability and efficiency.
* Empirical Validation: The empirical results show that RSA2C achieves competitive performance in continuous control tasks, demonstrating improvements in stability and interpretability without incurring significant computational overhead. This suggests practical applicability for safety-critical systems where interpretability is essential.
* Effective Use of RKHS: The paper demonstrates the advantage of using RKHS for efficient Shapley value computation, particularly in high-dimensional spaces where traditional methods would be computationally expensive.

### Weaknesses
Lack of a Thorough Analysis of the Motivation:

The main weakness of this paper lies in its insufficient justification for the core approach. The authors propose using state-level Shapley attribution to improve the interpretability and stability of the Actor-Critic (AC) framework, but they do not adequately explain why this approach is particularly effective for solving the interpretability issue in AC algorithms. While the authors describe the technical implementation and use of kernelized SHAP (RKHS-SHAP), the paper does not establish a strong connection between interpretability and AC optimization, nor does it explain why other interpretability methods (such as post-hoc approaches like LIME or saliency maps) cannot achieve similar results in the context of AC.

The authors jump straight into the kernelized implementation without first providing a more systematic analysis of the relationship between AC's interpretability challenge and state-level attribution. A deeper exploration of why state attribution is the most suitable tool for improving the interpretability of AC would significantly enhance the paper’s motivation and clarify the underlying rationale behind the algorithm design.

### Questions
* *Major Concern: Motivation and Justification of the Approach*
The primary concern with the paper is the lack of a thorough systematic analysis of the relationship between interpretability and AC optimization. While the paper presents the technical details of the kernel-based implementation, it jumps into this kernelized approach without providing sufficient justification as to why the state-level attribution is the best way to address the interpretability issue in AC algorithms.
Specifically, the authors do not explain why traditional post-hoc explanation methods, such as saliency maps or LIME, cannot be directly incorporated into the Actor-Critic framework. There is a missing bridge between the problem of AC's lack of interpretability and the solution based on state attribution. A more thorough discussion of why state attribution is the key to enhancing interpretability would strengthen the paper's motivation.
* *Minor Concern: Notation and Definitions*
The paper uses several notations without adequate definitions or clarifications. For example, in Eq. (2), the use of the $\otimes$ operator is not explicitly defined. Given the complexity of the notation, it is essential that all operators and their meaning within the context are clearly introduced before their use. Lack of proper definitions can confuse readers, especially when dealing with non-standard operations in kernel methods.
* *Minor Concern: Explanation of RKHS-SHAP and Its Link to AC*
While the paper successfully integrates RKHS-SHAP with Actor-Critic, the connection between these components could be explained more clearly. The reader could benefit from more context on how RKHS-SHAP specifically enhances AC’s learning process beyond just being a tool for computing state attributions. Providing a clearer conceptual framework would help reinforce the contribution of RKHS-SHAP to the overall algorithm's performance.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces the RKHS-SHAP-based Advanced Actor-Critic (RSA2C), a novel Actor-Critic (AC) algorithm designed to be both interpretable and provably stable. The core problem it addresses is that traditional AC methods are "black boxes," and existing explainable RL (XRL) methods are typically post-hoc, meaning they don't use interpretability to improve the training process itself.

RSA2C is a kernelized, two-timescale algorithm with three components: an Actor, a Value Critic, and an Advantage Critic, all operating in Reproducing-Kernel Hilbert Spaces (RKHS) . The key mechanism is as follows:

1.  The Value Critic learns the state-value function $V(s)$ in a scalar RKHS.
2.  State attributions (Shapley values) are computed from this Value Critic using RKHS-SHAP. This method uses kernel mean embeddings (KME) or conditional mean embeddings (CME) to analytically compute expectations for SHAP, avoiding costly sampling.
3.  These attributions are converted into "Mahalanobis-gated weights".
4.  These weights define an adaptive, attribution-weighted kernel.
5.  This special kernel is then used by the Actor and the Advantage Critic, effectively modulating their gradient updates based on the learned importance of each state feature.

### Strengths
Using RKHS-SHAP (KME/CME) is a clever, non-trivial, and analytical way to compute the required attributions without needing explicit density models.

The paper provides a rigorous global, non-asymptotic convergence bound (Theorem 4.10) for its complex, two-timescale kernelized algorithm, even under state perturbations. This is a substantial theoretical achievement.

The architecture thoughtfully separates the Value Critic (which generates SHAP) from the Actor/Advantage Critic (which use SHAP), avoiding circular dependencies and potential instabilities.

### Weaknesses
The proposed RSA2C algorithm is very complex. It requires three separate RKHS components, two different dictionaries, online dictionary sparsification (ALD), and the online computation of RKHS-SHAP (which, especially for CME, is computationally expensive). This high complexity may be a significant barrier to practical adoption.

The experiments compare RSA2C against its own ablations (Advanced AC, without SHAP) and a simpler kernel-based AC (RKHS-AC). While this is a good ablation, the paper is missing a comparison to standard, state-of-the-art non-kernel AC methods (e.g., A2C, PPO, SAC). Without this, it's difficult to judge the practical utility of this complex method. Do the gains in performance and stability justify the massive increase in complexity over standard deep RL methods?

The experiments are on low-to-medium dimensional environments (Pendulum-v1: 3 dims, BipedalWalker-v3: 24 dims). Kernel methods, even with sparsification, are notoriously difficult to scale to high-dimensional state spaces (e.g., hundreds of dimensions, or pixel inputs). The paper does not address or demonstrate how this approach would scale.

### Questions
1.  The primary question is about practical utility. How does RSA2C's sample efficiency and final performance compare to standard, widely-used (non-kernel) baselines like PPO or SAC on these same tasks? Do the stability and interpretability benefits justify the significant increase in algorithmic complexity?

2.  The scalability of kernel methods is a known challenge. How do you envision this method scaling to high-dimensional state spaces (e.g., $>$100 dims, or pixel-based inputs)? Does the computational cost of both the kernel operations (with ALD) and the RKHS-SHAP computation become prohibitive?

3.  The core mechanism involves a delay: the Value Critic at time $t$ provides SHAP attributions, which are then used to define the kernel for the Actor and Advantage Critic updates. How does this "lag" in the attribution signal (which is based on a slightly old value function) affect learning stability and convergence speed in practice?

4.  The paper argues that the CME variant's "balanced allocation of importance" is better than KME's sparse, single-feature dominance. Is this "balanced" attribution map provably more correct, or just different? How do you validate that the learned attributions are faithful to the environment's true dynamics?

### Soundness
2

### Presentation
2

### Contribution
2
