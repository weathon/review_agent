# Noise-Adaptive Layerwise Learning Rates: Accelerating Geometry-Aware Optimization for Deep Neural Network Training

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 8, 4

## Abstract
Geometry-aware optimization algorithms, such as Muon, have achieved remarkable success in training deep neural networks (DNNs).  These methods leverage the underlying geometry of DNNs by selecting appropriate norms for different layers and updating parameters via norm-constrained linear minimization oracles (LMOs). However, even within a group of layers associated with the same norm, the local curvature can be heterogeneous across layers and vary dynamically over the course of training. For example, recent work shows that sharpness varies substantially across transformer layers and throughout training, yet standard geometry-aware optimizers impose fixed learning rates to layers within the same group, which may be inefficient for DNN training.

In this paper, we introduce a noise-adaptive layerwise learning rate scheme on top of geometry-aware optimization algorithms and substantially accelerate DNN training compared to methods that use fixed learning rates within each group. Our method estimates gradient variance in the dual norm induced by the chosen LMO on the fly, and uses it to assign time-varying noise-adaptive layerwise learning rates within each group.
We provide a theoretical analysis showing that our algorithm achieves a sharp convergence rate. Empirical results on transformer architectures such as LLaMA and GPT demonstrate that our approach achieves faster convergence than state-of-the-art optimizers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses inefficiencies in geometry-aware optimization algorithms for deep neural networks, which typically use fixed learning rates across all layers performing updates defined with respect to the same norm. Recognizing that local curvature and gradient noise can vary across layers and over training, the authors propose a noise-adaptive layerwise learning rate scheme. They call their method (being a combination of Scion [1]/Gluon [2] with this adaptive scheme) LANTON. This method dynamically estimates gradient variance associated with each layer's norm-constrained update and adjusts learning rates online. The authors provide convergence guarantees and present experiments on transformer models (LLaMA and GPT), which demonstrate faster training compared to some existing optimizers.

[1] Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, and Volkan Cevher. Training deep learning models with norm-constrained lmos. arXiv preprint arXiv:2502.07529, 2025.

[2] Artem Riabinin, Egor Shulgin, Kaja Gruntkowska, and Peter Richtarik. Gluon: Making muon \& Scion great again!(Bridging theory and practice of LMO-based optimizers for LLMs). arXiv preprint arXiv:2505.13416, 2025.

### Strengths
- The work addresses a research area that is active and highly relevant today.

- The first 3 sections of the paper are generally well-written.

- The method proposed in the paper represents a sound idea, as evidenced by good empirical performance relative to several baseline optimizers.

### Weaknesses
While the idea behind the proposed learning rate scheme is promising, I have several serious concerns about this submission. The first half of the paper is clearly written, but issues arise once the authors introduce their method. These include undefined or incorrectly used notation, lack of acknowledgment of prior work, and unrealistic assumptions. In particular:

1. The main update rule in LANTON (line 10, Algorithm 1) is essentially identical to the update rule of Scion [1]/Gluon [2]. Hence, the main contribution of the paper is the introduction of an adaptive layer-wise learning rate scheme rather than an entirely new algorithm. The paper does not clearly acknowledge that the main algorithmic components are inherited from prior work. Proper citations to [1], [2], and [3] should be included in the "Algorithmic Framework" paragraph of Section 4.

2. There is a **substantial overlap with [2], which the authors do not acknowledge**:

    - Lines 44–45 claim that "existing geometry-aware optimizers simply assign fixed learning rates within groups of layers associated with the same norm choice". This is inaccurate: [2] introduced Gluon, a family of layerwise LMO-based optimizers (including Muon and Scion as special cases) that already use different learning rates across layers (see Theorems 1 and 2 in [2]). Meanwhile, [2] is cited only once in the main part of this paper (for the smoothness assumption), completely ignoring other contributions.
    
    - The notation in Assumption 5.1 is taken directly from [2], but simplified: [2] uses generalized smoothness, while this paper only assumes classical (layer-wise) smoothness. In Section 5, the paper switches from a single matrix space to a product space without explaining the setup (e.g., the gradient component notation $\nabla_l f(X)$ is not defined). The paper does not properly introduce this notation. In line 139, the authors say the optimization is over $X \in \mathbb{R}^{m \times n}$, but it is actually over a product space of matrices, as defined in the Introduction section in [2].
    
    - This (unintroduced) notation is used correctly in Assumption 5.1, but not in several other places in the paper, e.g., the algorithm. For example, the authors define $G_t^l = \nabla F(X_t^l;\xi_t^l)$, but it should be $\nabla_l F(X_t;\xi_t)$, i.e., the $l$th component of the gradient evaluated on $(X_t;\xi_t)$, not the gradient evaluated on the $l$th component of $X_t$; this is necessary for the dimensions to match ($F$ takes all the model parameters as input). The gradient must be computed on the full parameter vector $X_t$, not just the $l$th component $X_t^l$ due to the mechanics of backpropagation - see [2]. For the same reason, the same sample $\xi_t$ is used for the whole gradient, so the $l$ index on $\xi_t$ is wrong.
    
    - This misunderstanding of the setup introduces further errors. For example, in Theorem 5.3, expressions like $f(X_1^l)$ are incorrect because $f$ takes the full model $X$, not a single layer $X^l$.

    - Many proofs very closely follow [2]: Lemma C.1 replicates the first part of Theorem 8 in [2] exactly (simplified by setting $L_i^1 = 0$), and Theorem 5.3 follows closely the remainder of that proof, differing in the assumption on stochastic gradient noise. The mistake with $f(X_1^l)$ also appears here.
    
Overall, the authors not only borrow the setup and notation from [2] without acknowledgment, but also use it incorrectly. They claim their main contribution is being the first to introduce layer-specific learning rates in LMO-based optimizers, but this was already done in [2], which also proved results under more general assumptions (classical smoothness is known to be a poor model for neural network loss landscapes, while generalized smoothness better captures their behavior [4]).

3. I am very skeptical about the practicality of Assumption 5.2, which requires that
$$\underline{\sigma}_l\leq\|\nabla_l F(X,\xi) - \nabla_l f(X)\|_{(l)*} \leq\bar{\sigma}_l$$
with probability one across all layers. The uniform upper bound is already very restrictive, and the lower bound is generally unrealistic (for example, it is forced to be $0$ under interpolation). But when $\underline{\sigma}_l=0$, Theorem 5.3 requires that $1\leq \beta_2<1$, which is impossible.

[1] Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, and Volkan Cevher. Training deep learning models with norm-constrained lmos. arXiv preprint arXiv:2502.07529, 2025.

[2] Artem Riabinin, Egor Shulgin, Kaja Gruntkowska, and Peter Richtarik. Gluon: Making muon \& Scion great again!(Bridging theory and practice of LMO-based optimizers for LLMs). arXiv preprint arXiv:2505.13416, 2025.

[3] Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024.

[4] Zhang, Jingzhao, Tianxing He, Suvrit Sra, and Ali Jadbabaie. Why gradient clipping accelerates training: A theoretical justification for adaptivity. arXiv preprint arXiv:1905.11881, 2019.

### Questions
I think that the overall idea behind the submission is good, but the paper itself has serious flaws that make it not publishable in the current state. The setup is not introduced, it completely ignores that large chunks of the theory are borrowed from another work, and it misuses some chunks of this work by making errors. Hence, I do not think it is po
Could the authors please address my comments above?

- How can Assumption 5.2 be justified beyond analytic convenience?

- How does the proposed algorithm compare with other adaptive variants of Muon?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper finds that existing geometry-aware optimizers apply a single, fixed learning rate to groups of layers that share the same norm. However, the stochastic gradient noise could be highly heterogeneous _within_ these groups and dynamic throughout training, rendering a uniform learning rate suboptimal.
To address this, the paper proposes LANTON, a layer-wise adaptive learning rate schedule built on top of existing LMO-based optimizers (Scion). Its mechanism estimates the gradient variance for each layer in the dual norm space to scale that layer's learning rate. The principle is to assign smaller step sizes to layers with higher estimated gradient noise.
The authors provide a theoretical convergence analysis, claiming an improved rate compared with a uniform maximum method. Empirically, they demonstrate that LANTON accelerates the training of GPT2 and LLaMA models, achieving faster convergence and showing.

### Strengths
1. The motivation is clear. This paper is motivated by (Wang et al., 2025), which notices different loss curvature on different layers of Adam. This paper empirically shows a similar observation on Muon.
2. This paper applies the noisy adaptive idea within the geometry-aware LMO framework by estimating the variance in the dual norm space.
3. LANTON shows an improved convergence rate compared with Scion using a uniform learning rate.
4. The paper shows consistent improvement on both GPT2 and LLaMA models.

### Weaknesses
1. This paper doesn't discuss about computational overhead. LANTON needs to compute the dual norm additionally. Especially for hidden layers, it requires extra nuclear norm computation. I suppose the computation of the nuclear norm is not negligible. How to compute it in implementation? How much computational overhead does it require? Could you compare with other methods according to wall-clock time?
2. Miss image task comparison. This paper's experiments only focus on language tasks. I'm curious if LANTON can show improvement in image tasks like CIFAR-10?
3. I feel LANTON shows speedup at the beginning of training. Would the benefit diminish for long-term training?
4. I think Figure 4 (b) is an unfair comparison. If I understand correctly, Figure 4 (b) has the same setup as Figure 3 (left) except number of tokens / training steps. In Figure 3, LANTON shows a similar final performance to D-Muon. How could you claim a speedup if you run D-Muon 2$\times$ longer? Such speedup may be just brought by a different learning rate and its schedule.

### Questions
1. Do you compare the performance between options 1 and 2 in Algorithm 1?
2. I suppose y-axis of subfigures 2 and 4 in Figure 2 should be "Validation loss".
3. Is there any potential reason why D-Muon converges faster than LANTON in subfigure 4 of Figure 2?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a layer-wise learning rate scaling mechanism that works with existing geometry-aware optimizers. The main idea is to dynamically choose smaller learning rates for layers with higher stochastic gradient variance, and larger ones for layers with lower variance, based on online variance estimation. Geometry awareness is maintained through the use of the Linear Minimization Oracle (LMO) for norm-constrained updates. Even among layers that share the same type (and thus the same norm/LMO), the gradient noise and curvature can differ across layers and change during training. However, existing geometry-aware optimizers typically assign a fixed learning rate to all layers within a group, ignoring this variation. The paper shows both theoretical and empirical improvements: the proposed method achieves a better convergence rate than fixed-rate geometry-aware methods and demonstrates faster convergence and up to 1.5× higher sample efficiency on GPT-2 and LLaMA-2 experiments.

### Strengths
The paper is clearly organized and easy to follow. I am not deeply familiar with geometry-aware optimization but enjoy reading the manuscript.

The method is well-motivated. LANTON effectively tackles the overlooked issue of intra-group heterogeneity in gradient noise.

The proposed approach is plug-and-play and integrates naturally with existing geometry-aware frameworks such as Muon, as well as common learning rate schedules.

The experiments on GPT-2 and LLaMA convincingly support the paper’s claims, demonstrating faster convergence and better sample efficiency.

The convergence analysis is interesting and well connected to the method. However I do not have the technical background to verify all the derivations.

### Weaknesses
I do not see any major weaknesses. The additional computational and memory overhead introduced by maintaining per-layer noise estimates is expected and acceptable.

Empirical estimation of stochastic gradient variance may depend strongly on batch size, but the paper does not include an analysis or sensitivity study in this regard. Including such an analysis would strengthen the empirical validation.

### Questions
The theoretical bound includes constants depending on parameter dimensions. Can the authors clarify how large these constants are in practice, and whether they affect scalability to larger models?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a geometry-aware optimization algorithm named LANTON (LAyer-wise Noise-adaptive learning raTe scaling with Operator Norms). Building upon the frameworks of the D-Muon and Scion geometry-aware optimization algorithms, LANTON introduces noise-adaptive layer-wise learning rate scaling, which dynamically estimates the gradient variance in the dual norm induced by the selected Linear Minimization Oracle (LMO) and uses this estimate to assign layerwise learning rates that adapt over the course of training. This adaptive scaling enables the optimizer to assign smaller learning rates to noisier layers and larger learning rates to less noisy layers, thereby improving convergence efficiency.

### Strengths
1.	This paper proposes a new approach that combines geometry-aware optimization with noise-adaptive layer-wise learning rates.

2.	It achieves about 1.5× faster convergence than the baseline D-Muon in large-scale model training.

3.	The paper is clearly structured and well-organized, and Algorithm 1 is easy to understand.

### Weaknesses
1.	This paper essentially integrates Geometry-Aware Optimization with Layer-wise Adaptive Learning Rates. The method inherits the Geometry-Aware Optimization framework from D-Muon, and, building on that, adopts the concept from LAMB (Layer-wise) to apply noise-adaptive layer-wise learning rate scaling, assigning different learning rates to different layers within the same group. The noise estimation mechanism also resembles prior “variance-adaptive learning rate” methods such as AdaNoise, RAdam, and AdaBelief. It does not introduce any innovative improvements to the geometry-aware optimization, layer-wise adaptive learning rate, or noise adaptivity components themselves.
2.	The paper does not compare LANTON with more recent Layerwise Adaptive Learning Rate optimization methods.
3.	Lines 234–236 state that“Unlike Muon and D-Muon, which use AdamW for embedding and LM head layers, we adopt a geometry-aware framework (similar to Scion) and update these weight-sharing layers with Signum (see Table 1).” However, despite utilizing the Scion framework, the experiments do not include a direct comparison with Scion.

### Questions
1.	For the same group, different layers still share the same norms. What if this grouping were further refined or made dynamic?
2.	Why were experiments not conducted on larger models such as LLaMA-13B? In Appendix Section H, the paper states: “Averaged over three independent runs, D-Muon requires 20 h 05 m 24 s, whereas LANTON requires 20 h 55 m 37 s, about 4% more training time than D-Muon” if experiments were performed on larger models, would the computational overhead still remain around 4 % as reported in the paper? How would it change as the number of parameters increases?
3.	Have alternative noise-estimation methods been explored? Why was this estimation strategy chosen?
4.	Why does the experimental section not include a comparison with the Scion method?

### Soundness
3

### Presentation
3

### Contribution
2
