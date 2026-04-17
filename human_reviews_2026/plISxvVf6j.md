# Terminal Velocity Matching

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
We propose Terminal Velocity Matching (TVM), a generalization of flow matching that enables high-fidelity one- and few-step generative modeling. TVM models the transition between any two diffusion timesteps and regularizes its behavior at its terminal time rather than at the initial time. We prove that TVM provides an upper bound on the $2$-Wasserstein distance between data and model distributions when the model is Lipschitz continuous. However, since Diffusion Transformers lack this property, we introduce minimal architectural changes that achieve stable, single-stage training. To make TVM efficient in practice, we develop a fused attention kernel that supports backward passes on Jacobian-Vector Products, which scale well with transformer architectures. On ImageNet-256x256, TVM achieves 3.29 FID with a single function evaluation (NFE) and 1.99 FID with 4 NFEs. It similarly achieves 4.32 1-NFE FID and 2.94 4-NFE FID on ImageNet-512x512, representing state-of-the-art performance for one/few-step models from scratch.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a novel method for training one or few-step generative models by matching the terminal velocity. Through a clever rearrangement of the displacement equation between arbitrary points t and s along the ODE trajectory, the paper derives a novel objective which contains the instantaneous velocity at s and the generic Flow Matching loss. The formulation requires high order gradients through the JVP operation, but proposes improvements to the JVP computation to ensure scalability. The method further introduces improvements to DiT architecture and a way to include CFG conditioning. The method is tested on the common benchmark Imagenet 256x256 and achieves SOTA results for one and few-step generation.

### Strengths
The loss formulation is novel and well motivated, the paper introduces several relevant contribution such as DiT modification, scalable JVP computation, novel CFG training procedure. The results are strong and the method is scalable.

### Weaknesses
To me a weakness of the method it is hard to tell from the manuscript to what we can attribute the good performance. The overall results are great so there is no doubt that the proposed method is effective, but I wonder for example what results one could get from MeanFlow when using the improved DiT architecture. TVM is good but introduces some complications, such as using a EMA version of the model in the target, and having to backpropagate through JVP. While the latter is addressed, I wonder if just the improved architecture can work well on MeanFlow. I checked through the text and it doesn't specify whether meanflow is trained with standard DiT, so I am not sure what the retrained results from table 1 correspond to.

A common benchmark for these models is CIFAR10 but that was not reported. While it is a relatively small dataset, it can provide useful insight on the performance of the model without CFG, as well as on the sensitivity of the model to specific hyperparameters.

Overall, I think this method is promising, but seems like many details are left out, which makes it hard to fully understand the significance of the contribution (see questions). Also there is no open source code.

### Questions
- 1) Maybe I missed it but I could not find the exact number of training iterations for the results in table 3, same for batch size.
- 2) There also seem to be no info about the total training time. The method uses a forward pass with JVP plus a standard forward pass with the EMA model, and backward pass through JVP. Intuitively this should be much slower than MeanFlow or sCT, and it would be interesting to understand by how much.

### Soundness
3

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
The authors propose Terminal Velocity Matching (TVM), a few-step generation framework that matches terminal-time velocities instead of initial-time ones. To make TVM scalable for transformer architectures, the paper introduces minimal architectural modifications (e.g., RMSNorm-based AdaLN, Lipschitz initialization) and an efficient Flash-Attention JVP kernel that supports backward passes through Jacobian–Vector Products (JVPs). Combined with a simple training recipe—without curriculum training schedules or adaptive weighting—TVM achieves strong results on ImageNet-256×256, setting a new Pareto frontier for few-step generative models (FID = 3.30 @ 1-NFE, 2.49 @ 2-NFE) while maintaining training stability.

### Strengths
1. **Boundary condition at terminal time**

By enforcing velocity matching at the terminal rather than initial timestep, the method avoids evaluating JVPs involving guided velocities that often exhibit large norms and high variance during training. This could be more beneficial for stabilizing training when scaling to larger dimensionality, where the guided velocities often exhibit even larger norms and higher variance.

2. **Simple and stable training recipe**

The paper achieves stable one-stage training with only minor modifications to model architecture and without requiring any curriculum schedule, auxiliary objectives, or adaptive weighting heuristics. These simplifications make the approach both reproducible and again appealing for scaling to large models or high-dimensional data.

3. **Customized JVP kernel supporting efficient backpropagation**

The authors develop a fused Flash-Attention JVP kernel that supports backward passes on Jacobian–Vector Products within transformer blocks. This kernel significantly reduces memory usage and improves runtime efficiency during training. This engineering contribution is non-trivial and essential for making TVM practical at scale.

### Weaknesses
1. **Backpropagation through JVP.**

Unlike prior continuous-time consistency models where JVP terms are detached from the gradient graph (sCM, MeanFlow, etc.), TVM explicitly backpropagates through the JVP term introducing additional computational cost. This could become prohibitive for large-scale models. Providing quantitative analysis (e.g., runtime, memory, or gradient-cost overhead relative to MeanFlow) would help stress the concern.

2. **Insufficient ablations.**

While the design choices are well-motivated, the paper would benefit from deeper empirical validation. For instance, ablations comparing training with vs. without adaptive weighting on the training loss, the convergence of the original DiT versus the Lipschitz-controlled variant, and MeanFlow vs. TVM under identical random classifier-free guidance (CFG) conditions. Reporting gradient-norm / JVP-norm comparisons between TVM and MeanFlow would further support the claim of reduced JVP instability. Including recent baselines such as FACM (Peng et al., 2025) would also improve completeness.

3. **Need for deeper insight behind ablation trends,**

The empirical results raise several interesting but unexplained phenomena. For example, using any proportion of $t=s$ (pure flow-matching loss) degrades 1-step performance, and employing EMA targets accelerates convergence. A deeper discussion of why these trends arise would enrich the paper’s interpretability and practical guidance.

Peng, Yansong, Kai Zhu, Yu Liu, Pingyu Wu, Hebei Li, Xiaoyan Sun, and Feng Wu. "Flow-anchored consistency models." arXiv preprint arXiv:2507.03738 (2025).

### Questions
1. **Distributional guarantees vs. flow-map loss.**

The authors mention that Flow Map methods lack explicit distributional control, whereas TVM provides a 2-Wasserstein control. Could the authors elaborate on the difference? At a glance, the TVM loss resembles the Lagrangian distillation loss proposed in Flow Map—what key term enforces distributional regularity here?

2. **Effect of removing forced FM loss.**

TVM achieves superior 1-step generation performance even with 0% enforced Flow Matching samples. What intuition do the authors have for why completely removing the auxiliary FM supervision leads to better convergence?

3. **EMA acceleration.**

What is the authors’ intuition behind the observed acceleration of convergence when using EMA weights for the training objective? Does this act primarily as a variance-reduction mechanism?

4. **Distillation potential.**

Have the authors explored or considered using TVM as a distillation objective? Since TVM naturally supports terminal-condition regularization, it could serve as a stable distillation formulation for distilling multi-step teacher models into few-step generators.

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
This paper introduces Terminal Velocity Matching (TVM), a new training objective for generative modeling that generalizes Flow Matching (FM) by matching velocity fields at terminal rather than initial points along probability flow trajectories. The authors prove that the TVM loss upper bounds the 2-Wasserstein distance between data and model distributions, providing a principled connection between training dynamics and optimal transport theory. They also propose practical architectural modifications—such as semi-Lipschitz normalization, FlashAttention-based Jacobian-vector products, and scaled parameterization—that stabilize training and enable few-step sampling. Empirically, TVM achieves state-of-the-art single- and few-step image generation performance on ImageNet-256 with a DiT backbone.

### Strengths
1. The theoretical formulation is elegant and well-motivated, linking Flow Matching to a Wasserstein upper bound through terminal velocity constraints.
2. The method achieves excellent efficiency–quality trade-offs, outperforming existing one-step and few-step baselines (e.g., Consistency Models, MeanFlow) on ImageNet-256.
3. The architectural refinements (semi-Lipschitz normalization, FlashAttention JVP) are practically valuable contributions that could generalize to other diffusion or flow-based frameworks.

### Weaknesses
1. While theoretically grounded, the intuition behind “terminal velocity” could be elaborated further—especially how it differs in practice from midpoint or integral matching.
2. The scope of evaluation is limited to class-conditional ImageNet-256. Demonstrating robustness on higher-resolution or unconditional datasets (e.g., ImageNet-512, COCO) would strengthen generality claims.
3. The paper relies on a single architecture (DiT-XL/2). It is unclear whether TVM’s benefits extend to U-Net–based or continuous-time flow models.
4. The FlashAttention JVP optimization, though valuable, feels somewhat tangential to the conceptual contribution and could be relegated to an appendix.
5. Despite the theoretical connection to Wasserstein distance, there is no empirical verification of this bound (e.g., via transport cost evaluation), leaving its practical tightness untested.

### Questions
1. Can the authors clarify why terminal velocity matching improves convergence over initial or midpoint matching?
2. How tight is the Wasserstein upper bound empirically—has this been tested or approximated?
3. Is the FlashAttention JVP essential for convergence, or mainly for speed?
4. Could the method generalize to other architectures or modalities (e.g., U-Nets, text-to-image, or video)?

### Soundness
3

### Presentation
3

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
This paper proposes Terminal Velocity Matching (TVM), a single-stage objective that generalizes flow matching by learning ODE integrals between any diffusion timesteps via terminal velocity matching. TVM upper-bounds 2-Wasserstein distance under Lipschitz assumptions. On ImageNet-256×256, TVM achieves 3.30 FID in 1-NFE (SOTA) and a new few-step Pareto frontier (2.49 FID at 2-NFE), with natural n-step interpolation.

### Strengths
1. TVM reframes the problem of learning long-horizon ODE jumps as a terminal velocity condition (Eq. 6–7), providing a clean theoretical link between displacement error and velocity matching.
2. Theorem 1 establishes a distribution-level guarantee ($W_2$ upper bound) without requiring multiple particles (unlike IMM).
3. Duality with MeanFlow is clearly articulated: The paper shows MeanFlow matches initial velocity while TVM matches terminal velocity (Appendix E.1), offering a compelling symmetry and explaining why TVM is more stable under random CFG.
4. This paper introduces a custom Flash Attention + JVP kernel to accelerate.

### Weaknesses
1. While inference is fast, training cost (FLOPs, GPU-hours) vs. MeanFlow, sCT, or IMM is not reported.

### Questions
1. How sensitive is performance to the time sampling distribution $p(t,s)$? The paper uses a biased gap-based scheme — would uniform sampling or curriculum hurt?

### Soundness
3

### Presentation
3

### Contribution
3
