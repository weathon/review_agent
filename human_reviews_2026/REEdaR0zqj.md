# Beyond Uniformity: Regularizing Implicit Neural Representations through a Lipschitz Lens

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Implicit Neural Representations (INRs) have shown great promise in solving inverse problems, but their lack of inherent regularization often leads to a trade-off between expressiveness and smoothness. While Lipschitz continuity presents a principled form of implicit regularization, it is often applied as a rigid, uniform 1-Lipschitz constraint, limiting its potential in inverse problems. In this work, we reframe Lipschitz regularization as a flexible *Lipschitz budget framework*. We propose a method to first derive a principled, task-specific total budget $K$, then proceed to distribute this budget *non-uniformly* across all network components, including linear weights, activations, and embeddings. Across extensive experiments on deformable registration and image inpainting, we show that non-uniform allocation strategies provide a measure to balance regularization and expressiveness within the specified global budget. Our *Lipschitz lens* introduces an alternative, interpretable perspective to Neural Tangent Kernel (NTK) and Fourier analysis frameworks in INRs, offering practitioners actionable principles for improving network architecture and performance. Code and experimental results are available at: [https://lipschitz-inrs.github.io](https://lipschitz-inrs.github.io).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper revisits Lipschitz regularization in Implicit Neural Representations and introduces a K-Lipschitz budget framework that generalizes the traditional 1-Lipschitz constraint. The authors argue that uniform Lipschitz constraints (applied equally to all layers and components) are overly rigid and propose distributing a global Lipschitz budget non-uniformly across network components to balance smoothness and expressiveness. Experiments on signed distance fields (SDFs), medical image registration, and image inpainting demonstrate that flexible K-Lipschitz budgets improve stability, interpretability, and generalization.

### Strengths
- The idea of framing Lipschitz regularization as a budget allocation problem is fresh and theoretically well-motivated. It bridges a gap between rigid Lipschitz constraints and flexible, task-adaptive regularization schemes.
- The paper validates the theory across diverse INR applications—SDFs (geometry), deformable registration (medical imaging), and inpainting (vision)—showing generality.
- The Lipschitz lens provides a unifying view that complements NTK and Fourier analyses, contributing to a deeper understanding of implicit regularization in INRs.

### Weaknesses
- While the qualitative results are compelling, quantitative improvements (e.g., PSNR/SSIM/Folding Ratio) over strong baselines are small and sometimes marginal, with statistical significance not rigorously reported.
- I have some concerns regarding the oracle- and domain-driven estimates of K, but the estimation procedures are only sketched conceptually. Appendix references are insufficient for reproducing real-world tasks.
- Although the paper proposes an interesting conceptual framework (the Lipschitz-budget idea), most figures are purely experimental, focusing on results such as SDF reconstructions, lung deformation fields, and metric plots. The lungs in Fig.1 are quite confusing.

### Questions
- I have some questions about the fairness of comparison. The authors mention that all models are trained for a fixed number of epochs. However, since different architectures (e.g., SIREN vs. FFN vs. Gaussian-INR) and normalization methods may converge at different rates, a uniform epoch budget could introduce bias — some models might underfit or overfit relative to others. A convergence-based or early-stopping criterion would be a fairer comparison.
- What is the actual runtime or GPU memory overhead introduced by enforcing non-uniform Lipschitz constraints compared to the standard 1-Lipschitz setup?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper reframes Lipschitz regularization for Implicit Neural Representations (INRs) as a Lipschitz budget allocation problem. Rather than enforcing a strict 1-Lipschitz constraint, the authors treat the global Lipschitz constant K_B as a finite budget to be distributed non-uniformly across layers, activations, and embeddings. They derive layer-wise Lipschitz constants for common encodings, propose practical allocation strategies (uniform, linear, exponential, cosine), and demonstrate the approach on signed distance field reconstruction, deformable registration, and image inpainting. Empirically, non-uniform allocation improves the expressiveness-smoothness trade-off, and a data-driven oracle for estimating K_B yields interpretable, task-specific regularization.
The theoretical framing is sound: the derivations for per-component Lipschitz constants and the multiplicative composition rule are correct. The oracle-based Lipschitz budget estimation is mathematically consistent, though potentially sensitive to noise. Experiments are well designed and align with the hypotheses, though the product bound on Lipschitz constants is known to be loose, and the oracle lacks robustness analysis.
The paper is clearly written and well organized, with strong visual explanations and thorough appendices detailing hyperparameters, derivations, and code. Overall, the work unifies several regularization heuristics under a single “Lipschitz budget” framework. While the underlying tools are known, the reinterpretation and empirical synthesis are novel and practically valuable. The insights on how non-uniform allocation shapes expressivity and how Lipschitz analysis connects to NTK and Fourier perspectives provide novel intuition.

### Strengths
1. Conceptually elegant reframing of Lipschitz regularization as a quantitative resource-allocation problem.
2. Derivations for sinusoidal and random Fourier feature encodings connect spectral and geometric views.
3. Extensive appendices with reproducible setups (SDFs, registration, inpainting).
4. Empirical results consistent across modalities: tighter Lipschitz enforcement improves fidelity up to a task-dependent optimum.

### Weaknesses
1. The oracle estimator of K_B is a finite difference approximation, which is highly sensitive to discretization and noise. The paper lacks a rigorous analysis of this sensitivity
2. The product bound on Lipschitz constants is conservative; no tighter or data-dependent alternative has been explored.
3. The validation is on a small-scale dataset, e.g., Bunny SDFs, 256×256 inpainting, and a single lung registration pair.
4. No adaptive or learning mechanism for choosing K_B, for example, bi-level optimization 
5. The method does not guarantee a diffeomorphism resulting in folds; a comparison with diffeomorphic baselines would have been helpful.

### Questions
1.	Could the authors provide empirical estimates of the actual Lipschitz constant of trained models and compare them to the theoretical budget?
2.	How sensitive is performance to noise or discretization in the oracle K_B estimation?
3.	Would a bilevel optimization approach that learns K_B automatically yield comparable results?
4.	For registration, have the authors compared with diffeomorphic baselines that explicitly constrain Jacobian determinants?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on Lipschitz-constrained implicit neural representations (INRs). The authors propose to estimate a Lipschitz bound K using domain knowledge and explore different ways to distribute K across the components of the underlying neural network. They validate their framework on several tasks, including learning signed distance functions for object representation, image registration, and image inpainting. Overall, the paper builds on relatively simple and nice ideas and presents them clearly. The results are consistent with known trade-offs in Lipschitz-constrained networks.

### Strengths
Overall, the paper is easy to follow. There are extensive numerical experiments on different tasks that investigate the effect of different components. The integration of Lipschitz constraints into INRs seems like a nice line of research to explore.

### Weaknesses
Some parts could be improved (see questions).

### Questions
Could you include a few lines on how one should proceed with estimating the K bound? I know it is mentioned for each task, but I want a more general paragraph that tells what domain knowledge to look for and how to process estimating K when facing a new task. Could it be embedded into the training or form an optimization scheme to estimate it? 

Could you give an intuition why the budget distribution is not that effective in the SDF example, but it makes a difference for the image inpainting task? 

In Figure 2, why is the second bunny in row one different from the first of row two? Aren't they both with Bjoerick and ReLU? 

In the description of Figure 4 is written "the Lipschitz-regularized model exhibits significantly reduced folding artifacts while maintaining comparable TRE and the anatomical plausibility of the deformation"; however, in the Figure, there is no indication of Lipschitz / non-Lipschitz model, which confuses me. Could you clarify? 

Could you include visual examples (images) for the image inpainting? 

I suggest moving the citations in the conclusion to other parts and trying to summarize your own work there. 

I suggest more visual results (for the images) to help understand the effect of the methods.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a K-Lipschitz budget framework for implicit neural representations, deriving task-specific budgets and exploring non-uniform allocation strategies across network components. The work provides theoretical foundations and demonstrates applications in deformable registration, signed distance fields, and inpainting.

### Strengths
Writing is clear.

### Weaknesses
* The paper claims uniform 1-Lipschitz allocation is suboptimal but Figure 3 shows minimal differences between allocation strategies at K=1. Why is non-uniform allocation necessary if uniform allocation of a different budget K could achieve the same effect?

* Results conflate budget value K, allocation strategy, and
  normalization method, which factor actually drives performance?

* The inpainting oracle estimates Lipschitz constant from finite differences on the discrete image. Why should a continuous neural representation mimic the Lipschitz constant of its discrete sampling?

* The paper positions Lipschitz constraints as addressing "lack of implicit regularization" but never compares to actual implicit regularization from optimization dynamics or simply training longer. Does explicit Lipschitz regularization outperform other forms of regularization or architectural constraints?

* Only 5 hand-designed parametric schedules are explored with minimal
  justification. Given that optimal allocation appears task-dependent.
  The paper promises "actionable principles" but provides no mechanism
  for deriving allocations for new tasks.

* Experiments use small samples without confidence intervals or significance testing. Claims like "significantly sharper reconstruction" rely on visual inspection despite high variance in plots, making it difficult to assess whether observed differences are meaningful.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
