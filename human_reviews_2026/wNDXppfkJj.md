# Implicit Denoiser Structure in Robust Classifiers Explains Generative Capabilities

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Adversarially robust neural networks, while designed for classification, exhibit surprising generative capabilities when appropriately probed. We provide a theoretical framework explaining this phenomenon by connecting adversarial robustness to implicit denoising structure. Building on established results that robust training drives Jacobians toward low-rank solutions, we demonstrate that the Gram operator $\mathbf{J}^{\top}\mathbf{J}$ functions as an implicit denoiser, selectively preserving signal along discriminative subspaces while suppressing noise in orthogonal directions. This insight leads to Prior-Guided Drift Diffusion (PGDD), a simple algorithm that leverages this structure for generation through inference objectives rather than explicit Jacobian computation. PGDD requires no generative training or architectural modifications, yet produces class-consistent samples across different datasets and architectures. We extend our approach to standard networks via sPGDD, demonstrating that implicit generative structure exists beyond adversarially trained models. Our results establish a connection between discriminative robustness and generative modeling, showing that robust classifiers encode statistical priors that enable structured pattern generation without explicit generative objectives.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the connection between adversarial robustness and generative capabilities. The authors argue that the generative capacity of robust classifiers may originate from an implicit denoising structure within the Gram operator $\mathbf{J}^\top\mathbf{J}$. They propose the PGDD algorithm to leverage this structure for generative tasks. Surprisingly, they extend this mechanism to standard, non-robust models via 'smoothed PGDD.' The authors present numerical experiments to validate their findings.

### Strengths
* The authors conduct a careful review of related work, providing a solid foundation and context for their investigation.

* The PGDD algorithm is a conceptually novel and intuitive method. It leverages the hypothesized denoising structure of $J^\top J$ implicitly, bypassing the expensive explicit Jacobian computations.

### Weaknesses
* I feel the contributions of the paper are overstated. I find that the two main claims of the paper are not well supported by theoretical or empirical evidence: 
  * On the 'Implicit Denoiser' Hypothesis: The paper claims to establish a connection between the Gram operator and an implicit denoising structure. This claim is not substantiated with theoretical rigor, as no formal theorems are presented or proven. The provided numerical analysis only shows a difference in eigenvalue magnitudes between standard and robust models, but does not provide a mechanistic explanation for how this translates to a denoising process that facilitates generation.
  * Second, the authors propose two generating algorithms named PGDD and sPGDD. However, the evidence supporting their generative capabilities is weak and purely qualitative. The authors should have included standard metrics for generative modeling, such as FID or Inception Score, to assess the fidelity and diversity of the generated images.

* The paper is not easy to follow. Some notations, like the Gram operator $\mathbf{J}^\top\mathbf{J}$ and the $R^2$ statistics in Table 1, are not formally defined. (I also guess the energy ratio statistic should be something like $\mathbb{E}\|\mathbf{J}^\top\mathbf{J}\mathbf{\varepsilon}\|/\|\mathbf{\varepsilon}\|$). The description of algorithms like PGDD and sPGDD is too concise, especially for sPGDDs. If this is due to page limits, I suggest the authors shorten the "Conclusion" section or reduce figure sizes.

* Most of the numerical experiments are only conducted with small-scale neural networks on toy datasets like MNIST.

### Questions
* I am confused about the specificity of the 'implicit denoising structure'. The paper frames it as a property of robust classifiers that explains their unique generative abilities. However, the proposed sPGDD algorithm demonstrates that a similar generative mechanism can be exploited in standard, non-robust models too. Is this implicit denoising structure a unique characteristic of robust classifiers, or is it a general property of neural networks? If it is not unique, how can it serve as the key explanation for the generative performance of robust models in particular? If it is unique, what allows sPGDD to work on standard models?

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
3

### Summary
The authors show that the Jacobians of robust networks act as implicit denoisers. In particular, they use a simple gradient ascent algorithm, Prior-Guided Drift Diffusion, to generate class-consistent samples from adversarially robust networks trained without generative objectives.

### Strengths
- The authors provide ample experimental evidence for their claims, both quantitative and qualitative.
- The work provides additional insights on robust classifiers

### Weaknesses
- The contribution is mild, as generation from adversarially robust classifiers was previously observed in Santurkar et al. (2019, https://arxiv.org/pdf/1906.09453). Differentiating their objective reveals a very similar algorithm to PGDD, as the input-derivative of scalar outputs of a network all strongly involve the jacobian.
- In-depth analysis is limited to Cifar-10. It would strengthen the paper if this was done for larger datasets like ImageNet. Perhaps the computational cost is too large for full-resolution, but for artificially downsampled images it should be possible.
- Image plausibility is much weaker for standard networks, which limits the contribution in a novel area i.e. beyond robust networks.
- Normalized eigenvalue statistics are not provided (e.g. Table 1) beyond lambda_1 / lambda_2. It would strengthen the analysis if such statistics were reported, such as soft-rank (normalized eigenvalue entropy) and normalized-eigenvalue plots comparing standard and robust networks.

### Questions
- Are there greater connections between denoising diffusion models and the denoising capabilities of adversarially robust networks? Any further insight connecting with modern generative models would strengthen the work.
- The authors might consider interesting the work of Rodriguez-Munoz et al. (2024, https://arxiv.org/pdf/2409.20139), which showed a causal relationship of spectral properties -> robustness, as mentioned in the limitations section of the paper. In particular, Section 5.3 shows that aligning jacobians with natural image edges achieves robustness up to 60% of adversarial training.

### Soundness
3

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
2

### Summary
This paper proposes a theoretical framework to explain the emergent generative capabilities of adversarially robust classifiers. The central hypothesis is that adversarial training induces a low-rank structure in the network's input-output Jacobian, $J$. Consequently, the Gram operator, $J^TJ$, functions as an implicit denoiser, selectively preserving signal along a low-dimensional "discriminative subspace" while suppressing noise in orthogonal directions.

To leverage this insight, the authors introduce Prior-Guided Drift Diffusion (PGDD), a novel inference-time algorithm that generates images without requiring any generative training or architectural changes. PGDD works by iteratively updating an input to move its internal representation away from that of a noisy version of itself. The authors show that the gradient of this objective elegantly approximates a denoising step involving $-J^T J$. They extend this method to standard (non-robust) networks with smooth PGDD (sPGDD), which uses gradient averaging to stabilize the process.

### Strengths
- The paper's core contribution is providing a simple and principled explanation for a well-documented but poorly understood phenomenon. The identification of $J^T J$ as an implicit denoiser connects the spectral properties of robust networks (low-rank Jacobians) with the mechanisms of modern generative models (denoising).
- The paper is well-written and easy to follow. The introduction clearly motivates the problem and outlines the contributions.

### Weaknesses
1. Limited Scope of Full Spectral Analysis: While the spectral analysis on MNIST is excellent, it is limited to a small-scale problem. The paper's central theoretical claim about the $J^T J$ operator is not directly verified for the large-scale ImageNet models due to the computational intractability of computing the Jacobian. The success of PGDD on these models provides strong indirect evidence, but the paper relies on the assumption that the same low-rank spectral properties hold.

2. Lack of Quantitative Generative Metrics: The paper relies entirely on qualitative visual inspection for the generated ImageNet samples. While the images are coherent, their quality is visibly lower than state-of-the-art generative models. Including standard quantitative metrics like FID or Inception Score would be beneficial. Even if not state-of-the-art, these scores would provide a concrete way to compare generation quality across different models architectures, and between PGDD and sPGDD.

### Questions
1. Could you comment on the computational cost of PGDD/sPGDD for generating a single image compared to, for example, a forward pass in a comparably sized DDPM or GAN?

2. How were the hyper-parameters in Tables 3 and 4 selected? Was it manual tuning, or is there a more systematic procedure? For example, how does the generation trajectory change with different values of the Drift Noise Ratio or Step Size? Do you observe any failure modes (e.g., divergence, mode collapse) with improper settings?

3. While acknowledging the quality is not SOTA, could you provide FID/IS scores for the generated ImageNet samples? This would help quantitatively benchmark the effect of different robustness levels (ϵ), architectures, and the difference between PGDD on robust models vs. sPGDD on standard models.

4. Can you further elaborate on the theoretical relationship between the PGDD update direction $\nabla_x$LPGDD ≈ $-J^T J \epsilon$ and the score function $\nabla_x \log p(x)$? Is it possible to view PGDD as a form of score matching or Langevin dynamics on an implicit energy function defined by the classifier?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This study provides a theoretical perspective on why adversarial training elicits generative properties in robust image classifiers. It is demonstrated that the Jacobian Gram operator functions as an implicit denoiser, and an algorithm called Prior-Guided Drift Diffusion (PGDD) is developed to generate images. Furthermore, a sPGDD variant is proposed for standard (non-robust) classifiers by computing gradients over multiple independent perturbations of the inputs (akin to SmoothGrad). The study completes with empirical results aimed to demonstrate the implicit denoiser theory (on MNIST classifiers) and generations on ImageNet and MNIST classifiers.

### Strengths
- The works provides a novel (to the best of my knowledge) link between adversarial training and generative properties by formulating the implicit denoiser theory
- The authors propose PGDD to generate images by avoiding to compute $J^{\top}J$ explicitely.
- PGDD is further extended to sPGDD allowing generation via non-robust classifiers.
- The theory is (partially) backed by empirical evidence
- The limitation section is through and upfront

### Weaknesses
- The empirical evidence through the spectral analysis is very thin and only builds on one pair of networks (MNIST, $L_2$ and $\epsilon=1.5$, PGD-AT). If computing the Jacobian on ImageNet is too expensive, the paper could establish more evidence by recording the metrics as a function of $\epsilon$ where there should be a correlation.
- In similar fashion, the "visual residuals" analysis shown in Fig. 1 is thin by building on just one pair of images. Maybe this is a sign of my pareidolia, but I do see a rather clear "7" and opposed to the rather noisier sample of the robust network. It would be useful to show more examples (e.g., in the appendix).
- It is unclear how well these findings generalize to other forms of AT (e.g. as simple as FGSM or more complex), norms, network architectures (CNN vs. ViT) and datasets. 
- *Is this really generation?* The T-SNE plots are used as argument for generation rather than membership inference (memorization). I am not convinced that this is sufficient. The generated images lie different manifolds, but style-wise they are quite different, which may be the actual underlying cause. It would be great to show a few examples of generated images to the nearest training samples in a) image space and b) latent space.
- While I appreciate how upfront the limitations section is, I am still concerned about some of the limitations. E.g., L403 states “multiple runs frequently arrive at the same predicted class”. This may be conceived by a failure of PGDD to fully reconstruct all classes. Is there any evidence that PGDD can "reach" all classes?

### Questions
See above and additionally:
- It seems like there are some "attractor" samples which PGDD collapses to.  What is special about these samples or their classes? Is there a higher accuracy for them or any other correlation that the authors could establish?
- Is there a difference between and $L_\inf$ training?

### Soundness
3

### Presentation
3

### Contribution
3
