# Variational Inference for Cyclic Learning

- Decision: Accept (Poster)
- Scores: 8, 8, 4

## Abstract
Cyclic learning has emerged as a powerful paradigm for weakly-supervised learning. It involves training with pairs of inverse tasks and leverages cycle-consistency in the design of loss functions. However, its potential remains underexplored, as current methods are often narrowly focused on domain-specific implementations.
In this work, we develop generalized solutions for both pairwise cycle-consistent tasks and self-cycle-consistent tasks. By formulating cross-domain mappings as conditional probability functions, we reformulate the cycle-consistency objective as an evidence lower bound optimization problem via variational inference. Based on this formulation, we further propose two training strategies for arbitrary cyclic learning tasks: single-step optimization and alternating optimization.
Our framework demonstrates broad applicability across diverse tasks. In unpaired image translation, it offers a theoretical justification for CycleGAN and yields CycleGN—a competitive GAN-free alternative. 
In unsupervised tracking, following our conceptual design, CycleTrack and CycleTrack-EM achieve state-of-the-art results on multiple benchmarks.
This work establishes the theoretical foundations of cyclic learning and offers a general paradigm for future research.
The source codes for CycleGN and CycleTrack are publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper has a good intuition to use the latent variables bridging the two domains. They summarize the success of CycleGan by modeling the two terms of reconstruction and distribution alignment and propose several multi-variants like “CycleGN” (GAN-free)  and CycleTrack / CycleTrack-EM and achieve SOTA on unsupervised visual tracking tasks.

### Strengths
Sterngeths:


The paper establishes a framework to broad the cyclin-style loss functions in computer vision tasks. The paper has a clear concept by understanding the cycle term as reconstruction and the GAM term as the KL surrogate for making single-step optimization. Their proposed methods help address the failure conditions for the CycleGAN and are useful in the visual tracking tasks.

### Weaknesses
Weakness:


1 Overall the math deductions are good, but there are many typos and rigor issues, e.g.: In eq.3, where does the p_{data} come from?; Extra parenthesis in Eq. (16); what are the abbreviations IoU for in Eq. (17)? Should it be argmax based on the results in the table 3-4?


2 Figure 2 and Figure 3 are not clear enough to be understood. 


3 CycleGN is close but generally behind CycleGAN in the more challenging direction (photo→labels), suggesting that the Dsim/EM recipe doesn’t yet match a well-trained discriminator as a KL surrogate.

### Questions
Questions:


1  What are the abbreviations IoU for in Eq. (17)? Should it be argmax based on the results in the table 3-4?


2 Could you explain in more detail when your method will be successful? Will your methods be useful to the questions like domain adaptations in different modalities?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes a unified probabilistic framework to generalize cyclic learning, moving beyond the ad-hoc, task-specific implementations that currently dominate the field. The authors identify that while cycle-consistency is a powerful tool for weakly-supervised learning, it lacks a common theoretical foundation. To address this, they reformulate the cycle-consistency objective as a variational inference (VI) problem. The core of their approach is to model the cross-domain mappings (e.g., $A \rightarrow B$ and $B \rightarrow A$) as conditional probability functions and treat the intermediate generated data as latent variables. This allows them to re-cast the training objective as the optimization of an Evidence Lower Bound (ELBO) on the data log-likelihood.

From this single theoretical framework, the authors derive two distinct and general optimization strategies. The first is a "VAE-style" single-step loss that optimizes the full objective at once, which the authors show provides a new theoretical justification for the success of architectures like CycleGAN. The second is an "EM-style" alternating optimization algorithm that iteratively updates the forward and backward mappings, avoiding the need for an explicit KL divergence approximation (like a GAN's discriminator). The framework's effectiveness is demonstrated on two very different tasks: unpaired image translation, where their proposed GAN-free "CycleGN" is competitive with CycleGAN, and unsupervised visual object tracking, where their "CycleTrack" and "CycleTrack-EM" models establish a new state-of-the-art on multiple benchmarks.

### Strengths
- Novel Theoretical Contribution: Its primary strength is the novel and elegant reformulation of cycle-consistency as a variational inference (VI) problem. This connects a widely-used heuristic to fundamental probabilistic principles (like ELBO maximization, VAEs, and the EM algorithm) for the first time.

- Generalization: The framework is highly general, providing a unified theory for both paired cyclic tasks (like image translation) and self-cyclic tasks (like video tracking), which were previously treated with separate, task-specific methods.

### Weaknesses
- Analysis of EM-Style Failure Modes: The paper honestly presents failure cases (Fig. 4a) where the EM-based CycleGN achieves good cycle-reconstruction but produces poor-quality intermediate "fake" images. This suggests the model has learned an "incorrect mapping" that satisfies $g(f(x)) \approx x$ but where $f(x)$ is not a faithful member of the target domain $Y$. This is a crucial finding and a known risk of EM-style approaches converging to a local optimum. The paper would be improved by a deeper analysis of why this happens in the VI framework. Is it an inherent instability of the alternating optimization? Or does it confirm that the EM approach, by "lacking explicit constraints on latent variables" (Section 2.1), is more vulnerable to this specific failure mode than the single-step method, which explicitly constrains the intermediate variable with $D_{sim}$?

- Limited Scope of Image Translation Experiments: The validation of CycleGN is performed only on the Cityscapes (labels $\leftrightarrow$ photo) dataset. This is a highly structured translation task. The original CycleGAN paper demonstrated its robustness on much more "unstructured" and challenging tasks, such as horse $\rightarrow$ zebra or style transfer (Monet $\rightarrow$ photo). To compellingly claim CycleGN is a general, competitive alternative to CycleGAN, it should be tested on these more diverse and difficult translation tasks. It is possible the EM-style approach works well for structured tasks but struggles with more unconstrained mappings where a GAN's distributional matching is essential.

### Questions
Regarding the CycleGN failure case in Figure 4a (where reconstruction is good but intermediate images are poor), could you elaborate on the cause? Is this a local optimum that is inherent to the alternating EM optimization, or does this failure mode confirm that the EM approach is more vulnerable to "cheating" precisely because it lacks the explicit $D_{sim}$ (distributional) constraint on the latent variable that the single-step method has?

The claim that CycleGN is a general, competitive alternative to CycleGAN would be significantly strengthened by testing it on more unstructured translation tasks (e.g., horse $\leftrightarrow$ zebra, style transfer). Have you performed experiments on such tasks? How does the EM-style approach perform in these more unconstrained settings where the GAN-based $D_{sim}$ term is known to be critical?

### Soundness
4

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
This paper presents a general framework for training cyclic learning models, built on single-step and alternating optimization strategies. The framework's broad utility is shown in two areas: for unpaired image translation, it not only provides a theoretical basis for CycleGAN but also produces CycleGN and CycleTrack, a competitive model that doesn't require GANs. Overall, this work lays the theoretical groundwork for cyclic learning and offers a universal approach for subsequent research.

### Strengths
The paper presents a method (CycleGN) that avoids the bottleneck of the previously proposed cycle methods, like unstable adversarial optimization (GAN-style). This enables competitive results without explicitly using discriminators. The paper is well-structured and easy to follow.

### Weaknesses
1) The performance of the *CycleGN EM-based* approach was found to be *inconsistent across tasks* - showing stable results in some settings but degraded performance in others compared to CycleGAN (single-step with GAN).  This indicates that method still lack a unified mechanism to fully remove the D_{KL} surrogate across different problem domains.

  
2) A issue remains where models can achieve nearly perfect while the *intermediate translation remains unrealistic. This is an intrinsic limitation of cyclic training.

### Questions
**Q1** : The connection between the proposed EM-style framework and the classical EM algorithm remains unclear.  
    In the traditional EM formulation, the objective is to maximize the data likelihood.  
    Could the authors clarify how optimizing the cycle-consistency loss ($D_{cyc}$) in their setting corresponds to maximizing the likelihood?  
    Is it correct to interpret the cycle loss as a *surrogate reconstruction objective* that implicitly optimizes the likelihood function, similar to the role of the reconstruction term in variational inference?

    
 **Q2** :    Given the absence of explicit $D_{KL}$ control in the EM method, are there any theoretical or empirical evaluations confirming that the generated distributions $\mathcal{X}$ and $\mathcal{Y}$ approach the true priors, rather than merely achieving a cyclically consistent but unrealistic local optimum?
    

**Q3** :   Given the observed instability of results across different tasks, could the authors identify specific problem types or conditions under which the proposed EM-based framework is expected to outperform standard GAN-based models?  
    What are the authors’ insights regarding the comparative advantages of their approach?  
    In other words, what would constitute the \textit{ideal use case} where this framework would be preferable to a conventional GAN setup?

### Soundness
2

### Presentation
3

### Contribution
2
