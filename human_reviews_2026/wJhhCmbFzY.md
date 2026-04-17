# Discrete Variational Autoencoding via Policy Search

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Discrete latent bottlenecks in variational autoencoders (VAEs) offer high bit efficiency and can be modeled with autoregressive discrete distributions, enabling parameter-efficient multimodal search with transformers. However, discrete random variables do not allow for exact differentiable parameterization; therefore, discrete VAEs typically rely on approximations, such as Gumbel-Softmax reparameterization or straight-through gradient estimates, or employ high-variance gradient-free methods such as REINFORCE that have had limited success on high-dimensional tasks such as image reconstruction. Inspired by popular techniques in policy search, we propose a training framework for discrete VAEs that leverages the natural gradient of a non-parametric encoder to update the parametric encoder without requiring reparameterization. Our method, combined with automatic step size adaptation and a transformer-based encoder, scales to challenging datasets such as ImageNet and outperforms both approximate reparameterization methods and quantization-based discrete autoencoders in reconstructing high-dimensional data from compact latent spaces.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Discrete Autoencoding via Policy Search (DAPS), a novel training framework for discrete VAEs that avoids reparameterization or straight-through gradient estimation. DAPS formulates encoder learning as a KL-regularized policy search problem, and the parametric encoder is updated via weighted maximum likelihood using importance sampling. A trust-region parameter is adapted automatically using the effective sample size (ESS) to ensure stable training. Experiments on MNIST, CIFAR-10, ImageNet-256, and LAFAN motion data show that DAPS outperforms Gumbel-Softmax, GR-MCK, FSQ, and VQ-VAE in reconstruction quality and bit efficiency.

### Strengths
1. DAPS is a principled training framework that reformulates discrete VAE encoder optimization as a KL-regularized policy search problem, enabling stable and scalable training without reparameterization. The methods leverage modern RL techniques that have been underexplored in discrete VAEs.
2. DAPS demonstrates state-of-the-art reconstruction quality on ImageNet-256, and outperforms GR-MCK and Gumbel-Softmax in both log-likelihood and FID across MNIST, CIFAR-10, and ImageNet. It scales efficiently to sequence lengths of 1,024.
3. The step size of DAPS adapts automatically via ESS, eliminating the need for manual tuning of the trust-region parameter.

### Weaknesses
1. The baselines involved in the experiments lack the most recent work in 2024/2025; therefore, it is not clear how DAPS compares to recent SOTA methods.
2. Section 3 states that DAPS avoids computationally expensive and high-variance backpropagation through time by using weighted maximum likelihood. However, the paper doesn't empirically quantify the computational savings compared to Gumbel-Softmax or GR-MCK.
3. The abstract claims a 20% improvement on FID  for ImageNet 256. It would be beneficial to have a hypothesis test to determine whether this improvement is significant.

### Questions
1. Section 5 mentions that DAPS generates coherent and high-quality motion trajectories for LAFAN, but Figure 3 only shows a single frame without quantitative motion quality metrics. Could you provide quantitative metrics comparing DAPS against baselines on motion-specific evaluation criteria, and clarify how 'coherence' was assessed (e.g., via human evaluation or physical plausibility checks)?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces DAPS (Discrete Autoencoding via Policy Search), a novel method for training discrete variational autoencoders (VAEs) that leverages policy search techniques from reinforcement learning. The core idea is to frame the encoder update as a KL-regularized policy optimization problem, avoiding backpropagation through discrete sampling and enabling the use of autoregressive encoders. The method is evaluated on image (MNIST, CIFAR-10, ImageNet) and motion (LAFAN) datasets, demonstrating improved performance in reconstruction quality and downstream tasks like robot control.

### Strengths
The integration of policy search principles (like REPS and trust-region adaptation via ESS) into discrete VAE training is a novel and potentially impactful contribution.

The paper provides extensive experiments, showing that DAPS can scale to challenging datasets like ImageNet and outperform strong baselines such as Gumbel-Softmax, GR-MCK, FSQ, and VQ-VAE on key metrics (e.g., FID, log-likelihood).

The method demonstrates stable training.

### Weaknesses
The paper's logic is often indirect and difficult to follow. The connection between the core objective (discrete autoencoding) and the RL framework is not smoothly bridged, making the "big picture" and the intuitive justification for the approach unclear. In fact, both sides can be directly bridged via reverse KL minimization.

The notation is frequently confusing and inconsistent. 
- A critical example is the description of the parametric encoder $q_{\theta}(z|x)$. It is initially presented as a standard categorical distribution, but later described as an autoregressive model decomposed into a product of conditional (categorical) distributions. This conflict is deeply confusing. Furthermore, Figure 1 visually suggests a non-autoregressive structure (e.g., $z_1$, $z_2$, $z_3$ are not shown as inputs for $z_4$), creating a direct inconsistency between the text, the formula, and the diagram.
- Eqs. (3)-(5) are presented with terminology and notations from reinforcement learning, despite Section 3 being mainly about discrete autoencoding. At least, the Advantage $A(z,x)$ should be more explicitly and clearly connected to the VAE objective.
- The recognition loss in Line 9 of Algorithm 1 is different from Eq. (6), particularly considering the gradient with respect to the parameter $\theta$.

The introduction of the constraint $D_{KL}[q(z|x) || q_{\theta}(z|x)] \le \epsilon_{\eta}$ in Eq. 4 actually changes the optimization from a reverse KL (in the standard ELBO) to a forward KL w.r.t. parameter $\theta$. Reverse KL is mode-seeking, while forward KL is mode-covering. The paper provides no discussion or analysis of this fundamental shift, its implications for the latent representation, or its potential to cause conflicting behaviors during training.

The inclusion of a "Goal-Conditioned Robot Control" experiment feels disconnected from the primary narrative of discrete autoencoding. The paper fails to establish a clear, compelling link between this experiment and the core contribution discrete autoencoding.

### Questions
In Table 1, were the encoder architectures for all baseline methods (Gumbel, GR-MCK, FSQ, VQ-VAE) identical to the one used for DAPS? If not, how can we be sure that the performance gains are due to the proposed training algorithm and not a more powerful or better-suited architecture? Please clarify this in the experimental setup.

Please see also the above Weaknesses.

### Soundness
2

### Presentation
2

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
The main challenge in training discrete VAEs lies in backpropagating through the discrete sampling process. In this work, the authors propose Discrete Autoencoding Policy Search (DAPS), a novel training framework for discrete VAEs inspired by policy search methods from reinforcement learning, particularly Relative Entropy Policy Search (REPS). The method exploits the gradient of a non-parametric encoder (which can be optimized in closed form but not directly sampled) to update the parametric encoder without relying on reparameterization, while also incorporating automatic step-size adaptation. This approach eliminates the need for differentiable soft relaxations or zero-order gradient estimators, which often lead to biased or high-variance gradients and unstable training.

### Strengths
Given the increasing interest in learning discrete representations in the literature, I believe exploring new methods that enable the effective training of discrete latent variable models is relevant. While connections with reinforcement learning have previously been explored to address the challenge of backpropagating through discrete sampling (most notably through methods like REINFORCE) to the best of my knowledge, this is the first work to apply particle-based policy search methods to train discrete VAEs. This approach effectively circumvents the reparameterization problem and enables more stable training procedures. The authors also show promising results on complex, high-dimensional datasets, additionally highlighting the potential of this method for broader applicability in downstream robotics tasks.

### Weaknesses
-	**Method description.** Overall, I believe the description of the method could be improved. While some aspects may be intuitive for readers with a reinforcement learning background, they may appear opaque to others with expertise in deep generative models but not reinforcement learning (such as myself). For instance, why is the advantage function used instead of the reward function? In Equation (5), the loss for updating the encoder involves sampling from the parametric posterior, why does this not introduce the same backpropagation issue? I noticed that the training algorithm includes a stop-gradient operator, but I believe this should be explicitly discussed in the main text, since a central aspect of the method is preventing backpropagation through the discrete sampling process.

-	**Writing.** Additionally, I think the overall writing could be improved. In the introduction, several acronyms (e.g., ELBO) are not defined until later sections, and some variables referenced in the algorithm are never described in the text. Likewise, the metrics reported in the tables are not defined (e.g., log p, L2).

-	**Method.** While I find the framework interesting, it's unclear to me the ratio between performance gain and computational efficiency, as it is something that the authors do not address. They mention efficiency in terms of memory relative to Gumbel-Softmax-based methods, but not with respect to other baselines, nor in terms of training time. Also, as I stated before, I think key aspects of the method, such as the sampling procedure, remain unclear to me after reading the paper. I would appreciate if the authors can clarify it during the rebuttal.

-	**Evaluation.** I find the evaluation of the method limited. While the authors provide some reconstruction examples, showing that qualitatively their method performs comparably to state-of-the-art approaches, they do not include quantitative metrics to assess reconstruction quality, such as PSNR, comparing original and reconstructed samples. For generation, although FID scores are reported for some methods, the sample size used is not specified, which is important for comparison with state-of-the-art results. Additionally, it is unclear why these metrics could not be obtained for certain model configurations. Regarding the latent representations, the authors do not analyze their use. This is a notable limitation of VQ-VAEs that FSQ aims to address. It would be interesting to investigate how this alternative training framework, which allows for more stable training, impacts posterior inference. Furthermore, it would be valuable to study how the dimensionality of the latent space influences overall performance (could the improved training allow for more compact latent spaces?), especially since the latent representations appear to be high-dimensional. Also, I think the use of the ELBO as evaluation metric can be misleading, for example, a model with posterior collapse ELBO may still be high because reconstruction is good, but the latent code carries no information.

### Questions
-	Would this method still be beneficial when considering simpler models without an autoregressive structure, and a single vector as latent representation? One of the motivations of this approach is to improve memory efficiency with respect to approaches such as Gumbel-Softmax, does the same argument apply in simpler scenarios?
-	In this context, is generation performed by sampling from the uniform prior assumed during training? Similar to VQ-VAE, does this method require training an autoregressive prior in a second stage? How is the generation process?
-	In the appendix, the authors present label-conditioned samples. How is label-conditioned generation implemented? In the downstream robotics example, the high-level policy is also conditioned on language information, how is this text-conditioning realized in the DAPS generative process? 
-	According to the authors, what are the main limitations of the method, and what are the primary directions for future research?


I would be happy to increase my score if the authors adequately address the main weaknesses and questions.

### Soundness
2

### Presentation
2

### Contribution
3
