# Test-Time Anchoring for Discrete Diffusion Posterior Sampling

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 8, 2, 2

## Abstract
We study the problem of posterior sampling using pretrained discrete diffusion foundation models, aiming to recover images from noisy measurements without retraining task-specific models. While diffusion models have achieved remarkable success in generative modeling, most advances rely on continuous Gaussian diffusion. In contrast, discrete diffusion offers a unified framework for jointly modeling categorical data such as text and images. Beyond unification, discrete diffusion provides faster inference, finer control, and principled training-free Bayesian inference, making it particularly well-suited for posterior sampling. However, existing approaches to discrete diffusion posterior sampling face severe challenges: derivative-free guidance yields sparse signals, continuous relaxations limit applicability, and split Gibbs samplers suffer from the curse of dimensionality. To overcome these limitations, we introduce **Anchored Posterior Sampling (APS)** for *masked diffusion* foundation models, built on two key innovations---*quantized expectation* for gradient-like guidance in discrete embedding space, and *anchored remasking* for adaptive decoding. 
Our approach achieves state-of-the-art performance among discrete diffusion samplers across linear and nonlinear inverse problems on the standard benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Anchored Posterior Sampling (APS) for masked discrete diffusion models. APS is built on two key innovations: (1) quantized expectation, which enables gradient-like guidance in discrete embedding space, and (2) anchored re-masking, an adaptive strategy for decoding the most informative tokens early. Experiments demonstrate that APS achieves competitive performance among discrete samplers on linear and nonlinear inverse problems and enables training-free stylization, often outperforming continuous diffusion baselines at a lower computational cost.

### Strengths
- To the best of my knowledge, this work is original. It tackles a significant challenge regarding posterior sampling with discrete diffusion models with a novel combination of ideas: Quantized Expectation provides a principled, gradient-like signal that is far more effective than prior derivative-free or relaxation-based methods; Anchored Remasking​ adapts the concept of "anchor tokens" from language modeling to the visual domain for inverse problems.
- As far as I checked, this work is methodologically sound, grounded in theoretical derivations (Thm 3.1 & 3.2) that provide a solid foundation for the empirical contributions. The experimental evaluation is comprehensive and rigorous.
- The paper is well-written and well-structured. The appendix provides substantial additional detail, ensuring reproducibility.

### Weaknesses
- While the paper provides theoretical bounds (Thm 3.1 & 3.2), the connection between these bounds and the final Algorithm 1 (APS) feels somewhat loose. The method's core strength lies in the engineering innovation of ​Quantized Expectation​ and the heuristic of ​Anchored Remasking, which are motivated intuitively but not derived directly as the solution to optimizing the presented bounds.
- The comparison of runtime (Table 6) is valuable, but it lacks a breakdown. The 100 inner optimization steps per reverse diffusion step represent a significant overhead. The paper does not ablate the number of inner steps or analyze the trade-off between performance and compute.
- The generality of APS across different discrete diffusion architectures (e.g., MaskGIT) is not demonstrated. To strengthen the claim of generality, the authors should apply APS to at least one other masked discrete diffusion model (even if smaller) on a subset of tasks. This would show that the innovations are not overly specific to MMaDA's architecture.

### Questions
- The paper highlights the speed of APS compared to continuous methods, but the 100 inner-loop steps per diffusion step represent a significant cost. Could you provide an ablation study showing the performance (e.g., PSNR/LPIPS) versus the number of inner steps (e.g., 10, 30, 50, 100) for a key task like 4x super-resolution?

### Soundness
3

### Presentation
4

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
This paper proposes a novel posterior sampling method for masked discrete diffusion models, aiming to solve inverse problems like super-resolution and deblurring without task-specific retraining. The method builds on two key ideas: quantized expectation for gradient-like guidance in discrete space, and anchored remasking for adaptive decoding. Theoretical bounds (LDDPS and LAPS) are derived to support training-free inference. Extensive experiments on FFHQ and ImageNet show strong results, outperforming discrete baselines and competing with continuous diffusions.

### Strengths
- The proposed methods, quantized expectation and anchored remasking, empirically address challenges in discrete diffusion posterior sampling, like non-differentiability and suboptimal decoding order. This makes the method practical and efficient.

- I like the solid theoretical explanations, which have clear derivations of upper bounds that enable reusing pretrained models. This is valuable for further investigation.

- The experiments demonstrate superior performance on linear and nonlinear tasks, with notable improvements in LPIPS and PSNR. The extension to stylization and editing adds practical utility.

- The writing is clear and well-organized.

### Weaknesses
The method's generalization across different tokenizers, datasets, and noise levels could be further explored. For instance, it relies heavily on LFQ tokenization. Could you discuss more on results for varying noise strengths and multimodal tasks to highlight robustness?

### Questions
Could you discuss more on results for varying noise strengths and multimodal tasks to highlight robustness?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper develops an inference time only algorithm for solving inverse problems using pre-trained masked diffusion models. The paper proposes the optimization of a variational objective at test-time together with various tricks such as expected codebook quantization and anchored remasking.

### Strengths
- The paper is timely and tackles a problem that could be highly interesting for the ICLR community. 
- The paper presents experiments on inverse problems and stylization and the comparisons are performed against many baselines.

### Weaknesses
My first concern is regarding the methodological aspect of the paper. Overall I found the mathematical details to be somewhat poorly presented and the theoretical derivations seem to be disconnected from what the authors actually implement in practice. 
- The authors define a joint probability distribution $p \_\varphi(x, z_{0:1} | y)$ in line 206. Given the definition in equation 6, this is not a probability distribution: the tokenwise conditional is $p \_\varphi(z^l _s | z_t, y)$ does not even integrate to one (over $z^l \_s$), it ingrates to $q(y|x \_\varphi(z \_t))$. Furthermore, following this definition we would have that the conditional is equal to , why does this make sense?  $q(y|x \_\varphi (z \_t))^L q(z \_s| z\_t, x \_\varphi(z_t))$. The authors are trying to tilt the unconditional backward distribution but this is wrong. 
- Algorithm 1 in the appendix seems to be disconnected from the derivations in the main appendix. First, the authors seem to be only considering a reconstruction and perceptual loss. There doesn't seem to be anything loss term related to the pre-trained model as one would expect from the theory but also in general from variational inference methods (see G2D2). Next, even assuming that the authors want to implement the $L \_{APS}$ loss, this would require drawing a random timestep at each gradient step and then optimizing the associated term. This is not what Algorithm 1 does as the timesteps are scanned in decreasing order. Overall the theoretical framing does not match the actual implementation, despite the authors claiming for example in line 302 that the objective (7) is optimized at each time step. 

Regarding the experimental evaluation it seems that the authors use the perceptual and L1 reconstruction loss only for their method and not the other baselines (at least it seems to me that they mention nowhere that they use the same losses for the other methods. The code is not provided so I am unable to check this in the codebase). Furthermore, they seem to have performed tuning over the weights for both terms. In my opinion this is unfair for the other methods and biases the results as the losses should boost the performance. Second, it also seems that the comparison with existing diffusion baselines is not done using the same base model, which is also concerning and biases the results 

My overall opinion of this paper is that it is not yet ready for publication due to the many inconsistencies in the methodology and also the experimental section.

### Questions
- Why not compare against the same baselines on all the tasks?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a novel approach for posterior sampling based on a pretrained discrete diffusion model. The authors present two theorems demonstrating: 

1) An objective for training a discrete diffusion model capable of posterior sampling;

2) An alternative form of this objective that enables the reuse of a pretrained discrete diffusion model for posterior sampling without requiring backpropagation through it.

Furthermore, the authors propose two additional techniques: Quantized Expectation for differentiable likelihood evaluation and Anchored Remasking for adaptive unmasking. Finally, the paper reports experimental results that showcase the model’s capabilities.

### Strengths
- The paper is clearly written and provides detailed, step-by-step explanations, supported by extensive references to prior work.

- The proposed methods — Quantized Expectation and Anchored Remasking — have a notable positive effect on the final model’s performance.

- The experimental evaluation is comprehensive and effectively demonstrates the model’s efficacy across multiple scenarios.

### Weaknesses
Both theorems presented in the paper contain errors due to incorrect operations involving probabilistic distributions. Specifically, in Equation (6) (page 4), the authors incorrectly parameterize the conditional distribution as follows:

$$p_\phi(z^l_s \mid z_t, y) := q(z^l_s \mid z^l_t, x_\phi(z_t)) q(y \mid x_\phi(z_t))$$

The fundamental issue lies in the fact that the distribution on the left-hand side ($p_\phi(z^l_s \mid z_t, y)$) is defined over $z^l_s$ conditioned on $y$, whereas the right-hand side ($q(z^l_s \mid z^l_t, x_\phi(z_t)) q(y \mid x_\phi(z_t))$) represents a joint distribution over $z^l_s$ and $y$. In probabilistic modeling, these two forms cannot be interchanged. This inconsistency becomes critical because both theorems explicitly rely on treating these as probabilistic distributions. The issue appears in the Proof of Theorem 3.1 (the transition from Equation 10 to the equation immediately following it, page 16) and in the Proof of Theorem 3.2 (the transition from Equation 13 to Equation 14, page 20). 

While the final empirical procedure performs well, this success can be attributed to the structure of the training objective $\mathcal{L}_{\text{APS}}$ (Theorem 3.2, page 5) rather than to the theoretical results. Specifically, the right-hand term of the objective encourages the lightweight model’s output to align with a particular $y$, whereas the left-hand term regularizes the model by constraining it to remain close to the pretrained diffusion model. 

In conclusion, the current theoretical framework does not rigorously justify the proposed practically effective method. Given that the theoretical contribution constitutes a central component of the paper, this issue substantially undermines the overall validity of the work.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
3

### Contribution
1
