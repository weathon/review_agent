# Log-Normal Multiplicative Dynamics for Stable Low-Precision Deep Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 4, 8, 6

## Abstract
We design a new algorithm for stable low-precision deep learning motivated by the robustness of biological neural networks. It is well known that the stationary distribution of spine sizes follows a log-normal distribution and arises from noisy multiplicative dynamics. Building on these synaptic fluctuations that underlie neural computation, we propose the Log-normal Multiplicative Dynamics (LMD) algorithm for stable learning under low-precision computation. The method is derived by using variational training with a log-normal posterior distribution over the weights. LMD is a multiplicative weights update method that overcomes the scalability challenges seen in other multiplicative updates. We show empirically that LMD can learn stably under low-precision matrix multiplications during forward passes. It also gives accurate results for training-from-scratch for Vision Transformer and GPT-2 scale architectures. These results suggest that multiplicative dynamics, a biological feature, can maintain performance under low-precision computation, a promising direction for future energy-efficient hardware.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces LMD, a multiplicative weight update algorithm that draws on log-normal dynamics observed in biological synaptic spines. Experiments on ImageNet and OpenWebText show that LMD maintains stability and delivers robust performance even with low-precision forward passes. It also exhibits strong results when applied to GPT-2 and ViT models.

### Strengths
- This work achieves the first successful training of large networks using multiplicative weight updates (MWU). The proposed LMD algorithm even outperforms AdamW on ImageNet, clearly validating the effectiveness of the method.

### Weaknesses
- While this study builds primarily on the Lie-group Bayesian learning framework (Kiral et al., 2023), it fails to clearly delineate the key distinctions between LMD and the prior method, nor does it offer a direct performance comparison. The absence of such analysis undermines the apparent novelty and impact of the proposed approach.

- There is no theoretical support for why LMD achieves higher accuracy or performs well in low-precision scenarios. 

- Low-precision training is not a particularly important issue. Fig. 1 also does not show that LMD has a substantial improvement over the common AdamW.

### Questions
- On the theoretical side, a discussion of why LMD is expected to work better than previous algorithms would be welcome.

-  LMD uses Monte Carlo (MC) sampling. Can you explain the impact of S on computational cost? I don’t quite understand what problem the introduction of MC solves.

- Why it is effective in large networks, or if it works in both large and small networks?

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
This paper proposes the Log-normal Multiplicative Dynamics (LMD) algorithm for stable learning under low-precision computation (MXFP6).

### Strengths
1）This paper proposes the Log-normal Multiplicative Dynamics (LMD) algorithm for stable learning under low-precision computation and validated on vision and language models.

### Weaknesses
1)The paper lacks an evaluation of the actual training speed and memory consumption, likely because the proposed parameter optimization method is relatively complex and computationally intensive.

2)The ablation study is insufficient, as the main text does not include a quantitative comparison with the standard MWU method.

### Questions
1) The paper uses the low-precision MX data format for forward propagation and BF16 for gradient backpropagation. What would happen if the gradient backpropagation during training also used the MX data format?

2) How does the performance of this paper's model compare to that of a model trained with BF16 and deployed on low-precision hardware?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Inspired by the log-normal distribution of synaptic spine sizes in biological neural networks, this paper proposes the Log-normal Multiplicative Dynamics (LMD) algorithm for stable learning under low-precision computation. 

The LMD algorithm is built on the Lie group Bayesian learning rule, giving rise to multiplicative weight updates which achieve large dynamic range at limited bit widths. The algorithm also incorporates multiplicative noise injections which stabilize weights for low-precision forward passes, as well as multiplicative weight decay which regularizes and prevents excessive weight growth.

Experiments with the Vision Transformer and GPT2 demonstrates state-of-the-art accuracies when performing low-precision forward passes using the MX data formats. This is good news for the design of energy-efficient hardware for inference.

### Strengths
**Significance**
* Promising results on vision transformer and GPT2 architectures

**Originality**
* Proper treatment of multiplicative noise injection
* Combination of many useful ideas (signSGD, signed weights trick, decoupling of gradient and regularizer in the momentum, multiplicative weight updates, multiplicative weight decay, multiplicative noise injection) and showing that they actually work together well.

**Clarity**
* Simple clear implementation in PyTorch for existing models, via drop-in replacement of the usual Adam optimizer, and a simple adaptation of the existing parameter initialization scheme.

**Quality**
* High quality ablation studies show effects on accuracy, regularization and low-precision forward passes that come from multiplicative weight updates, weight decay and noise injection in the learning algorithm.

### Weaknesses
1. Hard to understand LMD Algorithm 1 without first understanding Lie Group BLR. One should assume that the reader is not familiar with LGBLR. It takes a while to see that Eq 5 and Algorithm 1 are largely the same, after features like signSGD and gradient/regularization separation are removed. For instance, how does one decide to add signSGD to v_temp and not to the regularizer R? Perhaps all this can be explained in more detail in the Supplementary Material, expanding the discussion in Section 2.3. After all, this paper is ultimately about the LMD Algorithm.

2. Scaling the gradient by the weight is only briefly mentioned in Eq 5, Section 2.3 and Section 4.2. Does this scaling come from Lie Group BLR? Please do explain this a little if you are going to talk about its effect on regularization. Intuitively, how does it arise naturally from multiplicative Lie groups, but somehow not in classical MWU? If there is not enough space, this discussion could go in the Supplementary Material.

3. In Section 3.2 on Initialization, it took me a while to understand that initialized values are sampled using m\epsilon where \epsilon is logNormal(0, \sigma^2), and that you need to apply E[m\epsilon] = m \exp(\sigma^2/2) to find the mean of the initialized values. This could be made more explicit.

### Questions
1. How is it different from replacing every parameter m in a given model with \exp \mu, rederiving the gradient updates, applying weight decay and applying noise additively to \mu?

2. Any kind of indication how much slower it is to run MWU on GPUs for training large neural networks, than running additive updates on GPUs?

3. Why then did the sign function in Eq 1 disappear in Eq 5? Is it because the signs of the weights are preserved in LGBLR?

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
3

### Summary
This work proposes parameter update rule based on multiplicative updates with log-normal noise. Prior works have attempted one of these two contributions but not both. The authors show that the new rule (LMD) enables stable low-precision training.

### Strengths
1. The paper's contributions enable low-precision (and therefore, low energy) training of large neural network which is timely and useful.
2. The paper's claims are well laid out and supported.

### Weaknesses
1. Evaluation is limited to only two model + dataset pairs.
2. Evaluation is limited to standard optimizers without focusing on low-precision schemes.

### Questions
Q1: How do you compare against other low-precision training schemes (e.g., [1] or [2])?

Q2: LMD seems to have overall lower training performance in Table 1. Why is that? Are there practical limitations to its convergence?

Q3: Could you estimate the potential compute savings? I know you cannot measure them due to emulation.

[1] : https://arxiv.org/abs/1803.03383

[2] : https://arxiv.org/abs/1710.03740

### Soundness
4

### Presentation
3

### Contribution
4
