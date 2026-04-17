# Half-order Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer

- Decision: Accept (Oral)
- Scores: 6, 8, 6

## Abstract
The probabilistic diffusion model (DM), generating content by inferencing through a recursive chain structure, has emerged as a powerful framework for visual generation. After pre-training on enormous data, the model needs to be properly aligned to meet requirements for downstream applications. How to efficiently align the foundation DM is a crucial task. Contemporary methods are either based on Reinforcement Learning (RL) or truncated Backpropagation (BP). However, RL and truncated BP suffer from low sample efficiency and biased gradient estimation, respectively, resulting in limited improvement or, even worse, complete training failure. To overcome the challenges, we propose the Recursive Likelihood Ratio (RLR) optimizer, a Half-Order (HO) fine-tuning paradigm for DM. The HO gradient estimator enables the computation graph rearrangement within the recursive diffusive chain, making the RLR's gradient estimator **an unbiased one with lower variance** than other methods. We theoretically investigate the bias, variance, and convergence of our method. Extensive experiments are conducted on image and video generation to validate the superiority of the RLR. Furthermore, we propose a novel prompt technique that is natural for the RLR to achieve a synergistic effect. The implementation is available at https://github.com/RTkenny/RLR-Optimizer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an innovative "half-order" fine-tuning paradigm, making a substantial contribution to diffusion model optimization. By intelligently combining three gradient estimation strategies, it achieves a high level of theoretical and practical application. Rigorous mathematical proofs and extensive experimental validation demonstrate the RLR method's significant advantages over traditional approaches.

### Strengths
1. The innovative concept of "half-order" fine-tuning paradigm is proposed, which fills the gap between traditional first-order and zero-order methods.
2. FO, HO and ZO complement each other's strengths and find the optimal balance between variance and computational cost by optimizing the h and j parameters, taking into account the actual computational budget constraints.

### Weaknesses
1. The problem with FO is its high cost, and the problem with ZO is its high variance, but the author does not provide a clear analysis to explain this. For example, for a specific scenario, how many NFEs are needed for FO, ZO, and HO respectively, how is this calculated, what are the variances of these three, and why is there such a large variance problem. I think this needs a clearer analysis.
2. The visualization results look average, and the improvement is not significant enough. Lacks comparison with recent works such as FlowGRPO and ReFL.

### Questions
I wonder when I use ZO, when the noise is large, do I still need to go through the entire denoising trajectory to get the reward? If this is the case, then ZO also seems to have a high cost problem, which might be solved by process reward model such as SPO.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the crucial task of efficiently aligning diffusion models (DMs) to meet downstream application requirements after pre-training.   Contemporary fine-tuning methods, such as Reinforcement Learning (RL) and truncated Backpropagation (Truncated BP), suffer from high variance and biased gradient estimation, respectively.  The authors propose the Recursive Likelihood Ratio (RLR) optimizer, a novel "Half-Order" (HO) fine-tuning paradigm. RLR constructs a new gradient estimator by cleverly combining First-Order (FO), Half-Order (HO), and Zeroth-Order (ZO) estimators within the recursive chain. The paper theoretically proves that the RLR estimator is unbiased (overcoming the defect of truncated BP) and has a lower variance than RL/ZO methods. Extensive experiments on text-to-image and text-to-video tasks validate the superiority of RLR. It not only outperforms baseline methods (like DDPO and Alignprop) on multiple reward benchmarks, but critically, it also avoids the "model collapse" problem caused by truncated BP9. Furthermore, the paper proposes a novel prompt technique called "Diffusive Chain-of-Thought" (DCoT), which synergizes naturally with RLR's HO estimator, allowing the model to optimize for specific generation scales (e.g., "fine-grained" details).

### Strengths
- This paper addresses the crucial task of efficiently aligning foundation diffusion models. This represents a highly important and practically valuable problem. 
- This paper proposes the Recursive Likelihood Ratio (RLR) optimizer, a novel "Half-Order" (HO) fine-tuning paradigm that successfully overcomes these challenges. T
- he paper also introduces a novel prompt technique that synergizes naturally with the RLR optimizer, further enhancing the originality of the contribution. 
- The paper rigorously demonstrates its method's advantages in terms of bias, variance, and convergence from both theoretical and experimental standpoints, while also proving its practical effectiveness.

### Weaknesses
- The paper exhibits significant inconsistencies in its core methodology description, particularly regarding the sampling strategy for the Half-Order (HO) sub-chain starting point, $j$. In Section 4.2 (Methodology), the paper describes $j$ as being sampled from a categorical distribution based on gradient norms. However, in Section 5.3 (DCoT Experiment), $j$ is described as being selected from a uniform distribution ($j \sim \mathcal{U}(1, T-h)$). This contradictory description makes it impossible to determine which sampling strategy the standard implementation of RLR is supposed to use.
- Furthermore, the implementation of DCoT (Diffusive Chain-of-Thought) introduces a critical external dependency. As shown in Section 5.3 and Appendix F, DCoT relies on an external Large Language Model (LLM) to generate the 'coarse-mid-fine' grained prompts. This raises doubts about its robustness in practical deployment.

### Questions
- As in weakness, why different descriptions occur, and how to choose the sampling strategy?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a Half-order Fine-tuning method for efficiently adapting large-scale diffusion models (e.g., Stable Diffusion) to downstream datasets.  The authors also present a theoretical proof that RLR is unbiased, has lower variance, and enjoys convergence guarantees. The experiments demonstrate RLR’s efficiency.

### Strengths
- The paper presents a novel fine-tuning scheme and devises gradient estimators for the diffusion model’s chain-of-thought, which appears genuinely innovative.
- The theoretical analysis is careful and offers a credible justification for the proposed approach.

### Weaknesses
- Missing a comparison with related diffusion model fine-tuning baselines, such as D3PO[1].
- The experiments are limited to SD 1.4 and SD 2.0, which are now dated. Moreover, the method’s generalization to the Flux architecture remains unclear.

$\text{[1] Yang, Kai, et al. "Using human feedback to fine-tune diffusion models without any reward model." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.}$

### Questions
NA.

### Soundness
3

### Presentation
3

### Contribution
3
