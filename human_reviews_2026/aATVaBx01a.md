# Self-Supervised Diffusion Model Sampling With Reinforcement Learning

- Decision: Reject
- Scores: 0, 2, 2, 6

## Abstract
Diffusion models have established themselves as the state-of-the-art for generative modeling, dethroning Generative Adversarial Networks (GANs) by generating higher-quality samples while remaining more stable throughout training. However, diffusion models generate samples iteratively and remain slow at inference time. Our work proposes to leverage reinforcement learning (RL) to accelerate inference by building on the recent framing of diffusion's iterative denoising process as a sequential decision-making problem. Specifically, our approach learns a scheduler policy that maximizes sample quality while remaining within a fixed budget of denoising steps. Importantly, our method is agnostic to the underlying diffusion model and does not re-train it. Finally, unlike previous RL approaches that rely on supervised pairs of noise and corresponding denoised images, our method is self-supervised and directly maximizes similarity in dataset feature space. Overall, our approach offers a more flexible and efficient framework for improving diffusion model's inference in terms of speed and quality.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This work introduces a reinforcement learning–based scheduler policy that optimizes sample quality under a fixed number of denoising steps, without retraining the diffusion model. The proposed self-supervised method accelerates inference efficiently while maintaining or improving generation quality, offering a flexible and model-agnostic enhancement to diffusion sampling.

### Strengths
The literature review seems to be comprehensive.

### Weaknesses
* The writing and the logic of this paper can be improved greatly. For example, the introduction is almost as the same length as the abstract. With such a short paragraph, the authors do not make it clear about the background and the motivation of their proposed methods. Overall, this paper gives me a feeling that the authors are a little bit rush to catch the ICLR due, and does not finish the paper.

* The novelty is limited. This paper is not the first paper to introduce reinforcement learning to learn a noise schedule in diffusion model sampling process. The authors should discuss their difference, and novelty compared the previous methods.

* The experiment is also not comprehensive enough. There are not compared methods in the experiment. The experiments are also conducted on few small-scale datasets. More evaluations and ablation study are needed to justify the advantage of the proposed method. Also, the experiment analysis is very limited.

* Typos: There are lots of typos in this paper, and the authors should fix them carefully. For example, in Line 121, “Or” after the comma should be lower bracketed.

### Questions
* I believe the quadric sampling performs the best in CIFAR10 and CelebA-HQ. Why do the authors not include the results of using quadric sampling?

* I am wondering if there is a constraint on the output of the policy $\pi_\theta$. As the experiment are conducted with the same time steps (like in 10 steps). How to ensure the proposed method can just complete the sampling process in 10 steps?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a reinforcement learning (RL) framework to accelerate diffusion model sampling without retraining the base model. Traditional diffusion models generate high-quality images but are slow at inference because of iterative denoising. The authors treat the denoising process as a sequential decision-making task, where an RL agent learns an optimal noise schedule to balance speed and sample quality.

### Strengths
1. Positioning RL as a learned scheduler for diffusion sampling is a forward-looking direction with clear potential: it decouples the sampler from the base model, enables adaptive step allocation under budgets, and could generalize beyond image generation to other iterative inference/integration tasks.

2. Self-Supervised Design: Avoids dependence on paired data or teacher models, reducing computational cost and improving scalability.

3. General Applicability: Works as a plug-in sampler for any pretrained diffusion model without retraining.

### Weaknesses
1. Limited Evaluation: Only small-scale datasets (CIFAR-10, CelebA-HQ) — no large or diverse benchmarks to test generalization.

2. Unclear Comparisons: Some baselines (e.g., DPM++) perform poorly but reasons are not analyzed; RL results on second-order solvers are weaker.

3. Missing Ablation Studies: No detailed analysis on hyperparameters, reward sensitivity, or computational overhead.

4. Paper organization & contribution strength. The manuscript’s sectioning and exposition require substantial tightening (clearer motivation, positioning vs. prior work). As presented, the contribution appears incremental with limited novelty and empirical breadth, which falls short of the ICLR acceptance bar.

### Questions
1. High-resolution & scale. Modern diffusion systems often operate at 512×512 or 1024×1024. How does the proposed RL scheduler perform at higher resolutions and on large-scale datasets (e.g., ImageNet-256/512)? Please report both quality (FID/KID) and wall-clock latency.

2. Real speedup under matched quality. Beyond NFEs, what is the end-to-end latency per image (and throughput) on a standard GPU compared to strong samplers when targeting the same quality? Include hardware, batch size, and confidence intervals.

3. Practicality: training cost and transfer. What compute/time is required to train the RL policy per model/dataset, and does the policy transfer across architectures or noise parameterizations? If retraining is needed, what’s the break-even point where the amortized gains outweigh the training overhead?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a reinforcement learning (RL)-based scheduler to accelerate diffusion model sampling. Unlike prior work that retrains or distills diffusion models, this approach learns an optimal denoising schedule via RL without retraining or requiring paired supervision. The method frames diffusion inference as an MDP, where each denoising step is an action, and a Gaussian policy learns to decide adaptive noise updates. A self-supervised reward is designed by measuring the maximum similarity between generated samples and real data features (using Inception-v3 embeddings). The approach is evaluated on CIFAR-10 and CelebA-HQ using DDIM and IADB diffusion models, achieving better FID scores with fewer function evaluations (NFEs).

### Strengths
1. The framing of diffusion sampling as a sequential decision-making problem is elegant and consistent with the Markovian structure of denoising.

2. The RL component (PPO-based scheduler) is model-agnostic—it does not modify or retrain the base diffusion network, enabling plug-and-play acceleration across architectures (DDIM, IADB).

3. Using feature-space similarity as a reward proxy allows the method to circumvent supervised alignment and label requirements.

### Weaknesses
1. A significant concern is that the proposed “self-supervised” reward actually depends on an Inception-v3 feature encoder pretrained on ImageNet, which likely leaks high-level real-image information into the training process. This coupling may bias the RL scheduler toward reproducing specific training-set patterns and artificially improve FID, since both the reward and evaluation metrics are derived from the same embedding space. Consequently, the reported performance gains might not reflect genuine improvements in sample quality or generalization. Future work should decouple the reward signal from FID features or use domain-specific encoders to ensure fairness and generalizability. 

2. Despite claiming theoretical soundness, the paper lacks a rigorous proof of convergence or guarantee of policy optimality. The “theoretical grounding” in Appendix A only discusses FID variance bounds, not RL convergence or sampling correctness.

3. The writing of the paper and the figure 1 seem inconsistent and is confusing in terms of whether the RL training leverages sparse vs dense reward. If the training requires sparse reward design, the PPO training will be unstable and requires more explorations and data collection for the RL algorithm to converge. The paper would benefit from a dense-reward design so that PPO works better though it comes with more computational cost. 

4. Key learning-based baselines [1,2] are missing. Without these, it’s difficult to attribute performance gains to the proposed self-supervised RL design.

5. No ablation studies are presented to isolate the contribution of self-supervision vs. policy learning.

[1] Lu, Cheng, et al. "Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models." Machine Intelligence Research (2025): 1-22.

[2] Xue, Shuchen, et al. "Sa-solver: Stochastic adams solver for fast sampling of diffusion models." Advances in Neural Information Processing Systems 36 (2023): 77632-77674.

### Questions
1. You claim that the method is self-supervised, yet the reward relies on features extracted from a pretrained Inception-v3 model. How do you reconcile this with the self-supervised claim, given that Inception features encode semantic information learned from large-scale labeled datasets?

2. The reward is sparse (terminal-only) and non-differentiable. How stable is PPO training under this setup? Did you attempt intermediate-step rewards or reward shaping to improve credit assignment?

3. PPO is known to be sensitive to hyperparameters. Could you report variance over multiple seeds or provide details on convergence behavior?

4. How does your policy network scale with higher-dimensional datasets or larger image resolutions (e.g., 512×512 or beyond CIFAR-10/CelebA-HQ)?

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
4

### Summary
This paper introduces a reinforcement learning (RL) approach to accelerate diffusion model sampling by learning an adaptive noise schedule policy. The method frames the diffusion sampling process as an RL episode where each denoising step corresponds to an environment step. The key innovation is a self-supervised reward based on maximum pairwise similarity between generated samples and real data in a pretrained feature space (Inception-v3), eliminating the need for teacher models or paired data. Experiments on CIFAR-10 and CelebA-HQ with DDIM and IADB models show that the learned scheduler achieves competitive or better FID scores compared to uniform and cosine schedules, particularly in low-compute regimes (10-30 NFEs).

### Strengths
- **Self-Supervised Reward:** Uses feature similarity instead of supervised signals, enabling broader applicability.
- **Model-Agnostic:** Works with any pretrained diffusion model (DDIM, IADB) and sampler order.
- **Efficiency:** Achieves better FID with fewer NFEs, especially in resource-constrained settings.
- **Theoretical Insight:** Appendix A provides a bound on FID estimation error due to finite sampling.

### Weaknesses
- **Inconsistent Second-Order Results:** The method underperforms on second-order solvers for CIFAR-10, raising questions about its generalizability.
- **Limited Reward Analysis:** The choice of Pearson correlation is not thoroughly justified; other metrics (e.g., cosine similarity) are not compared.
- **Ablation Studies Missing:** No analysis of the impact of batch size, feature extractor choice, or policy architecture.
- **Narrow Evaluation:** Only FID is used; no diversity metrics (e.g., Precision/Recall) or human evaluation.

### Questions
1. **Why does the method underperform on second-order solvers for CIFAR-10?** Is this due to the reward function or the complexity of the policy?
2. **Have you experimented with other similarity metrics (e.g., cosine similarity) or feature extractors?** An ablation would strengthen the reward design.
3. **How does the batch size of real samples \(D\) affect performance?** Is there a trade-off between reward quality and computational cost?
4. **Can the policy generalize to NFE budgets not seen during training?**
5. **Have you considered using dense rewards (e.g., intermediate similarity scores) to guide learning?**

### Soundness
3

### Presentation
3

### Contribution
4
