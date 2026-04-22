# Sample Reward Soups: Query-efficient Multi-Reward Guidance for Text-to-Image Diffusion Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Recent advances in inference-time alignment of diffusion models have shown reduced susceptibility to reward over-optimization. However, when aligning with multiple black-box reward functions, the number of required queries grows exponentially with the number of reward functions, making the alignment process highly inefficient. To address the challenge, we propose the first inference-time soup strategy, named Sample Reward Soups (SRSoup), for Pareto-optimal sampling across the entire space of preferences. Specifically, at each denoising step, we independently steer multiple denoising distributions using reward-guided search gradients (one for each reward function) and then linearly interpolate their search gradients. This design is effective because sample rewards can be shared when two denoising distributions are close, particularly during the early stages of the denoising process. As a result, SRSoup significantly reduces the number of queries required in the early stages without sacrificing performance. Extensive experiments demonstrate the effectiveness of SRSoup in aligning T2I models with diverse reward functions, establishing a practical and scalable solution. The code is available at https://github.com/EvaFlower/Sample-Reward-Soups-ICLR26.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
- This paper introduces SRSoup, a training-free and query-efficient diffusion-time alignment method for multiple rewards.
- The core idea is to avoid combinatorial blow-up by decomposing multi-objective optimization into $M$ simpler single-objective problems.
- The theory clarifies when the approximation is accurate: it holds when the objective’s curvature (second derivative) is small and the initialization is suitably chosen.
- In text-to-image experiments, SRSoup attains higher rewards under comparable query budgets.
- The empirical study also examines the roles of key hyperparameters.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Strengths
- The proposed multi-reward alignment method is training-free and query-efficient.
- Its query efficiency and accuracy are supported by both theory and experiments.
- The authors prove they can approximate the search gradient efficiently and accurately under stated assumptions, and they evaluate the finite-sample approximation error.
- The Pareto front is clearly visualized in experiments, and empirical Pareto near-optimality is demonstrated in Figures 3, 5, and 7.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Weaknesses
- As noted in lines 327–335, the main concern is that the assumption for Proposition 3 may not hold, leading to inaccurate approximations. The authors suggest switching from SRSoup to the standard method at the K-th step; however, the computational cost will then approach that of the standard “WeightedSum” method.
- For these reasons, I hope the computational cost as a function of K is clarified concretely in the main paper (I only found Table 4 in the appendix).
- The authors empirically validated density overlap using the Bhattacharyya coefficient (BC), whereas their theory uses Total Variation (TV). In the appendix and Proposition 4, they provide only an upper bound on TV. I think a lower bound on TV is also needed to justify the choice of K. The true TV might be small if BC is small.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Questions
- Could you extend your method to a per-subproblem optimization strategy with $M_2$ rewards per subproblem, where $M_2 < M$? I believe this could offer a favorable trade-off between quality and cost.
- For example, with 10 reward functions in total, you could approximately divide them into 5 subproblems, each containing 2 reward functions.
- Please note that I am NOT requiring any additional experiments for this.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Soundness
4

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
4

### Summary
This paper proposes Sample Reward Soups (SRSoup), a training-free, query-efficient multi-reward alignment method for text-to-image diffusion models.
The core idea is to pre-compute per-reward gradients (“search gradients”) and to linearly interpolate them to approximate the gradients for arbitrary reward weightings.
By sharing reward evaluations across weights during the early diffusion steps—when the sample distributions are still overlapping—the method aims to achieve efficient Pareto-optimal sampling under multiple rewards.
Experiments on Stable Diffusion 1.5 and SDXL demonstrate similar or slightly better image quality compared to fine-tuning methods (e.g., DDPO, TDPO) while reducing reward queries by ~1.8–2.7×.

### Strengths
1. The paper tackles the important problem of reducing query computation cost in multi-objective preference optimization.

2. The number of queries can be explicitly calculated (as shown in Section C.3), and each parameter can be set according to the desired reduction target.

3. The proposed method is clearly formulated and easy to understand; unifying the Gaussian initialization to achieve a first-order gradient approximation is an interesting design.

### Weaknesses
1. The paper does not discuss the impact of varying K on performance.

2. There is no comparison of processing time with existing methods other than weighted-sum.

### Questions
1. How does the performance change when K is varied?

2. Would it be possible to compare the inference time with existing methods such as Rewarded Soups, AlignProp, TDPO, and DDPO?

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
2

### Summary
This paper proposes a training-free multi-reward optimization framework that improves generated images of text-to-image diffusion models. In particular, the authors proposed a multi-reward alignment approach that optimize the denoising distribution at each sampling step using black-box optimization strategy. Additionally, they also proposed Sample Reward Soups, a mechanism that combines multiple reward objective that interpolates reward-guided search gradients. Experiment results show that the proposed inference-time framework achieves competitive performance compared to RL and other finetuning methods. They also demonstrated that the proposed Sample Reward Soups is more preferable than naive weighted sum of multiple rewards.

### Strengths
1. This paper is sufficiently novel in that it extends the idea of Reward Soap that interpolates multiple finetuned diffusion models to a training-free setup, making it more accessible while provided meaningful insight about the nature of this type of interpolation methods.
2. The author provided a detailed theoretical analysis of the characteristics of the proposed method, supported by toy experiments on mixture of Gaussian, which are informative.
3. The experiments are thorough. I especially appreciate the additional experiments and analysis in the supplementary material.

### Weaknesses
1. The author only showed experiments on SD-series U-Net, which is quite outdated at this time. I understand that many of the RL literature still focuses on SDv1.5 and SDXL due to computation constraints. However, as the proposed method is training free, it would be good to show results on state-of-the art rectified flow models based on diffusion transformers (DiT), such as Sana, SD3, Flux, etc. Such inclusion would make the paper more relevant.   It would also be interesting to see how the proposed method compared with the latest RL literature, such as Flow-GRPO. These latest works should also be discussed in the related works.

2. offline policy learning methods such as DPO are discussed but not included in the comparison. these model learns directly from human preference dataset which encompasses a diverse set of preference. It would be interesting to see however this method compare. While it is hard to apply soup to these models as they cannot be directly trained for different rewards, the author can still use other inference-time methods such as best-of-N sampling by matching the inference time compute.

3. for SDXL comparison in appendix, an RL baseline is missing, such as D3PO[2].


[1] Liu, Jie, et al. "Flow-grpo: Training flow matching models via online rl." arXiv preprint arXiv:2505.05470 (2025).
[2] Yang, Kai, et al. "Using human feedback to fine-tune diffusion models without any reward model." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
I currently recommend this paper for weak acceptance. 
I'm willing to increase my score based on author's responses. In particular, I hope the author can provide missing baselines and related literature.

### Soundness
3

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
The paper Sample Reward Soups (SRSoup) addresses the challenge of multi-objective optimization in Text-to-Image diffusion models by proposing the first query-efficient, inference-time "soup" strategy. The core problem is that standard methods for balancing multiple black-box rewards (like aesthetic quality and compressibility) require an exponentially large number of expensive reward queries across the preference space. SRSoup solves this by independently calculating a reward-guided search gradient for each individual objective at every denoising step, and then linearly interpolating these gradients (not the model weights or rewards themselves). This allows the model to efficiently share and reuse sample rewards across different weighted objective combinations, drastically reducing the number of required queries and enabling superior or comparable performance to the Weighted Sum baseline.

### Strengths
1. Novelty of Inference-Time Gradient Interpolation: The paper introduces the first inference-time “soup” strategy (SRSoup) for diffusion process, effectively adapting the “Model Soups” concept not to model weights but to search gradients. This approach achieves Pareto-optimal sampling across a preference space without the risks of reward over-optimization often seen in fine-tuning methods.
2. Addresses a Scalability Bottleneck with High Efficiency: The paper addresses a limitation in aligning with multiple reward functions, which requires exponentially growing number of reward queries, which makes the process highly inefficient. The proposed SRSoup resolves this by significantly reducing the number of queries in the early denoising stages, establishing a more scalable solution for multi-objective T2I alignment.
3. Strong Empirical Results and Practical Impact: The method achieves significant query efficiency (e.g., up to 2.7x speedup) over the weighted-sum baseline, as demonstrated by the Hypervolume (HV) versus reward query plots.

### Weaknesses
1. Lack of Cost-Benefit Justification: While the paper shows a reduction in the number of reward queries, it fails to provide a comprehensive cost-benefit analysis for the entire inference pipeline. The reward query computation time, which is model-dependent, must be weighed against the overhead of running multiple parallel single-reward guidance steps required by SRSoup's gradient estimation. 
2. Insufficient Analysis of Multi-Reward Trade-offs and Stability: The analysis, primarily relying on Hypervolume, does not rigorously demonstrate that SRSoup maintains a better balance or stability across the Pareto front compared to weighted-sum methods, especially with non-convex reward functions. This omission limits validation of the method's robustness against potential reward over-optimization.
3. Sensitivity and Justification of the Hybrid Strategy: The reliance on a hybrid strategy using SRSoup only for the first K steps lacks rigorous investigation into the determination and sensitivity of the boundary K.

### Questions
1. Please provide a detailed breakdown of the wall-clock time and GPU memory consumption across the full inference pipeline, explicitly weighing the reduced reward query time against the overhead of running M parallel guidance steps.
2. Given the risk of sub-optimal configurations in multi-objective optimization, how does SRSoup compare to simple weighted-sum methods in terms of output balance (i.e., avoiding cases where one reward is disproportionately maximized)?
3. Please provide a quantitative ablation study showing the performance (Hypervolume) and efficiency (Query Reduction) trade-offs across a range of K values (e.g., K=20, 40, 60).

### Soundness
2

### Presentation
3

### Contribution
3
