# TreeGRPO: Tree-Advantage GRPO for Online RL Post-Training of Diffusion Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Reinforcement learning (RL) post-training is crucial for aligning generative models with human preferences, but its prohibitive computational cost remains a major barrier to widespread adoption. We introduce **TreeGRPO**, a novel RL framework that dramatically improves training efficiency by recasting the denoising process as a search tree. From shared initial noise samples, TreeGRPO strategically branches to generate multiple candidate trajectories while efficiently reusing their common prefixes. This tree-structured approach delivers three key advantages: (1) *High sample efficiency*, achieving better performance under same training samples (2) *Fine-grained credit assignment* via reward backpropagation that computes step-specific advantages, overcoming the uniform credit assignment limitation of trajectory-based methods, and (3) *Amortized computation* where multi-child branching enables multiple policy updates per forward pass. Extensive experiments on both diffusion and flow-based models demonstrate that TreeGRPO achieves **2.4$\times$** faster training} while establishing a superior Pareto frontier in the efficiency-reward trade-off space. Our method consistently outperforms GRPO baselines across multiple benchmarks and reward models, providing a scalable and effective pathway for RL-based visual generative model alignment. The project website is available at https://treegrpo.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents TreeGRPO, an online RL algorithm for finetuning diffusion and flow matching models. Unlike existing RL methods that treat the multi step denoising process as independent full trajectory sampling, resulting in low sample efficiency and coarse credit assignment, TreeGRPO reformulates the process as a multi step decision problem optimized through tree structured RL. By modeling multi-step denoising as a search tree, it reuses shared prefixes to enhance sample efficiency and propagates node-level advantages for fine grained credit assignment. Experiments across multiple reward models show that TreeGRPO achieves markedly higher training efficiency and establishes a new Pareto frontier for RL based visual generative model alignment.

### Strengths
1. This paper is the first to apply tree search–based GRPO optimization to image diffusion and flow matching models, achieving substantial improvements in training efficiency.

2. The paper provides a comprehensive review of existing work and clearly articulates the similarities and differences between TreeGRPO and prior approaches.

3. The proposed method is concise and well designed, with comprehensive ablation studies conducted to thoroughly evaluate the effects of the introduced hyperparameters.

### Weaknesses
1. While TreeGRPO demonstrates impressive efficiency and gains on HPS-v2.1 and Aesthetic rewards, its performance on ImageReward and CLIPScore is weaker than DanceGRPO. This variability suggests that the proposed method may be sensitive to the type of reward signal and might not generalize equally well under different evaluation criteria.

2. Because TreeGRPO adopts a variable tree structure, it introduces additional hyperparameters. As shown in Tables 3 and 4, the optimal hyperparameters differ across reward objectives. This means that although each fine-tuning run becomes more efficient, achieving the best performance may still require multiple fine-tuning rounds for hyperparameter search, resulting in the overall computational cost not being reduced.

3. The paper contains several formatting and typographical issues, such as "rewardds" in the caption of Figure 1, "Where" that should be "where" on line 254, and an extra period on line 307. In addition, the manuscript does not include the required "Use of Large Language Models" statement. I recommend that the authors carefully review and correct these writing details.

### Questions
1. Could the authors elaborate on why TreeGRPO performs less favorably on certain reward metrics such as ImageReward and CLIPScore? Does this limitation stem from the local advantage propagation design or other factors like the tree structure or sampling schedule?

2. There is a concurrent work named BranchGRPO [1] that appears closely related to TreeGRPO in terms of motivation and overall design. Could the authors briefly compare TreeGRPO with BranchGRPO?

[1] Li, Yuming, et al. "Branchgrpo: Stable and efficient grpo with structured branching in diffusion models." arXiv preprint arXiv:2509.06040 (2025).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces an approach to fine-tuning diffusion models through a search tree.

### Strengths
The paper tackles the problem of making RL-based fine-tuning of vision-based generative models more efficient, which is a well-motivated and common problem in the existing literature. Additionally, TreeGRPO presents a significant improvement in runtime against the baselines considered, while matching or improving performance.

### Weaknesses
My main concern is the following, with some additional questions/concerns listed below. In Section 3.3., the paper claims that “Deterministic ODE solvers lack the transition probabilities required by policy-gradient RL...” and “we convert the probability-flow ODE... to an equivalent SDE that admits tractable likelihoods...”. Likelihoods necessary for RL are not tractable in SDE’s (e.g. due to the Brownian motion)? How is the paper computing log likelihoods for the advantage-based update in Equation 13? 

Additionally, the paper claims an improvement in sample efficiency, but the results are in terms of runtime. In Figure 1, how does the training on 8 GPUs correspond to the runtime on a single GPU? Are these different training setups? There is limited explanation of the implementation details provided. No appendix was provided in the PDF.

### Questions
See Weakness section.

### Soundness
2

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
3

### Summary
The paper proposes TreeGRPO, a tree-structured RL framework that reformulates diffusion-model denoising as a search tree, where shared prefixes and selective branching improve sample efficiency and step-wise credit assignment. The algorithm constructs trees by alternating deterministic ODE steps and stochastic SDE windows, then backpropagates rewards through branches to compute per-edge advantages for PPO style updates. Experiments on SD-3.5-Medium show TreeGRPO achieves comparable or higher alignment rewards than GRPO based baselines while cutting training time.

### Strengths
1.	Recasting diffusion denoising as a search tree with shared prefixes is a creative idea that directly addresses sample efficiency and credit assignment issues. The use of log probability weighted backup for per-edge advantages is theoretically sound.
2.	In terms of efficiency gains, TreeGRPO reduces per‑iteration training time by ~$2\times$ - $3\times$ while matching or surpassing baseline alignment scores. The method shows especially strong improvements in aesthetic scores.
3.	The paper provides a thorough empirical evaluation by comparing TreeGRPO against strong baselines across multiple reward models and both single reward and multi reward settings. Ablation studies systematically vary tree width, depth and sampling strategies, illustrating tradeoffs.

### Weaknesses
1.	While the method amortizes computation, branching multiple trajectories simultaneously increases memory usage, especially for large diffusion models. The paper does not quantify the computational overhead relative to baselines or provide strategies for memory management beyond acknowledging the issue.
2.	The performance improvement is somewhat marginal. Although TreeGRPO outperforms baselines on HPS and aesthetics, DanceGRPO achieves the highest ImageReward score in the single reward setting. The difference in ClipScore between TreeGRPO and baselines is small. Additional analysis could clarify why some metrics benefit less. Also, the Aesthetic metric is much easier to improve compared to others.
3.	Branching creates many edges from shared prefixes. That amortizes compute but correlates samples. So the effective on‑policy batch may be narrower than it looks. Importance weighting corrects policy drift but not correlation. Tracking effective sample size or using prefix‑level decorrelation, for example, multiple independent seeds per batch, would clarify the true statistical efficiency.
4.	The framework introduces extra hyperparameters, branching factor, tree depth, SDE window ratio, and number of trees, which significantly affect performance. Adaptive schedules are mentioned as future work but not explored.

### Questions
1. The tree structure seems designed for training; does it also accelerate inference or sample diversity at deployment? If not, could the branching be leveraged for generating diverse images at inference time?

2. The method combines tree search with GRPO; how much of the observed gain comes from tree exploration versus the per-edge GRPO update? Would a tree based extension of standard PPO yield similar benefits?

### Soundness
2

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
2

### Summary
The authors propose a tree-structured technique for advantage estimation and apply it to training diffusion models via GRPO.

### Strengths
(+) I haven't seen the idea of tree-structured advantage estimation before.

(+) I appreciate the ablations performed.

### Weaknesses
(-) From what I can tell, the results are not significantly stronger than similar baselines performance-wise. That said, requiring less compute is definitely a plus.

### Questions
1. Can you add in a comparison to https://arxiv.org/abs/2404.16767 if you have the compute?

2. Beyond just being empirically promising, I think this paper would be much stronger if there were a more compelling conceptual / theoretical reason why tree-structured advantage estimation is the "right" answer. I spent some time thinking about this but I couldn't come up with a reason why. Could you provide some thoughts on this, perhaps working things out on simple toy problems (e.g., flows between two Gaussians)?

### Soundness
3

### Presentation
2

### Contribution
2
