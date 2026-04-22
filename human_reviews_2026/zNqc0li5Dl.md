# ReMix: Reinforcement Routing for Mixtures of LoRAs in LLM Finetuning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
Low-rank adapters (LoRAs) are a parameter-efficient finetuning technique that injects trainable low-rank matrices into pretrained models to adapt them to new tasks. Mixture-of-LoRAs models expand neural networks efficiently by routing each layer input to a small subset of specialized LoRAs of the layer. Existing Mixture-of-LoRAs routers assign a learned routing weight to each LoRA to enable end-to-end training of the router. Despite their empirical promise, we observe that the routing weights are typically extremely imbalanced across LoRAs in practice, where only one or two LoRAs often dominate the routing weights. This essentially limits the number of effective LoRAs and thus severely hinders the expressive power of existing Mixture-of-LoRAs models. In this work, we attribute this weakness to the nature of learnable routing weights and rethink the fundamental design of the router. To address this critical issue, we propose a new router designed that we call **Re**inforcement Routing for **Mix**ture-of-LoRAs (ReMix). Our key idea is using *non-learnable* routing weights to ensure all active LoRAs to be equally effective, with no LoRA dominating the routing weights. However, our routers cannot be trained directly via gradient descent due to our non-learnable routing weights. Hence, we further propose an unbiased gradient estimator for the router by employing the reinforce leave-one-out (RLOO) technique, where we regard the supervision loss as the reward and the router as the policy in reinforcement learning. Our gradient estimator also enables to scale up training compute to boost the predictive performance of our ReMix. Extensive experiments demonstrate that our proposed ReMix significantly outperform state-of-the-art parameter-efficient finetuning methods under a comparable number of activated parameters.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes ReMix, a new approach to reinforcement-learning-based expert routing in Mixture-of-Experts (MoE) models. Traditional MoE routing methods (like top-k softmax gating or Switch Transformers) use deterministic or load-balanced gates that often underexplore the routing space and result in suboptimal expert utilization.

ReMix instead formulates expert selection as a policy optimization problem: a lightweight RL agent (the “router”) learns a routing policy to maximize token-level reward signals, balancing accuracy and load distribution.

### Strengths
1. Novel framing of expert routing as reinforcement learning.
The core conceptual shift of viewing routing as an RL problem is both natural and underexplored. Prior works (e.g., BASE layers, Switch) rely on differentiable load-balancing heuristics, while ReMix introduces a reward-driven control perspective.

2. Elegant algorithmic formulation.
The paper presents a clear KL-regularized objective.
This balances exploration and stability, akin to PPO or AWR-style updates. The derivation (Appendix B) is sound and bridges policy optimization and MoE training dynamics

3. Solid empirical validation.
Experiments cover multiple benchmarks: WikiText-103, The Pile, and C4. ReMix improves both routing diversity (entropy and expert usage) and model quality (perplexity and accuracy) across all tested scales (up to 7B parameters). Ablations on reward shaping, KL terms, and temperature annealing convincingly show robustness.

4. Good theoretical and empirical complementarity.
The authors provide intuitive and theoretical support for why reinforcement signals better capture dynamic routing needs—particularly in settings with dynamic data distributions or evolving expert specialization.

5. Clear presentation.
The paper is well-written and visually coherent. Figures 2–4 (routing entropy evolution, per-expert load distributions) effectively illustrate the stability and exploration benefits.

### Weaknesses
1. Limited analysis of computational overhead.
Introducing a per-token RL router adds non-trivial computational and communication overhead, especially during distributed training. While the paper claims “<3% runtime overhead,” the methodology for measuring this is unclear. Profiling on larger clusters would be valuable.

2. Unclear generality beyond language models.
All evaluations are in NLP. It remains to be seen whether ReMix generalizes to multimodal MoE systems (e.g., PaLI, Flamingo) or vision MoE architectures (e.g., V-MoE).

3. Comparison to recent stochastic routing methods.
The paper compares against standard deterministic baselines but omits stochastic routing methods, which also aim to encourage exploration via stochastic routing. These comparisons would better situate ReMix’s advantages.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work investigates Mixture-of-LoRAs and shows—both empirically and theoretically—that a few LoRA modules often dominate the routing, constraining the model’s capacity and expressiveness. The authors trace this issue to learnable router weights and propose using fixed, equal weights so that all selected LoRAs contribute equally. Since fixed weights preclude gradient-based training, they recast the problem as a reinforcement-learning problem in which the router is the policy model and the SFT loss functions as the negative reward. Overall, the paper is well structured and easy to follow. But I found there are some key weaknesses should be addressed before considering acceptance.

### Strengths
1.	The paper is well organized and clearly structured, making it easy to follow.
2.	It offers both theoretical and empirical analyses of the imbalance problem.
3.	The figures and illustrations are clear and easy to interpret.

### Weaknesses
1. The eq.(3) concerns only about utilization of LoRAs for each given input. So, it is sample-specific measurement, would your observations being biased due to samples?
	2. In figure1, you tract only the routing weights of the last layer. Would the pattern be the same across layers? It is better to illustrate with a heat map where each layers is also included.
	3. In section 3.1, you introduce hyper-parameters k which is the number of LoRAs you activated and /oemga which is the routing weights. How sensitive is your method to these hyper-parameters? How to choose different k and /omega for different models and tasks;
	4. You are assigning the same routing weights to all chosen experts, then it is equal to use a single LoRA with rank=k*r. The only difference is that you may choose different subset of k LoRAs for different samples. But there is no results showcasing with your method, the mixture of LoRAs are more equally activated across different samples. There could still be the case that it always choose the same k LoRAs.
	5. To overcome the non-learnable drawbacks, you propose to use RL techniques. But there are also other possibilities. For example, you could use soft mixture during training and hard top-k at evaluation. Another solution could be using Gumbel-top-k with straight-through estimators. The weakness here is that: (1) There should be a discussion or literature review for these solutions to non-differentiable problems since it is one of your core contribution. (2) At least there should be a toy experiment to showcase RL is better than these solutions.
	6. Too many details are absent in your experiments. For example, what hyper-parameters are used. What is the learning rate, optimiser, M, k and /omega?
	7. While you performing ablation study, when you said by removing RLOO, what training scheme do you use then?
	8. Table 2 is not convincing.  i would like to see the val acc versus steps or val acc versus wallclock time. especially you are using RL, i would guess your method needs more steps to converge. therefore, i think the training time efficiency is overclaimed;

### Questions
Below is a summary of my questions. Please also refer to weakness.
	1.	Sample bias: Since Eq. (3) measures LoRA utilization per input sample, could the observations be biased by the specific samples used?
	2.	Layer-wise consistency: Figure 1 only shows routing weights for the last layer—do similar imbalance patterns appear across other layers? A layer-wise heatmap would clarify this.
	3.	Hyperparameter sensitivity: How sensitive is the method to the hyperparameters k (number of active LoRAs) and \omega (routing weights)? How should these be chosen for different models and tasks?
	4.	Equivalence to larger-rank LoRA: If all selected LoRAs share equal weights, isn’t this effectively the same as using a single LoRA with rank = k × r? Also, is there evidence that the selected LoRAs vary across samples rather than always picking the same subset?
	5.	Alternative solutions to non-differentiable routing: Why choose a reinforcement-learning approach over other differentiable approximations like soft mixtures or Gumbel–Top-k with straight-through estimators? The paper should discuss these alternatives and ideally include a toy comparison.
	6.	Missing experimental details: Key hyperparameters (learning rate, optimizer, M, k, and \omega) are not specified, making experiments hard to reproduce.
	7.	Ablation clarity: In the ablation study, when RLOO is removed, what training scheme replaces it?
	8.	Training efficiency claims: Table 2 is unconvincing—validation accuracy over steps or wall-clock time should be shown, especially since RL may require more steps to converge. The claimed efficiency improvement may therefore be overstated.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper identifies a key limitation of current Mixture-of-LoRAs models: routers trained with continuous, learnable weights become highly imbalanced, often directing almost all traffic to one or two LoRAs, which sharply reduces model expressivity. To address this, ReMix introduces a reinforcement-based routing mechanism with constant routing weights applied equally across all activated LoRAs. This guarantees balanced participation of experts and avoids dominance collapse. Since constant weights break differentiability, the router is trained using reinforcement learning, treating the supervised finetuning loss as a negative reward. The authors develop an unbiased gradient estimator with Reinforce Leave-One-Out (RLOO) variance reduction, allowing stable and scalable optimization. Empirically, ReMix consistently outperforms prior PEFT methods for multiple tasks.

### Strengths
1. The paper is well-written and easy to follow.
2. The theoretical and empirical analysis are insightful.

### Weaknesses
1. For each theorem statement, providing a high-level explanation of the proof structure and key intuitions would significantly improve readability.
2. The imbalance routing issue is well-known in the area of mixture-of-experts. Related work should also discuss mixture-of-experts, compare mixture-of-experts and mixture-of-LoRA, and discuss how people resolve the imbalance issue for mixture-of-experts.

### Questions
1. Is introducing ESS as a tool for diagnosing routing‑weight imbalance new? Also, does Theorem 1 pertain solely to mixture‑of‑LoRA, or can it be applied to mixture‑of‑experts?
2. Could the author repeat the experiments in Figure 4(b) for other baselines such as HydraLoRA and Mix LoRA? Comparison with other baseline would strengthen the author’s claim. 
3. Could the author add more balanced MoE routing baselines?
4. Could the author empirically verify Theorem 2?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a solution to ensure that more than one LoRA adapter remains active in mixtures-of-LoRA settings. The authors provide theoretical analysis and an accompanying framework that enforces at least $k$ active adapters during routing, along with a discussion on the procedural differences between fine-tuning and inference.

### Strengths
*  Provides a theoretical proof showing that routing weights are almost surely imbalanced under standard conditions.
*  Introduces a framework guaranteeing at least $k$ active adapters at any given time.
*  Discusses implementation and procedural differences between fine-tuning and inference phases.

### Weaknesses
*  Unclear motivation: Having imbalanced routing is not necessarily undesirable. In practice, sparsity can be beneficial since less active adapters could be offloaded to slower memory tiers, improving efficiency. The paper lacks a deep discussion of why routing imbalance is inherently problematic.
*  Although the proposed method enforces at least $k$ active adapters, it does not guarantee diversity across selections. The same subset of $k$ adapters might always be activated together, which effectively collapses back to a single combined adapter.
*  The paper does not seem to discuss how the value of $k$ should be chosen.
*  The reported accuracy gains may correlate with the increased number of parameters rather than improved routing balance. The paper does not seem to disentangle these two effects or provide ablation results controlling for parameter count.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1
