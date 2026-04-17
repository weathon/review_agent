# Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Temperature is a crucial hyperparameter in large language models (LLMs), controlling the trade-off between exploration and exploitation during text generation. High temperatures encourage diverse but noisy outputs, while low temperatures produce focused outputs but may cause premature convergence. Yet static or heuristic temperature schedules fail to adapt to the dynamic demands of reinforcement learning (RL) throughout training, often limiting policy improvement. We propose Temperature Adaptive Meta Policy Optimization (TAMPO), a new framework that recasts temperature control as a learnable meta-policy. TAMPO operates through a hierarchical two-loop process. In the inner loop, the LLM policy is updated (e.g., using GRPO) with trajectories sampled at the temperature selected by the meta-policy. 
In the outer loop, meta-policy updates the distribution over candidate temperatures by rewarding those that maximize the likelihood of high-advantage trajectories. This trajectory-guided, reward-driven mechanism enables online adaptation without additional rollouts, directly aligning exploration with policy improvement. On five mathematical reasoning benchmarks, TAMPO outperforms baselines using fixed or heuristic temperatures, establishing temperature as an effective learnable meta-policy for adaptive exploration in LLM reinforcement learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an intuitive method TAMPO, which replacing the conventional static or heuristic temperature to a meta-policy and learning it online, thereby achieving trajectory-driven exploration–exploitation adaptation in LLM reinforcement learning. In detail, the framework consists two loop process: inner loop generates trajectories under temperatures determined by meta-policy and optimized (using GRPO) LLM, and outer loop optimize meta-policy based on temperature-specific advantage, which can be calculated by reused trajectories likelihood from inner loop under certain temperature. Experiments on mathematical reasoning tasks show robust improvements over several fixed or linearly scheduled baselines, along with ablations and analyses (e.g., temperature trajectories, T* distributions), demonstrating the potential and interpretability of the proposed method.

### Strengths
S1: Treating temperature as a meta-policy is a fresh and practical perspective in the context of LLM RL. It addresses a real limitation of current methods that use fixed or heuristically scheduled temperatures.
S2: TAMPO can be seamlessly combined with existing critic-free RL methods (such as GRPO and REINFORCE++) without adding additional sampling costs, and has good practicality and promotion potential.
S3: The paper is well-written and easy-to-follow.

### Weaknesses
W1: The current meta-policy employs a shared temperature distribution across the entire dataset, without explicit conditioning on the prompt, subtask, or training stage. This may lead to inefficiency in multi-task settings or under highly heterogeneous prompts.
W2: As the reinforce learning is known by its high variance, report the mean +- standard deviation along with significance tests under at least 3-5 independent runs is preferred to demonstrate the effectiveness of the methods.
W3: All experiments are conducted on a 1.5B parameter model and focus solely on mathematical reasoning tasks. It is unclear how well the method generalizes to larger models or other types of tasks (e.g., code generation, dialogue, summarization).
W4: Typo: Line 356 and 357, “using initial learning rate 1x10e-6” but “minimum learning rate of 0.1” is confusing and counterfactual.
W5: The range (0.6-1.5) and interval (0.1) of the candidate temperature set are manually preset without an adaptive adjustment mechanism. Moreover, the applicability of this fixed candidate set to different models or task scenarios is not verified, which may lead to missing better exploration intervals due to the limitations of the temperature search space.
W6: The trajectory likelihood relied on for meta-policy updates is strongly coupled with the dynamically iterated LLM policy in the inner loop. No mitigation measures are designed for the likelihood estimation bias caused by policy updates, which may result in the evaluation of temperature-specific advantages being affected by the current policy bias, reducing the stability of temperature adaptation.

### Questions
Please refer to Weakness.

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
This paper proposes Temperature Adaptive Meta Policy Optimization (TAMPO), a new framework that recasts temperature control in large language model (LLM) reinforcement learning (RL) as a learnable meta-policy rather than a fixed hyperparameter.
In typical LLM RL (e.g., GRPO or PPO-based RLHF), temperature controls the exploration–exploitation balance during text generation.
However, most existing methods use static or heuristic temperature schedules, which fail to adapt to the evolving training dynamics.
TAMPO formulates temperature control as a two-level learning process: 
In the inner loop, the LLM policy is optimized using critic-free RL (e.g., GRPO) at the temperature chosen by the meta-policy. 
In the outer loop, the meta-policy updates its temperature based on temperature-specific advantages by reusing the inner loop rollouts, reinforcing those that are more likely to generate trajectories with high advantage trajectories, without requiring additional rollouts.

Experiments on five distinct mathematics-focused benchmarks (AIME24, MATH-500, AMC23, Minerva, and OlympiadBench) show that TAMPO outperforms fixed or heuristic temperature baselines, in terms of Pass@1 and Pass@8 accuracy, by approximately 1.9% and 1.7%, respectively.

### Strengths
- This work provides a interesting way to control exploration in LLM RL by recasting temperature as a meta-policy variable.
- TAMPO efficiently reuses existing rollouts for meta-policy updates, introducing negligible additional cost.
- TAMPO demonstrates consistent improvements across multiple reasoning benchmarks using DeepSeek-R1-Distill-Qwen-1.5B.

### Weaknesses
- Experiments are restricted to mathematical reasoning tasks $\to$ generalization to other domains (dialogue, code generation, ...) remains untested.
- It’s unclear how TAMPO can be applied to critic-based or hybrid RLHF methods.
- The approach uses a fixed discrete set of temperatures {0.6, 0.7, ..., 1.4, 1.5}. Continuous temperature optimization might yield smoother adaptation.
- The paper does not include an ablation study for different base models.
- While results are strong, interpretability and deeper insight into how adaptive temperature affects reasoning quality are somewhat limited.

### Questions
- It would be helpful to know whether the authors have conducted any ablation studies to assess the generality of TAMPO: 
  1. Using base models other than DeepSeek-R1-Distill-Qwen-1.5B, 
  2. Across different domains such as dialogue or code generation, and
  3. With alternative critic-free reinforcement learning algorithms.
- Could the authors clarify what the optimal solution of the iterative optimization represents, both from a theoretical and an intuitive perspective?
- Could the authors elaborate on whether they plan to extend TAMPO to continuous temperature optimization rather than relying on a fixed discrete set of candidates?
- Why does adaptive temperature lead to improved reasoning quality? In addition, are there any potential negative effects of adaptive temperature, such as reduced exploration or excessive exploration?

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
4

### Summary
This paper proposes Temperature Adaptive Meta Policy Optimization (TAMPO) that can dynamically adjust the temperature of large language models (LLMs) for reinforcement learning (RL)-based post-training of LLMs. More specifically, TAMPO consists of a hierarchical two-loop process: inner loop and outer loop. The inner loop optimizes the LLM policy, and the outer loop learns the temperature meta-policy. This paper evaluates TAMPO on five math reasoning benchmarks including AIME24, MATH-500, AMC23, Minerva, and OlympiaBench. The experiment results show that TAMPO can achieve higher scores than a base model (i.e., DeepSeek-R1-Distill-Qwen-1.5B) and GRPO-based post-trained models.

### Strengths
- S1. [Presentation] First of all, this paper is well written and organized.

- S2. [Novelty] The basic idea of learning LLM temperature for RL-based LLM post-training (i.e., meta-policy learning) seems novel.

### Weaknesses
- W1. [Performance] For meta-policy learning, TAMPO additionally calculates likelihoods at virtual temperatures when training the policy model. This increases the computational complexity of the GRPO-based LLM post-training. However, compared to the basic GRPO-based post-training (pass@1: 42.6%), TAMPO provides a slightly higher average accuracy (pass@1: 44.5%). 

- W2. [Hyper-parameters] Even though this paper proposes to learn the temperature meta-policy, this may introduce additional hyper-parameters to search (e.g., EMA coefficient and sampling strategy).

### Questions
- Q1. [Overhead] Compared to the baseline method (e.g., GRPO-based post-training), how much computational complexity does the TAMPO have?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Temperature Adaptive Meta Policy Optimization (TAMPO), a framework to dynamically adjust the sampling temperature of a LLM during reinforcement learning. Instead of treating the temperature as a fixed hyperparameter, TAMPO treats it as a learnable meta-policy that is optimized alongside the main policy. The approach operates in a hierarchical two-loop process: an inner loop updates the LLM’s policy (using a critic-free RL algorithm like GRPO) with trajectories sampled at a temperature chosen by the meta-policy, while an outer loop updates the meta-policy by rewarding temperatures that yield high-advantage trajectories. The authors evaluate TAMPO on five mathematical reasoning benchmarks (e.g., AIME, MATH, AMC, Minerva, Olympiad datasets), comparing against fixed and heuristically scheduled temperatures

### Strengths
1. The technical approach is sound and well-motivated. The paper identifies a clear limitation of existing LLM RL methods – the use of fixed or manually tuned exploration temperature – and offers a principled solution.
2. Significant practical strength of TAMPO is its decoupled outer-loop update mechanism, which enables online adaptation without additional rollouts。

### Weaknesses
1. Failure to Contextualize within the Meta-Gradient Literature: This omission is compounded by the paper's flawed positioning within the meta-learning literature it does cite.The paper does cite "meta-gradient methods" (Xu et al., 2018) for learning other hyperparameters like $\gamma$ and $\lambda$. It fails to cite the direct combination of these two concepts: "Meta-SAC: Auto-tune the entropy temperature of soft actor-critic via metagradient" (Wang & Ni, 2020), which applies a meta-gradient approach to this exact problem.
2. The paper asserts: "The trajectory likelihood is unimodal w.r.t. $T$". This unimodality is what justifies searching for a single likelihood-optimal temperature $T^{\star}$ (Equation 7). No direct evidence found in the manuscript. Figure 2 provides only 8 illustrative examples, which is not a proof.If this claim is false (e.g., if the likelihood function $\ell_T(\tau_i)$ is multi-modal), the entire basis for the update mechanism is flawed. The method would reinforce a local likelihood-optimal temperature, and the neighboring contributions (§3.3.2) would be meaningless. This is a critical gap in the paper's technical soundness.

### Questions
1. Which prior approaches to adaptive exploration or hyperparameter tuning in RL are most closely related, and how does TAMPO differ from them? The submission would benefit from discussing additional related works that were not cited. For example, methods that dynamically adjust exploration parameters (entropy, $\epsilon$-greedy schedules, or temperature annealing strategies in RL) and meta-learning approaches for RL hyperparameters deserve mention
2. What is the effect of TAMPO’s design decisions such as using a discrete candidate set of temperatures and the chosen range [0.6, 1.5]? The method currently relies on a fixed set $\mathcal{T}$ of 10 temperatures and updates a categorical distribution over them. This discretization might limit the precision of the optimal temperature found.
3. Can the authors provide more intuition or evidence about the underlying assumption that trajectory likelihood vs. temperature is unimodal?
4. The paper's outer-loop update contains two confusing choices.a) Why use sparsemax to normalize the likelihoods? Sparsemax is designed to create sparsity, which seems to contradict the stated goal of giving attenuated positive contributions to neighboring temperatures. Would a standard softmax not be more appropriate?b) Why use min-max normalization (Equation 12) for the final policy update? This is a simple heuristic. 
5. Given the citation to meta-gradient methods, why not use a proper policy gradient update on the meta-policy $\pi(T)$, using the aggregated advantage as the advantage signal for the action $T_k$?

### Soundness
3

### Presentation
2

### Contribution
2
