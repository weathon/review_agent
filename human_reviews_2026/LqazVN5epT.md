# On Entropy Control in LLM-RL Algorithms

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 4

## Abstract
For RL algorithms, appropriate entropy control is crucial to their effectiveness. To control the policy entropy, a commonly used method is entropy regularization, which is adopted in various popular RL algorithms including PPO, SAC and A3C. Although entropy regularization proves effective in robotic and games RL conventionally, studies found that it gives weak to no gains in LLM-RL training. In this work, we study the issues of entropy bonus in LLM-RL setting. Specifically, we first argue that the conventional entropy regularization suffers from the LLM's extremely large response space and the sparsity of the optimal outputs. As a remedy, we propose AEnt, an entropy control method that utilizes a new clamped entropy bonus with an automatically adjusted coefficient. The clamped entropy is evaluated with the re-normalized policy defined on certain smaller token space, which encourages exploration within a more compact response set. 
In addition, the algorithm automatically adjusts entropy coefficient according to the clamped entropy value, effectively controlling the entropy-induced bias while leveraging the entropy's benefits. AEnt is tested in math-reasoning tasks under different base models and datasets, and it is observed that AEnt outperforms the baselines consistently across multiple benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a technique, AEnt, that makes the PPO's (GROP's) entropy regularization work with LLM-RL setting by clamping action-space, focusing on top-N tokens. Previously, the entropy regularization didn't work well in LLM-RL setup due to the extremely large action-space, compared to standard RL setup. In AEnt, this issue is mitigated by clamping the action-space and additionally introducing an algorithm that automatically tunes the entropy coefficient. In experiments, AEnt shows its strong performance improvements as well as the sane behavior of the policy entropy. Ablation studies show that the policy entropy cannot behave well without the proposed clamping mechanism and the automatic coefficienct tuning.

### Strengths
This paper has strengths as follows:
- The idea introduced in AEnt is simple and generic enough to be applied to general LLM-RL algorithms.
- In the main experiment, AEnt is shown to be improving performance from the baselines.
- In ablation studies, the changes introduced in AEnt is vital to gain performance improvements.

### Weaknesses
The paper is well written. Thus, there isn't obvious flaws in the manuscript. However, here is a list of comments to improve the quality of this paper.
- In the context of automatic entropy cofficient tuning, SAC has also proposed the similar mechanism that is implemented as a dual optimization problem. Authors should explain how the proposed tuning algorithm is selected over SAC's algorithm.
- In Figure 4, the response length plots aren't really described in the body texts. Since you plot them along with the entropy curves, authors should add some descriptions about correlation between the entropy and the response length at least.
- Authors should explain how many random seeds each does each experiment run with.

### Questions
The weakness includes my questions that I want to clarify.

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
4

### Summary
The author reveal why entropy regularization  no gains in LLM-RL training.
Through theoritic anysis, the author reveal that entropy will improve the convergence to optimal solution but bring bias.
The bias is closely related to the scale of action space and the sparsity of the optimal outputs.
Based on this anaysis, the proposed method calculated entropy only in a smaller(top 1-p) action space.
experiments in math-reasoning tasks shows that the proposed method outperforms the baselines.

### Strengths
I believe this is a strong paper with novel contributions and valuable insights for the LLM-RL community, particularly in the context of entropy tricks. I recommend accepting this paper.

1. The motivation for the work is clearly presented.
2.The theoretical analysis of entropy is well-executed and convincingly explains why the entropy trick cannot be applied directly in LLM-RL.
3.Simple Yet Efficient Method: The proposed method is both straightforward and effective.

### Weaknesses
I have a few questions regarding some details, and I believe additional clarity would be helpful for the broader application of this work.

1. **Choice of Top (1 - p) Percent Tokens:**  
   Why do you use the top (1 - p) percent of tokens in \( \pi \), rather than applying a threshold(e.g. \pi > \epsilon) to all tokens?

2. **Experimental Results in Figure 3:**  
   As shown in Figure 3, the experimental results indicate that the performance is similar during the first 90% of the training steps, with significant improvement occurring only in the last few epochs.  
   How does this happen? Could you provide a more detailed curve with additional steps to better illustrate its stability? An explanation would be greatly appreciated.

3. **Related Work in Entropy Analysis and Action Space Shaping/Pruning:**  
   Are there existing works that analyze entropy in large or continuous action spaces within traditional RL methods?  
   Additionally, is there prior work on action space shaping (continuous) or action space pruning (discrete) in RL methods (e.g., [1], [2], [3], [4])?  
   Could these methods be beneficial for applying the entropy trick in LLM-RL?

**References:**

[1] Zhong D, Yang Y, Zhao Q. No Prior Mask: Eliminate Redundant Action for Deep Reinforcement Learning[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(15): 17078-17086.

[2] Baram, N.; Tennenholtz, G.; and Mannor, S. 2021. Action redundancy in reinforcement learning. In Uncertainty in Artifcial Intelligence, 376–385. PMLR.

[3] Tennenholtz, G.; and Mannor, S. 2019. The Natural Language of Actions. ArXiv, abs/1902.01119.

[4] Tavakoli, A.; Pardo, F.; and Kormushev, P. 2018. Action branching architectures for deep reinforcement learning. In Proceedings of the aaai conference on artifcial intelligence, volume 32.

### Questions
See above

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the limited effectiveness of conventional entropy regularization in LLM-RL training, which contrasts sharply with its success in robotics and games. The authors provide theoretical analysis showing that entropy regularization suffers from bias that scales with response space size and optimal action sparsity. They propose AEnt, which employs two key mechanisms: (1) clamped entropy computed on a re-normalized policy over a reduced token space containing only the top probability tokens, and (2) an adaptive coefficient that automatically adjusts based on whether current entropy falls within target bounds. The method is evaluated on mathematical reasoning tasks using 1.5B parameter models (Qwen2.5-math-1.5b and DeepSeek-R1-distilled-Qwen-1.5b), demonstrating improvements over GRPO and standard entropy regularization baselines on 5 out of 6 benchmarks across two experimental settings.

### Strengths
- The paper provides clear theoretical motivation through Propositions 1 and 2, explaining why conventional entropy regularization underperforms in LLM settings due to large action spaces and sparse optimality, with rigorous proofs and synthetic validation in Figure 1.
- The proposed clamped entropy mechanism directly addresses the identified theoretical limitation by restricting entropy maximization to a compact, relevant token space, reducing bias from irrelevant tokens while maintaining exploration benefits.
- Experimental validation includes relevant ablation studies demonstrating the necessity of both adaptive coefficients (Figure 5) and appropriate clamping percentages (Figure 6), with consistent improvements across multiple mathematical reasoning benchmarks.

### Weaknesses
- Experimental validation is narrowly confined to mathematical reasoning with 1.5B models, providing insufficient evidence for a method claiming to solve a fundamental LLM-RL problem. Mathematical reasoning has unique properties (verifiable correctness, structured outputs) that may not generalize to open-ended generation, instruction following, or creative tasks where optimal response sets are less well-defined.
- The adaptive coefficient scheme essentially performs online hyperparameter tuning during training, introducing additional hyperparameters (H̃_low, H̃_high, β, λ_low, λ_high) that require careful task-specific tuning. This increases rather than decreases the complexity of deployment, contradicting the implicit promise of a more robust entropy control method.

### Questions
- Have you evaluated the method on larger models (7B, 13B, 70B parameters)? The experiments exclusively use 1.5B models, but larger models may exhibit fundamentally different entropy dynamics due to better pre-training, sharper probability distributions, or different exploration-exploitation trade-offs. Specifically, do larger models already concentrate probability mass appropriately, making clamped entropy redundant? Additionally, how does the computational overhead of computing clamped entropy scale with model size, and at what scale does this become prohibitive relative to the performance gains observed?
- Why specifically use top-probability tokens for defining the clamped space, and have you systematically compared against alternative criteria? The choice appears to assume pre-trained models already assign high probability to reasonable tokens, which may be circular reasoning if the goal is to improve exploration. Have you experimented with task-relevant vocabularies defined by domain experts or training data or learned token importance via a separate lightweight model? Furthermore, does the top-p criterion work equally well across different input contexts—for instance, does it fail when the model is uncertain and distributes probability broadly, or when pre-training biases are strong but incorrect for the RL task?
- How does performance vary across fundamentally different task domains beyond mathematical reasoning? Mathematical reasoning has unique properties—verifiable correctness, structured formats, relatively sparse optimal solution sets—that may make it particularly amenable to entropy control. Have you tested on tasks such as open-ended instruction following where response diversity is valued or multi-turn dialogues which require contextual adaptation? More critically, does your theoretical analysis suggesting sparse optimality as problematic even apply to domains where optimal response sets are large or poorly defined? Without broader evaluation, it remains unclear whether you have solved a mathematical reasoning-specific problem or a general LLM-RL challenge.

### Soundness
3

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
4

### Summary
This paper addresses the well-known problem of standard entropy regularization failing to provide benefits in LLM-RL training, unlike in traditional robotics or game environments. The authors provide a theoretical diagnosis, arguing this failure stems from the combination of the LLM's extremely large response space and the sparsity of optimal outputs, which introduces a significant entropy-induced bias. As a remedy, they propose AEnt, a straightforward entropy control method featuring two key components: 1) a "clamped entropy" bonus calculated only over a size-reduced, high-probability token subspace to mitigate this bias , and 2) an adaptive coefficient that dynamically adjusts to keep this clamped entropy within a target range, preventing training instability. The authors present empirical results on math-reasoning tasks to demonstrate that AEnt can stabilize training and outperform baselines like GRPO and traditional entropy regularization.

### Strengths
- The paper studies an important problem that why entropy regularization works for games or robotics but not for llm
- The motivation is strong; the reason why entropy regularization fails in LLMs—attributing it to the large and sparse action space—is intuitive and well-supported. This claim is convincingly backed by both the theoretical bound in Proposition 2 (which explicitly shows the bias term's dependency on action space size $|\mathcal{A}|$ and optimal response sparsity) and a compelling, clear toy experiment in the Figure.

### Weaknesses
- The paper's first method is 'clamped entropy' , which means maximizing entropy within the LLM's 'top-p' token range. In fact, this idea is quite straightforward; I have even tried it myself. However, this method is extremely sensitive to the hyperparameter, specifically the coefficient in front of the entropy loss. A good result can be obtained through extensive hyperparameter tuning, but it still requires re-tuning for a new model, and the performance is not very stable.
- The paper's second method, dynamically adjusting the entropy coefficient to maintain stable entropy, is also not a new concept. This adaptive mechanism is functionally similar to early KL control approaches (e.g., [https://arxiv.org/pdf/1909.08593.pdf]) designed to keep the KL divergence within a stable range, a method also integrated into the VeRL repository ([https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py#L150]). To my knowledge, practitioners today still prefer fixed KL coefficients precisely because these dynamic adjustment mechanisms are notoriously unstable. The method's core premise—that forcing entropy to stabilize within a fixed numerical range is beneficial—is unfounded. It is difficult to determine a priori what a "good" range for entropy or KL should be. Furthermore, it is questionable why entropy must be stable; a beneficial training trajectory might just as well involve entropy slowly rising and then slowly falling. The optimal shape of the entropy curve is likely case-by-case, and attempting to manually prescribe its behavior is a questionable engineering choice

- The paper's experimental results are not convincing.
    - In Figure 3, using the peak of the curve to represent the performance is flawed, as it could very likely be an artifact of training randomness.  
    - Table 1 only shows the final results for only one model,  but Figure 3 shows the training curves for DeepSeek-R1-distilled-Qwen-1.5b and    Qwen2.5-math-1.5b. Table 1 should be aligned with Figure 3.
    - I would expect the paper to demonstrate consistent improvements on at least three different models without requiring heavy hyperparameter tuning to truly prove the method's effectiveness.

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2
