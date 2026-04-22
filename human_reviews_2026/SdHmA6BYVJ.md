# On Predictability of Reinforcement Learning Dynamics for Large Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Recent advances in reasoning capabilities of large language models (LLMs) are largely driven by reinforcement learning (RL), yet the underlying parameter dynamics during RL training remain poorly understood. This work identifies two fundamental properties of RL-induced parameter updates in LLMs: (1) Rank-1 Dominance, where the top singular subspace of the parameter update matrix nearly fully determines reasoning improvements, recovering over 99\% of performance gains; and (2) Rank-1 Linear Dynamics, where this dominant subspace evolves linearly throughout training, enabling accurate prediction from early checkpoints. Extensive experiments across 13 LLMs and 10 algorithms validate the generalizability of these properties. More importantly, based on these findings, we propose AlphaRL, a plug-in acceleration framework that extrapolates the final parameter update using a short early training window, achieving up to 2.5× speedup while retaining > 96\% of reasoning performance without extra modules or hyperparameter tuning.  This positions our finding as a versatile and practical tool for large-scale RL, opening a path toward principled, interpretable, and efficient training paradigm for LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper analyzes the parameter dynamics of large language models during reinforcement learning (RL) training. The authors find that RL-induced parameter updates are rank-1 dominant, meaning that most reasoning improvements come from changes along a single dominant direction in parameter space. Moreover, they show that this direction evolves linearly across training checkpoints. Using these observations, the paper proposes AlphaRL, a method that predicts the final update trajectory using only 10-40 % of the training process and extrapolates the remaining steps.

### Strengths
- The paper evaluates several RL algorithms (e.g., PPO, GRPO, RLOO, DAPO), providing a comprehensive view of RL dynamics across diverse optimization paradigms.
- The Rank-1 phenomenon is demonstrated on models of different sizes (8B-32B), suggesting that the observed behavior is not limited to a specific scale.
- Results are reported using avg@32 across multiple reasoning benchmarks, ensuring statistical reliability and mitigating sampling variance.
-  The paper clearly motivates the need to understand RL dynamics in LLMs and shows the practical relevance with its AlphaRL method

### Weaknesses
- While the paper analyses important RL algorithms, there exist newer and more efficient policy-gradient methods like CISPO[1]. Including such methods would further strengthen the analysis. However, the current coverage is already broad and sufficient for the paper’s main claims.
- Generalisation of Rank-1 and Rank-1 Linear Dynamics
   - (a) in Table 1, the reported accuracies for the fully trained Qwen3-8B model appear notably low. For instance, the paper reports 28.5 % on AIME’24, whereas Qwen3-4B-Instruct achieves roughly 73%[2]. While these numbers are not directly comparable, as Qwen3-4B-Instruct benefits from distillation (which typically yields stronger models) the size of the gap is large. Replicating the entire industrial post-training pipeline of a model like Qwen3 is understandably beyond the paper’s scope, however, this discrepancy raises a concern: are the observed Rank-1 updates and Rank-1 Linear Dynamics an inherent property of RL training or an artifact of under-optimized or unstable training regimes. The same question holds for larger scale models: In Figure 1, DAPO* (a 32B model) only achieves 50% on AIME24 which is also far below Qwen-4B-Instruct.
   - (b) The analysis of the Rank-1 Linear Dynamics in Table 1 is limited to only 8B models, which are known to be less efficient to train under RL[3]. It remains unclear whether the reported phenomena would persist for larger models (e.g., 32B+), where RL tends to be more effective. Adding a table with AlphaRL on a 32B model would strengthen the result.
   - (c) Experiments are only on Qwen models on a single dataset. 
- The conclusions of this work seem to contradict some prior findings [4], which report that RL updates are sparse but full-rank. The paper would benefit from a discussion reconciling these differences.

1: MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention (https://arxiv.org/abs/2506.13585)  
2: https://qwen.ai/blog?id=1e3fa5c2d4662af2855586055ad037ed9e555125&from=research.research-list  
3: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (https://arxiv.org/abs/2501.12948)  
4: Reinforcement Learning Finetunes Small Subnetworks in Large Language Models (https://arxiv.org/abs/2505.11711)

### Questions
- **Q1:** The paper starts RL training from the Qwen3-Base model, whereas most recent RL works typically apply RL on top of an instruction-tuned model (i.e., following the standard SFT -> RL pipeline).
   - (a) Can the authors elaborate why they started from the base model?
   - (b) Are there experiments or results starting from an instruction-tuned checkpoint? If not, do the authors expect the Rank-1 properties to still hold under the standard SFT -> RL setup?
- **Q2:** For the evaluation on mathematical reasoning benchmarks, was an automated expression evaluation system (e.g., math-verify) or any symbolic equivalence checker used?
- **Q3:** Given the accuracy of Qwen3-8B after training is low compared to vanilla Qwen3-4B-Instruct, could the observed Rank-1 phenomenon & Rank-1 Linear Dynamics be specific to under-optimized or unstable training regimes rather than an inherent property of RL training?
- **Q4:** The experiments are limited to Qwen models trained on the same dataset. How dependent do the authors believe their findings are on the specific base model architecture or training data?
- **Q5:** Could the authors elaborate why they think their results differ from the conclusions in _"Reinforcement Learning Finetunes Small Subnetworks in Large Language Models"_?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work identifies two properties of RL-induced parameter updates in LLMs: (1) Rank1 Dominance, where the top singular subspace of the parameter update matrix nearly fully determines reasoning improvements, recovering over 99% of performance gains; and (2) Rank-1 Linear Dynamics, where this dominant subspace evolves linearly throughout training, enabling accurate prediction from early checkpoints. Extensive experiments across 8 LLMs and 7 algorithms validate the generalizability of these properties.

### Strengths
1. This paper is clearly written and easy to follow.
2. The findings are interesting and novel for me.

### Weaknesses
1. Lack of theoretical justification on the observations. The conclusions are primarily based on large-scale empirical observations, which uncover universal low-rank dynamics in RL training. However, these findings lack rigorous theoretical foundations. In particular, why does RL possess these characteristics while SFT does not?
2. Experiments are mainly on reasoning tasks. Investigating whether the findings hold for reasoning tasks only or not may help figure out the reason why the findings hold.

### Questions
1. How does RL methods like GRPO/PPO with learned reward models on alignment tasks perform? Do the observed phenomenon still occur? And what about DPO?

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
The paper studies parameter-update dynamics during the RL procedure of LLMs. The authors identify two general properties when inspecting the parameter difference between the post-trained model and the base model: 1) the top singular subspace of the parameter difference almost fully characterizes the "essence" of reasoning gains; 2) the dominant subspace evolves approximately linear during RL, which enables simple prediction rules given early checkpoints. Leveraging these properties, the authors also propose AlphaRL, which extrapolates the rank-1 trajectory and brings significant training speedup. Experiments across models from 7B to 32B and various RL algorithms validate the effectiveness of the method.

### Strengths
1. The identified phenomenon is indeed new (at least to me) and striking and further leaves room for further exploration.
2. The work includes results spanning multiple base models and RL algorithms, with controls showing the phenomena are specific to RL rather than SFT/distillation. It also demonstrates near-linear evolution of rank-1 subspaces links training dynamics to performance, offering a compact lens to monitor or forecast progress.
3. The work goes beyond observation to deliver a practical speedup recipe (up to 2.5x, >96% retention), potentially saving significant compute during RL runs.

### Weaknesses
1. The paper itself notes a lack of rigorous theory explaining why RL updates should be rank-1-dominant and linearly evolving. The true underlying mechanisms remain speculative.
2. Most of the evaluated tasks are math-based. It is unclear if the claimed properties generalized across different domains, such as coding, long-context planning, and tool use. The same question holds for completely different model families and datasets used during RL. 
3. If I understand correctly, AlphaRL assumes (and hence requires) stable linear trends in early checkpoints and relies on a number of early checkpoints to work. In practice, effectiveness is limited by the design and stability of the RL algorithm, as the paper acknowledges. It remains uncertain whether the proposed approach encounters failure modes.
4. Computing per-module SVDs of the parameter difference across all layers for large models can be memory-/I/O-heavy; the paper doesn't quantify overheads or detail engineering tricks needed for 10B–30B+ models (or even larger ones). I assume this should be doable for small models but am not sure for larger ones.

### Questions
1. Is SVD performed per parameter block or per layer (or in a different way)?
2. How does AlphaRL behave under changing reward models, curricula, or exploration regimes that violate linearity (e.g., entropy collapse or under non-verifiable reward)?
3. Is there a criterion to decide when rank-1 is insufficient and a small rank-k should be used?
4. The paper reports that SFT/distillation lack both properties. Can we attribute this to a specific ingredient in RL (on-policy sampling, verifiable rewards, credit assignment) that leads to the low-rank/linearity behavior?

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
This paper conducts an empirical analysis on the updated parameters induced by RL training and shows that they can be almost fully determined by the Rank-1 subspace of the update matrix. This is called Rank-1 dominance. Further, the paper shows that this dominant Rank-1 subspace evolves linearly throughout training. These two properties enable accurate prediction of future RL training accuracy improvements (called AlphaRL) simply from early training results, by using a simple ordinary least squares fit, and extrapolating performance improvements upto maximal accuracy. Results across different model sizes provide strong empirical evidence for the AlphaRL method.

### Strengths
- The paper is extremely well written and presented. The takeaways are presented clearly and the main results come off clearly from the fig 1 of the paper. 
- The empirical analysis is clearly grounded via a simple theoretical investigation, and this brings a strong justification for the AlphaRL method.
- The empirical results in tab 1 demonstrate the strength of the AlphaRL method.

### Weaknesses
- While the overall results are strong, a question that arises is: does the strength of the AlphaRL method depend on the compute budget used for RL? Can the prediction be done for arbitrary compute budgets of RL training, or is this restricted to the current setting used in the paper?
- Isn’t this method inherently upper-bounded by the noise celiling of the task? So shouldn’t the parameter update follow a sigmoidal rule rather than the linear rule outlined here? Currently, the paper uses simple linear regression and this might be a limiting choice?
- The current DIST method tested in the paper is an off-policy distillation method where the trajectories are generated by the teacher model. It would be interested to understand if an on-policy distillation method would also yield the low-rank yet effective update mechanisms. Particularly illuminating would be how the on-policy distillation method would compare with other methods in the fig 5 setting.
- All experiments are only conducted with Qwen models. Prior works (https://arxiv.org/abs/2506.10947, https://arxiv.org/abs/2507.10532) have shown biases and data contamination in Qwen models with regards to the benchmarks used for testing. Would the results also hold for non-Qwen models?

### Questions
All my questions are in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
