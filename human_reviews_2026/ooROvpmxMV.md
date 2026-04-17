# Agentic Reinforcement Learning with Implicit Step Rewards

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Large language models (LLMs) are increasingly developed as autonomous agents using reinforcement learning (agentic RL) that reason and act in interactive environments.
However, sparse and sometimes unverifiable rewards make it extremely challenging to assign credit when training LLM agents that serve as a policy.
Recent work attempts to integrate process supervision into RL but suffers from biased annotation, reward hacking, high-variance from overly fine-grained rewards or failures when state overlap is rare.
We therefore introduce implicit step rewards for agentic RL (**iStar**), a general credit-assignment strategy that integrates seamlessly with standard RL algorithms without relying on additional rollouts or explicit step labels.
Particularly, we alternatively optimize an implicit process reward model (PRM) with the policy model to generate step rewards for each action via a multi-turn DPO objective. Theoretical analysis shows that this learning objective produces a step-wise reward function learned from trajectory preferences.
Then the implicit step rewards are used to compute step-level advantages, which are combined with trajectory (or episode)-level advantages for policy updates, creating a self-reinforcing training loop.
We evaluate our method on three challenging agent benchmarks, including WebShop and VisualSokoban, as well as open-ended social interactions with unverifiable rewards in SOTOPIA. 
Crucially, our method shows superior performance over frontier LLMs and strong RL baselines across domains, achieving state-of-the-art results with higher sample-efficiency and training stability.
Further analysis also demonstrates efficient exploration by **iStar** with increased rewards in both step- and episode-level while maintaining fewer steps to achieve task success.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Online Reinforcement Learning (RL) has recently emerged as a powerful technique to improve LLMs’ capabilities in a wide range of tasks ranging from single step reasoning problems like math or coding to multi-turn agentic tasks such as web navigation/tool use. For multi-turn tasks where agents need to interact with an environment multiple times to successfully complete their objective, online RL is a promising approach. A key problem in this scenario is credit assignment — many tasks often have sparse or trajectory level reward only available at the end of the trajectory, and this makes it difficult to learn whether the intermediate actions taken by the agent are good or bad. This paper proposes a method for learning implicit step level rewards and leverage it for online RL training. Specifically, the proposed method, iStar, uses DPO-style log-likelihood ratio as an implicit reward for intermediate steps, and combines it with trajectory level reward (or advantage) to produce per step learning signal. The method is validated on multi-turn tasks like WebShop and VisualSokoban, showing gains over online RL methods using just trajectory level reward signal.

### Strengths
1. The proposed method is interesting. There is prior work that has used the DPO-style log-likelihood ratio as reward (and advantage), but I have not seen it being used as advantage in an online RL framework.

2. The paper is well-written and easy to follow.

### Weaknesses
(**Results are not strong enough**)

The gains we see in the method are not significant enough (in my opinion) compared to the level of complexity introduced in the paper. Specifically, if we focus on the results presented in Figure 4, the difference between the much simpler RLOO baseline and iStar is not significant enough. Moreover, to me it seems like the RLOO validation curve still has a positive gradient near the end of training, and hence more training steps might flip the order of performance between RLOO and iStar.

Moreover, RLOO is a less commonly used RL algorithm. Could the authors present the learning curves from GRPO and GRPO with iStar?

(**Comparison at more training steps**)

Based on the webshop and visual sokoban results in Figure 4, could the authors train for 300 steps (instead of 200) for all the baselines? I am not convinced that the gains may hold as one puts more compute in the baseline vs the proposed method, and would like to see longer training curves.

(**Comparison at GPU hours/FLOP level**)

Due to the training of the implicit PRM, which I believe is a separate model (correct me if I am wrong), iStar would spend significantly more compute compared to regular GRPO/RLOO. Hence, performance comparison at gradient step level is not a fair comparison. I would require a plot with

1. X-axis: GPU hours or total FLOPs

2. Y-axis: Validation performance

Where iStart shows superior performance over the baselines in order to increase my score of this paper.

(**Why does PPO perform so poorly?**)

In Table 1, PPO performs notably poorly, despite using the same idea — a trained critic function to estimate step level advantage. Could the authors explain this result?

### Questions
1. The implicit PRM is parametrized by $\phi$ (denoted by $\pi_\phi$), whereas the policy is parametrized by $\theta$ (denoted by $\pi_\theta$). In practice, what do these symbols mean? What model is used as the implicit PRM? Is $\phi$ different from $\theta$?

2. Why not just perform multi-turn DPO using trajectory level preference pairs constructed using task success/outcome level reward, but in an online fashion? Prior works [1, 2] has already used some variant of multi-turn DPO for sequential decision making tasks, albeit in the offline manner. I would like to see both the online/offline version of this as a baseline, and compare against [1, 2].

# References

[1] Your Language Model is Secretly a Q-Function, https://arxiv.org/abs/2404.12358

[2] Training a Generally Curious Agent, https://arxiv.org/abs/2502.17543

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes iStar, a general credit-assignment strategy for LLM-based agents in reinforcement learning. The method introduces implicit step rewards, which provide dense yet stable feedback signals without explicit step-level annotations. iStar jointly trains a policy model (the LLM agent) and an implicit process reward model (PRM) through trajectory-based direct preference optimization (DPO). The PRM generates implicit step rewards by comparing current actions to those from a previous policy snapshot, providing step-wise credit signals that are aggregated with episode-level advantages. The authors theoretically show that this learning objective is equivalent to a Bradley–Terry model with a step-wise reward function. Empirical results on WebShop, VisualSokoban, and SOTOPIA demonstrate that iStar improves performance and sample efficiency over several reinforcement learning baselines (PPO, GRPO, RLOO, REINFORCE++) and achieves state-of-the-art results. Ablations confirm that implicit step rewards enhance exploration and stability compared to token-level or handcrafted process rewards.

### Strengths
1. The integration of implicit step rewards with trajectory-level DPO represents a creative approach to balancing dense feedback with low variance in long-horizon RL. It avoids the pitfalls of both sparse end-of-trajectory rewards and noisy token-level signals.
2. The method achieves consistent improvements across diverse benchmarks, and this indicates good adaptability of the framework.
3. A solid theoretical proof is provided to ground the method. The equivalence proof connecting the DPO objective to a step-wise Bradley–Terry model adds credibility to the approach and helps ground the intuition behind.

### Weaknesses
1.  The framework’s performance likely depends on the capability of the base model, the nature of the task environment, and the quality of outcome reward verifiers. These dependencies are not theoretically analyzed, and the paper provides limited discussion about how iStar generalizes under varying conditions. This limits the method’s theoretical generalization and reproducibility.
2.  Some key components, such as the implicit PRM architecture, reward verifier setup, and hyperparameter tuning strategy (e.g., α, β), are insufficiently detailed. While the authors mention releasing code, more explicit descriptions or pseudo-code would strengthen reproducibility and credibility.

### Questions
1. How does iStar perform when using base models with weaker reasoning capabilities, and can the implicit reward model remain stable and effective when learning from low-quality rollouts?
2. what conditions—either theoretical or empirical—ensure that the learned implicit step rewards remain unbiased and consistent throughout training?
3. How robust is the framework when dealing with environments where outcome rewards are highly stochastic or unverifiable?

### Soundness
3

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
4

### Summary
The paper "Agentic Reinforcement Learning with Implicit Step Rewards" proposes a new RL-based optimization process for finetuning LLMs.   Inspired from implict rewards models that can be derivated from the DPO formulation, authors proposed a mechanism to train a step-wise function, that can be used to efficiently guide classical RL approaches, such as GRPO. Experimental results indicate the good behavior of the proposed learning scheme.

### Strengths
- Good Experimental results: Authors demonstrate the superiority of their approach on various tasks compared to different baselines. Differents ablations also allow to highlight the benefits of the different components of the approach. 

- Acting at a token-level is crucial for complex tasks

- Well written paper, easy to follow

### Weaknesses
- Rather straightforward approach that takes most of its contribution from zhong et al (for the definition of reward models at a token level). The main novelty is to me the use of a $\pi_{old}$ policy that is periodically updated rather than a constant reference policy. I think this is not a fully innovative contribution, as many works attempted to make evolve ths reference during training. 

- The use of GRPO with adavanges computed over any steps and trajectories in the group looks questionnable to me (see questions above).

### Questions
- For multi-turn RL settings, approaches such as POAD [] propose to modulate the discount factor at the utterace level rather than at a token level, with adapted critics. Are you doing something similar ? 

- Although originally designed for imitation learning rather than reinforcement learning, some GAN-based methods introduce token-level discrimination mechanisms that closely mirror several aspects of the implicit reward modeling proposed here. Notably, I think to [2], which considers an objective $\dfrac{\pi_\theta D_\phi}{\pi_{\theta^{old}}} \approx \dfrac{\pi*}{\pi_{\theta^{old}}} \dfrac{\pi_\theta}{\pi_{\theta^{old}} + \pi*}$, where $D_\phi$ is a discriminator trained to distinguish good from bad utterances at a token-level, $\pi*$ is the optimal policy and $\pi_{\theta^{old}}$ is the policy $\pi_\theta$ from the previous iteration. This looks very close to your training scheme. I think you should discuss the proximity with this work (for a different setting but this could be adapted) in your paper.

- The theoretical analysis section is fully taken from Zhong et al. The only difference looks to lie in the use of $\pi_{old}$ rather than $\pi_{ref}$, but this is the same at that point. I think authors should remove that part and only mention zhong when introducing their reward function. 

- I understand that the reward modelling is based on works that implicitly consider reward models through policy ratios. But in your case you train a reward model based on a policy fully dedicated for this. Is it still implicit ? For me this implicit denomination should be removed from title and many places of the paper, as it is not. You explicitely build a model of the reward. 

- From table 3, I am not sure what is new in w/iStar row compared to the w/ token-level process rewards. Please specify. 

- The authors use GRPO, which does not seem particularly suitable for handling step-wise (or token-level) rewards. In particular, I have concerns with the use of a global normalization in (4), that normalize all rewards from all timesteps and all trajectories from the group. Different rewards from different situations play at different scales. I feel there is a risk that only dominating rewards (not in the sense of a advantage over other tokens in different situations, only because for instance any correct sentence must contain the corresponding token)  have an impact on the training, hindering all the contribution from other positive markers in different situations. Wouldn't it be more relevant to use PPO with a trained critic to compute advantages ? 


- you say that one of the two main difference with DPO is that "preferences are learned from pairwise trajectories". Going back to the DPO paper I cannot see the difference regarding that aspect. Please clarify. 

[1] Wen, M., Wan, Z., Zhang, W., Wang, J., & Wen, Y. (2024). Reinforcing language agents via policy optimization with action decomposition. arXiv preprint arXiv:2405.15821.

[2] Lamprier, S., Scialom, T., Chaffin, A., Claveau, V., Kijak, E., Staiano, J., & Piwowarski, B. (2022, June). Generative cooperative networks for natural language generation. In International Conference on Machine Learning (pp. 11891-11905). PMLR.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the problem of sparse and unverifiable rewards for agentic LLMs. It proposed a credit-assignment strategy for LLM agents that introduces step-wise implicit step rewards learned by alternately training a PRM and the policy. The authors highlight that their approach does not need explicit reward annotations, can stabilize training with step-level rewards, and only relies on trajectory preferences.

### Strengths
1. The method is well motivated, introducing step-level reward for agentic tasks to avoid the training instability caused by either token-level reward signal being too dense or outcome reward being too sparse.   

2. The proposed method is flexible as it can be easily integrated into existing RL frameworks like GRPO, RLOO, etc.

### Weaknesses
1. The idea of step-level PRM has been studied in previous work[1]. 


2. The model introduces two hyperparameters: $\beta$ as reward temperature and $\alpha$ that balances step level and episode level advantages. A sensitivity sweep and guidance for setting them are important for practical adoption of the proposed method.


[1] Yuan, Lifan, et al. "Free process rewards without process labels." arXiv preprint arXiv:2412.01981 (2024).

### Questions
1. Can the authors elaborate on why dropping the KL divergence penalty when training the policy model? 


2. How sensitive is the model performance to the hyperparameters $\alpha$ and $\beta$?

### Soundness
3

### Presentation
3

### Contribution
3
