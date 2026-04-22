# Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
In long-horizon tasks, recent agents based on Large Language Models (LLMs) face a significant challenge that sparse, outcome-based rewards make it difficult to assign credit to intermediate steps. Previous methods mainly focus on creating dense reward signals to guide learning, either through traditional reinforcement learning techniques like inverse reinforcement learning or by using Process Reward Models for step-by-step feedback. In this paper, we identify a fundamental problem in the learning dynamics of LLMs: the magnitude of policy gradients is inherently coupled with the entropy, which leads to inefficient small updates for confident correct actions and potentially destabilizes large updates for uncertain ones. To resolve this, we propose Entropy-Modulated Policy Gradients (EMPG), a framework that recalibrates the learning signal based on step-wise uncertainty and the final task outcome. EMPG amplifies updates for confident correct actions, penalizes confident errors, and attenuates updates from uncertain steps to stabilize exploration. We further introduce a bonus term for future clarity that encourages agents to find more predictable solution paths. Through comprehensive experiments on three challenging agent tasks, WebShop, ALFWorld, and Deep Search, we demonstrate that EMPG achieves substantial performance gains and significantly outperforms strong policy gradient baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies that magnitude of policy gradients is inherently coupled with the policy’s entropy, and proposed EMPG to recalibrates the learning signal via self-claibrating gradient scaling and future clarity bonus. Experiments on WebShop, ALFWorld, and Deep Search show consistent improvements over GRPO and DAPO.

### Strengths
1. The paper formalizes the coupling between entropy and gradient magnitude and propose a correction mechanism rather than heuristic reward shaping.
2. Evaluations span three diverse benchmarks.

### Weaknesses
1. Why should the natural coupling between entropy and gradient magnitude be considered harmful rather than a desirable exploration mechanism? Could attenuating high-entropy gradients hinder the discovery of new behaviors? I understand that EMPG is actually anti-uncontrolled entropy coupling, but the proposed entropy-dependent gradient scaling introduces a biased learning signal, and may favor overly confident, low-entropy behaviors and converge to locally optimal or sub-optimal deterministic policies. It will be helpful to clarify whether the method prematurely collapses to confident but non-optimal trajectories.
2. Only three random seeds are reported.

### Questions
1. Why is Renyi-2 entropy chosen instead of Shannon entropy, which is more standard in RL formulations?
2. About the scaling function, how sensitive is the method to the hyperparameter $k$, any insight behind the chosen eponential form $e^{-kH_t}$? 
3. How sensitive are the results to  $k$, $k'$ and $\zeta$?
4. Appendix D interprets EMPG as optimizing a modified objective $J_{EMPG}$. Is there a guarantee that $J_{EMPG}$ has the same optimal policy as the original $J(\pi)$, if not, what are the advantages and properties of the surrogate objective?
5. Why is step-level entropy superior to token-level entropy (Fig. 3)?
6. How expensive is step-wise entropy computation for large-scale LLMs? And an open question on future directions, can EMPG scale to online or streaming RL settings without prohibitive cost?

### Soundness
3

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
3

### Summary
The paper proposes Entropy-Modulated Policy Gradients (EMPG) to improve reinforcement learning for long-horizon tasks with discrete actions and sparse rewards. 

EMPG introduces two entropy-based mechanisms. The first, Self-Calibrating Gradient Scaling, re-weights policy gradients by amplifying updates for low-entropy steps and attenuating those for high-entropy ones. SCGS is motivated by the phenomenon that the expected gradient norm is monotonically coupled with policy entropy. 

The second, Future Clarity Bonus, encourages the agent to take actions leading to lower-entropy future states, promoting more predictable and stable trajectories. 

Experiments on several LLM-based agent benchmarks demonstrate that EMPG significantly enhances both performance and training stability.

### Strengths
1. The paper addresses an important challenge in training LLM-based agents and provides a well-motivated solution.
2. The proposed Self-Calibrating Gradient Scaling and Future Clarity Bonus are novel techniques, each supported by clear theoretical motivation.
3. The method significantly improves the stability of reinforcement learning training.
4. EMPG is broadly compatible with existing policy gradient algorithms and can be easily integrated as an add-on module.

### Weaknesses
1. The paper lacks a comparison between EMPG and existing entropy bonus–based methods (e.g., SAC). Such a comparison would be valuable, since Self-Calibrating Gradient Scaling and traditional entropy regularization share conceptual similarities — both penalize overly uncertain (high-entropy) policies to converge and encourage more confident (low-entropy) ones to diversify.
2. Compared to Self-Calibrating Gradient Scaling, there is insufficient discussion on the motivation and theoretical justification of the Future Clarity Bonus. The paper only briefly refers to “information gain” without a formal analysis. Moreover, it remains unclear whether encouraging the agent to seek low-entropy states might inadvertently reduce exploration.
3. The interaction between the two proposed techniques appears inconsistent. While Self-Calibrating Gradient Scaling implicitly promotes higher-entropy policies, the Future Clarity Bonus drives the agent toward low-entropy future states. As shown in Table 2, combining them yields only marginal improvement compared to using either component alone.

### Questions
1. How does EMPG compare with existing entropy bonus–based methods?
2. What is the theoretical motivation behind the Future Clarity Bonus, and could emphasizing low-entropy future states unintentionally reduce the agent’s exploration capabilities?
3. Could the authors clarify how combining Self-Calibrating Gradient Scaling with the Future Clarity Bonus benefits agent training compared to using each method individually?

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
3

### Summary
This paper explores the use of a policy's entropy as an intrinsic signal for more efficient credit assignment in long-horizon LLM agents. First, the authors remark a general challenge in policy gradients called "the entropy-gradient coupling problem", where an uncertain step along a trajectory has larger gradients than steps with lower entropy, going against the intuition that confident steps should receive more credit than uncertain ones. To combat this challenge, this paper suggests a modified advantage function which emphasizes low-entropy steps along a trajectory, and attenuates the gradients of high-entropy steps. In addition, the authors introduce an additional reward signal at every step in order to minimize entropy at the next-step. The combined modulated advantage function can easily be applied to existing RL fine-tuning methods, and the authors demonstrate that both components of their method improve the final performance of existing methods on various long-horizon LLM agentic tasks.

### Strengths
- The authors present a clear and intuitive framework for reshaping rewards using an intrinsic signal within LLMs in the multi-turn RL fine-tuning environment. 
- While the use of entropy to regularize, motivate exploration/exploitation, or even self-validate RL agents has been explored in other works, the idea of using the step-level entropy of an agent to redistribute the sparse reward signal of long-horizon tasks is interesting, and the authors show that it is a direction worth exploring.

### Weaknesses
- The experiments lack statistical significance and robustness. While it would be nice to get more seeds for Table 1, it would also be informative to report any variance statistics of the three seeds that were collected (for example min/max). It is difficult to contextualize the improvements reported without a sense of variance in performance between seeds, for instance, is a +0.6% in success rate statistically significant?
- It would be nice to further contextualize the work within the space of traditional RL and credit assignment, as credit assignment in long-horizon RL is an existing litterature outside the scope of LLM agents. Particularly since the authors claims in line 479 "general-purpose method for variance reduction and credit assignment". It should be specified that this work is focused on LLM multi-turn agents, or additional evidence on traditional long-horizon RL environments should also be presented.
- Claims of variance reduction (by attenuating uncertain steps) would be nice to verify experimentally, even if it's in a toy experiment, since no theoretical results suggest any variance reduction by the proposed method.

### Questions
- In line 179, the paper says "a confident and correct step should be reinforced strongly, but its naturally small gradient limits its impact". Is there some way to verify this experimentally? If a step has low entropy, and was the correct step, why should the gradient be large if the model is already very likely to produce this correct step (due to its low entropy)?
- Similarly, it is noted that "the large gradients from highly uncertain exploratory steps can introduce noise and destablize training". Can we further verify this by showing the gradient variance of different training methods with and without **gradient scaling**? The results in Figure 2 show part of this answer, but fail to decouple the results from the Future Bonus, and does not directly measure the noise of the highly uncertain gradients steps. 
- How does the future clarity bonus differ from a general framework for step-wise entropy minimization? The authors state "the Future Clarity Bonus, which can be conceptually justified through the lens of information theory. [...] the bonus encourages actions that yield high Information Gain about the optimal future path". Can you elaborate on this? 
- I am not sure I understand the results from Figure 3. The distribution looks similar to the results from https://arxiv.org/abs/2506.01939 : lower entropy steps undergo smaller changes in entropy. Even if the opposite is true, how does that motivate the design of EMPG? 


### Minor comments
- Typo in line 176 "Equation equation 1".
- The font size of some figures are very small, especially Figure 3.
- Line 155: PPO is not a value-free policy gradient.
- Line 158: PPO does not rely on trajectory-level credit assignment, as it uses a value function to assign credit per-step despite a sparse reward signal.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the long-term credit assignment problem in multi-turn RL for LLM agents. The authors observe that the standard policy gradient tends to update logits more heavily for high-entropy steps, which may misalign credit assignment. They propose Entropy-Modulated Policy Gradient (EMPG), which introduces a self-normalized gradient term that prioritizes low-entropy steps and an auxiliary term that minimizes next-step entropy in proportion. Experiments on *web navigation*, *text-based embodied games*, and *information search* benchmarks using Qwen2.5-Instruct models show performance and stability improvements.

### Strengths
- Tackling multi-turn RL is valuable: this setting is more challenging than single-turn reasoning and deserves deeper investigation.  
- The paper is clearly written and easy to follow.  
- Applying advantage redistribution through entropy modulation is a simple and scalable method to recalibrate gradient credit over steps.  
- The proposed method yields consistent gains and stability improvements on multiple popular LLM-agent benchmarks.

### Weaknesses
**Unclear or unconvincing motivation for entropy modulation:**  
   The central inductive bias—that low-entropy steps deserve larger gradient updates—lacks theoretical justification and may even contradict empirical findings in recent literature. For example, [Cheng et al., 2025] and [Wang et al., 2025] show that **high-entropy steps often correspond to forking or decision-critical tokens**, while low-entropy ones are syntactic continuations. From this hierarchical view, high-entropy (forking) tokens arguably *should* receive stronger updates to improve exploration breadth, not weaker ones, as shown in [Wang et al., 2025]. The paper needs a clearer argument—either theoretical or empirical—for why emphasizing low-entropy steps improves long-horizon tasks.

[Cheng et al., 2025] Reasoning with Exploration: An Entropy Perspective on Reinforcement Learning for LLMs

[Wang et al., 2025] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

**Lacking detailed comparison on related methodologies:**  
 Equation 2 seems to minimize next-step entropy only, not the full trajectory entropy; this distinction should be clarified.  The connection to [Agarwal et al., 2025]—which discusses entropy minimization in reasoning tasks—is mentioned but not elaborated. A comparative discussion would help position EMPG among existing entropy-based RL variants.  

[Agarwal et al., 2025] The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning


**Empirical evaluation is limited:**  
   The study lacks analyses on the introduced temperature parameters $k$ and $k'$, both fixed to 1 without sensitivity checks. Evaluation is also limited to the Qwen2.5 family; testing on other base models would help demonstrate robustness.

### Questions
- Typo: in Eq. 1, $\pi$ on the RHS should be $\pi_\theta(s)$?
- Proposition 1 links expected gradient norm with Rényi-2 policy entropy. While interesting, this is not directly related to Shannon entropy used in practice. A clearer comparison like inequality between Rényi-2 and Shannon entropy would contextualize its relevance.

### Soundness
2

### Presentation
2

### Contribution
2
