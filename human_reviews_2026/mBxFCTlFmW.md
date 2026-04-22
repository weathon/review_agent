# Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Training large language models (LLMs) to reason via reinforcement learning (RL) significantly improves their problem-solving capabilities. In agentic settings, existing methods like ReAct prompt LLMs to explicitly plan before every action; however, we demonstrate that always planning is computationally expensive and degrades performance on long-horizon tasks, while never planning further limits performance. To address this, we introduce a conceptual framework formalizing dynamic planning for LLM agents, enabling them to flexibly decide when to allocate test-time compute for planning. We propose a simple two-stage training pipeline: (1) supervised fine-tuning on diverse synthetic data to prime models for dynamic planning, and (2) RL to refine this capability in long-horizon environments. Experiments on the Crafter environment show that dynamic planning agents trained with this approach are more sample-efficient and consistently achieve more complex objectives. Additionally, we demonstrate that these agents can be effectively steered by human-written plans, surpassing their independent capabilities. To our knowledge, this work is the first to explore training LLM agents for dynamic test-time compute allocation in sequential decision-making tasks, paving the way for more efficient, adaptive, and controllable agentic systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper addresses a critical limitation of existing LLM agents in sequential decision-making: extreme planning strategies (either "always planning" like ReAct or "never planning") lead to high computational costs/performance degradation or limited performance, respectively. To solve this, it proposes a conceptual framework for dynamic planning that formalizes the cost-benefit trade-offs of planning (balancing performance gains, computational costs, and behavioral instability).

The authors design a two-stage training pipeline to teach LLM agents when to allocate test-time compute for planning: (1) Supervised Fine-Tuning (SFT) on diverse synthetic trajectories (with 16 planning prompt styles) to prime models with planning behaviors; (2) Reinforcement Learning (RL) (via PPO) to refine dynamic planning in long-horizon environments.

Experiments on two environments (POGS, a partially observable graph search; Crafter, a Minecraft-inspired grid-world) show four key results: (1) Each task has a "Goldilocks" planning frequency (intermediate, outperforming extremes); (2) SFT with explicit plans improves imitation learning and reduces catastrophic forgetting; (3) SFT+RL yields agents with better sample efficiency that plan/replan adaptively; (4) Post-RL agents can be steered by human plans to achieve feats (e.g., collecting diamonds in Crafter) beyond autonomous capabilities.

### Strengths
1. An incremental exploration of dynamic test-time compute for LLM Agents: Breaking through the extreme strategies of "always planning / never planning" and proposing a formalized cost-benefit framework;

2. Two-Stage Training Pipeline: Injecting planning bias through SFT (Supervised Fine-Tuning) and optimizing dynamic strategies via RL (Reinforcement Learning), balancing efficiency and performance;

3. Potential of Human-AI Collaboration: Proving that the planning agent after RL can be guided by humans, enhancing controllability and security.

### Weaknesses
1. Overclaim: The authors assert that “this work is the first to explore training LLM agents for dynamic test-time compute allocation in sequential decision-making tasks.” However, this claim is inaccurate. For instance, Hu et al. (2024) have already investigated the same problem in the following work
@article{Hu2024,
  title={Enabling intelligent interactions between an agent and an LLM: A reinforcement learning approach},
  author={Hu, Bin and Zhao, Chenyang and Zhang, Pu and Zhou, Zihao and Yang, Yuanhang and Xu, Zenglin and Liu, Bin},
  journal={Reinforcement Learning Journal},
  year={2024}
}

Indeed, several findings reported in the current submission appear consistent with or have been previously revealed in Hu et al. (2024). A more comprehensive literature review is therefore recommended, particularly one that explicitly clarifies the relationship and differences between this work and Hu et al. (2024).

2. Model Scale Limitations: Only tested Llama-3.1-8B (fine-tuned) and Llama-3.3-70B (zero-shot), without verifying the scaling effects of larger models;

3. Environmental Scope Limitations: Experiments were only conducted in POGS and Crafter. The scope needs to be expanded to scenarios such as robot control and real-world decision-making;

4. Simplified Planning Mechanism: More complex planning strategies (e.g., hierarchical planning, multimodal planning) have not been explored.

### Questions
See the weakness Section.

### Soundness
3

### Presentation
3

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
The work presents a framework for training LLM agents to dynamically allocate test-time compute for planning. The framework is a 2 stage system, where the model is first SFT on synthetic data to seed the model for dynamic planning and then RL is applied to refine and enhance this capability to perform the best in long-horizon environments while reducing costs. Using this framework, the authors show that this setup can train dynamically planning agents which are simultaneously sample efficient and able to perform better on complex tasks.

### Strengths
The methodology section of the paper is quite thorough and provides a detailed breakdown of their dynamic planning architecture. The problem statement is formalized carefully and the authors do a good job to break down different aspects of the costs related to planning, as well as the dynamics of planning in Sections 3.1 and 3.2. The flow of the results section in Section 5 is quite good as the authors show a "planning Pareto frontier" and use this as motivation for their results in Section 5.2 and 5.3. Results show that their method outperforms baselines.

### Weaknesses
The main weakness is a limited number of datasets tests and a lack of experimental results overall. The main experimental results in this paper are completely contained with Figure 1, and the authors only show their main SFT and RL results using the Crafter dataset. Results would be significantly stronger if there was more than just 1 dataset and also 1 base model evaluated using their framework. 

Furthermore, there is no direct comparison between the RL trained model and the pure SFT based model. There is not a single table anywhere in the paper where the actual comparison between zero-shot, SFT, and SFT + RL is compared, which make it very difficult to evaluate how well this framework performs compare to pure SFT and prompting. For instance, in Figure 1a, it seems like a naive prompted model which plans every 4 steps achieves the same performance as the SFT based model, which looks concerning as I would expect SFT model to perform better. Figure 1a also uses "Normalized Score" on the Y axis, where as Figure 1c and 1e use "\% of max reward", are these the same metrics for Crafter? 

There is also not a single result looking at total number of tokens generated for plans or actions in the main paper. Figure 1e and 1f only show performance and planning frequency, but one of the benefits of their framework, theoretically, is reduced costs from getting rid of superfluous planning when it is unnecessary. Analysis would be significantly strengthened with some kind of results or table detailing these results.

Furthermore, in section 5.3, authors state that:

"We further observe that the planning cost penalty C tokens introduced in Section 3.3 effectively adjusts agent behavior
in proportion to the penalty magnitude, with higher costs leading to reduced planning frequency
and length during training while preserving task performance, indicating that agents can flexibly
adapt their planning strategies to different computational constraints" [lines 439-443] 

However, isn't this cost only used during RL Training? Once you actually deploy the model, it is not like if you change the cost of planning, suddenly the model will change its behavior.

In addition, authors claim to be the first work to explore training LLM agents for dynamic test-time compute allocation for planning; however, one prior work, UI-Tars [1] seems to be the first work to explore training LLMs for dynamic planning. In [1], Section 4.4, they use ActRe to collect trajectories to train the model to perform planning and reflection dynamically using SFT. At runtime, the model dynamically decides to use System-2 reasoning during inference without being prompted. 

Section 3.2 on Plan Drift not mentioned or referenced anywhere else in the paper. The authors seem to suggest a taxonomy for planning related errors, but there is no analysis or experiments related to this.

[1] Qin, Yujia, et al. "Ui-tars: Pioneering automated gui interaction with native agents." arXiv preprint arXiv:2501.12326 (2025).

### Questions
Can the authors use their framework on any additional datasets or any additional models to show that their result generalize?

Furthermore, can the authors provide a clear numerical table showing the Zero-shot Prompted, SFT (and variants), and RL + SFT (and variants) performance on these datasets?

It would also be very interesting to see an analysis of the # of tokens generated, as well as the prompting frequency for each of the different baselines and variants. Does this RL + SFT method successfully reduce the cost of running the model?

In Figure 1f, it seems to show that the best model seems to plans every 10?-ish steps? Does a naive prompted model that is prompted to plan every 10-ish steps perform worse?

### Soundness
3

### Presentation
2

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
This paper tackles the important question of when an LLM agent should engage in explicit planning versus acting directly, aiming to improve test-time compute efficiency.
The proposed method introduces a learnable gating policy that decides whether to invoke a planner.
The training procedure consists of two stages:

**1**.	Supervised Fine-Tuning (SFT) on synthetic demonstrations to imitate mixed plan-act behavior;

**2**.	Reinforcement Learning (RL) using a reward that trades off planning benefits against compute costs, regularized by a KL penalty to stabilize learning.

Experiments in synthetic or lightweight environments (POGS, Crafter) show that the learned agent can outperform both “always-plan” and “never-plan” baselines, suggesting a non-trivial tradeoff in reasoning allocation.

### Strengths
* This paper raises a relevant and not well explored question about adaptive reasoning allocation in LLM agents.
* This paper provides a conceptually clean formulation that distinguishes between planning and acting decisions.
* The two-stage training (SFT + RL) is straightforward and could serve as a reproducible baseline for future work.
* Preliminary results reveal a potentially meaningful “Goldilocks zone” between under- and over-planning.

### Weaknesses
1.	**Limited Scope of Environments**: The paper's broad claims about "LLM Agents" are supported by narrow evidence. The experiments are confined to two non-linguistic, grid-world environments (a synthetic graph search and Crafter), which limits the generalizability of the findings to more complex, language-native agent tasks.
2.	**Insufficient Baseline Comparisons**: The empirical credibility is undermined by weak baselines ("never-plan" and "fixed-frequency"). The paper fails to benchmark against established, relevant reasoning frameworks such as ReAct,ToT, making it difficult to assess the method's true advantage.
3.	**Incomplete Conceptual Framework**: A significant gap exists between the formal framework (Section 3.1) and its practical implementation. Key concepts, like the "Instability Cost" ($C_{noise}$), are described as conceptual but are never measured or explicitly optimized for; they are only assumed to be implicitly penalized.
4.	**Contradictory Ablation Results**: The paper's core premise is challenged by its own cost penalty ablation (Appendix C.3). The results show that while penalties reduce planning frequency, the final task score remains "largely unaffected". This finding questions the practical importance of the very cost-benefit trade-off the paper aims to optimize.
5.	**Missing Sensitivity Analysis**: The RL training relies on key hyperparameters, notably the KL coefficient, to stabilize the policy. The paper lacks a sensitivity analysis for this coefficient, making it unclear how robust the method is or if the policy is prone to degenerating into trivial "always-plan" or "never-plan" behaviors.

### Questions
1.	Can the authors formally define and measure the planning advantage metric rather than describing it qualitatively?
2.	How does the method behave under different KL penalty coefficients ($\beta$) — does the policy degenerate into trivial always/never behaviors?
3.	How would the model perform if applied to real reasoning benchmarks such as Mathematical reasoning (e.g., GSM8K, MATH) or program synthesis benchmarks (e.g., HumanEval)?”
4.	Does the gating module generalize across environments, or must it be retrained per task?

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
The paper studies how an agent can dynamically decide when to spend test-time compute on planning. The core idea is to treat planning as a state-dependent cost–benefit trade-off: plan when the expected value gain (a “planning advantage”) exceeds the costs (tokens/latency/instability). The training recipe has two stages: SFT priming on trajectories where a teacher plans at varied intervals, so the student learns both the format and the option to plan or not; then RL fine-tuning (PPO) to optimize task reward, optionally with a planning-cost penalty, so the agent learns when planning is worthwhile. In Crafter (and a synthetic POGS environment), the SFT+RL dynamic planners are more sample-efficient than non-planning baselines and can be steered by human-written plans to reach achievements beyond their solo ability.

### Strengths
1. The work isolates the when-to-plan decision, reports a “Goldilocks” effect (best performance at intermediate planning frequency), and motivates learning-based control of test-time compute rather than fixed schedules.

2. After RL, planning agents are more sample-efficient and reach more complex achievements than matched non-planning baselines; they can also be human-steered post-RL.

3. Zero-shot analyses with Llama-3.3-70B-Instruct in POGS/Crafter support the non-monotonic compute–performance relationship

### Weaknesses
1. The RL experiments fine-tune Llama-3.1-8B-Instruct; broader open-source families (e.g., Qwen/Mistral/OLMo) under the same RL protocol would better test robustness.

2. RL results are centered on Crafter. Additional partially observable, long-horizon tasks (e.g., ALFWorld) would test generality.

3. What is the training cost?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
