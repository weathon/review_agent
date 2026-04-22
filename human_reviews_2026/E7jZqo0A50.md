# MARTI: A Framework for Multi-Agent LLM Systems Reinforced Training and Inference

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 2, 8, 6, 2, 6

## Abstract
We present MARTI (Multi-Agent Reinforced Training and Inference), an open-source framework designed to facilitate scalable and efficient learning of multi-agent LLM systems. MARTI supports centralized multi-agent interactions and distributed policy training, with the added capability of multi-turn asynchronous rollouts to enhance training efficiency. The framework includes dynamic workflows for multi-agent interactions, which integrate both rule-based verifiable rewards and LLM-based generative rewards. We validate the effectiveness of MARTI through comprehensive experiments on diverse mathematical tasks, demonstrating that multi-agent LLM-based systems outperform single-agent systems within the same inference budget after convergence. Our contributions lay the foundation for exploring scalable collaborations within LLM-based multi-agent systems and advancing the capabilities of large reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces MARTI (Multi-Agent Reinforced Training and Inference), a unified and open-source framework that integrates multi-agent reinforcement learning (MARL) with LLM-based reasoning systems. The framework bridges the gap between multi-agent interaction (e.g., debate, chain-of-agents, mixture-of-agents) and reinforcement training by combining centralized reward modeling and distributed policy optimization. MARTI supports both rule-based verifiable rewards and LLM-based generative reward models, enabling flexible integration of verifiable and open-domain reasoning tasks. Experiments on several mathematical reasoning datasets (AIME24, AMC, MATH-500) show that multi-agent systems trained with MARTI achieve higher performance ceilings than single-agent baselines under equal inference budgets. The work provides a valuable infrastructure for scaling reasoning through multi-agent coordination and RL-based training

### Strengths
This paper stands out for its conceptual clarity and systems-level contribution. It proposes the first general framework that unifies multi-agent inference with reinforcement-based training, addressing an increasingly relevant problem at the intersection of MARL and LLM reasoning. The design of centralized reward modeling with distributed training is elegant and pragmatic, allowing various workflows (debate, mixture-of-agents, chain-of-agents) to plug into the same infrastructure. The authors also provide strong experimental validation with detailed comparisons and visualizations, showing tangible improvements in reasoning benchmarks. The open-source commitment adds further value, making this paper a meaningful step toward reproducible MAS + LLM research. Overall, the work is technically sound, well-positioned in current literature, and likely to have significant community impact.

### Weaknesses
While the framework is impressive, the evaluation scope remains narrow, focusing primarily on math reasoning benchmarks. It would be helpful to see more open-domain or tool-use tasks to demonstrate generality beyond symbolic reasoning. The paper could also discuss stability, scalability, and cost-efficiency—for example, how well MARTI handles growing agent populations or asynchronous rollouts at scale. Additionally, while reward modeling is a core innovation, the comparison between rule-based and generative rewards is underexplored; more ablations or sensitivity analyses would make the contribution stronger.

### Questions
- Could you clarify how MARTI handles credit assignment when generative reward models introduce noisy or inconsistent feedback across agents?

- Have you tested MARTI on non-mathematical reasoning or tool-use tasks, to validate that the improvements generalize beyond verifiable domains?

- How does scalability behave when increasing the number of agents or asynchronous rollouts—does training stability or throughput degrade?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
MARTI is a framework for multi-agent LLM training and inference, featuring centralised coordination, distributed policy learning, and asynchronous rollouts. It combines rule-based and LLM-generated rewards to optimise multi-agent collaboration. Authors claim that MARTI is achieving increased performance over single-agent systems on mathematical and reasoning tasks.

### Strengths
-

### Weaknesses
- pseudo-code in Listing 1 does not provide value or helps understanding
- Rule- and Generative Reward Shaping seems to be quite important, but is not part of the main contribution and does not show up in experiments, therefore it seems unnecessary.
- Untrained results of models that have not been RL Fine-tuned or MARTI Fine-tuned seem to be unnecessary and blur the picture
- Not showing MARTI Fine-tuned for Budget 1 and 6 in "Figure 2: Average scores of Qwen2.5-3B base and instruct models under different budget and settings." seems cherry-picked
- Same goes for "Figure 3: Average scores of reasoning models under different budget and settings." - no mention of MARTI Fine-tuned for Budget = 1, 3, 4
- There is quite some focus on Multi-Agent Debate (MAD) results, while the discussion and impact on the main hypothesis seems minimal
- The authors state "comprehensive experiments" in the Abstract; however, only a few benchmarks have been evaluated (AMC, AIME, MATH)

-> What I would have expected in the results section is the following experiments and metrics:
- Generally you should aim for 3-10 training runs and report average performances for this to be significant
- Single-Agent as a baseline is fine, but need to look at time/steps to convergence and final performance
- But mainly you should compare MARTI vs other MARL setups without the features and contributing advancements so we can see the effectiveness of such. For example MARTI asynchronous VS MARTI synchronous etc. or a collaborative setup VS non-collaborative setup

### Questions
Please see "Weaknesses" section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes MARTI (Multi-Agent Reinforced Training and Inference), an open-source framework that unifies reinforcement learning (RL) and multi-agent systems for large language models (LLMs). MARTI integrates centralized multi-agent interaction and distributed policy training, supporting asynchronous rollouts and both rule-based and generative reward models to improve training efficiency and scalability. MARTI also introduces modular reward shaping and credit assignment strategies to enhance long-term collaboration and stability during training. Experiments on mathematical reasoning benchmarks (AIME, AMC, MATH-500) show that multi-agent systems trained with MARTI outperform single-agent models under the same inference budget.

### Strengths
1. The framework provides a well-structured MARL formulation that combines centralized multi-agent interaction with distributed policy training, ensuring scalability and modeling consistency.
2. The introduction of a credit assignment mechanism reasonably decomposes global rewards, allowing more stable and interpretable multi-agent learning.
3. The inclusion of off-policy training enables the reuse of historical samples, improving sample efficiency and training stability.
4. The presentation of this paper is clear.

### Weaknesses
1. The paper lacks ablation studies on key components such as credit assignment or off-policy training, making it unclear how these mechanisms individually contribute to performance upper bounds or optimization efficiency.
2. Figure 1 presents the overall MARTI framework but remains too abstract, lacking detailed illustration of key components (credit assignment, reward shaping, and the rule-based verifier mechanisms).

### Questions
Please refer to weaknesses.

Discussion (won't affect the rating):
1. I'm curious whether the credit assignment in MARTI could be implemented using a counterfactual baseline proposed in COMA [1], and how this modification might influence the training stability, interpretability, and overall performance of the multi-agent reinforcement learning process.

[1] Foerster, Jakob, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon Whiteson. "Counterfactual multi-agent policy gradients." In Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1. 2018.

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
The paper introduces MARTI (Multi-Agent Reinforced Training and Inference), a framework that unifies multi-agent LLM inference workflows (Mixture-of-Agents, Chain-of-Agents, Multi-Agent Debate) with distributed RL training and centralized reward/credit assignment. It supports asynchronous multi-turn rollouts, rule-based verifiable rewards for math tasks, and LLM-based generative reward models, then trains per-agent policies with PPO/GRPO/REINFORCE++ (optionally mixing in SFT/DPO). Experiments on AIME24/AMC/MATH-500 claim that, under equal inference budgets, multi-agent RL reaches a higher performance ceiling than single-agent RL.

### Strengths
The paper’s key strength lies in providing a unified and scalable framework that bridges multi-agent LLM inference workflows with distributed RL training, filling a clear tooling gap. The modular design (multi-agent world, centralized rewards, agent-policy trainer) supports asynchronous rollouts, flexible agent compositions (MoA/CoA/MAD), and both verifiable and generative reward models, making it practical for real multi-agent RLHF experimentation. The reward shaping approach is intuitive and history-aware, improving consistency in multi-turn reasoning. Empirically, the framework demonstrates that multi-agent RL can achieve a higher ceiling than single-agent RL under matched inference budgets, with strong gains on math benchmarks (e.g., AIME) and evidence of good scaling behavior with concurrency.

### Weaknesses
1. The paper states that multi-agent settings are compared under “equivalent inference budgets,” but the budget is not formally defined (e.g., tokens, rollouts, GPU hours, or wall-clock time). A clear definition and a standardized reporting format would strengthen the fairness of comparisons.

2. Although substantial compute is implied (e.g., 8×A800 GPUs per agent node), the manuscript does not report training time or convergence duration. Providing training wall-clock time, GPU-hours, and throughput would help readers assess scalability and practicality.

3. The repository includes multiple RL algorithms, yet only REINFORCE++ results are shown. Including results for other supported algorithms (e.g., PPO/GRPO) would help confirm that the observed improvements are not algorithm-specific.

4. Experiments are conducted only with Qwen-based models. Adding results on another model family (e.g., LLaMA or Phi) would improve confidence in the framework’s generality.

5. Experimental figures lack narrative context.
Several plots are presented without a clear accompanying explanation of the experimental question and takeaway. Adding brief guiding paragraphs before each experiment would greatly improve clarity.

6. When running the released code, the MA-MAD results appeared noticeably lower than the reported values. This may be due to configuration differences; confirming the default settings and expected performance ranges in the repo would be helpful.

This work provides a very valuable and well-designed repository. I have tested multiple components and found the implementation largely correct and practical for multi-agent RL research. Addressing the points above — especially clarifying budgets, reporting compute costs, expanding baselines, and improving result presentation — would further strengthen the contribution. With these refinements, I would be pleased to give a higher score.

### Questions
Please see the weakness above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work presents a framework for training "multi-agent" LLM systems with reinforcement learning. In the context of math word problems, the proposed RL framework outperforms single-agent RL and inference-only multi-agent systems.

### Strengths
- The results confirm that the proposed method increases performance beyond that of baselines, whereas many multi-agent systems underperform majority voting

### Weaknesses
- This work only studies math "reasoning" problems instead of natively multi-agent LLM settings.
  - Examples of multi-agent LLM settings would include zero-sum games in SPIRAL (SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning), social deduction games (Training Language Models for Social Deduction with Multi-Agent Reinforcement Learning), or purely cooperative games (Ad-Hoc Human-AI Coordination Challenge)
- Only Qwen-based models are trained and tested in this work. However, these models behave very differently from other base models (like Llama), especially in terms of their RL performance, making it hard to determine if the findings are generalizable to LLMs in general.
- The results from "asynchronous generations" are underwhelming, underperforming synchronous behavior at a concurrency of 32 and having an unimpressive improvement for larger concurrency values.
- There is no ablation over the reward shaping terms or novel components of MARTI.

### Questions
Why does the response length significantly increase for 2 agents and decrease for one agent over the course of training in figure 5?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
MARTI is an open-source framework for scalable multi-agent LLM systems, integrating centralized interaction with distributed policy training to enhance reasoning via reinforcement learning. It introduces dynamic workflows with both rule-based and generative reward models, improving agent collaboration efficiency. Experimental results demonstrate that MARTI-trained multi-agent systems outperform single-agent baselines under equivalent inference budgets.

### Strengths
MARTI enables efficient, large-scale multi-agent learning through centralized interaction and distributed policy training.
Combines rule-based and generative reward models to enhance reward accuracy and adaptability.
Multi-agent systems trained with MARTI outperform single-agent baselines in reasoning tasks under equivalent resource constraints.

### Weaknesses
MARTI’s performance may degrade on tasks requiring unseen reasoning patterns, as pre-trained LLM capabilities still set an upper bound on performance.

### Questions
Although the algorithm’s performance was validated on competitive mathematics problems (e.g., AIME, AMC), can the MARTI framework maintain comparable efficacy in diverse, non-mathematical testing scenarios?

### Soundness
3

### Presentation
2

### Contribution
3
