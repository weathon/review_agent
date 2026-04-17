# When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training

- Decision: Accept (Poster)
- Scores: 10, 2, 6

## Abstract
While Large Language Models (LLMs) hold promise to become autonomous agents, they often explore suboptimally in sequential decision-making. Recent work has sought to enhance this capability via supervised fine-tuning (SFT) or reinforcement learning (RL), improving regret on the classic multi-armed bandit task. However, it remains unclear how these learning methods shape exploration strategies and how well they generalize. We investigate both paradigms by training LLMs with SFT on expert trajectories and RL with a range of tailored reward signals including a strategic, regret-shaped reward to reduce variance, and an algorithmic reward that enables oracle imitation. The resulting agents outperform pre-trained models and achieve performance comparable to Upper Confidence Bound (UCB) and Thompson Sampling, with robust generalization to 6$\times$ longer horizons and across bandit families. Behavioral analysis reveals that gains often stem from more sophisticated but greedier exploitation: RL/SFT agents are more prone to early catastrophic failure than pre-trained models, prematurely abandoning exploration. Furthermore, agents trained to imitate UCB learn to outperform their teacher by adopting more exploitative variants. Our findings clarify when each training paradigm is preferable and advocate tailored reward design and evaluation beyond average regret to promote robust exploratory behavior.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper investigates how Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) training paradigms shape the exploration strategies of Large Language Models (LLMs) when solving Multi-Armed Bandit (MAB) tasks, treating the LLMs as meta-bandit agents.

I currently think this paper provides a lot of value to the "learning to learn" / "learning to explore" community. However, I raised several questions and based on the authors' rebuttal response, I reserve the right to change/update my score.

### Strengths
1. The paper has extremely clear writing
2. The authors disambiguated different training strategies and setups very well. 
3. This paper introduced token-level reward derived from GAE, which improved upon prior work.
4. Each reward design is principled and reasonable -- incorporating all possible learning signals. 
5. The paper is well-situated compared to prior works, investigating areas where prior works didn't, adding critical knowledge/understanding to the entire teaching LLM to explore space.

### Weaknesses
The paper doesn't have obvious flaws/weaknesses, but here is some nitpicking weaknesses:

1. As a field, the learning to explore community needs to move beyond toy setting. [1] proposed and [2] included the MovieLens environment, which is a good first step. Behaviors of RL vs SFT and whether we need $\pi^*$ (optimal exploration policy) need to be answered in slightly more realistic setting. Many algorithms (like SGD) might be flawed in constructed setups, but they work really well in practice. Even though I think this paper has enough value without exploring those more realistic settings, I would encourage the authors to extend the paper to cover those settings.

2. The conclusion from the experiment seems a bit unclear (see the question part).

[1] Nie, Allen, et al. "Evolve: Evaluating and optimizing llms for exploration." arXiv preprint arXiv:2410.06238 (2024).

[2] Schmied, Thomas, et al. "Llms are greedy agents: Effects of rl fine-tuning on decision-making abilities." arXiv preprint arXiv:2504.16078 (2025).

### Questions
Typo 1: In Figure 3, it says RL-STG, it should be RL-STR 

1. RL-STR is not significantly better than RL-OG. Just comparing the average, in 3 out of 8 domains (Figure 3), it's better. It's worse in 3 out of 8. In 2 it's about the same? For people who might want to train exploration policy in a realistic setting, how would you give them suggestions on what reward design to use? 

2. Table 2, you did some behavioral analysis and citing a prior work. Krishnamurthy et al. (2024) also proposed MinFrac. Do you think this is a metric worth including? 

3. [1] also proposed to fit a function form over steps $T$ on average cumulative regret -- and try to describe the slope/speed of learning/unlearning. Would such analysis yield additional insights for your setting?

4. What future directions are opened up by your analysis, besides moving to more realistic domain? In your GRPO training (with RL-OG), do you see length hacking (i.e., with training continues, the length in thinking tokens becomes longer and longer)? In appendix, we saw that RL with ALG reward learns to compute UCB. Can you:
   - Provide sampled generations with base model but sampled using your prompt/pipeline? Do they already know that they should calculate UCB values?
   - Can you provide sampled generation from RL-OG and RL-STR for us so we know if these also learned (on their own) to compute UCB values?

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
This submission studies how LLMs learn to balance exploration and exploitation when trained to solve multi-armed bandit (MAB) problems, using both supervised fine-tuning and reinforcement learning. They train LLMs on various MAB environments using RL with three forms of reward design: original reward (i.e. the raw environment rewards), strategic reward (a normalized, regret-shaped reward that reflects how close the agent’s action is to the optimal arm at each tilmestep), and algorithmic reward (a signal that rewards the agent for matching the decisions of an expert oracle, specifically the UCB algorithm). For supervised fine-tuning, they train the LLM on synthetic data generated from UCB expert trajectories, including step-by-step chain-of-thought demonstrations of UCB value calculations. 

They evaluate the performance of these methods (when fine-tuning Qwen models) on Gaussian and Bernoulli MAB environments. They find that the fine-tuned models outperform vanilla models on these tasks, and that policies trained with UCB imitation perform best. These methods also perform comparably to UCB and Thompson sampling. 

Upon further analysis, the authors find that these gains often come from the models adopting greedy behaviors and under-exploring.

### Strengths
Using LLMs to solve decision-making tasks under uncertainty is an important next frontier, so the problem being studied is well-motivated. While the authors are not the first to fine-tune LLMs to solve these sorts of bandit tasks, their proposed reward designs are novel, from what I can tell. The finding that gains often stem from more greedy behavior is also interesting, and largely matches the findings of Krishnamurthy et al for vanilla models which are not fine-tuned.

### Weaknesses
I am not entirely sure what this paper is trying to accomplish. The typical motivation for studying LLMs’ ability to solve bandit tasks is because we want to design AI models capable of solving complex, text-based, decision making tasks under uncertainty. This is because we don’t currently have good algorithms for solving these types of tasks. Bandits get to the essence of the exploration/exploitation trade-off, and so they can be a useful abstraction. 

However, I am not sure what utility we get from fine-tuning LLMs to be more like existing algorithms, without evaluating their performance on some complex task that existing algorithms do not do well in (or are not even applicable to). (We don't need to reinvent the wheel to solve MAB tasks, as we have good algorithms for these types of tasks already.)

### Questions
What are you trying to show when evaluating these fine-tuned models on MAB tasks? It is not very surprising that a transformer will behave like UCB if you train it to.

### Soundness
3

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
This paper studies how different fine-tuning paradigms shape the exploration and exploitation behaviors of LLMs in multi-armed bandit (MAB) settings. The authors train Qwen-2.5 3B and 7B models on MAB tasks with different reward signals:
> RL-OG: standard stochastic rewards, 

> RL-STG: regret-shaped strategic rewards, and

> RL-ALG: imitation of the UCB oracle via algorithmic rewards.

The work demonstrates that both SFT and RL reduce regret and achieve performance comparable to theoretical baselines like UCB and Thompson Sampling, with generalization to unseen environments and 6$\times$ longer horizons. However, behavioral analysis shows that these gains arise from emergent exploitative tendencies since trained agents become greedier and sometimes abandon exploration prematurely. RL-ALG agents trained to imitate UCB outperform their teacher by adopting more exploitative variants of it.
The paper concludes that while both SFT and RL enhance decision-making, they also introduce bias toward short-term exploitation, highlighting the need for reward and data designs that explicitly sustain exploration.

### Strengths
- **Originality and significance**: This work provides a unified framework comparing SFT and RL fine-tuning for LLM agents on the same controlled MAB setup. The strategic and algorithmic rewards are well-motivated and address variance and credit-assignment issues in RL with LLMs. 
- **Quality**: This paper includes experiments across Gaussian and Bernoulli bandit families, including cross-distribution generalization and longer horizons, which strengthen the empirical claims. Metrics like suffix failure and greedy frequency are used to reveal qualitative behavioral shifts alongwith quantitative evaluation.
- **Clarity**: The paper is generally well-written with clear implementation details and code release commitments.

### Weaknesses
- Limited task diversity: Experiments are confined to simple bandit environments, therefore claims about general agentic learning remain somewhat speculative. 

- Model scalability: There is no discussion of whether the observed training and inference dynamics hold even with larger models and across different model families. 

- While the paper uncovers behavioral tendencies, it stops short of offering causal or mechanistic explanations of why greediness emerges under RL/SFT objectives.

- Compared to the analysis of RL approaches in LLM training, the SFT analysis is less thorough, especially regarding data sampling effects and robustness to arithmetic errors. 

- No explicit exploration-promoting baseline: Methods like intrinsic reward shaping or information gain are not compared, though they would provide valuable context.

### Questions
Exploration bias mitigation – Could future training include explicit exploration bonuses (e.g., curiosity or entropy regularization) to test if the emergent greediness persists?

Do the authors have an intuition of how findings might transfer to contextual or Markovian tasks where exploration requires memory and state estimation?

The finding that RL-ALG outperforms its UCB teacher is quite interesting. Could the authors analyze whether this over-exploitation emerges from token-level credit assignment or from the episodic reward structure?

### Soundness
2

### Presentation
2

### Contribution
3
