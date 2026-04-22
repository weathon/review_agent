# Meta-RL Induces Exploration in Language Agents

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Reinforcement learning (RL) has enabled the training of Large Language Model (LLM) agents to interact with the environment and to solve multi-turn longhorizon tasks. However, the RL-trained agents often struggle in tasks that require active exploration and fail to efficiently adapt from trial-and-error experiences. In this paper, we present LaMer, a general Meta-RL framework that enables LLM agents to actively explore and learn from the environment feedback at test time. LaMer consists of two key components: (i) a cross-episode training framework to encourage exploration and long term rewards optimization; and (ii) in-context policy adaptation via reflection, allowing the agent to adapt their policy from task feedback signal without gradient update. Experiments across diverse environments show that LaMer significantly improves performance over RL baselines, with 11\%, 14\%, and 19\%  performance gains on Sokoban, MineSweeper and Webshop, respectively. Moreover, LaMer also demonstrates better generalization to more challenging or previously unseen tasks compared to the RL-trained agents. Overall, our results demonstrate that meta-reinforcement learning provides a principled approach to induce exploration in language agents, enabling more robust adaptation to novel environments through learned exploration strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a meta-reinforcement learning framework, LaMer, which is designed to induce exploration and adaptation in large language model agents. Unlike conventional single-episode RL setups, LaMer uses a cross-episode training paradigm, where an agent interacts with the same environment across multiple episodes, balancing exploration and exploitation through a trajectory discount factor. The method leverages in-context policy adaptation via reflection, enabling agents to improve within trials without parameter updates.

The authors evaluate the method on Sokoban, MineSweeper, WebShop, and ALFWorld show that LaMer outperforms prompting-based and RL baselines (PPO, GRPO, GiGPO). It achieves a good marginal performance gain on complex environments and improved generalization to harder or out-of-distribution tasks. The framework introduces a bridge between meta-learning and RL for LLM-based agents.

### Strengths
- Novel framework design: The cross-episode MetaRL formulation extends standard RL to multi-episode exploration, introducing a new axis of “meta-time” learning that aligns well with LLMs’ in-context adaptation capabilities. The introduction of the trajectory discount factor provides interpretable control over the exploration–exploitation tradeoff.

- Improvements over the current baselines over benchmarks: LaMer consistently outperforms RL baselines across all benchmarks (Table 1), achieving +20% absolute improvement on Minesweeper and WebShop. Figures 3–5 show substantial increases in trajectory diversity and test-time scaling behavior, supporting the exploration hypothesis.

- Good generalization evidence: Demonstrated robustness to task difficulty (Figure 5) and generalization to unseen ALFWorld tasks (Table 2). Shows that meta-learned exploration strategies can generalize beyond the training distribution, which is an important step toward adaptive, open-ended agents.

- Broader impact: Provides a reasonable path to unify RL-based training and in-context learning for future meta-learning language agents. Bridges recent work in test-time compute scaling and reasoning through reflection.

### Weaknesses
- Limited novelty in algorithmic components: While the formulation is clean, most components, cross-episode return accumulation and reflection-based adaptation, are incremental combinations of known techniques. The paper could better articulate what is fundamentally new beyond adapting Meta-RL for LLMs.

- Insufficient analysis of failure cases: The paper lacks qualitative examples or ablation studies on when reflection adaptation fails (e.g., misleading feedback loops or hallucinated self-reflection). As well, it does not have analysis of how context length or reflection prompt quality impacts learning efficiency.

- Sequential dependency and compute inefficiency: Section 5.6 notes that MetaRL training is 2–3× slower due to sequential episodes, but does not propose mitigation strategies or quantify the trade-off precisely (e.g., wall-clock vs. sample efficiency).

- Evaluation fairness concerns: It is unclear whether RL baselines had access to equivalent multi-episode feedback or if they were constrained to single-episode objectives, potentially overstating MetaRL’s advantage.

### Questions
- Meta-objective design: How sensitive is performance to the trajectory discount factor? Could adaptive scheduling of this factor improve learning stability?

- Reflection mechanism: Did the authors experiment with alternative reflection prompt formats or automatic summarization of past episodes? It would be great to add it in the ablations. 

- Training efficiency: Can rollout dependencies be partially parallelized (e.g., via curriculum or pseudo-batch episodes) without breaking credit assignment? It will result higher training efficiency. 

- Exploration quality metrics: Besides trajectory diversity entropy, are there other quantitative metrics used to measure exploration effectiveness?

### Soundness
2

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
4

### Summary
The paper proposes a Meta-RL framework for LLM agents, combining cross-episode RL and in-context reflection. The idea is to train the model to learn to explore by first gathering information, then exploiting it across the following episodes. Results show clear gains over RL and prompting baselines across Sokoban, MineSweeper, Webshop, and ALFWorld. The approach seems effective at leveraging multi-episode structure and reflection to adapt at test time, though it’s unclear whether the gains truly stem from “learning to explore” or from other confounding factors such as longer contextual horizons or additional information flow between episodes.

### Strengths
The idea is clear and well motivated. It’s conceptually simple and general. The evaluation is broad and the study of generalization is interesting. The paper is clear and well-written.

### Weaknesses
**What is actually learned?**
It’s not so clear to me whether the agent is only conditioned on the reflection from the previous episode or on the previous history too? If conditioned on both, it would be good to ablate the reflection mechanism and see how performance holds. Ablating history and only keeping reflections might be interesting too. 

Is the feedback generation capacity trained too? Or just leveraging a frozen model? Maybe the approach doesn’t actually train exploratory behaviors but only improves feedback generation and feedback use in the following episode. 

To make sure it’s not the case, it would be interesting to visualize exploration strategies in the first episode: what are agents doing? Maybe through videos on a website. Looking and analysing generated feedback might be useful too. Overall I would like to know more about the exploratory behaviors this paper claims is being trained. 

It might be interesting too to feed a first episode collected by either 1) random actions, or 2) the base model; then generate feedback and run the second episode with the trained Meta-RL agent. Is there a drop in performance? If so, it means the meta-RL agent truly learned to explore. If not, then exploration doesn’t matter so much, what matters is the feedback offering a “second chance”. 

Are the different episodes in a group the same task instance? e.g. with mines in the same spot in mine sweeper? I assume it’s the case, otherwise exploration would be useless, right? In that case, in Sokoban and MineSweeper, it may not be so much about “learning to explore” as it is about giving a second chance the agent can actually leverage (second chances can’t be leveraged in the standard RL condition because there is no inter-episode memory). 



**Missing information about training task distribution**
It’s unclear how different training tasks are from each other. If they’re similar, “learning to explore” might just mean reusing the same strategy in all task instances? Are agents adapting their exploration (1st episode) to the task instance? Here again looking at replays would be useful.

**Comparisons**
The RL baseline controls for total experience but doesn’t control for the length of experience the agent has access to at any given moment. One way to control for this would be to run the RL agents on longer episodes instead of running it on longer trajectories. 

**Generalization to harder tasks**
I don’t know if you can say that meta-RL generalizes better here: performance drops by similar percentages in both cases. Meta-RL doesn’t “hold better”, it just starts higher. This is a bit of an overclaim. The AlfWorld generalization study is more convincing. 


**Typos:**
* fari comparison -> fair
* Tabel 1 -> Table 1

### Questions
* How is the “empirical distribution over distinct trajectories” computed exactly?
* Are prompting baselines also given access to memories across episodes? Reflection only? or histories too?
* In addition to “bold” indicating best performance, it is useful to make other performance that are not statistically significantly worse than the best (e.g. underlined).

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
5

### Summary
This paper uses meta-reinforcement learning (Meta-RL) to train large language models (LLMs) as language agents across multiple game environments. The inner loop of Meta-RL is implemented as a simple reflection step, where the LLM is prompted to analyze and reflect on previous episodes. The outer loop is optimized using the GiGPO algorithm. The authors show that this approach improves performance compared to both RL and prompting-based baselines across various games. Their analysis further suggests that Meta-RL enhances the exploratory behavior of LLM agents, leading them to produce more diverse trajectories.

### Strengths
- The paper is well written and easy to follow, with clearly stated hypotheses and a clean experimental design. The idea is simple and promising to help improve LLM-agents to explore in RL tasks. 

- It presents strong empirical results showing that meta-RL with reflection substantially improves the performance of LLM agents across multiple game environments.

### Weaknesses
- The paper does not analyze the reflections generated by the LLM. Do they make sense? Do they evolve over time? Is performance improvement mainly driven by changes in the policy or in the reflections themselves?
- The paper does not analyze the reflections generated by the LLM. Do they make sense? Do they evolve over time? Is performance improvement mainly driven by changes in the policy or in the reflections themselves?
- All experiments are conducted with a single LLM (Qwen3-4B). It would be important to test whether the results generalize across different model architectures and sizes.
- Lack of technical details:
    - The paper does not explain how trajectory diversity is computed.
    - Key training hyperparameters (e.g., batch size, learning rate, training budget) are not reported.
    - For procedurally generated environments such as Sokoban, it is unclear whether the same random seeds are used across episodes within a task—both in the RL and meta-RL settings.

### Questions
- Would your method also apply to more classical reasoning tasks where LLMs are trained with RL, such as MATH, BigBench, or similar benchmarks?

- Do you think your approach could work even without the reflection step — relying solely on in-context learning from previous trajectories provided in the prompt?

### Soundness
4

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
This paper introduces the LAMER meta-RL algorithm which makes agents explore better. The algorithm involves training the agent on sequences of related expisodes. After each algorithm there is a reflection step where the agent can write notes which are given to figure episodes. Since the meta-rl algorithm rewards the agent for reward across the entire sequence, the agent is incentivized to explore in earlier episodes and exploit in later ones.

They evaluate on Sokovan, Minesweeper, Alfworld, and Webshop, and find that LaMer produces both better exploration/diversity and higher overall reward.

### Strengths
The paper is well-motivated and is written clearly to explain how the authors are tackling the problem of exploration during RL.

While the Meta-RL framework itself is not novel, the application to LLMs is. The paper shows significant gains over single episode training across multiple benchmarks. The paper also shows out of distribution generalization compared to non meta learning on unseen benchmarks.

### Weaknesses
See the "questions" section.

Beyond this I would be interested in comparing against pass@k metrics for meta-exploration that have previously been explored in RL (for example, see Walder et. al Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems).  (I do not think this is necessary for this to be a good paper! Just a suggestion for extension.)

### Questions
- Why are there no seeds or error bars? RL experiments can have quite a lot of variance and it is typical to do at least 3 experimental seeds for each run.
- How much of the gain is attributed to the self reflection prompting versus the multi episode credit assignment?
   - Would be interesting to see an ablation of whether the reflection step is important or if merely showing the agent past episodes is sufficient.
- Qualitatively, did you notice the sampled trajectories to be more diverse? Entropy or similarity is an easily hackable objective. I'd also appreciate a bit more description on how exactly diversity was calculated, this was not entirely clear to me.

### Soundness
3

### Presentation
3

### Contribution
2
