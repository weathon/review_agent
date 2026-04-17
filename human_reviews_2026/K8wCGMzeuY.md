# Internalizing World Models via Self-Play Finetuning for Agentic RL

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Large Language Models (LLMs) as agents often struggle in out-of-distribution (OOD) scenarios. Real-world environments are complex and dynamic, governed by task-specific rules and stochasticity, which makes it difficult for LLMs to ground internal knowledge in those dynamics. Under such OOD conditions, vanilla RL training often fails to scale; We observe Pass@k–the probability that at least one of k sampled trajectories succeeds–drops markedly across training steps, indicating brittle exploration and limited generalization. Inspired by model-based reinforcement learning, we hypothesize that equipping LLM agents with an internal world model can better align reasoning with environmental dynamics and improve decision making. We show how to encode this world model by decomposing it into two components: state representation and transition modeling. Building on this, we introduce SPA, a simple reinforcement learning framework that cold-starts the policy via a Self-Play supervised fine-tuning (SFT) stage to learn the world model by interacting with the environment, then uses it to simulate future states prior to policy optimization. This simple initialization outperforms the online world-modeling baseline and greatly boosts the RL-based agent training performance. Experiments across diverse environments like Sokoban, FrozenLake, and Sudoku show that our approach significantly improves performance. For example, SPA boosts the Sokoban success rate from 25.6% to 59.8% and raises FrozenLake score from 22.1% to 70.9% for Qwen2.5-1.5B-Instruct model.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a world model-inspired finetuning approach for LLMs, called SPA, to solve symbolic board game like tasks. 
- First, they define a cleaner abstracted state representation of the game. 
- Then, they prompt the agent to first reason about the current state and the predicted state it should be in, before outputting the actual action: `<reasoning>`$\hat{s}, \hat{s}'$`</reasoning><answer>`$a$`</answer>`. The agent interacts with the environment to create ground truth training trajectories containing $s, s', a$, to supervise the reasoning and action outputs of the LLM. SFT is done on the agent on these traces to make it reason and predict what happens in the game before outputting the action.
- Finally, they do PPO to optimize the LLM's actions to maximize reward in the game. 

They evaluate SPA over a variety of models (Qwen, Llama, etc.) on Sokoban, FrozenLake, and Sudoku. They show that their method outperforms just doing PPO, and an alternative world model approach.

### Strengths
The idea of training the agent to do explicit world model reasoning as an SFT task before RL finetuning is nice, as it allows the agent to learn from dynamics data without needing reward labels.

The paper is nicely written, and easy to understand.

The experiments have through analysis and ablations, although I am concerned about their toy nature.

### Weaknesses
- The main weakness is the evaluation of the agent in seemingly  toy and synthetic tasks, like a very small 6x6 grid game. Are there more interesting and hard reasoning tasks where this method could work, like math, symbolic reasoning, etc? 

- The method does not seem too general, since it requires a handcrafted state representation which requires a domain expert to provide. 

- The data generation is susceptible to failure, since the agent is creating its own labels. If the agent is bad at navigating in the environment and exploring relevant parts of the task,  then wouldn't all of its ground truth trajectories be useless at helping the agent get better? For example if the agent in sokoban is stuck in a corner of the room. 
    - Using historical experience to create labels also seems related to the concept of hindsight experience replay or hindsight relabeling, from goal conditioned reinforcement learning. And these methods suffer from hindsight bias, which is the problem I mentioned above.

### Questions
See concerns above.

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
This paper introduces an agentic LLM learning method by incorporating the idea of model-based RL and building an internal world model inside the LLM. During policy training stage, the LLM agent with an internal world model can serve as a good initialization. The author does experiments on diverse environments like Sokoban, FrozenLake, and Sudoku to show the effectiveness of their approach.

### Strengths
1. The idea of learning an internal world model is worth further thinking. 
2. The author does some experiments to show the effectiveness of their method.

### Weaknesses
1. What you’re doing is a bit of overkill — you trained a whole world model just to initialize a policy, but the world model wasn’t actually used for simulated interactions to generate more data. Moreover, since your title mentions \textit{self-play}, I believe it should involve interacting with your own world model — that would be true self-play.
2. If you want to show \textit{transition-model learning is key to RL scaling}, I expect to see a larger scale experiment, at least in a more complicated, multi-task benchmark, e.g. a robotic environment, to show if it really helps for general agentic LLM learning.

### Questions
1. Since you have trained the world model with the environment data, I’m wondering if some offline model-based RL methods can be applied to learn the agentic model with the same environment data used to train the world model. 
2. I will consider improve the score if you show some results about the concerns mentioned in weakness and question part.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces SPA (Self-Play Agent), a two-stage framework to improve LLM agents in out-of-distribution (OOD) environments. The model learns an internal world model via self-play supervised fine-tuning (SFT) that (i) rewrites raw grid observations into structured state descriptions (“state estimation”) and (ii) trains transition modeling by predicting next states; ground-truth states from the environment replace model hallucinations in the SFT corpus. PPO is run on top of this initialization.

### Strengths
Clear, simple solution to internalize a world model in LLM agents via SFT, replacing beliefs with ground truth. This cleanly separates dynamics learning from RL credit assignment and contrasts with reward-shaping approaches like VAGEN. 

The decomposition (state estimation + transition modeling), the token-level loss mask, and prompt formats are explicit; training hyper-params and evaluation settings are documented. 

On multiple small models and environments, SPA consistently improves Pass@1 and Pass@k and often beats VAGEN; the paper also identifies and analyzes the OOD Pass@1 vs Pass@k divergence it seeks to fix.

### Weaknesses
All tasks are symbolic, fully observable text games. It’s unclear how the method scales to visual/embodied settings, partial observability, or continuous states where “ground-truth” next states are not trivially serializable. 

PPO is the sole RL algorithm; comparisons omit widely used model-free and model-based baselines for text games and modern agent scaffolds. 

SPA relies on trajectories where the model outputs formatted / and the system replaces them with ground truth. The paper notes instruction-following errors and shows filtering helps, but it does not quantify (i) the fraction of discarded samples, (ii) total SFT tokens vs PPO tokens, and (iii) wall-clock cost. 

The paper documents early exploration (Pass@k↑), then exploitation (Pass@k↓), but provides limited analysis on why SPA changes the diversity of trajectories

### Questions
How many SFT tokens are used relative to PPO tokens? What fraction of self-play samples are filtered for format issues? 

You mention fragility in stochastic settings; which component is most brittle (format compliance, prediction variance, critic learning)? Would modeling transition uncertainty (e.g., distributional next-state tokens or latent ensembles) help?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on tackling the performance of language models under OOD scenarios by learning a world model via self play before policy finetuning. The authors highlight that by cold starting with SFT self play to learn the world model then use it to predict future states, their model can broaden state coverage and better align with real dynamics. They demonstrate that their method can increase both pass@1 and pass@k performance in multiple unfamiliar environments, showing the model's performance gain as well as sample diversity.

### Strengths
1. How to incorporate world model is a timely and important research direction to ground LLMs.

2. The paper is well motivated and clearly presented, tackling the OOD problem with world modeling is intuitive and reasonable. 

3. Included detailed ablations demonstrating the effectiveness of learning transition modeling and better state representation.

### Weaknesses
1. The idea of self play explored in previous literature [1, 2] refers to a multi-agent setup, where LLM agents are used as different players in the game to interact with each other. This work, on the other hand, rolls out the base model in the environment for data collection and claims it to be self play. This is not self play widely considered in previous work in LLM or multi-agent RL[3, 4]. 

[1] Chen, Zixiang, et al. "Self-play fine-tuning converts weak language models to strong language models." arXiv preprint arXiv:2401.01335 (2024)

[2] Cheng, Pengyu, et al. "Self-playing adversarial language game enhances llm reasoning." Advances in Neural Information Processing Systems 37 (2024): 126515-126543

[3] Silver, David, et al. "Mastering chess and shogi by self-play with a general reinforcement learning algorithm." arXiv preprint arXiv:1712.01815 (2017)

[4] Heinrich, Johannes, and David Silver. "Deep reinforcement learning from self-play in imperfect-information games." arXiv preprint arXiv:1603.01121 (2016).

### Questions
1. Could the authors explain why learning the transition kernel expands reachable sets under k stochastic rollouts in Lines 90-91? Learning a transition kernel would help more with accurate planning instead of increasing coverage of reachable sets, which would also help lifting Pass@k. Also, this seems to overlap with the previous claim for broader state coverage. 


2. In Section 4.2, the authors test the effect of using ground truth coordinates as an additional state representation by replacing them with iid random coordinates and show that training collapsed. Random coordinates are likely to mismatch the actual state observation, thus no surprise that the training would collapse. A more meaningful ablation would be to compare the performance with just raw state observations. 


3. Related work should include world modeling and self-play for LLM.

### Soundness
2

### Presentation
3

### Contribution
2
