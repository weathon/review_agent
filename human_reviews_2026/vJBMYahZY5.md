# MSearcher: Self-Reflective Search Agent Empowered by Monte Carlo Tree Search Based Data Synthesis

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 4

## Abstract
Recent advances in reinforcement learning (RL) have enabled large language models (LLMs) to perform multi-turn chain-of-thought (CoT) reasoning with tool use, where web search serves as the most critical tool for answering complex questions. However, most existing methods apply RL directly to off-the-shelf models without a supervised fine-tuning (SFT) cold start, resulting in unstable training and limited tool invocations. This difficulty is exacerbated by the high cost of curating long reasoning trajectories, which are expensive to annotate and prone to factual drift. We propose MSearcher, a two-stage trained search agent that combines reflective thinking with robust tool use for complex reasoning. A central contribution is an efficient data construction framework based on Monte Carlo Tree Search (MCTS), which produces self-reflective reasoning trajectories for the SFT cold start. This framework leverages both correct and flawed rollouts to generate natural and diverse reasoning data. We adopt a two-stage pipeline, first applying SFT with our constructed data and then further training the model with RL, achieving substantial improvements on multi-hop question answering: 67.6\% on HotpotQA and 52.0\% on Frames. These results highlight the importance of high-quality SFT in stabilizing RL and equipping LLMs with robust long-horizon reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes MSearcher, which is a self-reflective search agent for open-domain multi-hop question answering task. The agent is trained with Supervised FineTuning (SFT) first, and then trained with Reinforcment Learning (RL) using Dynamic sAmpling Policy Optimization (DAPO). The data used for training is synthesized by rollout using Monte Carlo Tree Search (MCTS), where each node is a partition of tasks and the leaves are lists of atomic tasks. The atomic tasks are then answered by the rollout model in a logically topological order to form a trajectory towards the final answer. The trajectories that leads to the correct final answer are used for supervised learning and comparison with other trajectories; the rest are categorized as either retrieval, reasoning or decomposition error, and can be rewritten as a self-reflective trajectory that "turns" to the correct trajectory on the first incorrect step. The proposed model outperforms several baselines on in-domain and out-of-domain multi-hop search task.

### Strengths
1. The paper is well-written and easy to follow. The paper consists of two parts: MCTS data curation and SFT+RL training with the curated dataset, which are both very clearly conveyed. The MCTS's "exploration" and "simulation" are different from the usual use (a route from root to leaf on MCTS is not directly a solution, but only a division of tasks), but this is clearly explained in the paper and illustrated in Fig. 1.

2. The ideas are intuitive: the use of MCTS (which is also essentially a planner-executor multi-agent framework) does not only increases the possibility of successful rollouts with the limited model ability, but it also provides higher data efficiency - the "incorrect trajectories" can be recycled into trajectories with reflective behavior.

### Weaknesses
1. The purpose of using MCTS is to get rid of the depenence on expert large reasoning models (line 56), but the authors still use QwQ-32B, which is a much stronger model than the final MSearcher, to generate data. This design somewhat contradicts with the purpose - can the author further explain why do we not want to use expert large reasoning models in the first place, and how using QwQ-32B still supports this motivation? 

2. The empirical evaluation can be improved:

a)  the ablation study does not include any experiment about the hyperparameter of MCTS, or analysis on the dynamics of data curation (e.g. how many trajectories are successful, what is the average number of steps in total, what is the average number of steps before failure for failed trajectory, what is the ratio for each type of error defined in Sec. 3.1.4, etc.)

b) In Tab. 3, the result shows that MSearcher works better with self-reflective data. However, it is unsure whether this performance difference comes from the reflective behavior, or simply because it is now trained with less data.

**Minor Weakness**

1. Fig. 1,  contry -> country.

### Questions
I have two questions: 

1. Is the target question given in the form of multiple queries (i.e. $n>1$ in the prompt at line 90 $q=\{q_1,q_2,\dots,q_n\}$), or are all subquestions the product of MCTS?

2. in line 429, the authors mention that the performance decrease of SFT is because "a stronger bias toward tool usage introduced by the SFT data"; on the other hand, the lower average tool usage getting lower "consequently" leads to final performance decrease (line 425-426). Also, "SFT gives a higher initial tool usage, leading to better performance" (line 437-438). The effect of tool usage on final performance seems contradictory. Is more tool usage a good thing or bad thing?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MSEARCHER, a two-stage trained search agent designed to perform complex multi-hop question answering by combining reflective reasoning with robust tool use. The key innovation is a data construction framework based on Monte Carlo Tree Search (MCTS) that generates self-reflective reasoning trajectories for supervised fine-tuning (SFT), serving as a cold start before reinforcement learning (RL). The method leverages both correct and incorrect rollouts to train the model to recognize and correct its own errors.

### Strengths
1) The use of MCTS to generate diverse and self-reflective training data is creative and effective. 
2) The proposed two-stage training (SFT followed by RL) addresses a common issue in RL-based agent training: instability in early stages.
3) MSEARCHER achieves state-of-the-art or competitive results on multiple datasets.

### Weaknesses
1) While the MCTS-based data construction is innovative, the paper lacks a formal analysis or theoretical grounding for why this approach should yield better reasoning trajectories. For instance, why is binary decomposition of sub-questions optimal? Why not allow n-ary splits or dynamic decomposition strategies? The design choices appear heuristic and would benefit from ablation studies or theoretical motivation.
2) The MCTS framework requires multiple rollouts, simulations, and tree expansions, which can be computationally expensive. The paper does not provide a detailed complexity analysis or discuss the scalability of this approach to larger datasets or more complex reasoning tasks. It is unclear how feasible this method would be for real-time applications or deployment in resource-constrained environments.
3) Although the paper evaluates on a range of QA datasets, most are still within the realm of factoid or multi-hop question answering. The evaluation lacks diversity in task types (e.g., commonsense reasoning, dialog-based reasoning, or multimodal QA). This limits the generalizability of the claims about MSEARCHER’s reasoning capabilities.
4) The agent’s performance is heavily dependent on the quality and availability of external search tools. The paper does not analyze the impact of search engine failures, biased retrieval, or noisy documents. In real-world settings, where search results may be unreliable or adversarial, the robustness of MSEARCHER is questionable and untested.

### Questions
1) Why is binary decomposition (splitting one question into exactly two sub-questions) enforced at every MCTS expansion node, and what evidence is provided that this restriction is optimal compared with n-ary or adaptive decomposition?
2) The reward function is relatively simple. Did you experiment with more sophisticated rewards, such as step-level correctness or a reward for synthesizing information across multiple turns?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
MSEARCHER: Self-Reflective Search Agent Empowered by Monte Carlo Tree Search-Based Data Synthesis proposes a two-stage training framework that combines supervised fine-tuning (SFT) with reinforcement learning (RL) to improve long-horizon, multi-hop reasoning for large language models (LLMs). The key innovation lies in a Monte Carlo Tree Search (MCTS)-based data construction process that generates self-reflective reasoning trajectories, leveraging both correct and flawed rollouts to produce high-quality synthetic training data. This approach enables stable RL training, better tool-use decision-making, and strong generalization across diverse QA benchmarks. Experiments demonstrate significant improvements over state-of-the-art search agents (e.g., DeepResearcher, ASearcher), achieving 67.6% on HotpotQA and 52.0% on Frames, confirming the importance of self-reflective data for enhancing reasoning robustness.

### Strengths
1.  Introduces an MCTS-based framework that synthesizes self-reflective reasoning data without requiring large reasoning models, improving data diversity and efficiency.
2. Combines SFT cold-start with RL fine-tuning, effectively stabilizing early-stage training and enhancing reasoning depth.
3. Outperforms multiple advanced baselines (DeepResearcher, Search-r1, ASearcher) on both in-domain and out-of-domain multi-hop QA benchmarks.
4. Provides a clear, modular design for integrating decomposition, retrieval, and self-reflection—offering practical reproducibility and strong generalization.

### Weaknesses
1. Although the paper proposes a reflective data construction framework, it lacks a theoretical analysis of the convergence and sampling efficiency of MCTS in high-dimensional reasoning spaces.
2. While the paper categorizes retrieval, reasoning, and decomposition errors, it does not systematically discuss how these error types accumulate during the reinforcement learning stage.
3. Despite claiming efficiency, the paper does not provide detailed comparisons of computational resources, time costs, or scalability, leaving the practical extensibility uncertain.
4. Although partial ablation studies are conducted, the paper does not sufficiently demonstrate the independent contributions of different reflective signal types (e.g., retrieval vs. reasoning errors).

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
2

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
This paper introduces MSEARCHER, a self-reflective search agent designed to address the instability and inefficiency of training large language models (LLMs) with RL for complex reasoning tasks. The authors propose a two-stage training pipeline that begins with a supervised fine-tuning "cold start" to provide the model with a stable foundation. The core innovation is a data construction framework based on Monte Carlo Tree Search , which decomposes complex questions into smaller sub-problems. This framework generates high-quality, self-reflective reasoning trajectories by leveraging both correct and flawed rollouts from the search tree, effectively teaching the model error-correction and robust reasoning. Following the SFT stage, the agent is further trained with RL to enhance its performance. The results demonstrate that MSEARCHER significantly outperforms previous methods on multi-hop question-answering benchmarks like HotpotQA and Frames, highlighting the effectiveness of using a high-quality SFT phase to stabilize RL training.

### Strengths
1、The proposed method of using Monte Carlo Tree Search (MCTS) to synthesize reasoning trajectories is highly effective, with the generation of self-reflective trajectories being a particularly novel and valuable contribution.

2、This paper thoroughly explores the two-stage Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) paradigm, effectively demonstrating its power in enhancing an agent's reasoning capabilities.

3、The experiments are comprehensive and the results are significant, showing that the proposed agent consistently outperforms strong baselines on multiple challenging benchmarks.

### Weaknesses
1、The paper primarily quantifies the method's effectiveness through experimental results but lacks a deeper analysis, such as the underlying reasons for the observed improvements.

2、Based on the experimental results, MSearcher does not appear to have a substantial advantage, especially when compared to ASearcher.

3、Table 4 seems to indicate a performance drop of 4.8 on HotpotQA with SFT. What accounts for this decrease? Did you train a 7B-version of MSearcher, and what were its performance metrics from the base model to SFT and then to RL?

4、What would be the effect if, instead of using a complex algorithm like MCTS for data construction, a simpler method such as Rejection Sampling (RFT) were employed for SFT data generation? The authors need to elaborate on their motivation for using MCTS.

### Questions
Stated in Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
