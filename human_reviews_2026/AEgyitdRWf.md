# ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
The emergence of large language model (LLM)-based agents has significantly advanced the development of autonomous machine learning (ML) engineering. However, the dominant prompt-based paradigm exhibits limitations: smaller models lack the capacity to learn from execution trajectories for generalization, while large proprietary models incur high computational overhead, restricting accessibility and scalability.
Focusing on this, for the first time, we explore the paradigm of learning-based agentic ML, where an LLM agent learns through interactive experimentation on ML tasks using online reinforcement learning (RL). To realize this, we propose a novel agentic ML training framework with three key components:
(1) exploration-enriched fine-tuning, which enables LLM agents to generate diverse actions for enhanced RL exploration;
(2) step-wise RL, which enables training on a single action step, accelerating experience collection and improving training efficiency;
(3) an agentic ML-specific reward module, which unifies varied ML feedback signals into consistent rewards for RL optimization. 
Leveraging this framework, we train ML-Agent, driven by a 7B-sized Qwen-2.5 LLM for autonomous ML.
Despite training on only 9 ML tasks, our 7B-sized ML-Agent achieves comparable performance to agents using much larger proprietary LLMs (e.g., GPT-5) but at significantly lower computational cost, demonstrating strong performance and cross-task generalization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors design a training framework that enables LLMs to learn from environment interactions efficiently, leveraging exploration-enriched fine-tuning, step-wise RL, and a ML-specific reward module. The authors train ML-Agent, a 7B-parameter model based on Qwen-2.5, which outperformed most baselines, and showed strong generalization result.

### Strengths
1. Modular framework design. The three proposed components (exploration-enriched fine-tuning, step-wise RL, and reward unification) are intuitive, complementary, and grounded in practical RL challenges for ML agents.
2. Strong empirical results. The 7B- model trained with this framework rivals GPT-5-driven agents, and shows great generalization.

### Weaknesses
1. Sparse ablation and analysis. The contribution exploration-enriched fine-tuning is not clearly isolated in ablation studies.
2. Lack of novelty. The three components of the proposed framework, exploration-enriched fine-tuning, step-wise RL, and the agentic ML-specific reward module, appear to be additive rather than integrated or co-designed.

### Questions
1. Could you report quantitative ablations isolating each component, e.g., without exploration-enriched fine-tuning, without step-wise RL, and without reward normalization?
2. Does exploration-enriched fine-tuning empirically increase coverage of the action space?

### Soundness
2

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
The paper introduces ML Agent, a method for fine-tuning LLMs using one-step RL, a lightweight alternative to traditional multi-step RL algorithms like PPO or DPO.
Instead of simulating full dialogue trajectories or optimizing cumulative rewards, ML Agent applies a single-step policy improvement based on feedback signals (e.g., preference or quality scores) for individual responses.
The paper shows comparable results of post-training on a small-sized model compared to proprietary LLMs such as GPT-5.

### Strengths
The new RL update method, if proven efficient and comparable or better to SOTA techniques, could be a good tool for LLM post-training when tool calls (such as training a ML model) are expensive.

### Weaknesses
- I am having a hard time seeing the novelty of this approach. The lack of comparison with a broader range of papers in the literature (Reinforcement Learning for Machine Learning Engineering Agents, MLE-Dojo, MLGym, AgentGym...) makes it hard to really see how much  is brought by this paper.
- I don't see a theoretical justification for the One-step RL method, and the empirical validation is itself a bit weak. I think the authors should show that this is a sound thing to do by showing:
  - Better sample efficiency vs PPO, or
  - Equal or better final alignment compared to a well-tuned PPO model.
The statistical rigor (significance etc) is also limited.

### Questions
- Could the authors clarify how their method relates to recent frameworks like Reinforcement Learning for Machine Learning Engineering Agents, MLE-Dojo, MLGym, or AgentGym? The literature on agents / tools and ML scientists has exploded these past months. In particular, does ML Agent address any limitations or gaps identified in those works?
- The paper presents one-step RL as novel, but similar formulations (reward-weighted log-likelihood updates) have been studied extensively. What, specifically, is new here, the algorithm, the training pipeline, something else?
- Does the one-step approach introduce bias relative to multi-step RL methods such as PPO?
- Why do the authors not include comparisons against PPO-based RLHF, or at least controlled reimplementations using similar datasets and reward models? One could use a simple problem with a small model if compute availability is an issue.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces RL to train ML Engineering agents to learn from past experiences. They address three problems in training an ML Agent: first, small ML agents lack exploration; they address this by distilling from a larger model. Second, to overcome the slow feedback loop of typical ML experiments, the authors introduce a step-wise reinforcement learning (RL) paradigm. Instead of learning from entire trajectories, they reformulate the problem to learn from single action steps sampled from the pre-collected expert states. Finally, this RL process is guided by a carefully designed agentic ML-specific reward module. Experiments show comparable performance of a trained 7B model to a prompted 600B model. Although the experimental results are promising, the paper lacks discussion on highly probable reward hacking (due to reward design) and distribution shift problems (due to step-wise RL).

Overall, the paper introduces a novel learning-based paradigm that enables smaller models to achieve strong performance, addresses practical challenges. However, the approach has fundamental limitations in true exploration due to dependence on expert-generated state distributions, limited task diversity in training, incomplete cost analysis that excludes training overhead, and potential generalization concerns to tasks outside the benchmark distribution.

### Strengths
The paper's core strength is developing an autonomous ML agent that learns from experience rather than a prompt-engineered heuristic. The proposed step‑wise RL objective makes online training easier in ML settings with slow experiments. Empirical results show competitive performance of a 7B backbone model compared to Larger 600B size backbone models without training.
The paper systematically validates each component's contribution, showing that all three technical components are necessary. Despite training on only 9 tasks, the model shows meaningful performance gains on 10 held-out tasks.

### Weaknesses
1. Different execution times: The paper, although it proposes RL fine-tuning for MLE agents, does not consider a practical problem in training agents specifically for MLE tasks. Different actions from the same state could have different execution times, and if optimized in a vanilla fashion, would lead to more time-consuming yet optimal solutions being explored less. Consider that given a neural network, the agent adds extra layers in one action, opposed to changing the learning rate in a parallel action (people generally perform parallel rollouts for efficiency), then the second type of action would lead to faster rewards and hence be more frequent during training. 
2. Reward function is coarse: Mapping all the error cases to -1 and all the valid or corner edits to 0 would treat syntax issues, dependency issues alike, and all the types of out-of-memory issues (model loading OOM, memory leak OOM, and large batch size OOM) the same, although they convey different information and levels of mistakes and understanding. Moreover, the reward considers only the task accuracy and ignores compute, memory, cost, etc., important parameters. Although the reward is simple, it is prone to reward hacking. Another instance of reward hacking could be that the agent learns to find good seeds to make the performance increase, rather than actually learning to improve the architecture or training algorithm. 
3. Trajectory Collection is offline with respect to the training policy: The paper states that they use the collected trajectories and at each step, ask the smaller agent to propose an action on a state sampled from the larger agent. This might lead to the problems of distribution shift when using the smaller agent to generate the trajectories from scratch. This might also bias the learning towards expert behavior and limit exploration. This is more akin to a mix of off-policy behavior cloning and on-policy action generation. The agent cannot learn to recover from or explore states outside the expert trajectory distribution, which severely constrains the "learning through experimentation" claim. This raises the question, how would this approach extend to domains where high-quality expert trajectories aren't available?
4. Incomplete Cost Analysis: Figure 2 only compares the inference costs per trajectory but excludes the substantial training and inference costs required for data collection using GPT-4o-mini, supervised fine-tuning, and RL training. For a fair comparison, these costs should be amortized over expected usage.
5. Narrow Task Distribution and Generalization: Training on mainly regression and classification, and then testing on similar tasks raises the question of true cross-task generalization (except one generation task in testing). The paper lacks evidence of generalization to: (a) fundamentally different ML tasks, e.g., if trained on supervised learning, can it handle RL tasks?, (b) different data modalities not seen in training.
6. Evaluation Metric Limitations: The Performance gain delta metric depends heavily on the initial script quality, which may vary across tasks. A task with a poor initial script will show larger gains for the same absolute improvement. The paper would benefit from also reporting: (a) absolute performance metrics, (b) comparison against human expert solutions, and (c) success rate in reaching specific performance thresholds.
7. Script editing using a different model: Note that the training script is not edited by the model being trained. The actual editing is done by a different model (according to Table 4 and prompts shown), yet there is no discussion on which model or agent scaffold was used for this action. Moreover, if the policy being trained is not making edits, why is it penalized for errors induced by the editing model?

### Questions
1. How do you prevent the agent from 'reward hacking' by discovering shortcuts, such as finding optimal random seeds, rather than learning generalizable ML engineering improvements?
2. Since the trajectories are collected offline, how does this affect the test time performance of the model when it has to generate an action on its own trajectories that it has never seen during training time? What is the performance of your model when starting from an incorrect or suboptimal state compared to a pre-trained prompt-based method?
3. How do you account for more complex solutions that are less explored during RL due to the high time of execution, leading to fewer occurrences?
4. Which model is used to edit the script?
5. What is the average trajectory length at testing time? What is the average execution time of the solution generated by the trained model?
6. What is the total computational cost, including expert trajectory collection, fine-tuning, and RL training? How many trajectories would need to be run at inference time to amortize this cost compared to a strong prompt-based method directly?
7. How sensitive is the approach to expert trajectory quality? What happens if you use trajectories from a weaker model?
8. Can you provide evidence of generalization to qualitatively different ML tasks? For example, if trained on supervised learning tasks, can it handle reinforcement learning or unsupervised learning tasks?
9. How do you select the baseline script for each task? The reward and evaluation both depend on this selection.

### Soundness
2

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
3

### Summary
This paper proposes an online reinforcement learning agent training framework applied to agentic machine learning tasks. The proposed training framework involves three stages: exploration-enriched fine-tuning, step-wise RL, and rewards that are specific to agentic ML tasks. Experimentally, the authors show that with limited training, their 7B agent can match the performance of frontier models such as GPT-5.

### Strengths
- This paper provides strong evidence that RL can be used successfully to improve smaller (7B) models to the point of enabling complex agentic behavior on ML-specific tasks, which I view as a strong contribution. 
- The proposed technical solutions are quite reasonable. In particular, the authors note that ML agents can suffer from a lack of exploration, and provide an SFT technique that specifically targets this problem. The proposed exploration-enriched fine-tuning technique is both novel and well-executed. 
- The results on the tasks that were evaluated suggest that the method works and is competitive with frontier models. This is a strength, however, I have discussed other concerns about the evaluation details in the weaknesses section.

### Weaknesses
- The authors train on only 9 tasks from subsets of MLAgentBench and MLE-Bench. How were these tasks chosen? Details about the rationale behind the training task selection seem missing. 
- During evaluation, the authors evaluated on a small number of held-out tasks from MLE-Bench. It is unclear how this evaluation subset was selected from the larger set of tasks that were not used during training, even within MLE-Bench. A more convincing evaluation might involve evaluating on entire held-out benchmarks. 
- Training requires generating trajectories using GPT-4o-mini (or some other expert model) which might fundamentally limit the complexity of tasks on which it can be used, to tasks that are already solvable by frontier models.

### Questions
- Related to the first weakness point, can the authors provide the specific criteria used to select the 9 training tasks? 
- MLE-bench contains significantly more than 10 tasks. How were the 10 held-out evaluation tasks selected from the full set of unused tasks?

### Soundness
2

### Presentation
3

### Contribution
2
