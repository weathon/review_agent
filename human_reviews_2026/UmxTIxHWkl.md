# Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Unsupervised pre-training can equip reinforcement learning agents with prior knowledge and accelerate learning in downstream tasks. A promising direction, grounded in human development, investigates agents that learn by setting and pursuing their own goals. The core challenge lies in how to effectively generate, select, and learn from such goals. Our focus is on broad distributions of downstream tasks where solving every task zero-shot is infeasible. Such settings naturally arise when the target tasks lie outside of the pre-training distribution or when their identities are unknown to the agent. In this work, we (i) optimize for efficient multi-episode exploration and adaptation within a meta-learning framework, and (ii) guide the training curriculum with evolving estimates of the agent’s post-adaptation performance. We present ULEE, an unsupervised meta-learning method that combines an in-context learner with an adversarial goal-generation strategy that maintains training at the frontier of the agent’s capabilities. On XLand-MiniGrid benchmarks, ULEE pre-training yields improved exploration and adaptation abilities that generalize to novel objectives, environment dynamics, and map structures. The resulting policy attains improved zero-shot and few-shot performance, and provides a strong initialization for longer fine-tuning processes. It outperforms learning from scratch, DIAYN pre-training, and alternative curricula.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper tackles meta‑reinforcement learning by combining in‑context RL with a curriculum for self‑generated goals. An ICRL agent is pre‑trained to reach goals proposed by a curriculum pipeline. This pipeline first optimizes a goal‑search policy to estimate the agent’s current capability, then a selector chooses goals within calibrated success bounds. On XLand‑MiniGrid benchmarks, the proposed method, ULEE, improves exploration and adaptation, generalizing to novel objectives, dynamics, and map structures, and achieves better zero‑shot and few‑shot performance.

### Strengths
- *Intuitive formulation:* ULEE frames meta-RL as in-context adaptation guided by self-imposed goals, yielding a conceptually clean pipeline (goal search → capability estimation → goal selection). This makes the approach easy to reason about and implement.
- *Solid empirical evidence:* On XLand–MiniGrid, ULEE consistently improves zero-shot and few-shot performance over strong baselines (learning from scratch, DIAYN, alternative curricula), with generalization to new objectives, dynamics, and map structures.
-  *High-quality writing and presentation:* The paper is clearly written, with strong contextualization relative to prior work and a well-structured narrative that explains the curriculum components (goal search, capability estimation, selector) and their interplay, which aids reproducibility and understanding.

### Weaknesses
- Despite the intuitive design, the pipeline introduces multiple components and hyperparameters (e.g., difficulty/success bounds), raising concerns about training stability and sensitivity.
- With adversarial goal-generation, does the search ever propose “degenerate” goals (e.g., trivially satisfiable or reward-hacking)? Although the goal selector may mitigate this issue, randomness in sampling can lead to training collapse.

### Questions
- Why not integrate the selector’s criterion directly into the goal-search reward—for example, use a $|x−0.5|$-shaped reward over success probability or estimated difficulty to penalize goals that are too hard or too easy? What are the trade-offs?
- Could classical exploration designs like UCB (or Thompson sampling) be used to guide goal sampling—treating difficulty bands or goal families as arms—and does this improve sample efficiency or stability compared to the current selector?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ULEE (Unsupervised Learning of Efficient Exploration), an unsupervised meta-learning method designed to pre-train adaptive reinforcement learning policies. The core challenge it addresses is how to effectively generate and select self-imposed goals for an agent to learn from, particularly for broad distributions of downstream tasks. ULEE's main contribution is an automatic curriculum learning strategy guided by a novel post-adaptation task-difficulty metric. This approach optimizes for an agent's performance after a period of adaptation, rather than its immediate performance. The method combines an in-context learner with an adversarial goal-generation system that finds challenging yet achievable goals, effectively maintaining training at the "frontier of the agent's capabilities". On XLand-MiniGrid benchmarks, ULEE-pre-trained policies demonstrate improved exploration, adaptation, and generalization to novel tasks , resulting in better zero-shot and few-shot performance and providing a strong initialization for longer fine-tuning processes.

### Strengths
1. The post-adaptation task-difficulty metric, for me, is novel. It significantly departs from prior works in automatic curricula, which typically evaluate goal difficulty based on the agent's immediate performance. By defining difficulty as the agent's expected success rate after an adaptation budget, the method directly optimizes for the agent's capacity to learn rather than just its current knowledge. This aligns the pre-training objective more closely.

2. The paper originally combines three key components into a single system, ULEE. While concepts like meta-learning, adversarial goal generation, and in-context learners exist, ULEE integrates them synergically. It uses an adversarial "Goal-search Policy" to propose hard goals, a "Difficulty Predictor" network to estimate their post-adaptation difficulty, and an in-context learner (the "Pre-trained Policy") to meta-learn on a curriculum of goals selected for being at the "frontier of the agent's capabilities".

3. The authors evaluate the pre-trained policy across a wide spectrum of downstream scenarios.

### Weaknesses
1. The primary weakness of ULEE is its high methodological complexity. The system is not a single algorithm but a complex interplay of four distinct, learning-based components: the Pre-trained Policy ($\pi$), the Goal-search Policy ($\pi_{g, s}$), as well as the Difficulty Predictor. Are there practical bottlenecks (e.g., memory, wall-clock time) for this method?

2. The overall system's success depends on these components learning in lockstep. Can this co-adaptive process be brittle? E.g., will a relatively poor Difficulty Predictor lead to a collapse? Adding a robustness analysis to each design ingredient will greatly strengthen the paper.

3. The pre-training and evaluation tasks (4Rooms-Trivial, 4Rooms-Small, 6Rooms-Small) are all drawn from the same "family" of XLand-MiniGrid rules, which may not fully cover the claim “generalization to new goals, transition dynamics, and grid structures”.


Minors:
1. I recommend a figure to show the overall framework of the proposed, so that it can be more detailed and accessible.

2. There are some grammatical and typographical errors, for example. In line 65, “more tasks become too…”,

### Questions
1. There are many hyper-parameters in ULEE, learning rates, network architectures, buffer sizes, sampling bounds LB/UB, number of goal-search episodes etc. How critical are the specific hyper-parameter values?

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
The paper introduces Unsupervised Learning of Efficient Exploration (ULEE) as an unsupervised meta-learning approach that pre-trains a policy capable of rapid adaptation to new tasks. It achieves this by generating challenging-but-solvable goals based on the policy's estimated post-adaptation performance. The resulting policy is an in-context learner that adapts to new goals, dynamics and map structures using only its interaction history (observations, actions, rewards), requiring no explicit goal input. On the XLand-Minigrid benchmark, ULEE outperformed baselines in terms of fast adaptation, and provided a better initialization for both extended fine-tuning and supervised meta-learning. It also demonstrated generalization to novel environment structures. 

**Recommendation:**\
This paper falls outside my area of expertise, but appears to have a well motivated and interesting problem setting and strong empirical results. However, I have some questions about the methodology regarding seeding and validation/test sets. Therefore, in its current state, I will recommend to reject. However, I will be open to change my score if my questions are answered satisfactorily.

### Strengths
- Although this is not my area of expertise, the paper's motivation and positioning within existing literature appears strong. 
- The problem of pre-training for adaptation is very interesting.
- The empirical results are very strong.

### Weaknesses
- Section 4.3.1 does not sufficiently answer Q1. In this section the fraction of evaluation goals reached as a function of the number of evaluation episodes is shown in Figure 2. In my opinion this does not isolate exploration as the cause of evaluation goals reached, nor does it answer _"what exploration capabilities"_ the policy exhibits. For example, an increase in evaluation goals reached can also be due to zero-shot generalization, rather than improved exploration/adaptation. 
- Some parts of the experimental methodology is unclear. In particular when it comes to hyperparameter tuning and validation vs test split. 
- Certain details in the main text could be explained better.

### Questions
- Are there better ways to isolate and analyse the exploration capabilities of the Pre-trained Policy? For example, subtract all evaluation goals reached in a single episode and only include the ones reached with more episodes? Or visualize the exploratory behaviour of the policies in some way? 
- There appears to be no explicit mention of a test versus validation set. Are the final results evaluated on an independent testing set of environments (that has not been used for validation, hyperparameter tuning, or generally for algorithm design)? Similarly, were the seeds used for the final evaluation (testing) different from the ones used for validation, tuning and design? 
- I could not find mention of the hyperparameter tuning approach for your method and the baselines. How is it ensured that your approach did not accidentally benefit from an advantageous hyperparameter combination or tuning budget? Did you use separate seeds for tuning and final evaluation? 


**Things to improve that did not impact decision:**
- Some of the related work mentioned in Section 2 is missing an explicit comparison with the paper's approach.
- Line 227: The variable $n$ is introduced there but not defined or mentioned in the text.
- Line 294: I don't quite understand how $f_{counts}$ works. 
- Figure 3c: It is unclear to me what exactly Figure 3c is showing. Is it showing post-adaptation return on the evaluation set, evaluated at different points during pre-training?
- Table 1: The bold highlight is very difficult to differentiate from the regular numbers.

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
The paper proposes ULEE (Unsupervised Learning of Efficient Exploration), an unsupervised meta-RL pretraining framework that trains an in-context learner via an adversarial goal-generation strategy. Goal difficulty is defined by the post-adaptation success rate, yielding an intermediate-difficulty curriculum.

### Strengths
The paper is well-motivated. The curriculum is based on performance after in-context adaptation, not immediate performance, which aligns with the intended meta-RL setting. Empirically, ULEE improves exploration, shows faster few-shot adaptation, and provides stronger initializations for finetuning.

### Weaknesses
* The empirical impact of defining difficulty via post-adaptation performance, rather than immediate performance, remains unclear without an ablation. A direct ablation (e.g., a sensitivity study over $K$) would strengthen the paper.
* The baselines do not include recent meta RL and unsupervised RL methods.
* Experimental scope is limited to grid-world domains.

### Questions
* Does the learned difficulty correlate with intuitive task hardness? A qualitative or heuristic-based comparison between high and low-difficulty goals would be helpful.
* Does the goal-search policy reliably propose high-difficulty goals? What are the difficulty distributions of goals sampled by the goal-search policy (and a random policy)?
* What other environment information $\xi_M$ could be used, especially for environment domains other than grid-worlds?

### Soundness
3

### Presentation
3

### Contribution
3
