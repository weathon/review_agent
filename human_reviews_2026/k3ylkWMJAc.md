# Self-Aware Reinforcement Learning for Improving LLMs with Minimal Data

- Decision: Reject
- Scores: 4, 2, 6, 2, 4

## Abstract
Reinforcement learning (RL) has demonstrated potential in enhancing the reasoning capabilities of large language models (LLMs), but such training typically demands substantial efforts in creating and annotating data. In this work, we explore improving LLMs through RL with minimal data. Our approach alternates between the LLM proposing a task and then attempting to solve it. To minimize data dependency, we introduce two novel mechanisms grounded in self-awareness: (1) self-aware difficulty prediction, where the model learns to assess task difficulty relative to its own abilities and prioritize challenging yet solvable tasks, and (2) self-aware limit breaking, where the model recognizes when a task is beyond its capability boundary and proactively requests external data to break through that limit. Extensive experiments on nine benchmarks showing a 53.8% relative improvement with less than 1.2% extra data demonstrate the efficacy of self-aware RL and underscore the promise of self-evolving agent training. Our code is available at https://anonymous.4open.science/r/SARL-B7FE/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces self-aware RL, a paradigm for improving large language models through reinforcement learning with reduced data requirements. The approach features two mechanisms: (1) self-aware difficulty prediction, where the model learns to estimate task difficulty relative to its current capabilities, and (2) self-aware limit breaking, where the model occasionally requests external guidance for high-utility unsolvable tasks. The authors train Qwen2.5-Coder-3B on code generation tasks and evaluate across various benchmarks in mathematical reasoning and code generation, reporting improvements on math benchmarks.

### Strengths
- Addresses a problem of reducing data dependency in LLM training.
- The paper is well written and easy to follow.

### Weaknesses
Please see my detailed questions and concerns below.

### Questions
- What prevents reward hacking? The generator is rewarded for accurate difficulty prediction, but couldn't it hack this by generating only tasks of a narrow difficulty range where its predictions are accurate?
- How sensitive is the method to the target difficulty level mentioned in the prompt (lines 633)? You explicitly tell the model to aim for difficulty 4-5, isn't this just supervised prompting rather than "self-awareness"?
- Why use z-score normalization across the buffer? This makes the selection relative to recent tasks, but what if the buffer contains mostly easy or mostly hard tasks?
- Why measure novelty using perplexity from the old policy $\pi_{\theta_{old}}$ rather than the current policy? Shouldn't novelty be relative to what the current model knows?
- How do you ensure the generated tasks are valid, well-formed, and non-trivial? What quality control mechanisms exist beyond format checking?
- How do you handle the distributional shift as the model improves? Early-generated tasks may become trivial. Do you discard old data or retrain on everything?

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
3

### Summary
The paper introduces a self-aware reinforcement learning framework that allows an LLM to generate and solve its own tasks while using two mechanisms: 1) self-aware difficulty prediction, where the generator predicts task difficulty relative to the solver's ability; 2) self-aware limit breaking, where the model detects when it has hit the capability ceiling and requests external data for targeted improvement. Based on these two mechanisms, the paper designs reward functions and implement on REINFORCE++ algorithm. The paper evaluates the proposed framework with Qwen-Coder-3B model across 9 reasoning benchmarks.

### Strengths
1. The paper is well-motivated and addresses an important problem of reinforcement learning for LLMs.
2. The designs for difficulty prediction and adaptive external querying are reasonable and conceptually sound.

### Weaknesses
1. It is unclear from Equation (10) how the solver and generator are updated; the equation should explicitly write out the terms for both the solver and the generator to clarify their respective objectives.
2. The reward design for the generator seems too simple. Will the generator produce overly easy or overly difficult questions to maximize the difficulty prediction reward?
3. The experiments are confined to a single model and single run. The reported improvements on code benchmarks, despite training on code data, are marginal. It is hard to determine whether the method is consistently or statistically effective.
4. Lack of justification of choice of $\gamma$.
5. The paper writing should be improved, many typo and format errors are presenting, such as at line 273 and line 477.

### Questions
1. What is the rationale for multiplying $R_{format}$ in equation (8) and equation (9)?
2. In Figure 2, why does the training reward become negative, given that the defined rewards are non-negative? Additionally, it would be helpful to show separate curves for the format reward, outcome reward and difficulty prediction reward.

### Soundness
2

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
4

### Summary
The paper introduces a novel paradigm called Self-Aware RL, which allows the LLM to learn and improve with minimal external data by creating a self-task generation-solver design. To make sure the self-generated tasks are valid, this paper proposes two strategies to strengthen the usefulness of the generated samples. First, Self-Aware Difficulty Prediction: The LLM not only generates its own training tasks but also predicts the difficulty of that task relative to its current ability. It then learns to align this difficulty estimate with its actual success rate, enabling an adaptive curriculum that prioritizes tasks that are appropriately challenging, yet solvable ("just right"). Second, Self-Aware Limit Breaking: The model identifies its own capability limits and proactively seeks external guidance only when necessary to surpass its inherent capability limits.

### Strengths
1. This paper counts as a pioneer work on "zero-data" self-improving language models, directly comparing to Absolute Zero and showing decent performance improvement.

2. They validate their approach on a good amount of downstream tasks spanning from math to coding, showing steady improvement and good generalization.

3. They have a good cognitive science motivation behind by explicitly connecting LLM self-improvement to the concept of self-awareness provides a novel theoretical framing for future research into self-evolving agents.

### Weaknesses
- The entire system's success depends on the LLM first being able to accurately gauge its own difficulty, and we all know LLMs tend to overestimate their confidence for task difficulties. A poor start in self-awareness could easily cause the RL process to settle on an unproductive local minimum. There is currently no mechanism to break this local minimum loop.

- For the second component design, Self-Aware Limit Breaking, external knowledge guidance is used, which is basically a larger-sized LLM. However, this paper completely ignores a straightforward baseline where directly distilling a very small amount of data from this teacher model could yield competitive performance gain.

- The paper is missing experimental comparison with some closely related literature.

Wang et al. Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning.Zhou 
Zhou et all. Self-Challenging Language Model Agent


- Some typos and grammar should be better polished (for example, line 223 "self-aware self-aware RL") and references need to be made more professional.

### Questions
Please see weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a RL framework to improve LLM reasoning with minimal data. In this self-improving framework for LLM reasoning where generator agent generates the tasks and predicts success rate and solver agent tries to solve the task and selectively queries a stronger external solver when valuable unsolved tasks are identified.
Two mechanisms are introduced:
1. self-aware difficulty prediction to align task generation with model capability
2. self-aware limit breaking where agent request external guidance for high-utility failures tasks.

### Strengths
1. Practical and relevant direction toward data-efficient RL for reasoning models
2. Strong empirical improvements on multiple math reasoning datasets. Evaluated on wide number of benchmarks.

### Weaknesses
1. “self-aware” term is conceptually overstated. success-rate prediction alone does not fully justify the terminology.  the framework does not model its internal reasoning or belief states. A clearer positioning as capability-aware curriculum RL would improve conceptual accuracy.
2. In limit-breaking claim, this paper does not introduce a new learning paradigm or theoretical formulation, it simply adds gating logic to decide when to query an external model. Any performance gains depend on the external model’s capability.
3. Evaluation only compares against the base model and AZR, omitting other strong curriculum RL baselines.
4. this framework only tested on qwen-3B model, why not other model families? Have the authors tested different scales or model families? How would the approach transfer to other model families (e.g., Llama, DeepSeek)?

### Questions
Please refer to weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses data‐efficiency in RL‐fine­tuning of large language models (LLMs). The authors propose a “self‑aware RL” framework in which an LLM alternately (i) generates tasks and predicts their difficulty (self‑aware difficulty prediction), and (ii) attempts to solve them, with a mechanism to detect when tasks are beyond current capabilities and ask for external guidance (self‑aware limit breaking). The idea is to minimize reliance on large curated datasets by letting the model generate its own curriculum and seek external data only when necessary.

### Strengths
* The problem is timely: reducing dependence on curated data for reasoning-capable LLM fine-tuning is important.

* The idea of alternating roles (task generation / solving) is conceptually appealing and aligns with recent trends in self-supervision and synthetic data generation.

* The empirical results show non-trivial gains with relatively little extra data, which is promising for data-efficiency.

### Weaknesses
* The proposed framework is essentially a form of adversarial or dual-role training (generator and solver), yet the paper lacks analytical treatment of convergence, stability, or whether the iterative procedure reliably improves rather than degenerate (e.g., the generator produces trivial tasks, the solver overfits, task difficulty drifts). There is no deeper theoretical justification or monitoring of regime stability.
* My own experience (and known community experience) is that even SOTA LLMs generating tasks often yield unstable data (invalid tasks, unsolvable ones, or trivial ones). The paper does not sufficiently discuss how data quality is controlled, how “ground truth” solutions are validated (especially when the same model solves its own generated tasks), or how noisy/invalid samples are filtered. This raises questions about how much of the reported improvement is due to good synthetic data vs incidental or artifact effects.
* The “multiple model roles” (generator + solver + possibly verifier) increases complexity. The paper does not sufficiently discuss the scaling behaviour (model sizes, compute budget, how many rounds, etc.). It is unclear whether this method is practical for much more parameter models.

### Questions
* Could you include training curves (e.g., generator loss, solver loss, number of generated tasks, accuracy of solver over time) to illustrate training dynamics and stability?

* Is the method truly free of external data (i.e., “no external data” apart from initial model and minimal seeds)? If so, how is grounding / calibration ensured? If not, how much external data is needed and what is its role?

* Please report compute / training cost: number of model parameters used, number of rounds of generator/solver, GPU hours, wall-clock time. What is the overhead compared to standard fine-tuning?

### Soundness
2

### Presentation
3

### Contribution
2
