# AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Recent advances in large language models (LLMs) have sparked growing interest in building generalist agents that can learn through online interactions. However, applying reinforcement learning (RL) to train LLM agents in multi-turn, multi-task settings remains challenging due to lack of  scalable infrastructure and stable training algorithms. In this work, we present the AgentRL framework for scalable multi-turn, multi-task agentic RL training. On the infrastructure side, AgentRL features a fully-asynchronous generation-training pipeline for efficient multi-turn RL. To support heterogeneous environment development in multi-task RL, we design a unified function-call based API interface, containerized environment development, and a centralized controller. On the algorithm side, we propose cross-policy sampling to encourage model exploration in multi-turn settings and task advantage normalization to stabilize multi-task training. Experiments show that AgentRL, trained on open LLMs across five agentic tasks, significantly  outperforms GPT-5, Clause-Sonnet-4,  DeepSeek-R1, and other open-source LLM agents. Multi-task training with AgentRL matches the best results among all task-specific models. AgentRL is open-sourced at https://anonymous.4open.science/r/AgentRL-ICLR-C351 and has also been adopted for developing other open-source LLM agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents the AGENTRL framework for scalable multi-turn, multi-task agentic RL training. On the infrastructure side, AGENTRL
features a fully-asynchronous generation-training pipeline for efficient multi-turn RL. To support heterogeneous environment development in multi-task RL, authors design a unified function-call based API interface, containerized environment development, and a centralized controller. On the algorithm side, authors propose cross-policy sampling to encourage model exploration in multi-turn settings and task advantage normalization to stabilize multi-task training.

### Strengths
1, Develop an asynchronous, multi-task framework AGENTRL for scalable agentic RL training and robust heterogeneous environment deployment.
2,  Design a cross-policy sampling strategy to encourage exploration in multi-turn settings and task advantage normalization to stabilize multi-task RL training.
3, The experiments are comprehensive, including different models on different tasks.
4, The github seems good.

### Weaknesses
Please see questions for the details.

### Questions
So I am not very familiar with this topic. Could you maybe further explain why you choose Hephaestus and AgentLM as your baseline specifically and what's the difference between your proposed method and those two methods?

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
AGENTRL introduces a scalable, asynchronous multi-task RL framework for training LLM agents in multi-turn interactive environments. It combines cross-policy sampling to boost exploration and task-advantage normalization to stabilize training across heterogeneous tasks. Trained on five agent benchmarks, a single AGENTRL model matches specialist agents and outperforms GPT-5, Claude-Sonnet-4 and DeepSeek-R1, showing strong generalization to unseen tasks.

### Strengths
- The ablation studies are detailed and show the individual and combined effects of the two main algorithmic improvements.
- The introduction of cross-policy sampling to enhance exploration and task advantage normalization to address heterogeneity in multi-task RL are practical and well-motivated improvements.

### Weaknesses
- Mixing trajectories from different policy checkpoints introduces measurable off-policy bias, maybe more experiments and analysis to describe this bias is needed.
- Missing hyper-parameter sensitivity analysis: Kcross-policy sampling probability, advantage-normalisation batch size, async queue capacity are fixed to single values without grid-search or Sobol sweeps, so robustness of the reported gains is unclear.

### Questions
- Can the authors provide more granular per-task and per-instance breakdowns of reward scaling and failure modes to better demonstrate the stability of task advantage normalization?
- How sensitive is AGENTRL to choice of reward normalization and environment design? Is there any failure cases if task heterogeneity is heightened further?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents AGENTRL, a scalable framework for multi-turn and multi-task reinforcement learning of large language model agents. It introduces an asynchronous rollout–training pipeline, a unified function-call–based environment API, and novel algorithmic components such as cross-policy sampling and task advantage normalization, achieving state-of-the-art performance across five agentic benchmarks and demonstrating strong generalization to unseen tasks

### Strengths
- The asynchronous multi-turn training framework effectively eliminates the inefficiency of synchronous rollouts, reducing GPU idle time and improving overall throughput while maintaining stable policy updates.
  - The scalable multi-task environment infrastructure provides a unified function-call–based API and containerized deployment with centralized control, enabling efficient orchestration and management of heterogeneous environments at scale.
  - The algorithmic innovations, including cross-policy sampling and task advantage normalization, significantly enhance exploration and stabilize multi-task reinforcement learning, resulting in consistent performance gains across benchmarks.

### Weaknesses
- The proposed asynchronous generation-training architecture inevitably introduces a significant off-policy problem, where training data is generated by stale policies. However, for a framework based on an on-policy algorithm , AGENTRL fails to mention any mechanism to correct for the bias caused by this policy lag. This oversight of a core theoretical challenge leaves the stability and convergence of its training process theoretically unsubstantiated and makes the description of its asynchronous design ambiguous.
- The motivation and necessity of Cross-Policy Sampling are not sufficiently justified. This method, which samples actions from historical models, directly conflicts with the assumptions of on-policy algorithms and introduces uncorrected distribution shift risks. Furthermore, the paper links its motivation to solving "model collapse," yet provides no evidence that such collapse actually occurs. More critically, the experiments lack a comparison with simpler alternatives, such as standard experience replay, making it impossible to determine whether the complexity of this design is truly necessary or if its claimed benefits surpass those of established techniques
- Task Advantage Normalization (TAN) sounds complex but offers limited practical benefits. At its core, it simply performs z-score normalization independently for each task — a common trick that lacks higher-level insight.

### Questions
- Given the off-policy data introduced by asynchronous training, does your framework include any mechanisms (even implicit ones) to mitigate its negative impact on the on-policy algorithm (GRPO)?
- Regarding Cross-Policy Sampling, how do you theoretically justify that training directly on samples from older policies is stable and unbiased without applying off-policy corrections?
- Have you conducted experiments comparing Cross-Policy Sampling with standard experience replay? If not, what unique and irreplaceable advantages do you believe the former holds over the latter?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents AgentRL, a comprehensive framework for scalable multi-turn, multi-task reinforcement learning (RL) training of large language model (LLM) agents. The work addresses significant challenges in agentic RL by introducing: (1) a fully asynchronous generation-training pipeline for efficient multi-turn RL, (2) a unified function-call based API interface with containerized environment deployment and centralized control, (3) cross-policy sampling to enhance exploration in multi-turn settings, and (4) task advantage normalization to stabilize multi-task training.The authors evaluate AgentRL on five diverse agentic tasks (ALFWorld, DB, KG, OS, WebShop) across multiple model scales (3B-72B parameters). Results demonstrate that AgentRL-trained models significantly outperform strong baselines including GPT-5, Claude-Sonnet-4, and DeepSeek-R1, achieving state-of-the-art performance with an average success rate of 70.4%. Notably, a single multi-task trained model matches the best performance of five task-specific models while showing promising generalization to unseen tasks (BFCL-v3 benchmark).

### Strengths
Presents AgentRL, a comprehensive framework for scalable multi-turn, multi-task reinforcement learning (RL) training of large language model (LLM) agents, addressing infrastructure, environment, and algorithmic challenges simultaneously.

Achieves impressive performance gains across all tested tasks and outperforms strong proprietary baselines (GPT-5, Claude-Sonnet-4). 
The asynchronous pipeline achieves significant throughput improvements, making the approach practically viable at scale. 

Successfully demonstrates that a single multi-task trained model can match the best performance of five task-specific models while maintaining strong generalization capabilities.

### Weaknesses
Limited Theoretical Analysis: Cross-policy sampling's impact on policy distribution is mentioned but not formally analyzed

Evaluation Scope: Limited diversity in task types (no long-horizon embodied tasks, no code generation with execution, no real-world web automation); OOD evaluation limited to one benchmark (BFCL-v3)

Cross-Policy Sampling Analysis: "Stale engines" update schedule unclear (every "multiple steps" is vague); Limited analysis of the exploration-exploitation tradeoff; Section 4.3.1's inference experiments use different models (Llama vs Qwen) but training uses same model - this asymmetry needs justification

Scalability and Cost: No discussion of computational costs or training time

### Questions
1. How exactly is the "stale engine" update frequency determined? Is it task-dependent or fixed? What is the probability distribution for sampling from different policies at each step? Have you experimented with more than 2 policies? How does performance scale with the number of policies?

2.What is the maximum queue size for the data buffer mentioned in Section 3.1? How much off-policy bias is introduced? Can you provide a comparison with synchronous training in terms of training time, convergence speed, and performance ceiling? How do you ensure the trajectories remain "as up-to-date as possible" when the queue can accumulate data?

3. Why normalize at the token level rather than the trajectory level? Have you experimented with other normalization strategies？Does this introduce any bias when tasks have very different token-length distributions?

4.The BFCL-v3 results show modest improvements on single-turn tasks.  Can you provide more analysis on what makes the model generalize to new tasks? Have you tested on tasks completely different from the training distribution?

5. Section 4.3.1 uses heterogeneous models (Llama vs Qwen) for inference experiments, but training uses temporal versions of the same model family. Does this mean cross-policy sampling during training merely augments offline data diversity rather than introducing fundamentally different exploration mechanisms?

### Soundness
3

### Presentation
4

### Contribution
3
