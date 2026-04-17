# Efficient Multi-turn RL for GUI Agents via Decoupled Training and Adaptive Data Curation

- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Vision-language model (VLM) based GUI agents show promise for automating complex desktop and mobile tasks, but face significant challenges in applying reinforcement learning (RL): (1) slow multi-turn interactions with GUI environments for policy rollout, and (2) insufficient high-quality agent-environment interactions for policy learning.
To address these challenges, we propose **DART**, a Decoupled Agentic RL Training framework for GUI agents, which coordinates heterogeneous modules in a highly decoupled manner. DART separates the training system into four asynchronous modules: environment cluster, rollout service, data manager, and trainer. This design enables non-blocking communication, asynchronous training, rollout-wise trajectory sampling, and per-worker model synchronization, significantly improving the system efficiency: *1.6× GPU utilization for rollout*, *1.9× training throughput*, and *5.5× environment utilization*.
To facilitate effective learning from abundant samples, we introduce an adaptive data curation scheme: (1) pre-collecting successful trajectories for challenging tasks to supplement sparse success in online sampling; (2) dynamically adjusting rollout numbers and trajectory lengths based on task difficulty; (3) training selectively on high-entropy steps to prioritize critical decisions; (4) stabilizing learning via truncated importance sampling for policy mismatch between policy rollout and updating.
On the OSWorld benchmark, DART-GUI-7B achieves a 42.13% task success rate, a 14.61% absolute gain over the base model, and 7.34% higher than open-source SOTA.
To ensure reproducibility, ***we will fully open-source our training framework, data, and model checkpoints via*** [https://dart-gui-iclr26.github.io](https://dart-gui-iclr26.github.io) (anonymous), which we believe is a timely contribution to the open-source community of agentic RL training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DART, a decoupled RL framework for GUI agents that splits training into four async parts (environment cluster, rollout service, data manager, trainer) to boost throughput and hardware use. It also adds adaptive data curation: pre-collecting successful trajectories for hard tasks, adjusting rollout count and max length by task difficulty, training on high-entropy steps, and stabilizing updates with truncated importance sampling. On OSWorld, the 7B model trained with DART reaches 42.13% success, a 14.61% absolute gain over the base and above prior open-source results.

### Strengths
1. The paper splits RL for GUI agents into four async parts so training and interaction never block each other, which is a clear design shift from coupled systems. This decoupling is novel for this setting and is backed by analysis on different parts.

2. The method adds adaptive data curation at several levels (task, trajectory, step, token), including high-entropy step focus and truncated importance sampling to stabilize updates. The system describes concrete components and pipelines (Trainer, Rollout Service, Env Cluster, Data Manager) with practical details. It also documents system prompts, rollout settings, hardware, and database tables that track trajectories and checkpoints.

3. The approach improves utilization and throughput (up to 1.6× GPU for rollout, 1.9× training throughput, 5.5× env utilization). It also lifts OSWorld success to 42.13%, a +14.61 point gain over the base and above prior open-source results. 

4. The authors plan to open-source the framework, model checkpoints, and curated data, which can speed up progress on GUI agents. This combined with detailed system reporting, makes the work quite useful.

### Weaknesses
1. Training and evaluation are both on OSWorld, and the training pool is a 203-task subset from that same benchmark. This raises the risk of benchmark-specific tuning / overfitting; adding at least one other GUI suite or an OOD test would better show robustness.

2. Results compare models under different step budgets, which can blur sample-efficiency claims. The system also relies on substantial infra (~180 parallel envs), so adding cost-normalized metrics would clarify practical efficiency.

3. Ablations would be better with stronger controls. The ablation studies probe DR/DTL/HE/DA, but only on 45 training tasks. Expanding these to the full split or the test set would be more convincing. Also report “coupled + curation” vs. “decoupled + curation” to isolate the framework effect on final OSWorld scores.

4. Pre-collecting successful trajectories for hard tasks may nudge behavior toward memorizing specific solutions. Please add controls where such pools are withheld, and report how performance changes. The method also hinges on truncated importance sampling and a fixed high-entropy threshold (top 80% of steps). Please include sensitivity sweeps over the clipping cap, entropy cut, etc.

### Questions
1. Since the training and evaluation both center on OSWorld, can you add any OOD or cross-suite tests to show generalization? Or is there any difficulty doing so?

2. In Figure 1(c), why does the training accuracy fluctuate so significantly? In this case, would selecting a checkpoint at a local maximum result in an unreliable estimate?

### Soundness
3

### Presentation
4

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
This paper addresses the bottlenecks hindering the application of reinforcement learning (RL) to Vision-Language Model (VLM) based GUI agents: the inefficiency of training systems and the lack of high-quality interaction data. The paper introduces DART, a Decoupled Agentic RL Training framework, which presents a comprehensive solution. Architecturally, DART deconstructs the RL loop into four asynchronous, non-blocking modules, a design that achieve 5.5x increase in environment utilization and a 1.9x boost in training throughput. 

Algorithmically, this efficient data generation engine selects the most informative experiences for learning. Effectively solve challenges like sparse rewards and imbalanced learning. The resulting model, DART-GUI-7B, achieves a 42.13% success rate on the OSWorld benchmark  by solving tasks in fewer steps than its competitors. The work's commitment to fully open-sourcing the framework, data, and model further solidifies its contribution to the field.

### Strengths
1. The strength of this work lies in its sophisticated systems engineering with algorithmic design. 
2. The DART framework is a engineering achievement. by decoupling the training process, solving the system-level bottleneck of slow environment interaction, a claim substantiated by the transformative 5.5x improvement in environment utilization. 
3. High-entropy-driven step selection and Experience Pool for difficult tasks demonstrate a understanding of the learning dynamics in GUI environments.

### Weaknesses
1. Although the author claims the SOTA performance on open source model on OS-world. If look at all ranking in os-world, the top8 performance already can achieve over 40% performance. The top1 UI-TARS-2-2509 already achieve 53.1% performance. Even though they are not open source, but I think more discussion and analysis between proposed model and them will help.
2. For experimence pool, the paper mentions "pre-populate the Experience Pool by collecting and storing high-quality successful trajecctories by preliminary sampling". It will be good to explain more on how trajectories are generated, which model are used when generated the trajectory. The discussion on the computation cost on preliminary sampling will be good. More details on each RL training step's successful trajectory rate will help.
3. For framework effiecient analysis, more details on baseline experiment setting will help. Current academic and industry also have lots of work on multiturn RL framework. More performance comparison between the related work like sky-rl will help. In figure 4, it shows the performance comparision on 4 GPUs and 80 Environment, larger scale experiments setting is needed to enhance paper's performance statement.

### Questions
Please check weaknesses.

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
This paper introduces DART, a reinforcement learning framework designed to train vision-language model (VLM) based GUI agents more efficiently. The framework addresses two main challenges: (1) slow multi-turn interactions with GUI environments during policy rollout, and (2) insufficient high-quality agent-environment interactions for policy learning. DART separates the training system into four asynchronous modules (environment cluster, rollout service, data manager, and trainer) and introduces adaptive data curation strategies at multiple granularities (task, trajectory, step, and token levels). The resulting model, DART-GUI-7B, achieves strong results w.r.t. the current SOTA on OSWorld.

### Strengths
1. Strong engineering contribution: The decoupled architecture with four asynchronous modules (trainer, data manager, env cluster, rollout service) is well-designed and addresses real bottlenecks in GUI agent RL training.

2. Comprehensive data curation scheme: The multi-level adaptive strategy (task, trajectory, step, token) is thoughtful and well-motivated. Each component addresses a specific challenge in GUI agent training.

3. Thorough experimental validation: Good ablation studies demonstrating the contribution of each component.

### Weaknesses
1. **Limited algorithmic novelty beyond system engineering**: The paper primarily uses standard step-wise GRPO with techniques borrowed from prior work (e.g., high-entropy step selection from Wang et al., 2025b). While the system integration of vLLM, veRL, and K8s is solid, the core contribution is engineering-focused rather than algorithmic innovation. Critically, DistRL (Wang et al., cited in the paper) should be discussed as a highly relevant baseline since it also decouples training from interaction in an asynchronous RL framework. The paper should clarify what specific advantages DART provides over DistRL beyond implementation details.

2. Insufficient treatment of reward hacking and generalization challenges: The paper does not adequately address the fundamental problem of reward hacking in GUI agents, which is currently the main bottleneck for generalization. The discussion of distribution shift is obscured by terminology (OOD tokens, quantization mismatch, GRPO variants) without addressing the core challenge: how does the agent generalize beyond trajectory-level reward gaming to learn robust GUI interaction policies? This is especially critical given that the method trains on only **203** tasks but claims generalization.

3. Inadequate failure mode analysis and limited mechanistic understanding: Section A.6 presents only two cherry-picked failure cases without systematic analysis. You may need to give some details, including:

- What percentage of errors are due to reasoning failures vs. grounding errors vs. action space limitations?
- Which of the proposed methods (dynamic rollout, entropy filtering, etc.) are designed to address which specific failure modes?
- How do these failure modes correlate with task complexity, trajectory length, or application type?

### Questions
- See point 3 above

- Also, could you give out some trading off when scaling up the actors or interactive agents and going towards highly async environments, what will happen, and how does your method prevent the risks of unstable training?

- How to measure the robustness and generalization abilities of your models since your training set only includes 203 tasks?

### Soundness
3

### Presentation
4

### Contribution
2
