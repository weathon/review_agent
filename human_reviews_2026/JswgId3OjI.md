# ARGO: Asynchronous Rollout with Human Guidance for Research Agent Optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Large Language Model (LLM) agents have recently shown strong potential in domains such as automated coding, deep research, and graphical user interface manipulation. However, training them to succeed on **long-horizon, domain-specialized** tasks remains challenging. Current approaches either rely on dense human annotations through behavior cloning, which is prohibitively expensive for tasks that cost days/months, or on outcome-driven sampling, which often collapses due to the rarity of valid positive trajectories on long-horizon, domain-specialized tasks.
We introduce ARGO, a sampling framework that integrates **asynchronous human guidance with action-level data filtering**. Instead of requiring annotators to shadow every step, ARGO allows them to intervene only when the agent drifts from a promising trajectory, for example, by providing prior knowledge, or strategic advice. This lightweight, high-level oversight produces valuable trajectories at lower cost. ARGO then applies supervision control to filter out sub-optimal action, stabilizing optimization, and preventing error propagation. Together, these components enable reliable and effective data collection in long-horizon environments.
 To demonstrate the effectiveness of ARGO, we evaluate it using InnovatorBench. Our experiments show that when applied to train the GLM-4.5 model on InnovatorBench, ARGO achieves more than a 50\% improvement over the untrained baseline and a 28\% improvement over a variant trained without human interaction. These results highlight the critical role of human-in-the-loop sampling and the robustness of ARGO’s design in handling long-horizon, domain-specialized tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents ARGO (Asynchronous Rollout with Human Guidance for Research Agent Optimization), a framework aimed at improving long-horizon LLM agent training. ARGO allows human annotators to provide asynchronous, high-level interventions instead of continuous supervision and integrates an action-level supervision control module to mask suboptimal actions, reducing error propagation. Experiments on InnovatorBench demonstrate large improvements over baseline models, suggesting that asynchronous human-in-the-loop sampling can enhance agent learning efficiency and robustness.

### Strengths
1. The asynchronous human-guidance mechanism combined with action-level masking represents a creative and practical approach for improving long-horizon agent training.
2. The paper tackles a core problem in the development of autonomous LLM agents — reducing annotation cost while maintaining learning quality — which is highly relevant to the ICLR community.
3. The reported improvements on InnovatorBench and the supporting ablation studies demonstrate meaningful performance gains and validate the importance of human-guided rollouts.

### Weaknesses
1. The effectiveness of ARGO appears to depend heavily on the base model’s intrinsic capabilities, the task domain, and the expertise level of annotators. This introduces uncertainty about the method’s generalizability. Moreover, the framework lacks sufficient theoretical guidance or analysis explaining when and why asynchronous human guidance leads to stable learning outcomes across different settings.
2. Several key components — including the masking algorithm, the summarization operator, and the human feedback process — are insufficiently detailed. Without clearer algorithmic definitions and validation experiments, it is difficult to assess reproducibility or understand the exact mechanisms driving the reported gains.

### Questions
1. How sensitive is ARGO’s performance to the annotators’ expertise and intervention frequency?
2. Can the authors provide theoretical or empirical evidence on why asynchronous feedback stabilizes optimization?
3. How would ARGO perform on different task types (e.g., GUI automation or multi-modal reasoning)?
4. How robust is the method when base models vary in size or reasoning ability?

### Soundness
3

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
The paper introduces ARGO (Asynchronous Rollout with Human Guidance for Research Agent Optimization), a framework designed to train LLM-based research agents for long-horizon, domain-specialized tasks. ARGO enables asynchronous human guidance, where annotators provide high-level interventions only when the agent deviates from a promising trajectory, instead of continuous supervision. 

This process is combined with an action-level supervision control mechanism that filters out incorrect or unreliable actions before optimization. The system includes a Human–AI Interaction Interface that allows asynchronous annotation with low cognitive load, integrating visualization, state monitoring, and trajectory control. 

ARGO is evaluated on InnovatorBench, using GLM-4.5 as the base model. It achieves over 50% improvement compared to the untrained baseline and 28% improvement over the non-interactive variant. Ablation studies demonstrate that both asynchronous guidance and action-level masking are critical for robust gains, and test-time scaling experiments show that ARGO maintains performance improvements over extended training durations.

### Strengths
1.	New Framework Design – ARGO’s combination of asynchronous human oversight and step-level filtering offers a practical, scalable alternative to fully supervised or purely outcome-based training methods. The asynchronous annotation paradigm effectively reduces human cost while maintaining supervision quality.
2. Empirical Validation – Experiments on InnovatorBench show substantial performance gains over both open- and closed-source baselines, including Claude 4 and GPT-5, particularly in domains such as Data Collection, Data Filtering, and Loss Design.
3.	Clear Component Analysis – The ablation study clearly demonstrates performance degradation when removing either the human interaction component or the action-level masking mechanism, validating the necessity of each part.
4.	Scalability and Realism – The Human–AI Interaction Interface is designed for asynchronous updates and minimal cognitive load, making it feasible for long-term, complex research workflows.
5.	Interpretability and Transparency – The explicit action masking and trajectory visualization promote interpretability and enable humans to understand, verify, and improve model behavior efficiently.

### Weaknesses
1.	Lack of Theoretical Guarantees – Although conceptually inspired by MDP and ReAct paradigms, ARGO lacks a formal analysis of convergence or stability under asynchronous human guidance.
2.	Limited Evaluation Scope – Experiments are limited to InnovatorBench, without external or cross-domain benchmarks (e.g., GUI or embodied reasoning tasks), which restricts claims of generalization.
3.	Annotation Cost Analysis Missing – While the framework emphasizes reduced human cost, no quantitative comparison of human effort or annotation time is provided.
4.	Potential Evaluation Bias – Human-in-the-loop supervision may introduce domain-specific heuristics that unintentionally align with InnovatorBench’s evaluation metrics, creating potential bias.
5.	Insufficient Discussion of Failure Cases – The paper lacks qualitative examples of failure scenarios, such as incorrect masking decisions or ineffective human interventions.

### Questions
1.	How does ARGO handle inconsistent or noisy human feedback during asynchronous rollout? Is there a filtering or weighting mechanism to mitigate contradictory supervision?
2.	What is the measured reduction in annotation time or cost compared with dense, fully supervised baselines?
3.	Can ARGO generalize to multimodal or embodied agents, and how would asynchronous feedback be implemented in such environments?
4.	How is latency or synchronization delay between agent outputs and human responses addressed to prevent drift in long rollouts?
5.	Beyond symbolic masking, are there plans to quantitatively estimate action reliability using uncertainty or self-consistency signals?

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
4

### Summary
This paper introduces ARGO, a novel training framework for large language model (LLM) agents designed to tackle long-horizon, domain-specialized tasks such as automated research workflows. Traditional approaches like behavior cloning (requiring dense human annotations) and outcome-driven sampling (suffering from sparse rewards) are either too costly or unstable in such settings. ARGO addresses these issues by combining asynchronous human guidance with action-level supervision control, enabling efficient and stable training with minimal human effort. Its contributions are as follows:
1. An asynchronous human guidance mechanism that allows annotators to intervene only when the agent deviates from a promising trajectory, reducing annotation burden.
2. An action-level supervision control strategy that filters out low-quality or erroneous actions, improving training stability and preventing error propagation.
3. A human-AI interaction interface designed for low cognitive load and fine-grained control, making long-horizon annotation practical.
4. Comprehensive experiments on InnovatorBench, showing that ARGO improves over the untrained baseline by >50% and over a non-interactive variant by 28%.

### Strengths
S1: The paper identifies a real and important problem in training LLM agents for complex, long-horizon tasks.

S2: The combination of asynchronous guidance and action filtering is intuitive and technically sound.

S3: This paper includes ablation studies, test-time scaling analysis, and comparisons with both open-source and closed-source models.

S4: The human-AI interface is thoughtfully designed to support real-world annotation workflows.

S5: ARGO outperforms baselines across multiple task categories, especially in data collection, filtering, and loss design.

### Weaknesses
W1: While effective, the core ideas (human-in-the-loop guidance and action filtering) are not entirely new; the contribution is more of a careful engineering integration.

W2: Tasks are limited to AI research workflows; generalization to other domains (e.g., medicine, law) is not explored.

W3: The impact of annotator expertise or consistency is not studied, which could affect reproducibility and scalability.

W4: Although asynchronous, the system still relies on human judgment at critical points; unclear how it scales to more complex or frequent interventions.

W5: Missing comparisons with stronger baselines: No comparison with RL-based or multi-agent training methods, which limits understanding of ARGO’s relative advantage. Also, apart from using GLM-4.5 as the base model, how about the performance of other base models?

W6: There are some minor typos. For example, Line 112 is an action-level instead of a action-level;  Line 289 is slime code instead of smile code.

### Questions
Q1: How does the system handle inconsistent or suboptimal human feedback? Is there a mechanism to down-weight or correct such inputs?

Q2: How is "trajectory deviation" determined? Is it automated, or does it fully rely on human judgment?

Q3: Could the action filtering mechanism be too aggressive, potentially removing exploratory but useful actions?

Q4: Are there plans to test ARGO on non-research tasks or domains outside of AI development?

Q5: Could future versions replace human annotators with learned critics or other agents?

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
This paper proposes ARGO, an asynchronous human-integrated training framework, which includes asynchronous human guidance to construct better trajectory for agent training. After rolling out a trajectory with human guidance, an LLM will generate a summary trajectory based on the raw trajectory and the bad actions will be masked during training.

### Strengths
1. Integrating human guidance especially asynchronous guidance is an important yet underexplored topic.
2. The benchmark performance is competitive even compared to closed-source models

### Weaknesses
1. The methodology part is a bit unclear. Are the annotators given the whole history when they annotate? This also seems to impose a huge cognitive load. They may need to reload the information each time they annotate, especially if they have been away for a long time.
2. Lack of analysis to demonstrate why asynchronous human guidance is superior compared to other types of human guidance.
3. A fairer comparison is needed. The budget for human annotators should also be considered, and the budget spent by LLMs should match that of humans.
4. typos: 
- caption of Figure 2 "originla" -> "original"
-  line 289 smile -> slime
-  line 429 hugh -> huge

### Questions
1. How much effort do the annotators put into the experiments? I think a human baseline would be to measure it (e.g., time spent) and compare it to a post-hoc trajectory rewrite using the same amount of effort.
2. Is there any mechanism to reduce the cognitive load for human annotators? For example, can users/annotators chat with the LLM about the current status? It seems to be the case by looking at the screenshots but I'm not so sure.

### Soundness
3

### Presentation
3

### Contribution
3
