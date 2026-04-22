# Learning with Interaction: Agentic Distillation for Large Language Model Reasoning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 2

## Abstract
Recent advancements in large language models (LLMs) have demonstrated remarkable reasoning abilities to solve complex tasks. However, these gains come with significant computational costs, limiting their practical deployment. A promising direction is to distill reasoning skills from larger teacher models into smaller, more efficient student models, yet existing data-centric distillation approaches suffer from passive learning, over-learning on simple tasks, and persistent knowledge gaps. To overcome these limitations, we introduce Agentic Distillation, a novel framework for adaptive and active distillation. In Agentic Distillation, student LLMs interact with teacher LLMs modeled as environments, receiving feedback tokens to guide their reasoning process and selectively updating their capabilities when necessary. To address the off-policy and gradient vanishing challenges introduced by feedback tokens, we devise a tailored importance sampling and clipping strategy within a unified objective that both incentivizes reasoning and injects knowledge into student LLMs. Extensive experiments show that Agentic Distillation significantly enhances reasoning performance while improving efficiency, offering a scalable path for equipping compact LLMs with advanced reasoning abilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Agent Distillation to address existing data-centric distillation, which over learn on easy samples. They propose letting a student model interact with a stronger teacher model, and ask for the teacher's feedback when the student is not able to solve the problem on their own. This feedback will be used to guide the student, and it shows that learning from teacher-generated feedback effectively improves distillation performance.

### Strengths
1. The detailed discussion of several issues when trying to inject teacher-generated tokens into the student LM is insightful (e.g., being off-policy and gradient vanishing), and the author provides solutions to address these issues.
2. The evaluation is conducted on extensive tasks, and the benchmarks are well chosen.
3. The improvement is consistent across tasks, showing strong performance from learning from the teacher’s feedback.

### Weaknesses
1. Some qualitative analysis will help, for example, show what the student is actually generating after training, and how it improves the performance. Does the student also generate feedback-style reasoning during test time?
2. Adding a baseline on SFT from the teacher’s full trajectory (including the feedback) and then doing RL for correctness would further strengthen the claim. How important is the interaction? Can we collect feedback in an offline manner?
3. The method figure is not clear; for example, it's hard to see that the teacher is generating multi-turn feedback. Also, how to decide when the student model needs help is not clear either.
4. "When to use external feedback" is controlled solely by prompting the student model. However, models can be over-confident or ill-calibrated, necessitating the need for an analysis on how often the student is over-confident (does not call the teacher model but cannot solve the problem on its own).
5. Some variants on when to use external feedback are also necessary to justify this design choice.

### Questions
1. Typo in Fig 1’s caption: gap instead of grap.
2. Text in Fig 3 is too small, especially for the equations.
3. There is an additional a’ in line 119.
4. The teacher feedback is used when any single rollout fail, or when all rollouts fail for a question? If it’s the former, how does this method avoid overlearning, since the student model is actually able to solve this question (but not always correct).
5. Using the objective from equation 16, does the student model also generate self-correct or feedback-style content during test time?
6. What if the SFT baseline is also distilling teacher’s feedback (and followed by RL training for correctness)? What would the performance be like
7. In abstract, the statement "...reasoning abilities to solve complex tasks, which has propelled the progress toward artificial general intelligence" is not necessary or should be backed by citations.
8. In the Knowledge Boundary Expansion analysis, which dataset is this, and what is the sample size? In my opinion, the more direct way to test this is to see how many previously unsolvable problems become solvable after training.

Missing reference on distilling from interaction: https://arxiv.org/abs/2402.01620

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces AgenticDistillation, a knowledge distillation (KD) approach. The authors motivate the idea with two problems in standard KD: overlearning (overfitting to simple questions) and knowledge gaps between the teacher and student. Their method prompts the student to actively seek teacher feedback on particular steps and allows models to learn from the feedback the teacher provides. The paper then introduces a gradient clipping method for performing RL on and SFT on different parts of the response (question and response). The method is tested on datasets from math, code, and science domains, with improvements over the base models and ablations that they compare against

### Strengths
- Framing distillation as a method where the agent asks for information from an oracle is an interesting idea
- RL approach for learning from feedback appears novel
- The authors tested a variety of models, including reasoning and non-reasoning models.

### Weaknesses
- Under-reported baselines: In Table 1, the baselines the authors report seem lower than what has been reported in published work. For example, the Qwen 2.5 tech report (https://arxiv.org/pdf/2409.12122v1, Table 5) has AIME 24 performance at 5/30 (16%) while this paper reports 9%. Past work (e.g. https://arxiv.org/abs/2506.11902, Table 1) has also reported higher baseline numbers for MATH-500 (76.5%, rather than 73.00 reported here). In several cases, the gain reported from distillation largely disappears when considering the stronger baseline numbers. Can the authors explain why their baselines are consistently lower than prior work?

- No external baselines: All baselines compared against are internal models, but no other competing distillation methods were evaluated (e.g. https://arxiv.org/abs/2503.07067, https://aclanthology.org/2025.acl-industry.4/, https://arxiv.org/abs/2509.25837) although several are cited in related work. 

- Potential data leakage: how sure are the authors that none of the datasets tested on are included in the training data sourced from DAPO, OpenScienceReasoning, and Reasoning Gym?

- It's not clear to me what happens at test time. During training, the model asks for information from the oracle -- is this prompt also followed at test time? If not, why is the model improving from training?

### Questions
# Comments
- Small quibble: I find the motivation w.r.t. AGI unnecessary, and a bit of a leap (i.e. the connection between distillation and AGI is a bit tenuous)
- many of the sentences are incomplete and the paper could use more polishing (e.g. L061)
- L324 typo in Instruct

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
The paper proposes Agentic Distillation as a novel paradigm for knowledge transfer, aiming to distill complex reasoning capabilities from large, computationally expensive teacher models into smaller student models. Unlike traditional static knowledge distillation, this approach introduces an interactive and agent-based learning environment. The authors motivate this work by pointing out that current data-centric distillation methods suffer from passive learning, over-fitting on simple examples, and persistent knowledge gaps. While conceptually innovative, the method relies on dynamic interaction, which introduces non-trivial overhead and stability risks that must be comprehensively addressed.

### Strengths
1. The core idea of shifting from passive data-centric distillation (e.g., logit-matching) to an active, interactive, agent-based learning framework is a major intellectual contribution. It offers a genuine new direction for solving the knowledge gap problem.

2. The agentic distillation framework provides a flexible structure that could potentially integrate advanced components, such as tool-use or external memory, making the overall distillation process more comprehensive and future-proof.

### Weaknesses
1. The method contradicts its primary goal of efficiency by introducing significant training-phase complexity. The paper explicitly notes that training time "may grow considerably" with teacher complexity. This dramatically limits the ability of the deep learning community to reproduce, scale, or even test this method without substantial, often inaccessible, compute resources.

2. The success of the distillation is likely critically dependent on the specific design of the "interaction" protocol, the reward signals, and the complexity of the agent architecture. If the results are highly sensitive to these hyper-parameters, the methodology is not broadly applicable or robust.

3. Traditional distillation offers a clear, convex optimization target (e.g., KL divergence). Introducing complex, nested optimization loops and dynamic feedback makes the objective function non-trivial, harder to analyze, and obscures which components (the distillation loss, the agentic feedback, or the interaction environment) are providing the primary performance gains.

### Questions
The paper notes a risk of "unstable improvements across tasks" due to the dynamic, interactive nature of the training. Given that a key strength is the potential for generalized reasoning, how do the authors explicitly decouple the emergent reasoning skills from the specifics of the training interaction protocol? Specifically, how is the learned policy guaranteed to be a generalized reasoning model, rather than an agent that has merely overfit to the teacher's interactive prompt-response and self-correction style within the training environment, leading to a catastrophic collapse in performance on static, zero-shot benchmarks where the interactive scaffold is absent?

### Soundness
2

### Presentation
2

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
The authors propose Agentic Distillation, a distillation method wherein a student LLM optionally queries a teacher LLM, in the process obtaining feedback which is then jointly optimized with its own generated tokens using GRPO. To stably learn from the teacher's feedback tokens (sampled from the teacher's policy), the authors introduce an importance sampling coefficient and a clipping strategy. Experiments are conducted on different reasoning and coding benchmarks with multiple student+teacher combinations to show that Agentic Distillation outperforms SFT on teacher trajectories and RL with its own trajectories (w/o any teacher interaction).

### Strengths
* In off-policy distillation, a student passively learns from teacher trajectories instead of learning from the feedback obtained on its own (student's) trajectories. Agentic Distillation presents a way of mitigating this issue.

* Experiments are pretty thorough with the main result being that actively learning from teacher's feedback tokens can be more beneficial than imitating teacher's trajectories.

### Weaknesses
* An important missing baseline is on-policy distillation. It is the most common and effective way of distillation which has been shown to outperform off-policy distillation. It is also compute-efficient because querying the teacher’s log probabilities requires just a single forward pass from the larger model, while the trajectories are generated by the smaller and cheaper student.

* The paper lacks examples and analysis of the kind of queries that the student generates for the teacher and the teacher's subsequent feedback. Without such analysis, it's hard to understand if the student only asks for hints or full answers? Additionally, what is stopping the teacher from giving out complete answers in which case, agentic distillation will turn into vanilla off-policy distillation. In summary, I'm unsure how the student learns to balance between always asking the teacher for complete solutions versus never interacting. Even though the authors write "This trend suggests that early in training, the student LLM queries the teacher LLM frequently to learn new knowledge.", this requires more analysis, examples, and explanation.

[1] On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. Agarwal et al., 2023

### Questions
* How does your method compare to on-policy distillation? 

* "This method employs a temperature coefficient to sharpen the teacher LLM’s distribution" -- Could you explain this more? How do you use the temperature?

* Did you try experimenting with a weaker teacher and a stronger student? 

* Since your teacher LLMs are thinking LLMs, does the student also learn from the think tokens or only the response tokens? 

* Can you share some examples and statistics of the queries generated by the student? Similarly, the teacher's feedback would also be interesting to look at.

### Soundness
2

### Presentation
2

### Contribution
2
