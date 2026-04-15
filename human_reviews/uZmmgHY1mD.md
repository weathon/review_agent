# The Perfect Blend: Redefining RLHF with Mixture of Judges

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 6

## Abstract
Reinforcement learning from human feedback (RLHF) has become the leading approach for fine-tuning large language models (LLM). However, RLHF has limitations in multi-task learning (MTL) due to challenges of reward hacking and extreme multi-objective optimization (i.e., trade-off of multiple and/or sometimes conflicting objectives). Applying RLHF for MTL currently requires careful tuning of the weights for reward model and data combinations. This is often done via human intuition and does not generalize. In this work, we introduce a novel post-training paradigm which we called Constrained Generative Policy Optimization (CGPO). The core of CGPO is Mixture of Judges (MoJ) with cost-efficient constrained policy optimization with stratification, which can identify the perfect blend in RLHF in a principled manner. It shows strong empirical results with theoretical guarantees, does not require extensive hyper-parameter tuning, and is plug-and-play in common post-training pipelines. Together, this can detect and mitigate reward hacking behaviors while reaching a pareto-optimal point across an extremely large number of objectives. 

Our results show that CGPO consistently outperforms other commonly used SoTA RLHF algorithms (such as PPO and DPO) on a wide range of tasks -- general chat, STEM questions, instruction following, math, coding and knowledge. In particular, CGPO improves over PPO by 7.4% in AlpacaEval-2 (general chat), 12.5% in Arena-Hard (STEM\reasoning), 2% in IFEval (Instrcution Following), 2% in both MATH and GSM8K (Math\reasoning), 5% in HumanEval (Coding), and 2% in the ARC challenge (Knowledge). We also observe that PPO is susceptible to severe reward hacking behaviors (it exhibits severe regression in popular coding benchmarks) which can be addressed by CGPO. CGPO represents a breakthrough in RLHF, simultaneously addressing reward-hacking and extreme multi-objective optimization, and thereby advancing the state-of-the-art in aligning general-purpose LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduced a simple yet effective approach of multi-objective alignment using constrained optimization.
The method requires some engineering efforts in designing the judges.
The proposed framework is general, and is applicable to PPO / DPO / RAFT.
The authors conducted experiments with all those implementation variants on a variety of tasks.

### Strengths
The authors use a variety of tasks to demonstrate their method. They also implemented several variants of the CGPO framework.
The idea is simple, results look promising.

### Weaknesses
Some parts of the paper are not easy to follow. In the paragraph begining at line 77, I would recommend the authors to use a Figure 1 to explain their technical contribution rather than the illustrative MoJ, which is very clear - yet the primary-dual constrained optimization part could be further explained with a "figure 1"

Assuming prior knowledge to reward models / how could the hacks be is too strong and lacks a ground. 
The paper claims on improving pareto frontier at the intro section but there is no supporting evidence in the experiment section: missing comparison to baseline methods. e.g., [Reward Soups], the authors claimed problems/challenges with this type of methods, yet there is no empirical nor theoretical justification.

Lack of technical novelty. The idea is simple and seems effective, yet applying constrained RL in (multi-objective) alignment has been explored in e.g., [Safe RLHF: Safe Reinforcement Learning from Human Feedback]. This highly-relevant paper is not cited and not compared with.

The introduction of some terms is not necessary, e.g., why would the authors mention the primal-dual forms of constrained optimization? Simplicity could be an advantage if there is no need to make it more complex.

### Questions
On the discussion of Vulnerability to Reward Hacking: the claim on the amplification of reward hacking in multi-task learning is not convincing. e.g., the helpfulness can be hacked by generation length, yet the same hack does not work for harmless. Can the authors further justify their claim either theoretically or empirically?

How good is the proposed method when comparing with other multi-objective alignment baselines? 

In a multi-task setting, how do the idea differentiates between "objectives" and "constraints"? e.g., in a joint optimization for harmless and helpfulness, which one should be the objective? Will the implementation difficulty of rule-based judge be considered in such a process?

Practically, it is likely that not all of the constraints can be simultaneouly satisfied, how the method tackle this challenge? Will the learning / sample efficiency be extremely low in such cases?

In the experiment section, confidence intervals are only reported in some of the results, is there a specific reason?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a paradigm to align LLMs in constrained multi-objective settings. It first introduces the notion of ``mixture of judges’’, which utilizes rules or LLMs to check if a given response satisfies a set of constraints. Then, this work utilizes CGDO to optimize the reward while complying to the constraints. To extend the algorithm to the multi-objective setting, this work proposes CRPG, CRRAFT, CODPO that utilize multiple reward models and integrate the gradient updates.

### Strengths
- The manuscript is well-written; and the proposed algorithm is straightforward to understand.
- This work considers the constrained multi-objective alignment setting, which is novel and important. Also, compared to previous works which usually specify probabilistic constraints, this work specifies a stricter constraint, i.e., $P_{s \sim \mathcal{D}, a\sim\pi_{w}}((s,a) \in \Sigma) \geq 1$.
- According to Table 2, the proposed algorithm(s) outperform their DPO, PPO baselines.

### Weaknesses
- While the work studies a constrained optimization problem, the constrain satisfaction in Equation (3) is not verified. 
- In line 109, this work claims that the proposed method could avoid compromises due to conflicting goals from other tasks; however, in Algorithm 2, in the parameter updating step (step 6), conflicting goals might induce conflicting gradients ($\tilde g_l(\pi_{w_{t}})$), which stills leads to compromises.
- Since the MoJ is involved in the training process, the cost of LLM calling could be high. (As the practice in several previous works like [1], it is possible to model the model the constrains using collected safety data. This avoids calling LLMs during the training process. )

[1] Dai, etc. Safe RLHF: Safe Reinforcement Learning from Human Feedback.

### Questions
- What are the LLM models used as the LLM judges?
- What are FRR and SVR in Table 2?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel training paradigm called Constrained Generative Policy Optimization (CGPO), to tackle the challenge of multi-objective optimization and multi-constraints in RLHF. Under this new training paradigm there are several critical building blocks; the authors developed constrained RLHF optimizers, judges for handling the constraints, a new strategy of performing RLHF with the consideration of different types of data, reward models, and judges. The authors instantiated the proposed framework on Lllama3.0 and reported highly effective results compared to the baselines.

### Strengths
1. The authors proposed a new training paradigm for RLHF which handles multi-objective and multi-constraints, which can address several limitations of existing paradigms. The proposed paradigm is highly modular and has the potential to scale to more complicated future scenarios.
2. The experimental design is thoughtful, spanning a comprehensive set of tasks and constraints. The authors performed extensive evaluations and demonstrated the effectiveness of CGPO.

### Weaknesses
1. I appreciate the thoughtful design of the multi-objective and the multi-constraints, as well as the customizability of the framework. But there are a few natural questions following these:

    a) For all these "customized combinations" (line 292), "tailor the specific reward model to be applied for each task" (line 336), "uniquely tailored for each task" (line 347), "specifically tailored hyperparameter setup" (line 353) -- it's unclear how these design choices are made, and if one can reproduce these procedures, and how to use this framework for future needs. For the purpose of not only reproducibility but also the extensibility to new data/needs that may arise in the future, it's critical to: 1) document the procedures of making each specific design choice; 2) provide general guidelines that one can follow to develop these components; 3) discuss potential ways of automating some design choices.

    b) Given the complexity of the framework, it would be nice to 1) understand the contribution/sensitivity of each sub-component (with some ablation), 2) understand the complexity overhead. 

2. In terms of addressing the shortcoming of using a linear combination of reward models (a.k.a. linear scalarization) on all data, the authors proposed an approach that selectively applies compatible reward models to selective sets of data, which however still falls within the regime of linear scalarization types of methods. The literature [1] has pointed out that linear scalarization is incapable of fully exploring the Pareto frontier, while specialized multi-task optimizers, e.g. MGDA, can find more balanced solutions. It's worth considering the feasibility of applying MGDA and discussing the trade-off in complexity and performance.

**Minors**

- Typos: Line 181: optimizaiton; sec 3.2 header Multi-Taks.


**References**

[1] Hu, Yuzheng, et al. "Revisiting scalarization in multi-task learning: A theoretical perspective." *Advances in Neural Information Processing Systems* 36 (2024).

### Questions
See "weaknesses" section; would appreciate clarifications/discussions from the authors.

### Soundness
3

### Presentation
3

### Contribution
3
