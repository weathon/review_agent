# EXPLOITING TREE STRUCTURE FOR CREDIT ASSIGNMENT IN RL TRAINING OF LLMS

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 2, 2

## Abstract
Reinforcement learning improves LLM reasoning, yet sparse delayed reward over long sequences makes token-level credit assignment the key bottleneck. We study the verifiable-reward setting, where the final answer is checkable and multiple responses can be drawn per prompt. Reasoning tasks in math and medical QA align with this setup, where only a few decision tokens significantly impact the outcome. 
PPO offers token-level advantages with a learned value model, but it is complex to train both the actor and critic models simultaneously, and it is not easily generalizable, as the token-level values from the critic model can make training prone to overfitting. GRPO is critic-free and supports verifiable rewards, but spreads a single sequence-level return across tokens and ignores branching. We introduce Prefix-to-Tree (P2T), a simple procedure that converts a group of responses into a prefix tree and computes nonparametric prefix values V(s) by aggregating descendant outcomes. Built on P2T, we propose TEMPO (Tree-Estimated Mean Prefix Value for Policy Optimization), a critic-free algorithm that augments the group-relative outcome signal of GRPO with branch-gated temporal-difference corrections derived from the tree. At non-branch tokens, the temporal-difference (TD) term is zero, so TEMPO reduces to GRPO; at branching tokens, it supplies precise token-level credit without a learned value network or extra judges/teachers. On Qwen3-1.7B/4B, TEMPO outperforms PPO and GRPO on in-distribution (MATH, MedQA) and out-of-distribution (GSM-HARD, AMC23, MedMCQA, MMLU-Medical) benchmarks, and reaches higher validation accuracy with less wall-clock time.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the challenge of reward sparsity in RLVR. The authors introduce Prefix-to-Tree (P2T), a procedure that aggregates a group of responses into a prefix tree to compute a value function based on descendant outcomes. Building on P2T, they propose TEMPO, an algorithm designed to restore token-level credit assignment and mitigate reward sparsity in GRPO. TEMPO is evaluated on both in-distribution and out-of-distribution (OOD) benchmarks, demonstrating its robustness.

### Strengths
- The paper is clearly written and easy to follow.

- The proposed method is straightforward to implement, improving performance while maintaining the same training cost as the original GRPO framework.

- The approach is intuitively sound and memory-efficient, as it eliminates the need to train an additional value model to provide temporal-difference signals.

- TEMPO demonstrates robustness with small sample sizes. Although a potential concern is that TEMPO might require large group sizes to be effective, the experimental results show that it consistently outperforms the GRPO baseline even with small group sizes ( $\leq$ 9).
Weaknesses

### Weaknesses
- **Concern Regarding Novelty**

While the proposed method is appealing and intuitively sound, its novelty may be limited. As the authors acknowledge in Section 2, prior work [1] has also introduced a tree-search procedure into the GRPO framework. The similarity between TEMPO and TreeRPO raises a concern about the distinct contribution of this work.

[1] TreeRPO: Tree Relative Policy Optimization, arxiv 2025.

- **Missing Baseline**

TreeRPO [1] shares a great similarity with the proposed method TEMPO, yet it is currently absent from the main experimental results.

- **Lack of Justification for Advantage Computation**

The justification for the advantage calculation at line 299 requires clarification. Specifically, the rationale for standardizing the TD error by dividing it by $\text{std}(r)$ is unclear. A more conventional approach would be to normalize by the standard deviation of the TD error itself (e.g., $\text{std}(V(s_{t+1})-V(s_t))$). The authors should explain the motivation behind their chosen normalization method.

### Questions
See about.

### Soundness
3

### Presentation
3

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
This paper presents TEMPO, an RLHF training paradigm that introduces branching within rollouts to address the challenge of sparse rewards. Specifically, TEMPO generates multiple branches from each rollout, groups their final rewards to derive intermediate rewards, and integrates temporal-difference signals for improved training efficiency and stability. Evaluation results show encouraging speedup and better quality.

### Strengths
+ Improving RLHF efficiency and quality is an important problem.
+ The paper is overall well-written.
+ The design is straightforward and reasonable.
+ Evaluations report encouraging improvements.

### Weaknesses
- The contribution appears incremental, particularly compared to existing tree-based methods such as TreeRPO.
- The evaluation is limited, lacking comparisons with highly relevant baselines like TreeRPO.

### Questions
Thank you for submitting your work to ICLR. The paper tackles an important problem, and the idea of branching to collect finer-grained rewards is reasonable. However, the proposed design seems conceptually close to existing tree-based RLHF approaches, and the paper's discussion on novelty over prior work (e.g., TreeRPO) is not entirely convincing. The absence of direct comparison experiments further weakens the claim of contribution.

- Q1: It is unclear how and when the rollout is split to create new branches. Could the authors clarify the branching criterion and timing?

- Q2: The paper uses a fixed number of branches per prompt. It would be better to adapt it. 

- Q3: The paper should consider adding comparison experiments against other tree-based approaches, such as TreeRPO, to strengthen the evaluation.

- Q4: While both branching and temporal-difference learning appear beneficial, an ablation study would help isolate the contribution of each component.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper claims that by grouping the shared prefixes among a group of responses to a query when doing GRPO, one can extract fine-grained credit assignment. They show that this leads to gains in mathematical and medical tasks claiming this proves its generality. They analyze the branching tokens and the effect of the size of the group on the results providing further intuition.

### Strengths
I think the paper correctly claims that credit assignment is troublesome in current reasoning models. Exploiting the shared prefixes, if they exists, is a smart way to extract fine-grained credit assignment. The paper really does go deep in analyzing this behaviour and provides insights. It does a thorough related work section.

### Weaknesses
The problem with the work is that, it does not lead to significantly better credit assignment. The main problem is, the responses from a diverse reasoning model quickly diverge from each other and therefore, aside from the maybe first few thousand tokens, the rest will only have one response. It is very intuitive that this happens: the paths simply diverge due to sampling of tokens. This is very limiting. I have confidence that what authors have shown really holds: the extraction of the extra juice left in the shared prefixes for more accurate credit assignment is useful. But, the problem is that, the effects stop there. It cannot substantially improve: there is no knob to extract finer credit assignment. This paper can be a nice addition to a GRPO run, but it is not a significant contribution: it has small impact.

Aside from the above, I am wondering why the baseline numbers reported for MATH500 are different from Qwen3 official report. In theirs, the Qwen3-4B performance is 97% on MATH500 in thinking mode. However, the authors train and it starts from 82 and goes to 87% on validation. This seems contradictory to me and indicates maybe the authors tested their method on non-thinking mode. I think this is the case especially as the authors set the maximum response length to 1024 tokens: this is very low for reasoning models. I think if the authors tested their method on the correct setup, with high number of max responses tokens and in thinking mode, their method would have significant trouble because aside from the the first thousand tokens, the rest will only have just one response due to diverging branches. 

Unfortunately, these two weaknesses leaves me with no choice but to reject the paper.  I think the results of the paper hold and are correct, but does not scale to actual training runs with long reasoning chains.

### Questions
1-Are you using the thinking mode of the said models or their non-thinking mode?
2-If you set the maximum response length to a standard number, preferably 32K, if compute is limited 8K, would the results hold?

### Soundness
4

### Presentation
4

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
This paper tackles the token-level credit assignment challenge in reinforcement learning for large language models with verifiable rewards (e.g., math or medical reasoning). The authors propose TEMPO. TEMPO is a critic-free method that constructs a prefix tree from multiple sampled responses per prompt, computes nonparametric prefix values, and introduces branch-aware corrections. TEMPO provides fine-grained credit at branching tokens while preserving the efficiency and simplicity of GRPO-style training. Empirical results on MATH, MedQA, and several out-of-distribution benchmarks show that TEMPO outperforms PPO, GRPO, and HEPO in both convergence speed and accuracy using Qwen3-1.7B/4B models.

### Strengths
1. The studied problem is important. Sparse reward is a key limitation of current RL for LLMs, which can cause sub-optimal optimization.

2. The proposed idea is clear and easy to understand. Using a tree to calculate a more accurate reward for each token can be a good way to introduce such dense rewards.

3. The writing is clear and easy to understand.

### Weaknesses
1. The proposed method does not explicitly control the branch of the repeated sampling tree, but relies on implicit branching across sampled responses. In many prompts, only a small portion of tokens can actually form branch points, so most updates may reduce to GRPO with sparse rewards. Some further analysis can be conducted to verify why implicitly building the repeated sampling tree is a good choice and introduces enough dense reward signals.

2. Another concern is related to the experiment setting. As illustrated in Table 2, the maximum response length is only around 1k tokens, which is far shorter than the original model’s typical response length (up to 40k). Such a limited sequence length makes it difficult to fully justify the training effectiveness of the proposed method, as long-horizon reasoning and delayed credit assignment are precisely where token-level advantages should matter most. Moreover, this limitation is closely tied to the previous point on implicit branching; when the response length is long, the later tokens may not share any branches with other responses, which makes the reward very sparse.

3. The training step is also limited. Usually, in recent RL papers, > 800 training iterations are needed to make sure training converges. However, the number of training iterations on Math is 220, which can be too few. Despite that I feel that the overall idea is neat, but the evaluation part is not enough to support the major hypothesis of the paper.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
