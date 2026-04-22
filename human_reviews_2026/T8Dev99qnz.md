# Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 8

## Abstract
Group-based reinforcement learning (RL), such as GRPO, has advanced the capabilities of large language models on long-horizon agentic tasks. To enable more fine-grained policy updates, recent research has increasingly shifted toward stepwise group-based policy optimization, which treats each step in a rollout trajectory independently while using a memory module to retain historical context. However, we find a key issue in estimating stepwise relative advantages, namely context inconsistency, where steps within the same group may differ in their historical contexts. Empirically, we reveal that this issue can lead to severely biased advantage estimation, thereby degrading policy optimization significantly. To address the issue, in this paper, we propose Hierarchy-of-Groups Policy Optimization (HGPO) for long-horizon agentic tasks. Specifically, within a group of rollout trajectories, HGPO assigns each step to multiple hierarchical groups according to the consistency of historical contexts. Then, for each step, HGPO computes distinct advantages within each group and aggregates them with an adaptive weighting scheme. In this way, HGPO can achieve a favorable bias-variance trade-off in stepwise advantage estimation, without extra models or rollouts. Evaluations on two challenging agentic tasks, ALFWorld and WebShop with Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct, show that HGPO significantly outperforms existing agentic RL methods under the same computational constraints. Code is available at https://github.com/langfengQ/verl-agent/tree/master/recipe/hgpo.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes HGPO, a herarichical extension of group-based RL designed to address context inconsistency in long-horizon LLM-based agents. Empirical results on ALFWorld and WebShop show consistent improvements over GRPO and GiGPO.

### Strengths
1. The hierarchical grouping and adaptive weighting mechanism is conceptually simple and computationally efficient.
2. Experimental results demonstrate strong and stable improvements across benchmarks.

### Weaknesses
1. The Proposition 4.1 assumes bias decreases and variance increases monotonically with context depth. In practice, this assumption may not always hold, e.g., noisy long contexts could add both bias and variance.
2. The definition of hierarchical groups $G_k^H(s_t^{(i)})$ assimes exact equality of historical contexts. Does it hold in practice?
3. The notion of context inconsistency is intuitive but not rigorously formalized.

### Questions
1. Figures 2–5 mention oracle groups and context inconsistency bias. How exactly were these oracle advantages computed?
2. Only three random seeds are used. Given the small standard deviations reported for ALFWorld, can the improvements be considered statistically reliable?
3. HGPO is said to add “minimal extra time cost”. Can the authors report wall-clock training time or GPU hours to support that claim?
4. In Page 8, the paper claims that Figure 4 shows stable learning. Could the authors provide a quantitative evaluation to support the claim? It is not obvious that the curves of HGPO are smoother.
5. The weighting $\omega_k$ is heuristic and fixed across training, and I find the $\alpha$ also fixed for main exps. It is a good setting as it does not introduce extra tuning cost. But since variance and bias evolve during training, how does such a static schedule reflect the actual trade-off? 
6. The grouping is computed offline via hashmap lookups. As the policy evolves, group memberships change. Does the method recompute them every epoch? If not, the hierarchical structure may quickly become outdated.

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
3

### Summary
This paper identifies a critical problem in existing stepwise group-based Reinforcement Learning (RL) methods for LLM agents: "Context Inconsistency.", which is particularly unsolved in GiGPO.

The authors provide a strong empirical diagnosis (Figure 2) demonstrating that existing methods like GiGPO (which only groups by $s_t$) suffer from high bias, while a naive "Oracle" solution (using only steps with perfectly identical histories) is unusable due to high variance from data sparsity.

To solve this bias-variance dilemma, the paper proposes Hierarchy-of-Groups Policy Optimization (HGPO), with two key component:
1. Context-aware Hierarchical Grouping
2. Adaptive Weighting Advantage Estimation

Experiments on ALFWorld and WebShop show that HGPO significantly outperforms existing RL baselines, including GRPO and GiGPO, especially on out-of-distribution tasks.

### Strengths
1. The paper is well-written. Figure 2 of the "context inconsistency" problem clearly frames the bias-variance dilemma that existing methods face (GiGPO = high-bias, Oracle = high-variance).

2. HGPO is an intuitive and novel solution. 

3. HGPO achieves sota results, significantly outperforming its baselines.

4. The ablations are comprehensive.

### Weaknesses
1. The proof of the bias-variance trade-off in Appendix B relies on the assumption $b_k \ge b_{k+1}$ (bias decreases as context depth $k$ increases). This assumption is stated but never justified or proven. The authors must provide a formal argument for this assumption or re-frame Proposition 4.1 as a heuristic analysis.

2. The paper's claim to address "long-horizon" tasks is not supported by the evidence. Appendix C.3 reveals the maximum episode lengths are T=50 (ALFWorld) and T=15 (WebShop). A 15-step task is not "long-horizon."This is a major overclaim that misrepresents the scope of the paper's validation.

3. The paper makes the qualitative claim of "minimal additional time cost." This is highly questionable. GiGPO hashes one item per step ($s_t$), whereas HGPO (K=4) must construct and hash $K+1=5$ separate context sequences for every step. This is a non-trivial increase in computational overhead. The paper provides no wall-clock time or GPU-hour comparison against its baselines (GiGPO, GRPO) to substantiate its efficiency claim. 

4. The "adaptive weighting" scheme (Eq. 7) is not truly adaptive; it is a fixed heuristic weighting tuned by a single, manually-set hyperparameter, $\alpha$. Table 4 simply shows that $\alpha=1$ (linear weighting) works best empirically. This is parameter tuning, not an adaptive mechanism that responds to data (e.g., by adjusting weights based on group size/variance). The authors should discuss this in limitations.

### Questions
See weakness

### Soundness
2

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
4

### Summary
This paper identifies the "context inconsistency" problem in stepwise group-based reinforcement learning for LLM agents , arguing that grouping steps by their current state while ignoring their histories leads to severely biased advantage estimation. The proposed method, Hierarchy-of-Groups Policy Optimization (HGPO), addresses this by assigning each step to multiple hierarchical groups based on k-step historical context consistency, and applying an adaptive weighting scheme that assigns higher weights to advantages computed from more consistent groups. The authors provide a theoretical analysis of the bias-variance trade-off. Experiments on the ALFWorld and WebShop benchmarks demonstrate that HGPO significantly outperforms existing baselines, including the prior SOTA method GiGPO.

### Strengths
The paper's primary strength is its elegant and well-motivated mechanism for managing the bias-variance trade-off in advantage estimation. Rather than forcing a binary choice between GiGPO's high-bias/low-variance step-level group and a low-bias/high-variance Oracle group, HGPO proposes a more nuanced solution. By aggregating advantage signals across all K+1 levels of the hierarchy, it provides a principled interpolation that leverages the low-bias signal from high-consistency groups while retaining the low-variance stability from large, low-consistency groups. This mechanism is strongly supported by robust empirical results on recognized benchmarks. The ablation study is particularly convincing, as it clearly isolates the necessity of both the hierarchical grouping and the adaptive weighting components for the method's success.

### Weaknesses
1. Its validity is tightly coupled to the assumption that the agent's "memory" is equivalent to its "raw historical context," as implemented in the paper's prompts (Appendix C.5). This raw-history-as-memory paradigm, while common, is limited. The proposed $C_k$-based grouping would not directly generalize to more advanced agents that utilize summarized memory, as the grouping basis would misalign with the agent's true decision-making state. 

2. The claim of minimal additional time cost is unsubstantiated. The K+1-level hierarchical hash management is necessarily more computationally expensive than the baseline, and the paper provides no quantitative wall-clock time or memory overhead analysis. 

3. The method's scalability with respect to K is unclear. For truly long-horizon tasks (T > 100), a small K may be insufficient, while a large K would suffer from the same sparsity and high-variance issues as Oracle groups. 

4. The method's necessity is predicated on a sparse reward setting; its utility would likely be diminished in settings where effective process rewards are available.

### Questions
1. How would the $C_k$-based grouping function if the agent's policy $\pi(a_t | s_t, h_t)$ used a summarized memory state $h_t$ rather than the raw history?

2. The claim of minimal additional time cost is unsubstantiated; the paper provides no quantitative comparison of wall-clock time and peak memory usage (e.g., for K=4) against the baselines.

3. How should K be chosen relative to a task's average horizon T? How does the method avoid the Oracle group sparsity problem when K must be large for a very long-horizon task?

4. How significant do the authors believe the context inconsistency problem remains in settings with effective process rewards, as opposed to the sparse setting studied?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
To improve training efficiency in long-horizon agentic tasks, existing reinforcement learning (RL) methods typically adopt step-level policy optimization. However, when estimating the advantages of identical step groups, inconsistencies in historical context can lead to biased advantage estimation. A straightforward solution is to use oracle steps, which not only share identical current steps but also identical histories. Nevertheless, oracle steps are rare in practice, leading to the utilization of only a small fraction of the data and consequently causing high-variance advantage estimation. To balance the trade-off between bias and variance, this paper proposes Hierarchy-of-Groups Policy Optimization (HGPO), which aggregates advantages computed under different levels of historical context. This hierarchical aggregation enables HGPO to effectively balance bias and variance, achieving improved training stability and efficiency. Empirical results on ALFWorld and WebShop demonstrate the effectiveness of the proposed method.

### Strengths
1. This paper presents a well-motivated study that clearly illustrates the limitations of trajectory-level, traditional step-level, and oracle-step advantage computations in Figure 1 and 2, thereby making the proposed method appear both natural and intuitively justified.

2. Both the high-level idea and implementation of HGPO are simple yet well-grounded, making it an effective solution to address the context-inconsistency problem.

3. HGPO introduces almost no additional computational overhead compared to other group-based RL methods such as GRPO and GSPO. So, it effectively alleviates context inconsistency and maintains high efficiency.

4. The empirical results demonstrate the effectiveness of HGPO. Especially when K=4, we can observe a clear improvement over naive step-level methods. 

5. The paper combines theoretical justification with extensive empirical evaluation, resulting in a well-rounded and complete study.

### Weaknesses
1. The context inconsistency problem is only partially addressed in this paper rather than being fundamentally resolved. As the value of 
K increases, the number of grouped samples noticeably decreases. This indicates that for longer-horizon tasks, HGPO becomes less effective, since the higher-level history groups become increasingly sparse under HGPO’s grouping mechanism.

2. Therefore, HGPO is effective only when K is relatively small. In the experiments, K is set to 2 and 4, corresponding to 2–4 steps in agentic tasks, which represent relatively short-horizon settings.

However, I would not consider the above weaknesses as major limitations, since the authors have made a meaningful step toward addressing long-horizon tasks.

### Questions
I did few research on LLMs but lots on RL. So, I have the following questions:

1. How is the similarity between states (or histories) measured, and what specific criterion is used to group them?

2. How is the memory module maintained or updated throughout training?

3. In Eq. (8), is the memory utilized as context when computing the importance sampling ratio? And, which policy is updated, $\pi(a|x,s,h)$ or $\pi(a|x,s)$? In my view, $\pi(a|x,s,h)$ is preferred, but in the paper, it seems to be $\pi(a|x,s)$.

### Soundness
3

### Presentation
3

### Contribution
3
