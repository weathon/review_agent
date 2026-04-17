# Multi-Action Self-Improvement For Neural Combinatorial Optimization

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Self-improvement has emerged as a state-of-the-art paradigm in Neural Combinatorial Optimization (NCO), where models iteratively refine their policies by generating and imitating high-quality solutions. Despite strong empirical performance, existing methods face key limitations. Training is computationally expensive, as policy updates require sampling numerous candidate solutions per instance to extract a single expert trajectory. More fundamentally, these approaches fail to exploit the structure of combinatorial problems involving the coordination of multiple agents, such as vehicles in min-max routing or machines in scheduling. By supervising on single-action trajectories,  they fail to exploit agent-permutation symmetries, where distinct sequences of actions yield identical solutions, hindering generalization and the ability to learn coordinated behavior.

We address these challenges by extending self-improvement to operate over joint multi-agent actions. Our model architecture predicts complete agent-task assignments jointly at each decision step. To explicitly leverage symmetries, we employ a set-prediction loss, which supervises the policy on multiple expert assignments for any given state. This approach enhances sample efficiency and the model's ability to learn coordinated behavior. Furthermore, by generating multi-agent actions in parallel, it drastically accelerates the solution generation phase of the self-improvement loop. Empirically, we validate our method on several combinatorial problems, demonstrating consistent improvements in the quality of the final solution and a reduced generation latency compared to standard self-improvement.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel approach to solving combinatorial optimization problems that require simultaneous assignment between multiple agents and multiple tasks over time. Such assignment problems can generally be formulated as bipartite matching problems.
The authors propose a method that jointly embeds multi-agent and multi-task information to generate joint logits, performs assignment based on these logits, and trains a neural network for joint logit embedding using a self-improvement learning framework.
Experiments are conducted on FJSP, FFSP, and HCVRP combinatorial optimization problems, showing faster inference and higher solution quality compared to previous studies.

### Strengths
- The strategy of assigning all agents and tasks simultaneously through a single joint logit output is efficient and demonstrates strong performance.

- The proposed approach is expected to be useful for solving other similar types of combinatorial optimization problems or for inspiring further research in this area.

- The training procedure for producing joint logits is logically well-structured.

- The paper is overall well-organized and highly readable.

### Weaknesses
- The overall model structure based on joint logits is similar to prior works (e.g., Kwon et al., 2021), and its novelty is mainly limited to the use of joint logits for agent–task assignment and the application of self-improvement for model training.

- The application of self-improvement does not introduce substantial new ideas or enhancements, and it appears to largely adopt existing approaches without major modifications.

### Questions
None

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
The paper proposes MACSIM, a method to solve multi-agent CO problems, inspired by recent self-improvement methods for neural CO. More concretely, MACSIM learns a multi-agent policy for joint agent-task assignments, where the policy is learned via self-improvement: the policy is trained to imitate previously found good solutions. Through experiments on the FJSP, FFSP, and HCVRP problems, the authors show the efficacy of their method against relevant baselines.

### Strengths
- The paper is mostly well-written.
- The problem studied in this paper (multi agent CO) is of interest to the neural CO community and the proposed method (self-improvement with a joint policy) is novel.
- The empirical results are good.

### Weaknesses
- The authors claim that prior self-improvement methods assume a unique optimal action per step, which limits their applicability to multi-agent settings with symmetric solutions. However, since the policy here is stochastic, it’s unclear to me why it couldn’t naturally capture multiple optimal actions if the imitation data already reflects these symmetries. Could the authors clarify this?

### Questions
See weaknesses.

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
This paper identifies a critical limitation in existing self-improvement (SI) methods, such as those proposed by Corsini et al. (2024) and Pirnay & Grimm (2024). The authors argue that the standard "next-token" prediction paradigm, which learns from a single expert trajectory, fails to account for agent-permutation symmetries in multi-agent problems. This forces the model to learn an arbitrary agent order, hindering coordination and sample efficiency. To solve this, the paper proposes MACSIM (Multi-ACtion Self-Improvement). The core contribution is a framework that: 1) Uses a policy to predict a single, joint logit matrix ($M \times N$, a "$heatmap$" for all agents and tasks in one forward pass. 2) Employs a set-prediction loss ($\mathcal{L}_{CE}$) to supervise this heatmap on the set of expert assignments, making the training signal invariant to agent permutation. The authors claim MACSIM achieves higher solution quality, better sample efficiency, and "drastically accelerated" solution generation on multi-agent scheduling and routing problems.

### Strengths
1. The paper’s primary strength is its clear identification of the "agent-permutation symmetry" problem. The diagnosis that standard SI's next-token supervision (SLIM) implicitly punishes valid, symmetric solutions as "errors" is a significant contribution.
2. The decoupling of the (expensive) one-shot policy evaluation to get the joint logits from the (fast) $M$-step autoregressive sampling (Algorithm 1) is a clever way to amortize computation while guaranteeing a conflict-free joint action.
3. The experimental results show that MACSIM consistently outperforms its baseline (SLIM).

### Weaknesses
1. The paper's core claim is that it better handles multi-agent symmetries. The authors cite DPN (Zheng et al., 2024) in their related work, noting it also targets multi-agent permutation symmetries, albeit using a different method (symmetric baselines). However, despite both MACSIM and DPN being applied to min-max routing problems (HCVRP), DPN is completely absent from the experimental comparisons in Table 1. DPN is a state-of-the-art method that directly addresses the same core problem (multi-agent symmetry). Failing to compare against it empirically, while instead comparing against a re-implemented SLIM baseline, significantly weakens the paper's claims of superiority in this domain.
2. The paper positions itself as an improvement over standard self-improvement (SI) but only provides a direct experimental comparison against a re-implemented "SLIM" (Corsini et al., 2024). It fails to provide a direct experimental comparison to other significant and recent work in SI, notably Pirnay & Grimm (2024), which it cites alongside SLIM as the methods it aims to improve upon. While the authors group these methods together conceptually, they are not identical. A direct comparison against Pirnay & Grimm (2024) would be necessary to fully substantiate the claim of superiority over the class of standard SI methods.

### Questions
1. Your paper cites DPN (Zheng et al., 2024) and notes that it also addresses agent-permutation symmetries. Given that both your method and DPN are evaluated on min-max routing problems (HCVRP), why was DPN not included as a direct experimental baseline in your empirical comparison (Table 1)?
2. Could you provide a computational profile of your model at inference? Specifically, what percentage of the wall-clock time is spent on the $T$-step policy re-encodings versus the $M \times T$-step autoregressive sampling (Algorithm 1)? This would clarify if the accelerated part is a meaningful portion of the total inference time.
3. You train with the $\mathcal{L}_{CE}$ loss, which treats all $M$ agent assignments as independent classification problems. However, you infer using a sequential, dependent sampling algorithm (Algorithm 1). Why is this independent loss sufficient for learning a policy that must produce coordinated, dependent assignments?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a multi-agent extension of the self-improvement paradigm for neural combinatorial optimization (NCO). Instead of predicting one action per step, the proposed method jointly samples actions for all agents from the full joint-agent action space. By incorporating agent-permutation symmetry into both the policy design and training objective, the approach improves coordination among agents and produces higher-quality solutions compared to existing self-improvement baselines.

### Strengths
The paper effectively integrates ideas from reinforcement learning–based NCO and self-improvement learning. It makes good use of the MatNet architecture, originally developed for RL, and adapts it to the self-improvement setting. The main idea is simple yet powerful—using the joint action score matrix to derive agent–task matchings, which is both logical and effective. In addition, the paper is very well written, with rigorous equations and precise mathematical definitions.

### Weaknesses
While the paper presents a clear and elegant formulation, the novelty appears somewhat incremental, as it mainly builds upon existing self-improvement methods by reformulating them for multi-agent joint-action modeling. 

The method relies on full re-encoding of the problem state at each step, which may hinder scalability to larger instances. Moreover, since each re-encoding operates only on the updated state—without explicitly retaining information about past actions or the original problem context—the approach may have limited applicability to problems with temporal dependencies or long-horizon constraints.

### Questions
None

### Soundness
4

### Presentation
3

### Contribution
3
