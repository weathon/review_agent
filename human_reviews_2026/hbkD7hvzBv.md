# From Static to Dynamic: Adaptive Monte Carlo Search for Mathematical Process Supervision

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 0, 2, 4

## Abstract
The quality of process data plays a key role in training a Process Reward Model (PRM), which can enhance the complex mathematical reasoning capability of large language models. Existing methods estimate the quality of reasoning steps based on a fixed-budget sampling strategy and navigate a vast search space to perform path expansion during the automated data generation process, resulting in their inefficiency and inflexibility. To address these issues, we propose Adaptive Monte Carlo Search (AMCS), a framework that transforms data generation from fixed, static to adaptive, dynamic search at the level of node value estimation and path expansion. On one hand, AMCS adaptively refines estimation by allocating more samples to uncertain reasoning steps while using fewer samples for those that are easier to estimate. On the other hand, it enhances the path expansion through a Monte Carlo algorithm with a temporally adaptive policy that begins with broad exploration and gradually shifts toward exploiting the most promising directions. With AMCS, we construct a large-scale dataset MathSearch-200K of about 200K process supervision examples for training PRMs. To verify the effectiveness of our method, we conduct extensive experiments on four mathematical reasoning benchmarks. Experimental results show that Qwen2.5-Math-7B-PRM-AMCS achieves up to 76.2% accuracy on MATH500 with GLM-4-9B, outperforming all baseline PRMs. Notably, a 7B model supervised by Qwen2.5-Math-7B-PRM-AMCS surpasses a 72B model with weaker supervision. Moreover, Qwen2.5-Math-7B-PRM-AMCS maintains consistent advantages on out-of-distribution problems, demonstrating strong generalization capability. Our code is available at https://anonymous.4open.science/r/AMCS-065C/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a data synthesis engine for generating process supervision in mathematical reasoning. The key contribution of this proposed method (AMCS) is that it considers both generation confidence and solution complexity to encourage adaptive budget allocation when collecting process supervision data. The authors provide comprehensive experiments across multiple LLMs, various inference methods, and different mathematical reasoning benchmarks. However, there are several concerns: (1) lacking key ablation study on the utility of the Uncertainty-Driven Iterative Refinement and the change of value (Eq.6) in MCTS; (2) the trivial improvement compared with baselines on Table 1. Overall, this is a well-written paper with interesting ideas, but the concerns mentioned above need be discussed.

### Strengths
1. The paper introduces a novel formulation for calculating the value function in MCTS, which considers both generation confidence and solution complexity to encourage adaptive budget allocation. This adaptive approach is intuitive and particularly important for challening problems with high uncertainty.

2. The formulation of uncertainty quantification is thoughtful, considering both cluster-level and node-level uncertainty. The uncertainty measure $\delta$ converges as the sampling size $n$ increases, effectively guiding the exploration in MCTS.

3. This paper provides comprehensive experiments across multiple LLMs, various inference strategies with PRM, and different challenging mathematical reasoning benchmarks, demonstrating strong generalizaion across different settings.

### Weaknesses
1. It would be helpful if the authors could clarify what constitutes the basic unit in node estimation. According to Figure 2, it appears that the different rollout $r_i$ from the current step serves as units for clustering. For example, as node $S_{2,1}$, AMCS seems to sample different solutions from this node as the basic clustering units. The authors should provide more explanations of this process.

2. Missing critical ablation study. While the authors propose a new value function calculation considering both generation confidence and solution complexity, the paper lacks a key comparison with the vanilla MCTS value function based on both visit counts and cumulative rewards. This comparison is essential to demonstrate the utility of the proposed modifications. Without it, the current experiments do not sufficiently support the paper's core contribution.

3. The paper is difficult to follow as the formulation of uncertainty measure $\delta$ is relegated this to appendix, despite it appearing to be central to the entire method. Additionally, the introduction of uncertainty quantification (Appendix B.2) lacks the necessary references, such as Wilson score interval.

4. The use of squared weights in Eq.14 ($\frac{n_j}{n_{total}}^2$) lacks explanation. The authors should provide theoretical or empirical justification this design choice.

5. Limited improvement in main results. The performance gains in Table 1 appears to be marginal, with only +0.1% improvement in average acc for Llama3.2-MCTS, and +0.7% in Llama3.2-BeamSearch. Combined with the missing ablations mentioned in W2, the overall experiment in this work is insufficient to demonstrate the effectiveness of the proposed method.

6. In addition to the issues raised in W2 and W4, Table 1 lacks comparison with majority voting baselines, which is crutical for readers to understand the relative effectiveness of PRM-based methods versus voting approaches.

7. It would be helpful, especially for the readers unfamilar with this area, if the authors could explain the different PRM inference strategies in the appendix. For example, it is unclear whether the beam search method refers to the step-level beam search (e.g., as in [1]) or vanilla token-level beam search [2].

[1] AlphaMath Almost Zero: process Supervision without process
[2] Beam Search Strategies for Neural Machine Translation

### Questions
1. In Figure 2, could you clarify whether each rollout $r_i$ represents a complete solution path or partial trajectory? How do you handle rollouts of different lengths during clustering?
2. Did you experiment with vanilla MCTS value function as a baseline? If so, what were the results? If not, what technical challenges prevented this comparison?
3. What motivated the specific choice of squared weights in Eq.(14)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper addresses the inefficiency of fixed-budget sampling in generating data for Process Reward Models (PRMs). The authors propose Adaptive Monte Carlo Search (AMCS), a framework that shifts from static to dynamic data generation. Key components include:

1. Uncertainty-Driven Adaptive Sampling: Dynamically allocates more samples to uncertain reasoning steps by clustering diverse rollouts and iteratively refining estimates.

2. Adaptive Path Expansion: Uses MCTS with a temporally decaying exploration rate to balance discovering new paths and exploiting promising ones.

Using AMCS, the authors create MathSearch-200K. 

However, I don't see a reasonable performance gain compared to other existing datasets, and the gain of computation is not clear either.

### Strengths
Good exploration of uncertainty in process supervision.

### Weaknesses
1.  The experiments and results presented are not clear, for example for the BoN, how does the performance change as the N increase?
2. The authors claim the best performance of AMCS with 76.2% beats all baseline models, yet the other methods show similar number, especially PRM-800K. So performance-wise, I don't see a big gain of AMCS.
3. Similar to point 2, in Figure 3, the gain is too small to claim the best with out the CI provided.
4. Given the performance is close, I don't see the efficiency gain compared to other method like math-shepherd.

### Questions
1. "A 7B model paired with Qwen2.5-Math-7B-PRM-AMCS (70.6% on MATH500) substantially outperforms a 72B model with weaker supervision (65.0% with Deepseek-PRM), despite requiring approximately 10× fewer parameters." How was this experiment done?
2. Section 3.3 shows extreme similarity to Section 3.3 in https://arxiv.org/pdf/2406.06592. Especially the equation 7 and 8 in your paper vs equation 2 and 3 in the paper I provided.

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
The paper proposes Adaptive Monte Carlo Search (AMCS) — a dynamic framework for generating high-quality process supervision data used to train Process Reward Models (PRMs) that enhance the mathematical reasoning of LLMs. Existing Monte Carlo–based methods rely on fixed sampling budgets and static search strategies, leading to inefficiency and rigidity. AMCS addresses these issues through two innovations: 1. Uncertainty-driven adaptive sampling, which allocates more computation to uncertain reasoning steps and fewer to well-estimated ones. 2. Adaptive path expansion, which starts with broad exploration and gradually focuses on the most promising reasoning trajectories using a time-varying exploration–exploitation balance.

### Strengths
1. AMCS is a clear conceptual advance over traditional static Monte Carlo Tree Search (MCTS) methods used for process supervision.
2. It addresses two key inefficiencies in prior work. One is fixed-budget sampling, and another is static path expansion.

### Weaknesses
1. The novelty is incremental given previous related works within the scope of MCTS or PRMs. e.g., the two important missing references [1,2] (I think there are a lot of related works should be mentioned besides the two), and the PRM work [3] also designed a new MCTS with different expansion and exploration method. In summary, a simple modification of MCTS is not quite impressive.
2. Math-Shepherd is a method of Monte Carlo estimation, rather than focusing on tree search. It's more reasonable to compare with OmegaPRM[3], which is a relatively fair comparison.
3. The experimental results are not impressive, even on the saturated dataset AIME or Math 500. The performance of search method is far from the RLVR based methods, which is not even discussed or compared in this paper.

[1] Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning
[2] AlphaMath Almost Zero: Process Supervision without Process
[3] Improve Mathematical Reasoning in Language Models by Automated Process Supervision

### Questions
Refer to weaknesses.

### Soundness
3

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
4

### Summary
This paper proposes **Adaptive Monte Carlo Search (AMCS)**, a dynamic framework for generating high-quality process supervision data to train **Process Reward Models (PRMs)** for mathematical reasoning. Unlike existing methods that use fixed sampling budgets and static path expansion, AMCS introduces **uncertainty-driven adaptive sampling**—allocating more rollouts to uncertain reasoning steps—and a **temporally adaptive exploration-exploitation strategy** during path expansion, starting with broad exploration and gradually focusing on promising paths. Using AMCS, the authors construct **MathSearch-200K**, a dataset of ~200K annotated reasoning trajectories. Experiments on four mathematical benchmarks (e.g., MATH, AIME) show that a PRM trained with AMCS (**Qwen2.5-Math-7B-PRM-AMCS**) significantly outperforms existing PRMs, even enabling a 7B model to surpass a 72B model under weaker supervision. The method demonstrates strong generalization, especially on out-of-distribution and complex problems.

### Strengths
The paper is well-motivated, addressing a key inefficiency in the generation of data for process supervision: the uniform allocation of search budgets. The proposed AMCS method is a sound and innovative solution that dynamically allocates rollouts based on node uncertainty. In theory, this adaptive strategy should significantly improve the efficiency of data generation, allowing for more accurate value estimation and the discovery of more diverse reasoning paths within a comparable computational budget. The strong empirical results on downstream tasks suggest that the data generated by this method is indeed highly effective.

### Weaknesses
The primary weakness of this work lies in the experimental design, which does not strictly isolate the impact of the adaptive sampling strategy, making it difficult to draw a direct causal link between the method and the observed performance gains.

1. **Confounding Variables in Comparisons:** The main evaluation compares a PRM trained on the new AMCS-generated dataset with 200k paths against baselines trained on existing, often smaller, public datasets (e.g., 80k paths). This introduces significant confounding variables, most notably dataset size and the source of the underlying problems. Consequently, it is difficult to definitively attribute the performance improvement solely to the superiority of the adaptive sampling strategy rather than to these other factors.

2. **Lack of Robustness and Sensitivity Analysis:** The evaluation lacks a sensitivity analysis on the number of rollouts. The comparison is largely centered around a single operating point (a fixed budget of 16 rollouts), which leaves the method's performance-versus-cost trade-off unexplored. It is unclear if the reported advantages of AMCS would persist, diminish, or amplify under different computational budgets, which is a critical aspect for its practical application.

### Questions
1. To better isolate the effect of your adaptive sampling strategy, have you considered or performed an experiment where both the adaptive and a fixed-budget baseline generate datasets of the **same size** from the **same source problems** under a **comparable total computational budget**? Such a direct comparison would more rigorously validate the claimed benefits of adaptive sampling itself.

2. Could you provide an ablation study on the number of rollouts? A comparison between the fixed and adaptive strategies at both lower and higher rollout budgets would be highly valuable for understanding the practical applicability and robustness of your method.

### Soundness
3

### Presentation
3

### Contribution
3
