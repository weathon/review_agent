# Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 6, 6

## Abstract
We introduce an order-invariant reinforcement learning framework for black-box combinatorial optimization. Classical estimation-of-distribution algorithms (EDAs) often rely on learning explicit variable dependency graphs, which can be costly and fail to capture complex interactions efficiently. In contrast, we parameterize a multivariate autoregressive generative model trained without a fixed variable ordering. By sampling random generation orders during training—a form of information-preserving dropout—the model is encouraged to be invariant to variable order, promoting search-space diversity and shaping the model to focus on the most relevant variable dependencies, improving sample efficiency. We adapt Generalized Reinforcement Policy Optimization (GRPO) to this setting, providing stable policy-gradient updates from scale-invariant advantages. Across a wide range of benchmark algorithms and problem instances of varying sizes, our method frequently achieves the best performance and consistently avoids catastrophic failures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an order-invariant reinforcement learning framework for black-box combinatorial optimization (CO).
It builds on the principles of Estimation-of-Distribution Algorithms (EDAs) and policy-gradient reinforcement learning, using a neural autoregressive policy trained with random variable generation orders (to encourage order invariance).
The authors adapt the Generalized Reinforcement Policy Optimization (GRPO) objective for stable updates and evaluate their approach on synthetic benchmarks (NK, NK3, QUBO) comparing it against hundreds of baselines (Nevergrad algorithms, PBIL, MIMIC, BOA, Tabu search).

### Strengths
### Originality:
- The paper provides a clean formal connection between Estimation-of-Distribution Algorithms (EDAs) and policy-gradient reinforcement learning. This unification, while incremental, is conceptually tidy and may serve as a bridge between the EDA and RL communities.
- Adapts permutation-invariant training via random variable generation orders to the EDA framework, linking it to “information-preserving dropout.”
- The use of GRPO with rank-based advantages is a technically sound choice for the black-box setting and is well-justified.

### Quality & Clarity:
- The paper is technically precise and mathematically detailed, with complete derivations and careful notation.
- The proposed method is methodologically sound, deriving PPO/GRPO-like updates consistent with the Information-Geometric Optimization framework.

### Significance:

- The method avoids the NP-hard problem of learning explicit DAG structures required by classical multivariate EDAs like BOA and MIMIC.
- Demonstrated robustness across diverse landscape types (smooth K=1 to rugged K=8) without hyperparameter tuning, which is practically valuable.
- The neural parametrization allows polynomial scaling with problem size, unlike exponential growth in classical multivariate EDAs with contingency tables.
- The comparison against the massive suite of Nevergrad algorithms is commendable and shows the method is competitive against a broad spectrum of state-of-the-art black-box optimizers.

### Weaknesses
1. Limited Conceptual Novelty

    - The key innovation — training across random variable orderings to promote order invariance — has strong precedents in Neural Combinatorial Optimization (NCO) and symmetry-aware learning:

        - POMO [1] randomizes start nodes to enforce permutation invariance.

        - Sym-NCO [2] explicitly leverages symmetricities such as rotational and reflectional invariance

    - Consequently, the contribution feels more like an application of existing invariance principles to EDAs than a new algorithmic concept.

2. Lack of Comparison with Architecturally Invariant Models
    - Beyond stochastic invariance, there exists a well-established line of work on architectural permutation invariance, including Deep Sets [3] and permutation-equivariant GNNs (e.g. Transformers w/o PEs).
    - These models encode invariance by design, using symmetric aggregation or equivariant message passing, ensuring that outputs are independent of variable order. In contrast, the paper’s approach enforces invariance statistically by randomizing orderings during training.
    - A comparison or ablation against such architecturally invariant alternatives—for instance, parameterizing the EDA’s joint distribution with a Deep-Set–style aggregator—would clarify whether the proposed stochastic approach offers distinct advantages (e.g., improved exploration or diversity) or merely approximates an already well-understood symmetry principle.

3. Slow and Sample-Inefficient Convergence

    - Figures 1 and 2 show that the proposed (σ, σ′)-RL-EDA requires thousands of objective evaluations before matching or surpassing baselines, whereas simpler methods (e.g., PBIL, Tabu, or CMA-like evolutionary variants in Nevergrad) often reach comparable quality within hundreds of evaluations.

    - This indicates poor sample efficiency, which is particularly problematic in black-box optimization, where function evaluations are typically expensive.

    - The authors do not analyze this behavior or discuss why the policy-gradient dynamics are so slow (e.g., delayed credit assignment, stochasticity from random orders, small learning rates).

    - As a result, while asymptotic performance looks competitive, the method’s practicality for real-world black-box problems is questionable.

4. Marginal Empirical Advantage

    - Although the method shows competitive asymptotic performance on some large instances, it shows inconsistent performance across problem regimes: it underperforms on small-scale tasks, where simpler baselines like PBIL or Tabu reach near-optimal solutions quickly, and it fails to converge on highly rugged instances such as NK3 with $K=8$. 
    - This indicates poor sample efficiency and instability under both low- and high-complexity settings, raising concerns about the method’s robustness and practical applicability in black-box optimization problems with limited evaluation budgets.

5. Computational Cost and Missing Efficiency Metrics

    - The algorithm introduces nontrivial computational overhead: sampling new permutations for each trajectory, maintaining multiple neural policies (per-variable networks), and performing repeated PPO-style updates. Yet the paper reports no wall-clock times, runtime scaling, or resource usage.

    - Moreover, parameter sharing is mentioned as a means to scale to large $N$, but no experiment demonstrates it.

6. Clarity and Presentation Issues

    - The paper is dense and notation-heavy, reproducing standard policy-gradient derivations with extensive formalism that obscures the high-level idea.

    - The intuitive motivation — that random orders act like dropout on dependency structure — is insightful but underexplained.

    - Many main results are relegated to the appendices, and the narrative buries the empirical findings behind layers of mathematical restatement.

    - The contribution could be conveyed far more effectively through conceptual diagrams and lighter notation.

[1] Kwon, Y. D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020). Pomo: Policy optimization with multiple optima for reinforcement learning. Advances in Neural Information Processing Systems, 33, 21188-21198.

[2] Kim, M., Park, J., & Park, J. (2022). Sym-nco: Leveraging symmetricity for neural combinatorial optimization. Advances in Neural Information Processing Systems, 35, 1936-1949.

[3] Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., & Smola, A. J. (2017). Deep sets. Advances in neural information processing systems, 30.

### Questions
- Early-budget behavior: Can you report performance at fixed small budgets (e.g., 500/1k evals) and discuss how (σ,σ′) could be modified to improve early convergence (e.g., curriculum on β, λ)? 

- Runtime/compute: What are wall-clock times and GPU/CPU footprints vs. strong Nevergrad baselines and Tabu at matched budgets? 


- Ablate order count: How does # of permutations per update affect stability, speed, and final quality? (Training-time cost vs. benefit.) 


- Parameter sharing: Can you demonstrate shared-backbone variants (per-variable heads) to establish scaling viability?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper formulates the black-box combinatorial optimization problem as an MDP and reinforcement learning framework.

### Strengths
The paper formulates the problem of EDA as an MDP, allowing for the usage of RL. This allows for random orderings of variables during both training and inference, unlike other approaches (e.g. autoregression). RL-EDA outperforms 10 baselines on the QUBO and NK tasks considered, and the paper presents thorough ablations on key design questions. The paper presents possible explanations for the varying performance of various variable orderings in training and inference (i.e. $\delta, \delta’$-RL-EDA, $\delta, \sigma’$-RL-EDA, etc in Figure 1).

### Weaknesses
The paper does not present algorithmic novelty or new insights on the RL side, but seems to rather formulate the problem of EDA as an MDP and apply well-known, existing RL techniques to solve it. For example, the derivations in Appendix B and C seem to follow closely from proofs in the existing literature. 

The approach attempts to apply insights from GRPO to avoid learning critics in on-policy RL like PPO. What is the motivation for avoiding learning value functions? 

In Equation 7, Monte-Carlo samples are used for the approximation. Is there an ablation over the number of samples used here?

### Questions
See above section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces an order-invariant RL framework for black-box combinatorial optimization. Instead of learning explicit variable dependency graphs, it adapts an autoregressive model trained without a fixed ordering by sampling random generation orders during training. It uses GRPO as an RL solver for stable policy-gradient updates. Empirical study demonstrates robustness to structural uncertainty and adaptivity to complex, high-dimensional combinatorial search spaces.

### Strengths
1. this paper introduces a novel discrete black-box optimization framework. It uses neural networks to capture complex interactions between variables and uses policy gradient method for optimization;
2. the empirical study demonstrates superior performance and robustness.

### Weaknesses
1. the presentation of this paper could be improved. The MDP formulation in Section 3.1 is somewhat difficult to follow. Including a concrete example would help clarify the setup.
2. According to Figure 2, the baseline methods outperform the proposed approach when the number of calls to the objective function is small. Could the authors comment on this observation?
3. It would be helpful if the authors could discuss the computational complexity of the proposed method.

### Questions
See wekanesses

### Soundness
3

### Presentation
2

### Contribution
3
