# Communication-Efficient Decentralized Optimization via Double-Communication Symmetric ADMM

- Decision: Accept (Poster)
- Scores: 4, 8, 2, 8

## Abstract
This paper focuses on decentralized composite optimization over networks without a central coordinator. We propose a novel decentralized Symmetric ADMM algorithm that incorporates multiple communication rounds within each iteration, derived from a new constraint formulation that enables information exchange beyond immediate neighbors. While increasing per-iteration communication, our approach significantly reduces the total number of iterations and overall communication cost. We further design optimal communication rules that minimize the number of rounds and variables transmitted per iteration. The proposed algorithms are shown to achieve linear convergence under standard assumptions. Extensive experiments on regression and classification tasks validate the theoretical results and demonstrate superior performance compared to existing decentralized optimization methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes DS-ADMM, a decentralized optimization algorithm for composite problems that incorporates two communication rounds per iteration. The key innovation is a reformulation of consensus constraints that enables information exchange beyond immediate neighbors through a symmetric ADMM framework. The authors claim this approach reduces total communication cost despite increasing per-iteration communication. They provide convergence analysis showing sublinear rates generally and linear rates under metric subregularity conditions. Experiments on Lasso regression and SVM classification demonstrate superior performance compared to existing methods.

### Strengths
- The reformulation of consensus constraints using the symmetric form $u = \tilde{W} v, \tilde{W} u = v$ is creative and well-motivated. This differs meaningfully from prior work (Wang et al., 2018) and enables the application of Symmetric ADMM in a decentralized setting.
- The analysis in Section 3.3 carefully optimizes which variables to transmit and when. The identification that only two communication rounds are necessary and the strategic transmission of dual variables ($a_i^{(t)}$ and $b_i^{(t+1)}$) rather than primal variables shows careful algorithmic design.
- The paper provides both sublinear (Theorem 1) and linear convergence guarantees (Theorem 2). The characterization of sufficient conditions for metric subregularity (Proposition 4) covering PLQ functions and strongly convex cases is comprehensive and practically relevant.

### Weaknesses
- Limited Experimental Scope: Only two problem types (Lasso and SVM) and two relatively small datasets (a9a, a1a) are tested. Network size fixed at $n=30$ agents is small for decentralized optimization. Only random graphs with $p=0.5$ (and $p=0.2$ in appendix) are considered. No evaluation on structured topologies (ring, grid, etc.). No scalability analysis showing how performance varies with network size or sparsity.
- Parameter Sensitivity Not Addressed: The choice of $\tau = 0.01$ appears arbitrary with no justification or sensitivity analysis. The convergence rate depends on $r, \beta, \tau$, and $\rho$ (Theorem 2), but no guidance is provided on selecting these parameters. Different baseline methods may require different tuning efforts, so the fairness of the comparison is unclear.
- The notation switch between $w$ (dual variable blocks) and $w$ (concatenated variable including primal and dual) in Section 4 is confusing. The communication strategy in Section 3.3 is difficult to follow. Why specifically transmit $a_i^{(t)} = w_{2i}^{(t+1/2)} + 1/r(w_{2i}^{(t+1/2)} - w_{2i}^{(t)})$? The derivation or intuition is missing. Algorithm 1 uses $w_{2i}^{(-1/2)}$ in initialization, but this notation is never explained. The relationship between the proximal parameter Q and the mixing matrix W could be explained more intuitively.
- How is the optimal solution $u^{\star}$ computed for measuring suboptimality? What is the convergence criterion in Algorithm 1? Computational cost per iteration is not analyzed—only communication is counted. No discussion of numerical stability or practical implementation challenges.
- The introduction states that multi-consensus schemes "offer limited improvement to the quality of each iteration" but provides no formal analysis or empirical evidence quantifying this claim. The paper would benefit from ablation studies showing why two rounds specifically are optimal.

### Questions
Refer to the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper considers decentralized symmetric ADMM, an important optimization problem, and develops a variation that allows multi-round communications with each ‘iteration’. This is called decentralized symmetric ADMM (DS-ADMM), a decentralized version of S-ADMM. The key step is to formulate an auxiliary constraint that is embedded. This leads to a 2-round approach per iteration. Convergence is guaranteed analytically under standard assumptions, and numerical experiments illustrate the method.

### Strengths
The auxiliary constraint embedding (Prop 2) with proximal linearization is developed to be a useful method for obtaining the desired symmetric ADMM compatibility. 

Generally linear convergence is obtained, while communications overall are reduced because of fast convergence despite the 2-round per iteration needed.

### Weaknesses
The experiments are limited to a single group size, and are rather basic.

Robustness to dropouts or topology changes isn’t clear.  

Tuning is required and sensitivity and robustness to topology and number of agents are not explored.

### Questions
Improved numerical experimentation would strengthen the work.

What, if any, are the implications for scalability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper propose a variant of ADMM algorithm for decentralized consensus optimization which uses so-called double communication strategy

### Strengths
Due to weaknesses, I can not highlight any substantial strengths of the paper

### Weaknesses
It is stated that
> To our knowledge, this is the first decentralized optimization framework that achieves a net reduction in total communication by leveraging fixed multi-round communication within each iteration.

but multi-round schemes showed  communication acceleration in Scaman et al., 2017 and Ye et al., 2023, so the contribution is unclear to me.

There is no theoretical complexity comparison with SOTA decentralized optimization algorithms such as Mudag or even Symmetric ADMM, from which the algorithm was derived. The only comparison is through numerical experiments but they are not representative.

Overall, the contribution seems to be minor

### Questions
> Note that ProxMudag is not included in this comparison because it requires the nonsmooth term to be globally coupled, which is incompatible with this separable formulation (Ye et al., 2023).

Could you please explain the statement?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a decentralized optimization method, DS-ADMM, that embeds two rounds of neighbor communication inside each ADMM iteration, by designing a symmetric pair of linear consensus constraints and then applying S-ADMM with a graph-aware proximal linearization to enable separable subproblems. The algorithm communicates (per iteration) with two d-vectors perr agent in each of two communication rounds by sending cleverly chosen dual combinations a^(t+1) and b^(t+1) rather than two primals, which the authors argue minimizes what must be sent for symmetric ADMM to work. Theoretical results include a non-ergodic O(1/t) sublinear rate and Q-linear convergence under metric subregularity, with explicit constants depending on the spectral gap of the mixing matrix. Experiments on lasso and l2-svm over random graphs with n=30 show faster decrease in suboptimality per communication round than several baselines, including on a sparser graph.

### Strengths
(+) The constraint design is an elegant way to bake two-hop information flow into the constraints themselves. This makes the need for two communications per iteration principled rather than an ad-hoc multi-consensus loop. To my knowledge this particular symmetric, two-block dual realization of decentralized ADMM is novel. 
(+) The communication scheduling -- transmitting two surrogates rather than two primals -- is thoughtfully engineered so each block enables the other block's update with minimal payload. That design is interesting and practically motivated.  
(+) The communication-aware design is reflected in the evaluation. Testing on two graph densities is nice and qualitatively matches the theory.

### Weaknesses
(-) Counting "rounds" alone does not equal communication volume here because DS-ADMM transmits two d-vectors per round (four per iteration). Baselines often send a single d-vector per round. A better comparison should report total scalars (or bytes) transmitted per agent to reach a target accuracy. Without this "net communication reduction" claim is not fully substantiated.

### Questions
- How would you handle smooth but non-proximable f_i (e.g., logistic regression)? Is there a prox-gradient DS-ADMM variant that retains the same communication schedule? 
- Do your conclusions extend to directed or time-varying graphs? What breaks in the proof is W is not symmetric?

### Soundness
3

### Presentation
3

### Contribution
3
