# Parallelizing Tree Search With Twice Sequential Monte Carlo

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Model-based reinforcement learning (RL) methods that leverage search are responsible for many milestone breakthroughs in RL.
Sequential Monte Carlo (SMC) recently emerged as an alternative to the Monte Carlo Tree Search (MCTS) algorithm which drove these breakthroughs. SMC is easier to parallelize and more suitable to GPU acceleration. However, it also suffers from large variance and path degeneracy which prevent it from scaling well with increased search depth, i.e., increased sequential compute. To address these problems, we introduce Twice Sequential Monte Carlo Tree Search (TSMCTS). Across discrete and continuous environments TSMCTS outperforms the SMC baseline as well as a popular modern version of MCTS. Through variance reduction and mitigation of path degeneracy, TSMCTS scales favorably with sequential compute while retaining the properties that make SMC natural to parallelize.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper focuses on model-based reinforcement learning (MBRL) methods that leverage search algorithms.
Specifically, the paper claims that Monte-Carlo Tree Search (MCTS) does not fully utilize the GPU when compared to Sequential Monte Carlo (SMC).
However, the latter suffers from high variance and path degeneracy.
As a result, the paper proposes a method that combines the two, called Twice Sequential Monte-Carlo Tree Search (TSMCTS) that aims to improve upon both existing algorithms.
The paper demonstrates through both discrete and continuous environments TSMCTS is comparable if not better than existing algorithms in episodic returns, and further demonstrates that TSMCTS maintains smaller variance and less path degeneracy.

### Strengths
- The proposed method clearly yields better runtime while maintaining/improving performance when compared to Gumbel MCTS and SMC.
- The proposed method suffers less from high variance and path degeneracy.
- The complexity analysis in the appendix A.4 is helpful to understand the asymptotic trade-offs.

### Weaknesses
- Method
	- On line 268, keeping track of $t$ Q-values can be prohibitively expensive---instead something akin to $Q(\lambda)$ might just do the work.
	- It seems like sequential halving can suffer from high bias if the first iteration filters out good action early---this can easily be the case with stochastic dynamics and low search budget. If this is true, I think the paper should indicate this, otherwise it would be nice to provide a reference, demonstrating that this is not the case.
- Writing
	- On line 116, the notation $s_t$ conflates with the timesteps in the MDP. I suggest disambiguating them with, e.g., superscript, $\bar{s}_t$, etc.
	- Nit: On line 187, I think it should be $\tau_t = s_0, a_0, \dots, s_t, a_t, s_{t + 1}$
	- Theorem 1 should indicate that the assumptions are true $Q^\pi$ and transitions $P$, otherwise it can be misleading in the function-approximation case.
	- I think the writing can be clearer on line 234. If I understand correctly, $a_0^i \sim \pi_\theta(\cdot | s_0)$ means that most particles will start from the most-likely action, and consequently the empirical distribution will only have mass on said action. On the other hand, MCTS uses the estimated Q-value to form the empirical distribution.
- Experiments
	- The experiments should describe how the continuous environments are setup---is the action space discretized, or the cross-entropy loss is replaced with a squared loss, etc.?
	- How does this approach compare against Levin Tree Search and Luby Tree Search [1]?
	- This doesn't affect my score, but how would it perform when the transitions are approximated?

References:  
[1] Orseau, Laurent, et al. "Single-agent policy tree search with guarantees." Advances in Neural Information Processing Systems 31 (2018).

### Questions
See above

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
5

### Summary
This paper proposes TSMCTS, a new Sequential Monte Carlo (SMC) search algorithm for policy improvement. The method is presented as an alternative to MCTS that aims to solve the high variance and path degeneracy issues found in standard SMC for RL methods. The authors build their method incrementally: first, by formalising SMC beyond the Control-as-Inference (CAI) framework; second, by introducing SMCTS, which adds MCTS-inspired value backpropagation to mitigate path degeneracy; and finally, by integrating Sequential Halving at the root to create TSMCTS for better budget allocation. The paper's experiments show TSMCTS outperforming a baseline SMC and GumbelMCTS, claiming it yields lower-variance targets and scales properly with the sequential budget.

### Strengths
This paper's focus is particularly impactful as it targets two fundamental weaknesses that hinder the scalability and performance of Sequential Monte Carlo (SMC) in Reinforcement Learning:

* Targeting **path degeneracy** by preserving information about multiple actions at the root is a crucial research direction. This prevents the search from prematurely collapsing to a single trajectory, which leads to unstable and impoverished policy targets—a key limitation of standard SMC.
* Developing systematic **variance reduction** mechanisms is vital for making SMC a scalable planning algorithm. Focusing on techniques that combat the high variance growth with search depth is a key direction, as this is a fundamental bottleneck that prevents SMC from benefiting from deeper lookaheads.

### Weaknesses
### 1. Insufficient Baselines
* **Ambiguous "SMC Baseline":** The paper introduces an "SMC baseline" that doesn't correspond to a specific, recognized algorithm in the literature. This is confusing and unscientific. The existing methods are distinct: the SMC method from Piché et al. [3], SPO (which uses SMC for policy improvement) [2], and the extension by de Vries et al. with TRT-SMC (SPO + twisted proposals + revived sampling)[4]. The authors should compare against each of these established methods in all experiments. In addition, baselines such as SPO have open-source implementations, so their omission is a significant weakness of this paper. These baselines are a strict requirement, and I would strongly push for rejection without this being fully addressed.
### 2. Flawed Narrative and Positioning
* **Insufficient Coverage of Prior Work:** The related work section incorrectly frames the history of SMC in RL. It misses the seminal work by Lazaric et al. [1] that first introduced SMC for RL. The progression should be: Lazaric et al. [1] -> Piché et al. -> SPO (full policy iteration loop, highlighting parallelism) -> de Vries et al. (improvements on SPO via SMC and proposal changes).
* **Overstated Claims on Parallelism:** The title and framing imply that this paper is the first to identify the parallelism benefits of SMC for RL. This benefit was already robustly established and was a critical contribution of SPO. It is fine to reference this as a benefit of SMC-type methods, but the correct credit must be assigned to SPO where this was demonstrated.
* **Unclear Motivation for Decoupling from Control as Inference (CAI):** The paper doesn't provide a clear rationale for moving away from the CAI framework, especially since SMC is fundamentally a probabilistic inference method for which CAI is a natural fit.
### 3. Method Naming Issues
* **Confusing Introduction of Methods:** The paper introduces three methods (RL-SMC, SMCTS, TSMCTS). It does not make sense to name every iteration of the proposed methods. The paper should propose one method and investigate variations of it to understand the contributions of each change. This risks confusing the literature further, and it is unlikely that future work would use all of these ablations as baselines—just the final, aggregated method. Therefore, only this should be named.
### 4. SPO vs TRT SMC
* In multiple areas of the paper [4] is incorrectly referenced, such as in the experimental setup section. De Vries et al [4] is an iteration upon SPO which introduced the full experiment setup, environments and baselines, from which De Vries uses to investigate benefits to alternative SMC approaches. [4] should only be cited when referencing the specific SMC innovations used in their paper and SPO otherwise.
* SMC used as a policy improvement operator introduced in SPO needs to be made more clear in the related work section, this is a significant paper in the timeline of Sequential Monte Carlo for RL demonstrating not only policy improvement but the benefits of parallelism and competitiveness with MCTS.
---
### References
[1] Lazaric, A., Restelli, M., & Bonarini, A. (2007). Reinforcement learning in continuous action spaces through sequential monte carlo methods. In *Advances in Neural Information Processing Systems 20*.

[2] Macfarlane et al. SPO: Sequential Monte Carlo Policy Optimisation

[3] Piché et al. Probabilistic Planning With Sequential Monte Carlo Methods

[4] de Vries et al. Trust-Region Twisted Policy Improvement

### Questions
- What is the performance of the method from Piche et al (following SPO paper could be referred to as SMC-ENT), SPO and TRT-SMC for all experiments and how does it compare to the final method with all modifications, proposed in this paper

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Twice Sequential Monte Carlo Tree Search (TSMCTS), a planner built on a reformulation of SMC for RL (RL-SMC/SMCTS) that explicitly targets policy improvement at the root rather than trajectory inference. The authors prove that RL-SMC becomes a policy-improvement operator in the infinite-particle limit, then address two classical SMC issues—exploding variance with depth and path degeneracy—by integrating Sequential Halving (SH) at the root and aggregating SMCTS value estimates across SH rounds. Empirically, across discrete and continuous benchmarks, TSMCTS outperforms an SMC baseline and GumbelMCTS, with lower estimator variance, mitigated target degeneracy at the root, and better scaling with sequential compute while retaining SMC’s parallelization benefits.

### Strengths
- Clear conceptual shift. The paper reframes SMC for root-focused policy improvement, pairing importance-weighting/backprop with a value-based “SMCTS” that more closely mirrors MCTS, then layers SH to reduce variance and degeneracy at the root. The pipeline is well-motivated and algorithmically coherent.
 
- Parallelization narrative. The introduction positions SMC as easier to parallelize (GPU-friendly) than MCTS due to MCTS’s sequential nature and tree-memory overhead—useful context for why this line is promising.
 
- Addresses core SMC pain points. The paper explicitly diagnoses variance growth with depth and path degeneracy, then justifies why SH (known budget, repeated resets, per-action parallelism) should help at the root.

### Weaknesses
- Finite-sample guarantees are thin. The key theory (policy improvement) is stated for infinite particles; practical behavior with finite
N, finite evaluation accuracy, and shallow search is not quantified with explicit bias/variance or MSE bounds as functions of depth and budget.
 
- Comparative scope. Beyond GumbelMCTS, stronger or modernized MCTS baselines (e.g., robust PUCT variants, MuZero-style planners under equalized compute/memory) are not deeply explored; reproducibility notes exist, but compute normalization remains largely wall-clock-based, which can be hardware-dependent.
 
- Root-centric metrics. Degeneracy is principally assessed at the root (active actions). It is less clear how diversity behaves down the tree in deeper horizons, particularly for large or continuous action spaces.
 
- Trade-off analysis. While SH intuitively reduces variance, the paper does not provide formal rates showing how SH-aggregation plus per-action budget reallocation improve the estimator of the root value or decision quality relative to SMCTS/SMC at fixed compute.

### Questions
1. Finite N theory: Can you provide finite-particle error bounds (bias/variance or MSE) for the root value estimator and/or the action selection error under TSMCTS, explicitly showing dependence on depth and per-round budgets, and contrasting with SMCTS/SMC?
 
2. Robustness to poor priors. Early SH rounds may prune good actions if the policy prior is miscalibrated. Do you employ safeguards (temperature/bonuses or randomized inclusions) to prevent premature elimination? How sensitive is TSMCTS to prior entropy?
 
3. Beyond the root. Do you track effective sample size/diversity at deeper nodes to confirm that degeneracy is not merely shifted downward? If so, please include these diagnostics; if not, what is your expectation theoretically?
 
4. SH-induced bias. Since SH repeatedly resets to the root and aggregates across rounds, can you characterize the bias–variance trade-off more formally (e.g., a bound on regret at the root vs. horizon depth), and conditions under which SH yields decision-consistent improvements?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an advanced planning method, SMCTS, which integrates the backpropagation mechanism of MCTS into the classical SMC framework. To further enhance the root policy, the authors introduce TSMCTS, which employs Sequential Halving to focus search resources on a progressively smaller set of actions and runs SMCTS in parallel for each action. TSMCTS reduces variance through Q-value averaging and the SH mechanism while preserving the inherent parallelization advantages of SMC methods.

### Strengths
1.The authors present their method systematically, demonstrating step by step how their algorithm design addresses the limitations of existing approaches.
2. Combining the backpropagation mechanism of MCTS with the SMC framework is a novel and interesting idea. The inherent parallelization of SMC also shows strong potential for accelerating planning algorithms.
3. Unlike traditional MCTS, TSMCTS naturally supports both discrete and continuous environments, which is a significant step toward developing a more universal planner.

### Weaknesses
1. The paper includes extensive background material and relies heavily on prior works to explain the proposed methods. This makes it difficult to fully understand certain design choices without consulting the references. I suggest presenting a more self-contained description of the algorithms and moving some background discussions to the related work section.
2. There are several typographical errors and inconsistencies in the notations, suggesting that the notation system could benefit from more careful refinement.
3. The experimental section lacks detailed setup descriptions and thorough result analysis, particularly for the main experiments. The authors should explain the rationale behind the choice of environments, discuss why performance differs across them, and consider evaluating on more than five environments to strengthen the empirical support.

### Questions
1. Why does TSMCTS fail to outperform GumbelMCTS on the Rubik’s Cube task? Is this due to characteristics of the environment, or does TSMCTS generally lose its advantage on more complex tasks?
2. Please discuss the potential of applying TSMCTS to visual-observation tasks such as Atari, or to problems without an explicit model?
3. If my understanding of section 2 is correct, a policy improvement operator typically acts on a policy together with its value function. However, in Equation (16), the current policy is paired with the value of the improved policy. Could the authors please explain the theoretical justification for this formulation?
4. Please discuss how to choose the parameter $\beta$ in Equation (4)?
5. In my understanding, GumbelMCTS is designed for discrete tasks originally. How do the authors apply it to continuous tasks?
6. Other  issues: 
	In Equation (4), $\pi$ and $\pi_\theta$ seem to represent the same quantity. Similar inconsistencies appear elsewhere.
	In line 124, the state is equated to a probability, which seems incorrect or unclear.
	In line 128, what does $k$ represent? Likewise, in line 143, the meaning of $Q_i$ is unclear.
	in equation 7 and 8, the $A_soft$ is given in different forms.
	Algorithm 1 is never explicitly referenced in the text.
	In algorithm 2, I can't find where ancestor identifier is defined. And if the loop starts from $t=1$, $s_1$ is undefined. The same issue appears in Algorithm 3.
Overall, I suggest the authors carefully review the entire paper to correct unclear or inconsistent notations and definitions. At the moment this is one of the key points resulting in the selected score.

### Soundness
3

### Presentation
1

### Contribution
3
