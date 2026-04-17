# LMask: Learn to Solve Constrained Routing Problems with Lazy Masking

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Routing problems are canonical combinatorial optimization tasks with wide-ranging applications in logistics, transportation, and supply chain management. However, solving these problems becomes significantly more challenging when complex constraints are involved. In this paper, we propose LMask, a novel learning framework that utilizes dynamic masking to generate high-quality feasible solutions for constrained routing problems. LMask introduces the LazyMask decoding method, which lazily refines feasibility masks with the backtracking mechanism. In addition, it employs the refinement intensity embedding to encode the search trace into the model, mitigating representation ambiguities induced by backtracking. To further reduce sampling cost, LMask sets a backtracking budget during decoding, while constraint violations are penalized in the loss function during training to counteract infeasibility caused by this budget. We provide theoretical guarantees for the validity and probabilistic optimality of our approach. Extensive experiments on the traveling salesman problem with time windows (TSPTW) and TSP with draft limits (TSPDL) demonstrate that LMask achieves state-of-the-art feasibility rates and solution quality, outperforming existing neural methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents LazyMask (LMask), a new decoding mechanism coupled with refinement intensity embedding (RIE) that augments model state with search-trace information for solving hard-constrained routing problems, specifically TSPTW and TSPDL. The key design is integrating a backtracking-based masking strategy that enables neural constructive methods to generate feasible solutions more effectively. Results on TSPTW and TSPDL show that LMask achieves lower infeasibility rates and improved solution quality compared to baselines.

### Strengths
- The paper is generally well-written with clear presentation and logical structure. 
- The proposed LMask represents a meaningful advancement for neural constructive methods applied to TSPTW and TSPDL.
- The inclusion of theoretical analysis provides valuable insights.
- The experimental evaluation is extensive and the results are promising.

### Weaknesses
- The applicability of the Lmask seems limited. The discussions and experiments focus almost entirely on TSPTW and TSPDL, both of which involve temporal constraints. It is uncertain whether the backtracking-based masking strategy and RIE mechanism can be effectively extended to other complex constrained VRPs with different constraints.
- The paper lacks an in-depth analysis of the gap between the true potential set $S$ and its estimation $\hat{S}$. There is no empirical or theoretical investigation of how well such approximations work under different conditions.
- The selection of the backtracking budget $R$ appears empirical without principled guidelines.
- Infeasibility rates in Tables 1 and 2 are still not 0.00%, while classical methods like LKH3 achieve perfect feasibility. The paper should investigate why the Lmask still fails in those cases and what additional methods may be needed to fully close this gap.

### Questions
Please see my comments above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes LMask, a novel method for enhancing auto-regressive neural solvers on constrained routing problems. Specifically, it integrates a backtracking mechanism during decoding to prevent from infeasible solutions, and designs a refinement intensity embedding method to mitigate representation ambiguity induced by backtracking. The validity of the proposed method is theoretically analyzed under specific assumptions. Experiments on two typical constrained routing problems, TSPTW and TSPDL, demonstrate its superior performance over existing methods.

### Strengths
1. The proposed method is well-motivated. The paper is well-written and easy to follow.
2. On the two benchmark problems TSPTW and TSPDL, the results clearly demonstrate the superiority of LMask against competitors. 
3. The authors provide extensive ablation and hyperparameter studies, with clear analysis.

### Weaknesses
1. The meaning of the theoretical parts is limited, since the conditions and assumptions seems much too strict.

### Questions
1. How does LMask perform when paired with other architectures, such as LEHD?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes LMask, a learning framework for constrained routing that decodes with a LazyMask mechanism combining lightweight lookahead with adaptive backtracking. It introduces refinement intensity embedding (RIE) to encode the backtracking trace into the decoder. Theoretical guarantees are also provided: with unbounded backtracking, LazyMask generates only feasible solutions and assigns non-zero probability to all feasible ones, and a probabilistic optimality bound is analyzed.

### Strengths
1.	The proposed LMask contributes significantly to the current masking mechanism, which excludes the infeasible actions perfectly and efficiently via a smart backtracking design. 

2.	Proposition 4.1 provides a strong feasibility guarantee, lending theoretical grounding to masking-based constraint handling.

3.	The method achieves notable improvements on the TSPTW and TSPDL benchmarks, while maintaining competitive inference times and demonstrating robustness across instance scales and distributions

### Weaknesses
The theoretical contribution of Theorem 4.3 appears limited for practice. Its bound relies on an approximation assumption and does not directly inform the backtracking design or concrete hyperparameter choices, thus offering limited guidance for algorithmic tuning.

### Questions
Could the proposed backtracking mechanism be extended to large language model (LLM)-based solvers? While LLMs do not share the same MDP formalism, they are also autoregressive. I am curious about whether LazyMask-style backtracking, and RIE-like signals might transfer to general sequential models. This is only for discussion; no additional experiments are requested.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper aims to address the challenge of constrained routing problems. It introduces the LazyMask decoding method, which lazily refines feasibility masks with the backtracking mechanism. In addition, it employs the refinement intensity embedding to encode the search trace into the model. The proposed method is tested on TSPTW and TSPDL.

### Strengths
- This paper aims to address the challenging constrained routing problems in neural methods.
- The idea of using lazy masks to correct the decoding errors on the nonlinear routing problems is interesting.
- The performance on the complex TSPTW and TSPDL is good, with low infeasibility and competitive solution quality. The experiments are thorough.

### Weaknesses
- The backtracking mechanism introduces additional computational overhead. 
- The idea is not that impressive. PIP adopts a lookahead mask, whereas LMask uses backtracking. Also, it would be better to theoretically and empirically analyse the computational complexity of these two methods. Why is LMask, with a two-step lookahead and backtracking, reported to be faster than PIP, which also uses a two-step lookahead but without backtracking?
- The selection criteria for the backtracking budget on each problem are unclear.
- The theoretical guarantee provided for LMask relies on the assumption of an infinite computational budget, which is impractical in real-world scenarios.
- The proposed method appears to be specifically tailored for constrained nonlinear problems. Can it also be applied to constrained linear problems?

### Questions
-  Is TSL initialization the same as PIP?
- What are Greedy-C and Greedy-C in Table 9?

### Soundness
3

### Presentation
3

### Contribution
3
