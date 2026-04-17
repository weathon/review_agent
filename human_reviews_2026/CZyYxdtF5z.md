# Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Semidefinite programming (SDP) relaxation has emerged as a promising approach for neural network verification, offering tighter bounds than other convex relaxation methods for deep neural networks (DNNs) with ReLU activations. However, we identify a critical limitation in the SDP relaxation when applied to deep networks: a phenomenon we term interior-point vanishing, which leads to the loss of strict feasibility -- a crucial condition for the numerical stability and optimality of SDP. 
Through rigorous theoretical and empirical analysis, we demonstrate that interior-point vanishing creates a fundamental barrier to scaling SDP-based verification methods. Specifically, strict feasibility diminishes as the depth of DNNs increases. To address this issue, we design and investigate five solutions to enhance the feasibility conditions of the verification problem. Our methods successfully solve 88\% of the problems that could not be solved by existing methods, accounting for 41\% of the total. Our analysis also reveals that the valid constraints for the lower and upper bounds for each ReLU unit are traditionally inherited from prior work without rigorous justification. We find that these constraints are not only unbeneficial but, in fact, detrimental to the problem's feasibility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work analyses issues that arise when Neural Network Verification problems are solved with the help of Semidefinite Programming solvers based on interior point methods. The effectiveness of the off-the-shelf interior point solvers that are used in SDP-based verification relies on Slater's condition, but the authors find that the condition may not actually hold for neural network verification problems. When this occurs, it leads to various numerical issues for the solvers. As a consequence, the duality gap at optimality may be nonzero which would also be an issue for other (non-interior-point) solvers. The work identifies the constraints which are responsible for the interior point vanishing and the authors suggest a number of different ways of dealing with the issue. The methods put forward by the authors are backed by a theoretical underpinning but are also empirically evaluated through benchmarks which compare previously introduced SDP-based verifiers to the methods proposed by the authors. These experiments demonstrate that the proposed methods effectively resolve the issues.

### Strengths
- The identification of the issue of interior-point vanishing in SDP-based neural network verification is a novel and relevant contribution to SDP-based verification. As far as I'm aware, this is the first time this problem is identified.
- The authors propose multiple approaches for eliminating the interior-point vanishing problem. The experimental evaluation shows that they are effective at alleviating the problem and enable a higher number of verification queries to be solved in the majority of cases.
- The theoretical analysis in the problem seems sound and effectively supports the authors' argumentation

### Weaknesses
- In Appendix E the authors explain why they do not consider first-order methods in their analysis. While I do understand the issues that are mentioned, I would still argue that not being able to analyse whether the interior-point vanishing problem affects the faster first-order methods which often outperform interior-point solvers is a weakness of the work. The authors' argumentation sidesteps the most critical question of whether interior-point vanishing actually matters to the state-of-the-art first-order method solvers. The paper claims that the problem would be "even more severe" for those but provides no evidence to support these claims.
- The evaluation in the paper is based on the "success rate", i.e. whether the solver converged, and the objective gaps. However, the paper completely omits other metrics which are normally reported by neural network verification papers. The goal of verification is to either prove or disprove a robustness property at hand as quickly as possible. Besides analysing the convergence of the algorithm, the authors should therefore also compare
  - the time required to solve the verification problems. This would enable an assessment of whether the different methods put forward by the authors make the problems more computationally expensive to solve. It would also make it possible to assess the scaling of the algorithms as the network depth grows.
  - the success rate of the different verification methods. The authors analyse the objective value gaps in Figure 2, but it would be helpful to see the impacts of the gaps on the verification success rate.
- The analysis focuses too much on the convergence of the algorithms rather than the success rate of verification. Even if a solver does not converge, it might be able to provide a dual bound that is sufficient for proving the robustness of a given sample (and the failure to converge would therefore not matter). This should be discussed in the paper and additional metrics (such as the success rate mentioned above) should be reported to assess the relevance of this point.
- There is no comparison of the methods proposed by the authors with other state-of-the-art neural network verification tools. Besides this, the networks that the authors conduct their experiments on are small toy networks and do not conduct any experiments on popular benchmarks such as those from recent Verification of Neural Networks (VNNCOMP) competitions. This makes it difficult to assess the relevance of the work for verifying neural networks in practice which would normally be much larger than those evaluated by the authors. SDP-based verification seems to have been surpassed by various branch-and-bound schemes which employ looser relaxations but operate on subproblems that are much easier to solve. I doubt that this work would contribute to SDP methods being able to rival such methods (such as GCP-CROWN) and its practical relevance is therefore limited. Appendix D shows that the interior-point vanishing problem does occur in networks from prior verification works, but the authors do not provide any evidence that their proposed methods help solve the issues on these larger networks.

### Questions
- Could the authors provide verification times and success rates for their experiments?
- Could the authors comment on the scalability of their methods?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies why SDP-based verifiers for ReLU networks often lose strict feasibility (no interior point) as depth increases. It formalizes a simple diagnostic feasibility problem, analyzes structural causes (inactive neurons,  tiny per-row weight norms), and evaluates five practical strategies for alleviating this issue: $\epsilon$-SDP (soften ReLU equalities), LeakySDP, diagonal scaling (D-Scale), weight scaling (W-Scale), and removing intermediate bound constraints (B-Remove). Experiments over 50 instances per setting (10 inputs × 5 training seeds) show strict-feasibility margins shrink with depth, but $\epsilon$-SDP and especially B-Remove substantially improve solver success with modest loss in tightness.

### Strengths
The paper studies a relevant question from a novel perspective, namely strict feasibility as the real cause of SDP verification failures. It offers practical and simple fixes that restore solver stability at depth with modest loss in tightness. It also clarifies which constraints actually harm feasibility.

### Weaknesses
The writing and presentation require substantial revision. While the experiments illustrate interior-point vanishing, they do not clearly demonstrate that the proposed fixes translate into more verified instances in practice. The theoretical analysis would be more impactful if some insights were included on, for example, how to train neural networks in a way that mitigates the vanishing problem and thereby facilitates SDP verification subsequently.

### Major points

- Writing is informal or imprecise (draft style) at times, with many typos. Some examples include: line 067 (using *actual*), line 101 (problem is *true*), line 117 (boundary propagation method..), line 234 (that SDPA-GMP successfully terminated its computation), line 963 (more negative eigenvalue), line 311 (Appendix 3.4), line 397-399, line 407 (which typically help higher converging), line 356 (when $x>0$), line 471-473, etc. Many repetitions are present. For example: repeating *we first introduce*, then Proposition 3.1 (that misses some connection words), just repeats the sentence above, etc. Figures and tables are not well placed, making the reading difficult, e.g., Figure 1 is referenced almost 2 pages after being introduced, Table 3 is misplaced, etc.

- The experimental section is limited: while it illustrates the phenomenon, the comparisons are not exhaustive. In particular, no baselines that tighten SDP bounds via linear reformulations and quadratic constraints, or compute SDP-based upper bounds on Lipschitz constants, are included. Several experiments are also "underpowered"; for example, Table~4 draws conclusions from a single verification instance (one image).

### Minor points

- Why is problem (7) referred to as *MEM* on Line 785?
- Line 851 - you meant associated problem (7) for each model?
- Notation $a\pm b \operatorname{E}-k$ could be clarified to avoid ambiguity, i.e, whether the exponent applies to both $a$ and $b$ or $b$ only.
- Clarify whether success rate (Table 2) means that vanishing problem is solved or whether the instance robustness was verified.

### Questions
## Questions

1. In line 317, shouldn't you use  $(\textbf{P}[x_ix_i^\top])_{jj}$?
2. Is it reasonable to use W-Scale approach if the smallest norm is close to zero? Wouldn't dividing by $\check{w}_i$ induce more instability?
3. Your approach demonstrates that bound constraints are sometimes harmful for deeper networks? Is it true for all types of redundant constraints which are commonly used to increase tightness of SDP relaxations?
4. You only presented results with $\epsilon=\alpha=0.01$. What is the sensitivity of the interior-point vanishing with respect to $\epsilon$ and $\alpha$?
5. At fixed depth, do you observe similar results for different network widths?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The submission identifies a problem within SDP-based algorithms for incomplete neural network verification.
Specifically, the lack of strict feasibility, which the authors call "interior-point vanishing", which prevents some SDP solvers (based on primal-dual interior-point methods) from converging to the optimal solution, and causes numerical instabilities in the others.
The authors first demonstrate that the strict feasibility does not hold on a series of networks, then propose a series of remedies, whose efficacy is empirically investigates on small toy networks.

### Strengths
The main strength of the work is its novelty: the problem of strict feasibility was not identified before in the SDP-based neural network verification literature. The authors also propose a simple solution (B-remove) that successfully addresses the problem without any significant gap in optimal objective value.

### Weaknesses
The main weakness of the work lies in the specificity of its subject and on its potential impact. Indeed, the experiments only focus on the impact of strict feasibility in isolation, without ever considering whether this is any useful in practice towards obtaining an effective neural network verifier. Given the venue, this alone places the work markedly below the acceptance threshold, I believe.

More in detail:
- Some of the motivational text makes exaggerated claims. Lines 67-68: *The interior-point vanishing could be the actual reason hindering the development of the SDP-based approach in this area*. The main reason is their lack of scalability. By the time an SDP solver returns a bound, branch-and-bound based on LP relaxations will have returned a significantly tighter one. 
- In order to show the impact of the method, the authors should show that, once the strict feasibility problem is fixed, SDP-based branch-and-bound attains significantly better trade-offs between the accuracy of the bounds and the time to compute them. This is virtually the only metric in the area.
- The authors exclude first-order methods from their study because they cannot check optimality. However, given the claim that strict feasibility is important there too because of numerical stability, it is extremely important to check whether this has any practical impact.
- I think it would be quite important to evaluate the impact of the mitigation strategies on networks from previous works too (currently, those experiments are only carried out on networks trained by the authors, which are much narrower and much deeper than those used in the literature).

### Questions
- Rather than showing the average objective and the percentage of solved problems (for instance, in Table 1), I think it would be more informative to report the share of problems for which the objective was negative (i.e., strict feasibility was lacking).
- When you evaluate robustly-trained networks from previous work, the average objective is not negative for all networks. Why do you claim that this "consistently" shows interior-point vanishing?
- I am a bit confused by the claim that inactive neurons will lead to interior-point vanishing. It seems to me that for this to happen, $u_i$, which is a known pre-computed pre-activation bound, should be non-positive. But if this is the case, then such inactive neuron can and should be removed, as you also state. You elaborate that some neurons may be inactive but the loose pre-activation bounds may not spot it. But isn't the lack of strict feasibility linked to the employed numerical $u_i$ value, rather than the "true" inactivity?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a critical limitation of semidefinite programming (SDP) relaxation, which is a state-of-the-art convex relaxation method for verifying deep neural networks (DNNs) with ReLU activations, by identifying and resolving the interior-point vanishing problem. SDP relaxation is widely recognized for providing tighter bounds than other convex methods (e.g., linear programming) in DNN verification, but its scalability to deep networks has remained elusive, traditionally attributed to high computational cost.

In my view, there are four core contributions:
1. The paper rigorously demonstrates that as network depth increases, the SDP's feasible set likely lacks interior points, rendering standard interior-point methods numerically unstable and often failing to converge.
2. The paper offers both theoretical proofs and extensive empirical evidence using high-precision solvers to validate the phenomenon's prevalence. They trace interior-point vanishing to two root causes: (a) unidentifiable inactive neurons forcing diagonal entries of the SDP matrix to zero, and b. small row norms of the extended weight matrix constraining the SDP matrix’s minimum eigenvalue to near zero.
3. The paper proposes five distinct methods to restore strict feasibility: (a) $\varepsilon$-SDP (tolerating small violations of ReLU equality constraints), (b) LeakySDP (replacing ReLU with Leaky ReLU), (c) B-Remove (removing unnecessary intermediate-layer bound constraints), (d) D-Scale (diagonal scaling to improve numerical conditioning), and (e) W-Scale (weight scaling to enlarge small row norms).

The paper still falls short in terms of scalability. However, such a limitation is inherent to the verification tasks for neural networks.

### Strengths
- The paper is the first to define and analyze interior-point vanishing in DNN verification, moving beyond incremental improvements to address a foundational limitation. The theoretical link between inactive neurons/weight norms and SDP feasibility is novel and rigorously proven.
- The use of high-precision solvers (SDPA-GMP), controlled depth experiments, and benchmark network validation ensures results are reliable and generalizable. The 88% success rate on unsolvable instances is a concrete, impactful outcome.
- The methods are lightweight and compatible with existing SDP solvers (SDPA, SDPA-GMP), requiring no specialized hardware. B-Remove, in particular, is easy to implement (removing constraints) and delivers significant gains.
- The evaluation is holistic, diagnosing the problem and evaluating solutions on both feasibility and quality metrics.

### Weaknesses
- The paper only evaluates fully connected ReLU networks. Convolutional Neural Networks or transformers have distinct weight structures that may alter interior-point vanishing dynamics. Extending validation to CNNs would strengthen generalizability.
- The paper’s maximum depth is 16 layers. For modern deep networks, it remains unclear if the proposed methods (especially $\varepsilon$-SDP and B-Remove) can maintain feasibility. Evaluating 20–30-layer networks would better assess scalability.
- LeakySDP, D-Scale, and W-Scale show minimal gains but lack detailed failure analysis. For example, why does LeakySDP underperform?

### Questions
1. Your analysis is specific to ReLU networks. Do you believe the interior-point vanishing problem is an issue for SDP relaxations of networks with other activation functions (e.g., sigmoid, tanh)? Would your solutions, particularly $\varepsilon$-SDP and LeakySDP, be easily adaptable?
2. For networks with more layers, do $\varepsilon$-SDP and B-Remove still maintain feasibility? If not, is there a threshold beyond which interior-point vanishing reoccurs, and can you theoretically predict this threshold?

### Soundness
3

### Presentation
3

### Contribution
3
