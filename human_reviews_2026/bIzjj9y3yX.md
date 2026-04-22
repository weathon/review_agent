# Experimentation for Different Scheduling Policies on Queues: Mixed Differences-in-Q Estimators Based on Little’s Law

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
In data centers, tasks are dispatched to various servers to evenly distribute the workload. When a data center considers implementing a new scheduling algorithm, it typically conducts an A/B test prior to deployment to assess the real-world impact of this new method. However, a straightforward A/B test might be interfered with so-called ``Markovian'' interference. We utilized the Differences-in-Q estimator, as developed by Farias et al. (2022), and introduced mixed Differences-in-Q estimators grounded in Little's Law. We show that our A/B testing methods significantly reduce bias and variance when testing various scheduling policies. Extensive simulations were conducted under scenarios like non-stationary arrival rates, heterogeneous service rates, and communication delays. These simulations highlight the robustness and efficacy of our A/B testing approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of interference in A/B testing of scheduling policies in large-scale queuing systems, such as data centers. Standard A/B tests can produce biased estimates because treatment and control policies interact through shared system states (e.g., queue lengths).

The paper's main approach is to build on the Differences-in-Q (DQ) estimator of Farias et al. (2022), which reduces bias due to Markovian interference, the authors propose a mixed DQ estimator grounded in Little’s Law. The main contribution seems to be the idea of using two DQ estimators, one based on response time and one on queue length, and combine them (in a weighted way) to reduce variance. The paper provides some connections to Little’s Law. The authors perform extensive simulation results (focusing on variants of "choose the best of d queues" as policies, but adding variations to that basic setup, such as heterogeneous servers) to show that the mixed DQ estimator substantially reduces both bias and variance.

### Strengths
The new estimator appears to be a nontrivial extension of the past framework.

It's an interesting way to use LIttle's Law.  

The control-variate formulation and the explicit derivation of the optimal mixing weight is provided well from a mathematical standpoint.

The simulation results are comprehensive: they cover variations such as service heterogeneity and communication delays.  The results from simulation are good.

### Weaknesses
The paper uses a lot of terminology and setup that would benefit from more explanation. For example, the paper defines the Global Treatment Effect (GTE) as the difference in long-run average performance (response time or queue length) between two policies. But it never clearly motivates why estimating this GTE matters in an experimental sense or its specific relation to A/B testing.

Similarly, the paper asserts that the DQ estimator “addresses Markovian interference,” but doesn't really explain how.  They seem to rely on the reader being familiar with Farias et al. (which I am not).  Markovian interference (temporal dependence between treatments) is apparently the motivation for DQ estimators, and  the authors assume the reader already knows how the DQ estimator corrects for it. They neither visualize nor walk through how Q-value differences remove bias from the evolving queue state.

(Said another way: Mixing two scheduling policies in one system creates interactions that do not correspond to how either policy performs alone -- that's the Markovian interference as I understand it.  But this makes it unclear, at least to me, what “ground truth” the estimators are meant to recover, especially under the interference that mixing creates. The paper needs a clearer explanation of what the mixed experiments are intended to demonstrate.)

The artificial setup of the power-of-d policies hurts the papers in various ways. First, because it's clear what the "right answer" is the impact is lessened.  Second, it's not clear why one would ever run a mixed systems (some jobs get d1 choices, some jobs get d2) and try to reverse engineer the right answer, rather than just run each system separately (which the authors do at some point in the paper).  

The estimator’s variance minimization is derived under sample-based variance–covariance assumptions.

The truncation parameter L affects both bias and variance. The paper does not propose a principled selection rule. 

All of the experiments are simulation-based. Even a small-scale real or trace-based evaluation (e.g., on a queueing simulator or an open-source cluster model) would strengthen the applied relevance.

### Questions
In the weaknesses I think are embedded a series of questions -- how do these estimators work?  Why are we looking at mixed systems, and why is it really OK to do so (particularly in this setting)?  How should L be chosen?  etc.

Maybe one point to make is that I got intrigued when I saw the appendix where communication delay was considered, because the power-of-d choices with communication delay has stranger properties -- in particular a smaller number of choices CAN be better because of "herd behavior" explained in the Mitzenmacher paper cited in that section.  I think that would have been a much better "test case" for the main paper, because the answer it not obvious.  (And because it complicates the question of how does the "mixed process" tell you about individual processes -- I'm not convinced that 2 choices doing better in a mixed process, for instance, means it would be better in an unmixed process when delay is involved!)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies A/B testing for load-balancing (scheduling) policies in parallel-server queueing systems, where naive A/B comparisons suffer from Markovian / intertemporal interference. Building on the Differences-in-Q (DQ) estimator of Vivek Farias et al. (2022), the authors introduce a mixed Differences-in-Q estimator that combines a response-time-based Q and a queue-length-based Q using weights motivated by Little’s Law (L = λW). They derive an explicit variance expression for the mixed estimator and an analytic optimal weight α* (estimated from data) that minimizes variance; they describe estimators for the required variance/covariance terms and practical implementation details (including estimating λ and µ_i), and present extensive simulation studies (homogeneous/heterogeneous services, non-stationary arrival rates, delayed communication, power-of-d policies and others) showing markedly reduced MSE relative to naive and to single-type DQ estimators. The paper also discusses doubly-robust extensions and provides inference heuristics.

### Strengths
The paper presents a variance expression and an algebraic optimal weight α*, but does not give rigorous statistical guarantees for finite samples or for the consistency/convergence of an estimate of α̂ to α*. The Authors rely mainly on heuristic/sample formulas and Monte Carlo evidence; there is no main result (theorem) for basic properties of the estimator, like its consistency, asymptotically normality, or even that α̂ is well-behaved under the mixing experiment design. So, this seems to be a major concern; when the key aim/claim is variance reduction.


The truncation parameter (L) is used heavily and is admitted to cause non-negligible truncation bias, yet the paper gives minimal formal analysis of truncation error or principled rules to choose (L). 

====

Presentation: Overall the exposition is clear and the paper is well organized (intro, the problem setting, definitions of estimators, mixing derivation, implementation, comprehensive experiments, appendices with proofs). Figures and Tables (e.g., tables for MSE/Std Dev across λ values, violin plots) support the running text. 


Variance and SE estimates used for inference are heuristic (Section C.3). There is no careful finite-sample coverage study (e.g., empirical CI coverage over replications is not reported), so the computed CIs may be misleading.

=====

Contribution: This paper makes a solid, original contribution at the intersection of experimental design for systems and statistical estimation:

Conceptual novelty: the idea to mix queue-length and response-time Q-estimates via Little’s Law (a domain-informed control variate) is simple, elegant, and novel in the context of DQ estimation for Markovian interference.


Some other positives of the work: the estimator is easy to implement, leverages observables in typical dispatchers, and substantially reduces MSE in diverse, realistic scenarios—making it attractive to practitioners who run online experiments in data centers.


Some parts of the empirical computations are through, though: wide experiment coverage (heterogeneity, delays, non-stationarity) and ablations (power-of-5/3, doubly-robust variants) support the robustness of the proposed scheme.

### Weaknesses
The paper considers M/M/1 queue; this is a good model to start with. Perhaps, some other service times could have also been considered (M/G/1 queue). 

The chosen heuristic (L=\lfloor 30 N\lambda\rfloor) appears a bit ad hoc; the empirical advantage of mixDQ may hinge on this choice (and the direction of truncation bias seems to help the mixed estimator in some regimes). So, it is not clear how the proposed mixed estimator is generally better.

The inference/SE scheme (Section C.3) uses a simple variance estimator and asymptotic CI; some more discussion about finite-sample validity or bootstrap alternatives would help.

A few places could be improved for clarity and reproducibility: The discussion of truncation bias (choice of L and its impact) is informative but could be expanded (guidance on choosing L in practice, sensitivity plots). 

A scholarly survey is by Ward Whitt; among others, it offers a sample path based proof (a minor correction was provided later). A bit surprising that it is not referenced. 

A major comment: there are _glarring_ typos in the references, including names of Authors of the papers. The first formal proof of Little's law is due to John D.C. Little (the paper is about this relation; the result has been used since decades before, we were told); a reference lists `Little and D. C. John'. 

Another one is `OR', not `Or'. And similar ones.

### Questions
*) How sensitive are the results to the truncation length L? Can the Authors conduct a small extra computational experiment that brings out  the bias/variance trade-off (or MSE) as L varies (for at least one λ setting)?


*) As this paper considers M/M/1 queue, perhaps, the Authors comment on M/G/1 queue. 

*) Some comments on the per-arrival computational/memory overhead to compute Q̂mix online would be useful. Can the method be implemented in streaming systems with low latency?

*) Typos listed above can (and should) be addressed.

### Soundness
2

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
2

### Summary
The paper studies experimentation for evaluating scheduling policies in queueing systems (e.g., data center) under Markovian interference, where the performance of one policy can affect that of another through shared system states. Building on the Differences-in-Q estimator proposed by Farias et al. (2022), the authors introduce a mixed Differences-in-Q estimator that leverages Little’s Law to combine queue-length–based and response-time–based estimators. This method is shown theoretically and empirically to reduce the estimator’s variance while maintaining low bias. The paper provides detailed simulation studies across a variety of settings—including heterogeneous service rates, non-stationary arrivals, and communication delays—to demonstrate robustness and accuracy.

### Strengths
The paper extends the Differences-in-Q estimator to a queueing context through an elegant use of Little’s Law, yielding a mixed estimator with improved bias–variance performance. The idea of combining queue-length–based and response-time–based estimators via an analytically derived mixing weight is original within the experimentation and queueing literature, and the empirical evaluation is extensive, systematic, and convincingly supports the claims. The work is clearly motivated by real-world data-center scheduling and includes comprehensive numerical experiments. Overall, this paper effectively bridges causal inference methodology with queueing theory and offers practical insight into experimentation for dynamic systems.

### Weaknesses
While the paper is technically sound and makes a meaningful queueing-theoretic contribution, its core novelty lies primarily in the application of Little’s Law to construct a variance-reduced estimator, rather than in advancing causal inference or learning methodology. As such, the work reads more as an operations research study on experimental evaluation in queueing systems than as a contribution to causal reasoning or machine learning. In contrast, the Differences-in-Q estimator by Farias et al. (NeurIPS 2022) was framed as a general method for causal inference under Markovian interference—an issue broadly relevant to ML experimentation and policy evaluation—whereas the current paper focuses specifically on queuing systems.

### Questions
To strengthen its fit for ICLR, the authors could better articulate how the proposed estimator generalizes beyond queueing contexts or connects to broader themes such as off-policy evaluation, dynamic treatment effects, or learning under interference.

1. Could the authors elaborate on whether the Little’s Law–based variance reduction idea can extend to other Markovian environments or experimental settings with temporal interference?

2. How does the proposed estimator relate conceptually to methods in off-policy evaluation or causal inference under dependent data? For example, could it be interpreted as a form of variance reduction or control variate in the context of policy evaluation?

3. Do the authors see a pathway for connecting their approach more directly to broader machine learning themes such as dynamic treatment effects, sequential experimentation, or learning under interference?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies A/B testing for queueing-based scheduling under Markovian interference and proposes a mixed Differences-in-Q (DQ) estimator that blends a response-time DQ with a queue-length DQ using Little’s Law. They show response-time and queue-length GTEs are equivalent up to λ (Proposition 1), then derive an optimal mixing weight α via a control-variate variance calculation, and validate on supermarket-style load balancing with heterogeneous service, non-stationary arrivals, and delays.

### Strengths
1. The mixing idea is indeed interesting: Blending two DQs linked by Little’s Law and choosing α to minimize the mixed Q-variance is neat and practically motivated. The paper explicitly derives Var(Q̂_mix(α)) and the closed-form α* (Eqs. 15–16), then shows large variance drops versus plain DQ in experiments.

2. Breadth of experiments: They cover homogeneous/heterogeneous service, non-stationary λ(t), policy families beyond power-of-d, doubly-robust approaches of estimating Q, and communication delay; mixed-DQ tends to achieve the lowest MSE.

### Weaknesses
1. Any analysis for the mixing idea rather than the empirical correlation? – You do set up a principled control-variate estimator and give a closed-form α* (Eq. 16). That’s good. But it would be nice to have some intuition for the strong correlation between these two Q-functions. 

2. Any value decomposition trick to reduce dimensionality? For estimating the Q function of Q(q1, q2, ... qN), one trick is to estimate Q(q1), Q(q2), ..., Q(qN) and add them up, so-called value decomposition trick. Have you considered this approximation in the doubly-robust estimation? 

3. In addition, have you considered estimating V-value function and use Monte-carlo to estimate the advantage function (similar to what is used in TRPO), usually that is a good trick for variance reduction in advantage function/Q function estimation

4. The motivation for this problem can be further highlighted: in data center, why can't we run a switchback experiment for this?

### Questions
See the weakness above.

### Soundness
3

### Presentation
3

### Contribution
3
