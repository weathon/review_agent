# Decision Making under Imperfect Recall: Algorithms and Benchmarks

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
In game theory, imperfect-recall decision problems model situations in which an agent forgets information it held before. They encompass games such as the "absentminded driver" and team games with limited communication. In this paper, we introduce the first benchmark suite for imperfect-recall decision problems. Our benchmarks capture a variety of problem types, including ones concerning privacy in AI systems that elicit sensitive information, and AI safety via testing of agents in simulation. Across 61 problem instances generated using this suite, we evaluate the performance of different algorithms for finding first-order optimal strategies in such problems. In particular, we introduce the family of regret matching (RM) algorithms for nonlinear constrained optimization. This class of parameter-free algorithms has enjoyed tremendous success in solving large two-player zero-sum games, but, surprisingly, they were hitherto relatively unexplored beyond that setting. Our key finding is that RM algorithms consistently outperform commonly employed first-order optimizers such as projected gradient descent, often by orders of magnitude. This establishes, for the first time, the RM family as a formidable approach to large-scale constrained optimization problems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper makes two primary contributions. First, it introduces an extensive benchmark suite for decision-making under imperfect recall, with problem categories motivated by real-world concerns. Second, it provides a comprehensive empirical evaluation of algorithms for finding equilibria in these problems. The key finding is that the family of RM algorithms, particularly RM⁺, consistently and significantly outperforms standard first-order optimizers like PGD and its variants in terms of convergence speed, often by orders of magnitude. This establishes RM as a highly effective, parameter-free approach for this important class of problems.

### Strengths
1. The creation of the first large-scale benchmark for imperfect-recall decision problems is a significant and timely contribution. The problems are well-motivated and will serve as a valuable resource for the community, filling a critical gap.

2. The evaluation across 61 tasks and multiple algorithms is thorough, and the performance difference is so pronounced that it provides a clear, actionable takeaway for practitioners in this domain.

3. The paper is well-written and structured. It effectively introduces the necessary background, the benchmark design, and the algorithms, making it accessible to a broad audience.

### Weaknesses
1. The most significant weakness is the lack of a theoretical explanation for RM's superior performance. As noted, RM's efficacy in perfect-recall games is well-documented; however, its robust performance in the more complex and non-convex setting of imperfect-recall optimization demands deeper investigation. The paper would be greatly strengthened by providing theoretical intuition or analysis that explains why the RM update rule is so effective here, moving beyond the empirical demonstration [1-3]. 

2. The benchmark generation methods, while excellent for ensuring variety and avoiding cherry-picking, are largely random and syntactic. They may not fully capture the strategic, domain-specific abstractions [4, 5] or automated abstractions [6-8]. This could limit the immediate applicability of the findings to some real-world problems where imperfect recall is intentionally designed for efficiency.

3. Some novelty has been claimed. RM performs well in imperfect-recall game has been stated in [9], which asserted "CFR applied to any abstraction belonging to our general class results in a regret bound not just for the abstract game, but for the full game as well". Imperfect-recall abstraction performance has been tested in poker and randomly generated games with support [10, 11].

4. The paper does not propose a new algorithm but rather performs a systematic comparison of existing ones. While the empirical finding is highly valuable, the novelty is centered on the benchmarking and evaluation, not on methodological innovation.

If these issues can be resolved, I would be willing to raise my rating.

[1] Christian Kroer, Tuomas Sandholm. Imperfect-Recall Abstractions with Bounds in Games. EC 2016

[2] Christian Kroer, Tuomas Sandholm. A Unified Framework for Extensive-Form Game Abstraction with Bounds. NeurIPS 2018

[3] Nicolas S. Lambert, Adrian Marple, Yoav Shoham. On equilibria in games with imperfect recall. Games and Economic Behavior 2019

[4] Sam Ganzfried, Tuomas Sandholm. Potential-Aware Imperfect-Recall Abstraction with Earth Mover’s Distance in Imperfect-Information Games. AAAI 2014

[5] Noam Brown, Sam Ganzfried, Tuomas Sandholm. Hierarchical Abstraction, Distributed Equilibrium Computation, and Post-Processing, with Application to a Champion No-Limit Texas Hold'em Agent. AAMAS 2015

[6] Jirí Cermák, Viliam Lisý, Branislav Bosanský. Automated construction of bounded-loss imperfect-recall abstractions in extensive-form games. IJCAI 2020

[7] Boning Li, Zhixuan Fang, Longbo Huang. RL-CFR: Improving Action Abstraction for Imperfect Information Extensive-Form Games with Reinforcement Learning. ICML 2024

[8] Boning Li, Longbo Huang. Efficient Online Pruning and Abstraction for Imperfect Information Extensive-Form Games. ICLR 2025

[9] Marc Lanctot, Richard G. Gibson, Neil Burch, Michael Bowling. No-Regret Learning in Extensive-Form Games with Imperfect Recall. ICML 2012

[10] Jirí Cermák, Branislav Bosanský, Viliam Lisý. An Algorithm for Constructing and Solving Imperfect Recall Abstractions of Large Extensive-Form Games. IJCAI 2017

[11] Jirí Cermák, Branislav Bosanský, Michal Pechoucek. Combining Incremental Strategy Generation and Branch and Bound Search for Computing Maxmin Strategies in Imperfect Recall Games. AAMAS 2017

### Questions
1. The strong performance of RM in both perfect and imperfect-recall settings suggests a fundamental advantage. Can you provide any intuition or preliminary analysis for why the RM update rule is so well-suited to constrained optimization over the product-of-simplex domain, even in non-convex problems like these?

2. How do you envision the community building upon your benchmark? Do you believe your conclusions about RM would hold for more strategically-generated, domain-specific imperfect-recall abstractions, and would you encourage the creation of such instances within your framework?

3. Could you test different imperfect-recall algorithms in common games? Could you test how different degrees of forgetting affect the solution?

### Soundness
2

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
The paper shows that variants of regret matching may be a suitable heuristic for solving imperfect recall decision problems. It argues why solving these problems is important, proposes several scalable benchmarks, and compares the achieved value and computations requirement of the several iterative algorithms to the optimal solution computed by Gurobi. They conclude that on a range of domains, RM based solutions perform close to optimum (where computable) in a fraction of time.

### Strengths
* The paper explains reasonably well why imperfect recall problems are important.
  * It presents a sufficiently wide range of experiments to convince me that RM based methods are worth considering for this problem.
  * It promises public release of a range of problems, that may become a common benchmark for future solvers
  * No need to tuning parameters compared to GD

### Weaknesses
* Even the local convergence is supported only empirically
  * Negative examples where local optima are bad are hidden in the appendix and referenced to a different paper instead of clearly discussing the limitations
  * The main results in Table 1 are hard to interpret


Further suggestions:

The paper at places seems to make claims about general constraint polynomial optimization, not just imperfect recall games (L99,L255, 278), which looks confusing to me. Either make clear what wider class of optimization problems you suggest RM should work in, or stick to imperfect recall decision making.

There are more advanced algorithms for solving imperfect recall games, such as [A]. It would be nice to comment on why those cannot be used, instead of "Gurobi". Also, for the "Gurobi" method, I would appreciate the exact formulation you used to solve the problem, in the appendix. This (and related) papers also introduce a parametrized class of imperfect recall random games, which can be related to yours. 

[A] Čermák, Jiři, Branislav Bošanský, and Michal Pěchouček. "Combining incremental strategy generation and branch and bound search for computing maxmin strategies in imperfect recall games." Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems. 2017.

I believe it would be better to clearly describe the examples of imperfect recall decision making problems, where RM fails in the main body of the text.

It is very hard for me to read Table 1. Can you add some aggregate statistics over the columns or something that would clearly compare the algorithms without staring at the table for 10 minutes?

### Questions
Why are Gurobi times for Rand-*k not reported? Why are OGD times not reported for some games, where the value is reported?

I did not fully get the domain motivation of the detection domain. Can you elaborate more on a specific real world situation modeled by the game?

### Soundness
3

### Presentation
3

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
This paper introduces the first benchmark suite for decision-making under imperfect recall, a setting in which an agent may forget previously acquired information. The authors design three parametric classes of problems—simulation games (for AI safety and testing), subgroup detection under privacy constraints, and random decision problems—implemented using the LiteEFG framework. They evaluate a range of first-order optimization algorithms for computing Causal Decision Theory (CDT) equilibria, including projected gradient descent (PGD), AMSGrad, and a newly introduced family of regret matching (RM) algorithms for nonlinear constrained optimization. Experiments across 61 instances show that RM-based algorithms, especially RM+, consistently outperform gradient-based optimizers in convergence speed (often by orders of magnitude) while achieving comparable or better objective values. The paper provides both theoretical motivation and large-scale empirical results, establishing RM algorithms as strong general-purpose optimizers beyond their traditional game-theoretic use.

### Strengths
S1. Novel benchmark suite fills a clear gap in the literature.

S2. Bridges two research communities —game theory and optimization—by adapting regret matching to nonlinear constraints.

S3. Extensive experiments across diverse problem types and scales, with clear reporting of performance metrics and runtime.

S4. Strong practical insight: RM+ shows remarkable stability and speed, potentially influencing solver design for large imperfect-information systems.

S5. Clarity and reproducibility: methodology, code availability, and hyperparameter details are well-documented.

### Weaknesses
**W1.** Lack of theoretical guarantees for RM convergence in general nonconvex constrained settings.  While empirical evidence is compelling, a formal proof (even partial or asymptotic) would strengthen the claim that RM can act as a “first-order optimizer.”

---

**W2.** Limited analysis of failure cases.  The appendix briefly mentions instances in which RM converges to poor local optima, but a deeper investigation of when and why this occurs would enhance understanding.

---

**W3.** Connection to learning-based methods (e.g., policy gradient or actor-critic under imperfect recall) is only briefly mentioned; integrating these discussions could expand relevance to the broader ICLR audience.

---

**W4.** Benchmark diversity: all benchmarks are tabular; extending to function approximation or continuous domains is left for future work.

### Questions
**Q1.** How sensitive are RM variants to initialization or stochastic noise compared to gradient-based methods?

**Q2.** Could the authors theoretically link RM’s updates to mirror descent or other known first-order schemes under certain convexity assumptions?

**Q3.** In the simulation benchmark, does varying the simulation–deployment ratio qualitatively change RM’s convergence behavior?

**Q4.** Would integrating RM with adaptive step sizes (like AMSGrad) further improve performance?

**Q5.** Could the benchmark be extended to multi-agent imperfect-recall games (beyond single-player decision problems)?

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
The paper aims to study algorithms for imperfect recall decision making problems.  They introduce three parameterized benchmark environments and propose to assess algorithms by how well they compute CDT equilibria, which amount to finding first order optimal, the very thing projected gradient descent algorithms do.  They then explore the use of regret matching for nonlinear constrained optimization and find that it outperforms projected gradient descent in this class of problems.

### Strengths
The paper is tackling an interesting problem.  The empirical evaluation is quite large, spanning not only multiple, but also quite different environments.

The observations on the performance of RM algorithms in these problems is interesting, and deserves further attention and investigation.

### Weaknesses
It is not entirely clear to me what is the contribution of the paper. It seems to be spread across presenting benchmarks, arguing for a particular evaluation metric, and then a set of evaluations that suggest RM methods should get more attention.  I'm not sure it does the first two that effectively.  I'm not sure the latter is sufficiently impactful, given the long history of imperfect recall being explored in games (using exactly RM-based algorithms; see work coming out of the AI poker competitions from 2007 to 2017).

Some of the motivation for imperfect recall "will play a key role in AI" are unconvincing.  Solutions to imperfect recall seek to cope with the forgotten information, but may effectively do so by using other remembered cues to recover that information.  Many of the examples aim to force forgetting as a principle (for privacy or simulation testing adherence purposes), where an agent's recovery of that information would be harmful to the mechanism's purpose itself.  I do think imperfect recall is an important thing to study purely from the framing of bounded rationality and long-lived agents.  Remembering one's entire past is clearly impractical beyond very short horizon settings.

Given that the paper purports to introduce benchmarks for evaluation, I would expect more discussion of their choice of evaluation metric.  They seem to settle on CDT equilibria without much justification beyond the fact that optimality (or approximation of it) is NP-complete.  That may be, but then that doesn't make just any tractable evaluation metric a good choice.  In fact, maybe it suggests the endeavor is hopeless, and we would be better off finding tractable subclasses (see work on well-formed games; Lanctot et al., ICML 2012; Kroer et al., EC 2016).  Why would reaching a CDT equilibria be desirable or sufficient?

Algorithm 1 is incomplete.  What is GetX?  What is Step?

"for generic nonconvex optimization problems such as ours"... Huh?  How is your subclass of non-convex optimizations somehow generic?!

I don't understand the subgroup detection problem at all.  What's the sequence of decisions?  What's the utilities?

"The confidence intervals represent the 30th and 70th percentile"  Does that mean these represent 40% confidence intervals?!?  I'm not sure you have enough iterations to be expecting to produce accurate confidence intervals, but would use is a 40% confidence interval anyway?  It seems like you're just trying to show your plots have tight shaded regions!  With under 15 samples, showing dots for all the samples in addition to the mean, or showing bars giving min and max would be more fair and honest.

Another related concept to CDT equilibria is "action deviations" from the literature on regret algorithms (Morrill, ICML 2021).  It may be worth relating to that work.

### Questions
You mentioned the Gurobi solver is guaranteed to find the global optimum if it terminates, yet it failed to do so in a couple of your problems. How can that be the case?  There's also times it terminated but you didn't report a running time.

### Soundness
3

### Presentation
2

### Contribution
2
