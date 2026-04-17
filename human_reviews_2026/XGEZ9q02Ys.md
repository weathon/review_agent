# Decision-Theoretic Approaches for Improved Learning-Augmented Algorithms

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
We initiate the systematic study of decision-theoretic metrics in the design and analysis of algorithms with machine-learned predictions. We introduce approaches based on both deterministic measures such as distance-based evaluation, that help us quantify how close the algorithm is to an ideal solution, and stochastic measures that balance the trade-off between the algorithm's performance and the risk associated with the imperfect oracle. These approaches allow us to quantify the algorithm's performance across the full spectrum of the prediction error, and thus choose the best algorithm within an entire class of otherwise incomparable ones. We apply our framework to three well-known problems from online decision making, namely ski-rental, one-max search, and contract scheduling.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
There has been a long misalignment among different standards that are considered in learning-augmented algorithms to model the trade-off between consistency (performance with perfect prediction) and robustness (performance with arbitrarily bad prediction), including Pareto-optimality (direct trade-off between consistency and robustness) and smoothness (performance as a measurement of prediction accuracy). 
This paper provides two more metrics to resolve the misalignment from the decision-theoretic view, including a distance-based measurement and a risk-based measurement. 
They are applied to the problems of ski rental, one-max search, and contract scheduling to inspire new algorithms, which seem to perform better than previous methods.

### Strengths
+ The paper is well-motivated in that existing metrics for learning-augmented algorithms could be brittle in different ways. 
+ Under the given metrics, optimal algorithms behave better than previous SOTA on some datasets.

### Weaknesses
- To me, the major problem is that the intuitions behind the proposed metrics are vague. In fact, these metrics are somehow unnatural and may not extend well to other more complicated problems. E.g., is the "ideal solution" defined for distance-based metrics always solvable? Further, is the algorithm that maximizes the three metrics always solvable? 
- The relationship between distance-based and risk-based metrics is not addressed. How are they compared?
- The optimal algorithms corresponding to each metric are not explicitly provided in the main body. 
- The experiments seem incomplete. It is not clear whether the parameter choices are set to "exploit" the baselines. Especially, for the one-max search, why do the authors use the inputs for evaluating only the worst-case performance instead of the average-case? 

Overall, I feel that the paper can benefit from justifications on the proposed measures, a better-understandable writing, and more rigorous experiments.

### Questions
See the above "weakness" part.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
* This paper provided a systematic study of decision-theoretic approaches in learning-augmented algorithms. The system defines a unifying framework, characterizing online algorithms with advice according to their distance and risk measures. An online algorithm is said to be “ideal” if it lies on the robustness-performance Pareto frontier.
* For the continuous ski rental problem, the paper characterizes the performance ratio of ideal algorithms, and provides a CVaR-based risk analysis. For the one-max problem, the paper characterizes the ideal algorithm, gives an analytical solution for the unweighted maximum distance, and a risk-based analysis. Analysis of contract scheduling is mentioned and deferred to the appendix. 
* Empirical evaluation shows favorable performance on synthetic datasets, and evaluation on empirical data is included in the appendix.

### Strengths
* Systematic approach helps illustrate parallels between related problems, and provides a principled way to choose the "best" algorithm from a set of options.
* Risk-oriented analysis is well-motivated by the increasing application of online learning methods in high-stakes applications.
* Algorithmic results are evaluated both theoretically and empirically.

### Weaknesses
* Scope of novel contributions is unclear. Apart from the value of a unified perspective, it is not completely clear how algorithmic results compare to existing upper and lower bounds known in the literature.
* Limitations and practical applicability are not discussed explicitly.
* Empirical evaluation in the body of the paper is limited to synthetic data. Appendix D.3 seems to provide some empirical evaluation, but analysis does not seem to be conclusive (i.e., there doesn't seem to be an algorithm which performs best on all dataset, but the paper does not seem to provide any further insights regarding the root cause).
* It seems that graphically illustrating the empirical results would make them easier to interpret.

### Questions
* Is it possible to briefly summarize the novel contributions of the paper? (i.e. introduction of a unified framework, new settings previously not investigated, and relation between presented results and existing literature)
* Which underlying properties of a dataset might guide a practitioner in choosing the right performance criterion for their application?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper studies learning augmented algorithms. Algorithms are classically studied by interpolating between consistency and robustness (performance function), and in many cases cannot be compared with each other. The authors provide two measures of quantitatively comparing these algorithms, (i) distance based methods and (ii) risk measures; the latter exploits distributional properties on the quality of prediction. Existing methods include (i) pareto optimality and (ii) tolerance based methods which are special cases of the methods that the authors propose. 

The paper then analyzes 3 problems: ski rental, one-max search and contract scheduling. These examples demonstrate how their measures can practically be computed and how they can be useful in characterizing performance of algorithms.

### Strengths
-The paper is generally easy to read and follow; it is well motivated and easy to understand even for someone outside the field. 
-The choice of CVAR is one that is easy to accept, being commonly used in decision theory.
- For a theory paper, it is nice to see some experiments, though they seem to be slightly contrived.

Disclaimer: I am not from the area and cannot confidently comment on novelty nor quality.

### Weaknesses
- No major issues here from me.
- Minor criticism : Based on my understanding of the experiments, it seems that the authors are showing that by directly optimizing their metrics, better "results" are obtained based on those very metrics. This is of course unsurprising, so claims like "distance-based algorithms offer considerable improvements over the sota" aren't that fair.

### Questions
- Computing the performance ratio and optimal solutions (for the authors' metrics) does not appear to be easy, and indeed is one of the key contributions of the paper. Can the authors comment on whether there are general techniques that distance measures and risk based analysis that apply to a broader class of online problems (e.g., k-server)? This would seem to greatly strenghten the paper.

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
2

### Summary
A learning-augmented online algorithm is an online algorithm (in the Theoretical Computer Science sense) that takes a machine-learned prediction as extra input, uses it to guide decisions, and still guarantees a bounded worst-case loss when the prediction is wrong. The advice can be bad, so the algorithm is designed with two goals: Consistency (near-optimal when the prediction is accurate) and Robustness (provable cap on the competitive ratio even under bad predictions). This dual-goal structure induces a family of Pareto-optimal algorithms that are hard to compare.

This paper uses decision theory to score an algorithm’s full error-vs-performance curve against a principled yardstick. Given the r-robustness constraint, the decision-theoretic part of this paper concerns how to choose among all r-robust (r-competitive for every input) algorithms using principled objectives. They define "ideal" comparator $I_r$ as the omniscient algorithm that knows the input but is forced to be r-competitive too. 

For the first objective they consider, the "distance to the ideal", they score any r-robust algorithm $A$ by its weighted max distance and weighted average distance from $I_r$ 's performance curve over the entire prediction-error range. A user-chosen weight function encodes preferences over error regions. Such “pick the action whose loss curve is closest to an ideal benchmark” idea is what they borrow from decision theory.

For the second objective, they consider risk with CVaR; they also view the prediction as a distribution and minimize $\alpha$-consistency, which uses Conditional Value-at-Risk to weight the worst fraction of outcomes. $\alpha=0$ recovers expected performance, $\alpha \rightarrow 1$ stresses the worst-case mass in the prediction range. Given r, the goal is to find an r-robust algorithm with minimum $\alpha$-consistency. So this becomes a constrained risk minimization problem.

The paper empirically investigates how these ideas play out in classic problems such as ski rental, one-max search, and contract scheduling.

### Strengths
Pre-existing work centers on consistency/robustness trade-offs and Pareto-optimality (e.g., caching, one-way trading, search) or on tolerance windows and distributional advice. Comparison of algorithms among those Pareto optimal algorithms is a new thing. The frameworks for distance measures and CVaR-based risk are both nice, actionable, and well-posed. 

They benchmark on synthetic and real data and report better average ratios or profits than Pareto-optimal (PO) and $\delta$-tolerance baselines in ski-rental/one-max/contract scheduling for both distance measures and CVaR-risk.

### Weaknesses
I actually don't understand the benefit of going beyond Pareto, or why we wouldn't just let practitioners choose one algorithm from the Pareto set. Everyone has different preferences for trading off Consistency and Robustness; given the Pareto-optimal algorithms curve, a practitioner's preferences uniquely pinpoint one algorithm (as in economics class, where the optimal consumption position is where the indifference curve is tangent to the budget line).

### Questions
Just theoretical questions:

Can one derive lower bounds that show the paper’s optimizers are information-theoretically tight for broad classes?

Can we relax the unimodality assumption on the prediction distribution?

Can you formalize when Pareto-optimal designs are provably brittle near tiny errors and show how distance-to-ideal fixes this? Can you provide sharp transition thresholds?

### Soundness
2

### Presentation
2

### Contribution
2
