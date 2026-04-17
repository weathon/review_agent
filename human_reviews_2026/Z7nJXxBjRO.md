# Conformal Non-Coverage Risk Control (CNCRC): Risk-Centric Guarantees for Predictive Safety in High-Stakes Settings

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Standard Conformal Prediction (CP) guarantees that prediction sets contain the true label with high probability, but it is *cost-blind*, treating all errors as equally important---a critical limitation in high-stakes domains. We introduce **Conformal Non-Coverage Risk Control (CNCRC)**, a framework that replaces coverage frequency with direct risk control. CNCRC guarantees an upper bound on catastrophic **non-coverage risk** while actively reducing **ambiguity risk**, providing prediction sets that are both safe and usable. This is achieved through a principled decomposition of decision risk and the design of risk-weighted nonconformity scores that balance robustness with efficiency. Experiments show that CNCRC reliably satisfies strict risk constraints in adversarial settings and outperforms all baselines on a large-scale clinical benchmark. By offering practitioners a choice between maximum robustness and maximum efficiency, CNCRC provides a practical and theoretically grounded toolkit for deploying genuinely risk-aware machine learning systems in safety-critical applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a conformal framework Conformal Non-Coverage Risk Control (CNCRC) that provides an upper bound on the non-coverage risk of a model. The authors decompose the decision risk into (i) non-coverage risk, which is the expected cost suffered when the prediction set does not contain the true label, and (ii) ambiguity risk, which is defined as the cost of the worst distractor within a prediction set. The goal is to satisfy the non-coverage risk bound while reducing ambiguity risk in order to ensure robustness and efficiency in high-stakes settings. The authors show some theoretical guarantees for their method and demonstrate empirical evaluation on a clinical task.

### Strengths
The motivation behind the risk control guarantees is an important consideration for high-stakes settings. The paper explains the method clearly and the presented theorem agrees with the motivation. Additionally, the use of a real application in the experimental evaluation is appreciated.

### Weaknesses
1. Availability/accurate construction of a cost matrix is a strong assumption. Moreover, the proposed framework uses the cost in its score function and also uses it for non-coverage risk evaluation; whereas the baselines do not seem to use it (see further comments in questions below).
2. Comparison with Cost-aware CP and CRC should mention the guarantees and assumptions/requirements of the methods for fair comparison and for providing complete context to the reader.
3. Missing class-conditional baseline methods: given the motivation of reducing the risk associated with missing the true class, I strongly believe the paper should include comparison with class-conditional methods. While these methods do not provide guarantees on risk, they do provide classwise coverage guarantees and some of the methods handle long-tails as well.
4. I am concerned about the scope and scalability of the framework due to both the availability of knowledge bases in all scenarios, as well as the risk mapping using the presented pipeline.

### Questions
1. Appendix A explains cost matrix construction, however this is a fairly limited description. Who provides the “transparent, auditable rules”, and how do we ensure this matrix is accurate up to some error?
2. I don’t believe it is fair to say ‘CRC failed on efficiency’ (p7 l376) as the difference between 4.60 and 4.63 APS doesn’t seem significant. Can you clarify this?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a variant of conformal prediction called Conformal Non-Coverage Risk Control (CNCRC), which replaces the standard coverage objective with direct control of non-coverage risk, which is the expected cost when the true label is missing from the prediction set. The method also defines an ambiguity cost to quantify the potential harm from incorrect labels included in the set. The authors provide theoretical guarantees under standard conformal assumptions and evaluate the method on synthetic and real datasets.

### Strengths
1.	The paper addresses an important and practically relevant problem: controlling the cost of critical prediction failures rather than focusing solely on coverage rates.
2.	The proposed method is simple, interpretable, and easy to implement, building on conformal prediction.
3.	The writing is mostly clear, and the examples are well chosen to illustrate key ideas.

### Weaknesses
1.	The theoretical contributions are limited. The main results follow almost directly from existing conformal prediction theory, and the novelty lies primarily in the new formulations rather than new derivations.  
2.	The proposed approach is a marginal extension of existing cost-aware conformal techniques, not a fundamentally new paradigm.
3.	The proposed risk metrics are not sufficiently motivated. For example, the definition of $R_{NC}$ seems to ignore the overall structure of the prediction set. Likewise, using the maximum cost in the ambiguity metric feels somewhat arbitrary; using a minimal or average cost might yield a more intuitive measure of how “close” the prediction set is to the correct label.
4.	The experiments lack additional details such as ablation studies for the hyperparameters, error bars, or box-plots.
5.	The empirical evaluation is narrow. The datasets are few and do not convincingly demonstrate robustness or generality.

### Questions
### Questions
1.	While section 3.2 helps to construct the cost function, it is not complete. Specifically, how does one implement the risk mapping function in practice?
### minor comments
1.	The text in the figures is too small.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a reformulation of conformal prediction that aims to consider the coverage and risk control at the same time. This work joins two lines of work in conformal prediction: coverage (including cost-aware coverage) and risk control (loss defined by a cost function with prediction and ground truth). The authors provide theoretical results on the algorithm to achieve these two goals at the same time and also validate their results by experiments.

### Strengths
- The motivation of the paper is clear. The conformal set needs to cover the true label and also avoid covering other useless labels. 
- The combination of non-coverage risk and ambiguity risk provides an interpretable structure to understand the coverage and usefulness in conformal prediction.
- The main theorem provides guarantees that extend classical CP’s marginal coverage to include explicit bounds on both non-coverage and ambiguity risk.

### Weaknesses
- The connection to real-world high-stake decision-making scenarios is not clear. Specifically, the automatic derivation of the cost matrix via external knowledge bases (Sec. 3.2) is insufficiently justified. The risk mapping $\mathcal{R}$ is arbitrary and domain-dependent. There is no sensitivity analysis or justification for the chosen $\mathcal{R}$.
- It seems that some property of score function are not discussed in the theoretical results. For example, how does the choice of $s_{max}$ and $s_{sum}$ impact the results? It shows difference in the result section but was not discussed in depth.

### Questions
- See weaknesses.
- How do the authors see the work to be applied to decision-making tasks? For example, some works (such as Kiyani et al. 2025) show that the decisions with optimal worst-case performance is essentially the decisions optimized under the conformal set. How do the authors relate their results on ambiguity risk to such decision-making?

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
3

### Summary
This paper proposes a framework that extends conformal prediction to handle asymmetric costs. The method replaces the standard coverage frequency guarantee with direct control over decision risk, decomposed into non-coverage risk and ambiguity risk. They propose risk-weighted nonconformity scores and empirically validate their approach on synthetic and real benchmarks, outperforming previous CP baselines.

### Strengths
- Addresses a very important limitation of standard CP methods.
- The adversarial stress test experiment demonstrates why previous methods fail when rare but "poisonous" conditions exist.
- The method empirically outperforms the considered baselines on the large clinical task.

### Weaknesses
- The cost matrix construction relies on structured knowledge bases which limits the applicability to where such resources are not available.
- Limited empirical evaluation. The empirical evaluation focuses heavily on CP-based baselines (standard CP, cost-aware CP, and CRC). The claims would be strengthened if it included other risk-sensitive UQ methods / bayesian approaches etc, to show CNCRC's advantages beyond just the family of conformal approaches.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
