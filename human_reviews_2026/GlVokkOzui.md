# Post-Processing Approach for Distributive Fairness in Multi-Class Federated Learning

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Distributive fairness is a critical concern in the application of Federated Learning
(FL) to decision making. Three concepts of distributive fairness are recently con
sidered important in FL: global, local group and client fairness. Global fairness
addresses disparities among legally protected groups across the entire population.
Local group fairness addresses disparities between protected groups within indi
vidual clients. Client fairness focuses on disparities across clients. These concepts
of distributive fairness coexist in FL and achieving one does not guarantee the
others. Most FL studies focus on only a single concept. In real-world applications,
however, different stakeholders often require fairness from different perspectives
simultaneously. Enforcing those fairness concepts inherently incurs an accuracy
cost. This paper investigates that, for a given FL setup, the maximum achievable
accuracy under various combinations of distributive fairness, i.e., all three, any two,
or just one, depending on the application. We propose a post-processing algorithm
that returns a model with the near-optimal accuracy while satisfying pre-specified
fairness constraints. Experimental results show that our algorithm outperforms
the current state of the art (SOTA) in terms of the fairness–accuracy tradeoff,
computational and communication efficiency. Code is available on Github.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a framework for multi-class FL that balances various combinations of distributive fairness, which is a pioneer work to that seeks to optimize global group, local group, and individual fairness collectively.

### Strengths
1. The theoretical analysis of this paper is solid.

2. This work is the first to propose a unified framework that seeks to optimize global group, local group, and individual fairness collectively.

### Weaknesses
1. There are only 2 clients for the experiments on Adult, and 5 clients for the experiments on HM10000. Note that in real-world FL settings, the number of clients could range from 100 to 10000. Although the paper uses 50 clients for the ACS dataset, this is only one special case not a common setting throughout the experimental evaluation.

2. The paper lacks evaluation on real-world large-scale vision/text datasets.

3. The assumption that each client has access to all sensitive groups is not very practical. Also, there may be incompatibility between client fairness, local fairness, and global group fairness.

### Questions
Please refer to the weakness part.

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
This paper studies the problem of fairness in federated learning, considering three definitions: global group fairness, local group fairness, and client fairness. The question is what is the maximum achievable accuracy under various combinations of fairness, i.e., all three, any two, or just one. The paper also proposes a post-processing algorithm to obtain a model with near-optimal accuracy while satisfying pre-specified fairness constraints. Experimental results are included on Adult, ACSPublicCoverage and HM10000 datasets.

### Strengths
-- New formulation considering three types of fairness simultaneously. While local and global fairness have been looked at, including their tradeoffs via a convex optimization, the introduction of client fairness into the formulation is quite interesting.

-- Leads to a nice convex optimization to find the optimal accuracy under constraints on the different fairness criteria.

-- Proposes a post-processing algorithm that further tries to achieve the optimal accuracy. The algorithm describes the role of each client in the execution of the strategy, showing how it will be implemented in a distributed setting.

-- Experiments are provided on 3 datasets, and multiple baselines have been considered. Adult has 2 clients while PublicCoverage has 50 clients.

### Weaknesses
-- Figures could be better for comparing the tradeoffs rather than tables. There are some figures in the Appendix, but the captions/legends were not very clear to understand how it outperforms SOTA.

-- It is a bit confusing to say it outperforms SOTA. It seems they achieve better fairness by paying a cost in accuracy (unless I am misunderstanding the table)? The benefit will be clearer in a tradeoff plot. (communication/computation benefit is acknowledged)

-- Another option is to consider a radar chart to better understand the performance comparison.

-- Some grammar issues and typos were noted. 

125. (1) We formally defines
129. We defines

A few others as well.

### Questions
Q1. Could you explain how and in what aspect these algorithms outperform SOTA (other than communication/computation benefit)? Or, is there a better way to visualize the accuracy-fairness tradeoff if there is one?

Q2. The paper has discussed extensions to some other group fairness metrics in the Appendix. I was wondering if it is possible to define a general class of fairness metrics over which this kind of a convex optimization formulation can be extended? 

Q3. This is optimal for post-processing techniques, but is it possible that there may exist other in-processing/pre-processing techniques that could lead to an even better tradeoff?

Q4. Are there any assumptions that the initial FedAvg solution needs to satisfy?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a post-processing framework for federated learning that enforces three distributive fairness notions—global group fairness, local group fairness, and client fairness—while selecting an outcome predictor that maximizes accuracy under explicit constraints.   It extends ROC-based post-processing from binary to multi-class settings by defining a multi-class ROC surface, characterizing the region under this surface via supporting hyperplanes, and proving that this region is convex and contains all achievable true-positive vectors, which yields a convex program for optimal accuracy under fairness constraints. Experimental result on three datasets reduces global/local/client disparity with competitive accuracy.

### Strengths
- The paper addresses a timely challenge in federated learning, reducing global, local, and client disparity with competitive accuracy.

- Clear formalization of three fairness notions in multi-class FL with precise constraints for global, local, and client fairness.

- Empirical efficiency: consistent reductions in three disparities with fewer communication rounds compared to baselines.

- The paper is well organized and easy to read.

### Weaknesses
- The theoretical framework presented in this paper is nearly identical to that proposed in [1], substantially diminishing its technical contribution and novelty. I would be willing to raise my score if the authors can clearly differentiate their approach and address this concern.

- Local statistics may reveal important properties of the local datasets in high-stakes scenario, even without containing any per-user information. Differential privacy alone is insufficient to prevent such leakage. The authors ought to consider encrypted computation on the server side.

- Accuracy cost can be substantial, especially when enforcing all three notions on some datasets. The Pareto frontier is referenced but not visualized in the main text.

[1] The Cost of Local and Global Fairness in Federated Learning, AISTATS 2025.

### Questions
1. Why does the theoretical approach in this paper appear overly similar to that of [1]? Is this essentially the same method applied to a different problem setting?

2. How sensitive are results to Bayesian score miscalibration?

### Soundness
3

### Presentation
3

### Contribution
3
