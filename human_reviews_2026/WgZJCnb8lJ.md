# Fair Graph Machine Learning under Adversarial Missingness Processes

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Graph Neural Networks (GNNs) have achieved state-of-the-art results in many relevant tasks where decisions might disproportionately impact specific communities. However, existing work on fair GNNs often assumes that either sensitive attributes are fully observed or they are missing completely at random. We show that an adversarial missingness process can inadvertently disguise a fair model through the imputation, leading the model to overestimate the fairness of its predictions. We address this challenge by proposing Better Fair than Sorry (BFtS), a fair missing data imputation model for sensitive attributes. The key principle behind BFtS is that imputations should approximate the worst-case scenario for fairness---i.e. when optimizing fairness is the hardest. We implement this idea using a 3-player adversarial scheme where two adversaries collaborate against a GNN classifier, and the classifier minimizes the maximum bias. Experiments using synthetic and real datasets show that BFtS often achieves a better fairness x accuracy trade-off than existing alternatives under an adversarial missingness process.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a critical and underexplored problem in fair graph machine learning: the effect of adversarial missingness of sensitive attributes on fairness-aware Graph Neural Networks (GNNs). The authors argue that prior fairness methods assume that sensitive attributes are either fully available or missing completely at random (MCAR), which is unrealistic in practice. To overcome this limitation, the paper introduces Better Fair than Sorry (BFtS), a 3-player adversarial framework that jointly learns a node classifier, a fairness adversary that predicts sensitive attributes from learned embeddings, and an imputation adversary that imputes missing sensitive attributes to approximate the worst-case fairness scenario. Moreover, the authors provide theoretical analysis showing that BFtS corresponds to a min–max optimization that minimizes classifier bias under worst-case imputations.

### Strengths
1. This paper has principled theoretical grounding, clear definitions, and basic robust justification.
2. This paper proposes an innovative framework with 3-player adversarial setup elegantly unifies imputation, fairness estimation, and classification.
3. Extensive experiments on both synthetic and real-world benchmarks and the robustness is shown under varying degrees of missingness.

### Weaknesses
1. The 3-player adversarial training could be computationally heavy, particularly on large-scale graphs. A detailed runtime or memory comparison would strengthen the empirical analysis.
2. The study focuses on Demographic Parity (ΔDP) and Equality of Opportunity (ΔEQOP). It would be beneficial to include other fairness notions (e.g., Equalized Odds, Counterfactual Fairness) for completeness.
3. While the adversarial imputation approach is conceptually powerful, there is little discussion of how it performs under realistic partial observability (e.g., when only 5–10% of sensitive data is known).
4. The intuition behind the imputation adversary’s learned behavior could be further explored, perhaps via visualization or sensitivity analysis.

### Questions
1. How sensitive is BFtS to hyperparameter tuning, especially the fairness weight (α) and imputation adversary weight (β)? Could the authors provide empirical or theoretical guidance on selecting them?
2. Does the degree-bias assumption for adversarial missingness generalize to graphs with highly non-homophilous structures?
3. Could BFtS be extended to handle multi-valued or continuous sensitive attributes rather than binary ones?
4. Is there any evidence of mode collapse or instability during the 3-player adversarial training, and if so, how is it mitigated?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies graph fairness when demographic attributes are partially observed under adversarial missingness processes.
It introduces BFtS (Better Fair than Sorry), a 3-player adversarial learning framework combining a GNN classifier, a fairness adversary, and an imputation adversary. BFtS imputes worst-case sensitive attributes to make fairness evaluation robust under adversarial missingness. Both theoretical analyses and empirical results demonstrate superior tradeoffs between fairness and accuracy compared to baselines.

### Strengths
- The work identifies a realistic yet overlooked issue: adversarial missingness of sensitive attributes, which can mislead fairness evaluations in graph learning, and formalizes two adversarial missingness problems (AMAFC, AMADB) 

- Theorems 2 and 3 clearly demonstrate that BFtS minimizes worst-case bias and approximates robust fairness

- Extensive evaluations are conducted across synthetic and real-world datasets. Empirical results consistently show superior fairness–accuracy trade-offs and robustness under limited or missing sensitive data

- The paper is well organized and easy to follow

### Weaknesses
- The practicality of adversarial missingness is vague. Whether a value is missing or not seems to be difficult to control by adversaries. If the adversaries can deliberately drop some values, in this case, modifying these values seems to be a stronger adversary we can think about. It would be helpful to add more discussions on the practical scenarios of adversarial missingness in real-world cases.

- It would be helpful to add an introduction of the threat model setting in the main text.

- The bilevel optimization raises concerns about training stability. Although the authors propose the loss curve in Figure 6, it seems that Figure 6 is plotted with only a few points. It would be helpful if the training stability could be better shown in the experiments.

- It seems that the proportion of missing sensitive attributes is more than 50% of the total nodes in Figure 8. It would be helpful to see more experimental results under more stealthy attack settings.

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the problem of fair graph learning when sensitive attributes are missing under an adversarial missingness process. The authors propose Better Fair than Sorry (BFtS), a three-player adversarial framework involving a graph classifier, a bias discriminator, and a missing-value imputer. The method aims to enhance fairness robustness by simulating worst-case imputations. Both theoretical arguments and empirical evaluations are presented, demonstrating that BFtS achieves superior fairness–accuracy trade-offs on multiple graph datasets compared to existing methods.

### Strengths
1. Novel and important problem
The paper targets a realistic setting where sensitive attributes are not missing at random, which is often overlooked in existing fair graph learning literature. Formulating this as an adversarial missingness problem is both intuitive and practically meaningful.

2. Methodological soundness
The three-player adversarial design is conceptually well-motivated and integrates ideas from fairness, robust optimization, and adversarial learning into a unified framework. The training procedure is clearly described and the objectives are well defined.

3. Comprehensive experiments
The evaluation covers both synthetic and real-world datasets, reporting multiple fairness and accuracy metrics. The results consistently show that BFtS outperforms baseline methods under different missingness settings.

### Weaknesses
1. Limited theoretical depth
The theoretical results provide general insights but remain high-level. The proofs are brief and do not include convergence or generalization guarantees for the proposed min–max training objective. A more formal analysis of the optimization dynamics would strengthen the paper.

2. Comparison to related methods
While the paper compares BFtS with existing fair graph learning approaches, it could more clearly articulate how its mechanism differs from other fairness-aware imputation or robustness frameworks. The conceptual novelty may appear incremental without deeper discussion.

3. Experimental diversity
The adversarial missingness is modeled primarily through degree bias, which may not capture all possible real-world scenarios. Including other structural or attribute-based missingness patterns would make the empirical evaluation more convincing.

4. Fairness metrics and discussion
The choice of fairness metrics (Demographic Parity and Equality of Opportunity) is standard, but the paper could briefly justify why these particular measures were selected and whether the method generalizes to others.

5. Presentation details
Some notations are inconsistent between equations, and the visual presentation of a few figures could be improved for clarity.

### Questions
1. Limited theoretical depth
The theoretical results provide general insights but remain high-level. The proofs are brief and do not include convergence or generalization guarantees for the proposed min–max training objective. A more formal analysis of the optimization dynamics would strengthen the paper.

2. Comparison to related methods
While the paper compares BFtS with existing fair graph learning approaches, it could more clearly articulate how its mechanism differs from other fairness-aware imputation or robustness frameworks. The conceptual novelty may appear incremental without deeper discussion.

3. Experimental diversity
The adversarial missingness is modeled primarily through degree bias, which may not capture all possible real-world scenarios. Including other structural or attribute-based missingness patterns would make the empirical evaluation more convincing.

4. Fairness metrics and discussion
The choice of fairness metrics (Demographic Parity and Equality of Opportunity) is standard, but the paper could briefly justify why these particular measures were selected and whether the method generalizes to others.

### Soundness
3

### Presentation
3

### Contribution
3
