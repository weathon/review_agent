# PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 4, 8

## Abstract
Shapley values have emerged as a central game-theoretic tool in explainable AI (XAI). However, computing Shapley values exactly requires $2^d$ game evaluations for a model with $d$ features. Lundberg and Lee's KernelSHAP algorithm has emerged as a leading method for avoiding this exponential cost. KernelSHAP approximates Shapley values by approximating the game as a linear function, which is fit using a small number of game evaluations for random feature subsets.

In this work, we extend KernelSHAP by approximating the game via higher degree polynomials, which capture non-linear interactions between features. Our resulting PolySHAP method yields empirically better Shapley value estimates for various benchmark datasets, and we prove that these estimates are consistent.

Moreover, we connect our approach to paired sampling (antithetic sampling), a ubiquitous modification to KernelSHAP that improves empirical accuracy. We prove that paired sampling outputs exactly the same Shapley value approximations as second-order PolySHAP, without ever fitting a degree 2 polynomial. To the best of our knowledge, this finding provides the first strong theoretical justification for the excellent practical performance of the paired sampling heuristic.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes an extension of KernelSHAP by fitting higher-order polynomials to approximate the Shapley value. The results show improvement with higher order polynomials. Interestingly, the authors find that fitting a 2nd order polynomial to approximate the Shapley value is equivalent to fitting KernelSHAP with paired (antithetic) sampling.

### Strengths
-The paper is proposing a novel idea to capture non-linear interactions between features.

-The authors provide theoretical support to the proposed idea.

### Weaknesses
-KernelSHAP with paired sampling has better convergence, and simply using the unbiased KernelSHAP [1], or vanilla KernelSHAP, with more data samples will improve the accuracy as well. 

-In addition to the number of samples needed for PolySHAP, there is an additional cost to retrieve the Shapley values ($O(d·d')$), and this additional cost does not exist with KernelSHAP. 

-KernelSHAP with paired sampling is equivalent to 2-PolySHAP, and k-PolySHAP representation is equivalent to order-k Faith-SHAP. Moreover, for tabular data, RegressionMSR generally outperforms PolySHAP. So why does the user need PolySHAP?

-In the experiments, the number of examples (10 randomly selected instances) is very small. I am not convinced that we can generalize the findings given this sample size.

-The unbiased KernelSHAP [1] is missing from the comparison.


[1]-Covert, I. and Lee, S.-I. Improving kernelshap: Practical Shapley value estimation using linear regression. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130, pp. 3457–3465, April 2021.

### Questions
-Why widely used image datasets, e.g., CIFAR-10, CIFAR-100, and Imagenette, were not included in the experiments?

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
3

### Summary
The paper introduces PolySHAP, a generalization of KernelSHAP that fits higher-order interaction terms through a polynomial (technically multilinear) regression model to better approximate the underlying Shapley game. This allows the method to capture non-additive feature interactions beyond what KernelSHAP can represent. The authors prove that PolySHAP yields consistent Shapley value estimates as the number of samples grows and establish a theoretical equivalence between KernelSHAP with paired sampling and second-order PolySHAP, providing an elegant explanation for the strong empirical performance of paired sampling. Experiments on various tabular, image, and text datasets demonstrate that higher-order PolySHAP variants improve accuracy over existing estimators, with paired KernelSHAP performing comparably to 2-PolySHAP but at a lower computational cost. Overall, the paper is clearly written, the experiments are extensive, and the connection between paired sampling and interaction modeling is both interesting and useful.

### Strengths
The paper is well written, conceptually clear, and provides a theoretical and empirical investigation of Shapley value estimation. The most notable strength is the discovery of a relationship between paired sampling and polynomial (interaction-based) fitting in KernelSHAP, which offers a new and elegant theoretical explanation for a long-standing empirical observation in the literature. This connection is both novel and insightful, bridging a practical heuristic with a principled mathematical foundation.

### Weaknesses
While I found the relation between paired sampling and polynomial (or mulilinear) fitting intriguing, I believe the PolySHAP formulation is not entirely novel. 

The regression in Definition (4) corresponds exactly to the multilinear extension of cooperative games. Consequently, several theoretical results, including Theorem 4.2 follow directly from earlier works, particularly Owen’s papers on multilinear extensions [1] - See Section 2 for instance, and the Mobius representation of multilinear games that gives the same results as Theorem 4.2. This close relationship is not (explicitly) discussed in the paper. 

That being said, the idea of using multilinear extensions in explainability is not entirely new. Methods such as Faith-SHAP already build on this formulation, and the equivalence stated in Corollary 4.5 seems straightforward in light of that prior work. There are also other recent studies exploring similar multilinear formulations for feature attribution [2] - they have an exact formulation of PolySHAP via the multilinear extension of games. 


[1] https://www.jstor.org/stable/2661445
[2] https://ojs.aaai.org/index.php/AAAI/article/view/34149

### Questions
The model in Definition 4.1 effectively uses monomials (or multilinear extension) rather than full polynomials, which is the correct approach since for binary variables, higher-order powers are redundant. However, this point deserves explicit clarification, as using true polynomials would add unnecessary complexity without additional representational benefit.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper focuses on the Shapley value estimation problem in explainable artificial intelligence (XAI) and proposes a new method, **PolySHAP**, as an improvement over the widely used **KernelSHAP**. Traditional KernelSHAP employs a linear regression to approximate the relationship between model outputs and feature subsets, which limits its ability to capture feature interactions. This paper generalizes the approximation from a linear to a polynomial form, allowing the method to capture higher-order nonlinear interactions and improve the accuracy of feature attribution. In addition, the paper provides a theoretical explanation for the paired sampling mechanism in KernelSHAP, showing that it is equivalent to 2-PolySHAP.

### Strengths
* Presents a sound and theoretically grounded generalization of KernelSHAP by introducing polynomial regression to model nonlinear feature interactions, addressing a well-known limitation of additive interpretability methods.
* The connection between paired sampling and 2-PolySHAP is a notable theoretical insight, offering a formal explanation for an empirical heuristic widely used in practice.
* PolySHAP is shown to strictly subsume existing variants and guarantees convergence and consistency under reasonable assumptions.
* Supplementary experiments (Figures 4–5, Table 4) and ablation studies further support the robustness and generality of the proposed approach.

### Weaknesses
* **Limited scalability in high-dimensional or strongly non-additive settings.** Due to the combinatorial explosion, experiments with (k \geq 3) are restricted to small datasets, leaving open questions about PolySHAP’s feasibility in real-world large-scale scenarios.
* **Rapid growth in sampling and computational complexity.** The paper does not sufficiently discuss the trade-off between accuracy and computational efficiency when (k > 2).

### Questions
1. The paper states that KernelSHAP’s paired sampling is equivalent to 2-PolySHAP. Have the authors analyzed or compared the **computational efficiency** of the two methods? For example, do they exhibit similar runtime, sampling complexity, or scalability?
2. **Table 3** contains missing results. Could the authors clarify the reason for these omissions? From the current presentation, it appears that **3-PolySHAP becomes infeasible for large feature dimensions (d)**, suggesting scalability limitations. If so, what practical advantages does PolySHAP retain compared to KernelSHAP?
3. What was the **rationale behind the choice of sampling budgets and interaction orders** in the experiments? Were these hyperparameters tuned empirically, fixed a priori, or derived from theoretical considerations? Clarifying this would help assess the fairness and generality of the results.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper outlines theoretical and practical analysis of a method of obtaining more accurate shapley value estimates. The method essentially recognizes that shapley values are linear least square coefficients, so it computes higher-order pynomial leaest-sqaure estimates, then distributes higher-order coefficients down to individual variables using the shapley value-value property for projecting pure interaction functions. Comparison/euqivalencies to existing shapley work, including paired kernelShap, are given, as well as emperical convergence analysis.

### Strengths
- relevant theoretical results
- sound theoretical basis
- useful experimental results
- reasonably clear

### Weaknesses
- I expect more experinental comparisons to other existing shap-acceleration methods in terms of speed and accuacy, as such, it is difficult to assess the effectiveness of the method.

### Questions
- Please comment on the results of the experimetns sectin, as it appears that regression MSR outperforms polySHAP.
- Have you considered applying the approximate-up-then-project-down approach to other shapley-like methods that are computationally expensive and require random sampling, such as the Shapley-Taylor Interaction Index?

small issues:
- table 1 needs more visual separation from paragraph text
- 228 - "this is approach is"

### Soundness
3

### Presentation
3

### Contribution
3
