# Group Fairness Under Distribution Shifts: Analysis and Robust Post-Processing

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Group fairness, as a statistical notion, is sensitive to distribution shifts, which may invalidate the fairness guarantees of classifiers trained with non-robust algorithms. In this work, we analyze randomized fair classifiers and derive upper bounds on fairness violation and excess risk under distribution shift, decomposed into covariate shift, and concept shift—changes in the distribution of group labels (and other variables considered by the fairness criterion) conditioned on the input. Our bounds are general and apply to both multi-class and attribute-blind settings; notably, we show that attribute-blind classifiers incur an additional dependency on the fairness tolerance in their excess risk, suggesting the robustness benefits of attribute awareness. Next, we propose a robust post-processing algorithm that learns fair classifiers with respect to an uncertainty set constructed by modeling the potential covariate and concept shifts, aligning with the decomposition in our analysis. We evaluate our algorithm under geographic shifts in the ACSIncome dataset, demonstrating improved fairness on unseen regions, with additional evaluations performed under noisy group labels and worst-case covariate shifts.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies how group fairness guarantees are getting worse under distribution shifts and proposes a robust post-processing algorithm to address this issue. The authors first provide a theoretical analysis for randomized fair classifiers, deriving upper bounds on fairness violation and excess risk when the test distribution differs from training. This result also showed that the bound can be decomposed into covariate and concept shifts. Thus, their results reveal that smoother classifiers are more robust to such shifts. Building on this insight, they extend the LinearPost algorithm into a robust variant that iteratively learns classifiers satisfying fairness constraints across multiple perturbed distributions within an uncertainty set modeling potential shifts. Experiments on the ACSIncome dataset under geographic shifts demonstrate that the proposed method significantly improves fairness on unseen regions—albeit at some cost to accuracy—showing that explicit robustness modeling can enhance the transferability of fairness guarantees.

### Strengths
* The paper provides a rigorous theoretical analysis of how fairness degrades under domain shifts, deriving upper bounds on fairness violation and excess risk. The result also offers insightful decomposition of the fairness degradation into covariate shift and concept shift, helping clarify which aspects of distribution change most strongly affect fairness guarantees.
* The work extends the LinearPost algorithm to a robust version capable of handling multiple worst-case distributions simultaneously, allowing fairness constraints to generalize across an uncertainty set.
• The proposed method shows strong empirical robustness on the ACSIncome (Adult) dataset, achieving improved fairness across unseen geographic regions compared to existing baselines.

### Weaknesses
* It is unclear how much novel theoretical contribution the analysis provides beyond existing results. The first part of Theorem 3.1 reproduces known bounds, and while the second part introduces a new decomposition, the resulting insights are not fully explored or clearly leveraged in the paper. Similarly, in Theorem 3.2, the first two terms of the excess-risk bound have appeared in prior analyses of domain shifts, and the new term $\varepsilon / (\alpha + \varepsilon)$ seems to have limited influence on robustness, as all three terms are dominant for the upper bound in the inequality. That is, even if the new term is increased or decreased depending on the $\alpha$, the robustness for the risk is less likely to be affected if the last two terms have large values.

* The proposed robust post-processing algorithm appears somewhat disconnected from the theoretical results. Its effectiveness heavily depends on accurately estimating the conditional probabilities of $(A, Z)$ given $X$, which may not be realistic in many domains—especially for high-dimensional or unstructured data such as text or images (e.g., CelebA or CivilComments), where predicting sensitive attributes or fairness-relevant variables is inherently unreliable.

* The experimental evaluation is limited to a single benchmark (ACSIncome), making it difficult to assess the generality of the approach. Additional experiments on diverse datasets or modalities would strengthen the empirical evidence, particularly to test the issues mentioned in the second weakness, the algorithm’s dependence on the choice of uncertainty set Q and its robustness under different types of distribution shifts.

### Questions
The robust post-processing algorithm introduces two regularization parameters, $\lambda_{\text{IW}}$ and $\lambda_{\text{CS}}$, which control the strength of covariate and concept shift modeling. How were these parameters tuned in practice, and how sensitive are the results to their values? It would be helpful to know whether they were selected through validation, grid search, or fixed heuristics, and whether the observed robustness improvements remain stable across different choices of these hyperparameters.

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
3

### Summary
This paper studies the robustness of group fairness guarantees when machine learning classifiers are deployed under distribution shifts. The authors develop theoretical bounds on both fairness violation and excess risk for randomized fair classifiers, decomposing the effects of covariate and concept shifts. They further show that attribute-aware classifiers (which explicitly include sensitive attributes) enjoy stronger robustness guarantees than attribute-blind ones.

Building on these insights, the authors propose a robust post-processing algorithm—a cutting-set method that alternates between identifying the worst-case distributional perturbation (pessimization) and re-optimizing the classifier to maintain fairness across all discovered perturbations. The algorithm extends the LinearPost framework to multiple distributions and defines the uncertainty set in terms of parameterized covariate and concept shift models.

Empirically, the method is evaluated on the ACSIncome dataset under geographic shifts, showing improved fairness generalization across regions at the cost of reduced accuracy on the training distribution.

### Strengths
Overall, the paper is well-written, logically structured, and effectively connects the theoretical analysis with algorithmic design. It provides rigorous and thorough theoretical analysis by providing general, decomposed bounds on fairness violation and excess risk under distribution shift. The cutting-set approach with a pessimization oracle is conceptually clean and theoretically justified. Convergence guarantees are provided, and the method integrates neatly with existing post-processing frameworks.

### Weaknesses
The theoretical results, including the techniques and types of bounds, are not particularly surprising, as similar findings exist in prior literature (e.g., Chen et al., 2022), including the conclusion that label shift does not affect fairness. On the practical side, the pessimization step and parameterized shift models introduce several hyperparameters ($\lambda_{IW}$, $\lambda_{CS}$) that may be difficult to tune, and scalability to large datasets or complex shifts is a bit unclear. In addition, the cutting-set iterations could impose significant computational overhead when uncertainty sets are large or fairness constraints are tight.

### Questions
1. How sensitive are the results to the hyperparameters defining the uncertainty set ($\lambda_{IW}, \lambda_{CS}$)?
Can practitioners estimate these parameters realistically without labeled target data?

2. How does your approach compare empirically and conceptually to adversarial robustness–based fairness methods (e.g., FR-Train, adversarial reweighting)?

3. The theoretical results are about randomized classifiers; in practice, deterministic deployment is often required. How do the authors envision translating this to deployed models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the problem that group fairness guarantees often fail when a machine learning model is deployed in an environment where the data distribution has shifted from the one it was trained on. The authors first provide a theoretical analysis of how distribution shifts—decomposed into covariate shift and concept shift—affect the fairness and accuracy of randomized classifiers. They derive upper bounds on both the fairness violation and the excess risk under these shifts. Based on this analysis, the paper proposes a new robust post-processing algorithm designed to learn classifiers that maintain fairness guarantees within a defined "uncertainty set" of potential distribution shifts. This algorithm iteratively finds the worst-case distribution shift that maximizes fairness violation and then retrains the classifier to be fair against this new shift. The algorithm is evaluated on the "Retiring Adult" dataset, where it is tested against geographic distribution shifts. Additional results on noisy sensitive attributes and covariate shift are included in the Appendix.

### Strengths
- While other works have analyzed fairness under distribution shifts, Theorem 3.1 which decomposed the fairness violation and excess risk bounds into distinct covariate shift and concept shift components is a novel and insightful result.
-  The introduction and preliminaries do an excellent job of motivating the problem with a clear example (Fig. 1) and precisely defining the key concepts, including the fairness criteria and types of distribution shifts. The structure of this paper is easy to follow, and the results are presented clearly. There are rich contents in the Appendix with more theoretical proofs and experimental results.
- The problem of fair classification under distribution shifts is one of the most critical and practical challenges in operationalizing fair ML. This paper provides a tangible, algorithmic solution for this.

### Weaknesses
- The proposed post-processing algorithm is an extension of LinearPost (Xian & Zhao, 2024). This reliance limits the work's originality, as the novel solution (Theorem 4.2) is a multi-distribution version of the original Theorem 4.1, and the proof techniques naturally follow the same ideas (as shown in Appendix D.2). More importantly, while the theoretical insights on decomposing shifts (Theorem 3.1) are general and broadly applicable, the proposed solution is narrowly tied to this specific post-processing framework.
- The entire optimization framework hinges on the specification of the uncertainty set $Q$, which is modeled via parameterized neural networks and regularized by a pair of $\lambda_{IW}$ and $\lambda_{CS}$. And the worse-case perturbation $q$ is optimized by gradient ascent on two-layer LeakyReLU network. This seems not making too much sense. On one hand, the distribution shifts are fully controlled by the hyper-parameters $\lambda_{IW}$ and $\lambda_{CS}$. The performance will vary significantly with different settings of $\lambda$s. In a real-world scenario, a practitioner will not have access to multiple target distributions to validate these hyperparameters. On the other hand, the distribution shift itself is modeled by the simple network $q$. This adversarial learning procedure will limit the sensitivity and robustness of the distribution shifts. If the true, real-world distribution shift is more complex than this simple network can represent, the algorithm will only be robust to a small, non-representative subset of shifts, and its fairness guarantees will fail.
- The optimization of Algorithm 1 is problematic as the number of constraints is growing with the number of iterations $T$. The linear program could become very large and computationally expensive to solve.

### Questions
1. How tight is the bound of fairness violation in Theorem 3.1?
2. In a real-world scenario where a practitioner has no access to target data, how would you do the model selection regarding different $\alpha$, $\lambda_{IW}$ and $\lambda_{CS}$ values?
3. How did you conclude that Algorithm 1 only need 5 to 20 iterations to converge? If we set $N = 1,000$ samples and $K = 2$ for binary classification, with $\tau = 0.1$ we will need $10^{2000}$ iterations to converge.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the core issue of robustness in group fairness guarantees under distribution shifts. It recognizes that statistical fairness of a model's outputs on a data distribution can be easily invalidated when a classifier trained to be fair on a source distribution is used in a new target distribution. This is empirically demonstrated with an income prediction task where a classifier fair on data from California becomes increasingly unfair and less accurate when evaluated on other states, correlating with the magnitude of the distribution shift. Then the work provides a theoretical analysis of this problem and an robust algorithm derived from this analysis.

### Strengths
1. This paper's primary strength lies in its deep theoretical analysis. The decomposition of fairness violation into covariate and concept shifts provides a clear and actionable framework. The discovery of the dependency in the excess risk bound for attribute-blind classifiers is also an important contribution. The Reviewer did not find major incorrectness within the theoretical proof.
2. The usage of the proposed method combined with post-processing is considered to be a reasonable approach. The extension of LinearPost to the multi-distribution setting is a technical contribution. The uncertainty set is then defined via parameterized models, the algorithm is general and can be adapted to various shift scenarios.
3. The proposed algorithm is directly motivated by the theoretical findings. The construction of the uncertainty set $Q$ by parameterizing covariate and concept shifts directly mirrors the decomposition in the fairness violation bound (Theorem 3.1). This connection between analysis and method makes the overall work compelling.

### Weaknesses
1. The core of the robust algorithm is in the **pessimization** phase, which finds the worst-case distribution by optimizing the parameters of neural networks via gradient ascent. This is a non-convex optimization problem nested within the main loop, and the authors acknowledge that this step may not be performed exactly and that its performance can be variable. This introduces potential instability and makes the algorithm highly sensitive to the choice of regularization hyperparameters and optimization details (learning rate, number of steps), which may be difficult to tune in practice. 
2. Following the above question, the bilevel min-max optimization framework to optimize the worst-case performance sounds not novel, such as re-weighting including but not limited to distributionally robust optimization, see [1,2,3,4]. Could the authors please distinguish the proposed framework with other worst-case optimization methods? 
3. The overall algorithm appears computationally intensive, which involves an outer loop adding new constraints, and an inner loop for the adversarial training of neural networks. While the paper notes that the algorithm typically converges in 5-20 iterations, it does not provide a direct comparison of computational costs against the baselines. This computational overhead could be a barrier to applying the method to large datasets or large scale networks, limiting the reproducibility.

[1] Online fairness-aware learning under class imbalance, ICML 2020.
[2] FIFA: Making Fairness More Generalizable in Classifiers Trained on Imbalanced Data, ICLR 2023.
[3] Fairness-aware class imbalanced learning on multiple subgroups, UAI 2023.
[4] On the Inductive Biases of Demographic Parity-based Fair Learning Algorithms, UAI 2024.

### Questions
Please see above weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
