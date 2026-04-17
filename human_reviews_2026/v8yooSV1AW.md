# From Fragile to Certified: Wasserstein Audits of Group Fairness Under Distribution Shift

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Group-fairness metrics (e.g., equalized odds) can vary sharply across resamples and are especially brittle under distribution shift, undermining reliable audits. We propose a Wasserstein distributionally robust framework that certifies worst-case group fairness over a ball of plausible test distributions centered at the empirical law. Our formulation unifies common group fairness notions via a generic conditional-probability functional and defines $\varepsilon$-Wasserstein Distributional Fairness ($\varepsilon$-WDF) as the audit target. Leveraging strong duality, we derive tractable reformulations and an efficient estimator (DRUNE) for $\varepsilon$-WDF. We prove feasibility and consistency and establish finite-sample certification guarantees for auditing fairness, along with quantitative bounds under smoothness and margin conditions. Across standard benchmarks and classifiers, $\varepsilon$-WDF delivers stable fairness assessments under distribution shift, providing a principled basis for auditing and certifying group fairness beyond observational data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies the instability of existing group fairness functionals under distributional shifts. It proposes a distributionally robust framework that assesses fairness in the worst case over all test distributions within a Wasserstein ball centered at the empirical data. The key concept, $\varepsilon$-Wasserstein Distributional Fairness (WDF), provides a formal criterion for certifying fairness robustness. The authors present theoretical results ensuring feasibility and consistency, showing that the proposed framework offers stable and reliable fairness evaluation beyond a single observed dataset.

### Strengths
The main contribution lies in formulating fairness auditing as a distributionally robust optimization (DRO) problem under Wasserstein uncertainty. This approach generalizes group fairness notions such as demographic parity and equalized odds within a unified framework. The paper derives a tractable reformulation via strong duality and establishes finite-sample guarantees connecting empirical and population-level fairness. The method provides formal robustness certification, addressing a key limitation of prior fairness approaches that lack distributional reliability. Overall, the work offers a clear theoretical foundation and a systematic formulation for fairness auditing under distribution shift.

### Weaknesses
While the paper presents a clear and well-motivated formulation, its degree of theoretical novelty appears somewhat limited. The proposed framework essentially extends existing Wasserstein DRO theory to a fairness functional, without introducing new theoretical or algorithmic components. Although the application of DRO to fairness auditing is conceptually meaningful, many of the theoretical results—such as dual reformulation and finite-sample guarantees—are direct consequences of established DRO literature rather than novel contributions.

Beyond the issue of limited novelty, several other practical aspects deserve further consideration:

1. The claim of improved stability under distributional shifts is not fully substantiated. Figure 2 demonstrates that WDF serves as an upper bound on fairness violation; however, this alone does not establish empirical stability (in fact, the variability between the empirical estimates and the WDF estimates appears to be quite similar). The authors motivate the introduction of WDF by showing in Figure 1 that existing fairness functionals are fragile under distributional shifts. However, this qualitative observation does not quantitatively confirm that WDF indeed yields more stable fairness estimates. Analyzing the variability of fairness functionals under resampling or perturbations would provide stronger evidence for the robustness claims beyond merely showing an upper bound. Such an analysis would offer a more comprehensive validation of the proposed framework’s robustness.

2. The framework is theoretically sound but its scalability to high-dimensional or large-scale fairness auditing tasks remains unclear. Solving Wasserstein-based optimization problems typically requires large-scale linear programming or optimal transport computation, which may be computationally demanding in practice. Moreover, the proposed formulation may not readily extend to general fairness functionals, as many fairness notions are non-differentiable or non-convex, making their integration into the Wasserstein framework non-trivial. A discussion of approximate or scalable alternatives would enhance the practical impact of the work.

### Questions
The choice of the Wasserstein radius $\delta$ is fixed (e.g., 0.01) without explicit justification or sensitivity analysis. Since $\delta$ controls the size of the uncertainty set, it plays a crucial role in determining the robustness–accuracy balance. Providing a principled selection rule or an empirical calibration strategy for $\delta$ would strengthen the practical validity and reproducibility of the results.

### Soundness
3

### Presentation
2

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
The paper introduces a Wasserstein Distributionally Robust Optimization (DRO) framework to certify group fairness under distributional shifts.

### Strengths
- The paper tackles an important problem, i.e.,  the fragility of fairness metrics under distributional changes. This is a critical issue in several real-world applications.
-  By unifying multiple fairness notions, the paper provides a flexible and comprehensive tool for fairness certifications.
-The introduction of DRUNE offers a practical method for estimating worst-case fairness, making the approach applicable to real-world scenarios.
- I did not read all the proofs in details but the results seem reasonable.

### Weaknesses
- The analysis assumes that the empirical distribution accurately represents the population, which may be unrealistic in practice, especially for rare subgroups or OOD samples. How robust are the certificates if this assumption fails? 

- Although DRUNE is proposed for efficiency, the DRO optimization may remain computationally intensive for large-scale or high-dimensional datasets. Can the method scale in practice?

- Certification relies on finite-sample empirical distributions. For small datasets or complex models, the finite-sample bounds may be loose, which is related to the phenomenon of fairness overfitting (Laakom et al., Fairness Overfitting in Machine Learning: An Information-Theoretic Perspective, ICML 2025), potentially leading to over-optimistic fairness guarantees.  Are the finite-sample bounds reliable enough for your approach to certify fairness in practice, or could they overestimate true robustness?

### Questions
See section above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the fairness assessment under distribution shifts. The core contribution of this work is the proposal of a Wasserstein distributionally robust optimization framework to certify worst-case group fairness. Specifically, they define a $\epsilon$-Wasserstein Distributional Fairness ($\epsilon$-WDF) as a robust audit target that quantifies worse-case group fairness over a set of plausible test distributions, modeled as a Wasserstein ball centered at the empirical data distribution. Recognizing the computational challenges of optimizing over infinite-dimensional sets of distributions, the authors derive tractable reformulations for $\epsilon$-WDF and its associated DRO regularizers. To address the out-of-sample problem, where only empirical data is observed, the paper establishes finite-sample certification guarantees for auditing fairness. And the theoretical insight is that the worst-case fairness estimated from the finite samples could upper bound the true worse-case disparity under shifts within an $\epsilon$-Wasserstein ball.

### Strengths
**Originality**:  
The paper's primary original contribution lies in the novel formulation of $\epsilon$-WDF. By proposing a Wasserstein distributionally robust framework, the authors introduces a new and more resilient approach to fairness assessment that directly addresses the fragility of existing metrics under distribution shifts. This re-framing of the fairness audit problem, moving towards worse-case certification over plausible distributions, is a significant contribution. While DRO and Wasserstein distances have been applied in various machine learning contexts (especially distribution shifts), their comprehensive integration specifically for certifying group fairness showcases the originality of this framework.

**Quality**:
The theoretical development is rigorous, starting from a clear problem statement and building towards a robust solution. 

**Clarity**:  
The paper is well-written and generally clear. The logical flow from defining the problem to proposing $\epsilon$-WDF, detailing its theoretical underpinnings, and outlining the estimation method is easy to follow. The mathematical notation is consistent.

**Significance**:  
The significance of this work is substantial, with implications for both theoretical research and practical applications. By providing a framework to certify worse-case worse-case group fairness, the paper offers a more reliable basis for auditing algorithmic systems. The proposed DRUNE estimator has the potential to be adopted in practical auditing and training pipelines.

### Weaknesses
- **Computational Complexity**: the authors acknowledged that the stage 1 of Algorithm 1 has an $O(d^3)$ per-point time cost and stage 2 runs in $O(N\log N)$ time. While it might be efficient for a handful of closest-point computations, it could still pose a significant bottleneck for high-dimensional feature space (especially embedded with textual or visual embeddings).
- **Comparison to Related Works**: The related work section is underdeveloped and lacks discussions with a broad range of relevant studies. In particular, it fails to explicitly discuss or compare the proposed framework with (Chen et al., 2022), which also addresses fairness certification in the target distribution under small distributional shifts.

[1] Yatong Chen, Reilly Raab, Jialu Wang, and Yang Liu. 2022. Fairness transferability subject to bounded distribution shift. In Proceedings of the 36th International Conference on Neural Information Processing Systems (NIPS '22). Curran Associates Inc., Red Hook, NY, USA, Article 819, 11266–11278.

### Questions
1. The notations of $\mathcal{S}_{\delta, q}$ and $\mathcal{I}\_{\delta, q}$ are a bit repetitive. If $f$ was defined as the absolute value, i.e.,  $f=|h(x) \phi(\cdot, \cdot)|$, then the two quantities $\mathcal{S}\_{\delta, q}$ and $\mathcal{I}\_{\delta, q}$ could be consolidated into a single, unified notation $\mathcal{S}\_{\delta, q}$. In that way, Eq. (9), (10), (11) could be furthered reduced into only one regularizer. Similarly, you won't need to swap the 0 and 1 in the afterwards expressions.  
2. The time complexity of the DRUNE estimator is $O(d^3)$ to compute $d_i$ for each data point and $O(N\log N)$ for solving the knapsack problem. However, if one wants to incorporate the estimator into a real fairness optimization problem, it may incur such time cost during each iteration (as the model parameters will change). This may not be effective for training a fair classifier.   
3. How does the Wasserstein regularizer compare to the bounds in (Chen et al., 2022)? Basically, they also provided provable guarantee of group fairness at the worst case. If we only look at the data examples within the thin boundary, it seems that the regularizers in your work could be converted into the bounds in their work.

[1] Yatong Chen, Reilly Raab, Jialu Wang, and Yang Liu. 2022. Fairness transferability subject to bounded distribution shift. In Proceedings of the 36th International Conference on Neural Information Processing Systems (NIPS '22). Curran Associates Inc., Red Hook, NY, USA, Article 819, 11266–11278.

### Soundness
4

### Presentation
3

### Contribution
3
