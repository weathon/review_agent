# Measuring Invariance in Representation Learning: A Robust Evaluation Framework

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Distribution shifts challenge reliable deployment even when in-distribution accuracy is high. Invariant representation learning aims to mitigate this challenge by learning feature spaces that remain invariant across diverse out-of-distribution (OOD) scenarios.  However, a critical gap exists in directly and efficiently evaluating the true invariance of learned representations across varied environments. To address this, we introduce DRIC, a novel and computationally efficient criterion designed for the direct assessment of invariant representation performance. DRIC establishes a formal link between the conditional expectation of invariant predictors and environmental diversity through the density ratio, providing a theoretically sound and practical evaluation framework. We validate the effectiveness and robustness of DRIC through extensive numerical experiments on both synthetic and real-world datasets, demonstrating its utility in quantifying and comparing the invariance of learned representations, ultimately contributing to the development of more robust machine learning models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DRIC, a new framework for quantitatively measuring invariance in learned representations under distribution shifts. DRIC provides a normalized, environment-agnostic metric that directly assesses how invariant a model’s latent features are across environments by leveraging density-ratio estimation between distributions. The authors develop theoretical guarantees showing DRIC’s convergence and derive bounds relating invariance and predictive accuracy, highlighting the trade-off between them. Experiments on synthetic, benchmark (DomainBed), and real-world datasets demonstrate that DRIC not only correlates with out-of-distribution generalization performance.

### Strengths
- The work builds on and unifies ideas from IRM, VREx, DANN, and causal invariance frameworks, positioning DRIC as a generalizable tool for the invariant learning community.

- DRIC not only serves as a diagnostic metric but can also be integrated as a regularization term to improve invariant learning, showing practical value beyond evaluation.

- The authors provide code to support reproducibility.

### Weaknesses
- The motivation is questionable. The authors argue that over-enforcing invariance can degrade accuracy, implying that model selection should balance both accuracy and invariance. However, their framework performs selection solely based on the proposed DRIC metric, which measures invariance alone. This raises the concern that models with trivially invariant yet uninformative representations (e.g., random or collapsed embeddings) could be favored. Moreover, the paper does not provide clear practical guidance on how DRIC should be used in conjunction with accuracy or other performance metrics to achieve a meaningful trade-off, leaving its real-world applicability ambiguous.

- While the paper emphasizes the need for measuring invariance, it does not clearly justify why existing invariance proxies (e.g., risk variance across domains, domain classifier accuracy) are insufficient or how DRIC fundamentally improves beyond them.

- The density-ratio estimation step can be unstable, especially in high-dimensional feature spaces or when environment overlap is weak; the paper does not explore robustness to these conditions.

- Using DRIC as a regularizer (Eq. 9) is conceptually interesting but lacks detailed design choices or analysis on optimization stability and convergence.

- Experiments rely mainly on standard benchmarks (SEM, DomainBed, small real datasets) and do not test DRIC on large-scale, complex, or high-dimensional real-world tasks (e.g., vision, language).

- There is no experiment demonstrating failure modes or conditions under which DRIC may produce misleading scores.

- DRIC values are reported without clear thresholds or interpretability guidelines. It is unclear what constitutes “strong” or “weak” invariance in practice.

### Questions
- Since DRIC only measures invariance, how should practitioners use it in conjunction with accuracy to avoid over-selecting trivial invariant representations? Can the authors provide quantitative or heuristic guidelines for choosing this trade-off in real applications?

- Could the authors clarify more concretely when DRIC would be more informative or reliable than existing invariance proxies such as risk variance (VREx) or domain classifier accuracy? In what practical scenarios (e.g., fairness auditing, model debugging, or OOD selection) would DRIC provide unique insights that these existing metrics cannot?

- How sensitive is DRIC to inaccurate density-ratio estimation, especially in high-dimensional or weak-overlap environments? It would be helpful if the authors could provide empirical sensitivity analysis or propose stabilization strategies for density-ratio estimation.

- What range or magnitude of DRIC values should be considered indicative of “strong” versus “weak” invariance? Is there any empirical calibration or threshold that practitioners can use for interpreting DRIC in a standardized way?

- How well does DRIC scale to high-dimensional data (e.g., image or text embeddings) where density-ratio estimation can be computationally difficult?

- Since the paper mentions DRIC’s potential relevance for fairness and trustworthiness, could the authors elaborate or provide an example of how DRIC might help identify bias or environment-specific leakage in real-world data (e.g., gender, hospital site)?

- Could the authors provide more quantitative correlation analyses between DRIC and OOD accuracy across different methods and datasets? Demonstrating a strong statistical relationship would strengthen confidence in DRIC as a reliable diagnostic measure.

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
4

### Summary
This paper proposes the Density-Ratio–based Representation Invariance Criterion (DRIC), a new quantitative metric for evaluating representation invariance across environments. Under the “expected invariance” assumption, DRIC measures differences in conditional expectations between environments after density-ratio reweighting, yielding an environment-independent and scale-invariant criterion. The authors develop a plug-in estimator with theoretical consistency and provide computational complexity analysis. Extensive experiments on both synthetic data and real benchmarks demonstrate that DRIC correlates well with out-of-distribution (OOD) performance and can also serve as a regularization term for training invariant models.

### Strengths
(1) Proposes a clear and theoretically grounded metric (DRIC) that directly quantifies representation invariance through environment reweighting, achieving environment-independence and scale invariance.
(2) Provides practical plug-in estimators with rigorous theoretical guarantees, including estimator consistency and information-theoretic lower bounds.
(3) Demonstrates an efficient and scalable implementation with analyzed time and space complexity, making DRIC practical for multi-environment evaluation.
(4) Presents extensive empirical validation across synthetic data, the DomainBed benchmark (CMNIST), PACS, and multiple real-world datasets, showing a consistent correlation between DRIC scores and OOD performance.
(5) Extends DRIC beyond evaluation: incorporating it as a regularization term during training further improves invariant model performance, underscoring its methodological value.

### Weaknesses
(1) The robustness boundaries of environment specification and density-ratio estimation are not clearly analyzed. If environments are misspecified or imbalanced, the estimated \( \hat{Q}^{\phi} \) may be biased (see Concluding Remarks; Remark 3.1). This issue limits the reliability of DRIC under weak overlap or heterogeneous domain conditions.  
(2) Although Appendix B.4 compares DRIC with several invariance testing methods such as KMaxIC, subgroup-based tests, and domain-classifier accuracy, the main text provides little discussion on their complementary roles or applicability boundaries. A clearer methodological positioning of DRIC among existing invariance evaluation tools would strengthen the contribution.  
(3)Provably invariant learning without domain information, ICML2023, this paper should also be compared.
(4) The paper does not include interpretability or feature-attribution analyses, making it unclear which features or environments most influence the DRIC score. Such analysis could enhance understanding of how invariance manifests in learned representations.

### Questions
(1) Regarding environment specification, could the authors elaborate on how sensitive \( \hat{Q}^{\phi} \) is to environment imbalance or mis-specification? In particular, how does the metric behave when environments have weak overlap or partially disjoint covariate supports (see Remark 3.1 and Concluding Remarks)?  
(2) The comparison with group-invariance testing methods (Appendix B.4) is insightful. Could the authors clarify under what conditions DRIC should be preferred over formal statistical tests such as KMaxIC or subgroup-invariance tests? What are the trade-offs between DRIC’s continuous, quantitative nature and the hypothesis-testing frameworks used in those approaches?  
(3) The concluding section mentions plans to calibrate \( \hat{Q}^{\phi} \) with uncertainty quantification. Could the authors elaborate on how uncertainty estimation would be incorporated, and how this calibration might influence DRIC’s interpretability and practical usage?

### Soundness
3

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
4

### Summary
This paper proposes DRIC, a normalized, environment-agnostic metric for assessing how invariant a representation is across domains. DRIC measures variation in conditional predictions across environments via density-ratio reweighting, comes with a simple classifier-based estimator, and has consistency guarantees. The metric is then used for model selection and as a regularizer to favor invariant representations. Experiments on synthetic data and a DomainBed setting illustrate that lower DRIC aligns with better out-of-domain behavior.

### Strengths
- DRIC provides a direct, normalized criterion for representation invariance, improving on ad-hoc proxies and making scores comparable across datasets and methods. 
- The classifier-based density-ratio estimator is simple to plug in, and the paper outlines a usable path for model selection and training with an invariance penalty. 
- Consistency of the estimator and an information-theoretic bound linking invariance to label–environment dependence make the metric interpretable and well-grounded. 
- DRIC separates "how invariant" a learned representation is from raw accuracy, helping analyze failures of OOD generalization beyond test error alone.

### Weaknesses
- Remark 3.1 treats disjoint supports as a degenerate case, but realistic domain shifts often have partial overlap (some features absent in some domains). How DRIC behaves and should be interpreted under partial support mismatch is underexplained. 
- There is limited comparison to prior invariance-oriented selection or regularization criteria, for example CLOvE [1] and related calibration-based OOD selection. Without this, DRIC’s added value over known metrics is hard to gauge. 
- DomainBed [2] is mentioned, but only CMNIST is used; broader, more "real" settings (e.g., PACS, Office-Home, TerraIncognita) and stronger, recent baselines are missing. 
- A light editorial pass would help resolve minor ambiguities and make key design choices more transparent, for example, standardizing references like "2.2" in "invariance property 2.2", (line 60-65). In addtiion, The normalization and practical interpretation of absolute DRIC values could be clearer, including guidance for thresholds and how to balance DRIC versus accuracy in model selection.

### Questions
- How should DRIC be computed and interpreted when environments share only a subset of support (e.g., features like color absent in sketch but present in photo). Can you detect and adjust for partial overlap rather than treating near-zero ratios as merely degenerate. 
- Could you compare DRIC against CLOvE[1] w.r.t model selection and training with regularization form, reporting correlation with OOD performance and showing when DRIC is more predictive. 
- DomainBed breadth: Will you add evaluations on other DomainBed Datasets (PACS, Office-Home, or TerraIncognita) with recent SOTA (e.g. [3],[4]) baselines to demonstrate utility beyond CMNIST. 
- Could you clarify whether "invariance property 2.2" refers to Definition 2.2? Plus, when selecting models, how should practitioners trade off DRIC versus accuracy? Can you provide practical guidance or target ranges for DRIC that indicate sufficient invariance in common settings? 

If these points, especially partial support handling, stronger metric baselines, or broader DomainBed evaluation, are clarified or strengthened, I would be inclined to raise my score.

[1] Wald, Yoav, Amir Feder, Daniel Greenfeld, and Uri Shalit. "On calibration and out-of-domain generalization." Advances in neural information processing systems 34 (2021): 2215-2227.

[2] Gulrajani, Ishaan, and David Lopez-Paz. "In Search of Lost Domain Generalization." In International Conference on Learning Representations.

[3] Nguyen, Toan, Kien Do, Bao Duong, and Thin Nguyen. "Domain generalisation via risk distribution matching." In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2790-2799. 2024.

[4] Kim, Taero, Subeen Park, Sungjun Lim, Yonghan Jung, Krikamol Muandet, and Kyungwoo Song. "Sufficient invariant learning for distribution shift." In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 4958-4967. 2025.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces **DRIC**, a measure for assessing invariance in representation learning across environments. The measure is computationally efficient and easy to estimate.

### Strengths
The paper is clearly written and easy to follow, with an intuitive presentation of the proposed DRIC metric.

### Weaknesses
### Unclear contributions \& problem framing
The paper claims to propose the first “environment-agnostic” metric for measuring representation invariance, but it never clearly states what specific problem it solves or how it differs from existing approaches such as domain-classifier accuracy, HSIC, or MMD. In short, the novelty and unique contribution are unclear and the paper fails to explain what gap DRIC actually fills.

### Weak literature review
1. No formal comparison/contrast with mutual-information/HSIC-style penalties, or domain classifier metrics have been made. I believe a comparison table to highlight the gaps and what unique challenges were addressed in this paper would be helpful. 

### Logical inconsistency around “DRIC = 0”
* Under the condition $Y\perp E\mid \phi(X)$, DRIC can be zero. However, In Section 3.3, the paper asserts that DRIC=0 is unattainable. Please reconcile this contradiction. 
* I do understand that Invariance implies DRIC = 0. However, it's not clear the inverse holds. How do we guarantee that DRIC = 0 implies invariance? Authors implicitly implies that DRIC = 0 encourages Invariance (e.g., DRIC as a penelizer), however, it's not clear to me, given that DRIC is a proxy for expectation-level invariance. 


### Theorem 3.5
Authors claimed that DRIC is invariant to the scale of Y ("Scale-invariant"). However, COV(Y, E) is variant in scale of Y, and can be arbitrarily large by rescaling. I think this is contradictory. As it stands, Theorem 3.5 is likely false or at least misstated. 

### Weak Simulation 

The simulations are quite weak overall. They only test DRIC as a post-hoc metric not as part of a new/novel learning method. The authors simply compare DRIC values among a few existing algorithms (ERM, IRM, VREx, GroupDRO, IB-IRM, LISA) without introducing or analyzing any new representation-learning procedure. The results mostly confirm well-known trends, such as ERM gives the highest “invariance violation,” and VREx or IB-IRM the lowest. No novel insights are provided/demonstrated by this work beyond prior findings.

### Questions
Please see weakness.

### Soundness
2

### Presentation
2

### Contribution
1
