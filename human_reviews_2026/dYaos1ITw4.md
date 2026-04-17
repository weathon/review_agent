# Topological Causal Effects

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Estimating causal effects is particularly challenging when outcomes arise in complex, non-Euclidean spaces, where conventional methods often fail to capture meaningful structural variation. We develop a framework for topological causal inference that defines treatment effects through differences in the topological structure of potential outcomes, summarized by power-weighted silhouette functions of persistence diagrams. We develop an efficient, doubly robust estimator in a fully nonparametric model, establish functional weak convergence, and construct a formal test of the null hypothesis of no topological effect. Empirical studies illustrate that the proposed method reliably quantifies topological treatment effects across diverse complex outcome types.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces a novel framework for topological causal inference, combining the Potential Outcome framework with Topological Data Analysis tools in order to estimate causal effects when the outcome lies outside Euclidean spaces. The authors detail the identification and estimation steps and provides interesting real-world experiments.

### Strengths
This paper is strong, all claims are well explained and supported by theoretical and experimental results. All proofs are left in the appendix which allows for an easy read. This work seems quite novel and very useful as multiple real world examples are provided.

### Weaknesses
The biggest weakness in my opinion lies in the strength of the assumptions required.

### Questions
* I believe the assumptions to be quite strong. Could you explain how one can test these assumptions. Are these assumptions satisfied in the real-world examples provided?
 * There exists two main causal frameworks: The potential outcome framework and Pearl's DAG framework. I wonder what arguments motivated the authors to chose the potential outcome framework? Is this work compatible with the DAG framework?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a novel notion of treatment effect for causal inference, defined through algebraic topology.
Given unstructured or high-dimensional treatment outcomes, the idea is to apply a dimensionality reduction approach from 
topological data analysis to extract homology features that represent the underlying topological structure of the data.
This allows comparing potential outcomes before and after an intervention from differences in these features. 
To this end, the approach constructs persistence diagrams from the homology features (e.g., Fig 1 right) and 
compares their expected difference in silhouette score, resulting in the topological treatment effect measure (Eq. 3).  

The remainder of the work considers estimation, asymptotic properties, and empirical study of the topological TE. They study the plug-in (PI) and inverse-probability-weighted estimators (IPW and AIPW), show under which conditions they are unbiased 
and how to construct confidence bands, and derive a test for the null hypothesis of no topological TE.
Evaluations cover three benchmarks derived from real-world datasets, and show that in particular the AIPW gives a close approximation of the true silhouette function.

### Strengths
The connection between algebraic topology and causal inference is original and has a clear practical value, 
given that it allows addressing high-dimensional and/or structurally complex outcomes, for which few methods exist.
The presentation is clear and provides example illustrations of the important concepts. There is a careful theoretical analysis with independently interesting results, such as the stability bounds for weighted silhouettes, and the experiments match the theoretical results and are convincing. 
Overall, studying a topology-aware variant of treatment effect is a worthwhile contribution and could open an interesting future work direction integrating insights from TDA to causal inference.

### Weaknesses
The experimental evaluation would be stronger if a comparison to existing TE estimators were given if applicable, and the related work could be expanded with references to previous methods (esp. addressing a high-dimensional setting). Furthermore, while some comments on practical considerations are given (ln. 256-), the authors might consider including a section on implementation considerations and details in the main text or at least Appendix.

### Questions
- Related to the above weakness, can an existing TE be included in the experiments, for example to the ORBIT benchmark? 
- Given your claim in ln. 354-355 that the hypothesis test (Cor 5.5) is of primary interest to most practitioners, it would be interesting to study it empirically on the datasets.
- Can you comment on the connection between the functional ATE (Testa et al., 24) and the topological one, as alluded in line 192?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a framework for topological causal inference that defines treatment effects through changes in the topology of subjects. The authors utilize persistent homology from TDA to capture structural shifts due to intervention by a treatment, proposing topological average treatment effect defined as expected differences in power-weighted silhouettes of persistence diagrams between treated and control groups. They develop estimators, derive asymptotic properties, establish stability bounds for weighted silhouettes, and provide a formal hypothesis test for the null of no topological effect. Experiments on semi-synthetic datasets demonstrate the method's effectiveness in capturing topological causal effects.

### Strengths
**S1.** The work is novel and makes a foundational contribution that meaningfully extends causal inference to the topology of treatment units, addressing an important gap where structural changes encode scientifically relevant effects.

**S2.** The authors propose a solid theoretical framework with comprehensive analysis.

**S3.** The paper introduces a hypothesis test for no topological effect with consistency guarantees.

**S4.** Experimental validation demonstrates effectiveness on semi-synthetic data.

### Weaknesses
**W1.** The authors do not discuss scalability and computational complexity in the main paper. This is mentioned briefly in the Discussion section as a limitation, but given the claim for practical relevance, I believe this needs to be discussed in the paper.  

**W2.** The silhouette is an aggregated representation of topological features, which could potentially obscure the change in each homological feature. Ideally this should also be discussed further in the paper.

**W3.** The weighted silhouette depends on the hyperparameter $r$. There is lack of guidance in the paper on how this is/should be selected, and how it influences the results.

**W4.** While the paper is indeed novel, it is not the first bridge TDA and causal inference. To my knowledge, a recent work [1] connects the two. [1] is fairly recent and it is understandable if the authors have missed it in their literature review, but the claim in being the first to bridge TDA with causal inference should be corrected. That being said, the paper is still novel in characterizing topological causal effect.

[1] Farzam, A., Aloui, A., Tarokh, V., & Sapiro, G. Topology-Aware Robust Representation Balancing for Estimating Causal Effects. In High-dimensional Learning Dynamics 2025.

### Questions
**Q1.** What is the computational complexity of the proposed method? Can you provide empirical runtime comparisons or discuss scalability strategies for large datasets?

**Q2.** How sensitive are the results to the choice of power parameter $r$ in the weighted silhouette? Are there specific guidelines for selecting this parameter (e.g., for different applications)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper bridges topological data analysis and causal inference to estimate causal effects for non-Euclidean outcomes, like molecular graphs and point clouds. Core contributions include defining the Topological Average Treatment Effect as the expected difference in power-weighted silhouettes of persistence diagrams between treatment groups, proposing plug-in, inverse probability weighting, and augmented IPW estimators, deriving theoretical guarantees, including asymptotic distribution, Lipschitz stability, convergence, and hypothesis testing. The performances of estimators are validated on three semi-synthetic/synthetic datasets with visual evidence of AIPW outperforming alternative estimators.

### Strengths
The paper’s primary strength lies in its innovative problem formulation: reframing causal effects as treatment-induced topological changes (rather than scalar/vector shifts) and integrating topological data analysis tools to quantify these changes. This opens a new direction for causal inference on non-Euclidean data, with clear relevance to biomedicine, neuroscience, and other fields where structural outcomes matter. Experimental design uses well-characterized semi-synthetic datasets with known true topological effects, providing a controlled setting to test proposed estimators. The paper also effectively breaks down complex topological and causal concepts, with concrete examples for basic definitions and straightforward interpretation of the topological average treatment effect, enhancing accessibility for readers unfamiliar with topological data analysis.

### Weaknesses
W1: The paper suffers from inadequate theoretical rigor. It does not provide explicit bias and variance formulations for IPW and AIPW estimators. While claiming AIPW some kind of double robustness benefit? But it offers no verification, for instance, analyses of how misspecified propensity score or outcome prediction models, extreme predictions, or skewed propensity scores impact estimation performance are entirely absent. There is also no discussion of whether robustness holds under realistic conditions like moderate nuisance model misspecification or unmeasured confounders.

W2: Experimental validation is insufficient. Results rely solely on visual comparisons of silhouette curves, with no tabular reporting of quantitative metrics (such as estimated bias, mean squared error, variance, confidence interval, or coverage etc.) for estimation accuracy. Statistical tests to confirm AIPW’s performance gains over other estimators are not discussed, leaving uncertainty about reproducibility or whether results are specific to chosen random seeds.

W3: Critical robustness analyses are omitted. The paper does not evaluate how misspecifying propensity score models (e.g., linear models for non-linear true scores) or outcome prediction models (e.g., incorrect Fourier basis sizes) affects estimator bias and variance. It also fails to test performance under skewed or extreme propensity score distributions to contextualize the proposed framework’s advantages.

### Questions
Q1: Can you provide explicit mathematical formulations for the bias and variance of your proposed estimators? How do misspecifications of propensity score or outcome prediction models, or skewed propensity scores, affect these quantities? Then how do you rigorously verify AIPW’s DR property? Do extreme predictions or skewed propensity scores erode double robustness?

Q2: Will you provide tabular experimental results with quantitative metrics for your estimators across datasets? Such as estimated bias, variance, or AUC based on your outcome type? Can you confirm that AIPW’s performance gains over the others are statistically significant? Are your experimental results seed-specific? Please provide pairwise t-test results across multiple random seeds to demonstrate reproducibility.

Q3: How is the comparison between your proposed methods and other existing functional causal inference baselines (such as Testa et al., 2025) in terms of accuracy and robustness?

### Soundness
2

### Presentation
3

### Contribution
3
