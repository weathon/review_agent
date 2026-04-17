# Locally adaptive conformal inference for operator models

- Decision: Reject
- Scores: 6, 2, 2, 8

## Abstract
Operator models are regression algorithms between Banach spaces of functions. They have become an increasingly critical tool for spatiotemporal forecasting and physics emulation, especially in high-stakes scenarios where robust, calibrated uncertainty quantification is required. We introduce Local Sliced Conformal Inference (LSCI), a distribution-free framework for generating function-valued, locally adaptive prediction sets for operator models. We prove finite-sample validity and derive a data-dependent upper bound on the coverage gap under local exchangeability. On synthetic Gaussian-process tasks and real applications (air quality monitoring, energy demand forecasting, and weather prediction), LSCI yields tighter sets with stronger adaptivity compared to conformal baselines. We also empirically demonstrate robustness against biased predictions and certain out-of-distribution noise regimes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Local Sliced Conformal Inference (LSCI), a CP algorithm for neural operator models that constructing locally adaptive, function-valued prediction sets. The method cleverly uses $\Phi$-depth functions (inf of a family of linear maps) for functional conformity scores and similarity-localized calibration to generate prediction sets that adapt to heterogeneous residual distributions. The authors provide finite-sample validity guarantees under local exchangeability and demonstrate improvements over existing conformal baselines on synthetic and real-world datasets.

### Strengths
I have to admit that the operator model exposition is a little beyond me, so I'll be commenting mostly from a conformal prediction perspective. 

- It is a nice contribution to extend CP to operator learning and translate both the algorithm and guarantees to functional space. The authors have showed that this extension is nontrivial and yields better intervals than other (learning based) uncertainty quantification methods.

- Other than good direct experiment results, I find the ablation study / robustness analysis in Figures 1 and 2 and Tables 2,4,5 to be thoughtful and convincing. The authors demonstrated that coverage is stable across various choices of localizers, feature maps, projection families, and depth functions (since CP should be model-agnostic), and show meaningful advantages over baselines in biased prediction / distribution shift scenarios.

- On theory, the authors drew the connection between the localized calibration bound (Eq 13) to the tradeoff between localization and calibration data availability (Eq 14). Although more clarity can be desired (for example through experiments), the guidance is helpful for readers and practitioners.

### Weaknesses
I found the paper to be a bit difficult to follow due to my lack of background in operator learning. (might not be a weakness). 

For example, I didn't understand the significance of the statistical knockoff, why the Tukey (half-space) depth was selected, how exactly $\lambda$ is tuned to balance the trade-off, and how FPCA sampling recreates the conformal interval.  Although the authors did try to explain these choices/algorithms, the current explanations are either rushed or a little hand-wavy and could benefit from more principled explanations, through equations and examples and plots, utilizing space in the appendix. 

Another question that I had after reading this paper is, how is this UQ useful for Operator learning? The authors introduced 4 metrics, but in my experience did not explain what are the implications of each thoroughly. (I think FC and IS are intuitive for me, but how should I interpret the other two in the context of operator learning?) Maybe it is obvious for the operator learning community, but more likely it's the case that neither community knows what to do with this nice method you created. Some discussion on the properties and usefulness of the prediction sets, that needs to be created from rejection sampling and then empirical quantiles.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors propose a method for constructing prediction sets for functional data. Using the tool of $\Phi$-depths, the authors establish local $\Phi$-scores, which act as localized conformity measures on residuals. Furthermore, by employing this score along with conformal prediction methods, the authors construct prediction sets for functional data. The authors also provide a more intuitive form of the constructed prediction sets through sampling. Finally, the authors present relevant theoretical properties and experimental results. The experimental results demonstrate that the proposed method outperforms baseline methods in terms of both coverage rate and the size of the prediction sets.

### Strengths
The authors employ the tool of $\Phi$-depths to establish conformity measures suitable for functional data. Additionally, the authors utilize local information to construct the score function, thereby endowing the proposed method with enhanced robustness.

### Weaknesses
1. For methodology, in the literature on localized prediction sets (e.g., Guan, 2023[1]; Hore & Barber, 2023[2]; Barber, 2023[3]), the threshold $q_{\alpha}(f_{n+1})$ is typically defined as the $1-\alpha$ quantile of a weighted distribution $\sum_{i=1}^{n+1}w_i\delta_i$, where the weight $w_i$ assigned to each data point reflects its contribution to the construction of the prediction set. However, in line 201, the authors adopt a different quantile definition. Constructing prediction sets in this way may substantially undermine the validity of the proposed approach.

2. Theoretically, in line 223, the theoretical result presented by the authors is not supported by the literature Barber (2023)[3], as the underlying methodologies differ. Moreover, the authors do not provide a detailed proof for it.

### Questions
1 In this paper, local information is used to construct the $\Phi$-scores, yet no local information is utilized when calculating threshold $q_{\alpha}(f_{n+1})$. This differs from all existing conformal prediction frameworks (e.g., Guan, 2023[1]; Hore & Barber, 2023[2]; Barber, 2023[3]). Why not use existing localized conformal prediction methods to compute threshold $q_{\alpha}(f_{n+1})$? Is the currently used threshold a reasonable one?

2 The theoretical results presented in Line 223 are referenced to Barber (2023) [3]. However, there is a methodological discrepancy between the two works, particularly in the calculation of the threshold. Given this difference, the results from Barber (2023) [3] cannot directly support the authors' claim. It is hoped that the authors can provide a corresponding explanation or present the specific proof process.

3 Depth-based prediction sets are defined implicitly as subsets of the function space. Therefore, I am curious about how the metrics in the experimental section were calculated, particularly the band width (BW)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a locally adaptive conformal score for models that map between function spaces. The present a depth-based conformal score function using a localized empirical cumulative distribution function with weights determined by a similarity kernel. They provide provable upper bounds on the coverage gap obtained from breaking the global exchangeability assumption. They validate their method on synthetic data and real-world data (Air Quality, Energy Demand, and Weather data).

### Strengths
The authors validate their method with an intuitive upper bound for the coverage gap, which makes it clear how coverage suffers when local exchangeability is weakened. The experiments are quite robust and further strengthen their proposed method.

### Weaknesses
Main Weaknesses
* The proposed method isn’t well-motivated. The paper didn’t cite any examples where global exchangeability might break or local exchangeability might hold with functional data. 
* It’s not easy to see why depth-based scores are important to obtain local marginal coverage or tight prediction sets. In experiments, it’s clear that LSCI outperforms all the conformal baselines in the Interval Score metric, but there is no intuition behind why depth-based score can reduce Interval scores.  
* There doesn’t seem to be any experiments validating the coverage gap bound in Proposition 3, which appears to be key result.

Writing-related weaknesses
* The conformal prediction background section needs to cover local exchangeability and local adaptive conformal inference in the finite-dimensional setting. 
* The introduction has a lot of unnecessary background on neural operators and fails to motivate the problem well.

### Questions
* Can a depth-based score in a finite-dimensional setting outperform conformal baselines?

### Soundness
4

### Presentation
1

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
The authors introduce LSCI, a novel, distribution-free UQ framework for operator models, such as NOs. LSCI provides statistically rigorous, function-valued prediction sets that are locally adaptive. The authors propose local Phi scores based on Phi-depth, allowing the method to measure the centrality or typicality of a residual function relative to a local distribution of residuals, rather than just using a single scalar value. LSCI computes a local, test-specific quantile by weighting calibration samples based on their feature-space similarity to the test input.

### Strengths
1. The paper is technically sound
2. The paper has strong empirical evidence and is convincing. The experiments show it produces prediction bands that are appropriately tight in low-variance regions and wider in high-variance regions, leading to more informative and useful uncertainty estimates.
3. The theory relies on local exchangeability, a far more realistic assumption for complex, non-stationary data than the standard global exchangeability required by standard CP methods.
4. The authors provide a practical algorithm

### Weaknesses
1. The method's adaptivity hinges on the choice of the localization kernel H, the bandwidth lambda, and potentially a feature map phi. While the paper ablates these (in fig 1) and suggests tuning lambda, it offers little guidance on how to choose H or other hyperparameters and analyzes how it affects the resulting efficiency. A discussion on how to choose these parameters would be beneficial.
2. While weaker than global exchangeability, the assumption that residual distributions vary smoothly could be violated in scenarios with abrupt shifts or phase transitions. The paper does not test the method's robustness to such sharp breaks in the data-generating process. A discussion on the failure modes would be beneficial.

### Questions
1. Table 4 shows that coverage is robust to the choice of projection Phi. But how does this choice affect the tightness and shape of the prediction bands?
2. Can you provide more intuition or formal guidance on how to select the similarity kernel H and feature map for a new problem?
3. Why were Bayesian Neural Operators (BNOs) or other probabilistic operator models not included as baselines? While they are not distribution-free, they are a primary competing approach for UQ in this domain.
4. How does the method perform if the calibration set is very large? Does the need to compute n local scores for each test point become a practical bottleneck?
5. How does LSCI's coverage and tightness behave if the model is significantly mis-specified or poorly trained (i.e., the residuals are very large and structured)?

### Soundness
4

### Presentation
4

### Contribution
4
