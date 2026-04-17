# JAPAN: Joint Adaptive Prediction Areas with Normalising Flow

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Conformal prediction provides a model-agnostic framework for uncertainty quantification with finite-sample validity guarantees, making it an attractive tool for constructing reliable prediction sets. However, existing approaches commonly rely on residual-based conformity scores, which impose geometric constraints and struggle when the underlying distribution is multimodal. In particular, they tend to produce overly conservative prediction areas centred around the mean, often failing to capture the true shape of complex predictive distributions. In this work, we introduce JAPAN (Joint Adaptive Prediction Areas with Normalising-Flows), a flow-based framework that uses density estimates for several conformal scores. By leveraging flow-based models, JAPAN estimates the (predictive) density and constructs prediction areas by thresholding on the estimated density scores, enabling compact, potentially disjoint, and context-adaptive regions that retain finite-sample coverage guarantees. We theoretically motivate the efficiency of JAPAN and empirically validate it across multivariate regression and forecasting tasks, demonstrating good calibration and tighter prediction areas compared to existing baselines. Furthermore, several density-based conformity scores showcase the flexibility of our proposed framework.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a conformal prediction framework that constructs prediction sets in the output space using the log-likelihood from normalizing flows as the conformity score. This design avoids fixed geometric assumptions and enables multi-modal, disconnected prediction regions. The authors provide a rank-preserving efficiency bound and propose a latent space estimator for computing region area. Experiments on several regression and time-series datasets show smaller regions at comparable coverage.

### Strengths
1. This paper replaces residual based conformity scores with output space likelihoods from a normalizing flow is a simple yet powerful idea that removes restrictive geometric assumptions and enables nonconvex, multi-modal prediction regions.
2. The work subsumes prior CP variants (residual, latent, posterior) as special cases, and the paper demonstrates several practical extensions including conditional andτ-adaptive versions.
3. The experiments span both regression and time series tasks, using eight diverse datasets, and consistently show smaller prediction areas at target coverage.

### Weaknesses
1. The paper provides intuitive rank-preserving arguments but stops short of offering stronger formal results, there is no finite-sample or conditional coverage bound. Besides the author should incorporate more theoretical evidence.
2. The efficiency claim hinges on the flow preserving likelihood ranking. There is no evidence that this holds across architectures, capacities, or under mild misspecification. A few diagnostic plots or ablations could make the claim much more convincing.
3. Although the experiments cover many datasets, several baselines (notably RCP, MCQR, and CopulaCPTS) are numerically unstable or poorly tuned. As a result, it’s unclear whether the observed area reductions stem from the proposed idea or from implementation details.

### Questions
See above

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
4

### Summary
The author proposes JAPAN, a multidimensional conformal prediction method that is based on Normalising flows to efficiently construct prediction regions. The author argues that current multidimensional conformal prediction methods are usually imposing restrictive geometric constraints like rectangular or spherical which often fail to capture more complex, multimodal distributions, thus leading to overly conservative sets that lose efficiency. By training a flow model, it allows for complex shape of regions which focus on the high-density areas, thus greatly improving efficiency. Besides, the author provides theoretical guarantees showing that when the density estimate is correct, the resulting prediction area is optimal. Then the author validates the performance of JAPAN by comparing it with various benchmark conformal methods.

### Strengths
Strength:
1. Provide theoretical guarantee showing that the region is optimal when the models correctly estimate the distribution
2. Comprehensive benchmarking in the experiment showing the efficiency and accuracy of the method

### Weaknesses
Weakness:
1. Lack coverage guarantees which are important in conformal prediction
2. Not comparing with other related flow-based conformal prediction method

### Questions
1. I notice there is other conformal prediction method for multidimensional time series ''Flow-based Conformal Prediction for Multi-dimensional Time Series'', how would you compare their method with yours?

2. It seems that the theory part is lacking the coverage guarantee. Would the author explain more in detail about this?

3. While establishing the optimality of the prediction region, the author assumes that the flow model can approximate the true density. Would the author explain more about when the assumption might approximately hold?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces JAPAN, a flow-based conformal prediction framework that uses density estimates (e.g., log p(y|x) from normalizing flows) as conformity scores to build compact, potentially disjoint prediction areas with finite-sample marginal coverage. It uses density-based conformity scores rather than residuals to avoid rigid geometries, enabling multimodal and context-adaptive regions.
The method constructs prediction areas by thresholding the estimated density and compute their volume efficiently via sampling in the flow’s latent space. It also provide theoretical motivation for efficiency under rank-preserving mappings and bounded misranking. The paper demonstrates versatility through variants: unconditional density, conditional on base predictions, posterior density, latent-space scoring, and a “tau-adaptive” variant to adjust thresholds with context. It conducts extensive experiments empirically evaluating across multivariate regression and time-series forecasting, reporting consistent calibration and smaller areas than strong baselines.

### Strengths
The strengths of the paper include: 
1. JAPAN’s “geometry-free” density-thresholded sets can be disjoint and adapt to x, and the framework accommodates several practical scoring variants (conditional, posterior, latent), which broadens applicability.
2. Proposition 3 gives a tractable estimator via latent-space sampling and change-of-variables of the area/volume.
3. The experimental results consistently show comparable coverage and significantly smaller areas than baselines like CQR, PCP, CONTRA, copula-based methods, etc.

### Weaknesses
The main weakness is that the theoretical assumptions in the paper are strong and somewhat informal: Propositions 1 and 2 rely on strict monotonic transformations and density homogeneity assumptions. Proposition 4 uses a misranking mass argument without explicit finite-sample rates or dimension dependence; these read more as plausibility results than tight guarantees. Also, the claimed equivalence to CONTRA is supported empirically but not formalized.

### Questions
Could you include a formal argument showing JAPAN in latent is equivalent to CONTRA, 
or CONTRA is a special case within JAPAN, and under what conditions?

What is the variance of the area estimator (Proposition 3) ? This estimator appears unstable due to the appearance
of the density function in the denominator.

Could you please clarify the second condition in the proof of Proposition 1 (page 24), which states that the conditional densities 
p(y∣x) are homogeneous across x? Additionally, could you provide some examples where this condition is satisfied?

JAPAN appears to only provide marginal guarantee. What about conditional converage?

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
4

### Summary
This paper proposes JAPAN, a novel conformal prediction framework that defines flexible prediction regions using normalizing flows. Unlike CONTRA that relies on latent spaces to form conformity scores, JAPAN directly uses the loglikelihood as scores.

This enables the prediction sets to capture arbitrary, disjoint, and non-convex, non-connected shapes, while satisfying finitesample,
distribution-free marginal coverage guarantees.

The approach is evaluated and performed well on synthetic datasets, multivariate regression. The method has also been extended to handle interesting time series tasks.

### Strengths
Presentaiton-wise, the paper is well-structured and highly informative.

Essentially, JAPAN can provide flexible prediction sets because it is based on estimating conditional densities, and normalizing flow is a good tool to achieve this.

The application to Time series is very interesting. (I am not an expert of CP used in this area of application. I do see there are multiple papers cited. Could the authors briefly talk about how the exchangeability-backed CP theory works in times series applications? Any tips on checking the validity of marginal coverage in theory and in practice? Given a new timeseries, how do I know if JAPAN produced CP are trustworthy?)

The paper provides some theory of coverage guarantees, rank preservation, and a volume estimator that supports efficient sampling.

The method is evaluated on over mulitple datasets, including time series data, and compared with many alternative methods.
JAPAN consistently achieves lower prediction volume while maintaining coverage level.

### Weaknesses
The authors talked bout how to estimate the area of the proposed CP using Monte Carlo, a sensible solution. In contrast, there was a lack of mention of how to identify points y in the proposed CP, \Gamma_\epsilon(x). That is, even if one has the analytical form of the (log) probability density function, looking for all the points with density higher than a given threshold is itself a computing challenge. (A related problem is contour finding). I am myself not an expert, but I wonder what is the computing complexity of this step, and if this could quickly become prohibitively expensive when the dimensionality of outcome increases? Is it possible to take advantage of the smooth property of \hat{p} to get more efficient solutions? How this step is achieved and visualized in the current examples and how this can be efficiently achieved in higher-dimensional examples is something worth more discussion. 

The paper does not specify whether JAPAN, CONTRA, and PCP use the same training setup. For
a fully controlled comparison, it would be helpful to clarify whether the same flow architecture and
training procedure were used across methods.

 Like most conformal methods, JAPAN does not guarantee conditional coverage in theory. But since JAPAN estimates conditional p(y|x) at its core,  perhaps some empirical studies can be added to show if it does reasonably well in conditional coverage?

 The method requires training normalizing flows and sampling extensively to approximate the region.
Runtime and efficiency were not clearly discussed in the main text.


Presentation-wise, the paper lacks consistency and rigoricity in symbol usage and mathematical languages. Most are minor issues, but worth a good round of proofreading. I am only giving three examples below, among other similar problems:

In the statement of Proposition 3, I think “z” was abused to denote both a sample of size N (z_1,\cdots, z_n) and just one instance/ in h^{-1}(z, x)? Also, both z and z_i appeared in the Area estimation formula. And mathematically, it’s not tight to write z \sim p_X(z), with the same symbol z on both sides…

The introduction, followed by a review of NF and inductive conformal prediction is generally well-done. The detailed description provided for ICP is actually for a most popular case of ICP, the split conformal. So that part could be a little misleading.

Sec 3.1 \hat{p} was not officially defined before it appears inside the conformal prediction set.

### Questions
My questions are embedded in my comments for Strengths and Weaknesses above. Thanks.

### Soundness
3

### Presentation
2

### Contribution
3
