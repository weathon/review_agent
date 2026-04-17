# Diffusion and Flow-based Copulas: Forgetting and Remembering Dependencies

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Copulas are a fundamental tool for modelling multivariate dependencies in data, forming the method of choice in diverse fields and applications. However, the adoption of existing models for multimodal and high-dimensional dependencies is hindered by restrictive assumptions and poor scaling. In this work, we present methods for modelling copulas based on the principles of diffusions and flows. We design two processes that progressively forget inter-variable dependencies while leaving dimension-wise distributions unaffected, provably defining valid copulas at all times. We show how to obtain copula models by learning to remember the forgotten dependencies from each process, theoretically recovering the true copula at optimality. The first instantiation of our framework focuses on direct density estimation, while the second specialises in expedient sampling. Empirically, we demonstrate the superior performance of our proposed methods over state-of-the-art copula approaches in modelling complex and high-dimensional dependencies from scientific datasets and images. Our work enhances the representational power of copula models, empowering applications and paving the way for their adoption on larger scales and more challenging domains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel framework for copula modeling using diffusion and flow-based methods, focusing on the processes of "forgetting" and "remembering" dependencies within data. The authors propose two new models: the classification-diffusion copula and the reflection copula. The CDC is designed as an effective density estimator by framing the problem as a classification task to estimate the ratio of copula densities at different stages of a diffusion process. The Reflection Copula is a generative model built on a flow architecture that enables efficient sampling. Theoretically, both models are shown to recover the true underlying copula at optimality.

### Strengths
**S1**. The core idea of explicitly modeling the "forgetting" (diffusion to independence) and "remembering" (density estimation/generation) of dependencies provides a fresh and theoretically grounded perspective on copula modeling.

**S2**. The introduction of the classification-diffusion copula (leveraging classification for density estimation) and the reflection copula (a dedicated flow for generation) are sound and well-motivated solutions.

**S3**. The most significant strength is the demonstrated ability to achieve SOTA results on complex, high-dimensional data, including image datasets. The models clearly outperform classical (e.g., Vine) and other deep copulas, which struggle with scalability and complexity. The paper also provides thorough empirical validation across diverse datasets, using multiple metrics and visualizations to support its claims.

### Weaknesses
**W1**. While the reflection copula samples quickly, the training times for both models, especially the CDC, are reported to be very long (e.g., up to 4 hours for CDC on CIFAR). This high computational cost could hinder practical application.

**W2**. For image generation, the CDC samples are noted to be "slightly noisier and grainier" compared to the smoother samples from the reflection copula, indicating a potential trade-off between accurate density estimation and sample fidelity.

**W3**. The paper could benefit from more detailed ablation studies to isolate the contribution of specific architectural choices or loss components to the overall performance.

### Questions
No other issues; please refer to the Weaknesses section for specific points.

### Soundness
3

### Presentation
3

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
The authors propose a type of flow mapping that maps a copula to the independence copula, i.e. the authors define a mapping that forgets dependencies. The model borrows techniques inspired by diffusion models to define a mapping that forgets dependencies such that an appropriate independent copula can be reversed and transformed to new samples that map to the observed data. The authors propose two variants: one that provides likelihoods and another that only provides samples. The authors illustrate the method on a number of high dimensional datasets to showcase the extent to which the method can be applied.

### Strengths
The authors propose an interesting direction to enable high dimensional sapling from copulas, a difficult task that extends the applicability of copulas.

The method scales reliably to high dimensions which has been somewhat of an elusive property of copulas up to now.

### Weaknesses
The architectures prescribed do not preserve the necessary components to ensure a valid copula is being recovered, which can make some of these tools a bit difficult to employ in practice (e.g. if one needs specific families of marginals to be used).

The authors also miss some relevant works on deep learning and scaling copulas which should be taken into account, especially since these consider high dimensional questions using the stochastic representations [1,2,3].

[1] Inference and Sampling for Archimax Copulas, NeurIPS 2022

[2] Generative Archimedean Copulas, UAI 2022

[3] Copula Flows for Synthetic Data Generation, arxiv:2101.00598

### Questions
While the experiments on image data are quite interesting to showcase the capabilities of the copula, can the authors describe some additional sets of data that would more amenable to this method? Most image data would be well suited for diffusion models etc, but the interesting components of the copula are specifying possibly different parametric families for the marginals and it would be nice to describe that within the paper. 

What is the computational cost of this method? 

How often are the conditions of the copula empirically violated in practice?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel framework for modeling multivariate dependencies using copulas, leveraging principles from diffusion and flow-based models. The core idea is to design forward processes that progressively "forget" inter-variable dependencies while provably preserving the marginal distributions. The authors propose two such processes: one based on an Ornstein-Uhlenbeck process on a Gaussian-transformed space, and another based on a reflection process on the copula hypercube.

Based on these forward processes, the paper develops two distinct copula models. The first, the "classification-diffusion copula" (cdc), is designed for accurate density estimation. It learns the copula density by framing the problem as classifying the "time" or level of dependence in the forward process. The second, the "reflection copula," is a generative model optimized for efficient sampling by learning to reverse the reflection process. A key achievement of this work is demonstrating that these models can effectively capture complex, multi-modal dependencies and scale to very high dimensions (d > 1000), a significant advancement over existing copula methods. Empirical results on scientific datasets and images show good performance.

### Strengths
1. The core idea of modeling dependence by learning to reverse a dependence-forgetting process is both elegant and powerful. It provides a new perspective on generative modeling that explicitly disentangles marginal behavior from joint dependence.
2. The paper is built on solid theoretical ground. The authors provide proofs for the validity of their forward processes and demonstrate that their proposed models can recover the true copula at optimality. This rigor adds significant credibility to the methods.
3. The models achieve impressive results, outperforming existing copula methods on complex scientific datasets.
4. The most significant strength is the demonstrated ability to scale copula models to thousands of dimensions, as shown with the image datasets. This was previously considered infeasible for flexible copula models and represents a major leap forward for the field.
3. The paper is clear, well-structured, and easy to read, which is commendable given the technical depth of the topic.

### Weaknesses
1. The loss function for the cdc model (Theorem 5) involves a hyperparameter α that balances the cross-entropy and score-matching terms. The paper states this is chosen to balance the magnitude of the terms, which is a common but heuristic approach. The paper would be stronger with a more detailed analysis of the sensitivity to this parameter or a more principled selection method.
2. The paper correctly notes that the models are only guaranteed to be valid copulas (i.e., have uniform marginals) at optimality. While the appendix includes rank histograms as a diagnostic, a more quantitative analysis (e.g., using statistical tests for uniformity) of how close the generated samples are to being marginally uniform in practice would be beneficial.
3. The FID score for the cdc model on the digits dataset is notably worse than other methods, including the simpler reflection copula. While the authors suggest the samples are "noisier," a deeper explanation for this specific performance drop would be helpful. It raises questions about the robustness of the cdc approach on certain data types.

### Questions
The reflection copula initializes velocities from an isotropic Gaussian distribution. Have you considered alternative or learnable initial velocity distributions? Could this provide a better inductive bias for certain types of dependencies and improve performance or training efficiency?

### Soundness
4

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
The paper proposes a principled framework that unifies copula theory with continuous-time generative modeling. It defines marginal-preserving forward processes (diffusions or flows) that progressively remove dependence while keeping all marginals uniform, so every intermediate distribution remains a valid copula. The core mechanism uses an Ornstein–Uhlenbeck diffusion in Gaussian space followed by the Gaussian CDF, which preserves uniform marginals and converges to the independence copula. Reverse dynamics are learned either as a score-based diffusion (likelihood) or a deterministic flow (fast sampling). Experiments report strong performance on high-dimensional data versus vine and neural-flow baselines. Conceptually, the work reframes dependence modeling as a diffusion/flow on the copula manifold, yielding a mathematically grounded and scalable approach to complex dependencies.

### Strengths
Originality: Defining stochastic/deterministic dynamics directly on the copula manifold (uniform marginals at all times) is novel and technically 

Quality: The marginal-preserving construction (OU + CDF map) is well-motivated and, as stated, supported by a formal proposition; the dual reverse instantiations (score vs. flow) are complementary and well engineered. Empirically, results indicate improved likelihoods and scalability where vines or generic neural copulas struggle.

Clarity: Exposition and notation are clean.

Significance: Establishes a geometric generative foundation for high-dimensional copulas and clarifies a long-standing issue of marginal inconsistency in prior deep copula models.

### Weaknesses
In practice, sampling or training in copula space requires numerical integration on $[0,1]^d$ or its Gaussian transform.  The paper does not describe how boundaries $u_j \in \{0,1\}$ are handled under discretization, nor how marginal uniformity is maintained when the OU process is approximated with finite steps.  Since the theoretical result assumes continuous time and ideal mapping, discretization could introduce marginal leakage.

The paper benchmarks primarily against classical vine and neural-flow baselines but omits newer methos such as [1].
Although the paper claims linear scaling in dimension, no runtime, or memory/parameter analysis are presented compared to other methods and as the dimension increases (in addition to Table 4).

The weighting function governs how quickly dependence is forgotten. This choice may strongly influences the likelihood–sample-quality trade-off.  The paper fixes a single schedule without ablation, leaving unclear how robust the results are to this design.

 
Refs:
[1] Kamthe, Sanket, Samuel Assefa, and Marc Deisenroth. "Copula flows for synthetic data generation." arXiv preprint arXiv:2101.00598 (2021).

### Questions
Your Proposition 2 shows marginal preservation for the OU + CDF forward process. Does this extend to other forward processes (e.g., non-OU or state-dependent drifts/β-schedules)? If not, is OU intended as the recommended path?

Your results report overall log-likelihoods and FID-type metrics, but these conflate marginal and dependence contributions. Since the defining advantage of a copula-based model lies in capturing joint dependence given fixed marginals, how do you disentangle and assess the quality of dependence fit separately from the quality of marginal fit?

### Soundness
3

### Presentation
3

### Contribution
3
