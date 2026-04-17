# Bayesian Influence Functions for Hessian-Free Data Attribution

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Classical influence functions face significant challenges when applied to deep neural networks, primarily due to non-invertible Hessians and high-dimensional parameter spaces. We propose the local Bayesian influence function (BIF), an extension of classical influence functions that replaces Hessian inversion with loss landscape statistics that can be estimated via stochastic-gradient MCMC sampling. This Hessian-free approach captures higher-order interactions among parameters and scales efficiently to neural networks with billions of parameters. We demonstrate state-of-the-art results on predicting retraining experiments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new method BIF to analyze how individual training samples affect model behavior without requiring Hessian inversion. Classical influence functions break down for deep neural networks because their loss landscapes are highly singular and BIF addresses this by replacing Hessian based inverses with covariance estimates computed over a localized Bayesian posterior using stochastic gradient MCMC sampling. This approach defines influence as the negative covariance between a sample’s loss and an observable, capturing higher-order geometric effects in the loss landscape that classical IF miss. The authors provide a scalable implementation that can handle billion-parameter models, demonstrating strong agreement with approximations like EK-FAC while offering better computational scalability and robustness.

### Strengths
1. The paper introduces a principled bayesian generalization of classical influence functions, replacing Hessian inversion with covariance estimation over a local posterior, thus making influence computation well defined even for singular deep networks. The method is architecture agnostic, works on any differentiable model. In addition, it scales to models with billions of parameters, avoiding the cubic complexity and instability of Hessian based methods like EK-FAC.
2. The authors rigorously connect BIF to classical IF through Laplace approximation, showing that local BIF recovers damped IFs as a first-order limit and extends them to higher-order geometric effects. And experiments on large models (Pythia-2.8B and Inception-V1) models demonstrate that BIF achieves good data attribution performance.

### Weaknesses
1. Although the method avoids explicit Hessian inversion, it still requires running multiple long SGLD chains and forward passes over both training and query sets, making it computationally intensive and potentially slower than classical methods.
2. The theoretical connections to classical IF rely on Laplace approximations and assume locally Gaussian posteriors, which are often invalid for highly non-convex neural loss landscapes. Hence, the higher-order generalization claim might not fully hold in real world networks.

### Questions
see weaknesses.

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
2

### Summary
The paper introduces the Bayesian Influence Function (BIF), replacing the ill-posed inverse Hessian in classical influence functions with a local posterior covariance estimated via SGLD around the trained weights. On LDS benchmarks, BIF matches strong baselines while avoiding Hessian approximations, and scaling studies show favorable wall clock time compared to EK-FAC on larger models.

### Strengths
The paper is well motivated, and the proposed Bayesian Influence Function (BIF) is natural and conceptually sound; the theoretical results clearly establish its relationship to the classical influence function. The scaling demonstrated by BIF is quite favorable.

### Weaknesses
1. It is surprising to see sign flips between BIF and EK-FAC, given that both methods are intended to approximate classical influence function. Although the authors provide some explanations, I am not fully satisfied with them. For me, this suggests one of the methods is approximating IF badly. Could the authors add comparisons to IF ground truth? Smaller-scale models are acceptable.
2. To prove practical value, please consider adding at least one downstream evaluation where BIF guides an actionable intervention and improves an objective, especially for the language model settings. The Pythia results are currently limited to qualitative visualizations, which are insufficient to establish the method's usefulness.

### Questions
1. How sensitive is BIF to hyperparameters?
2. This method seems to be distributional in nature. Can authors make some connection/comparison with the recent work on distributional data attribution [1]?

[1] Mlodozeniec, Bruno, et al. "Distributional Training Data Attribution." arXiv preprint arXiv:2506.12965 (2025).

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
4

### Summary
This paper introduces local Bayesian Influence Functions (BIF) for the training data attribution task. BIF estimates data influence through Bayesian covariance sampling using stochastic gradient Markov chain Monte Carlo. The method works for large models, avoids numerical instability, and provides fine-grained, per-token attribution. Experiments on vision and language models show that BIF matches or outperforms some existing methods while being more scalable and flexible.

### Strengths
1. The paper introduces the local Bayesian Influence Function with both theoretical grounding and empirical validation.

2. The proposed estimator is architecture-agnostic and scales to billions of parameters.

3. The authors provide thorough implementation and experimental details, which support reproducibility.

### Weaknesses
1. The experimental evaluation could be strengthened by including more comparison methods and a broader range of datasets.

2. A robustness analysis or ablation study on key hyperparameters would further enhance the credibility of the results.

3. A brief description of the datasets used would improve the paper’s clarity and completeness.

### Questions
Why was Pythia-2.8B chosen over more widely used or newer LLMs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces the Local Bayesian Influence Function (BIF), a novel approach to Training Data Attribution (TDA) for deep neural networks. It addresses the limitations of classical influence functions, which rely on Hessian inversion and struggle with the non-invertible Hessians and high-dimensional parameter spaces of modern deep learning models. The proposed BIF replaces Hessian inversion with covariance estimation over the local posterior, leveraging stochastic-gradient MCMC sampling for efficient computation. The method is architecture-agnostic and scales to models with billions of parameters. Empirical results demonstrate that BIF achieves state-of-the-art performance in predicting retraining experiments and offers computational advantages over classical methods like EK-FAC, particularly for fine-grained and per-token attribution tasks.

### Strengths
- The paper provides a principled extension of classical influence functions to the Bayesian setting, addressing the challenges posed by non-invertible Hessians in deep neural networks. The proposed BIF method also scales efficiently to models with billions of parameters, making it suitable for modern large-scale deep learning architectures.

- The BIF enables per-token influence computation, which is particularly useful for language models and provides insights into semantic relationships between tokens. Moreover, unlike classical methods, BIF can be applied to any differentiable architecture, including attention-based models, making it more versatile.

- The paper provides a thorough comparison with state-of-the-art methods like EK-FAC, highlighting the advantages and trade-offs of the BIF approach.

### Weaknesses
-  While BIF avoids the high up-front fitting cost of EK-FAC, its computational cost scales with the number of posterior draws, which can be expensive for large datasets or models.

- While the BIF provides interpretable results in many scenarios, there are cases, particularly in language modeling, where the most influential samples are not immediately intuitive.

- The authors are requested to provide some initial experiments on more modern deep learning models and slightly larger models.

- The paper primarily focuses on comparing BIF with EK-FAC, leaving other TDA methods like TRAK and GradSim less explored in depth.

### Questions
Please see the Weaknesses.

Overall, the paper makes a significant contribution to the field of AI interpretability and training data attribution by introducing a scalable, theoretically grounded method that addresses key limitations of classical influence functions. The empirical results are promising, and the method's ability to handle large-scale models and provide fine-grained attribution is a notable advancement.

### Soundness
4

### Presentation
4

### Contribution
3
