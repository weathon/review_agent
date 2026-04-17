# Robust Generalized Schr\"{o}dinger Bridge via Sparse Variational Gaussian Processes

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 8

## Abstract
The famous Schr\"{o}dinger bridge (SB) has gained renewed attention in the generative machine learning field these days for its successful applications in various areas including unsupervised image-to-image translation and particle crowd modeling. Recently, a promising algorithm dubbed GSBM was proposed to solve the generalized SB (GSB) problem, an extension of SB to deal with additional path constraints. Therein the SB is formulated as a minimal kinetic energy conditional flow matching problem, and an additional task-specific stage cost is introduced as the conditional stochastic optimal control (CondSOC) problem. The GSB is a new emerging problem with considerable room for research contributions, and we introduce a novel Gaussian process pinned marginal path posterior inference as a meaningful contribution in this area. Our main motivation is that the stage cost in GSBM, typically representing task-specific obstacles in the particle paths and other congestion penalties, can be potentially noisy and uncertain. Whereas the current GSBM approach regards this stage cost as a noise-free deterministic quantity in the CondSOC optimization, we instead model it as a stochastic quantity. Specifically, we impose a Gaussian process (GP) prior on the pinned marginal path, view the CondSOC objective as a (noisy) likelihood function, and infer the posterior path via sparse variational free-energy GP approximate inference. The main benefit is more flexible marginal path modeling that takes into account the uncertainty in the stage cost such as more realistic noisy observations. On some image-to-image translation and crowd navigation problems under noisy scenarios, we show that our proposed GP-based method yields more robust solutions than the original GSBM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a GP prior GSBM algorithm for robust generalization. This allows a sparse variational free energy formulation that ensures effective posterior path measures using GP-based mean and covariance derivation on pinned marginals. The noticeable benefits are robustness in noisy scenarios and generality for various SDE settings. The authors presented various experiments demonstrating its effectiveness.

### Strengths
* The proposed method and ELBO optimization method can be applied in various circumstance.
* The overall manuscript is clearly written.
* The authors put considerable effort into experiments, and they successfully found the method's strength in robustness.

### Weaknesses
* Although it is referred to as generalization, I think the GP prior is another constraint or additional premise on SB that would only favor some portion of SDE problems in theory. I think a GP-based formulation can be overly restrictive for very high-dimensional modalities with additional geometric constraints that have branching paths as solutions.
* The precise notion of robustness is unclear, and why the GP prior recovers the "original" solution amidst noise is black-boxed.
* There is a room for further theoretical discussion on the GP prior which might benefit the clarity manuscript.

### Questions
.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper extends the Schrödinger Bridge (SB) framework by introducing a robust generalized formulation that unifies diffusion-based generative modeling and SB matching under a common theoretical lens. The authors develop a new objective function incorporating robustness to model mismatch and derive corresponding training dynamics. The work provides both theoretical guarantees and empirical validation on complex datasets, aiming to improve the stability and expressiveness of SB-based generative models. But to be honest, this paper is too mathematically dense and is hard for me to follow.

### Strengths
The paper tackles an ambitious and mathematically sophisticated problem, presenting a unifying and robust extension to the Schrödinger Bridge framework. The theoretical development appears rigorous, and the results—both in proofs and experiments—suggest meaningful advances in understanding and improving bridge-based generative modeling. The connection drawn between generalized SB matching and diffusion-based models is conceptually strong and of potential significance to the community.

### Weaknesses
The paper is mathematically dense and can be challenging to follow, even for readers with backgrounds about SB (not GSB). Some of the derivations could benefit from more intuition or interpretive commentary. It is also difficult, without deep verification of each step, to fully assess the soundness of the theoretical proofs.

### Questions
Could the authors provide a more intuitive explanation or diagram illustrating how the generalized robust SB formulation differs from the classical one, and how robustness manifests in practice?

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
This study proposes a variational solution that applies a Gaussian Process (GP) to efficiently solve the existing Generalized Schrödinger Bridge (GSB) problem. To define the Gaussian process, Gaussian distributions are first established based on the data from two domains, $x_0$ and $x_1$. A GP prior is then constructed between them, and learning is conducted by optimizing the Evidence Lower Bound (ELBO) over intermediate timesteps. Using this approach, the proposed method achieves superior performance compared to existing approaches.

### Strengths
- Simplification through a variational approach: The proposed method applies a Gaussian Process (GP) to the GSB problem, allowing convenient formulation of intermediate distributions through a simple prior. This variational perspective simplifies the overall problem structure while maintaining flexibility.

- Visualization of results: The paper thoroughly validates the proposed methodology through diverse visual analyses presented in both the main manuscript and the appendix. These visualizations effectively demonstrate the distinctions and advantages of the proposed approach.

### Weaknesses
### Major Weaknesses
- Time Complexity: As also mentioned by the authors, the proposed method has significantly higher time complexity compared to the original GSBM. According to Table 5 in the Appendix (Page 26), the method additionally requires $O(n^3)$ time. A strategy to further reduce the sampling time is needed. Considering the current computational cost, the performance improvement over GSBM does not appear dramatic.

- “Sparse” Variational Gaussian Process: The proposed method constructs a flow by interpolating simple single Gaussian kernels defined on two domains, $x_0$ and $x_1$. While this simplifies the problem, it does not fully capture the global distribution, making the selection of data pairs highly influential to overall performance.

- Uncertain Performance Difference: The proposed approach modifies the GSBM structure only by formulating $V_t$ as a GP problem. Based on the reported results, including those in the Appendix, the improvement over GSBM appears only incremental. It would strengthen the paper to include concrete experimental cases where GP-GSBM successfully solves problems that GSBM fails to address.

### Minor Weaknesses

- Choice of Primary Area: The proposed method utilizes a Gaussian Process (GP) as one possible solution to the GSB problem. In my opinion, this work may fit more appropriately under the area of generative modeling rather than GP itself.

- Formatting in the Related Work Section: It might be better to remove quotation marks from the paragraph titles in the Related Work section for cleaner presentation.

### Questions
See Weaknesses Section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This manuscript tackles the generalized Schrodinger Bridge (GSB) problem that incorporates an extra stage cost function on top of the original SB problem. Given that the previous efficient GSB solver only offers a point estimate of the path measure, the authors propose to treat the GSB as a posterior inference problem. Specifically, they treat the GSB objective as a stochastic likelihood function and leverage a Gaussian path prior, and aim to estimate the posterior distribution. To address the intractable posterior, the authors propose to leverage fully factorized variational inference, where the variational posteriors are tractable GPs. Empirical evaluation demonstrates that the proposed solution has competitive performance with the GSBM baseline and superior performance on the scenarios where there is uncertainty or under a proxy cost function.

### Strengths
* The writing is clear, with good contextualization of related works.
* The motivation of this paper is good.
* The proposed technique is sound, with well-executed experiments.

### Weaknesses
Overall, I have no major concerns about this manuscript. 

### Minor
* The implications or limitations of some design choices should be highlighted. For example, will the independent prior (regularization) be suitable for all applications? The same question applies to the homogeneous kernel function across dims (though this is common) and the form of the variational posterior. However, adding complexity to these designs may further increase computational cost.
* The figure's resolution can be improved. The current form is somewhat hard to read when the paper is printed.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
