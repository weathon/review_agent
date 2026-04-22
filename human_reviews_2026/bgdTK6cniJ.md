# CaNDiCE: Causal Discovery of Nonlinear Dynamics Through Counterfactual Explanations

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
The problem of discovering governing equations from noisy observational data has broad applications in scientific discovery, control, and prediction of complex systems. However, existing approaches that infer dynamics directly from data—whether symbolic regression (e.g., tree-based methods) or sparse identification with pre-defined basis functions—often suffer from poor generalizability, sensitivity to noise, and the inclusion of spurious terms. In this work, we present a causality-preserving counterfactual explanations framework for discovering governing equations in dynamical systems. Counterfactuals in this setting are hypothetical governing equations obtained by minimally perturbing basis function coefficients to induce out-of-distribution trajectories. By penalizing counterfactuals that deviate from the observed topological causality, a measure of directed effective influence between state variables, the resulting trajectories remain consistent with the causal structure of the true dynamics inferred from observed data. As such, resulting counterfactuals are obtained only by perturbing causal terms in the governing equation, while spurious terms are naturally suppressed since their perturbations violate causal consistency.  We evaluate our approach across a range of dynamical system benchmarks and show that it outperforms state-of-the-art methods, including symbolic regression, library-based sparse regression, and deep learning models, in identifying robust and parsimonious governing equations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an equation discovery framework for dynamical systems based on topological causality. It first defines a world model as the posterior of the sparse regression parameters given the bootstrapped data samples and updates the posterior using a stochastic inverse approach. Then, it trains a generator that generates perturbations to the sampled parameters from the world model using a combination of GAN loss and several regularization terms that promote sparsity and causal consistency of the perturbations. The terms whose coefficients are often perturbed by the trained generator are deemed causal and retained in the function library for sparse regression.

### Strengths
* The paper provides extensive context about related topics, such as topological causality and counterfactuals.
* The experimental results are impressive on the Lotka-Volterra equation and the Lorenz equation. The proposed method did very well when limited data were available.

### Weaknesses
* I find the writing in the methodology section unclear in general. For example, in Section 3.1, the kernel density estimator and stochastic inverse approach are crucial for estimating and updating the posterior $\pi(\Theta\Lambda|\mathbf y)$. These techniques should be (at least briefly) introduced in this problem context. Also, the notation $\pi(\Theta\Lambda|\mathbf y)$ is slightly confusing. I suppose $\Theta\Lambda$ here in fact refers to the simulated trajectory from $\dot {\mathbf x} = \Theta\Lambda$, but this notation alone might suggest the RHS expression itself.
* The presentational issues become more severe when it comes to the counterfactual model (Section 3.2 and 3.3). I have a lot of questions regarding these sections. In the CGAN formulation, what is the difference between the role of the discriminator and the classifier? They seem to both classify between the unperturbed (real) parameters and the perturbed (fake) parameters. Also, from the algorithm, it seems that the update of D is decoupled from that of G, which is different from the original GAN formulation. Why is that? I cannot understand eq. (10) either. The classification label $z$ is never explained or defined in the text. And the $\mathbf y$ inside the expectation does not make sense, since this is a scalar equation.
* The writing should be improved in general. For example, the second point of the contribution list says that "... generating counterfactual instances that lead to *out-of-distribution* trajectories... Counterfactuals are obtained by... *in-distribution* trajectories." Before reading the later sections, I could not figure out whether the counterfactual model should lead to in-distribution or out-of-distribution trajectories. On second thought, I can understand this statement where "minimally perturbing" seems to serve as a negative, but the unnecessary complexity in writing has made the paper difficult to understand.
* It would be great to include a figure explaining the entire pipeline, in addition to Algorithm 1.
* The experiments only considered two simple dynamical systems. While the results on these two systems are impressive, it remains to be seen whether this can be generalized to other datasets.
* The experiments did not compare with weak SINDy, which is specifically designed for noisy data.

### Questions
* L60: missing cross-reference
* distinguish between \citet and \citep
* L219: Can you elaborate on why you need the causal consistency constraint?
* How many samples are needed to train the CGAN counterfactual model? Since it involves some neural networks, does it do well with the small sample sizes in the experiments?
* How time-efficient is the proposed method compared to baseline SINDy?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of equation discovery from noisy and limited data, which is a long-standing challenge in scientific machine learning. The authors introduce CaNDiCE (Causal Discovery of Nonlinear Dynamics through Counterfactual Explanations), a framework that integrates counterfactual reasoning and topological causality to recover parsimonious, causally meaningful governing equations.
The method builds a world model over the coefficients of a predefined basis library, generates counterfactual coefficients via a conditional GAN that satisfy sparsity and causal consistency constraints, and then identifies the minimal causal set of terms to refit a sparse regression model.
Empirical evaluations span five benchmark dynamical systems: Lotka–Volterra, Lorenz, Van der Pol, Rössler, and a ball-drop experiment with air resistance, under varying signal-to-noise ratios (SNR) and data regimes. Results show large improvements compared to classical and recent baselines such as SINDy, ESINDy, and SPL, especially in low-data and high-noise conditions.

### Strengths
1. **Conceptual novelty.** The paper introduces a fresh causal perspective on equation discovery by embedding counterfactual generation within a causal consistency regularization. This adversarial counterfactual setup, where coefficient perturbations are constrained by topological causality, is an original and very interesting idea.

2. **New inspiration for physics discovery using causal constraints.** The incorporation of topological causality (via cross-mapping on reconstructed manifolds) provides a principled way to ensure causal consistency in dynamical systems where classical DAG-based causality notions fail. This is a meaningful step toward causal interpretability in scientific ML.

3. **Clear writing and methodological exposition.** The paper is well structured and mathematically precise. Algorithm 1 and the breakdown into “world model”, “counterfactual model”, and “minimum set discovery” make the pipeline understandable.

### Weaknesses
1. **Missing key baselines and contextualization.** The paper omits several recent transformer- or diffusion-based approaches to symbolic regression and equation discovery, notably "ODEFormer: Symbolic Regression of Dynamical Systems with Transformers (ICLR 2024)", which introduces the ODE-Bench dataset. Including such models would better position CaNDiCE within the current landscape of neural-symbolic discovery.

2. **Theoretical grounding and identifiability.** While the paper discusses causal constraints qualitatively, it lacks a formal analysis of parameter identifiability or conditions under which the causal coefficients are recoverable. For example, under what assumptions does topological causality regularization guarantee recovery of the correct sparse structure?

3. **Computational complexity and scalability.**
The combination of stochastic inversion, bootstrapping, and GAN training raises concerns about computational efficiency. The paper would benefit from a runtime or memory comparison with baselines, especially for higher-dimensional systems.

4. **Interpretability and intuition gaps.**
The notion of topological causality may be unfamiliar to much of the ICLR audience. A concise **visual** example—e.g., a 2-variable dynamical system with the corresponding manifold mappings—would greatly help build intuition. 

5. **Library dependence and limitations.**
As shown in the ball-drop case study, CaNDiCE’s success depends heavily on whether the causal functional forms are representable within the predefined basis library. The paper should discuss possible extensions to mitigate this limitation.

### Questions
1. **Identifiability and guarantees**
Under what assumptions does CaNDiCE provably recover the correct causal terms? Is there an identifiable mapping between topological-causality preservation and coefficient consistency?

2. **Complexity and scalability**
What is the asymptotic or empirical computational cost relative to SINDy/ESINDy/SPL? Could the GAN-based counterfactual generation become a bottleneck for high-dimensional systems?

3. **Robustness to library misspecification**
Can the model adaptively extend or refine its basis set when key functional forms are missing? 

4. **Ablation or sensitivity analysis**
How sensitive are results to the hyperparameters λ_TC and λ_sp? Does the balance between sparsity and causality penalties substantially affect which terms are identified as causal?


I am happy to raise my score if the concerns in questions and weaknesses are addressed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors introduce a couterfactional penalty term to model discovery methods, specifically the spare identification of nonlinear dynamics algorithm.  They then show the performance of this method against some of the variants of SINDy.

### Strengths
The ideas of the paper are actually quite nice.  It seems a rather nice innovation to include in the regression architecture.  I would strongly encourage the authors to continue pursuing this line of work as it has great potential.

### Weaknesses
Unfortunately, the method seems rathe immature to me at this point.  Specifically, the two models demonstrated in the paper are the Lotka-Volterra and Lorenz system, both of which are very basic models.  It certainly fine to demonstrate initially on these models, but it certainly would be expected for an ICLR to have much more challenging models to explore. 

Additionally, the comparisons to SINDy, eSINDy, SPL, while good, are certainly not state-of-the-art methods.  In fact, these methods are not really aimed at causal inference.  Much more serious comparisons should be made against what are considered causal learning methods.  So the comparisons are simply not up to what would be expected.

### Questions
Only two main questions:

How does this work for more challenging models than Lorenz/Lotka-Volterra?  Especially with noise?

How does the method actually hold up in comparison with leading causality inference methods?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a new approach to select relevant basis functions for sparse identification of governing equations based on topological causality. The authors propose to train a GAN to sample counterfactual trajectories that preserve the causal structure as measured by a topological causality metric. Examining the distribution of sampled counterfactuals then allows for the construction of a minimal library of basis functions by eliminating terms that do not generate causally consistent counterfactuals.

### Strengths
This paper is well-written and uses a recently developed theoretical tool for causal analysis of dynamical systems to improve system identification. The experiments show that this approach has a significant advantage over standard non-causal system identification, especially for noisy data where standard SINDy fails.

### Weaknesses
My main concern is regarding the scalability of the counterfactual generation. It seems to be at least quadratic in the number of initial library terms to even evaluate the loss for the GAN. Furthermore, the GAN must effectively sample all sparse combinations to truly find all good counterfactuals. As mentioned in the paper, this is NP-hard. The world model construction may also run into scalability issues due to the need to estimate a stochastic inverse.

### Questions
1. How computationally expensive is constructing the world model and computing the topological causality metric? How does it scale?
2. How stable and reproducible is the GAN training, and are there failure modes that you observe? Does failure to sample enough counterfactuals result in a library that is too small to fully capture the dynamics?
3. Can you run a test on a much higher-dimensional system to give a sense of the scaling behavior?
4. I'm a bit confused by the diversity-promoting loss in equation 11. If gamma > 0, wouldn't the loss encourage the new counterfactual candidate to be similar to pre-existing counterfactuals (so reducing diversity)? Also, what does it mean to subtract a coefficient from a set of coefficients (are you averaging here)?

### Soundness
4

### Presentation
4

### Contribution
3
