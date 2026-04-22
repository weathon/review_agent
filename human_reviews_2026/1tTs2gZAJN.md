# Stochastic Neural Networks for Causal Inference with Missing Confounders

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Unmeasured confounding is a fundamental obstacle to causal inference from observational data. Latent-variable methods address this challenge by imputing unobserved confounders, yet many lack explicit model-based identification guarantees and are difficult to extend to richer causal structures. We propose Confounder Imputation with Stochastic Neural Networks (CI-StoNet), which parameterizes the conditional structure of a causal directed acyclic graph using a stochastic neural network and imputes latent confounders via adaptive stochastic-gradient Hamiltonian Monte Carlo. Under SUTVA and overlap, and assuming that the structural components of the data-generating process are well approximated by a capacity-controlled sparse deep neural network class, we establish model identification and consistent estimation of the mean potential outcome under a fixed intervention within this class. Although the latent confounder is identifiable only up to reparameterizations that preserve the joint treatment–outcome distribution, the causal estimand is invariant across this observationally equivalent class. We further characterize the effect of overlap on estimation accuracy. Empirical results on simulated and benchmark datasets demonstrate accurate performance, and the framework extends naturally to proxy-variable and multiple-cause settings with overlap diagnostics and bootstrap-based uncertainty quantification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes CI-StoNet (Confounder Imputation with Stochastic Neural Networks), a deep learning framework for causal effect estimation when confounders are unobserved. The key idea is to model the causal DAG using a stochastic neural network where missing confounders Z are treated as latent variables and imputed from the conditional distribution π(Z|A,Y). The authors use adaptive stochastic gradient Hamiltonian Monte Carlo (SGHMC) to simultaneously impute missing confounders and train the neural networks. They provide theoretical guarantees showing that causal effects remain identifiable even though the missing confounders can only be identified up to loss-invariant transformations. The method is evaluated on simulated data, proxy variable settings, and benchmark datasets

### Strengths
1. The paper provides convergence guarantees (Lemma 1) and consistency results (Theorems 1-2) for the proposed approach under sparse deep learning theory, which is a notable contribution.
2. The Markovian structure of CI-StoNet allows modeling diverse causal structures, including multiple causes and various proxy variable settings (Sections 3, A2).
3. The paper makes an interesting observation that causal effects can remain identifiable even when confounders are only identified up to loss-invariant transformations (Remark 3).
4. The paper includes simulation studies with both separable and non-separable confounding, and evaluates on multiple benchmark datasets with comparisons to many baseline methods.

### Weaknesses
1. While the paper uses "stochastic neural networks," the role and necessity of stochasticity is not clearly explained. The model (4) appears to be a standard neural network with additive Gaussian noise. Why is this stochastic formulation necessary versus standard deterministic neural networks with probabilistic inference? The connection between stochasticity and identifiability of causal effects needs clarification.
2. Assumption 1(ii): Assuming the true model is exactly a sparse StoNet is very strong and unlikely in practice. This eliminates model misspecification concerns but is unrealistic.
3. Mixture Gaussian prior (Equation 6): The choice of this specific prior with independent components seems arbitrary. No justification is provided for why this particular prior structure is appropriate for neural network weights in causal inference. The sensitivity to hyperparameters (λ_n, σ_0, σ_1) is not investigated.
4. Assumption 2 (Overlap): Requires overlap on the imputed confounder Z, but how can this be verified when Z is unobserved?
5. The paper does not provide clear, verifiable conditions to determine when causal effects are vs. aren't identifiable
6. The relationship between "loss-invariant transformations" of confounders and causal effect identifiability needs more rigorous treatment
7. What happens when Assumption 1(ii) is violated (i.e., true model is not a sparse StoNet)?
8. All experiments use relatively small networks and simple settings
9. Missing ablation studies on key components (e.g., impact of sparsity, SGHMC vs. standard SGD)
10. Computational complexity not discussed. No analysis of computational cost, scalability, or convergence speed compared to baselines.

### Questions
1. Why is stochasticity needed and why is it helpful to identify causal effect with missing confounders? Is the stochasticity in Z essential, or just a convenient modeling choice?
2. How do you determine if the causal effect is identifiable? According to Theorem 2, causal effects are identifiable when assumptions 1-2 hold (including assuming true model is a sparse StoNet and conditions of Lemma 1 and Theorem 1 hold). However, these are not practically verifiable. A practitioner cannot check if the true model is a sparse StoNet. 
3. How do you determine if confounders (missing) are non-identifiable? The paper claims confounders are non-identifiable due to "loss-invariant transformations" but then states causal effects are identifiable (Theorem 2). Remark 1 mentions equivalence classes, but how does one determine which equivalence class the learned confounder belongs to? Can you provide examples where two different confounder values yield the same causal effect estimate?
4. How realistic is the mixture Gaussian prior assumption? This assumption in Equation 6 is quite strong. Assuming all θ_i are independent is unrealistic for neural network weights, which typically have complex dependencies. Also, why is this two-component mixture appropriate? No justification provided
5. The conditions in (13) are complex. How sensitive are results to the choices of your hyperparameters?

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
3

### Summary
This paper proposes a new latent variable modeling approach called Confounder Imputation with Stochastic Neural Networks (CI-StoNet) to handle missing confounders when using observational data. CI-StoNet leverages stochastic neural networks to jointly model outcomes and unobserved confounders, and employs an adaptive Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) approach for both imputation and parameter learning. Theoretical guarantees are provided for convergence and consistency, and experiments on both simulated and benchmark datasets are conducted to confirm the effectiveness of the proposed method.

### Strengths
1. The paper tackles an important problem (i.e., missing confounders) in causal inference and proposes a new neural network-based method (CI-StoNet) that overcomes some limitations of prior latent variable approaches, such as limited applicability to nonlinear models and consistency issues.
2. The authors provide theoretical support for the convergence and consistency of the proposed method.
3. Experiments on both simulated and benchmark datasets are conducted to evaluate the performance of the proposed method.

### Weaknesses
1. As noted in the limitations, the method assumes that the underlying causal structure (DAG) is correctly specified, which may be difficult in real applications. Is it possible to provide some experimental results when the DAG is misspecified?
2. The approach involves training deep neural networks with adaptive MCMC, which can be computationally intensive. Complexity analysis or comparison with baselines should be provided.
3. The two types of baselines are inconsistently used in Table 1, S1 and S2. The differences and reasons should be clearly described.
4. The analysis in the paper relies on several technical assumptions, which may not always hold.

### Questions
Please refer to Weaknesses 1-3.

### Soundness
2

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
This manuscript proposed to estimate the causal effects with the missing confounder through a stochastic neural network. The authors provide some preliminary theoretical justifications, with some experimental justifications on both synthetic and small real world scenarios.

### Strengths
Overall the authors present the idea in a reasonable manner, with reasonable improvement on the performance.

### Weaknesses
The assumption of the existence of the underlying stochastic neural networks as inductive bias can be a chick and egg problem on identifiability. Meanwhile, the theoretical justifications are mainly on consistency, instead of convergence rate based on some standard assumptions.

### Questions
My questions are mainly two-fold:
* Think about the case that we only have $(A, Y)$, where $A$ is just some binary treatments, then I don't think we can perform reasonable ATE and hence if the conclusion claimed in this manuscript is correct. The only explanation is that we embed some inductive bias in the function class of $\mu$ (similar to the consistency claim in Theorem 2 that depends on $\theta^*$). If that's the case, we hide the missing confounder issue by adding additional identifiability issue on the function class. Is that correct?
* Regarding the theoretical guarantee, if the density ratio $P(A|X)/P(A)$ is negligibly small but not zero, then we will have a extremely bad convergence rate. However, Theorem 2 only provides asymptotic convergence, and I'm generally wondering the dependency of some other quantities like the one I just mentioned. Otherwise it's not a real causal effect estimation solution but just an attempt to directly apply the stochastic network's results to causal effect estimation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes CI-StoNet, a latent-variable framework that treats missing confounders as stochastic hidden states and jointly models (i) their conditional distribution and (ii) the outcome with two interconnected neural networks. Training alternates between imputing the latent confounders via adaptive SGHMC and updating network parameters, enabling consistent causal-effect estimation even when the confounders themselves are non-identifiable up to loss-invariant transformations. The approach targets complex, nonlinear settings; retains identifiability of causal effects under mild assumptions; and extends naturally to multiple-cause and proxy-variable scenarios. The authors prove convergence of the adaptive SGHMC estimator and consistency of the resulting causal-effect estimates, then demonstrate state-of-the-art accuracy on simulations and benchmarks (ACIC 2019, Twins). They note limitations around DAG specification (e.g., mediators/colliders) and current lack of explicit uncertainty quantification, and outline how CI-StoNet’s Markovian structure can be adapted to broader proxy-based causal graphs.

### Strengths
- The paper offers a strong conceptual framing for missing confounders by treating them as latent states within a StoNet and jointly learning their distribution with treatment and outcome modules.
- The proposed method reaches state-of-the-art ATE accuracy on ACIC-2019 and performs strongly on the Twins dataset; performance remains comparatively stable even when a key confounder is removed, indicating robustness of the latent-imputation mechanism.

### Weaknesses
1/ Your inference of $Z$ conditions on each datum’s $(A_i,Y_i)$. When both variables are binary while $Z$ is multi-dimensional, recovering $Z$ from $(A,Y)$ alone is under-determined; the latent confounders are generally non-identifiable without extra structure or additional observables. Could you formalize when $(A,Y)$ contain \emph{enough} information about $Z$?

2/ You state that causal effects remain identifiable “under mild conditions” even if $Z$ is only identifiable up to loss-invariant transformations. Please state these conditions precisely for the binary–binary case and add guidance about what fails if they are violated. Under what restrictions on $g_1,g_2$, noise, or overlap does $E[Y(a)]$ remain identifiable despite non-identifiable $Z$?

3/ Because $D_i=\{A_i,Y_i\}$ drives the imputation of $Z_i$, the method’s success hinges on how informative $(A,Y)$ are about $Z$. Please articulate minimum “richness” assumptions (beyond overlap) under which the estimator is well-posed—e.g., conditions ensuring that $\pi(Z\mid A,Y)$ is sufficiently concentrated to support accurate effect estimation.

4/ You provide proxy-variable extensions for “outcome-depends-on-proxy” and “treatment-depends-on-proxy.” It would help to add practical criteria or diagnostics for when $(A,Y)$ are too sparse and $X$ must be incorporated (e.g., signal tests or ablations showing failure without $X$). Could you include a short decision guideline indicating when to introduce proxies?

5/ The convergence of adaptive SGHMC is asserted under specific conditions and in a reduced parameter space (due to loss-invariant transformations). In practice, what diagnostics should users monitor to assess mixing/convergence and to detect cases where $Z$ is poorly informed by $(A,Y)$? Provide a checklist (e.g., effective sample size, potential scale reduction, stability of $E[Y(a)]$ across chains).

6/ You already conduct a robustness test by removing a key confounder in \emph{Twins}. Please add a complementary experiment where $A$ and $Y$ are binary and $Z$ is high-dimensional, to quantify: (i) degradation in effect estimates, and (ii) the (non-)recoverability of $Z$. How does performance change as $\dim(Z)$ increases under fixed binary $(A,Y)$?

7/ Your ATE estimator is a Monte Carlo average over imputed $Z$, which suggests posterior uncertainty is available. Please report credible intervals (or bootstrap CIs) for $\widehat{E}[Y(a)]$ and $\widehat{\tau}$, and compare coverage when $(A,Y)$ are low-information versus when proxies $X$ are added.

8/ Given the potential information bottleneck when $(A,Y)$ are discrete, practical users would benefit from sample-size versus complexity guidance. Please provide rules of thumb linking $n$, $\dim(Z)$, and network sparsity/width needed to stabilize effect estimation.

### Questions
Please refer to the questions in Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
