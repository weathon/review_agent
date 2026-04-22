# Bayesian Parameter Shift Rules in Variational Quantum Eigensolvers

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Parameter shift rules (PSRs) are key techniques for efficient gradient estimation in variational quantum eigensolvers (VQEs). In this paper, we propose their Bayesian variant, where Gaussian processes with appropriate kernels are used to estimate the gradient of the VQE objective. Our Bayesian PSR offers flexible gradient estimation from observations at arbitrary locations with uncertainty information, and reduces to the generalized PSR in special cases. In stochastic gradient descent (SGD), the flexibility of Bayesian PSR allows reuse of observations in previous steps, which accelerates the optimization process. Furthermore, the accessibility to the posterior uncertainty, along with our proposed notion of gradient confident region (GradCoRe), enables us to minimize the observation costs in each SGD step. Our numerical experiments show that the VQE optimization with Bayesian PSR and GradCoRe significantly accelerates SGD, and outperforms the state-of-the-art methods, including sequential minimal optimization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work proposes the Bayesian parameter shift rule (PSR), which uses Gaussian process with VQE kernel, to estimate the gradient of objective in variational quantum eigensolver. Furthermore, guided by the posterior uncertainty, the stochastic gradient decent (SGD) step is completed in the gradient confident area. Numerical results demonstrate that the proposed optimization strategy can achieve faster convergence than the vanilla SGD.

### Strengths
- The manuscript provides a comprehensive theoretical analysis and numerical results, which are convincing.

### Weaknesses
- The proposed method is a natural application of Bayesian optimization with Gaussian process to VQE gradient estimation, which offers limited novelty.
- The motivation of employing the Bayesian PSR is not clearly clarified. It would be more convincing to discuss the challenges the proposed method tries to address.
- The manuscript lacks a formal statement or theorem guaranteeing the existence of an optimal sequence for both the shift parameter and shot allocation that ensures the fastest convergence.

### Questions
Please see the Weakness section

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors propose a method for optimizing variational quantum eigensolvers. The method aims to reduce the number of measurement shots by estimating gradients using Gaussian Processes (GPs). This Bayesian approach allows the algorithm to reuse previous observations in its gradient estimates. The authors combine this with an adaptive observation cost strategy, "GradCoRe", and show that this combined method outperforms comparable approaches for a fixed measurement shot budget.

### Strengths
* The problem it targets—reducing measurement shots—is high-impact and central to near-term quantum algorithms.
* They obtain rigorous estimates for the mean and covariance of the gradient GP (Theorem 3.1) and show that their Bayesian PSR converges to the standard PSR in the noiseless limit.
* In a simplified setting (Theorem 3.2), they use this estimate to demonstrate that the common $\pi/2$ shift is optimal for minimizing uncertainty, justifying a technique used in the literature.
* The proposed method is a clever combination of existing tools (SGD, GPs, and physics-informed kernels).

### Weaknesses
* The method appears to be a combination of existing methods (which, as noted, is also a strength), making the novelty of the contribution somewhat unclear.
* The experiments seem fairly limited, primarily focusing on a $(Q=5)$-qubit system. A comparison on a higher-dimensional problem would be needed to substantiate the "state-of-the-art" claim.
* The graphs comparing techniques (e.g., Figure 4) are hard to parse. I would suggest experimenting with ways to represent the uncertainty without using shaded fills, which tend to obscure each other.
* The derivation of the derivative GP (Section 2.1) is essential to the method. While the paper states the kernel modifications and alludes to Appendix A, a clearer, self-contained (or better-referenced) derivation in the main text would be beneficial.
* While the paper focuses on quantum cost (measurement shots), it neglects the *classical* computation cost. The method requires GP regression and solving an optimization problem (Eq. 17) at each step. An empirical comparison of this classical overhead would be helpful.

### Questions
* Is it sensible or standard to assume that the observation noise in the model is Gaussian??
* The experimental setup (Ising model, $(Q=5)$, $(L=3)$) is taken from Nicoli et al. (2023a). Is this 5-qubit problem considered a sufficient benchmark for "state-of-the-art" claims in the field?

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
3

### Summary
This paper presents a Bayesian variant of the well-known parameter shift rules (PSRs). Bayesian PSRs enable flexible gradient estimation from observations at arbitrary locations and unify the standard PSRs. Connecting the Bayesian PSRs with SGD, the paper proposes a new optimizer named gradient confident region (GradeCoRe), which outperforms many STOA methods for VQE training.

### Strengths
- Bayesian PSRs are a natural generalization of the standard PSRs, and they enable more efficient exploitation of previous measurement data by taking uncertainty information into account. 
- The theoretical relationship between Bayesian PSR and existing PSRs is established. Bayesian PSR reduces to standard PSRs in certain cases. Moreover, the optimality of PSRs is established. 
- Leveraging this new gradient Bayesian-based gradient estimation technique, gradient-based optimizers demonstrate improved numerical performance in various VQE problems.

### Weaknesses
- The current Bayesian PSR is derived based on Gaussian processes. Is this an appropriate assumption for VQE? How to justify the GP model?
- Is there a quantitative way to compare Bayesian PSRs and standard PSRs in terms of the quality of gradient estimation and sample complexity?

### Questions
- Is there an intuitive explanation for why the uncertainty dependence on $\alpha$ is a "U" shape curve (as in the last panel in Figure 2)?
- Even with Bayesian PSRs, it is a bit counterintuitive that an SGD-based algorithm can outperform Bayesian optimization methods. Is there an explanation for this?
- Is it possible to establish the optimality argument in the second order? If not, what are the challenges?
- Does the momentum information (e.g., momentum SGD + Bayesian PSRs) further accelerate the optimizer?

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
3

### Summary
This paper introduces the Bayesian Parameter Shift Rule (Bayesian PSR), a method that uses Gaussian Processes (GPs) with a physics-informed VQE kernel to estimate the gradient of the VQE objective. A key feature of this Bayesian estimator is its flexibility, allowing it to reuse observations to improve the accuracy of the gradient estimate. The paper also proposes a novel adaptive cost strategy called the Gradient Confident Region (GradCoRe), which leverages the gradient's posterior uncertainty provided by the GP. At each SGD step, the GradCoRe strategy solves for the minimum number of measurement shots needed to ensure the gradient's uncertainty falls below a dynamically updated accuracy threshold. Numerical experiments show this combined SGD-GradCoRe method accelerates optimization, as measured by the number of shots on the quantum side, and outperforms existing state-of-the-art methods.

### Strengths
This paper's primary contribution is building on recent work by Nicoli et al. and Anders et al. to adapt their concepts to a more general framework. It takes the Confident Region (CoRe) idea, which was previously applied to 1D subspace methods, and adapts it for the D-dimensional gradients used in SGD, creating the Gradient Confident Region (GradCoRe). 

The quality of this adaptation is well-supported on two fronts: theoretically, it provides a solid proof (Theorem 3.1) that its Bayesian PSR estimator is a valid generalization of the standard Parameter Shift Rule, and empirically, it demonstrates superior performance against strong, state-of-the-art baselines, including SubsCoRe from Anders et al that can be seen as a precursor model to the current one. This leads to a reduction of  the total number of required measurement shots by approximately 10-20% to reach the same energy levels in the benchmarks. The paper also maintains a high degree of clarity, using its experiments (Fig. 3) to effectively untangle the two sources of its performance gain. The results in Figure 3 indicate that while data-reuse alone (Bayes-SGD) provides no benefit over standard SGD, the entire performance lift is achieved by the adaptive measurement strategy (GradCoRe), which uses the GP's uncertainty to minimize shot cost.

Its significance comes from successfully broadening the utility of these physics-informed, adaptive-cost Bayesian methods beyond specialized algorithms, making them accessible to the widely-used SGD framework

### Weaknesses
The paper's novelty is somewhat limited, as its main contributions are largely an incremental adaptation of the VQE kernel  and Confident Region (CoRe) concept  developed by Nicoli et al. and Anders et al. While adapting this to the more general SGD framework is a valuable step, it relies heavily on these recently established foundations. 

The paper's primary ablation study (Figure 3) reveals that Bayes-SGD, which incorporates the Bayesian PSR and data reuse, offers no clear performance benefit over standard SGD. This suggests that the paper's performance gain is derived exclusively from the GradCoRe adaptive shot-counting strategy. This undermines the presented value of the Bayesian PSR framework itself.

Additionally, a limitation in the empirical validation is the inconsistent use of the gamma kernel hyperparameter; Bayes-SGD use gamma=1 while all key baselines are set to gamma=3. The paper lacks an ablation study to quantify the performance sensitivity of all methods to this parameter, making it difficult to fully isolate the architectural gains of GradCoRe and Bayes-SGD from a potentially more favorable hyperparameter setting. 

Lastly a very minor thing. I would argue that the paper overstates the novelty of Theorem 3.2, which proves the optimality of the π/2 shift. The proof primarily serves as a validation that the Bayesian framework aligns with existing knowledge, rather than being a new discovery.

### Questions
Figure 3  shows that Bayes-SGD (with data reuse) performs no better than standard SGD. This implies the performance gain of GradCoRe comes from its adaptive shot strategy, not from the Bayesian PSR or data reuse. Do the authors agree? If so, should the paper be re-framed to de-emphasize the Bayesian PSR and focus on GradCoRe as an adaptive shot-counting heuristic that can be applied to any PSR?   

Please justify the use of a different kernel ($\gamma=1$) for GradCoRe versus the baselines ($\gamma=3$). This is a critical confounding variable that is not mentioned in the main text. Does GradCoRe still outperform SubsCoRe if both use $\gamma=3$? Or if both use $\gamma=1$?"

Since the practical contribution of this work is the non-insignificant reduction in number of shots, can you please expand a bit on the computational overhead (on the classical side) introduced by your model. Presumably this should scale as $\mathcal{O}(N^3)$ and you have chosen N=400 as mentioned in the appendix. The major drawback that VQE suffers from in terms of scalability is the amount of time required for training a model with a substantial number of parameters. A discussion on this computational tradeoff could really help your work.

### Soundness
3

### Presentation
3

### Contribution
2
