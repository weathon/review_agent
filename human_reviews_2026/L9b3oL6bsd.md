# Functional Distribution Networks (FDN)

- Decision: Reject
- Scores: 6, 0, 2, 4

## Abstract
Modern probabilistic regressors often remain overconfident under distribution shift. Functional Distribution Networks (FDN) place input-conditioned distributions over network weights, producing predictive mixtures whose dispersion adapts to the input; we train them with a Monte Carlo $\beta$--ELBO objective. We pair FDN with an evaluation protocol that separates interpolation from extrapolation and emphasizes simple OOD sanity checks. On controlled 1D tasks and small/medium UCI-style regression benchmarks, FDN remains competitive in accuracy with strong Bayesian, ensemble, dropout, and hypernetwork baselines, while providing strongly input-dependent, shift-aware uncertainty and competitive calibration under matched parameter and update budgets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Functional Distribution Networks (FDN), a method that places input-conditioned distributions over network weights to produce predictive mixtures with adaptive dispersion. The approach uses lightweight hypernetworks to amortize a variational posterior $q_{\phi}(\theta|x)$ and trains with a $\beta$-ELBO objective. The authors propose two variants (IC-FDN conditioning only on inputs, and LP-FDN conditioning on previous layer activations) and evaluate them on synthetic regression tasks with an explicit interpolation/extrapolation protocol. Under matched parameter and update budgets, FDN demonstrates competitive in-distribution performance and strong calibration on smooth distribution shifts, though it struggles with highly oscillatory out-of-distribution scenarios.

### Strengths
* The paper addresses a critical problem in neural regression (overconfidence under distribution shift) with a principled approach. The explicit evaluation protocol that separates interpolation from extrapolation and uses $\Delta$ Var, MSE-variance slope/intercept, and Spearman correlation provides interpretable diagnostics for uncertainty quality. This makes it clear what "good" OOD behavior should look like (widening uncertainty, positive $\Delta$ Var,, near-unity slope).
* The controlled comparison is commendable, matching parameter budgets (~1000 parameters) and update budgets across all baselines (BNN, Deep Ensembles, Dropout, Hypernetworks). This eliminates common confounders in uncertainty quantification studies where methods differ dramatically in capacity or training cost, making the empirical findings more trustworthy and the conclusions more actionable.
* FDN is positioned as a drop-in replacement for standard MLP layers, making it easy to retrofit into existing architectures without redesigning the predictive head or training pipeline. The use of lightweight hypernetworks to generate layer-wise weight distributions keeps the approach computationally tractable, and the β-ELBO training objective integrates naturally with standard gradient-based optimization.

### Weaknesses
* The evaluation is restricted to three synthetic 1D regression tasks, and FDN's calibration breaks down significantly on the sine task (highly oscillatory OOD).  The lack of experiments on higher-dimensional, realistic datasets limits our understanding of how FDN scales beyond toy problems.

* The decision to fix observation variance σ² to a constant is justified as isolating epistemic uncertainty, but it fundamentally limits the model's expressiveness. Real-world regression often exhibits heteroscedastic aleatoric uncertainty that varies with input regions, and forcing homoscedasticity may cause the epistemic component (weight uncertainty) to compensate inappropriately. 

* The authors fail to discuss post-hoc calibration techniques for regression models [1,2,3]. I recommend that the authors include a discussion of post-hoc calibration approaches to better ground the paper and provide appropriate context for related work.


**Refs**

[1] Accurate uncertainties for deep learning using calibrated regression

[2] Distribution Calibration for Regression.

[3] Quantile Regularization: Towards Implicit Calibration of Regression Models

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper introduces function distributional network. The main idea is that the neural network weights are sampled from a (variational) distribution that is input dependent, i.e., the parameter distribution can capture input dependent uncertainties. The paper introduces a vartiational bayes method using a beta-VAE to optimize the likelihood. Experimentes are illustrated on very simple 1-D regression tasks.

### Strengths
- the paper is relatively well written
- the idea is novel even though I am very sceptical that the approach can be scaled to anything more complex than the presented 1D tasks.

### Weaknesses
- the experiments are way too simple, only presenting tiny networks for 1-D regression. the 1-D regression tasks are not even randomized but presented for specific functions, so it is hard to say whether the results are just artifacts of the 3 functions that have been presented. The authors should look into the neural process literature on what family of functions are used there. If we would have the average performance over many instances of the same family, than it would at least be statistically more convincing.
- I do not see how this approach should be scalable in any ways. A standard DNN has O(N^2) parameters (N being the number of neurons per layer). Yet, this approach (at least the LP-FDN) would scale with O(N^3), which is completely unrealistic to use for a reasonable network size. Also the IC-FDN scales poorly with O(D*N^2), where D is the number of input dimensions.
- The plots are not readable (too small font, huge legends etc...) and very hard to understand.

### Questions
Please investigate the scaling properties further and present more standardized medium size regression tasks (as promised in the first sentence of the experiment section... but I could not find any other regression tasks except for the 3 almost trivial once presented).

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
3

### Summary
The paper introduces Functional Distribution Networks (FDNs), a method for per-input amortized variational inference in multi-layer perceptrons (MLPs).

The proposed variational posterior factorizes across layers and each factor is conditioned on either the input location (IC-FDN) or the previous layer's activations (LP-FDN). The layer-wise posteriors are set to be diagonal-covariance Gaussians, and the conditioning is realized via layer-specific hyper-networks, mapping the conditioning signal to the Gaussian's parameters. A zero-mean isotropic Gaussian prior is placed on the MLP parameters, allowing for analytic computation of the standard evidence lower bound objective (ELBO). The variational parameters of the hyper-networks are learned by optimizing a variant of the ELBO, where the KL from posterior to prior is scaled by the regularizer, $\beta$. 

The authors compare their approach to a number of uncertainty-quantifying regression methods on three 1D regression tasks.

### Strengths
- Quantifying regression predictive uncertainty in a calibrated manner is an unsolved research problem. The paper aims to tackle a highly relevant issue, of great interest to the field.
- The split of the input space into ID and OOD is useful for assessing OOD detection capabilities.
- The proposed method is mostly well explained.
- The positioning of this work relative to related literature is made clear and relevant baselines are included.

### Weaknesses
- Figures 1 and 2 are quite hard to understand. This is due to font sizes, legend positioning, overlapping lines and (lack of) scaling of axes. Sometimes there are multiple lines of the same color (e.g. 1e) and it is not clear what they represent. It should be stated clearly in the caption that Figure 1 shows results on both ID and OOD input locations.

- The quantities used to compare methods (CRPS, AURS, ...) should be mathematically defined, including how they are computed on finite data, at least in the appendix. 

- The evaluation in general is rather slim. For instance, comparing the MSE of different methods would have been interesting. This would allow statements about the trade-off between uncertainty quantification and accuracy.

- Since these are 1D tasks, a simple plot showing the training data and predictive mean and, e.g., two standard deviations would have been relevant. This would help to build intuition for the model's behavior.

- In Section 3, relevant symbols and their dimensionalities should be explicitly defined. This would make the exposition easier to follow.

- Throughout, more references would be helpful, but particularly in Appendix A.

-  The conclusions one can draw from experiments using a single hidden layer MLP and 1D regression tasks are quite limited. What about higher-dimensional inputs and outputs?

- The step function task is not well defined. Using $\theta$ both for this function and the MLP parameters is suboptimal.

- The code should be made available to reviewers, e.g., via https://anonymous.4open.science.

- The introduction could be sharpened. Discussions of training objective and evaluation protocols may be better put in Sections 3 and 4, respectively.

- The treatment of isotropic, diagonal and full covariance in Section 3.1 seems overly elaborate, given that the observation noise is simply fixed to 1. This would largely be better placed in the appendix.

- Shuffling the rows in Tables 3, 4 and 5 seems unnecessary, same with the columns of the individual plots in Figure 3.

### Questions
- Can the authors connect their work to [1]?
- How many training examples are used for each task?
- How should we interpret the huge  $\Delta\text{MSE}$ values for the sin task (e.g. Table 4)? Does this just mean the predictions here are extremely poor?
- Are the training data function values noised?

[1] Jafrasteh, B., Villacampa-Calvo, C., & Hernández-Lobato, D. (2021). Input dependent sparse Gaussian processes.

### Soundness
2

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
This paper proposes Functional Distribution Networks (FDN), an architecture where the network’s parameter distribution is conditioned on the input. By dynamically adapting weight distributions, FDN aims to better capture epistemic uncertainty, particularly under out-of-distribution (OOD) conditions. The method is positioned as distinct from Neural Processes, Hypernetworks, and standard Bayesian approaches. Experiments cover both in-distribution (ID) and OOD scenarios, with comparisons to several stochastic baselines and analyses under a fixed parameter budget.

### Strengths
- The idea of conditioning parameter distributions on inputs is interesting and well-motivated for handling OOD uncertainty.
- The paper provides a clear conceptual positioning relative to related work.
- The evaluation considers both ID/OOD performance and parameter efficiency.

### Weaknesses
- The experimental results do not convincingly show clear gains over strong baselines.
- No real-world experiments are presented, limiting practical validation.
- Figures and presentation could be clearer

### Questions
While the approach is interesting, this paper requires more convincing experiments to show the merit of the architecture. I will be open to raising my score if the following questions are addressed: 
- Evaluation: How does FDN perform compared to existing methods on real datasets? How about with other standard calibration metrics e.g. ECE and NLL?
- Robustness: Since the parameter distribution can shift a lot depending on the input, how robust is FDN to noise in the data?
- Generalization: How does FDN perform in comparison to other baselines in terms of generalization for ID and OOD? A low $\Delta$MSE in Fig. 3 seems to indicate that the method might be generalizing well even under OOD case.

Additional feedbacks:
- Figures are very hard to read/comprehend (especially Fig. 1)
- Provide more detailed explanations of IC/LP-FDN in the method section
- Line 246 repeats Line 204
- Visualization of the interpolation/extrapolation results would be nice to have.

### Soundness
2

### Presentation
2

### Contribution
3
