# metabeta - A fast neural model for Bayesian mixed-effects regression

- Decision: Reject
- Scores: 6, 2, 2, 2

## Abstract
Hierarchical data with multiple observations per group is ubiquitous in empirical sciences and is often analyzed using mixed-effects regression. In such models, Bayesian inference gives an estimate of uncertainty but is analytically intractable and requires costly approximation using Markov Chain Monte Carlo (MCMC) methods. Neural posterior estimation shifts the bulk of computation from inference time to pre-training time, amortizing over simulated datasets with known ground truth targets. We propose metabeta, a transformer-based neural network  model for Bayesian mixed-effects regression. Using simulated and real data, we show that it reaches stable and comparable performance to MCMC-based parameter estimation at a fraction of the usually required time.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents a simulation-based inference approach for Bayesian mixed-effects regression called metabeta. Hierarchical models are an important and widely used class of models, and efficient inference for them remains an active research topic. The authors train a single model that supports multiple priors, instead of training one model per prior as in previous work, and they employ a transformer-based architecture. The approach aims to achieve MCMC-level accuracy with greatly reduced inference time. The paper is clearly written, well-motivated, and addresses an interesting and timely problem in Bayesian inference for hierarchical models. It builds upon recent advances such as those by Habermann et al. and  Hollmann et al., extending neural posterior estimation to a broader mixed-effects setting with multiple priors. This extension is both conceptually interesting and practically meaningful.

### Strengths
This paper addresses an important and timely problem: Bayesian inference for hierarchical models. The proposed metabeta framework is interesting, combining transformer-based neural posterior estimation with amortization across multiple priors. The approach shows promise and could meaningfully reduce inference cost in mixed-effects regression. Strengths are
- Novel and well-motivated application of neural posterior estimation to mixed-effects regression, and effectively leveraging the available likelihood for post-hoc refinement.
- The integration of multiple priors into a single amortized inference model represents a significant step toward general-purpose Bayesian inference.
- Comprehensive benchmark suite covering both in-distribution and out-of-distribution data, which supports the claims of generalisation and robustness.
- Clear presentation and placement within the existing literature, helping the reader understand the contributions of the approach.

### Weaknesses
HMC diagnostics are not reported and convergence is doubtful 

The paper reports “divergence and strong outliers” (l.129) in HMC runs but does not provide diagnostics and the authors do not define what constitutes “outliers” (l. 129). Moreover, they do not report effective sample sizes, divergence counts or a similar convergence diagnostics. Unconverged chains should be excluded and metrics computed on all converged runs. The current MAD-based chain selection may bias variance estimates, potentially causing the selected chain of HMC to underestimate posterior variance. This should be justified or replaced with a more standard convergence diagnostic. If convergence issues persist, HMC might need a longer warm-up or more iterations.

More concretely:
Figure 1D shows a discrepancy between HMC and metabeta. Given that HMC has asymptotic convergence guarantees, and that HMC uses the “true priors and generative model” (l.128) , this raises questions about whether metabeta is overconfident (e.g., for $\beta_0$) or whether HMC was improperly tuned. If the latter, longer runs or a different parameterization may be required. Please clarify the setup and report diagnostics. To support the claim that both pipelines are correctly specified, including at least one simple case with a known analytical posterior could be helpful to demonstrate that HMC converges and metabeta matches it. Even when HMC turns out to perform better, the amortisation part is still very valuable. Interesting would also be to compare the methods, when the prior is misspecified.

A quantification of the differences between HMC and metabeta in the posterior predictive distributions would further strengthen the results.

Minor comments
- In Section 2.1, $S$ is undefined (l.95).
- “we simulate hierarchically structured datasets using PyTorch“ (l.110), what is the benefit of using PyTorch here for simulation?
- Figure 1 needs to be improved in terms of overall readability: missing labels (panel C), unreadable fonts, neural networks are badly visible, and too small text in subplots. Make clear that D is only inference time, not training time.
- Font sizes in all main figures are too small. In general, the resolution of the figures should be improved. 
- The importance sampling refinement is a nice idea. Showing results without this step to isolate its contribution would further improve the paper.
- The code is not provided, preventing verification of the claim that open-source software and pretrained models are or will become available. Please check out https://anonymous.4open.science/.

### Questions
The authors claim that prior work “at best nullifies the runtime advantage of NPE” (l. 68). How does this compare to the authors’ method? In particular there are serval situations, which are not discussed enough in the opinion of the reviewer, such as:

- How can missing data be handled? Many hierarchical datasets contain incomplete observations, and prior work uses masking to circumvent this issue (e.g., “All-in-one simulation-based inference“ by Gloeckler et al.).

- The paper claims full amortization, yet separate models are trained for different parameter dimensionalities. This weakens the claim of applicability for practitioners if the needed parameter dimensionality is not available. The authors should clarify early in the manuscript that amortization holds only for fixed dimensionality $d$ and group structure $q$ and that multiple models are trained. Also, here they could connect to recent work, such as "Compositional amortized inference for large-scale hierarchical Bayesian models" by Arruda et al.

- How are priors incorporated as inputs: through direct parameterization or learned embeddings? What about priors which are not part of the training data? This should be clarified in the main maunscript and potentially the relation to Whittle et al. “Distribution Transformers: Fast Approximate Bayesian Inference with On-The-Fly Prior Adaptation“ discussed, which seems closely related.

Please report computational cost for training metabeta, including total training time for the different models and resource usage. This is essential for assessing amortization trade-offs.

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a meta-learning framework for estimating the posterior distribution of coefficients in a Bayesian linear mixed-effects model, leveraging transformer architectures in a manner similar to TabPFN. To enhance the calibration of the predictive distribution, an additional post-hoc refinement stage—such as importance sampling or conformal prediction—is incorporated. The proposed approach is evaluated on both synthetic and semi-synthetic toy datasets.

### Strengths
Employing transformer-based meta-learning for diverse forms of amortized Bayesian inference is a compelling and important research direction. This work represents a valuable contribution to this growing area.

### Weaknesses
As noted by the authors, a key limitation is that a pretrained metabeta model can only be applied to datasets with a specific number of fixed effects and random effects. This constraint reduces the practical impact of the approach, as the model must be retrained for each new dataset with differing dimensionality. In such cases, the benefit of amortization becomes less compelling than using e.g., MCMC directly.

Another limitation is that all experiments are conducted on small-scale synthetic and semi-synthetic datasets, leaving the performance on real-world datasets uncertain.

### Questions
1. Could the authors describe the model details of the normalizing flow, particularly clarifying how the input s is utilized within the flow?

2. Did the authors employ both post-hoc refinement methods in the experiments, or was only one method used?

3. Could the authors provide experimental results on real-world datasets? Although the ground-truth effects are unknown, the model’s performance could be assessed by comparing the prediction accuracy of a linear mixed-effects model whose coefficients are estimated using metabeta or HMC.

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
4

### Summary
This paper proposes a method for neural amortized Bayesian inference (ABI) for linear mixed-effects (multilevel/hierarchical) models. The results are overall good but the contribution may be a bit small in light of existing literature. I am short on time due to the semester start. Apologies if my reviews are a bit short. I am happy to engage in reviewer discussion should be concerns not be clear.

### Strengths
- The paper works on an important topic. 
- The applied and combined methods are sensible.
- The presentation is easy to follow for somone familiar with mixed-effects models.

### Weaknesses
- The contribution is overall small. Already previous papers provide ABI for multilevel models (and are correctly cited in the paper). The main addition here is the amortization over prior hyperparameters, but this has also been suggested in other places (https://arxiv.org/abs/2310.11122), althought admittedly not in a multilevel context.
- The HMC baseline seems to be incorrectly or at least not well implemented. For such simple multilevel models, HMC in PyMC or other PPLs should not struggle with any convergence or recovery issues. I assume your parameterization of the model wasn't quite right or optimal. Consider using a non-centered parameterization for the random effects. Or compare with an existing implementation of such model, e.g., via the brms R package using Stan as PPL backend, or bambi in python using PyMC as backend. 
- The authors only focus on *linear* multilevel models where the error distribution is Gaussian. This unecessarily restricts the flexibility of the framework. 
- No correlations between random effects are considered. 
- The general formulation in terms of design matrices X and Z suggest the possibility of multiple grouping factors and corresponding random effects. Yet, your implementation just supports a single grouping factor as far as I can tell. I don't expect you to generalize your framework to multiple grouping factors right away. Just make this point more explicit. 
- The term "transformer-based" perhaps oversells the point a bit that you use set transformer as summary networks.

### Questions
- The authors mention the aim of releasing a pretrained version of the model for free use. However, I am not sure if an "aim" is already a contribution of the paper. How far is the pretrained version?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes metabeta, an amortized Bayesian inference framework for mixed-effects
(hierarchical) regression. The method draws on techniques from simulation-based
inference (SBI), particularly neural posterior estimation (NPE) with normalizing flows
and permutation-invariant set encoders. By pretraining on many simulated hierarchical
datasets under varying priors, metabeta learns to approximate posteriors over both
global and local parameters of linear mixed-effects models (LMEs). At inference time,
users can provide their own priors and data, and the network outputs approximate
posteriors within seconds—potentially replacing MCMC for common hierarchical modeling
tasks. To improve calibration, the authors add post-hoc importance sampling (IS) using
the analytic LME likelihood and conformal prediction to adjust coverage. Experiments
compare metabeta to Hamiltonian Monte Carlo (HMC) on toy and real data, showing large
speed gains and competitive accuracy.

### Strengths
- Timely and practically motivated: The work targets a real bottleneck (computational
  cost of Bayesian mixed-effects regression) and adapts amortized SBI tools for this
  setting. This is an interesting application domain for amortized inference.
- Clear architecture design: The hierarchical set-transformer encoder combined with
  flow-based posterior heads is a reasonable and interpretable choice. The post-hoc
  importance sampling and conformal calibration steps are conceptually clean and
  computationally lightweight.
- Empirical performance: On toy and real hierarchical datasets, metabeta achieves
  parameter recovery and coverage comparable to HMC at a fraction of inference time. The
  results demonstrate that amortized models can produce usable posterior approximations
  in classical regression problems.
- Potential impact: If validated under fair comparison, such amortized inference could
  make Bayesian mixed-effects modeling far more accessible to applied fields (social
  sciences, bioinformatics, etc.) where MCMC remains the default.

### Weaknesses
### Conceptual framing and related work

- The paper is not truly an SBI setting: the LME simulator and likelihood are
  analytically known. metabeta uses SBI tools for amortized efficiency, not because
  inference is likelihood-free. This distinction should be made explicit and more prominent.
- The historical narrative around NPE and BayesFlow is inaccurate. Neural Posterior
  Estimation (also the amortized version) was introduced by Papamakarios et al. (2016) and extended by Lueckmann et
  al. (2017) and Greenberg et al. (2019). BayesFlow (Radev et al., 2020) later provided
  a practical amortized inference framework with set encoders, not transformers.
  Transformer-based amortized inference (e.g., Whittle et al., 2025; Mittal et al.,
  2025; Reuter et al., 2025) represents a distinct and more recent line of work. The
  related work section should reflect this chronology.
- The related work section omits key hierarchical SBI approaches such as Rodrigues et
  al. (NeurIPS 2021). The absence of these citations distorts context and weakens the
  claim of novelty.

### Methodology and baselines

- The HMC comparison is potentially unfair. Divergences in HMC typically signal poor
  tuning rather than algorithmic failure. [Non-centered parameterizations](https://sjster.github.io/introduction_to_computational_statistics/docs/Production/Reparameterization.html), robust
  step-size tuning, multiple chains, and R-hat diagnostics are standard. The authors
  should verify that best practices were followed; otherwise, the claimed accuracy
  advantage is not meaningful.
- Missing other baselines for hierarchical inference: The comparison currently focuses on HMC, but omits several established fast Bayesian or approximate inference methods that directly apply to mixed-effects models. For example: Variational Inference (VI) offers scalable approximate posteriors and would provide a relevant amortized or single-dataset baseline. INLA (Integrated Nested Laplace Approximation), a deterministic and highly efficient method for latent Gaussian models, widely used for Bayesian mixed-effects and spatial models; often matches MCMC accuracy at a fraction of the cost. Laplace / GLMM approximations — the classical second-order Gaussian approximation around the MAP, as implemented in standard GLMM software. Including these would contextualize metabeta's speed and accuracy gains relative to well-known fast alternatives rather than only to a potentially under-tuned HMC baseline.
- The posterior quality is inconsistent: in the toy example (Fig. 1C), metabeta produces irregular shapes for otherwise Gaussian-like posteriors. This raises questions about the flow architecture and whether the networks are overflexible or underregularized.
- Importance sampling is introduced without sufficient justification. If NPE is trained
  on the correct generative model, IS should not be required. The authors should explain
  whether IS corrects training–prior mismatch or residual amortization bias. Similar
  post-hoc IS refinements have been proposed in SBI (e.g., Dax et al., 2023) and should
  be cited.
- Evaluation metrics focus on parameter RMSE and correlation, which are not appropriate
  for Bayesian inference. The true parameter need not coincide with posterior means.
  Calibration metrics such as simulation-based calibration (SBC) or log-probability of
  the true parameters would be more informative.
- Runtime comparison is anecdotal (“orders of magnitude faster”) and lacks hardware
  disclosure or amortized cost estimates. Practitioners need wall-clock times on
  standardized setups.

### Presentation and reproducibility

- Figure 1 is difficult to interpret; caption variables do not align with figure
  notation. Posterior density plots based on kernel estimates obscure the fact that the
  NPE model defines a parametric PDF.
- Quantitative results: Table 1 lacks error bars or multiple-seed repetitions, making it
  unclear how stable the reported metrics are.
- Coverage statements (“good coverage”) are qualitative; comparative or numerical values
  are needed.
- The software contribution is underspecified. Code is “hidden” for review, but an
  anonymous repository is easily possible. Since the method’s accessibility depends on
  this, it should be part of the submission.
- Tone in the introduction (“prohibitively long inference times”) overstates the
  limitations of MCMC.

### Questions
1) Clarification of training data generation: How are the simulated training datasets
   (X, Z) constructed? Are they drawn from distributions intended to mimic real-world
   predictors, or are they generic Gaussian designs? How does the method generalize if
   real data differ strongly from the training simulations?
2) Role of real data: If the model is trained entirely on synthetic data, what exactly
   is the pipeline for applying it to real datasets? Are priors and covariate
   distributions assumed to match?
3) Calibration discrepancy: The paper attributes poor calibration to the forward-KL
   objective’s mass-covering tendency, whereas SBI literature (e.g., Hermans et al.
   2022), often finds flows too narrow. Can the authors reconcile this difference? Might
   the issue arise from too little training data or too simplistic flows? 
4) Alternative posterior networks: Could score-based or flow-matching estimators
   mitigate the need for IS and conformal calibration? These avoid the forward-KL bias
   entirely.
5) Importance sampling diagnostics: What are the effective sample sizes and weight
   variances of the IS correction? Without them, it is hard to assess stability.
6) Scope beyond Gaussian LMEs: Does metabeta handle generalized mixed-effects models
   (e.g., logistic, Poisson)? If not, how would the method need to change to support
   non-Gaussian likelihoods?

### Soundness
2

### Presentation
2

### Contribution
3
