# PriorGuide: Test-Time Prior Adaptation for Simulation-Based Inference

- Avg Score: 6.80
- Decision: Accept (Poster)
- Scores: 6, 10, 6, 8, 4

## Abstract
Amortized simulator-based inference offers a powerful framework for tackling Bayesian inference in computational fields such as engineering or neuroscience, increasingly leveraging modern generative methods like diffusion models to map observed data to model parameters or future predictions. These approaches yield posterior or posterior-predictive samples for new datasets without requiring further simulator calls after training on simulated parameter-data pairs. However, their applicability is often limited by the prior distribution(s) used to generate model parameters during this training phase. To overcome this constraint, we introduce *PriorGuide*, a technique specifically designed for diffusion-based amortized inference methods. PriorGuide leverages a novel guidance approximation that enables flexible adaptation of the trained diffusion model to new priors at test time, crucially without costly retraining. This allows users to readily incorporate updated information or expert knowledge post-training, enhancing the versatility of pre-trained inference models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes PriorGuide, a method to adapt diffusion-based amortized
simulation-based inference (SBI) models to new priors at test time without
retraining. The key idea is to express the target-posterior score as the original score
(under the training prior) plus a guidance term involving the prior ratio
$(r(\omega)=q(\omega)/p_{\text{train}}(\omega))$. To make the guidance tractable, the
reverse transition kernel is approximated as Gaussian and the prior ratio is
approximated by a Gaussian Mixture Model (GMM), which yields a closed-form guidance
update. The method is evaluated on a suite of SBI problems for both posterior and
posterior-predictive inference. Empirically, PriorGuide attains competitive performance
relative to baselines, and the paper further studies a compute–accuracy trade-off
using interleaved Langevin refinement.

### Strengths
**Quality.** The paper’s theoretical foundation is solid: the score decomposition with a
guidance term is standard and correctly instantiated for the SBI setting. The Gaussian
reverse-kernel and GMM prior-ratio approximations are reasonable and enable a practical
algorithm. The analysis of test-time compute via Langevin refinement is appropriate, and
the extended experiments cover canonical SBI tasks. The ablation and sensitivity studies
in the appendix further strengthen the empirical grounding.

**Clarity.** The manuscript is exceptionally well written, with well-structured
exposition and clear intuition for the guidance term. Notation is introduced early and
consistently; figures and tables are easy to interpret. The appendix integrates detailed
methodological clarifications—GMM fitting, diagnostics, and runtime analyses—that
improve reproducibility. The discussion of the Pareto front between diffusion and
Langevin steps is particularly insightful.

**Significance and Contribution.** The work addresses a practically important
problem—changing priors after training—where retraining amortized SBI models can be
prohibitively expensive. Adapting diffusion guidance to encode new priors at inference
is an appealing and useful contribution. The empirical scope is strong, especially given
the additional baselines in the appendix (Rejection Sampling, Sequential Importance
Resampling, NLE+MCMC), a sensitivity analysis, and a 20D test demonstrating scalability.
The contribution remains primarily empirical but offers meaningful methodological value
for the SBI community.

**Originality.** Applying diffusion guidance to test-time prior adaptation in amortized
SBI is a natural but novel design choice. The GMM ratio approximation provides a
practical mechanism to operationalize this guidance, and the extended analyses confirm
that the method maintains competitive performance where simpler baselines fail.

### Weaknesses
**Scope and positioning.** While the paper clearly demonstrates PriorGuide’s strengths,
it could offer more explicit practical guidance for practitioners—specifically, when
PriorGuide should be preferred over simpler baselines such as NLE + MCMC, RS, SIR, or
related methods like SIMFORMER and ACE and how to balance its additional test-time
computation against retraining or posterior-correction methods. While this is discussed
qualitatively in the appendix, a concise summary in the main text would make the
contribution more actionable.

**Dimensionality constraints.** Although the new 20D Gaussian Linear results demonstrate
scalability, these remain relatively simple benchmarks. A more complex or real-world
high-dimensional example would strengthen the claim of general applicability.

**Approximation limits.** The Gaussian reverse-kernel approximation, while adequate for
the reported tasks, remains a simplifying assumption. Even with Langevin refinement
mitigating some of its effects, further diagnostic guidance for practitioners would be
useful.

### Questions
1. Could the authors provide clearer practical guidance on when PriorGuide is
   preferable to simpler or amortized alternatives? For instance, how should
   practitioners trade off its added test-time computation against retraining costs or
   simpler posterior-correction methods?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
PriorGuide introduces an extension to framework of simulation-based inference (SBI) with diffusion models that allows pre-trained diffusion models to incorporate new prior distributions at inference time without requiring retraining. This is achieved using a novel Gaussian mixture model approximation to make the target prior tractable for guiding the diffusion sampling process. Empirical results on synthethic data demonstrate PriorGuide’s effectiveness in accurately recovering posterior and posterior-predictive distributions across various SBI problems. The method also allows for refinement through Langevin dynamics, enabling a balance between computational cost and accuracy.

By enabling adaptation to new priors without retraining, PriorGuide exemplifies a “test-time compute” approach, extending the capabilities of pre-trained models with targeted computations. This decoupling of simulator runs from prior specification offers practical benefits especially in the case where simulations are costly, including the ability to perform post-hoc prior sensitivity analyses and incorporate domain expert knowledge after training, ultimately reducing the computational burden of scientific workflows which employ SBI.

### Strengths
The manuscript proposes an extension of SimFormer modeling to the challenging problem of representing complex joint probability distributions. The core idea is relatively straightforward, yet the authors provide a rigorous mathematical treatment that appropriately identifies both the potential benefits and inherent limitations of this approach, acknowledging the necessary reliance on approximations. The empirical evaluation is a strength of the paper; the results presented demonstrate promising performance, particularly in the accuracy of the reported uncertainty estimates (Tables 1 & 2). The selection of benchmarking use cases is clearly motivated and relevant to the stated goals. Finally, the authors effectively utilize concise and informative visualizations to support their claims and illustrate the underlying hypothesis.

### Weaknesses
The manuscript would benefit from a more careful review of its language, as instances of anthropomorphism (e.g., “the model having seen few training examples”) can detract from the scientific rigor. Additionally, the presentation of the derivation for 'r' is hampered by misaligned page breaks, separating key explanatory text and hindering comprehension. More significantly, while the paper thoroughly explores the technical capabilities of PriorGuide, it lacks a discussion of the broader implications of manipulating the prior distribution within a Bayesian framework. The authors appear to assume a familiarity with this interplay between prior, simulator, and the scientific endeavor, which may limit the accessibility and impact of the work for a wider audience.

### Questions
- page 4, the derivation of eq 6-9 is central to the paper, please try to rearrange so that the text in lines 216-220 is readily visible close the equations (e.g. by removing lines 191-195)
- the code for the experiments was not found in the paper, i.e. through an anonymised git repo, this should be corrected for publication
- line 232: please do not use anthropomorphic language to describe model behavior "the model having seen few training examples", i.e. the model has consumed or been trained on
- line 234: "using standard OOD metrics" as such a standard exists only colloquially, please remove "standard"
- equ 13: the text lacks a discussion if `K` is a free parameter and how to choose it
- line 274: "Notably, when p(θ) is uniform, r(θ) reduce*s* to q(θ), which can be directly specified as a Gaussian mixture." perhaps also add hint why r reduces to q (due to using the expectation I guess)
- line 310: "To incorporate this with regular diffusion ..." perhaps comment on the computational cost here already.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
**PriorGuide** targets a practical SBI need: adapting a trained diffusion-based posterior estimator to **new priors at test time** with **no retraining**. The paper derives a clean **score decomposition** in which the target posterior score equals the trained score plus a **prior-ratio guidance** term . To make this usable, the authors (i) use a *standard* Gaussian reverse-kernel approximation (via Tweedie’s formula) and (ii) fit a Gaussian mixture to the prior ratio, yielding a *novel* closed-form guidance update that plug directly into the diffusion sampler. Optional few-step Langevin corrections at low noise tighten asymptotics and expose a neat compute–accuracy knob alongside the diffusion steps. Empirically, PriorGuide improves both posterior and posterior-predictive metrics across several SBI tasks while keeping the implementation lightweight and training unchanged.

### Strengths
- **Well written easy to follow**:The derivation is easy to follow; the GMM prior-ratio leads to implementable, closed-form guidance. Propositions are sound and proven in the Appendix.
- **Good empirical results with clear trade-offs and ablations**: Consistent gains on posterior and predictive metrics; straightforward ablations over diffusion/Langevin steps make hyperparameter selection transparent. The paper is upfront about approximation choices and where Langevin steps matter, which helps practitioners reason about when to use the method.

### Weaknesses
- **Prior family diversity**: Main experiments emphasize changes among Gaussian priors (often **diagonal covariance**). This is required for fair ACE comparisons but does not demonstrate  the claimed method’s generality. A demonstration on **non-factorized** or more structured priors (complicated priors) would strengthen external validity. For example one use-case that requires flexible priors would be for "i.i.d" data which also can be handled by sequentially updating the "prior" with the "current posterior".

### Questions
1. **Non-factorized complex priors**: Have you tried dense-covariance Gaussians or general complicated priors? Have you tried inference on iid data using PriorGuide?
2. **Better Gaussian reverse Kernel approximations**: There exists better reverse kernel approximations (i.e. [1] and related). These usually are more costly i.e. require Jacobians or other terms. Would be interesting to see if these also help in that case but atleast should be discussed in the manuscript.
3. **Matrix calculations**: The authors state that a current limitation is that "matrix-operations  may-pose a scalability issue". A straight forward approach to avoid this is to constrain the involved covariance to be *diagonal* (or other approx.). After all a mixture of Gaussians with diagonal covariance is still a universal approximation family (just may needs more components). As the backward kernel approximation is by design also diagonal the whole Guidance terms are reduced to the diagonal. This hence makes *PriorGuideDiag*  an interesting alternative to the current approach.

[1] Boys, Benjamin, et al. "Tweedie moment projected diffusions for inverse problems." _arXiv preprint arXiv:2310.06721_ (2023).

### Soundness
3

### Presentation
4

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
The paper consideres an interesting problem of adapting a diffusion model pretrained to sample from a posterior when a different (unknown at the training state) prior model is adopted by the user. The proposed algorithm, PriorGuide, is able to correct biase of the pretrained diffision model during the sampling stage and providing a test time adaptive inference. 

The guidance is intractable but is proximated using Gaussian mixture models. 

The experiments show that on a wide variety of bechmark datasets, the proposed method outperforms similar diffusion methods with no-prior adaptation.

### Strengths
Overall, I think this paper makes a solid contribution to the SBI community. 

The problem of changing priors is indeed common in many applications, and the ability to adapt a pre-trained posterior model without retraining the entire model is highly desirable and can make a lot of sense when the user's prior is different and comes in at a later stage. 

Diffusion guidance is a well-studied technique in the generative model adaptation, so it is natural to leverage this approach to adapt SBI diffusion models. To my best knowledge, the proposed idea is novel. 

The proposed method is mathematically sound, and its resulting formulation is natural, separating the pre-trained model from the guidance term (the expected prior shift). 

Although the guidance term itself is intractable, authors show it can be effectively approximated.

### Weaknesses
My concerns are mostly on the literature review: 

In the main text, the authors do not provide a comprehensive review or comparison of prior adaptation methods in the existing literature.  As briefly mentioned in the introduction and the experimental section, there are already methods that consider adaptive priors. Although the authors include a separate section in the appendix, a dedicated subsection in Section 2 would be appreciated, as readers would like to understand the specific limitations of these prior adaptation methods and how (or whether) they relate to the proposed guidance method.

In Section 2, recent progress (2022 ~ ) on applying diffusion models to SBI should also be comprehensively reviewed and cited as they provide the backdrop (the pre-trained posterior score) to the proposed method. To the best of my knowledge, the earliest attempt to use diffusion models in SBI is often attributed to Geffner et al. (2023), first published on arXiv in 2022. Simformer is a subsequent work built on Geffner et al. and others, introducing masks and a transformer architecture to handle missing observations and more versatile inference tasks.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers the amortized simulator-based inference framework and one of its major limitations, i.e., prior distributions used to generate model parameters during training. The authors propose a novel method, PriorGuide, which leverages a guidance approximation for flexible test-time prior adaptation. The method was evaluated on a set of benchmarks. The most significant strength is requiring no retraining.

### Strengths
The paper has a few strengths:
1. The "prior-rigidity" of amortized SBI models is a well-known and significant bottleneck for scientific application. This paper offers a direct, "plug-and-play" solution that enables, for the first time, post-hoc prior sensitivity analysis and the incorporation of new expert knowledge using expensive, pre-trained models.
2. The method is built on a sound mathematical foundation. The derivation of the target posterior as a tilted distribution (Proposition 1) is clear and correct.
3. The experiments are convincing. The paper correctly identifies that its method is an approximate sampler and intelligently introduces Langevin correctors to manage the accuracy-compute trade-off.

### Weaknesses
However, despite it’s strengths, there are also weaknesses:
1. The entire method hinges on a critical, and likely fragile, approximation: fitting the prior ratio $r(\theta) = q(\theta) / p_{train}(\theta)$ with a GMM. The paper understates the difficulty of this step. This is a density-ratio estimation problem, which is notoriously difficult, especially as the dimension of $\theta$ increases. The paper's solution (a gradient-based $L_2$ fit) is a heuristic that is not guaranteed to be stable or accurate, particularly if $p_{train}(\theta)$ is non-trivial.
2. The paper employs two major approximations: the GMM for the ratio and a simple diagonal Gaussian for the reverse kernel. It is the combination of these (especially the latter) that biases the sampler and requires the heavy use of Langevin correction steps to achieve good results. This means the method, while "retrain-free," has a high and non-trivial per-sample inference cost.

### Questions
Besides the Weaknesses, I have also a question:

1. The GMM approximation of the prior ratio $r(\theta)$ appears to be the most critical (and fragile) part of the pipeline. How does the $L_2$ fitting procedure scale with the dimensionality of $\theta$? Have you considered replacing this heuristic fit with more robust, modern density-ratio estimators (e.g., flow-based, MINE, etc.)?

### Soundness
3

### Presentation
3

### Contribution
2
