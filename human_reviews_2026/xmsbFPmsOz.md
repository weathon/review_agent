# Marginal Girsanov Reweighting: Stable Variance Reduction via Neural Ratio Estimation

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Recovering unbiased properties from biased or perturbed simulations is a central challenge in rare-event sampling. Classical Girsanov Reweighting (GR) offers a principled solution by yielding exact pathwise probability ratios between perturbed and reference processes. However, the variance of GR weights grows rapidly with time, rendering it impractical for long-horizon reweighting. We introduce Marginal Girsanov Reweighting (MGR), which mitigates variance explosion by marginalizing over intermediate paths, producing stable and scalable weights for long-timescale dynamics. Experiments demonstrate that MGR (i) accurately recovers kinetic properties from umbrella-sampling trajectories in molecular dynamics, and (ii) enables efficient Bayesian parameter inference for stochastic differential equations with temporally sparse observations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In the study of molecular dynamics Monte Carlo simulations are frequently used to estimate the transition probabilities between metastable states. If the separating potential barrier is high such a simulation may take a long time. Modifying the potential V to V' can facilitate the climb over the energy barrier, but this perturbs the original statistics. Therefore a reweighting scheme is needed to compute the original transition probability from the one observed in the perturbed simulation. This reweighting factor can be written as an expectation, conditioned on the endpoints, of the Radon-Nikodym derivative of the original process driven by V with respect to the perturbed, simulated path driven by V'. 
    For two processes sharing the diffusion coefficient Girsanov's theorem furnishes an expression for this Radon Nikodym derivative, and a method using a time-discretized version of the theorem in the study of molecular dynamics has been proposed, analyzed and used by Donati et al (2017). The paper to review now points out that the logarithm of the Girsanov derivative suffers from an increase of variance both with underlying dimension and length of trajectory. This problem is then addressed by applying Girsanov's theorem as in Donati et al (2017), but only over short time intervals and extending it to longer trajectories with a machine learning technique. There a classifier, trained to detect the different between (already trained) pairs of joint distributions from the original and the perturbed, is used recursively to extend the estimate of the likelyhood ratio, beginning from short time intervals where the Girsanov method is more reliable.

### Strengths
The proposed method seems quite original and the paper presents an impressive number of experiments to support the proposed method. The problem considered seems to be of considerable practical importance.

### Weaknesses
1) The authors have chosen to reweigh the transition probability using Girsanov theorem. They improve previous results based on this method using a clever machine learning trick. They show that they can obtain results that were impossible to get with previous methods. However, there is another way that was recently pointed to recover the same kind of information as done in the present work: instead of reweighting the individual transition probability, one can learn the eigenfunctions of the infinitesimal generator ("Devergne et al. Neurips 2024, https://openreview.net/forum?id=TGmwp9jJXl&noteId=txQ3k2U5vR) by reweighting accordingly using the umbrella sampling formula <O> = <O e^{\beta V}> /<e^{\beta V}>. It seems important to cite and compare to the method in this paper.

2) The description of the method is rather opaque. The back and forth between continuous and discretized processes as in appendix A (variance explodes) and other places is confusing.
    
3) The theoretical support for the method is very weak. In particular the authors fail to explain how the noise inherited from the discretized Girsanov method does not accumulate in the recursive training procedure. 
    
4) The use of the strong law of large numbers in appendix C is not justified by any independence assumptions. For long trajectories, are there any mixing assumptions on the original and/or the perturbed process? Are the multiple trajectories at least independent?

### Questions
See weaknesses above.

In line 128 it should be $t \in [T - \tau]$ instead

At the end of line 207 shouldn't it be $(x_t,x_{t+k\tau})$?

(4) is not really an identity, because (3) is not.

Line 731: "We consider two probability on $\mu$..." delete \mu and parentheses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a machine learning method for performing marginal Girsanov Reweighting. The idea is simple and well-executed, demonstrating benefits on a suite of benchmarks. While simple, the proposed method could substantially improve rare event sampling in molecular dynamics (MD) and provide an alternative method for performing Bayesian posterior inference for stochastic differential equations (SDE) such as the Lotka-Volterra model. I have minor critiques and questions.

### Strengths
The authors clearly explain the method and relevant background, such as MD. The exposition of Girsanov reweighing is clear and discussion of related work is clear, and, to my knowledge, comprehensive of relevant papers. The paper's novelty is conceptual clarity and practical stabilization.
- Their SDE posterior point estimate could be used in Simulation-based inference (SBI) and we recommend the authors consider applications there.

### Weaknesses
The authors address many of the limitations of the paper in the choice of the neural network used as the classifier. I address a few minor critiques below.
- Not really Bayesian inference of the SDE if you're estimating a point value in the SDE problems
- Nit-pick importance sampled expectation is missing marginal likelihood in expansion of the posterior but that cancels with the introduction of reference distribution.
- Saying hypothesis in line 410 is a little confusing as I was ready for a hypothesis test. Threw me off for a second thinking what you meant and might be more straightforward to say "ground-truth" or "reference true parameter" instead.
- Related to the SBI note, while outside the scope of this paper, calibration of parameter estimates of SDEs in Bayesian posterior estimation is an important consideration and is missing here. Examining the performance of the posterior parameter estimate on a held-out true point estimate is a good start but often in real-world SDEs there's a requirement for calibration since ground truth is never really known. The method actually provides a classifier in the vein of classifier-based SBI that could adapt already-existing calibration checks for SBI classifiers; but, this is a nice-to-have.
- The authors mention TRAM as an alternative method for transition-based reweighting along with GR, so why wasn't that evaluated? This seems Important to compare.

### Questions
- The paper relies on sampling endpoint pairs $(x_t, x_{t+k\tau})$ from trajectories under the perturbed dynamics $\tilde{\mu}$ and uses the Strong Law of Large Numbers (SLLN) to justify convergence of weighted averages. How robust is MGR when the perturbed dynamics are not ergodic, or when the sampled trajectories do not adequately explore the full state space?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces **Marginal Girsanov Reweighting (MGR)**, a method that combines classical stochastic reweighting theory with modern neural ratio estimation.   The classical **Girsanov reweighting (GR)** technique allows expectations under one stochastic differential equation (SDE) to be recovered from simulations under a different, typically more tractable SDE. However, as also shown by the authors GR suffers from **exponential variance growth** with time horizon, making it unstable for long trajectories. MGR mitigates this by **marginalizing over intermediate paths**: instead of pathwise weights, it learns **endpoint-based density ratios** using a **classifier-based neural ratio estimator**. The method bootstraps short-lag GR weights (where the variance is still manageable) to compose longer-lag ratios iteratively, yielding stable, scalable weights. Experiments on molecular dynamics (umbrella-sampled four-well and alanine dipeptide systems) and Bayesian SDE inference (Ornstein–Uhlenbeck and Lotka–Volterra models) show that MGR produces accurate long-horizon estimates where GR fails.

### Strengths
- **Well written and methodologically sound**: The paper presents a technically solid and clearly explained method, bridging stochastic analysis and modern density ratio estimation. The theoretical results (variance analysis, consistency proofs, and training algorithm) are rigorous.
- **Comprehensive empirical evaluation** The two application domains—molecular dynamics and Bayesian SDE inference—are complementary and well chosen. Results are convincing: MGR maintains higher effective sample size (ESS), better implied timescales, and accurate posterior estimates, even when GR and standard inference methods fail.

### Weaknesses
- **Related work:** The approach (specifically in the context of Bayesian inference for SDEs) is very much related to simulation-based inference [1] methods (NRE, NLE and specific specialization of these for SDEs). These should be atleast discussed in related work. Specifically [2,3] also a targeted towards Bayesian inference on SDEs and also are motivated by learning a certain property of transition densities that can be used to perform inference.
[1] Cranmer, Kyle, Johann Brehmer, and Gilles Louppe. "The frontier of simulation-based inference." _Proceedings of the National Academy of Sciences_ 117.48 (2020): 30055-30062.
[2] Gloeckler, Manuel, et al. "Compositional simulation-based inference for time series." _arXiv preprint arXiv:2411.02728_ (2024).
[3] Cai, Xin, et al. "Simulation-based transition density approximation for the inference of SDE models." _arXiv preprint arXiv:2401.02529_ (2023).

- Motivation can be made clearer:  The overall motivation can be made more clear for readers not familiar with GR or molecular dynamics some parts are somewhat confusing i.e. "direct simulation from the target law is often infeasible ..., which motivates us to introduce a perturbation ... perturbed dynamics are  easier to simulate ..." (page 3, top). But then GR reweighting requires anyway to evaluate both drifts $f$ and $\tilde{f}$  along the trajectory which has the same cost than the unperturbed process... . The answer, is somwhat implicit in the text, should be stated explicitly: *direct simulation of the unperturbed SDE may be computationally infeasible due to slow mixing or interest in rare events (under the target law)*. GR then allows to bias towards e.g.  rare events and correct it later.

### Questions
- For the Bayesian inference experiments, a _reference process_ must be defined, i.e., a particular choice of parameters and initial conditions under which the trajectories are simulated. While the manuscript specifies which reference settings were used, it does not explain **why** these choices were made or how they affect performance. Since the stability and effectiveness of MGR likely depend on how close the reference dynamics are to the true (target) parameters, a discussion of **how to select or tune the reference process in practice** would be very valuable. In its current form, it is unclear how one would make this choice from scratch, or how sensitive the method is to this design decision.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes marginalised Girsanov weighting method, where the standard Girsanov weighting between trajectories is extended to between measures.

### Strengths
The method is principled and well motivated. The method is original, and seems to be effective at small demonstrations. The paper is overall clearly presented.

### Weaknesses
The work does not demonstrate much significance. The results are toy'ish small demonstrations, where the innovation does work over the vanilla method. There is however no "real-world" experiments, or any comparisons to competing methods other than the vanilla Girsanov. 

The novelty is also somewhat limited as the work combines known techniques in the ratio estimation.

The work could also improve clarity by visualising the setting with some SDE paths or measures that highlight the problem of using drift f, and show the need to move towards tilde{f}

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
