# Manifold-Constrained Gaussian Process Inference for One-shot Learning of Unknown Ordinary Differential Equations

- Avg Score: 3.00
- Decision: Reject
- Scores: 6, 2, 2, 2

## Abstract
Learning unknown ordinary differential equations (ODEs) from a single trajectory of scarce, noisy data is challenging, especially with partial observability. We introduce MAGI-X, an integration-free framework that couples a neural vector field with a Gaussian process prior over trajectories and enforces ODE consistency via a GP manifold constraint, thereby circumventing traditional numerical integration. Across canonical examples (FitzHugh--Nagumo, Lotka--Volterra, and Hes1), MAGI-X achieves better accuracy in both fitting and forecasting while requiring comparable or less computation time than benchmark methods NPODE and Neural ODE, with runtime scaling linearly in state dimension. MAGI-X offers a practical solution for \emph{partially observed} systems without bespoke priors or imputation heuristics, where existing methods struggle. The GP posterior further yields calibrated uncertainty, and experiments demonstrate robustness across initial conditions. We show practicality on seasonal flu data with rolling multi-week forecasts from noisy signals.
 These properties establish MAGI-X as a fast, accurate, and robust tool for data-driven discovery of nonlinear dynamics from a single noisy trajectory.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper suggests a framework to learn a semi-linear system of ODEs and use the learned system to make predictions into the future. This is achieved by traning a Gaussian processes on (potentially little) observed data and modeling the ODE via a neural network.

### Strengths
- I like the idea of the paper, in particular the probabilistic approach and the combination of easy techniques to construct a powerful framework.
 - I like the diversity of experiments.
 
In general, I guess the paper would be a reasonable addition to the ICLR community.

### Weaknesses
- I think the abstract contradicts the introduction. The abtract claims to learn an ODE, whereas the introduction stresses predictions using the learned ODE. Which one is it? (Probably both, which is fine. But the wording is confusing.)
 - Reading the paper for the first time was very hard, while reading it for a second time was rather easy, at least for me. I think the paper makes a suboptimal introduction into its approach. 
 - I do not get why you call give the name "manifold constraint" to "W=0". What does this have to do with a manifold? It is an equality of functions.
 - The training approach seems finicky.

### Questions
- Do you need the assumption of a semi-linear ODE as stated in (1)? Either say so early and add this assumption or tell me how you could do without this assumption.
 - Which form of a neural network do you use for $f$? (I am asking because ODEs care about differentiability of $f$.)
 - While (5) cannot be computed symbolically, there are good (and quick to compute) approximations like [Tebbe et al., Efficiently Computable Safety Bounds for Gaussian Processes in Active Learning]. Would such a bound help?
 - In Algorithm 1, why to you do blockwise updates instead of a joint update?
 - It seems that there are few comparisons to baselines. Can you comment?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes MAGI-X, a method for learning unknown ODE systems from a single noisy trajectory without numerical integration. The approach combines a neural network parameterization of the vector field with a Gaussian process prior over trajectories, enforcing ODE consistency via a manifold constraint. The authors demonstrate competitive or superior performance compared to NPODE and Neural ODE on three benchmark systems (FitzHugh-Nagumo, Lotka-Volterra, Hes1), with faster runtime and natural handling of partial observations.

### Strengths
1.The paper demonstrates that replacing GP modeling of f with neural networks can reduce computational complexity in the MAGI framework, particularly for higher-dimensional systems where the cubic cost O((DM)³) becomes problematic.

2.The integration-free training approach avoids the computational overhead of repeated ODE solves, which provides practical benefits compared to methods requiring numerical integration.

3.The GP framework naturally handles partial observations with asynchronous measurement times, which is demonstrated in challenging settings where only one component is observed at each time point.

4.The writing is generally clear and the paper is well-organized with extensive supplementary materials.

### Weaknesses
1. The core contribution is severely limited. The paper essentially takes the MAGI framework from Yang et al. (2021a) and replaces the GP modeling of f with a neural network. This is acknowledged in Line 206: "This matches the two-step approach of Ridderbusch et al. (2020), but with a neural model for f*." The manifold-constrained GP objective, the treatment of partial observations, and the overall inference framework are entirely inherited from MAGI. The alternating optimization between x(T) and θ is a standard block coordinate descent procedure without theoretical justification. There is no fundamental conceptual advance or novel insight that would constitute a significant contribution to the field.

2. The experimental comparison is critically incomplete and appears designed to avoid the most relevant baselines. The paper only compares against NPODE (Heinonen et al., 2018) and Neural ODE (Chen et al., 2018), both from 2018 and both requiring numerical integration. The most important missing comparisons are: (a) MAGI with GP modeling of f (Yang et al., 2021a) - the parent method from which this work directly inherits its framework. The authors claim O((DM)³) complexity is prohibitive but provide no empirical evidence. For the small systems tested (D=2-3), GP modeling may be perfectly viable and potentially more accurate. Not comparing against the original method while using its framework is academically problematic. (b) Ridderbusch et al. (2020) - the paper explicitly models its initialization after this two-step gradient matching approach and claims alternating optimization is more robust (Line 097), yet provides no experimental evidence. This is the most directly comparable baseline as it also uses gradient matching without integration. (c) Other gradient matching methods mentioned in related work (Heinonen and d'Alché-Buc 2014, Wenk et al. 2019, 2020) are cited but not compared.

3. The experimental setup exhibits unfairness that undermines the claimed advantages. MAGI-X uses 2500 iterations (Appendix C.2) while NPODE and Neural ODE are limited to 500 iterations, with Neural ODE specifically capped "due to its longer runtime." This makes runtime and accuracy comparisons meaningless. When NPODE is given 2000 iterations (Table 4), its performance improves substantially, with LV forecasting matching or exceeding MAGI-X on some metrics. This suggests the baselines may be undertrained. The comparison strategy appears to showcase MAGI-X against slow integration methods while avoiding fast integration-free alternatives.

4. The optimization exhibits significant instability requiring multiple restarts. The paper acknowledges divergence issues (Figure 8, Appendix C.1) where "small errors accumulate in the trajectory propagation via numerical integration in the forecasting phase." The high-dimensional experiments show 10-11% divergence rates (Table 6), defined as forecasting RMSE exceeding 5. The solution proposed is to "restart MAGI-X with multiple random seeds," which is a workaround rather than a principled solution. No convergence analysis is provided, and the root cause of instability is not addressed. The method relies heavily on heuristic learning rate schedules (l^(-0.6)) and engineering tricks (time standardization, Cholesky parameterization) that lack theoretical justification.



5. The scalability claims are not well supported. While Table 5 shows linear scaling in runtime, the synthetic high-dimensional experiments (Table 6) reveal degraded performance with higher divergence rates. The main experiments focus on 2-3 dimensional systems, which are too small to validate the claimed advantage over GP-based methods. For small D, the O((DM)³) complexity of MAGI-GP may not be prohibitive, making the motivation for neural network parameterization weak in the tested regime.

6. The real-world application (Section 3.6) provides only qualitative assessment. The influenza forecasting shows no baseline comparisons, no quantitative metrics, and no comparison with standard epidemiological models. The visual results show some overshoots and undershoots that are difficult to evaluate without proper benchmarking.



7. Several implementation details appear to be ad-hoc engineering fixes rather than principled choices. The time standardization scheme (Appendix C.4) that scales all systems to a time range of 8 is acknowledged as "an engineering solution" with "a better standardization procedure to be investigated in future works." The choice to freeze GP hyperparameters after initialization may limit adaptability. The tempering parameter |T|/N_d lacks principled selection criteria.

### Questions
1.Why is there no comparison with the original MAGI method (Yang et al., 2021a) using GP modeling of f? You claim O((DM)³) is prohibitive, but provide no empirical evidence for D=2-3 systems. Without this comparison, it is impossible to assess whether neural networks actually improve upon the parent method.

2.Why is Ridderbusch et al. (2020) omitted from experiments? You explicitly state that your initialization replicates their two-step approach (Line 206) and claim alternating optimization is more robust (Line 097), yet provide no supporting evidence. This is the most directly comparable baseline.

3.MAGI-X uses 2500 iterations while baselines use only 500. Can you provide fair comparisons with equal computational budgets? Table  shows NPODE improves substantially with more iterations, suggesting your baselines are undertrained.

4.What is the fundamental advantage of alternating optimization over simpler two-step approaches beyond using more iterations? Can you provide ablation studies isolating the benefit of alternating versus just training longer?

5.What causes the 10-11% divergence rate in high dimensions requiring random restarts? Can you provide convergence guarantees or at least characterize when and why the method fails?

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
The paper presents a integration-free framework (MAGI-X) that couples neural vector fields with Gaussian process priors via a
manifold constraint, unifying GP-based and neural ODE paradigms.

### Strengths
The paper is well-organized, so it is not hard to follow the general idea. The paper introduces MAGI-X, an integration-free framework that combines neural vector field learning with Gaussian process priors through a manifold constraint. This unified formulation not only bridges GP-based and neural ODE approaches but also handles partially observed data by leveraging the probabilistic structure of the GP prior.

### Weaknesses
The main concerns are:

* The proposed MAGI-X framework is conceptually very close
to Yang et al., PNAS 2021 (“Inference of Dynamic Systems via
Manifold-Constrained GPs”). Both share the same probabilistic manifold-constrained formulation and similar optimization strategy; MAGI-X's primary innovation in my eyes is replacing the vector field parameterization with a neural network. Consequently, the methodological novelty may appear very incremental, a point that is not fully acknowledged in the main text. The authors must explicitly and carefully delineate their advance over Yang et al. to clarify the true contribution of this work.

* To better position the contribution of MAGI-X, the experimental section should be expanded to include comparisons with more recent benchmarks. The current analysis, which relies on methods from 2018 (Neural ODE and NPODE), does not fully capture the current state of the field and may overlook more competitive alternatives.

### Questions
* Regarding the learning of GP hyperparameters, using the noisy observations y directly to fit the GP seems like a bad approximation. Could the learning framework be extended to jointly learn the GP hyperparameters and the neural network parameters, rather than treating them separately?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a method for learning the dynamics of an unknown system given corresponding time-series data. They model the time-derivative of the system using a neural network and  full trajectories using a GP. During training, they impose an approximate manifold constraint on the trajectories, such that they match the learned time derivative. Training takes place by alternating between generating input output pairs using the GP, which are then used to train the neural network, which in turn is used to retrain the GP.

### Strengths
The authors address an important and challenging problem. The paper is easy to read and their choices are well motivated.

### Weaknesses
The statement in line 51 that f* is completely unknown is misleading. The authors model it using a NN and a GP, which implies some regularity assumptions. This is not the same as a completely unknown function.

The considered benchmarks correspond to fairly low-dimensional systems. This leaves questions with respect to the scalability of the method. Moreover, the considered baselines are relatively old, and it is unclear how the method compares to newer baselines.

The paper does not have a section dedicated to discussing its limitations.

### Questions
Is the notation in eq. (4) correct?

Line 117: What guarantee are the authors referring to here? The paper only contains theoretical results in the appendix.

Line 326: This sentence is incorrect.

The authors assume that the dimensions in X are conditionally independent. is this problematic?

What is the motivation for allowing "different components to be observed at different sets of time points"? This is a somewhat unusual definition of partial observability, since it implies that some of the states can always be observed. How does the proposed method perform if none of the states are directly observed (i.e., they are a latent variable that has to be inferred)?

What is the motivation for learning the noise variances but not the remaining hyperparameters during training? Is it only efficiency? It seems to me that this may overfit the noise.

### Soundness
2

### Presentation
3

### Contribution
2
