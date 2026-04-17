# Injecting Sensitivity Constraint Into Continual Learning Significantly Enhances Surrogate-Aided Optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 6

## Abstract
A myriad of scientific and engineering optimization and learning tasks involve
running a numerical model to guide optimization directly or generate training
data for function mapping algorithms. Surrogate models can greatly accelerate
these tasks, but they often fail to capture the true input-output relationships (sensi-
tivities) so they lose the ability to guide high-dimensional and long-horizon op-
timization. Online continual learning (OCL) – iteratively obtaining numerical
results to continue training the surrogate – can mitigate this issue, but may still
be insufficient. Here we propose scheduled injection of sensitivity constraints
(SC, matching the Jacobian of the surrogate model with that of the true numer-
ical model) for the surrogate into OCL to enforce realistic output-parameter re-
lationships. We evaluate this approach across diverse datasets and optimization
frameworks where continual surrogate training is used: (1) multi-objective multi-
fidelity surrogate-assisted Bayesian optimization and Pareto front exploration; (2)
hybrid end-to-end training of coupled neural networks and process-based mod-
els; and (3) a modified unifying framework for generative parameter inversion
and surrogate training. Across all of these tasks, inserting SC accelerates the de-
scent to optimality and consistently improves the main optimization outcome, as
it critically improves the future trajectory of optimization. OCL improves data
relevance and SC ensures sensitivity fidelity, and they together produces an ef-
ficient surrogate model that almost achieves the same effect as the full physical
model, only achievable by OCL+SC. It consistently outperform pretrained-only
surrogate models with SC or OCL without SC, not to mention the pretrained-only
model without SC, so the benefits of two procedures reinforce each other. Even
infrequent surrogate finetuning with SC injection (once every 5 epochs) can in-
duce large benefits in optimization outcome. Together, these results demonstrate
the possibility to enable large-scale optimization of complex systems for big-data
learning and knowledge discovery

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors study how to introduce Jacobian matching between surrogate and oracle functions into online continual learning - specifically in how to improve the surrogate model by matching not only the function values, but also their derivatives. The method is explored across 4 different tasks, including a combination of both synthetic benchmarks and real-world environments.

### Strengths
1. The contribution of this work is interesting and intuitive.
2. I appreciate the ablation over the frequency of SC injections, and also the experiments on how the method affects the overall runtime.

### Weaknesses
3. The sensitivity loss weight hyperparameter lambda seems to be manually tuned per task. Since gradient magnitudes differ dramatically across models and domains, performance may be sensitive to lambda. There is no relevant ablation study or discussion of this limitation from what I can see.
4. It is challenging for me to understand the Introduction within the context of the larger scope of this work, and required multiple reads at least on my end. For instance, what was the point of discussing knowledge discovery in the second paragraph? Catastrophic forgetting is also not explored at all in this work.
5. The sensitivity loss term necessitates the ability to compute (or at least approximate) first order derivatives of the true numerical model. This is not necessarily the case in practice (e.g., discrete design spaces, wet-lab experiments, MD simulators, etc).
6. $n = 5$ seeds is too few to make any meaningful conclusions – for example, I believe the 95% CIs for Offline + SC and OCL + SC (2) would overlap for the $R^2$ metric for the Hydrologic Model in Table 1.
7. There doesn’t seem to be actual experimental evidence of catastrophic forgetting in the paper. Did the authors observe any signs of catastrophic forgetting or degradation in performance outside the immediate optimization regions (e.g., checking overall $L_2$ error on the entire initial buffer B) in the OCL-only versus OCL+SC cases?
8. The manuscript would generally benefit from additional proofreading and editing. I generally do not feel super strongly about having a submission being 100% perfect in terms of grammar and spelling, but there are a large number of grammatical mistakes in this work that made it challenging to read through and required a couple of iterations to understand. I started detailing a few below in the "Minor Comments" section, but stopped somewhere in the middle so the list is by no means exhaustive.
9. I understand that the SC loss term is not the main contribution, but rather empirically studying how it can be effectively incorporated into OCL in different settings. However, this was only studied in only 3-4 applications, and the ablation studies are significantly lacking (e.g., see point 1 above, also ablating buffer size, number of warm-up epochs, dataset sizes, Jacobian approximation accuracy, etc). This makes my enthusiasm for the empirical contribution of this work significantly tempered.
10. I would also consider adding experience replay, elastic weight consolidation, and gradient episodic memory as baseline methods to compare against.

Minor Comments:
 - Abstract: "they together produces" should be "they together produce"
 - Abstract: "an efficient surrogate model" - I don’t think the authors mean "efficient" here, should it be "effective"?
 - Abstract: "consistently outperform" in line 31 should be "consistently outperforms"
 - Line 68: I’m not quite sure what "rigidity" means here.
 - Line 77: "infinite-dimension" should be "infinite-dimensional"
 - Line 90: "solely the recent examples" should be "solely on the recent examples"
 - Line 181: I don’t think HV-KG is a well-known acronym – the authors should define it first.
 - Line 181: The sentence that begins with "In the HV-KG framework" is not a complete sentence.
 - In general, it is unclear when better metric values are higher vs lower. Up and down arrows should be added to clarify.

### Questions
11. What is the definition of $M$ in equation (1)?
12. How is the value of lambda in Algorithms 2 and 3 determined?
13. In line 255, the authors state that a subset of the gradients were used to estimate the full Jacobian? What exactly was this algorithm? 
14. Why is the performance of OCL + SC (2) better than OCL + SC (1) in Table 1? Shouldn’t more frequent continued training improve the performance?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper addresses the challenge of reduced guidance capability in high-dimensional, 
long-horizon optimization. By integrating sensitivity constraints (SC) with online continual learning (OCL), 
the authors present a generalizable framework that can be applied across diverse modeling contexts.
The paper demonstrated their effectiveness of this approach in several domains, including multi-objective multi fidelity 
Bayesian optimization, hybrid training of coupled neural networks and process-based models, and generative parameter inversion with surrogate training, 
highlighting its versatility and potential broad impact.

### Strengths
1. The OCL+SC framework is versatile and can serve as a plug-in for multiple models, 
potentially broadening its impact across various applications.
2. The paper provides a clear and detailed explanation of the experimental settings, including application in multiple tasks

### Weaknesses
1. The paper does not clearly specify which Online Continual Learning (OCL) algorithm or variant 
is employed among the many existing approaches (e.g., RAR[1], OCA[2],...) 

2. The design of the sensitivity constraint is insufficiently justified. 
There are multiple ways to inject or formulate sensitivity information 
(e.g., input–output Jacobian norms, local Lipschitz bounds), yet the paper adopts one specific form without discussion. 
It would be valuable to clarify why this particular sensitivity formulation was chosen, 
how it affects optimization behavior, and whether alternative forms were tested.

3. Experimental comparison is limited; important baselines such as MF-OSEMO [3] and iMOCA [4] are missing in MO-MFBO.
The evaluation on MO-MFBO were mainly on Branin–Currin; including other synthetic benchmarks (e.g., Park, Levy, Rosenbrock) 
and real-world problems (e.g., Mechanical Plate Vibration Design, Thermal Conductor Design, NAS) would strengthen the evidence.
4. The paper lacks comparisons to recent joint forward–inverse operator methods such as Latent Neural Operator (LNO) [5]. 
I also recommend evaluating the FUSE pipeline on additional PDE tasks beyond Darcy flow (for example airfoil, Navier-Stokes)

The paper lacks empirical comparisons with crucial baselines that could be straightforwardly adjusted to the proposed setting. 
This omission weakens the overall experimental support for the claimed effectiveness of the method.

### Questions
1. Which specific OCL strategy was implemented or was it simply a replay buffer? What motivated this choice?
2. Beyond computational savings, does the use of OCL provide any additional benefit over 
simply retraining the surrogate model with concatenated data in an online optimization setting? If so, can you explain it 
3. Could you please provide an analysis showing how sensitive your results are to the choice of $\lambda$ (the weighting factor for the 
sensitivity-constraint term), and explain how this parameter should be selected for each task?

**Missing related works** : 

[1] Repeated Augmented Rehearsal: A Simple but Strong Baseline for Online Continual Learning

[2] Online Curvature-Aware Replay: Leveraging 2nd Order Information for Online Continual Learning

[3] Multi‑Fidelity Multi‑Objective Bayesian Optimization: An Output Space Entropy Search Approach 

[4] Information‑Theoretic Multi‑Objective Bayesian Optimization with Continuous Approximations 

[5] Latent Neural Operator for Solving Forward and Inverse PDE Problems

[6] Holistic Physics Solver: Learning PDEs in a Unified Spectral-Physical Space

[7] Parameterized Physics-informed Neural Networks for Parameterized PDEs

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes combining Online Continual Learning (OCL) with Sensitivity Constraints (SC) to improve surrogate model performance in optimization tasks involving expensive numerical simulations. The core methodology augments standard data fitting loss with a sensitivity-matching term that aligns the Jacobian of the surrogate model with that of the physical model. The authors evaluate this approach across three optimization frameworks: (1) multi-objective multi-fidelity Bayesian optimization using the Branin-Currin benchmark, (2) hybrid end-to-end training of neural networks coupled with differentiable hydrological (dHBV) and ecosystem (δpsn) models, and (3) the FUSE framework for joint generative parameter inversion and surrogate training on 2D Darcy flow. Results show that OCL+SC consistently outperforms offline surrogates, OCL-only, and offline+SC baselines, with R2 improvements from 0.45 (offline) to 0.71 (OCL+SC) in the hydrological case and from 0.21 to 0.45 in the ecosystem case.

### Strengths
The evaluation across three structurally different optimization frameworks (MOMF-BO, hybrid differentiable training, joint generative-forward modeling) provides evidence of generality beyond a single application context. Each framework uses surrogates differently, yet OCL+SC consistently improves performance.

The failure of accurate surrogate models to provide good optimization guidance is a genuine challenge in computational science and engineering. The paper correctly identifies that sensitivity fidelity is often overlooked in surrogate model evaluation.

The hydrological experiments use the widely-recognized CAMELS dataset (531 catchments), and the ecosystem experiments use SAPFLUXNET (100+ sites, 120+ species), lending credibility to the validation on real observational data rather than only synthetic problems.

### Weaknesses
The paper extensively discusses SC-FNO (Behroozi et al., 2024), DINO (O'Leary-Roseberry et al., 2024), and DE-DeepONet (Qiu et al., 2024) but does not include them as baselines. The "Offline+SC" baseline appears to be a simple implementation rather than these published methods. The central claim that OCL+SC outperforms existing approaches cannot be validated without direct comparison. Specifically, the paper does not test whether applying online continual learning to SC-FNO or DINO would yield similar results, which is the most direct competing approach.

With only 5 random seeds and no hypothesis testing, the statistical validity of the results is questionable. The ecosystem model result where OCL+SC surpasses the physical model benchmark (0.45 > 0.438) is particularly concerning and receives no investigation. Effect sizes, p-values, and confidence intervals are entirely absent, making it impossible to assess whether observed differences are meaningful or due to chance.

The FUSE experiment uses 1000 samples on a 32×32 grid, which is orders of magnitude smaller than typical neural operator papers in 2024-2025 (often 10,000+ samples on 256×256+ grids). The Branin-Currin benchmark is a 2D analytical function, not a physical simulation, contradicting the paper's emphasis on "expensive physical models" and "high-dimensional problems." The experimental scale does not match the claimed scope.

Table 5 claims 1/30 speedup but the comparison is unfair. The physical model's 4176s/epoch likely includes adjoint gradient computation, which the authors also use for SC. The analysis omits initial buffer generation cost, cumulative OCL evaluation costs, and end-to-end time-to-convergence. A complete cost accounting would likely show much smaller speedups. The claim should compare total cost to achieve equivalent performance, not per-epoch runtime for different operations.

Key design decisions lack justification: (a) Why constrain only the "four most sensitive parameters" in the hydrology experiment rather than all 12? (b) Why evaluate gradients at only "middle and last timesteps" rather than all 730 timesteps? (c) How sensitive are results to λ (set to 1 in FUSE, unspecified elsewhere)? (d) How does the number of SC sampling points M affect results? (e) What sampling strategy for SC evaluation points is best (random, uniform, adaptive)? Without these ablations, it is unclear when and how to apply the method.

### Questions
Why are SC-FNO, DINO, and DE-DeepONet not included as baselines? These are the most directly relevant competing methods. How does OCL+SC compare to simply applying online continual learning to SC-FNO? Without this comparison, the contribution cannot be validated. If computational constraints prevented full comparison, can you at least test SC-FNO on one domain?

Please provide statistical significance tests. For all results in Table 1, report p-values (paired t-tests or Wilcoxon signed-rank tests) comparing OCL+SC to baselines. With standard deviations of 0.02-0.07 and only 5 seeds, are the observed differences statistically significant at p<0.05?

How does the ecosystem model surrogate outperform the physical model benchmark (R2: 0.45 vs 0.438)? This is physically implausible. Is this: (a) overfitting to the validation set, (b) an issue with the physical model implementation, (c) noise in observations, or (d) statistical fluctuation? Please investigate and explain.

Please provide complete computational cost accounting. For the ecosystem model, report: (a) cost to generate initial buffer, (b) cumulative cost of all physical model evaluations during OCL, (c) gradient computation cost for SC, (d) end-to-end time to reach R²=0.42 for each method. What is the true total speedup when all costs are included?

What accounts for the inconsistency where OCL alone helps in hydrology but hurts in FUSE? Under what conditions does OCL improve vs. degrade performance? This inconsistency suggests important boundary conditions that should be characterized.

### Soundness
2

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
3

### Summary
This paper proposes combining online continual learning (OCL) with scheduled sensitivity constraints (SC) to improve surrogate-based optimization. The idea is to continually refine a surrogate model with incremental data while periodically enforcing consistency between surrogate sensitivities (Jacobians) and those of the true numerical model. The authors argue that this hybrid approach better preserves meaningful gradients for long-horizon, high-dimensional optimization tasks. The method is evaluated across several settings including multi-fidelity Bayesian optimization, hybrid physics–ML models, and generative parameter inversion frameworks.

The paper tackles an important problem with broad applicability, namely improving surrogate fidelity, particularly with respect to sensitivities, to support efficient optimization. The results generally support the claim that OCL combined with SC improves performance over OCL or SC alone. However, the contribution would benefit from clearer positioning relative to extensive prior work in active learning, adaptive surrogate refinement, and dynamic model updating, as well as more clarity on the application context and practical meaning of improvements. While promising, the novelty and scope currently feel somewhat diffuse due to the breadth of examples and insufficient literature contextualization.

### Strengths
Originality: The paper addresses sensitivity preservation in surrogate modeling, a recognized challenge in surrogate-based optimization. Novel combination of continual learning and sensitivity constraints, and the scheduling strategy for SC injection is interesting. Diverse illustrative applications across optimization, hybrid modeling, and inverse problems show potential generality.

Quality: Empirical results generally support claims that OCL+SC yields better optimization trajectories than alternatives and the paper demonstrates that sparse or infrequent SC finetuning can still give meaningful gains.

Clarity: The motivation is clear, and algorithm steps are reasonably described. Claims are consistently stated and supported numerically. It is really positive that the authors based their work on available open-source codes that are acknowledged and discuss the generalizability of this approach as differentiable programming becomes more available (which the Reviewer agrees is true).

Significance: If properly contextualized, the idea could be useful in large-scale scientific surrogate-assisted optimization. The observation that limited initial data can suffice (if demonstrated rigorously) could be impactful for expensive simulations.

### Weaknesses
1. The manuscript engages in an active research area (surrogate-assisted learning, continual learning, active/adaptive data acquisition) but does not sufficiently clarify: How OCL relates to or differs from adaptive learning, online learning, or active learning in scientific modeling Whether prior works combining sensitivities with incremental data exist in related communities (e.g., multifidelity active learning, physics-guided update strategies)? Without clearer boundaries, the novelty claim (first to combine SC with OCL) is difficult to verify. Actionable suggestion: explicitly define continual learning vs. adaptive surrogate refinement vs. online active learning, and cite key lines of work in each.  In literature review, explain examples from a clearly continual learning need (e.g., not general field of "design using surrogates", which is a huge area where is continual learning needed, or not?)

2. Related to above comment, it is not clear whether the method targets: Dynamic systems with evolving parameters, or static systems where data is progressively acquired to improve surrogates, or both? The mixed examples blur this distinction and make it harder to map contributions to existing streams. Clarify target problem class and restructure literature and examples accordingly.

3. Practical significance not well discussed: While results improve metrics, the real impact is unclear. For instance: In the hydrological case, what does a validation error difference of 1.3 vs 1 translate to physically? Does the sensitivity improvement significantly change real-world decisions? Suggestion: add brief discussion connecting numerical improvements to domain relevance.

4. Experimental clarity: Dimensionality and complexity of the hydrological model are insufficiently specified. Results focus on relative loss improvement; little insight into uncertainty, stability, or robustness. Suggestion: provide dimensionality details, and if possible discuss sensitivity to noise/initialization.

5. Breadth vs depth: Multiple different applications are showcased, which suggests generality, but also makes evaluation feel surface-level. Suggestion: Consider deepening analysis in one domain or adding a conceptual unifying framework to help the reader navigate the diversity of settings.

### Questions
1. How exactly do you define “continual learning” in this context, and how does it substantively differ from well-established active learning, adaptive surrogate modeling, or online Bayesian optimization?

2. Is the approach primarily intended for dynamic systems that evolve over time, or static systems where additional samples are sequentially acquired, or both? Specifically in case of static systems, what are some practical examples of a steady-state design problem that would need continual learning vs. active learning for example?

3. Can you detail the dimensionality and computational scale of the hydrological test case? How challenging is it relative to existing hydrology benchmarks?

4. What does the improvement in surrogate loss translate to in terms of real decision-making or physical interpretability in the hydrological and other settings?

5. Did you consider uncertainty-aware baselines (e.g., BO with uncertainty-driven refinement)? If so, how does the method compare?

6. Is SC applicable if the physical model does not support efficient Jacobian computation, and what are scalability limits?

### Soundness
3

### Presentation
2

### Contribution
2
