# Mean-Field Neural Differential Equations: A Game-Theoretic Approach to Sequence Prediction

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
We propose a novel class of neural differential equation models called mean-field continuous sequence predictors (MFPs) for efficiently generating continuous sequences with potentially infinite-order complexity. To address complex inductive biases in time-series data, we employ mean-field dynamics structured through carefully designed graphons. By reframing continuous sequence prediction as mean-field games, we utilize a fictitious play strategy integrated with gradient-descent techniques. This approach exploits the stochastic maximum principle to determine the Nash equilibrium of the system. Both empirical evidence and theoretical analysis highlight the unique advantages of our framework, where a collective of continuous predictors achieves highly accurate predictions and consistently outperforms benchmark prior works.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
Unable to assess the work's impact/relevance, abstaining from reviewing process

### Strengths
Unable to assess the work's impact/relevance, abstaining from reviewing process

### Weaknesses
Unable to assess the work's impact/relevance, abstaining from reviewing process

### Questions
Unable to assess the work's impact/relevance, abstaining from reviewing process

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a Mean-Field Neural Differential Equation (MFP) framework that reformulates continuous-time sequence forecasting as a mean-field game, and introduces a gradient-update algorithm based on forward–backward stochastic differential equations (FBSDEs) to compute Nash equilibria. On the theoretical side, it studies the coupled partial differential equations—Hamilton–Jacobi–Bellman (HJB) and Fokker–Planck–Kolmogorov (FPK)—and provides proofs of distributional convergence. Empirically, the method shows promising performance on the evaluated benchmarks, offering partial evidence of effectiveness and indicating potential research value.

### Strengths
1. A nontrivial methodological contribution within the neural differential equation (NDE) literature: the paper frames continuous-time sequence modeling with a coherent design that explicitly targets irregular sampling and asynchronous events.
2. Strong theoretical development with careful mathematical derivations, yielding a principled objective and training procedure; the technical depth increases the credibility of the modeling choices.
3. On the evaluated (discrete, finitely sampled) benchmarks, the method attains competitive performance, indicating practical promise for continuous-time parameterizations (notwithstanding the limitations noted below).
4. The problem is central to continuous-time ML/NDEs and is timely for the community, with potential downstream impact on temporal point processes, latent dynamics, and long-horizon forecasting.

### Weaknesses
1. Template noncompliance: multiple figures (e.g., Fig. 3) place captions above the panels, and Table 2 does not follow the ICLR table style; these violate camera-ready formatting requirements and must be fixed.
2. Heavy notation with scattered definitions: the symbol usage lacks a consolidated reference, hurting readability. Provide an appendix table (symbol, meaning, shape/dimension, domain, scalar/vector) to standardize notation.
3. Reproducibility gap: no code or artifacts are released. Public implementations (training scripts/configs, environment, seeds, and optionally checkpoints) are necessary to substantiate the claims and enable adoption.
4. Evidence does not support the stated extreme-regime claim (Δt→0, events→∞): the experiments operate on discretely and finitely sampled datasets, without convergence studies vs. step size/solver tolerance, stress tests in dense-event regimes, or analyses of error accumulation in long-horizon rollouts. As written, results demonstrate performance in standard discrete settings rather than the claimed limit.
5. Text–figure inconsistency: Ablation Study I states long-term prediction on EigenWorm, while Fig. 4 is titled “Ablation studies on the MIT Humanoid Robot dataset,” creating a contradiction that undermines confidence in the reporting.
6. Missing experimental setup details and pointers to the appendix: data preprocessing/windowing/normalization, evaluation protocol and metrics, solver choice and tolerances, training hyperparameters (optimizer, schedule, batch size), random seeds, compute budget, and early-stopping criteria are not clearly specified, making it hard to assess fairness and reliability.

### Questions
1. The introduction states the goal as modeling in the extreme regime Δt→0 and events→∞, yet the experiments and conclusions primarily show that MFPs perform well on standard discrete forecasting. Please clarify whether the actual objective is an extreme-regime continuous-time capability or competitive performance under discrete sampling.
2. Apart from weaknesses 1, 2, and 6, the remaining listed weaknesses likewise raise concerns; please clarify each or provide supporting evidence.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Mean-Field Predictors (MFPs), a novel class of neural stochastic differential equation (SDE) models, for continuous-time sequence prediction. The core idea is to model the prediction problem as a mean-field game (MFG), where a continuum of predictors (agents) interact via a neural graphon to collectively predict future events. The authors develop a gradient-based training algorithm using forward-backward SDEs (FBSDEs) and provide both theoretical convergence guarantees and empirical validation.

### Strengths
Originality: The paper introduces a novel formulation of sequence prediction as a mean-field game, combining neural SDEs with neural graphons in a creative way.
Quality: The methodology is technically sound with solid theoretical grounding and strong empirical results across multiple datasets.
Clarity: The paper is generally well-written, with clear definitions and helpful illustrations that support understanding.
Significance: The approach is broadly applicable to complex, irregular, and continuous-time sequence data, making it impactful for several domains.

### Weaknesses
The paper uses \varepsilon-Nash equilibrium as a practical solution but lacks analysis on how \varepsilon impacts prediction quality or how it behaves over time.
Key recent models like continuous-time graph or attention-based architectures are missing from the comparisons.
All datasets are low-dimensional; results on high-dimensional or real-world multivariate sequences would strengthen the claims.

### Questions
Could you provide a more intuitive explanation of why sequence prediction benefits from being framed as a mean-field game? How does this view improve over traditional ensembling or neural SDE approaches?
How about the impact of different ensemble-based predictors? Could we use simpler alternatives?
How about the computational cost of training interacting predictors?

### Soundness
3

### Presentation
4

### Contribution
3
