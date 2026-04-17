# Towards Identifiability of Interventional Stochastic Differential Equations

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
We study identifiability of stochastic differential equations (SDE) under multiple interventions.  Our results give the first provable bounds for unique recovery of SDE parameters given samples from their stationary distributions. We give tight bounds on the number of necessary interventions for linear SDEs, and upper bounds for nonlinear SDEs in the small noise regime.  We experimentally validate the recovery of true parameters in synthetic data, and motivated by our theoretical results, demonstrate the advantage of parameterizations with learnable activation functions in application to gene regulatory dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper studies identifiability of stochastic differential equations from stationary distributions under multiple interventions. It gives the first theoretical bounds on the number of interventions required in both linear and nonlinear (small-noise) regimes, supported by synthetic and semi-synthetic experiments.

### Strengths
•	Rigorous mathematical development with clear proofs.
	•	Theoretical results are novel within the causal inference/SDE literature.
	•	Experiments confirm the identifiability thresholds.

### Weaknesses
•	The setting (identifiability from stationary SDEs) is quite narrow and primarily of mathematical interest.
	•	The nonlinear result holds only under restrictive assumptions (contractive drift, small noise).
	•	No real connection is made to learning algorithms or generative diffusion models, which would be essential for ICLR relevance.
	•	The applications is minimal and does not add conceptual depth.

### Questions
•	How do the identifiability results for stationary interventional SDEs inform or improve learning algorithms used in practice?
	•	Can the presented theory be applied to diffusion-based generative models, where SDEs define learned data distributions?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles identifiability of SDE models using only stationary snapshots under multiple interventions (no trajectories). It provides theoretical guarantees for when drift parameters are recoverable from moment information across interventions, covering both linear and nonlinear settings. It validates the ideas on synthetic and semi-synthetic GRN benchmarks.

### Strengths
1. **Well-motivated problem.**
The paper addresses the challenge of recovering system dynamics from stationary, intervention-only data, a realistic and important setting for many scientific domains (e.g., biology), where collecting time-series trajectories is often infeasible.
2. **Novel theoretical contribution.**
The work provides, to the best of my knowledge, the first provable identifiability guarantees for SDEs observed only through stationary interventional distributions, covering both linear and nonlinear cases.

### Weaknesses
1.  **Theoretical presentation lacks clarity.**
The main theorems are difficult to follow because they do not explicitly list all required assumptions. For instance, Theorem 4.4 depends on distributional/genericity assumptions (Assumption 4.2) and a known $D$ (Assumption 4.3), yet these are not stated in the theorem itself but scattered in the text. This weakens the precision and reproducibility of the claims.
2.  **Restrictive linear setup.** 
The linear identifiability results rely on strong and somewhat unrealistic assumptions (e.g., requiring certain structural components or known parameters) which limit their practical applicability and make the “identifiable with $r$ interventions” message feel narrower than presented.
3. **Theory–practice gap in the nonlinear setting.**
The nonlinear guarantees assume globally contractive and monotone activations, but the experiments with learnable activations (generic MLPs) do not appear to enforce these constraints. This creates a noticeable mismatch between the theoretical results and the empirical demonstrations.
4.  **Limited regime of validity.**
The main results hold only in the small-noise regime. The paper itself notes that the proposed losses (e.g., KSD/Sinkhorn) become numerically unstable as noise increases, and that empirical benefits diminish in that regime, which limits the broader applicability.

### Questions
1. **Clarify theorem statements.**
Please restate Theorems 4.4 and 4.8 with their assumptions explicitly enumerated (e.g., “Under Assumptions 4.2–4.3 …”; “Under Assumption 4.5, $||A\|, ||B|| \leq 1$, and i.i.d. interventions ..."). This would significantly improve readability and rigor.
2. **Activation constraints in experiments.** 
In the nonlinear experiments with learnable activations, were monotonicity or contractivity constraints enforced (e.g., via constrained layers or regularization)? If not, how should readers reconcile the theoretical assumptions with the empirical results?
3. **Sample complexity and scalability.**
Could you report the approximate sample complexity (e.g., cells per intervention) needed for stable recovery in both linear and nonlinear regimes, and discuss how computational cost scales with $n$ and $r$?
4. **Related work suggestion.** 
In the “Dynamical System Methods” section of related work, please consider including  _Wang et al., NeurIPS 2023: "Generator identification for linear SDEs with additive and multiplicative noise"_, which provides identifiability results for linear SDEs based on the generator. This paper is highly relevant and would help position your contribution more clearly within the broader identifiability literature.

### Soundness
2

### Presentation
2

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
This paper considers a system with state space R^d and evolution described by a first order stochastic differential equation (SDE), with identity diffusion. The authors assume to have access to 'interventions' which means that the drift coefficient can be modified by addition of a constant term. They ask the question of identifiability of the drift from the stationary distribution of the SDE for multiple values of the intervention.
Two type of results are presented:
1) Linear case. In this case the drift is linear with corresponding matrix given by the sum of a rank r term and an arbitrary known term.
Under a a probabilistic model for the low rank component (that in particular ensures genericity) tyhey prove that r-2 interventions are necessary and r are sufficient.
2) Nonlinear case. The drift is assumed to be parametrized by a two-layer neural network with r hidden neurons. This case is treated in the limit in which the stochastic component of the SDE vanishes, reducing to the linear case by a perturbative argument.

### Strengths
In many systems that are modeled by an SDE is unrealistic to assume that we can observe trajectories, and it is instead more common to have access to the stationary distribution. The stationary measure does not uniquely identify the drift and hence the plan of studying this problem under interventions is well motivated and interesting.
The presentation is clear, and the results are easy to understand.

### Weaknesses
Establishing identifiability is only the first step towards understanding estimation accuracy; optimal procedures; computational complexity and so on.
The identifiability result in the linear model appears a relatively direct fact of linear algebra.
As for the nonlinear case, the result is purely perturbative and non-quantitative. It requires the drift to be a contraction with a unique fzero, enabling perturbative argument. No quantitative estimate is given on how small \epsilon must be for the identifiability to hold.

### Questions
1) Assumption 4.2 is stated in a form that is not very transparent. I believe that what is required is really certain deterministic conditions on A, B, C. Instead the authors choose A,B,C to be random so that those conditions are satisfied a.s. I think it would be much better to express the conditions in deterministic form. Also "each column is drawn iid" probably means that the columns c_1, c_2,.. are iid.  

2) Why the low-rank model, and the corresponding r hidden neurons network are good models for the problems proposed in the introduction?

### Soundness
3

### Presentation
3

### Contribution
2
