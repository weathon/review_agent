# Learning-Based Autonomy from Kernel-Embedded Multi-modal Fusion to Feedback Control

- Decision: Reject
- Scores: 0, 0, 2, 4

## Abstract
In this work, we develop an end-to-end autonomy loop that couples \emph{kernel-embedded} multi-modal fusion with data-driven dynamics learning and feedback control. Heterogeneous sensor streams are embedded into a joint Reproducing Kernel Hilbert Space (RKHS) via additive/product kernels and conditional mean embeddings; dynamics are learned with kernel ridge regression (KRR), Deep Kernel Learning (DKL), or Bayesian deep neural networks (BDNNs); and policies are synthesized via dynamic programming (discrete and continuous-time HJB) or reinforcement learning with RKHS value functions. We present closed-form estimators, finite-sample and iteration-complexity characterizations, risk-sensitive planning with uncertainty, and safety via control barrier functions. We provide deployable algorithms, results and experiment in simulated robotics and precision irrigation.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes an end-to-end "kernel-centric autonomy loop” that fuses multi-modal observations via kernel mean embeddings in a joint RKHS, learns dynamics with KRR, DKL, or BDNNs, computes policies via RKHS-based fitted value iteration or a kernel Galerkin approximation to the continuous-time HJB, and enforces safety with a CBF-based shield and introduces risk-sensitive planning using predictive uncertainty. Algorithms and complexity statements are provided; experiments span CartPole, Dubins, Quadrotor, and Precision Irrigation with multiple metrics.

### Strengths
The paper's stated goal to unify sensing and data fusion, dynamics learning, and control in an RKHS framework with built-in safety and risk awareness is commendable. It builds on established tools such as kernel ridge regression and RKHS. It gives a brief overview over each of the concepts required for building its pipeline.

### Weaknesses
The submission feels like an early‑stage draft. There is little to no clear novelty beyond assembling well‑known components (KRR/DKL/BDNN, RKHS value‑function approximation, CBF shields), and major parts of the stated contributions are either missing or only gestured at without concrete methods, experiments, or analysis. Several sections are left near‑empty (7, 8, 9) or largely restate textbook material (e.g., Sec. 4.1–4.3), experiments and baselines are not adequately described or justified, crucial information about the full setup (modalities, representations, optimization routines, hyperparameters, seeds, compute, uncertainty modeling) is absent, and there is no substantive discussion or interpretation of results. Despite a claim of a “reproducible deployment protocol,” none is actually provided or exemplified in the paper.

Overall, the paper needs serious work in terms of articulating a focused and genuinely novel contribution, filling in missing technical and experimental details, and substantially improving the clarity and rigor of the presentation and analysis.

The following is an incomplete list of specific issues that stood out:

## Technical novelty and substance

- Sec. 4.1 restates standard kernel ridge regression; no new estimator, analysis, or integration result is provided.
- Sec. 4.2: L155’s “optimized to capture task-relevant structure” is vague; no specification of how the feature extractor is learned (architecture, loss, optimizer). L167 acknowledges DKL is established, with no additional novelty in the paper.
- Sec. 4.3 is a textbook recap of BDNNs and VI; no new modeling, inference, or control coupling is introduced.
- Sec. 4.5 repeats the same bound and the same CBF-QP twice, including duplicated text and equations ((1) and (2)), without added content or proof.

## Clarity, notation, and presentation
- Notation inconsistency: On L119 the discrete model uses $\epsilon$, while L120 refers to $\omega$ as process noise—symbols do not match.
- Formatting/citation artifacts: Broken encodings and citations (e.g., “Sch”olkopf,” citation formatting on L32) impede readability.
- Algorithms lack context: Sec. 5 lists procedures (Algorithms 1–3) without detailing inputs/outputs, optimization steps for min over u, or how RKHS elements are concretely represented for DKL/KRR.
- Redundancy: Sec. 6 repeats the same complexity facts twice (L280 and L298), both restating prior work.
- CBF integration unclear: The CBF constraints are not motivated (assumptions for convexity, control-affinity, feasibility) nor connected to how they interface with the learned policy and action optimization.

## Experimental shortcomings

- Baselines repeated (L286 and again in Sec. 7) without discussion. No description of modalities, sensors, data splits, horizons, or hyperparameters. No scalability experiments.
- Results are not analyzed: No discussion of trends; units are missing; line plots are used for categorical comparisons (L326 onward).
- Dubious tables/figures: Tables are scattered into/after the references section; values show six decimals yet the last four digits are always zero, suggesting post-rounded data with misleading precision.
- Safety inconsistency: “Safety filter” is claimed, yet Safety Violations (SV) are present across many experiments; the paper does not explain whether/when the CBF shield is active, nor why SV > 0 if guarantees are expected.
- Reproducibility gap: Despite claiming a “reproducible deployment protocol” in the contributions, no such protocol is described. The Reproducibility Statement only promises future code release; critical implementation details are missing.

### Questions
Since there are several larger issues with the paper, the questions focus on the higher-level:

- Core contribution and throughline: Can you state one focused, novel contribution (algorithmic or theoretical) and show how the method, theory, and experiments specifically realize and validate it?
- Multimodal fusion representation: How is the fused RKHS object made usable in practice (finite-dimensional representation for KRR/DKL)? Please specify the representation choice and how it is learned or selected.
- Safety via CBF: When and how is the CBF shield applied, under what assumptions is it feasible, and why do safety violations still appear?
- Experimental setup and baselines: Can you provide a compact but complete description of each task (modalities, data sizes, horizons), baseline configurations, and report means/std over multiple seeds with brief result interpretation?
- Reproducible deployment protocol: The conclusion mentions a deployment protocol—can you summarize it concretely (components, configurations, and steps) or provide a minimal recipe/code link to substantiate this claim?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes several different methods to achieve a fully data-driven approach to control while achieving safety. It considers deep kernel learning and Bayesian neural networks to model the dynamics, and ostensibly achieves safety using control barrier functions.

### Strengths
The authors address an important problem, where a fully data-driven approach is used to control a system while guaranteeing safety.

### Weaknesses
This paper presents an agglomeration of several well-known concepts, without providing any clear contribution. The problem is not clearly stated (the Problem Formulation section only mentions model learning). There is no related work section. The authors claim to provide theory supporting their method (Line 371), even though their theoretical analysis is limited to well-established results that only apply under certain conditions. The experimental section is very limited, does not contain SOTA approaches, and has no error bars. The limitations section (Section 8) does not discuss limitations and is only two lines long.

### Questions
Line 116: Is epsilon a stochastic variable? If so, do the authors consider an SDE? This is unclear.

Line 120: \omega is not defined. Also, there is a notation conflict in line 162, where \omega is used to refer to parameters, not noise.

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper tries to merge kernel mean embeddings, kernel ridge regression, reinforcement learning, and barrier functions into a framework for learning control policies. The main motivation is drawn from real-world autonomous systems that need to provide safety guarantees while receiving uncertain, multi-modal data streams as input. Kernel methods could then be a possibility to learn representations of the data distribution. The authors present an algorithm that combines all those methods and compares it against baseline algorithms on standard benchmarks.

### Strengths
The general problem considered is indeed of interest, and kernel methods could play a role in resolving it. As such, the approach considered in the paper is generally interesting.

### Weaknesses
1) There are various important details missing. ICLR offers the possibility of submitting an appendix, so if space is an issue, additional details can be presented there. Also, references do not count against the page limit, so there would have been more space.  As an example, the authors say that multi-modal observations (how are those defined?) are fused through kernel mean embeddings. How? It is unclear what is happening. 

2) Notation is unclear. For example, the noise variable $\epsilon_t$ is never properly defined (i.e., what is its distribution). It is instead said that $\omega$ represents process noise, but $\omega$ does not appear in the system definition. Instead, $\omega$ will later on represent something different. 

3) A discussion of related literature is missing.

4) The authors claim safety guarantees, but in Section 4.5, they just state that under a Lipschitz assumption, the true and estimated value functions are also close to each other, without any reference or proof for this statement. Then, they say that safety is enforced by a control barrier function, again without any proof. Also, (1) and (2) seem to be the same. 

5) There are way too few details about the experiments and no discussion of the results. From the plots and tables, it is unclear what the takeaways are. The tables don't even have captions. 

6) The authors claim to discuss kernel choice, robustness, etc., but it is unclear to me where this happens.

7) There seems to be a LaTeX error with Schölkopf's last name in the references.

8) Sections 4.3 and 4.4 seem to introduce new problem definitions. It would be nice if we could stick to the one that was originally defined.

### Questions
1) Can you provide more details on the method itself? How do you embed multi-modal measurements? What is the promise of approximating the value function via RKHS embeddings? What is then happening with the $\tilde{y}_i$? What about scalability of this approach?

2) Can you elaborate on the safety guarantees? Just stating that they hold is insufficient without proof or reference.

3) Can you discuss your results?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an end-to-end learning-based architecture that integrates kernel-embedded multi-modal fusion, nonparametric dynamics learning, and control via dynamic programming (DP) in a unified Reproducing Kernel Hilbert Space (RKHS). The approach fuses heterogeneous sensory modalities (e.g., vision, proprioception, environment data) through additive or product kernels using kernel mean embeddings and conditional mean operators. The learned system dynamics are modeled with Kernel Ridge Regression (KRR), Deep Kernel Learning (DKL), or Bayesian Deep Neural Networks (BDNNs), and controllers are synthesized via discrete or continuous-time Hamilton-Jacobi-Bellman (HJB) formulations using RKHS value functions. The framework provides finite-sample and iteration-complexity guarantees and demonstrates safety enforcement via control barrier functions (CBFs). Experiments across robotic and environmental control tasks (CartPole, Quadrotor, Dubins car, Precision Irrigation) showcase performance and sample efficiency gains relative to standard RL baselines.

### Strengths
1) The paper is well-written and presents a comprehensive  synthesis of kernel-based methods, uncertainty quantification, and control theory into an end-to-end loop. The RKHS unification of sensing, dynamics learning, and control is interesting and differentiates this work from modular deep-RL architectures.

2)  The use of control barrier functions on top of kernel-based controllers provides a rigorous and safety-aware mechanism that is relevant to practical deployment in robotics and autonomous systems.

### Weaknesses
1)  The paper's claim of "end-to-end autonomy" would be better supported by engaging with the vast literature on adaptive and model-based RL (e.g., PILCO, system identification for control, or recent work on neural Lyapunov and policy gradient control). Without this, the reader may overestimate the novelty of the proposed learning–control loop.

2)  Each module (KRR, DKL, CBF-based safety) is individually well-established. The main contribution is the system-level integration, but the paper could be more explicit about what new algorithmic or theoretical principles emerge from this combination.

3) The proposed method relies on kernel matrix operations with cubic complexity, and though approximations are mentioned, the paper does not evaluate scalability on larger datasets or higher-dimensional sensory inputs.

### Questions
1) How does the proposed RKHS-based fusion compare to learned representation or latent-space fusion methods in terms of downstream control performance and generalization?

2) How are kernel hyperparameters or fusion weights selected across modalities, and is there a procedure to learn them end-to-end?


Minor comments: 

1) The exposition is thorough and sound, but the related work section should explicitly connect to multi-task control (e.g., data-efficient meta-RL).

2)  The discussion on Deep Kernel Learning could benefit from clearer justification of when it outperforms purely neural alternatives in control tasks.

### Soundness
3

### Presentation
2

### Contribution
3
