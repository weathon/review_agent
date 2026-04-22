# Geometry of Nash Mirror Dynamics: Adaptive $\beta$-Control for Stable and Bias-Robust Self-Improving LLM Agents

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Self‑improving agents learn by playing competitive, often non-transitive language games (e.g., generator–solver, proposer–verifier) where training can oscillate or drift toward undesirable behaviours. We study this scenario through the lens of reverse‑KL regularised Nash learning, showing how the regularisation strength $\beta$ shapes both where agents converge and how they get there. We derive a continuous‑time view of Nash Mirror Descent (Nash‑MD), revealing a simple geometry: trajectories are spirals on the simplex whose damping grows with $\beta$, while $\beta$ simultaneously pulls equilibria toward the reference policy—amplifying any existing biases. We prove last‑iterate convergence to the $\beta$‑regularised Nash equilibrium, quantify its first‑order shift from the unregularised solution, and link convergence speed to the spectrum of the linearised dynamics.

Building on this geometry, we introduce two adaptive $\beta$ controllers: (i) a Hessian‑based rule that targets a desired damping–rotation ratio to accelerate without overshoot, and (ii) a bias‑based rule that caps measurable bias (e.g., output length, calibration, hallucination proxies) while retaining speed. On toy games (e.g. Rock–Paper–Scissors) and small open‑model reasoning benchmarks, our controllers deliver faster, more stable convergence with bounded bias, outperforming baselines. The result is a practical recipe: tune $\beta$ as a control knob to make self‑improving LLM agents both faster and safer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a theoretical and methodological analysis of Nash Mirror Descent dynamics in two-player zero-sum games under reverse-KL regularization, a setting highly relevant for self-improving Large Language Model agents trained via competitive preference games. The core contribution is a precise geometric characterization of how the regularization strength influences the learning dynamics and the resulting equilibrium. The authors derive a continuous-time limit of the dynamics, proving that the local behavior near equilibrium is characterized by decaying spirals whose exponential damping rate is exactly the regularization parameter, while the rotational frequency is determined by the game structure and is independent of it. They further prove that the regularized Nash equilibrium is biased towards the reference policy by an amount proportional to the regularization strength. Leveraging this geometric insight, the paper introduces two novel adaptive controllers for the regularization parameter: the first controller adjusts the parameter to achieve a target damping-to-rotation ratio for accelerated and stable convergence. The second controller adjusts the parameter on a log-scale to enforce a user-specified budget on an operational bias metric. The theoretical findings are validated on a toy game, and preliminary feasibility is demonstrated on small-scale reasoning benchmarks, showing that the proposed controllers can lead to faster convergence and better bias control compared to fixed-parameter or heuristic baselines.

### Strengths
1. The spectral decomposition result is elegant, intuitive, and powerful. It cleanly separates the effect of the regularizer from the game structure. This provides a fundamental understanding that was previously missing in the literature.

2. The paper is theoretically rigorous. It provides a comprehensive set of proofs under clear assumptions, including the derivation of the differential equation limits, characterization of fixed points, convergence rates, and a first-order sensitivity analysis quantifying the equilibrium bias. The connection to preference games solidifies the practical relevance.

3. The proposed adaptive controllers are a direct and clever application of the theoretical insights. Moving beyond heuristics, the first controller provides a principled way to tune for convergence speed, while the second controller offers a novel mechanism for explicitly controlling undesirable behavioral biases, a critical concern in model alignment.

4. The paper is generally well-written. The "geometry" narrative is compelling and effectively unifies the theoretical and methodological contributions. The use of a toy game for sanity checks and visualizations is excellent for building intuition.

5. The work addresses a central challenge in modern model research—stabilizing and controlling self-improving loops—and does so by building upon the increasingly important Nash learning and reverse-KL regularization framework.

### Weaknesses
1. The most significant weakness is the scale of the empirical validation. The language model experiments are described as "micro-experiments" and "single-run prototypes," which, while sufficient for a feasibility study, fall short of demonstrating the practical impact of the method on state-of-the-art alignment pipelines. The community would be more convinced by results on larger models and standard, extensive benchmarks or a full-scale training run.

2. The entire theoretical framework and the control guarantees are local, valid only near an interior equilibrium. The performance and stability of the controllers in real-world scenarios, where these assumptions may be violated, remain an open question. The discussion of limitations is appropriate but underscores that the global picture is not yet covered.

3. The first controller requires estimating the spectral radius via Jacobian-vector products. While the cost is noted as negligible compared to a model forward pass, implementing this robustly in a distributed training setup for large models could present engineering challenges and requires careful hyperparameter tuning. The paper would be strengthened by a more detailed discussion of these implementation complexities.

4.  The presentation of the model results is somewhat confusing. The methods are renamed, and the description feels more like an outline than a full results section. This makes it difficult to assess the concrete performance gains in the practical setting.

### Questions
1. How do the proposed controllers perform in a full-scale training run with a large model and a diverse, large-scale preference dataset? Did you observe any instability or failure modes when moving beyond the "micro-experiment" setting?

2. To what extent did you find the key local assumptions to hold in your model experiments? Were there cases where these assumptions clearly broke down, and how did the controllers behave in those situations?

3.  For the first controller, how sensitive is the performance to the choice of the target damping ratio and the smoothing parameter? Similarly, for the second controller, how critical is the choice of the clamp value and the log-step size?

4. The paper positions itself as complementary to methods like Mirror-Prox. In your model experiments, did you find that the first controller applied to standard descent could outperform or match the performance of Mirror-Prox with a fixed parameter? Is the primary benefit of your method the automatic tuning, or does it achieve a fundamentally better performance profile?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies the training dynamics of self-improving LLM agents. It models the process as reversed-KL regularized Nash Mirror Descent to understand the role of the regularization strength $\beta$. The priamry theoretical contribution is an ODC analysis showing that the learning trajectories are spirals and that $\beta$ serves to stabilzie the spirals. They also show that $\beta$ pulls the final equilibrium to the reference policy. Because of this the authors propose two $\beta$ controllers for faster convergence and control on bias. Experiments on toy settings and LLM reasoning benchmarks validate the theory.

### Strengths
The paper addresses the problem of instability and bias amplification inherent in self-improving LLM agent training loops. There is a core theory contribution on spectral separation of the Nash mirror dynamics. The paper provides a formal description of the trade-off on the stabilizing and bias shift from $\beta$. This motivates the useful perspective of treating $\beta$ as an adaptive control knob and not fixed hyperparameter, this control lever appears practical and is backed by theoretical rigor.

### Weaknesses
The main weakness is that the LLM experiments are very small scale.  That makes it hard to believe the stronger claims about LLM agents. Also, the main LLM results in Table 1 use a “earned 3×3 payoff, which is really just another toy setup rather than a real test of these controllers in a high dimensional, language based environment.

The theoretical analysis is also quite local, it depends on interior-equilibrium and local-monotonicity assumptions (A1, A2) and doesn’t say anything about global convergence or what happens far from equilibrium, which is usually where instability problems show up.

In general it feels difficult too evaluate the significance of this contribution without more compelling empirical results.

### Questions
For $\beta$, it’s not included in the LLM results in Table 139. If it was implemented for the LLM task, how well did it work?

Since the LLM setup is described as a "micro-scale, single-run prototype," it’d help to know what’s blocking scaling up. Is the problem mainly that the two-player local model stops applying in more complex settings, or that the controllers are too computationally heavy to run at scale?

### Soundness
3

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
The paper studies the scenario where self-improving agents learn by playing competitive, often non-transitive language games, where training can oscillate or drift toward undesirable behaviours. The paper claims that it shows that the regularisation strength $\beta$ shapes both where agents converge and how they get there, and derives a continuous-time view of Nash Mirror Descent (Nash-MD), revealing a simple geometry: trajectories are spirals on the simplex whose damping grows with $\beta$, while $\beta$ simultaneously pulls equilibria toward the reference policy—amplifying any existing biases.
The paper also claims that last-iterate convergence to the β-regularised Nash equilibrium was also proven.

### Strengths
Unfortunately, the paper is almost impossible to properly evaluate as the narrative is almost impossible to follow. The ideas in the abstract seem promising, but the paper itself is very difficult to read. Please see the weakness section below.

### Weaknesses
Unfortunately, the paper is almost impossible to properly evaluate as the narrative is almost impossible to follow. There is not a single reference connecting the setting of the work with previous works (the first reference is in the related work section of the paper, and all references are only there). Unfortunately, there is no proper presentation of the notation and the problem of interest, and the whole paper is essentially a series of bullet points without clear connections to each other. 

The authors admit (see last page of appendix) that they use LLMs for assisting with (i)  manuscript structuring and editing, (ii) mathematical exposition (proof sketches and readability), and (iii) LaTeX engineering. I appreciate that the authors share that, but in my view, the presentation and most parts of the main paper look like they are fully written by an LLM. There is no connection between sections, no flow in the narrative, no explanation of related works, and no clear presentation of the theory.

These factors make the reader's task very challenging, making it difficult to absorb the presented information.

At this time, I believe this work falls short of the standards of a major ML conference and requires substantial rewriting to be readable.

### Questions
Please see part on Weaknesses

### Soundness
1

### Presentation
1

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
This paper studies how LLMs can be trained through self-play or competitive “preference games,” where standard learning dynamics often cycle or diverge. It analyzes reverse-KL regularized Nash mirror descent and shows that the regularization strength $\beta$ is used to control both to stabilize and bias toward a reference model.

The authors derive a continuous-time view showing that learning trajectories follow damped spirals, with $\beta$ setting the damping rate. They prove that higher $\beta$ speeds up convergence but also pulls solutions closer to the reference model. Based on this geometry, they design two adaptive $\beta$ controllers:
- Hessian-$\beta$: adjusts $\beta$ to maintain a target damping ratio for faster, more stable learning.

- Bias-$\beta$: adapts $\beta$ to keep a measurable bias (e.g., output length, calibration) within a set limit.

Experiments on rock-paper-scissors and small open-source LLMs confirm the theoretical predictions and show more stable and bias-robust training compared to fixed or heuristic baselines.

### Strengths
- does a nice balance between theoretical groundness and addressing a relevant problem; I also like that the paper points out and uses a trade-off between learning-stability vs system bias.
- it is nice that the paper provides interpretable results on LLMs

### Weaknesses
- Limited comparison: Relationship to standard optimizers like extragradient or optimistic methods is only discussed qualitatively.
- Heuristic link to bias metrics: The operational notion of “bias” (length, calibration) is intuitive but not rigorously tied to the theory.
- computational cost: Spectral estimation for Hessian-β may be expensive or noisy in large systems.

# Theoretical concerns

Several theoretical concerns arise. 
- lacks global convergence guarantees even in simpler settings, and rate and computation comparison

- Because the $\beta$-regularized Nash equilibrium shifts $O(\beta)$ toward the reference policy, varying $beta$ dynamically changes the equilibrium itself; decreasing $\beta$ after a high-$\beta$  phase reintroduces cycling [1,2] and thus can “undo’’ previous stabilization (without an appropriate game optimizer). 

- Conceptually, the same stabilization effect—strengthening the potential (symmetric) component of the game operator—can often be achieved with simpler or cheaper regularization-based methods, such as entropic or Tikhonov regularization that bias the solution but make the field more potential [3,4,5]. Closely related ideas appear in adaptive operator-mixing and interior-point methods, which explicitly add a potential term with a decreasing weight to ensure convergence [6]. Finally, there is a parallel to inertial systems with Hessian-driven damping in continuous-time optimization [7,8], which also introduce curvature-dependent damping to increase the potential part of the dynamics and promote stability.

### Comment on novelty/positioning

Some parts are presented as novel findings, but these are well known, for instance, in the abstract "we derive a [...] revealing a simple geometry: trajectories are spirals [...].
The writing should better reflect the novelty, for instance, "our framework re-affirms known rotational dynamics".


## Minor


Abstract.
- The sentence 20-22 ("we prove last-iterate") is easy to misunderstand: it can easily be interpreted as global last iterate proof, at that point, it is unclear how strong the regularization should be. Also, there exists a small $\epsilon$ for $\beta$ for which this statement "last-iterate.." violates known negative results. 
- Some parts are incomprehensible without knowing the background or reading the paper in detail, for instance, "hallucination proxies," etc. Better to avoid and/or explain better.



-----
[1]  Hsieh Y., Mertikopoulos P., Cevher V., The Limits of Min-Max Optimization Algorithms: Convergence to Spurious Non-Critical Sets, ICML 2021.

[2] Abe, K., Ariu, K., Sakamoto, M., & Iwasaki, A. (2024). Adaptively Perturbed Mirror Descent for Learning in Games. ICML 2024.

[3] Mertikopoulos, P., Papadimitriou, C., & Piliouras. Cycles in Adversarial Regularized Learning. 2018.

[4] Nemirovski, A. (2004). Prox-method with rate of convergence O(1/t) for variational inequalities with Lipschitz continuous monotone operators and Smooth Convex-Concave Saddle Point Problems. SIAM J. Optim., 15(1), 229–251.

[5] Juditsky, A., Nemirovski, A., & Tauvel, C. Solving Variational Inequalities with Stochastic Mirror-Prox Algorithm. 2011.

[6] Yang T., Jordan M., & Chavdarova T. Solving Constrained Variational Inequalities via a First-order Interior-Point-based Method. 2023.

[7] Attouch, H., Chbani Z., Fadili J., & Riahi H., P. First-order Optimization Algorithms via Inertial Systems with Hessian Driven Damping. Optimization, 2016.

[8] Bôt, R. I., Sedlmayer M., & Vuong P. T. A Relaxed Inertial Forward-Backward-Forward Algorithm for Solving Monotone Inclusions with Application to GANs, 2023.

### Questions
1. Does the stabilizing effect of $\beta$ persist under stochastic gradient noise or changing reference policies?

2. How would adaptive $\beta$ interact with existing acceleration or extragradient methods?

3. Which practical bias metrics most faithfully reflect the theoretical “pull” toward the reference model?

4. How robust is the spectral estimate needed for Hessian-$\beta$ in high-dimensional spaces?

5. Could the approach generalize to multi-agent or non-zero-sum games?

### Soundness
3

### Presentation
3

### Contribution
2
