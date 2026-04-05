=== CALIBRATION EXAMPLE 29 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately signals the core topic: enforcing monotonicity in neural networks via constrained learning. However, “with Constrained Learning” is somewhat broad given that the paper’s actual technical vehicle is a stochastic primal-dual method with a chance-constrained surrogate.
- The abstract does state the problem, the proposed approach, and the claimed benefits fairly clearly.
- The strongest claim in the abstract is that the method “does not impose any constraints on the neural network architectures” and “needs no pre-processing such as tuning of the regularization.” This is mostly supported by the method section, but the claim of “small extra computations” is not quantified. For ICLR standards, the abstract is a bit promotional relative to the amount of evidence provided.

### Introduction & Motivation
- The problem is well-motivated, and the introduction correctly identifies the two major existing families: architecture-constrained methods and regularization/certification-based methods.
- The gap is reasonably clear: existing methods either restrict architectures or require regularization tuning/certification overhead.
- The contributions are stated clearly, but several are slightly overstated:
  - “General architectures” is true in the sense of not requiring monotone-specific layers, but the experiments still use relatively simple MLPs, and the method depends on differentiability of the chosen network.
  - “Strong adaptability” is plausible, but the paper does not really demonstrate adaptation across substantially different settings beyond a few benchmark tasks.
- Overall, the introduction is aligned with an ICLR-style applied methods paper, but it would benefit from a more careful statement of what is actually guaranteed versus empirically observed.

### Method / Approach
- This is the most important section, and it raises several substantive concerns.
- The core idea is to convert pointwise monotonicity into a chance constraint, then relax it into the continuously differentiable surrogate in Eq. (6b) using the auxiliary variable \(t\), and solve via a primal-dual method in Eqs. (7)–(11). The high-level approach is understandable.
- However, there are logical gaps in the derivation:
  - **Eq. (5) to Claim 1:** The equivalence between the chance constraint and the expectation of the indicator is standard, but the step introducing \([t + g(x)]_+\) as a sufficient surrogate is only shown as an inner approximation, not a tight one. The paper does not characterize how conservative this approximation is, nor whether it meaningfully preserves the original monotonicity objective.
  - **Claim 1:** The claim is a sufficient condition, not an equivalence, yet the exposition sometimes blurs that distinction. This matters because the final optimization is over a relaxed objective/constraint, so the method is not directly solving the original monotonicity-constrained problem except in the idealized \(\alpha = 0\) discussion.
  - **Uniform sampling over the whole input space \(X\):** The paper says it computes the expectation over \(\mathrm{Uni}(X)\) instead of the data distribution. This is a very strong modeling choice. In realistic problems, the monotonicity requirement may only be relevant on the data manifold, and uniform sampling in high-dimensional boxes may be extremely inefficient or misleading. The paper does not justify this choice beyond citing generalization, and it does not discuss the mismatch between the true data distribution and the uniform domain distribution.
  - **Reproducibility of the gradient terms:** Eqs. (10)–(11) are not fully clear in the extracted text, but more importantly, the method relies on differentiating the input-gradient \(\partial f_\theta/\partial z_m\) w.r.t. \(\theta\), which is a second-order derivative computation. The paper claims “small extra computations,” but does not discuss the actual cost, memory overhead, or numerical stability of computing these higher-order gradients during training.
- Missing or under-discussed failure modes:
  - If the auxiliary \(t\) is fixed small, the relaxed constraint may become either too weak or numerically unstable. The paper briefly says one “may also consider to fix” \(t\), but that undermines the claim that the method avoids tuning.
  - The dependence on Monte Carlo samples from \(\mathrm{Uni}(X)\) may be severe in high dimension; no discussion of sample complexity or coverage is given.
  - There is no theoretical convergence analysis for the stochastic primal-dual updates in the nonconvex neural network setting, which is expected at ICLR if the method claims algorithmic novelty.

### Experiments & Results
- The experiments do test the central claim: whether the method can enforce monotonicity while preserving predictive performance on benchmarks and in a control task.
- The baselines are broadly appropriate and include several important prior monotonic neural network approaches: Certified MNN, COMET, LMN, Constrained MNN, SMNN, etc.
- That said, there are several important evaluation gaps:
  - **No explicit monotonicity violation metrics are reported.** The paper repeatedly claims monotonicity is satisfied, but the tables mainly report predictive accuracy/RMSE and parameter counts. For a monotonicity paper, the central evaluation should include the measured violation rate, certified monotonicity status, or an equivalent quantitative monotonicity metric. The narrative says methods “can produce monotonic controllers” or “passes certification,” but this is not tabulated systematically.
  - **No ablation on \(\alpha\), \(t\), or the number of uniform samples \(N\).** These are essential to the proposed method. Without them, it is hard to know whether the performance comes from the primal-dual formulation itself, the choice of hyperparameters, or simply the network architecture.
  - **No ablation of the chance constraint surrogate versus direct penalty.** Since the paper’s novelty is the constrained-learning formulation, it would be important to compare against a simpler adaptive penalty baseline and to show that the primal-dual update is genuinely better than heuristic penalty escalation.
  - **No statistical rigor beyond mean±std on some datasets.** The paper reports standard deviations, which is good, but the “best five results” protocol is not ideal and can be somewhat cherry-picky. It is also unclear whether all methods received equivalent hyperparameter search budgets.
  - **Fairness of baselines:** The paper compares against methods whose original networks and tuning regimes may differ significantly. For example, the model sizes in Tables 1–2 are not always directly comparable. In some cases the proposed method uses fewer parameters, but this is partly because the architecture is smaller, not solely because the training framework is better.
  - **Interpretation of results:** The gains are modest on some datasets. On COMPAS and Loan Defaulter, the method is competitive, but not decisively superior. On Blog Feedback, SMNN appears slightly better on RMSE, so the claim of general superiority is not fully supported.
- The control experiment is interesting and relevant, since it shows the method can be used outside supervised learning. However:
  - The experiment is more of a case study than a rigorous comparison.
  - It uses a specific existing RL setup and only compares against two monotonic controller variants.
  - There is no uncertainty analysis over disturbances or multiple random seeds beyond “best results” plots.
- Overall, the results are promising, but for ICLR the empirical evidence is not yet as strong as the claims made in the introduction and abstract.

### Writing & Clarity
- The paper is generally understandable at a high level, but several parts of the method are hard to follow, especially the derivation from Eq. (4) to Claim 1 and the construction of the surrogate optimization in Eq. (6).
- The most significant clarity issue is conceptual rather than stylistic: it is not always clear whether the algorithm is guaranteeing monotonicity, approximating it probabilistically, or merely encouraging it. The paper uses all three framings at different points.
- Figures and tables are mostly informative in intent, but the parser-extracted text makes them hard to inspect here. Based on the surrounding discussion:
  - Figure 1 seems to support the 2D toy example, but the paper does not explain enough about how the contour or parameter counts relate to the monotonicity constraint.
  - Figures 2 and 3 are relevant and likely useful, but the discussion is qualitative and would benefit from explicit metrics alongside the plots.
- The main readability issue for a reviewer is that the mathematical development does not fully bridge the gap between the constrained monotonicity goal and the implemented primal-dual surrogate.

### Limitations & Broader Impact
- The paper has an ethics statement, but it is very brief and does not really discuss limitations.
- Key limitations that should be acknowledged:
  - The method requires sampling uniformly from the full input box, which may be impractical in high dimensions.
  - The method’s guarantee is on a surrogate chance constraint, not on exact global monotonicity in the original problem.
  - The approach appears to rely on differentiable activations and second-order derivatives; this may complicate scalability.
  - The control application is domain-specific and does not establish broader safety guarantees.
- Broader impact is mostly positive, as the method aims to improve safety and interpretability. However, the paper should more explicitly discuss where monotonicity might be inappropriate or insufficient as a safety mechanism, especially in high-stakes applications where local monotonicity does not imply overall reliability.

### Overall Assessment
This is a solid and relevant paper with a genuinely interesting idea: use a primal-dual constrained-learning framework to enforce monotonicity without architectural restrictions. That said, the ICLR bar is high, and the current version leaves important questions unresolved. The main concerns are the conservativeness and practical meaning of the chance-constrained surrogate, the lack of quantitative monotonicity evaluation and ablations, and the absence of convergence or scalability analysis for the higher-order-gradient primal-dual training. The empirical results are encouraging and the method likely has merit, but the paper does not yet fully justify the strength of its claims relative to existing monotonicity methods.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a primal-dual constrained learning framework for training partially monotonic neural networks with general architectures. The key idea is to replace a hard pointwise monotonicity requirement with a chance-constraint-style formulation over sampled inputs, then optimize it with a stochastic primal-dual algorithm that adaptively enforces monotonicity during training.

### Strengths
1. **Addresses an important and well-motivated problem.** Monotonicity is indeed relevant for interpretability, fairness, and safety-critical applications, and the paper connects this to concrete domains such as credit risk, healthcare, and frequency control.
2. **Architecture-agnostic formulation is appealing.** Unlike many prior monotonic network methods that require specialized architectures, the proposed method is designed to work with general neural networks, which is a practical advantage if the method is correct and stable.
3. **Adaptive constraint handling is a nice optimization idea.** The primal-dual perspective is a natural way to avoid manual tuning of a regularization weight, which is a real pain point in prior regularization-based monotonic learning methods.
4. **Broad empirical coverage.** The paper evaluates on five public datasets and one control task, spanning both supervised learning and a reinforcement-learning-style setting, which is broader than many monotonicity papers.
5. **Competitive reported performance.** The results tables suggest the method is often competitive with or better than recent baselines, and the control experiment claims improved objective cost over SMNN and a monotonic SNN.

### Weaknesses
1. **The core methodological novelty appears incremental relative to standard constrained optimization.** The paper’s main mechanism is a primal-dual method applied to a relaxed monotonicity constraint. While this is reasonable, the review question for ICLR is whether this is a substantial ML contribution; as presented, it reads more like a domain adaptation of known constrained learning ideas than a clearly novel learning principle.
2. **The theoretical justification is weak and somewhat underdeveloped.** The chance-constraint reformulation and the auxiliary-variable approximation are not accompanied by strong guarantees about actual monotonicity satisfaction, convergence, or approximation quality. In particular, replacing the original monotonicity requirement with an expectation over uniformly sampled points is a surrogate, and the paper does not provide a rigorous bound relating this surrogate to global monotonicity over the whole input space.
3. **Potential mismatch between claims and what is actually guaranteed.** The abstract and introduction suggest monotonicity is “induced” and “continuously and adaptively enforced until the constraint is satisfied,” but the method uses sampled points from the input space and a softened chance constraint. This does not clearly guarantee monotonicity everywhere, especially for deep nonconvex networks.
4. **Reproducibility is only partially convincing.** Although the paper includes dataset descriptions and some hyperparameters, several details critical for reproduction are missing or unclear: exact network widths/depths for all baselines, precise sampling procedure over the input domain, how monotonicity is measured/certified at test time, how hyperparameters were selected, and implementation details for the primal-dual updates.
5. **Empirical evaluation is not fully sufficient for ICLR-level confidence.** The comparisons are mainly against prior monotonicity methods, but the paper does not clearly report monotonicity violation rates, calibration of the chance constraint, or ablations showing the effect of the dual updates, auxiliary variable, sampling density, and the choice of α. These are important to validate the claimed contribution.
6. **The paper’s presentation is hard to follow in places.** Even accounting for parser artifacts, the formulation is somewhat difficult to parse, especially around the reformulation from pointwise monotonicity to chance constraints. The algorithmic steps and the relationship between equations (4)-(6)-(9) are not explained with enough clarity for easy verification.
7. **Some claims are overstated.** Phrases like “does not impose any constraints on architectures” and “needs no pre-processing” are attractive but not fully substantiated, since the method still requires careful choice of α, sampling over the input domain, and training stability settings. Also, using ReLU-based networks in experiments does not fully demonstrate arbitrary-architecture support.
8. **The novelty relative to concurrent monotonic-network work is not sharply delineated.** The paper compares against recent methods like SMNN and Constrained MNN, but it does not crisply explain why the proposed method is fundamentally different beyond optimization style, nor does it isolate what new capability it enables that prior methods cannot.

### Novelty & Significance
**Novelty:** Moderate to low. The use of primal-dual optimization for monotonic constraints is sensible, but the underlying idea appears to be an adaptation of standard constrained learning rather than a fundamentally new monotonicity paradigm. The chance-constraint surrogate and auxiliary-variable construction are interesting, but the paper does not convincingly show a deep methodological breakthrough.

**Clarity:** Moderate to low. The high-level motivation is clear, but the mathematical development is difficult to follow and the exact guarantees are not cleanly stated. The paper would benefit from a more precise problem formulation and clearer explanation of what is and is not guaranteed.

**Reproducibility:** Moderate. The paper provides dataset summaries and some training settings, but not enough detail to confidently reproduce all reported results without additional code or supplementary material. For ICLR, stronger experimental transparency would be expected.

**Significance:** Moderate. Monotonic neural networks are a meaningful area, and an architecture-agnostic constrained-training method could be practically valuable. However, to meet ICLR’s acceptance bar, the paper would need stronger evidence that the method is both technically novel and reliably superior in practice, especially on the core property it claims to enforce.

### Suggestions for Improvement
1. **Add rigorous guarantees.** Provide a theorem or bound connecting the chance-constrained surrogate to actual monotonicity violations, and clarify what conditions are needed for convergence of the primal-dual procedure.
2. **Quantify monotonicity directly.** Report explicit monotonicity violation metrics on test data and, if possible, certification results or worst-case checks over input regions.
3. **Include ablation studies.** Isolate the roles of the auxiliary variable \(t\), the dual updates, the sampling size from the input domain, and the choice of α. Compare against a simple penalty baseline with tuned weights.
4. **Clarify the novelty relative to prior work.** Explicitly state what is fundamentally new compared with regularization-based monotonic methods, counterexample-guided learning, and existing constrained optimization approaches.
5. **Improve experimental rigor.** Report variance across runs more systematically, detail hyperparameter selection for all methods, and ensure baseline comparisons use equally strong tuning budgets.
6. **Strengthen reproducibility.** Provide precise implementation details for domain sampling, monotonic feature handling, optimization schedules, stopping criteria, and certification or evaluation procedures.
7. **Tone down overclaims.** Rephrase claims about “guaranteeing” monotonicity and architecture independence to reflect the actual surrogate-based, sampled nature of the method.
8. **Expand the comparison to stronger general constrained-learning baselines.** Since the method is framed as constrained optimization, it would help to compare with generic penalty, augmented Lagrangian, and primal-dual baselines adapted to the same task.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Add a direct feasibility/monotonicity verification benchmark on the full input domain, not just test performance.** The core claim is that the method enforces monotonicity with a chance constraint, but the paper never reports the actual violation rate or certified monotonicity accuracy on held-out samples or via exact verification for the trained models. Without this, the claim that the constraint is being “continuously and adaptively enforce[d] until satisfied” is not convincing.

2. **Compare against a stronger unconstrained-or-partially constrained tuning baseline with matched architecture and explicit monotonicity penalties.** The paper mostly compares to older monotonic architectures and one or two regularization methods; ICLR reviewers will expect a careful baseline like a well-tuned gradient-penalty/constraint-penalty model with the same MLP backbone, same training budget, and multiple penalty schedules. Otherwise the gains could just be from better optimization or larger effective capacity.

3. **Run ablations for the key components: chance constraint \(\alpha\), auxiliary variable \(t\), dual update, and uniform-domain sampling size \(N\).** The paper’s main novelty is the primal-dual reformulation, but there is no evidence that each part is necessary or stable. Without ablations showing what breaks when \(t\) is fixed, when \(\alpha\) changes, or when the dual update is removed, the method’s claimed flexibility is not established.

4. **Add scalability experiments on larger/deeper models and higher-dimensional monotonic subsets.** The paper claims architecture-agnostic applicability, but all experiments are small MLPs on modest tabular datasets. ICLR reviewers will want evidence that the method remains practical when the number of monotonic features, network depth, or input dimensionality increases, since the added sampling of input-domain points may become expensive.

5. **Include a runtime and training-stability comparison against certification-based methods.** The paper argues that certification is expensive, but provides no wall-clock, convergence, or failure-rate comparison under equal compute. Because the method introduces an inner-loop stochastic constraint penalty, it is important to show it actually reduces cost and does not introduce new optimization instability.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantify the trade-off between monotonicity satisfaction and predictive utility as \(\alpha\) varies.** The central claim is controllable flexibility, but the paper only reports one fixed \(\alpha=0.1\). Add curves showing accuracy/MSE versus violation rate across several \(\alpha\) values; without this, the “chance constraint” story is not empirically validated.

2. **Analyze whether the constraint is enforced on the true input domain or only on sampled points from a uniform surrogate.** The method replaces the data distribution with uniform sampling over \(\mathcal{X}\), which is a major modeling choice and could misrepresent the real constraint under the data distribution. The paper needs to explain when this surrogate is valid and whether monotonicity guarantees hold in practice outside the sampled grid.

3. **Report optimization dynamics of the primal-dual method.** The paper claims adaptive enforcement, but does not show trajectories of constraint violation, dual variables \(\mu\), or gradients over training. These plots are necessary to judge whether the method is genuinely adaptive or just another penalty method with implicit tuning.

4. **Evaluate sensitivity to the sampled-domain budget \(N\) and learning rates.** Since the constraint is enforced using Monte Carlo samples from the input domain, performance may depend strongly on \(N\) and the dual/primal step sizes. Without sensitivity analysis, it is unclear whether the method is robust enough for the “immediately ready” claim.

5. **Clarify the theoretical gap between the original chance constraint and the inner approximation.** The paper introduces an approximation via \( [t + g(x)]_+ \), but does not analyze how conservative it is or whether solving the relaxed problem implies meaningful control over the original monotonicity probability. This weakens the theoretical basis of the framework.

### Visualizations & Case Studies
1. **Plot monotonicity-violation heatmaps over the input domain before/after training.** For 2D or low-dimensional cases, this would show exactly where the model violates monotonicity and whether the proposed method actually suppresses violations everywhere. This is the clearest way to validate the claim beyond aggregate accuracy.

2. **Show training curves for task loss, constraint violation, and dual variables.** These curves would reveal whether the primal-dual updates stabilize, oscillate, or collapse, and whether the constraint is actually being driven down during training. Without them, the optimization behavior is opaque.

3. **Provide side-by-side prediction-vs-input plots for each monotonic feature on real datasets.** Reviewers need to see whether the learned function is monotone in the intended variables while still maintaining reasonable shape elsewhere. A few representative cases where the baseline violates monotonicity and your method does not would be far more convincing than the current aggregate tables.

4. **Add failure cases where the method loses to a baseline or over-constrains the model.** The paper currently highlights successes but does not show when the method hurts performance or truncates feasible output ranges. A case study on a difficult dataset or a tight constraint regime would expose whether the method is robust or overly conservative.

### Obvious Next Steps
1. **Evaluate on more standard monotonic learning benchmarks and stronger modern baselines under identical backbones.** For ICLR-level evidence, the paper should include a more comprehensive benchmark suite and architecture-matched comparisons, not just a small set of tabular tasks and one control application.

2. **Add exact or approximate certification of the learned models.** Since the paper’s contribution is constraint enforcement, the obvious next step is to check whether the trained networks satisfy monotonicity under post hoc verification, especially on the full domain, not just sampled points.

3. **Generalize the framework beyond monotonicity to other inequality constraints and show one example.** The paper itself suggests this direction, and it would materially strengthen the contribution by demonstrating that the primal-dual formulation is not specific to monotonicity.

4. **Demonstrate compute/memory advantages over certification- and architecture-based methods.** The paper claims small extra computations and better trainability, but does not substantiate this with timing and resource measurements. This should have been in the paper because it directly supports the method’s practical contribution.

# Final Consolidated Review
## Summary
This paper proposes a primal-dual constrained-learning framework for partially monotonic neural networks. The method replaces pointwise monotonicity with a chance-constrained surrogate over sampled inputs, then optimizes a Lagrangian with stochastic gradient updates to adaptively enforce the constraint during training. The paper’s ambition is practical—architecture-agnostic monotonicity with less manual tuning—but the current evidence does not fully justify the strength of the claims.

## Strengths
- The problem is important and well motivated: partial monotonicity is genuinely useful for interpretability and safety in tabular prediction and control, and the paper connects the idea to concrete applications rather than treating it as a toy constraint.
- The proposed training framework is architecture-agnostic and conceptually appealing. Using a primal-dual method to adaptively handle the monotonicity constraint is a natural way to avoid specialized monotone architectures and manual penalty tuning, and the paper demonstrates the idea on both supervised benchmarks and a control case study.

## Weaknesses
- The central guarantee is weaker than the paper’s rhetoric suggests. The method does not directly enforce exact global monotonicity; it optimizes a sampled chance-constraint surrogate over \(\mathrm{Uni}(X)\) with an auxiliary-variable relaxation. The paper does not quantify how conservative this surrogate is, nor does it provide a rigorous link between the relaxed constraint and actual monotonicity over the full domain.
- The empirical validation is incomplete for the paper’s core claim. For a monotonicity paper, it is a major omission that the results focus on accuracy/RMSE and parameter counts, while not reporting explicit monotonicity violation rates, certification status, or other direct feasibility metrics. Without this, it is hard to tell whether the method truly enforces monotonicity or merely improves predictive fit on the chosen benchmarks.
- The method’s practical stability and cost are under-explored. The algorithm relies on gradients of input-gradients with respect to parameters, so it involves higher-order differentiation and Monte Carlo sampling over the input domain; yet the paper gives no runtime, memory, convergence, or sensitivity analysis for \( \alpha \), \(t\), or the sampling budget \(N\). This leaves the “small extra computations” and “immediately ready” claims insufficiently supported.
- The novelty is somewhat modest relative to the state of the art. The paper’s main idea is a standard constrained-optimization recipe applied to monotonicity, and the manuscript does not crisply show a capability that prior monotonic-learning methods fundamentally cannot achieve beyond flexibility in optimization.

## Nice-to-Haves
- An ablation study isolating the effect of the chance constraint coefficient \(\alpha\), auxiliary variable \(t\), dual updates, and the number of sampled domain points \(N\) would make the method much more convincing.
- Direct runtime and stability comparisons against certification-based or penalty-based monotonic methods would help substantiate the practical advantage claimed by the paper.

## Novel Insights
The most interesting aspect of the paper is not simply that it enforces monotonicity, but that it reframes monotonicity as a constrained optimization problem over the input domain and then uses the dual variables themselves as an adaptive penalty mechanism. That is a reasonable and potentially useful viewpoint, especially for settings where monotonicity is a soft preference rather than an absolute hard constraint. However, the same reframing also exposes the paper’s main limitation: the final method is only as strong as its sampled surrogate, so the work is closer to a pragmatic training heuristic for approximate monotonicity than to a principled solution to global monotone network design.

## Potentially Missed Related Work
- Generic augmented Lagrangian / primal-dual constrained learning methods — relevant because the proposed method is essentially an application of constrained optimization machinery to monotonicity, and direct comparisons would clarify whether the contribution is in the optimization setup or in the monotonicity-specific surrogate.
- None identified beyond the monotonic-network papers already cited — the paper’s related work coverage is reasonably broad on monotonic architectures and regularization-based methods.

## Suggestions
- Add direct monotonicity evaluation: report violation rates or certified feasibility on held-out samples, and for low-dimensional settings include exact or approximate verification over the domain.
- Include ablations for \(\alpha\), \(t\), \(N\), and the dual update to show which components actually matter and whether the method is robust.
- Report wall-clock time, memory use, and optimization dynamics, especially because the method uses higher-order gradients and uniform-domain sampling.
- Tone down the claims about guaranteed monotonicity and architecture independence to match the actual sampled, surrogate-based formulation.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
