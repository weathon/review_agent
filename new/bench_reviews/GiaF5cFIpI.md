Now I have a thorough understanding of the paper. Let me compose the final review.

Key observations:
1. The simulated stimulation concern is valid and central — the paper's core claim is about adaptive stimulation of latent neural dynamics, but all validation uses simulated stimulations (AR(1) model) on real neural data.
2. The paper does test a "non-trivial" stimulus-response mapping (closed-loop, Fig. 5), but the description of what this mapping actually is is extremely minimal — the reader learns it's "non-simple" and "non-trivial" but not what specific functional form it takes.
3. The baseline comparison issue is real — for stimulation optimization, they compare only against random baselines. For the response model, they compare only against a "blind" model. No comparison to cited methods like Bayesian optimization (Minai et al. 2024) or variational inference (Draelos & Pearson 2020).
4. The L0/L1 approximation concern: The paper does acknowledge this is an approximation, saying "Rather than employ the L0 constraint on the number of neurons, which would make the problem NP hard in general, we use an L1 constraint on u offset by N to encourage a solution with the number of non-zero elements close to n." This is a known relaxation and the authors acknowledge it. The concern that L1 doesn't strictly bound the number of nonzero elements is technically valid but the authors present it as a relaxation, not a strict constraint.
5. The paper does demonstrate the adaptive kernel regression recovery from mapping changes (flip/rotation experiments in Fig. 2).
6. Runtime benchmarks (<10ms average, <100ms max) are a genuine practical strength.
7. The paper makes a genuine contribution in formulating the optimization framework with realistic constraints (non-negativity, sparsity) for high-dimensional stimulation.

The fundamental question is: how seriously does the simulated stimulation weaken the paper's claims? The paper's central claim is about adaptive stimulation design. The fact that all stimulations are simulated with a simple AR(1) model means the paper has not demonstrated its core contribution in its intended domain. However, this is a methodological paper proposing a framework — the authors acknowledge this as a limitation ("performed offline, though in a realistic streaming setting"). The question is whether this is a fatal gap or a valid scope limitation.

I think it's a major weakness but not necessarily fatal — the paper is presenting a computational framework and demonstrating its viability, not claiming to have run closed-loop experiments. But the abstract's claim of evaluation on "real neural data" is somewhat misleading since the stimulations were simulated.

Let me now write the final review.</think>Now I have a thorough understanding of the paper and calibration anchors. Let me compose the final review.

## Summary

This paper presents a streaming framework for adaptive stimulation of latent neural dynamics, integrating (1) streaming latent space construction (including a novel sjPCA method), (2) adaptive dynamical modeling with model selection across multiple latent spaces, (3) a nonparametric kernel regression estimator (Ŝ) for learning stimulus-response mappings that can track non-stationarity, and (4) a constrained optimization procedure for designing high-dimensional stimulation patterns that drive dynamics in desired latent directions under biologically realistic constraints (non-negativity, sparsity). The system is demonstrated on real neural data (calcium imaging and electrophysiology) with simulated stimulation effects and achieves end-to-end runtimes under 10ms on average.

## Strengths

- **Well-integrated system design**: The end-to-end pipeline—streaming latent space construction, adaptive model selection, kernel regression for stimulus-response mapping, and constrained optimization—is coherently designed. Each component serves a clear purpose and the modular architecture allows individual components to be swapped (e.g., non-linear latent space methods).

- **Practical feasibility**: The demonstrated runtimes (<10ms average, <100ms maximum) on standard hardware directly support the claim of real-time applicability for both calcium imaging (15 Hz) and electrophysiology (30 Hz), which is essential for the stated goal of in vivo closed-loop stimulation.

- **Adaptive stimulus-response model**: The kernel regression estimator Ŝ with temporal discounting (Eq. 7) addresses a real practical concern (non-stationary response mappings). The flip and rotation experiments in Fig. 2e demonstrate that the model can recover from abrupt shifts (within ~15s) and continuously adapt to drift, which are genuinely useful properties for experimental deployment.

- **Realistic feasibility constraints in optimization**: Equation 8 incorporates non-negativity and sparsity constraints that directly correspond to excitation-only optogenetic stimulation and limited simultaneous targets—constraints that are experimentally meaningful and rarely addressed together in prior work.

## Weaknesses

### Fatal
None that definitively invalidate the paper's contribution, though there is a major gap in validation scope.

### Major

- **All validation uses simulated stimulation effects, not real perturbations**: The paper's core contribution is adaptive stimulation design and response modeling. However, every experiment—including those on real neural data—applies simulated stimulations using a simple AR(1) model (Sec. 3: "yt = rt + at, at = 0.8·at−1 + ut"). The motivation in Sec. 2.3 explicitly states that real stimulus-response mappings are "nontrivial," state-dependent, and nonlinear, yet the simulated response model has none of these properties. Even the "non-trivial" mapping tested in Fig. 5 is barely described—the reader learns only that it is "non-simple" without understanding its form. The optimization results in Fig. 4 (517/600 optimizations achieving <1° misalignment) are obtained with the optimizer having access to the same simple models the simulator uses. While the authors acknowledge this as a limitation ("performed offline, though in a realistic streaming setting"), the framing understates the issue: the central problem is not offline vs. online, but simulated vs. real perturbation responses. Without testing against actual neural stimulation data, the paper cannot fully substantiate its core claims about adaptive stimulation of latent neural dynamics. This is acknowledged but the language of the abstract—"we demonstrate our approach on both simulated and real neural data"—could mislead readers about the nature of validation on real data.

- **No comparison to principled baselines for stimulation optimization or response modeling**: For stimulation optimization (Fig. 4), comparisons are limited to random single neurons, random groups, and shuffled versions of the designed stimuli—all unprincipled baselines. The paper cites relevant alternative methods (Bayesian optimization from Minai et al. 2024, variational inference from Draelos & Pearson 2020), but does not compare against any of them. For the response model (Figs. 2e, 3c), the sole comparison is a "blind" model that ignores stimulation entirely—a trivially easy baseline. The absence of any principled comparison makes it impossible to determine whether the proposed method is competitive with the state of the art or merely better than strawmen.

### Minor

- **The "non-trivial" stimulus-response mapping is insufficiently described**: Fig. 5 tests a closed-loop mode where Ŝ learns a "non-trivial" mapping, but the functional form of this mapping is never clearly specified in the main text. The result that Ŝ learns simple and non-simple mappings "at approximately the same rate" (2.21 ± 0.9 vs. 1.95 ± 0.79) raises questions about whether the non-trivial mapping is challenging enough to stress-test the method. More detail (or at minimum a mathematical specification) of this mapping would significantly strengthen the evaluation.

- **Optimization formulation underspecification**: The notation ||u||₀^max in Eq. 8 is never precisely defined. While the text explains it as an L1 approximation to an L0 constraint, L1 regularization promotes soft sparsity rather than bounding the number of non-zero elements—a solution could stimulate all N neurons at tiny amplitudes with small L1 norm. The authors acknowledge this is a relaxation, but the practical implications (e.g., how often solutions violate the intended sparsity target) are not analyzed. Additionally, the optimization solver is not specified, and the non-convexity of s(u) (which involves kernel regression) is not discussed.

- **The assumption that only one stimulus can be pending at a time** (line 99) is a significant practical constraint that is not discussed as a limitation, even though real closed-loop experiments may need overlapping stimulus effects.

### Trivial

- The abstract's phrasing "evaluated on real neural data" could more precisely note that stimulations are simulated, though this is partially clarified in the methods.

## Nice-to-Haves

- Demonstrating the method on actual neural stimulation data (even a small pilot with optogenetic perturbations) would substantially strengthen the paper. However, given that the paper scopes itself as a computational framework, this is aspirational rather than essential for acceptance.
- Comparison against Bayesian optimization (Minai et al. 2024) or at least one other principled stimulation design method would clarify the method's relative contribution.
- An ablation studying how many stimulus-response observations are needed before ŝ predictions are reliable, and how performance degrades with increasingly complex/noisy response models, would strengthen the empirical evaluation.
- Analysis of whether adaptive model selection (choosing between latent spaces/dynamic models) translates to better stimulation outcomes, not just better prediction error.

## Removed Points

- **Harsh critic: "The paper claims evaluation on 'real neural data' but this is misleading since stimulations are simulated"** — While the concern is valid and kept above as a major weakness, the harshest framing (calling it "misleading" or questioning the integrity of the claim) overstates it. The paper does use real neural data for the latent space construction and dynamics; only the stimulation effects are simulated. The claim is technically true but incomplete.

- **Harsh critic: "Optimization results are nearly tautological because the optimizer has access to the same simple model the simulator uses"** — This is partially true for the open-loop case (S = Q^T u), but the closed-loop case (Fig. 5) where ŝ is learned from data adds genuine uncertainty. The concern is kept but softened.

- **Harsh critic: "No evaluation of whether alternative stabilization approaches would work for sjPCA"** — This is a generic request for more alternatives and the sjPCA convergence is demonstrated. Removed as scope creep.

- **Harsh critic: "Delayed response model and temporal spreading model are mentioned but never evaluated"** — The temporal spread model is described as optional ("we optionally regress a small number of coefficients β") and the delay is tested in Fig. 3 on real data. This is overstated.

- **Harsh critic: "The 'Feasible' direction condition stacks the deck"** — The feasible condition explicitly targets directions intentionally designed to be reachable, which is the whole point of demonstrating that the optimization works when the problem is well-posed. This is a fair experimental design choice, not stacking the deck.

- **Strength finder: "Closed-loop advantage quantified over open-loop baselines (Fig. 5b)"** — This is a genuine result but the strength is weakened by the fact that the "non-trivial" mapping is barely described and may not be very challenging. Kept as a supporting strength but qualified.

- **Strength finder: "Demonstrated real-time computational feasibility"** — Kept, as runtime benchmarks are concrete and verifiable.

- **Strength finder: "Validation on real neural data across two modalities"** — Removed from strengths because the real data is only used for the latent space; stimulation effects are simulated, which significantly tempers this claim.

## Novel Insights

The paper identifies an important structural gap in the neuroscience methods literature: while closed-loop stimulation of neural activity is becoming experimentally feasible (e.g., holographic optogenetics), there exists no integrated computational framework that simultaneously handles streaming latent space estimation, adaptive response modeling, and constrained optimization for stimulation design. The modular architecture is well-suited for the iterative refinement cycle of closed-loop experiments. However, the gap between the paper's aspirational vision (real adaptive stimulation with non-trivial, state-dependent responses) and its validation (simulated AR(1) responses, simple non-trivial mapping, no principled baselines) limits confidence that the framework would work as intended in its target domain. The contribution is best viewed as a systems-level proof-of-concept rather than a validated method.

## Suggestions

- Clearly specify the functional form of the "non-trivial" stimulus-response mapping in Fig. 5, ideally with a mathematical definition and a comparison against at least one published stimulation design method (e.g., Minai et al. 2024 or Draelos & Pearson 2020).
- Add a parametric sweep over stimulation response properties (magnitude, temporal profile, state-dependence) in the simulated experiments to characterize how the method degrades under more challenging conditions.
- In the abstract and introduction, be explicit that stimulations are simulated rather than delivered—this strengthens rather than weakens the paper by setting reader expectations correctly.
- Report how often the L1 relaxation of the sparsity constraint violates the intended number of non-zero targets in practice.

## Score and Decision

**Calibration comparison:**

| Anchor | Score | Comparison |
|--------|-------|------------|
| Low: synthetic data as validation (ywD00GsxgD) | 2.6 | This paper has more structure but shares the weakness of simulated-only validation for core claims |
| Low: HVAC RL simulator (5XUlfPcQnG) | 2.5 | Much weaker paper overall; our paper has real data and meaningful framework design |
| Medium: realSEUDO (b7DsNJYmeo) | 4.67 | Similar profile: streaming neuroscience method, real-time emphasis, limited novelty beyond engineering integration |
| Medium: fMRI-PTE (BZkKMQ25Z7) | 4.0 | Overclaimed scope with limited validation; our paper is more honest about limitations |
| Medium-high: SIMPL (9kFaNwX6rv) | 6.25 | Stronger empirical validation on real neural data with real baselines; our paper is weaker on baselines |
| High: NetFormer (bcTjW5kS4W) | 7.5 | Much stronger: validated on real data with ground-truth connectivity, compared against baselines |

This paper is below the SIMPL/NetFormer tier due to the simulated-only stimulation validation and lack of principled baselines. It is above the purely synthetic weak papers because it does use real neural data for latent space construction, demonstrates practical feasibility, and proposes a coherent integrated framework. It is similar in spirit to realSEUDO (streaming real-time neuroscience method) but with somewhat more novelty (the optimization framework and adaptive model selection). The gap between the aspirational claims and the evidence is the primary issue, keeping the score in the borderline range.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>