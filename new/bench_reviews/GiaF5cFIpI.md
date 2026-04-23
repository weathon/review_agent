Now I have all the information I need. Let me compose the final review.

## Summary

This paper presents a streaming framework for adaptively modeling and driving latent neural dynamics via high-dimensional stimulation patterns. The method integrates streaming latent space construction (including a novel sjPCA), adaptive nonparametric stimulus-response modeling via kernel regression (Eq. 7), and constrained optimization (Eq. 8) to design stimuli that perturb dynamics along desired directions in latent space. The framework is validated on a 3D toy model and on two real neural datasets (calcium imaging, electrophysiology) with simulated stimulations, demonstrating sub-10ms end-to-end runtimes.

## Strengths

- **Well-motivated and timely problem formulation.** Designing closed-loop stimulations to drive latent neural dynamics is an important and underaddressed problem at the intersection of computational neuroscience and optogenetics. The decomposition into streaming latent space construction, adaptive response modeling, and constrained optimization is logically coherent (Sec 2).

- **The toy model experiments (Fig. 2) demonstrate genuine adaptive capability.** The kernel regression estimator Ŝ recovers from both an abrupt 180° flip in the ground-truth mapping at t=25s (within ~15s) and continuous drift (1 revolution/30s starting at t=45s), while a non-adaptive model suffers persistent elevated error (Fig. 2e). This is the clearest demonstration that the temporal kernel mechanism provides useful non-stationarity tracking.

- **Closed-loop optimization outperforms open-loop when Ŝ is non-trivial.** Figure 5b shows that stimuli designed using the learned Ŝ achieve a higher proportion of their observed response magnitude aligned with the target direction v compared to open-loop stimuli designed assuming S(u) = Q^⊤u. This directly validates the utility of the adaptive learning component.

- **Real-time computational feasibility is convincingly demonstrated.** End-to-end runtimes average <10ms and never exceed 100ms (Sec 3), satisfying the timing requirements for closed-loop optogenetic experiments on standard workstation hardware.

- **The framework is modular and extensible.** Each component (latent space, dynamics model, response model, optimizer) can be independently swapped, which is a practical design choice for real experimental settings where different recording modalities may require different components (Sec 2).

- **Validated on two real neural modalities with different timescales.** Calcium imaging (592 neurons, 15 Hz; Fig. 3) and intracortical electrophysiology (130 units, 30 Hz; Sec 4.2) demonstrate applicability across recording modalities.

## Weaknesses

### Fatal
None.

### Major

- **The core contribution — closed-loop adaptive stimulation with a non-trivial Ŝ — is only validated on a 3D toy model, not on real neural data.** The paper's central motivation is handling complex, state-dependent, potentially nonlinear stimulus-response mappings where "responses are driven by network structure and the state of the neural system" (Sec 2.3). Yet on all real data, stimulations are simulated using a trivially simple AR(1) model: y_t = r_t + a_t, a_t = 0.8·a_{t-1} + u_t (Sec 3), making S(u) = Q^⊤u (the open-loop case). The closed-loop adaptive loop — where Ŝ is learned from observations and used for optimization — is the paper's primary technical contribution, but it is only demonstrated on a 3D toy model with binary stimulation (Fig. 5b). While the authors acknowledge that "real data experiments were performed offline" (Sec 5), the gap between the claimed capability and the validation is significant: the method could be fundamentally sound, but the current evidence does not establish that it works under the challenging conditions it claims to address.

- **The optimization is compared exclusively against random strategies.** The paper compares its designed stimuli against random single-neuron stimulation, random multi-neuron stimulation, and shuffled versions of its own stimuli (Sec 4.2, Fig. 4a). The introduction cites multiple prior methods — Bayesian optimization (Minai et al., 2024), input-output dynamical modeling (Yang et al., 2021), and active learning (Wagenmaker et al., 2024) — yet none are implemented as comparative baselines. Without comparing against any principled alternative, it is impossible to assess whether the proposed optimization offers advantages over existing approaches or merely outperforms random selection. Even one implemented baseline would substantially strengthen the empirical evaluation.

### Minor

- **The sparsity formulation in Eq. 8 appears incorrectly specified for its stated goal.** The penalty λ₁(‖u‖₀^max − ‖u‖₁) under box constraints 0 ⪯ u ⪯ 1 minimizes (constant − ‖u‖₁), which maximizes ‖u‖₁ and encourages entries toward 1 — i.e., denser solutions, not sparser ones. The text states the goal is to "encourage a solution with the number of non-zero elements close to n" (Sec 2.4), but the formulation as written appears to work against this. The optimization may still produce reasonable results because the cosine similarity term dominates, but the claimed sparsity property is unsupported by the mathematical formulation as stated. This needs either correction or clarification.

- **The abstract's claim of demonstration on "real neural data" could set misleading expectations.** While technically true (real neural recordings are used), the stimulations on real data are entirely simulated with a trivial AR(1) model. A qualifier would better set reader expectations.

## Nice-to-Haves

- A sensitivity analysis of how optimization degrades as Ŝ becomes less accurate — how many stimulation observations are needed before the optimization produces useful results, and how does error in Ŝ propagate to error in designed stimuli?
- A full closed-loop simulation trace showing the adaptive loop in action on a non-trivial system (initially inaccurate Ŝ → designed stimulations → observed responses → Ŝ update → improved stimulations).
- Comparison against at least one principled baseline for stimulation optimization (e.g., the Bayesian optimization approach of Minai et al., 2024).
- A partnership with an experimental lab for a proof-of-concept in vivo demonstration, even a simple one.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Misleading abstract" (harsh critic)**: The abstract says "real neural data" which is technically correct. The concern about simulated stimulations is captured in the Minor weakness above, but calling it "misleading" overstates the issue — the neural data itself is real, and the paper does use the term "simulated effects" in the Discussion. Demoted to a minor clarity point.
- **"Combinatorial argument mismatch" (harsh critic)**: The claim that the combinatorial motivation (10^45 combinations) mismatches the continuous L1 approach is a strawman — the motivation establishes why the problem is hard, and the L1 relaxation is a standard approach to handle it. The scale of the motivation and the method's approach are not contradictory.
- **"sjPCA novelty difficult to assess without full algorithm specification" (harsh critic)**: The Sherman-Morrison update is a standard technique; the Orthogonal Procrustes stabilization is described in Eq. 2. Demanding full derivation is a nitpick about appendix-deferred details.
- **"Streaming estimator novelty unclear" (harsh critic)**: The framework for parallel evaluation and adaptive selection of representations (Fig. 1c) is a useful engineering contribution even if individual components are straightforward. Demanding novelty in each sub-component is scope creep.
- **"Computational scaling concern for kernel regression" (harsh critic)**: This is a generic concern about kernel methods that applies to any kernel regression paper. The paper demonstrates real-time feasibility empirically (<10ms), which addresses this concern in practice.
- **"Feasible directions mostly validate easy problem" (harsh critic)**: While true that feasible directions are defined as reachable with <30 neurons, the systematic evaluation across Negative, Dense, Random, Feasible, and Q0 targets (Fig. 4b) provides useful characterization of the optimization's capabilities and limitations.
- **"Did not include explicit consideration of effects on behavior" (harsh critic)**: The authors explicitly acknowledge this in Sec 5 and outline how it could be extended. Criticizing its absence is scope creep — the paper is about driving latent dynamics, not behavioral effects.
- **"Non-linear latent spaces not tested" (harsh critic)**: Acknowledged in Sec 5 as a limitation with a clear extension path. Scope creep.
- **Strength Finder's "rapid learning from few stimulations"**: While mentioned in the abstract, this isn't backed by a specific quantitative result in the paper beyond the toy model convergence. Moved to nice-to-have.

## Novel Insights

The key tension in this paper is between the ambition of the problem formulation (adaptive, nonparametric, state-dependent stimulus-response modeling for high-dimensional neural stimulation) and the modesty of the validation (only the simplest case on real data). The toy model results (Fig. 2) are genuinely promising — the adaptive kernel regression does recover from non-stationarities — but the paper leaves the most critical question unanswered: whether the gap between Ŝ and the true S on real neural data (with real optogenetic stimulation, real state-dependence, real non-stationarities) is small enough for the optimization to produce useful stimuli. The modularity of the framework is both a strength and a risk: it allows incremental validation and component swapping, but it also means the paper validates components individually rather than the integrated system under realistic conditions.

## Suggestions

- **Add one experiment with a non-trivial simulated Ŝ on real neural data.** Even without real optogenetic stimulation, you could apply a non-trivial response mapping (e.g., state-dependent, or with nonlinearities like per-neuron saturation) to the real neural data and test whether the closed-loop optimization recovers. This would bridge the gap between the toy model and the trivial AR(1) model currently used on real data.
- **Implement at least one principled optimization baseline** — even a simple greedy neuron-selection strategy based on Q-loadings would provide a more informative comparison than random selection.
- **Clarify the sparsity formulation in Eq. 8** — either correct the sign/convention or explain how the current formulation produces sparse solutions despite appearing to encourage density.

## Evaluation

**Originality:** The framework is novel in its integration of streaming latent space construction, adaptive response modeling, and constrained optimization for neural stimulation. Individual components (kernel regression, streaming PCA, L1 sparsity) are not novel, but their combination for this problem is.

**Importance of research question:** High. Closed-loop adaptive stimulation of latent neural dynamics is an important and underaddressed problem with clear applications in basic neuroscience and brain-machine interfaces.

**Claims supported by evidence:** Partially. The open-loop optimization and response modeling components are well-supported on real data. The core claim of closed-loop adaptive optimization with non-trivial Ŝ is only supported by toy model evidence.

**Soundness of experiments:** The experiments are competently designed within their scope, but the scope is limited by the trivial stimulation model on real data and the absence of principled baselines.

**Clarity of writing:** Generally clear, though the sparsity formulation in Eq. 8 and the notation ‖u‖₀^max could be improved.

**Value to the research community:** Moderate to high. The framework provides a useful starting point for real-time adaptive stimulation experiments, though practitioners will need to validate it on their own systems before trusting the closed-loop optimization.

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Coupled Transformer Autoencoder | /home/wg25r/review_agent/human_reviews_2026/oeoCgcYIyf.md | 7.00 | Strong real-data validation on multiple datasets; our paper is weaker due to validation gap for core contribution |
| Model-Guided Microstimulation | /home/wg25r/review_agent/human_reviews_2026/S4B7Iq7S3C.md | 6.00 | Directly comparable topic (neural stimulation optimization) with real in-vivo monkey experiments; our paper is weaker because it lacks any real stimulation validation |
| MindPilot | /home/wg25r/review_agent/human_reviews_2026/7jdmXx869Q.md | 5.50 | Closed-loop brain stimulation with human experiments; our paper has broader scope but bigger validation gap |
| PSID with smoothing | /home/wg25r/review_agent/human_reviews_2026/2CJBngOZh5.md | 5.00 | Real data validation but core contribution incremental; our paper has more novelty but less validation |
| STEER framework | /home/wg25r/review_agent/human_reviews_2026/kc5jbYHedw.md | 4.50 | Core contribution only on synthetic data; comparable validation gap |
| FCCA | /home/wg25r/review_agent/human_reviews_2026/geLzACYeE0.md | 4.00 | Insufficient baselines, similar concern; our paper has stronger engineering contribution |
| SYNAPSE | /home/wg25r/review_agent/human_reviews_2026/6YktIxJTJr.md | 4.00 | Simulation benchmark without real validation; our paper has more methodological novelty |
| EEG source localization | /home/wg25r/review_agent/human_reviews_2026/rIIpKdKzkC.md | 1.60 | Entirely simulated with no real data; our paper is significantly stronger |

The paper sits between the 4.5 (STEER, Accept Poster with synthetic-only core validation) and 5.0 (PSID, Reject with real-data validation but incremental contribution) anchors. It has more methodological novelty than PSID but a bigger validation gap than STEER (which at least validates its core claim on synthetic data). Compared to the microstimulation paper (6.0), the absence of any real stimulation experiments is a ~1.5 point penalty. I place this at 5.0 — borderline, with the framework contribution and real-time feasibility barely compensating for the validation gap.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>