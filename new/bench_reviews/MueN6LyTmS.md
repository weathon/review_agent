## Summary
This paper proposes *EquiEvo*, a co-evolution framework that injects geometric symmetry into morphology-behavior co-evolution in 3D environments by learning dynamic Local Reference Frames (LRFs) via subequivariant graph neural networks. It introduces the *3DS-MB* setup—navigation and sumo tasks with variable-direction goals—and reports that EquiEvo outperforms non-equivariant and static-morphology baselines, with qualitative visualizations of evolved morphologies.

## Strengths
- **Novel problem setup.** The paper correctly identifies that prior co-evolution benchmarks (e.g., Transform2Act, CompetEvo) focus on fixed-direction tasks and proposes *3DS-MB* with variable-direction navigation and adversarial sumo, formalized via subequivariant graphs under $E_g(3)$ rather than unstructured topology graphs (Table 1, Section 3.1).
- **Principled geometric mechanism.** EquiEvo integrates dynamic, learned LRF canonicalization via subequivariant message-passing networks into the co-evolution loop, projecting steerable vector states into an invariant frame (Section 3.2, Eq. 5–12). This is technically grounded in recent geometric deep-learning work.
- **Internal actor-critic ablation.** Figure 8 isolates the role of equivariance within the same architecture, showing a clear ranking of EquiActorCritic > EquiActor > EquiCritic > NoEqui on EvoAnt, which supports the claim that symmetry constraints benefit both policy and value estimation.
- **Qualitative morphology visualizations.** Figures 6 and 10 provide interpretable evidence that task reward structure correlates with qualitatively different evolved body plans (e.g., radial vs. lateral symmetry in ants).

## Weaknesses

### Fatal
None.

### Major
- **Confounded baseline comparisons undermine causal attribution to subequivariance.** The central claim is that subequivariance drives improved co-evolution, but the non-equivariant *Evo* baseline and the hand-crafted normalization ablations (*Evo+HN/DN*, Figure 7) all **treat vector features $\vec{z}$ as scalars**, destroying geometric information (Section 4.4). Without a non-equivariant baseline that retains raw vector features (e.g., a standard GNN or MLP on vector coordinates without LRF constraints), the observed gains may reflect merely *using* geometric information that the Evo baseline discards, rather than symmetry constraints. This is a critical methodological gap that directly weakens the paper’s core argument.
- **Generalization claims are unsupported by experiments.** The abstract and introduction assert that EquiEvo improves “generalization ability” and enables the joint policy to “generalize to diverse task spatial structures.” However, all experiments train and evaluate on the same distribution (e.g., goals sampled within a fixed radius $[3,4]$, fixed arena dimensions). No zero-shot or few-shot out-of-distribution experiments (larger arenas, farther goals, unseen opponents) are provided (Abstract, Section 1, Section 4.1).

### Minor
- **No statistical validation despite strong language.** The abstract and results repeatedly use “significant,” yet no statistical significance tests are reported, and all learning curves (Figures 4, 5, 7, 8, 9) omit error bars or confidence intervals despite training with only three seeds. This makes it impossible to assess whether performance differences are robust or spurious (Section 4.2).
- **Inconsistent effect of morphology transform is not reconciled.** *EvoAnt* improves over the static *Ant* baseline (Figure 4a), whereas *EvoHumanoid* degrades below the static *Humanoid* baseline (Figure 4b). The paper explains the latter via search-space expansion but offers no systematic account for why the same phenomenon does not occur in the Ant task, weakening the central narrative that subequivariance is what makes morphology evolution viable (Section 4.3).
- **Qualitative morphology claims lack quantitative support.** Visualizations of evolved morphologies (Figures 6, 10) are accompanied by anecdotal descriptions but no quantitative metrics (e.g., bilateral-symmetry scores, center-of-mass analysis, moment of inertia), limiting the strength of conclusions about task-driven morphological emergence.
- **Degenerate case in LRF construction is unaddressed.** The orthonormal basis construction (Eq. 9–11) divides by $\|\bar{u} - \langle\bar{u}, \bar{e}_3\rangle \bar{e}_3\|$; if the predicted vector $\bar{u}$ is parallel to gravity $\vec{g}$, this denominator vanishes. The paper does not discuss how this singularity is handled (Section 3.2).
- **Overstated benchmark framing.** *3DS-MB* comprises only three task variants and lacks standardized evaluation protocols or broad baseline comparisons. Framing it as a “benchmark” rather than a task suite is an overstatement (Abstract, Section 3.1).

### Trivial
- **Imprecise conceptual framing.** The abstract describes equivariance as “effectively reducing a 2D/3D problem to a simpler 1D/2D one,” which is misleading: equivariance enforces consistent transformation laws under group actions, not dimensional reduction or equivalence of all directions (Section 1).

## Nice-to-Haves
- Out-of-distribution generalization experiments (e.g., larger goal radii, novel arena sizes, cross-play against held-out opponent morphologies) to substantiate generalization claims.
- A non-equivariant geometric baseline that processes raw vector features without LRF canonicalization, to disentangle the benefit of subequivariance from simply using geometric information.
- Success-rate or time-to-goal metrics for navigation tasks to complement cumulative reward.
- Quantitative morphological metrics and trajectory visualizations on identical initial states to clarify *how* symmetry exploitation improves behavior.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **“Strong empirical evidence that subequivariance is critical”** — This strength is undermined by the confounded baseline comparison. Because the *Evo* baseline discards vector features, the ablations do not isolate subequivariance as the causal driver of performance.
- **“Systematic ablations validating design choices” (Figure 7)** — The hand-crafted normalization ablation is confounded: *Evo+HN/DN* treat vectors as scalars, so the comparison does not cleanly separate learned equivariance from manual symmetry priors.
- **Criticism that the paper fails to explain how subequivariance affects the morphology-transform policy** — The paper explicitly scopes morphology attributes as scalar and invariant, and the morphology transform operates on these scalar features (Section 3.2). Criticizing the absence of equivariance in morphology transform is scope creep.
- **Concerns that cited benchmarks/models are unreleased or unverifiable** — The paper cites Transform2Act, CompetEvo, MuJoCo, and standard geometric deep-learning libraries; all exist and are publicly available. Doubting their existence reflects a reviewer knowledge gap, not an author error.
- **Missing appendix / deferred proofs** — The paper references Appendix A, C, D, and E; these sections were stripped by the parser. The original submission contains them.
- **Typos, grammar, and formatting artifacts** — These are parser errors, not present in the original submission.
- **Demand for real-world robot experiments** — The paper is a simulated co-evolution study; requiring physical robot validation is outside the standard scope of this subfield and constitutes scope creep.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- **Add a non-equivariant vector-feature baseline.** Retain raw 3D vector inputs (positions, velocities, goal directions) in a standard GNN or MLP without LRF constraints, matched for capacity, to isolate the effect of subequivariance from generic geometric feature usage.
- **Report variance and statistical tests.** Plot standard deviation or confidence intervals on learning curves and report paired statistical tests across the three seeds to justify claims of “significant” improvement.
- **Temper generalization claims or add OOD experiments.** Either conduct explicit out-of-distribution tests or revise the abstract and introduction to avoid claiming generalization that is not empirically demonstrated.
- **Quantify morphological properties.** Compute symmetry indices, center-of-mass trajectories, or moment-of-inertia statistics to strengthen the qualitative claims about evolved morphology.

## Score and Decision

**Calibration anchors used:**
- **High (≥6):** *BodyGen* (avg 7.5, Spotlight) and *HEPi* (avg 8.0, Oral). Both feature rigorous baselines, strong empirical validation, and clear technical novelty. EquiEvo’s empirical argument is notably weaker due to confounded comparisons and unsupported generalization claims, placing it well below this band.
- **Medium (~5):** *RDC-SAT* (avg 5.75, Poster) and *SiT* (avg 5.67, Reject). *RDC-SAT* provides systematic experiments and cross-dataset generalization despite some A+B criticism. *SiT* offers architectural novelty but suffers from weak baselines and limited RL evaluation. EquiEvo is comparable in novelty to *SiT* but its baseline confounding is more severe because it directly threatens the causal interpretation of the central claim, and it lacks the experimental breadth of *RDC-SAT*.
- **Low (≤4):** *Gaitor* (avg 4.5, Reject) and *Reducing Symmetry Mismatch* (avg 3.5, Reject). *Gaitor* has real-world experiments but methodological confusion and limited comparisons. *Reducing Symmetry Mismatch* is a minimal preprocessing contribution. EquiEvo is more technically substantive than the low anchors and presents a clearer problem formulation than *Gaitor*, but its core empirical argument is undermined by the confounded baseline and unsupported generalization claims.

**Comparison:** Relative to these anchors, EquiEvo sits near the lower end of the medium band—above the purely weak/low contributions but below papers with solid, unconfounded empirical validation. The confounded baseline and unsupported generalization claims are serious enough to weigh against acceptance.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>