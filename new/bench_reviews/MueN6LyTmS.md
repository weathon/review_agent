Now let me read the calibration papers to help with scoring.Now I have enough information to write the consolidated review. Let me read a few more calibration papers quickly.## Summary

EquiEvo introduces subequivariant graph neural networks (GNNs) into the morphology-behavior co-evolution framework by constructing dynamic, learnable Local Reference Frames (LRFs) to project vector states into an invariant form. The paper also proposes 3DS-MB, an extended benchmark based on MuJoCo featuring variable-direction navigation and adversarial sumo tasks, where spatial geometric information is richer than prior "move forward" benchmarks. Experiments show that EquiEvo outperforms internal ablation variants in reward and win rate across three task settings.

---

## Strengths

- **Principled motivation for invariant value estimation in co-evolution.** The "Invariant of Morphology Value" argument (Section 3.1) — that consistent feedback for morphology transformation requires an invariant value function, which in turn requires equivariant behavior control — is conceptually sound. If the value estimate changes with the agent's global orientation, the morphology update signal is corrupted; subequivariance addresses this systematically.

- **Learned LRF outperforms hand-crafted normalization.** Figure 7 compares EquiEvo against two principled hand-crafted alternatives (goal-direction LRF and heading-direction LRF). Learned subequivariant canonicalization outperforms both, demonstrating the value of data-driven LRF discovery over rigid task-specific heuristics.

- **Meaningful morphology-task analysis.** The forward-reward ablation (Figures 9–10) is genuinely insightful: switching from a distance-only reward to a forward-reward causes the evolved morphology to shift from radial symmetry to lateral symmetry, demonstrating that morphology tracks task structure rather than predefined biases.

- **Clear architectural contribution.** Distinguishing the morphology graph $\mathcal{G}_m$ (not environment-interacting) from the state graph $\tilde{\mathcal{G}}_g$ (environment-interacting) is conceptually helpful and correctly scopes where equivariance is needed.

- **Benchmark extension.** The 3DS-MB environments (variable-goal navigation, multi-agent sumo) do address a genuine gap relative to fixed-direction locomotion benchmarks.

---

## Weaknesses

### Fatal
*(None. The paper has real contributions and the core mechanism is sound.)*

### Major

- **All primary baselines are internal ablations.** Section 4.2 defines four variants — EquiEvoX, EquiX, EvoX, X — all derived by removing components from the authors' own pipeline. There is no comparison against any independent co-evolution method or alternative symmetry-aware architecture. This makes it impossible to assess whether subequivariance is the decisive advance versus one of many possible architectural improvements. Given that the abstract claims to "significantly outperform existing approaches," the baseline scope is far too narrow for this framing.

- **Three seeds is insufficient for RL/co-evolution claims.** Section 4.2 explicitly states "3 seeds." For PPO-based morphology learning, which is notoriously high-variance, three runs provide unreliable variance estimates. The abstract states the method "consistently and significantly outperforms existing approaches," but this strength of claim requires either larger seed counts or pairwise significance tests, neither of which is provided. This is especially problematic for the sumo comparisons, where adversarial training further amplifies variance.

- **Evaluation metrics do not isolate morphology quality.** The paper's central contribution is improving *morphology evolution*, yet the only reported metrics are cumulative reward (navigation) and win rate (sumo). These are task-performance metrics; they cannot distinguish whether gains come from better morphology, easier policy learning due to equivariant inductive bias, or reward shaping interactions. Quantitative morphology metrics — such as symmetry scores of evolved bodies, inter-seed consistency of evolved structures, or decoupled evaluation of evolved morphologies under a fixed control policy — would be needed to substantiate the morphology-specific claims.

- **Mixed evaluation regimes across tasks limit generalizability of the main claim.** Ant Navigation includes full structural transform (from an atomic torso), while Humanoid Navigation and Sumo omit structural transform entirely (attribute-only adaptation). These are materially different co-evolution settings. As the paper itself states: *"To maintain the humanoid structure, we skip the structural transform stage"* and *"Following the CompetEvo setup, the structure transform stage is omitted."* While each choice is individually justified, the result is that co-evolution with full structure search is only tested in one of three tasks, which limits how broadly the paper's claims about co-evolution can be stated.

### Minor

- **The "reduces a 3D problem to a simpler 1D/2D one" claim is unsubstantiated.** The introduction states: *"under rotational symmetry, states and actions in any direction can be treated as equivalent, effectively reducing a 2D/3D problem to a simpler 1D/2D one."* This intuitive claim is never formalized, quantified, or experimentally demonstrated. No representation dimensionality analysis, effective complexity measure, or action-space comparison is provided.

- **LRF robustness not discussed.** Equations 9–11 show the orthonormalization procedure using $\bar{u}$ and $\bar{v}$ predicted from the root node. The paper does not address how the method handles cases where $\bar{u}$ or $\bar{v}$ are near-zero (e.g., for morphologies with symmetric configurations), which could make the resulting frame degenerate or highly sensitive to noise.

- **EvoHumanoid failure is insufficiently analyzed.** Figure 4b shows EvoHumanoid performing *worse* than the no-evolution Humanoid baseline — a dramatic optimization failure. The paper attributes this to search space expansion, but the mechanism is left unexplored. Given that EquiEvo's key claim is *rescuing* this regime, a more principled analysis of when and why morphology transform without equivariance fails would strengthen the paper.

- **Actor/critic equivariance ablation is restricted to one setting.** Figure 8 ablates Actor vs. Critic equivariance only on EvoAnt. If the paper claims a general principle about the necessity of equivariance in both Actor and Critic for co-evolution, this ablation should be replicated on at least one other setting.

- **Morphology-Behavior Mapping analysis is deferred to appendix.** Section 4.4 states: *"We conduct experiments on the Ant Navigation task to analyze robust Morphology-Behavior Mapping, with details in Appendix E.3."* Since this is one of the paper's three stated contributions (Abstract), its main-paper treatment is too minimal.

### Trivial

- **"More akin to that of a professional athlete"** (Section 4.3) is colloquial and not a scientific statement. The morphology visualizations in Figure 6 are useful, but the interpretive language should be tightened.

---

## Nice-to-Haves

- **Test-time rotational generalization experiment.** Training in one set of orientations and testing on rotated environments would provide the most direct evidence that equivariance actually functions as claimed, rather than inferring it from architecture design alone.
- **Quantitative symmetry metrics for evolved morphologies.** A bilateral/radial symmetry score for evolved limb configurations would ground the qualitative claims in Figures 6 and 10.
- **Computational cost analysis.** Reporting wall-clock time and parameter counts relative to non-equivariant baselines would help practitioners assess the practical trade-off.
- **More seeds and/or significance tests.** Even 5 seeds with standard deviations prominently reported would substantially improve the credibility of the statistical claims.
- **Structure transform search space details.** For Ant Navigation starting from an "atomic morphology," the maximum number of limbs, connectivity constraints, and overall search space size are not stated in the main paper.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **[Harsh Critic] "Benchmark lacks external comparisons therefore nothing can be concluded."** Removed as overstated — the paper is transparent about using internal ablations, and the hand-crafted normalization comparison provides meaningful external signal. The concern about narrow baselines is kept but moderated.
- **[Harsh Critic] "Bilevel objective is not principally solved."** Removed — the paper is building on established co-evolution practice (Transform2Act) and does not claim to solve the bilevel problem exactly. This is standard in the field.
- **[Harsh Critic / Spark] "No evaluation on prior fixed-direction benchmarks."** Removed as scope creep. The paper explicitly focuses on direction-aware 3D tasks; evaluating on fixed-direction tasks the method was not designed for is not a fair requirement.
- **[Human Finder / Spark] "Real-world applicability."** Removed — this is a simulation-focused paper. Demanding real-robot evaluation is outside the paper's stated scope.
- **[Human Finder] Specific citation of other papers as missing related work.** Removed per hard rules — external references cannot be confirmed.
- **[Harsh Critic] "Sumo opponent sampling protocol omitted."** Partially valid but the paper cites Bansal et al. (2018) for the protocol and states teams using different methods compete against each other. Moved to minor/nice-to-have level.
- **[Harsh Critic] "Philosophical claim that bilateral constraints diverge from evolutionary principles."** Removed as a style/philosophical nitpick — the paper's argument stands regardless of this framing.

---

## Novel Insights

The most genuinely novel observation across all three reviews is the **value-consistency argument for morphology evolution**: existing co-evolution frameworks that lack geometric symmetry produce inconsistent advantage estimates for morphology transform actions when the agent's global orientation changes, effectively injecting irrelevant variance into the morphology update signal. EquiEvo's use of an invariant value function directly addresses this previously unstated problem. The forward-reward morphology ablation (Figure 10b vs. 10c) further demonstrates that under EquiEvo, evolved morphologies track task symmetry in a way that non-equivariant methods cannot achieve at comparable sample budgets — a finding that suggests symmetry is not merely an inductive bias for control but a prerequisite for reliable morphology-behavior co-evolution in direction-rich environments.

---

## Suggestions

1. **Expand baselines to at least one independent co-evolution method** to support the "outperforms existing approaches" framing in the abstract.
2. **Increase to at least 5 seeds** and report pairwise tests or confidence intervals on the main navigation/sumo results.
3. **Add a quantitative morphology quality metric** (e.g., bilateral symmetry error, cross-seed morphology consistency) to the main evaluation section to substantiate morphology-specific claims.
4. **Move the Morphology-Behavior Mapping analysis** from the appendix to the main paper, or expand Section 4.4 to summarize the key finding.
5. **Discuss LRF degeneracy handling** — what occurs when predicted vectors $\bar{u}$, $\bar{v}$ are near-zero or nearly parallel.
6. **Provide a test-time rotation experiment** as a direct validation that the equivariance mechanism generalizes at inference time, not just architecturally.

---

## Score and Decision

**Calibration:**

| Paper | Topic | Score | Decision |
|---|---|---|---|
| BodyGen (cTR17xl89h) | Embodiment co-design, strong experiments, 60% gain, principled contributions | 8, 6, 8, 8 | Accept (Spotlight) |
| Beyond Canonicalization (vDp6StrKIq) | Local reference frames, equivariant GNN | 6, 8, 5 | Accept (Poster) |
| MECE (BgzE4zwkFW) | Morphology-environment co-evolution, narrow baselines | 5, 6, 5 | Reject |
| CoSIL (Iz230vHUy0) | Morphology-behavior co-adaptation, poor baselines | 3, 3, 5, 3 | Reject |
| Equivariant RL (vl3F3s8OMg) | Euclidean symmetry in RL, toy experiments | 3, 3, 6, 5 | Reject |

This paper sits between MECE and Beyond Canonicalization in quality. Like MECE, it has narrow baselines and borderline statistical rigor; unlike MECE, it has a clearer technical contribution (learned LRF for co-evolution value consistency), a non-trivial benchmark extension, and the morphology-task analysis is genuinely informative. It is substantially weaker than BodyGen, which had comprehensive baselines and a 60% improvement claim backed by strong evidence. The use of 3 seeds and internal-only baselines prevents it from matching the empirical standard of vDp6StrKIq (score ~6, accepted). The theoretical motivation is sound but the claims in the abstract overreach relative to what the experiments establish.

**Assessment axes:**
- *Originality:* Moderate — applying subequivariant GNNs to co-evolution is novel, though the individual components (subequivariant GNNs, LRF canonicalization, Transform2Act-style co-evolution) are from prior work.
- *Importance of research question:* High — morphology-behavior co-evolution in 3D is a meaningful open problem.
- *Claims well-supported:* Weak — the headline claims about morphology improvement are backed by qualitative figures and reward curves only.
- *Soundness of experiments:* Fair — the experimental design is reasonable but underpowered (3 seeds, internal baselines only).
- *Clarity of writing:* Good — the paper is generally readable; the method is described clearly.
- *Value to community:* Moderate — the benchmark extension and the LRF-for-co-evolution framing have value, but the paper needs stronger empirical backing to be fully convincing.

**Final Score: 5.0** — Positioned above CoSIL/vl3F3s8OMg (score 3–4, rejected for weak baselines and poor evidence), comparable to MECE (score 5, rejected for narrow contribution scope), and below vDp6StrKIq (score ~6.5 avg, accepted as poster with stronger empirical domain). The paper has a real idea but insufficient experimental support for its claims as currently written.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>