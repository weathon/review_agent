Good, I've read the full paper. Now let me run calibration searches in parallel.Now I have sufficient information to write the consolidated review. Let me synthesize everything.

---

## Summary

This paper develops a kernel theory of compositional generalization for models with compositionally structured (e.g., disentangled) representations. The central result, Theorem 4.2, proves that such models are constrained to "conjunction-wise additive" computations on test inputs—they can only sum values assigned to component conjunctions seen during training. This partitions the compositional task space, ruling out transitive equivalence while permitting transitive ordering and context dependence. Building on this, the paper analytically characterizes a "memorization leak" failure mode (Proposition 5.1, giving a closed-form expression for the generalization distortion as a function of representational salience and training size) and empirically characterizes a "shortcut bias" failure mode. The theory is empirically validated on ConvNets, ResNets, and ViTs across MNIST and CIFAR-10 compositional tasks, showing qualitative consistency with the theory's predictions.

---

## Strengths

- **Theorem 4.2 gives a precise, general characterization** of what compositionally structured kernel models can compute on the test set (conjunction-wise additivity, Eq. 2), providing an exact partition of the compositional task space (Fig. 2d). This is a meaningful, non-trivial result that does not follow from prior work.

- **Sharp transitive equivalence vs. transitive ordering distinction (Section 4.3):** The observation that these superficially similar tasks fall on opposite sides of the solvability boundary—one component-wise additive (ordering), the other fundamentally not (equivalence)—is a non-obvious theoretical consequence that explains empirical confusion in prior literature.

- **Proposition 5.1 delivers a concrete, testable closed-form prediction** (Eq. 3): the memorization leak slope $m = \frac{p \cdot S(1;2)}{1+(p-2)S(1;2)}$ depends only on salience and training set size, and notably predicts no difference between interpolation and extrapolation regimes. This is the kind of falsifiable, quantitative statement that makes theoretical work valuable.

- **Representational salience $S(k;C)$ (Section 5.1)** reduces the representational geometry to $C-1$ free parameters, giving an interpretable and manipulable summary—further supported by the empirical finding (Fig. 5a) that spatial distance of digits in ConvNets provides a practical handle on $S(1;2)$.

- **Multi-architecture empirical validation (Section 6)** covers ConvNets, ResNets, and ViTs across MNIST and CIFAR-10, confirming qualitative predictions about proportional compression (Fig. 5b), slope increasing with distance and training set size (Figs. 5c–d), and shortcut-driven failure on CD-3 (Fig. 5e).

- **Principled benchmark design implication:** Conjunction-wise additivity tells us how to construct tasks that are unsolvable by kernel models (Appendix E.2.3), directly grounding the development of compositional benchmarks.

---

## Weaknesses

### Fatal
None.

### Major

- **Section 6 framing overstates the theory–empirics connection.** The section title states the theory "can describe the behavior of deep networks," but the Discussion explicitly acknowledges "we do not provide any quantitative bounds." The validation is qualitative: the paper confirms directional trends (slope increases with distance, with training set size) but never plots predicted vs. observed slope using the closed-form Eq. 3. Since $S(1;2)$ is measured from an intermediate ConvNet layer (Fig. 5a) and all ingredients are available, a quantitative comparison would have directly tested whether the kernel theory describes deep network behavior numerically, not just directionally. As stated, the evidence supports "qualitative consistency" rather than the stronger "describes." The authors should either scope the claim down explicitly throughout Section 6 (not just in Discussion) or add the quantitative comparison.

- **Asymmetry in theoretical treatment of the two failure modes.** Memorization leak is formalized as Proposition 5.1 with a closed-form expression and precise conditions ($\mathcal{V}$ and $\mathcal{W}$ zero-mean). Shortcut bias is analyzed only empirically—the paper identifies a qualitative regime (Fig. 4c: high $S(2;3)/S(1;3)$ yields generalization) but never derives the boundary analytically. Both are presented as coequal theoretical contributions, yet only one achieves formal characterization. The absence of a theorem for the shortcut bias boundary is a gap that limits the precision of this contribution.

### Minor

- **Kernel regime assumption for deep networks is not defended in Section 6.** The paper correctly notes in the Discussion that the theory applies to the kernel regime, but Section 6 uses fully end-to-end backprop-trained networks (explicitly stated). There is no experiment comparing linear-probe (kernel-regime) training to full fine-tuning to establish whether the qualitative behavior originates from kernel-regime dynamics or other mechanisms. Even a small ablation showing that fixed-feature + linear-readout models exhibit the same qualitative patterns would meaningfully tighten the connection between theory and experiment.

- **"Much subtler effect for ViTs" on slope–distance (Fig. 5c) is acknowledged but unexplained.** A brief discussion of why ViTs (with global attention) would differ from ConvNets (local weights) would connect the architecture-specific finding back to the theory, since the theory's prediction about salience and distance implicitly relies on local weight structure.

- **Zero-mean assumption in Proposition 5.1.** The requirement that $\mathcal{V}$ and $\mathcal{W}$ have zero mean is stated but its necessity for the result is never discussed. Training sets in practice need not be zero-mean; clarifying whether the conclusion is approximately valid or fails significantly for non-zero-mean sets would be useful.

### Trivial

- **Salience $S(k;C)$ is formally defined only in Appendix B.1.** Main-text readers must take its properties on faith until the appendix. A one-sentence definition in Section 5.1 (alongside the intuitive description already present) would improve self-containedness.

---

## Nice-to-Haves

- A predicted-vs.-observed slope plot (using Eq. 3 and measured $S(1;2)$) across training set sizes and architectures would transform the empirical validation from qualitative to quantitative—a straightforward addition given that all components are already measured.
- Empirical validation of the transitive equivalence impossibility result (Section 4.3): confirming that ConvNets/ResNets/ViTs fail on transitive equivalence while succeeding on transitive ordering would provide direct experimental grounding for the theory's most striking theoretical prediction.
- Analysis of sensitivity to violations of Definition 3.1 (the exact-count symmetry assumption): the randomly permuted category assignment controls for one confound, but characterizing how well real networks actually satisfy the assumption, and what approximately holds when they don't, would strengthen the theoretical framework's applicability.
- Extension to asymmetric compositional structure (where different components have different intrinsic saliences) would broaden the scope of the theory.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "Definition 3.1 imposes a very specific symmetry—sensitivity to violations is never analyzed."** Partially addressed: the paper controls for non-equidistance of categories via random permutation of category assignments, and the theory is explicitly for the idealized symmetric case. The criticism about sensitivity analysis is reasonable as a nice-to-have but the existing control is adequate for the empirical section.

- **Harsh Critic: "Space of achievable salience profiles for practical networks is not characterized."** This is outside the stated scope—the paper characterizes the impact of salience on generalization given a salience profile, not the engineering question of what architectures achieve what salience. This is scope creep.

- **Harsh Critic: "$n=10$ seeds is low."** With random permutation of category assignments, 10 seeds is standard for this kind of controlled experiment on MNIST/CIFAR; this is not a meaningful weakness.

- **Harsh Critic: "Proposition 4.1 is expected given the literature."** True but it is a necessary lemma; calling it "expected" does not diminish its role in the paper's argument.

- **Harsh Critic: "Theorem 4.2 proof is deferred—cannot be verified."** Per hard rules, appendix proofs exist in the original submission; this is not a weakness.

---

## Novel Insights

The conjunction-wise additivity framework provides a genuinely useful lens: it shifts the question "can a kernel model generalize compositionally?" from an architectural question (is the model expressive enough?) to a structural one about the interaction between representational geometry, dataset statistics, and the algebraic form of the target function. The sharp distinction between transitive ordering (additive, solvable) and transitive equivalence (non-additive, unsolvable) is a particularly illuminating consequence — not because equivalence is hard in some informal sense, but because it provably cannot be expressed as a conjunction-wise sum over component pairs. The identification that memorization leak is controlled by exactly two parameters ($S(1;2)$ and training set size $p$, with interpolation/extrapolation playing no role) is a surprising and clean quantitative insight that could directly guide dataset design choices.

---

## Suggestions

1. **Re-scope Section 6's title and framing** from "our theory describes deep network behavior" to "our theory's qualitative predictions are consistent with deep network behavior," matching the Discussion's more cautious language throughout.
2. **Add a quantitative predicted-vs.-observed slope comparison** using Eq. 3 and measured $S(1;2)$ values, with error bars across seeds. This is a minor implementation effort with outsized payoff for the paper's empirical credibility.
3. **Provide at least an empirical characterization of the shortcut bias threshold** (the regime boundary in $S(2;3)/S(1;3)$ space), even if a full analytic theorem is deferred, to put the two failure modes on more equal footing.
4. **Add a linear-probe baseline** in one architecture (e.g., fixed ConvNet features + linear readout) to show that the kernel-regime approximation captures the qualitative behavior of the full network, directly connecting theory to the architecture tested.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Score | Comparison to paper under review |
|------|-----------|----------------------------------|
| `/human_reviews/7VPTUWkiDQ.md` | 7.33 (oral) | Closest topical match — provable compositional generalization via identifiability theory, similar scope (one architecture type, synthetic data); this paper has broader empirical scope (3 architectures, 2 datasets) but the theory-to-empirics gap is larger |
| `/human_reviews/3SJE1WLB4M.md` | 8.0 (spotlight) | Tight kernel analysis with precise bounds; higher bar than this paper (stronger quantitative theory-empirics fit) |
| `/human_reviews/GH2LYb9XV0.md` | 5.5 (poster) | Clean solvable theory model (grokking), qualitative predictions confirmed, some novelty doubts; similar in spirit but that paper had conceptual disputes about whether the phenomenon modeled was genuine — this paper's modeling choices are less contested |
| `/human_reviews/D1w3huGGpu.md` | 4.75 (reject) | Compositional generalization in modular architectures; limited novelty and weaker theoretical grounding than the paper under review |
| `/human_reviews/RIaIpdUCPb.md` | 3.0 (withdrawn) | Geometry + compositional generalization, weak novelty and no formal main theorem; substantially weaker than this paper |
| `/human_reviews/2NwHLAffZZ.md` | 2.33 (reject) | Kernel theory lacking convincing validation; much weaker theoretical contribution than this paper |

**Assessment:** The paper sits clearly above the medium anchor (GH2LYb9XV0, 5.5) and the medium-reject (D1w3huGGpu, 4.75): it has a real, general theorem (not just a special-case solvable model), a closed-form proposition, and multi-architecture validation. It falls somewhat below the high anchor 7VPTUWkiDQ (7.33) primarily because the theory-to-deep-network framing is stronger than the evidence warrants and because the shortcut bias contribution lacks a formal characterization parallel to Proposition 5.1. The 7VPTUWkiDQ paper also had limited experiments but its theory directly implied the experiments; here the gap between kernel-regime theory and full backprop training is a genuine (if acknowledged) unresolved issue.

Placing this paper at **6.5**: above poster-level medium but below spotlight/oral, reflecting a genuine and useful theoretical contribution with an honest but incompletely bridged gap between the kernel-regime theory and deep network behavior.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>