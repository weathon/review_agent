Now let me search for calibration papers.Now I have enough material for calibration. Let me compose the final review.

---

## Summary
This paper proposes using manifold capacity theory (MCT), an existing representation-based framework, to operationalize the lazy-versus-rich feature learning dichotomy and to reveal geometric subtypes of feature learning. The authors provide a theorem connecting capacity monotonically to the degree of richness in a simplified 2-layer/one-step setting, empirically demonstrate that capacity tracks feature learning in deeper networks (VGG-11, ResNet-18), and apply the geometric lens to characterize learning stages/strategies, structural inductive biases in RNNs, and correlates of OOD generalization failure.

---

## Strengths

- **Theorem 1 constitutes a non-trivial technical extension**: The paper extends Ba et al. (2022) from a regression to a classification setting, proving a monotone relationship between the learning rate, manifold capacity, and prediction accuracy via Gaussian equivalence tools. Within the scope of the one-step, 2-layer, proportional-asymptotic setting, this derivation is careful and required substantial technical adjustment (analyzing the margin of the Gaussian equivalent model, Proposition 2 relying on Montanari et al., 2019).

- **Representation-based methodology addresses a genuine neuroscience gap**: Unlike weight-change or NTK-based measures that require access to weight matrices, MCT operates on neural activity representations. This is a concrete advantage explicitly demonstrated in Section 5.1, where the RNN analysis would not be possible with weight-space measures in experimental neuroscience.

- **Geometric measures reveal learning strategies and stages beyond the binary dichotomy**: Figure 4a,b shows distinct learning trajectories in (R_mf, D_mf) space depending on richness and initialization wealth — some regimes compress both radius and dimension, others sacrifice radius for lower dimension. Figure 4c identifies four qualitative stages in VGG-11 training that persist long after accuracy has saturated, showing geometry continues to evolve post-convergence in test accuracy. These are novel descriptive insights that a scalar measure cannot provide.

- **Capacity-gap reframing is conceptually useful**: Section 5.1's observation that the gap between initialization capacity and final-epoch capacity (rather than the final value itself) distinguishes RNN learning regimes is a clean, conceptually productive reframing, consistent with the wealthy-lazy vs. poor-rich terminology.

---

## Weaknesses

### Fatal
None.

### Major

- **Theory–experiment gap undermines generality claims**: Theorem 1 is proved only for a 2-layer fully-connected network, with *fixed* readout weights, after *exactly one* gradient step, in the proportional asymptotic limit (P, d, N → ∞), under a teacher-student data model. The paper explicitly acknowledges this in footnote 6: "we follow the convention in (Ba et al., 2022) and study only the first gradient step as the key Gaussian equivalence step might not hold for more steps." Yet the primary experimental contributions involve VGG-11 and ResNet-18 trained to convergence and RNNs trained for 10,000 SGD steps — settings categorically outside Theorem 1's scope. There is no theoretical bridge (not even informal) connecting the one-step, two-layer result to multi-epoch, multi-layer, convolutional, or recurrent settings. This means the theoretical and empirical halves of the paper support each other only weakly; the generalization of the framework rests almost entirely on empirical justification, which is fine but should be presented as such rather than a joint "theoretical and empirical" demonstration as claimed in the abstract.

- **Superiority-of-capacity claim established only on synthetic data**: Section 3.2 and Figure 3 assert "capacity is better than conventional measures in quantifying the degree of feature learning," but the comparison against weight change, NTK-label alignment, and representation-label alignment is performed exclusively on the synthetic 2-layer/Gaussian-cloud testbed. On VGG-11 and ResNet-18 (Figure 2b), only capacity itself is shown — the comparison against other measures is absent. To be credible in the settings that matter most to the paper, this comparison should be replicated on DNNs across the range of scale factors.

### Minor

- **OOD results overclaim "explanation"**: Section 5.2 and Figure 6c caption state that "the expansion of manifold radius and the increase of center-axis alignment explain the failure of OOD generalization in the ultra-rich regime." The data show only correlation: models pre-trained on CIFAR-10 with extreme richness show simultaneous OOD accuracy drop and geometric changes. No intervention is performed (e.g., regularizing radius and checking if OOD accuracy recovers), and no generalization across other OOD pairs is shown. The term "explain" overstates what is demonstrated; "correlate with" or "are concurrent with" would be accurate. The conclusion appropriately leaves intervention as future work, but the caption and abstract do not reflect this limitation.

- **Learning stages characterized from a single run on a single architecture**: The four-stage description in Figure 4c is based on one VGG-11 run on CIFAR-10. Stage boundaries appear to be identified by visual inspection of the heatmap; they are not operationally defined and have not been validated across seeds, architectures, or datasets. Whether these stages generalize beyond this specific setting is an open question that the paper does not address.

- **Missing explanation for why representation-label alignment fails in Figure 3b**: The paper demonstrates that CKA-based rep-label alignment gives the "wrong ordering" of initialization wealth, but provides no account of why this happens. Without a mechanistic explanation, it is unclear whether capacity's advantage is robust or specific to the synthetic Gaussian-cloud setup. This limits how much one can generalize the superiority claim.

### Trivial

- The RNN near-tautology observation: different (R_mf, D_mf, alignment) configurations can sum to the same capacity by construction from MCT's formula. While it is useful to *show* this empirically, the framing could be sharpened so readers understand why equal capacity with different geometry is interesting (e.g., it predicts functionally different behavior that awaits future testing).

---

## Nice-to-Haves

- An intervention-based test of the OOD geometric hypothesis (e.g., regularize manifold radius and measure whether OOD accuracy improves) would transform the OOD section from a descriptive observation to mechanistic evidence.
- Replication of the VGG-11 learning stages on at least one other architecture (e.g., ResNet-18) to check robustness of the stage sequence.
- Comparison of conventional measures vs. capacity on DNN settings (Figure 2 setting) in addition to the synthetic setting.
- Extension of the RNN analysis to connect geometric differences to behavioral predictions (noise robustness, generalization to novel task conditions), which would substantially strengthen the functional interpretation.
- Even an informal multi-step or linear-network argument bridging Theorem 1 to the DNN setting would help anchor the theoretical contribution.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"Tautology" in RNN capacity analysis** (Harsh Critic): The critic argues it is tautological that different geometric configurations can achieve the same capacity. This misunderstands the paper's point: the claim is not that this is mathematically surprising, but that it reveals empirically distinguishable representational strategies at equal performance — a functionally meaningful observation that existing weight-based measures cannot reveal. *Removed as strawman.*

- **Missing comparison with intrinsic dimensionality / linear probing / stronger baselines** (Harsh Critic): These are reasonable future extensions but are not standard baselines in the MCT literature. The paper uses baselines standard to the lazy/rich literature (weight change, NTK alignment, CKA). Criticism about missing TwoNN or other intrinsic dimensionality methods is scope creep. *Weakened to nice-to-have.*

- **Superiority claim vs. DNNs with stronger baselines** (Harsh Critic): The core concern (comparison only on synthetic data) is kept in Major. The specific demand for "linear probing accuracy, CKA between intermediate and output representations" is removed as scope creep beyond what the paper targets. *Partial removal — core concern kept, specific baseline demands removed.*

- **Missing appendix and missing proofs** (any reviewer): Removed per hard rules — parser strips appendix sections.

- **RNN prediction: geometric difference should predict noise robustness, animal data correspondence** (Harsh Critic): These are interesting extensions, but the paper explicitly scopes the RNN section as a demonstration of the method's applicability. Demanding full behavioral validation is outside the paper's stated scope. *Moved to nice-to-have.*

---

## Novel Insights

The most genuinely novel observation synthesized from this paper and the reviewer analysis is the **wealth × richness interaction in manifold geometry**: the paper shows that the geometric *path* a network takes through (R_mf, D_mf) space depends on both the richness of training and the wealth of initialization, and that the same final capacity can be achieved by fundamentally different geometric strategies. This is qualitatively new insight that neither the lazy/rich scalar dichotomy nor weight-space measures reveal, and it suggests that capacity alone (as a scalar) is not a sufficient characterization — the full geometric decomposition carries additional information that may predict robustness, generalization, or biological plausibility of representational strategies.

---

## Calibration Summary

| Anchor Paper | Avg Score | Comparison |
|---|---|---|
| `k9t8dQ30kU` — *Task structure and nonlinearity jointly determine learned representational geometry* | 6.75 (Accept, poster) | Very similar spirit: one-hidden-layer networks, empirical representational geometry, no strong theory. Richer mechanistic dissection of one phenomenon; paper under review has broader coverage but shallower in each area. |
| `vt5mnLVIVo` — *Grokking as transition from lazy to rich training dynamics* | 6.00 (Accept, poster) | Similar setting (lazy/rich dichotomy, 2-layer network theory, DNN empirics). Grokking paper has a crisper unified story; present paper has more breadth but more fragmented. |
| `CtiFwPRMZX` — *A simple connection from loss flatness to compressed representations* | 5.00 (Reject) | Also connects existing tools to a new phenomenon with limited theory and correlational findings; present paper is stronger in experimental breadth and has a genuine theorem (even if limited). |
| `A9yKCUQNnc` — *Understanding connection between low-dimensional representation and generalization* | 3.00 (Reject) | Also claims to bridge representation geometry and generalization theory, but was rejected for weak theory and superficial treatment. Present paper is clearly stronger in execution. |
| `dEypApI1MZ` — *How Feature Learning Can Improve Neural Scaling Laws* | 7.20 (Accept, Spotlight) | Stronger theoretical results, tighter theory-experiment connection; paper under review falls short of this bar. |

The paper under review sits above CtiFwPRMZX (richer experiments, genuine theorem, multiple applications) and above the low-scoring anchors (A9yKCUQNnc), but below vt5mnLVIVo (cleaner story, tighter claim-to-evidence fit) and well below dEypApI1MZ (theoretically tighter). The cluster of most similar papers falls in the 5–6.75 range; given the paper's genuine breadth but meaningful gaps in claim substantiation, a score of **5.5** is appropriate — above borderline reject, reflecting a paper with real contributions but requiring non-trivial revisions to properly scope its claims and demonstrate superiority in DNN settings.

---

## Score and Decision

**Originality**: Moderate. MCT itself is not new; applying it to feature learning regimes is the novel angle. The geometric decomposition of learning strategies (Figure 4) is a genuinely new observation.

**Importance of research question**: High. The lazy/rich dichotomy is a central question in both ML theory and computational neuroscience, and a representation-based approach is genuinely needed.

**Claims vs. support**: Mixed. The theory supports a very narrow setting; the empirical claims for DNNs are supported but comparisons with baselines are incomplete. The OOD "explanation" is overclaimed.

**Soundness of experiments**: Adequate. The experiments on VGG-11/ResNet-18 are real, but single-run without replication across seeds or architectures for some key findings.

**Clarity**: Good overall.

**Value to research community**: Moderate-to-high for the neuroscience community; the MCT framework with geometric decomposition is likely to be useful as a tool even if individual results need follow-up.

**Final Score: 5.5** — Marginally below the typical acceptance bar for a strong venue. The paper makes genuine contributions but the theory-experiment gap and incomplete comparison study are substantive concerns that require revision.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>