Now let me search for calibration anchors.Now I have enough information to write the final review. Let me synthesize everything.

---

## Summary

This paper develops a kernel theory of compositional generalization for models with "compositionally structured" representations — those whose kernel depends only on the number of overlapping components between inputs. The central theoretical result (Theorem 4.2) proves that such models are constrained to *conjunction-wise additive* computations, partitioning the compositional task space into solvable and unsolvable tasks. A key corollary is that kernel models cannot transitively generalize equivalence relations. Within the solvable regime, Proposition 5.1 gives an exact closed-form formula for "memorization leak" — a systematic distortion of generalization due to ℓ₂-norm minimization — as a function of representational salience and training set size. The authors also identify a "shortcut bias" failure mode on context dependence. All theoretical predictions are empirically tested in ConvNets, ResNets, and Vision Transformers on MNIST/CIFAR-10 compositional tasks.

---

## Strengths

- **Theorem 4.2 (conjunction-wise additivity)** is a clean, general, non-trivial characterization of what compositionally structured kernel models can compute. The result partitions the compositional task space (Fig. 2d) in a way prior work had not formalized and has clear downstream consequences — including the impossibility of transitive equivalence generalization.

- **Proposition 5.1** yields an exact closed-form memorization-leak factor $m = \frac{p \cdot S(1;2)}{1 + (p-2)S(1;2)}$, quantitatively relating representational salience and training set size to systematic generalization distortion. This level of analytical precision is uncommon in the compositional generalization literature.

- **The transitive equivalence / transitive ordering distinction** (Section 4.3) is formally sharp and practically important: two tasks that are routinely conflated in cognitive science differ fundamentally in what kernel models can express. This kind of formal discrimination is exactly what the field needs.

- **The representational salience metric** $S(k;C)$ (Section 5.1) reduces the full compositional kernel structure to $C-1$ free parameters, providing an interpretable and practically measurable descriptor. The manipulation of $S(1;2)$ via spatial distance between digits (Fig. 5a) is both creative and experimentally effective.

- **Three distinct predictions of Proposition 5.1 are confirmed in deep networks** (Fig. 5b–d): (i) proportional distortion of test predictions, (ii) slope increases with $S(1;2)$ via spatial distance, (iii) slope increases with training set size. The same pattern holds across ConvNets, ResNets, and ViTs, strengthening confidence that the theory captures real phenomena beyond the strictly theoretical domain.

- **Shortcut bias analysis for context dependence** (Fig. 4c–d) mechanistically explains why CD-3 causes 0% accuracy in many networks while CD-1/CD-2 succeed — connecting spurious correlations in the training data to specific representational geometry conditions.

---

## Weaknesses

### Fatal
None.

### Major

- **Gap between kernel-regime theory and feature-learning empirics, not fully bridged.** The theory characterizes kernel models with *fixed* compositionally structured representations — explicitly the kernel/NTK regime (large initial weights, wide networks, training only the linear readout). Yet Section 6 trains ConvNets, ResNets, and ViTs fully via backpropagation, firmly in the feature-learning regime (as acknowledged in Section 2: "the feature-learning regime yields more substantial changes in neural networks' internal representations"). The paper empirically shows that the theory's *predictions* hold qualitatively in these networks, which is a valid and interesting finding. However, the section title ("Our theory can describe the behavior of deep networks") and phrases like "validate our theory" are somewhat stronger than what is formally established — the paper demonstrates correlation, not a theoretical derivation. The authors acknowledge this in the Discussion ("we demonstrate that our theory captures many qualitative phenomena in deep neural networks, but do not provide any quantitative bounds") and in the Limitations section, but the distinction between "theory applies" and "theory correlates empirically" deserves more careful framing throughout Section 6.

- **Definition 3.1's symmetry condition is not verified for trained networks.** The entire theoretical framework rests on Definition 3.1 (the kernel depending *only* on the count of overlapping components, not on which components overlap). The paper measures $S(1;2)$ from intermediate-layer representations (Fig. 5a), implicitly treating these as approximately compositionally structured. However, the full overlap-count symmetry condition is never formally checked for any trained architecture. Without this, it is unclear to what degree Definition 3.1 holds approximately, and thus how far the theoretical analysis extends to the tested networks. This is a gap that could at minimum be addressed qualitatively.

### Minor

- **The most important theoretical prediction — that kernel models cannot solve transitive equivalence — is never empirically tested in deep networks.** The theoretical result is proven, but the paper does not check whether trained ConvNets/ResNets/ViTs also fail on transitive equivalence and for the predicted reasons. Without this, it remains open whether feature-learning networks overcome this limitation (which would complicate the paper's central message). Even a single experiment showing failure would meaningfully strengthen the paper.

- **No quantitative comparison between Proposition 5.1 predictions and observed slopes.** Figures 5c/5d show that the slope increases with $S(1;2)$ and $p$, confirming the *direction* of the prediction. But the paper never plugs the empirically measured $S(1;2)$ (available from Fig. 5a) and training set size $p$ into Eq. (3) to predict a numerical slope, then compares to the observed one. Quantitative agreement would be far more compelling evidence that the kernel formula captures the operative mechanism rather than a generic "distribution mismatch" effect.

### Trivial
None.

---

## Nice-to-Haves

- A dedicated experiment testing transitive equivalence in the trained deep networks would complete the empirical picture and directly test the paper's most striking impossibility result.
- A figure overlaying the predicted slope from Eq. (3) against empirically observed slopes (using measured $S(1;2)$ values) in a single plot would convert a qualitative into a semi-quantitative test.
- Testing the full Definition 3.1 symmetry condition approximately (e.g., by verifying that kernel similarities between pairs with the same overlap count are approximately equal) would clarify the scope of the theoretical framework's empirical applicability.
- An experiment with frozen backbone + trained linear readout (genuine kernel regime) would cleanly test whether the kernel mechanism is causally responsible, as opposed to merely correlated.

---

## Removed Points

*These points are flagged for removal — treat with caution.*

- **Harsh critic, Point about Proposition B.2 being misapplied to trained finite networks.** This criticism confuses the paper's use of this proposition — it is presented as a theoretical analysis of random-weight infinite-depth networks (Section 5.1, Fig. 3), not as an empirical claim about trained networks. The distinction is clear in context.

- **Harsh critic's concern about n=10 permutations being "small."** Error bars in Fig. 5 are reported to be small (often not visible), and this sample size is standard for randomized category assignment experiments. No grounds to call this a flaw.

- **Harsh critic's concern about different training set sizes for different architectures (20k vs 40k).** The paper uses different architectures on different datasets (MNIST vs CIFAR-10), and training size differences are expected and do not confound cross-architecture comparisons directly. This is not a methodological problem.

- **Strength Finder's claim that the paper "provides a concrete explanation for inconsistent findings in disentangled representation learning."** While the paper mentions this as a possible implication, this is a speculative extrapolation beyond what the experiments directly demonstrate. The paper tests specific compositional tasks; inferring a general resolution to the disentanglement debate is too strong a claim for a strength.

---

## Novel Insights

The paper's clearest novel insight — going beyond cataloguing what both reviewers describe — is the identification of *conjunction-wise additivity* as a principled partition of compositional task space that arises naturally from the symmetry properties of compositionally structured representations under ℓ₂-norm minimization. This partition has an immediate and non-obvious corollary: transitive equivalence, a cornerstone of relational cognition, is not solvable by compositionally structured kernel models, while transitive ordering — its superficially similar sibling — is. This distinction, combined with the closed-form memorization-leak formula, provides a theoretical basis for the long-observed empirical inconsistency in the disentanglement literature: it is not simply that disentangled representations "help" or "don't help," but rather that the particular task's position in the conjunction-wise additive partition, combined with the specific representational geometry (quantified by $S(k;C)$), determines success or failure in a formally characterizable way.

---

## Suggestions

1. In Section 6, consistently distinguish between "the theory's predictions are consistent with deep network behavior" (empirically supported) and "the theory formally applies to these networks" (not established). The current phrasing sometimes conflates these.
2. Add a figure checking, for one trained architecture, that kernel similarities between representation pairs with equal overlap counts are approximately equal — a direct diagnostic for Definition 3.1.
3. Add at minimum one experiment testing transitive equivalence in a trained deep network to probe the key impossibility result empirically.
4. Report predicted vs. observed slope values from Eq. (3) for at least one architecture (using measured $S(1;2)$) to convert the qualitative validation to a semi-quantitative one.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Score | Notes |
|------|-----------|-------|
| `7VPTUWkiDQ.md` | 7.33 (Accept, oral) | Provably guarantees compositional generalization for object-centric representations via identifiability theory. Theory directly applies to the models tested. Stronger theory-to-experiment bridge than paper under review. |
| `hKMPz3wkPV.md` | 6.75 (Reject) | Formal theory of compositionality via algorithmic information theory. Despite 8,8 scores from two reviewers, rejected (likely due to gap in experimental validation and theoretical precision). Similar scope to paper under review. |
| `dggRphAcCj.md` | 6.33 (Withdrawn) | Compositional generalization via geometric constraints; less rigorous theory, weaker formal grounding than paper under review. |
| `kOtFuzoA93.md` | 4.00 (Withdrawn) | Kernel/NN theory without clear practical connection; weak contribution. |
| `fUz6Qefe5z.md` | 3.00 (Reject) | NTK derivative paper with serious technical gaps. Much weaker than paper under review. |
| `RIaIpdUCPb.md` | 3.00 (Withdrawn) | Brain-inspired geometry for CG — weak theoretical grounding, unsupported claims. |

**Comparison:** The paper under review is clearly stronger than the ≤4-scoring papers (kOtFuzoA93, fUz6Qefe5z, RIaIpdUCPb), which lack clean theoretical contributions. The paper's Theorem 4.2 and Proposition 5.1 are substantially sharper results. Compared to the high-scoring 7VPTUWkiDQ (oral, 7.33), the paper under review has a more significant theory-empirics gap: the identifiability paper's theory directly applies to the architectures tested, while here the kernel theory formally applies only to a regime different from the experimental one. The hKMPz3wkPV paper (6.75) is instructive — it too proposes a formal theory of compositionality with good theoretical novelty but got mixed scores due to empirical validation gaps.

The paper under review sits between hKMPz3wkPV and 7VPTUWkiDQ: it has cleaner, more predictive theoretical results than hKMPz3wkPV (an exact formula vs. a metric definition), and its empirical validation is more targeted and convincing, but it falls short of 7VPTUWkiDQ's fully provable end-to-end guarantee. The key major weakness — the kernel/feature-learning regime gap — is real but acknowledged and does not invalidate the empirical findings; it limits the claim strength. Given the genuine theoretical contributions, the creative empirical design (spatial distance manipulation, three-architecture validation), and the appropriate hedging in the Discussion, I place this in the **accept** range at the lower end.

**Final Score: 6.0 — Borderline Accept (Poster)**

The theoretical core is sound and makes contributions the field has lacked; the empirical validation is qualitatively convincing across three architectures; the acknowledged limitations are real but do not undermine the core claims if framed carefully.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>