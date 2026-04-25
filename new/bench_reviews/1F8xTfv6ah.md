Now let me search for calibration papers.Now I have enough context to write the final review. Let me verify one last claim — the multi-layer feature usage versus NAC comparison numbers from Table 1.

From the paper (lines 154, 188-189, 203-204):
- Section 3.1 explicitly states multi-layer feature integration, citing NAC (Liu et al., 2024a)
- CIFAR-10: KAN 94.12 ± 0.59 vs NAC 93.37 ± 0.64
- CIFAR-100: KAN 83.44 ± 1.99 vs NAC 83.36 ± 0.84
- ImageNet-200: KAN 71.46 vs NAC 60.05
- CIFAR-100 Near OOD: KAN 77.42 ± 0.35 vs RMDS 83.40 ± 0.46 — RMDS wins substantially

These facts are confirmed. Now I'll write the consolidated final review.

---

## Summary
This paper proposes a novel OOD detection method that exploits the local neuroplasticity of Kolmogorov-Arnold Networks (KANs): by comparing the activation patterns of a trained KAN against an untrained copy, regions adapted to in-distribution (InD) data are distinguished from unadapted regions. The core detection score (Eq. 5) measures the overlap between spline coefficient changes and the test sample's activated basis functions. The method is validated across seven benchmarks in image classification and tabular medical data domains.

---

## Strengths

- **Clear mechanistic grounding (Eq. 3–5):** The KAN OOD score is derived from first principles via B-spline basis function properties. The decomposition in Eq. 5 — magnitude of coefficient change times the activated basis function — provides genuine interpretability beyond black-box scoring functions. Figures 2 and 3 build intuition effectively for 1D and 2D cases.

- **Exceptional dataset-size robustness (Table 6):** At 0.1% of CIFAR-10 training data, the KAN detector maintains 93.21% AUROC while KNN collapses to 8.15% and VIM drops to 76.38%. This is a genuine, practically important property that the paper demonstrates clearly with a controlled ablation.

- **Cross-domain generality:** The method is evaluated on both image benchmarks (ResNet backbone) and tabular medical data (FT-Transformer backbone), spanning 7 benchmarks. Most OOD papers stay in the vision domain; this breadth is a real plus.

- **Meaningful ablation on design choices (Table 7):** Systematic evaluation of partition count (P=1 to 30) and grid size (G=5 to 200) shows the method saturates at P=10 and G=100. The histogram baseline experiment (85.29% vs 94.12%) isolates the contribution of spline smoothing within the same framework.

---

## Weaknesses

### Fatal
None.

### Major

- **Multi-layer feature advantage is uncontrolled, inflating comparisons against most baselines.** Section 3.1 explicitly states: *"We leverage information from multiple latent layers of the pre-trained backbone. As demonstrated by Liu et al. (2024a), this multi-layer integration enriches the feature representation."* The baselines in Tables 1–2 (KNN, VIM, RMDS, GEN, ASH, ReAct, etc.) are evaluated in the standard OpenOOD configuration that uses only the penultimate layer. Only NAC (Liu et al., 2024a) also uses multi-layer features, because it is the method from which this design choice is borrowed. The large margins over KNN (94.12 vs 92.19 on CIFAR-10) and over RMDS, VIM, etc., cannot be cleanly attributed to local neuroplasticity — they may reflect multi-layer feature richness. The only fair architecture-controlled comparison is KAN vs. NAC: on CIFAR-10, KAN is 94.12 ± 0.59 vs. NAC 93.37 ± 0.64 (statistically tied per the paper's own Welch test criterion); on CIFAR-100, KAN is 83.44 ± 1.99 vs. NAC 83.36 ± 0.84 (essentially tied). A simple ablation running single-layer KAN and multi-layer KNN/VIM/RMDS side by side is absent, making the primary performance claims difficult to attribute. The ImageNet results (KAN 71.46 vs NAC 60.05) show a larger gap that may reflect a genuine KAN advantage, but this too cannot be confirmed without controlling for feature configuration. This is a key methodological gap.

- **Near-OOD CIFAR-100 underperforms RMDS significantly; "superior performance on both benchmarks" overclaims.** Table 1 shows that on CIFAR-100 Near OOD, KAN achieves Avg Near = 77.42 ± 0.35 vs. RMDS's 83.40 ± 0.46 — a ~6 AUROC point gap in favor of RMDS. The KAN's overall CIFAR-100 advantage (83.44 vs 82.00) is driven entirely by far-OOD datasets (MNIST, SVHN, Textures). Since near-OOD is the harder and more practically relevant sub-task (semantic novelty detection as opposed to clearly different-modality rejection), this gap is meaningful. The abstract's claim of "superior performance" and Section 3.2's claim that "the KAN detector outperforms all previous methods on both benchmarks" are therefore overstated.

### Minor

- **P=1 below-chance behavior (46.08% AUROC) is unexplained.** Table 7 shows that without partitioning, the detector performs worse than random (50%). The paper proposes partitioning as the remedy but does not explain why the unpartitioned score is sub-random rather than just sub-optimal. A below-50% AUROC means that InD samples score *lower* than OOD samples on average, implying the method inverts its intended direction in this configuration. This inversion — its cause and implications — deserves brief discussion. Understanding it would also help clarify whether the gains from partitioning come from resolving this inversion or from improved density estimation.

- **Histogram baseline configuration underspecified.** Section 3.3 states the histogram baseline achieves 85.29% vs. the KAN's 94.12%, concluding spline smoothing provides ~9% AUROC gain. However, it is not stated whether the histogram baseline also uses class-conditional partitioning (P=10). If the histogram uses P=1 (no partitioning) and the KAN uses P=10, the comparison entangles partitioning and spline smoothing — two different contributions. Clarifying the baseline setup is needed to correctly attribute the ~9% gain.

- **Places365 mentioned in Section 3.1 but absent from Table 1 columns.** Section 3.1 lists Places365 as a far-OOD dataset for the CIFAR-10 benchmark, but Table 1 shows only MNIST, SVHN, and Textures as far-OOD columns. Whether Places365 is included in the "Avg Far" aggregation or omitted is not stated.

### Trivial
None beyond formatting artifacts from the parser.

---

## Nice-to-Haves

- **Ablation: single-layer vs. multi-layer features for the KAN detector.** Running the KAN with only last-layer features would directly quantify whether the multi-layer integration or the KAN architecture is the primary driver of performance. This would substantially strengthen the paper's core attribution.

- **Fair multi-layer comparison for KNN/RMDS/VIM.** Giving the same multi-layer feature concatenation to the strongest single-layer baselines would clarify how much of the performance advantage over those methods is architectural vs. feature engineering.

- **Explanation of why P=1 is sub-random.** Even a brief geometric argument (e.g., mixed-class marginals are too broad and encompass OOD regions) would satisfy this.

- **Class-conditional density baseline.** Since the partitioned KAN is functionally close to per-class density estimation with splines, comparing against per-class KDE or per-class GMM on the same multi-layer features would situate the spline contribution more clearly within the density-estimation literature.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic's structural claim that "local neuroplasticity" is entirely misleading:** The critic argues the method "reduces to class-conditional B-spline density estimation" and is not meaningfully different from per-class KDE or GMM. While the resemblance to class-conditional density estimation is real and worth noting, the critic overstates this as invalidating the paper. The KAN mechanism — specifically the spline smoothing advantage over the histogram baseline (~9% AUROC, Section 3.3) and the dataset-size robustness (Table 6) — constitutes a real contribution beyond naming. The framing as density estimation is a nice-to-have clarification, not a fatal flaw.

- **Harsh Critic's claim that the Eq. 5 conflation of coefficient magnitude and activation location is a meaningful theoretical error:** The critic notes that |c_trained − c_untrained| is a weight magnitude, not an exact positional indicator. This is a reasonable precision note but is a known property of B-splines (compact support, not indicator functions). The paper's interpretation is an approximation that is standard in the KAN literature and does not undermine the method.

- **Harsh Critic's claim about "histogram normalization" being an uncontrolled advantage:** The paper applies histogram normalization to handle skewed latent feature distributions. There is no evidence it is applied asymmetrically to only the KAN. This is speculation.

- **Strength Finder's claim "State-of-the-art across all seven benchmarks":** Removed as a standalone strength because on CIFAR-100 near-OOD, RMDS outperforms the KAN (83.40 vs 77.42). The overall average wins are real but the phrasing is imprecise.

- **Harsh Critic: Training-size robustness attributed to B-spline data efficiency rather than KAN:** The critic suggests KDE-per-class would show similar robustness. This may be true but is speculative — no experiment demonstrates it — and doesn't negate the observed advantage of the KAN detector.

- **Harsh Critic: tabular medical benchmark results near-chance:** Table 4 shows results near 50%, which is fair to note, but these near-OOD age splits are known to be intrinsically hard for all methods, not a failure specific to KAN. Removed as a targeted weakness of this paper.

---

## Novel Insights

The most genuinely novel observation buried in the paper is that the KAN detector's performance is *inverted* (below random) when P=1 but jumps dramatically to state-of-the-art performance with class-conditional partitioning. This is not just an ablation result — it reveals something fundamental about what the method is actually doing: the local plasticity signal is too ambiguous across a mixed-class distribution but becomes highly discriminative when restricted to within-class marginals. This suggests that local neuroplasticity is a *within-class* phenomenon more than a global one, and future work could leverage this insight to design even more structured partitioning (hierarchical or prototype-based) that goes beyond flat class partitions while retaining the spline advantage over nearest-neighbor methods.

---

## Suggestions

1. Add an ablation comparing (a) KAN with last-layer only features vs. (b) KAN with multi-layer features, and (c) KNN/RMDS with multi-layer features. This single experiment would resolve the main methodological concern.
2. Clarify the histogram baseline setup in Section 3.3 (does it use P=10 class-conditional partitioning or not?).
3. Provide a brief explanation for why P=1 gives sub-random (not merely sub-optimal) performance.
4. Qualify the "outperforms all previous methods on both benchmarks" claim to refer specifically to the overall average AUROC, explicitly acknowledging that RMDS dominates on near-OOD CIFAR-100.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Human Score | Relevance |
|---|---|---|---|
| NECO (neural collapse OOD, post-hoc) | `9ROuKblmi7.md` | 5.75 (Accepted Poster) | High — novel post-hoc OOD with mechanistic grounding; stronger theoretical backing than KAN OOD paper |
| SCALE (activation shaping OOD analysis) | `RDSTjtnqCg.md` | 6.25 (Accepted Poster) | Medium — solid analysis + new method; cleaner causal attribution than KAN paper |
| HAct (activation histogram OOD) | `Oo5spZRpH6.md` | 3.67 (Rejected) | Medium-high — structurally similar (activation-based post-hoc OOD), but weaker experiments and missing baselines |
| Shallow features for OOD | `UTnq6hJJYa.md` | 5.33 (Rejected) | High — multi-layer features for OOD with confounded comparisons, overclaimed results |
| SISOM (unified AL+OOD) | `rcKzU0Vns0.md` | 2.50 (Rejected) | Low — weak experiments, missing baselines |
| KAN GNN | `udfjje2xXb.md` | 3.40 (Rejected) | Medium — KAN applied to a new domain, minimal contribution |

**Positioning:** This paper is substantively stronger than HAct (3.67) — it has 7 benchmarks, cross-domain experiments, clear mathematical formulation, and a genuine robustness advantage. It is comparable to or slightly below UTnq6hJJYa (5.33): both apply a multi-layer feature approach in OOD detection with some confounding, both have overclaimed results, but this paper is broader and has stronger experiments. It falls short of NECO (5.75) and SCALE (6.25) because those papers resolve attribution questions more cleanly. The multi-layer confound is a real issue that prevents clear attribution of the headline gains, but the dataset-size robustness (Table 6) and the histogram comparison (Section 3.3) are genuine contributions that are not confounded. I settle on **5.0** — marginally below acceptance, acknowledging genuine novelty and broad evaluation but penalizing for the unresolved multi-layer attribution gap and overclaimed overall performance.

**Originality:** Good — applying KAN local plasticity to OOD is novel and well-motivated.  
**Importance:** Moderate — OOD detection is important; dataset-size robustness is practically valuable.  
**Claims support:** Weak — headline performance claims are confounded by multi-layer features; the fair comparison (vs. NAC) shows marginal improvement on CIFAR.  
**Soundness:** Moderate — core mechanism is sound; multi-layer confound is a methodological gap.  
**Clarity:** Good — well-written, with clear figures and ablations.  
**Community value:** Moderate — introduces a new perspective (KAN for OOD) with practical dataset-size robustness, but the primary claim needs better experimental support.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>