Now let me look at the calibration papers mentioned by the Human Finder reviewer.Now I have enough information to write the final consolidated review. Let me synthesize.---

## Summary

NIMBA proposes a proximity-preserving greedy reordering strategy for converting point cloud patch centers into 1D sequences for Mamba-based state space models. The core idea is to ensure consecutive patch centers in the constructed sequence are spatially close (‖Cᵢ – Cᵢ₊₁‖ < r = 0.8), eliminating the sequence triplication of PointMamba and positional embeddings. The paper provides formal theoretical grounding via propositions on permutation invariance of attention vs. Mamba, and demonstrates consistent improvements over PointMamba and Point-MAE baselines on ModelNet40, ScanObjectNN, and ShapeNetPart.

---

## Strengths

- **Theoretical framing.** Propositions 1 and 2 formally establish that attention is permutation-equivariant while Mamba is not, cleanly motivating the need for principled sequence construction. This elevates the work beyond a purely engineering contribution.

- **Compelling positional embedding (PE) ablation.** Table 5 is the strongest result in the paper: removing PE degrades NIMBA by only 1.68% (89.80 → 88.12) compared to 4.11% for PointMamba and 6.53% for Point-MAE. This is a clear, reproducible, and informative result demonstrating that NIMBA's ordering encodes substantially more positional structure.

- **Genuine efficiency improvement.** By maintaining sequence length N instead of 3N, NIMBA achieves 14–17% reduction in training time at the same parameter count (Table 3), which is a practical advantage that is well-evidenced.

- **Reproducibility and transparency.** All experiments are conducted from scratch (not fine-tuned), mean ± std is reported over 3 runs, and learning-rate tuning is applied equally to all baselines, making comparisons fair.

- **Consistent benchmark improvements over PointMamba.** NIMBA outperforms PointMamba across all reported datasets and scales (up to ~1.8% on ScanObjectNN OBJ-ONLY, ~1% class-mIoU on ShapeNetPart), without positional embeddings.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing key SSM baselines in main experimental tables.** OctreeMamba, Point Cloud Mamba (PCM), and Mamba3D are acknowledged in Table 1 as competing SSM-based point-cloud methods but are entirely absent from Tables 2 and 4. This is a significant gap: the paper's main empirical claim (SOTA Mamba-based performance) cannot be evaluated without comparing against these methods. Notably, Table 5 already includes PointTramba—which clearly outperforms NIMBA with PE (92.42% vs. 89.80% on OBJ-BG)—yet this model is absent from the classification comparison in Table 2. The current comparison set is insufficient to support SOTA claims.

- **No ablation on the core hyperparameter r.** The threshold r = 0.8 is the single most important design choice in NIMBA—it controls which reorderings occur and therefore the entire sequence structure. Its selection is justified only by a heuristic argument ("40% of the distance from scene center to border"). There is no sensitivity analysis over r values (e.g., r ∈ {0.2, 0.4, 0.6, 0.8, 1.0, 1.5}), nor over the choice of initial axis (x, y, z). Without this, it is impossible to know whether the method is robust to this choice or whether r has been inadvertently tuned to the benchmark.

### Minor

- **"Almost permutation-invariant" claim is unsupported.** The abstract says NIMBA enables processing "in an almost permutation-invariant manner," but no formal definition or quantitative evidence is provided. NIMBA deterministically maps any input cloud to a single canonical ordering, which makes the model permutation-equivariant under that mapping—not permutation-invariant in any standard sense. No experiment measures sensitivity to different random permutations of the same input cloud. The claim should either be made precise and tested, or removed.

- **Overclaiming throughout.** (a) "Drastically improves robustness" (Introduction, point 3): Figure 3 shows approximately 1–2% differences between NIMBA and PointMamba across noise conditions, with no standard deviations. The word "drastically" is not supported. (b) "State-of-the-art results… surpassing Transformer-based models": on ModelNet40, Point-MAE has a higher mean (92.30 vs. 92.10); PointTramba achieves 92.42% on OBJ-BG; stronger transformers (PointGPT, PointNeXt) are not compared against. The framing should be revised to "competitive with" rather than "surpassing."

- **Algorithm under-specified.** Section 3.3.2 describes the proximity check as "look for a center along the sequence that is near enough to the starting center and place it next to it," without specifying whether this is a forward scan, a global search, a swap or insertion, or how ties are handled. No pseudocode is given. Additionally, the paper says r = 0 is "computationally expensive" (each center compared to all others) but also says it "will result in an ordering identical to the initial axis-wise order, as no centers will be considered close enough to trigger reordering." These two statements are contradictory: if no pair satisfies the threshold, no comparisons trigger swaps and there is no overhead.

- **PE analysis restricted to a single dataset and task.** Table 5 reports PE effects only on ScanObjectNN OBJ-BG classification. Whether the same conclusion holds on ModelNet40 or ShapeNetPart segmentation is unknown. A broader PE analysis would strengthen the generality of the claim.

- **Hydra/bidirectional experiment underdeveloped.** Table 6 reports only two numbers with no variance, no discussion of why replacing Mamba with Hydra hurts, and no alternative bidirectional strategy (e.g., bidirectional Mamba1 as used in PCM). Given that several competing methods use bidirectionality, this section needs substantially more depth.

### Trivial

- Internal inconsistency in describing behavior at r = 0: the paper simultaneously claims it is computationally expensive and that it produces no reordering—these two cannot both be true.

---

## Nice-to-Haves

- Sensitivity ablation over r and initial axis choice, to validate that the chosen defaults are principled rather than benchmark-tuned.
- Comparison with at least one explicit locality-preserving serialization baseline (e.g., Morton/Hilbert-curve ordering of FPS centers), to isolate whether NIMBA's greedy strategy is essential or whether any locality-preserving traversal yields similar gains.
- Inference throughput (FPS) and memory measurements to complement training-time comparisons.
- Visualizations of actual NIMBA orderings on real ModelNet/ScanObjectNN shapes (color-coded sequence indices), rather than the toy 8-point illustration in Figure 1.
- Scaling curve (accuracy vs. parameters) for NIMBA vs. PointMamba, to better characterize the acknowledged scaling limitation.
- A mechanistic explanation or analysis of *why* spatial proximity in the sequence benefits Mamba's recurrence specifically.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic: No comparison to space-filling curves / MST-TSP / learning-to-order schemes as explicit baselines.** While scientifically legitimate, demanding these comparisons goes beyond the paper's stated scope (the paper's contribution is framed relative to existing point-cloud SSM strategies, not the classical spatial serialization literature). Moved to Nice-to-Haves.

- **Harsh critic: PE analysis doesn't show r=0 description is "obviously incorrect."** The critic flags the r=0 description as problematic, but the paper's actual error is the inconsistency between "expensive" and "no reordering" — this inconsistency is real but minor (kept as Trivial). The critic's meta-claim that the whole statement is wrong is itself inaccurate.

- **Harsh critic: Claim about O(n²) complexity undermining efficiency.** The training time comparison (Table 3) shows NIMBA is 14–17% *faster* than PointMamba in practice, implying that any preprocessing overhead is empirically negligible. Absent evidence that the reordering cost is significant at the scales tested, this concern is not supported.

- **Harsh critic: Sensitivity of FPS/KNN to noise undermining ordering stability.** Robustness experiments (Figure 3) directly test performance under jitter, rotation, and dropout, providing empirical coverage of this concern. Criticizing that FPS/KNN is stochastic without evidence that this materially disrupts NIMBA is a strawman given the empirical results.

- **Human finder: Insufficient justification for why NIMBA outperforms Transformers.** This is the same question posed for the SU3lZ8jrRD spectral paper and similar SSM papers — it is a general "why does Mamba beat Transformers" question that is well beyond the scope of this specific paper and is not a weakness per se.

- **Human finder/Spark: Comparison with stronger Transformer baselines (PointNeXt, PointGPT, Point-BERT).** The paper does not claim to be SOTA over all methods on all benchmarks in an unconstrained sense; the relevant comparison is within the same training-from-scratch regime. These stronger models typically use pre-training or different training protocols, making direct comparison unfair to both sides.

---

## Novel Insights

The most genuinely insightful finding in this paper is the positional embedding dependency analysis (Table 5): the substantial difference in PE sensitivity across architectures (1.68% for NIMBA vs. 4.11–6.53% for PointMamba/Point-MAE/PointTramba) is a concrete empirical result suggesting that SSM-based point-cloud architectures whose ordering is spatially coherent can effectively internalize positional information into the sequence structure itself. This observation — that sequence construction quality and PE redundancy are inversely correlated — is a practically actionable insight that could guide the design of future SSM-based 3D architectures.

---

## Suggestions

1. Add OctreeMamba, PCM, Mamba3D, and PointTramba to Tables 2 and 4, or provide a clear justified reason for exclusion (e.g., different pre-training). Without these comparisons, SOTA claims are unsupported.
2. Run ablations over r ∈ {0.2, 0.4, 0.6, 0.8, 1.0, 1.5} and all three initial axes; report the sensitivity in a small table or figure.
3. Replace "almost permutation-invariant" with a precise statement, or add an experiment measuring output variance under random permutations of the same input cloud.
4. Revise overclaiming language: change "drastically improves robustness" to "consistently improves robustness," and qualify "state-of-the-art" with respect to the specific comparison class (same-scale, same-backbone, training from scratch).
5. Add a pseudocode block for the proximity reordering procedure, and correct or clarify the r = 0 description.
6. Extend the PE ablation to ModelNet40 and ShapeNetPart to show generalizability of the main insight.

---

## Score and Decision

**Calibration:**

- **XKQ2qzajbU (GlobalMamba)**: avg 5.0, withdrawn/reject. Marginal improvements, unclear source of gains, no ablation on key hyperparameters. NIMBA is somewhat stronger: the PE ablation is a more convincing result than anything in GlobalMamba, and the efficiency argument is cleaner.
- **SU3lZ8jrRD (Spectral Spatial Traversal in Point Clouds with Mamba)**: avg 4.75, withdrawn/reject. More theoretically sophisticated method but similar issues with missing baselines and overclaiming.
- **E1ML0nEReb (MEEPO)**: avg 6.2, rejected. Had substantially stronger empirical results (surpassing PTv3 on large-scale ScanNet/nuScenes) and more thorough ablations.

NIMBA sits closer to the GlobalMamba/SU3lZ8jrRD tier than to MEEPO. Its genuine contributions (PE ablation insight, efficiency, clean theory) place it marginally above GlobalMamba, but the missing key SSM baselines and absent ablation on the central hyperparameter r are major gaps that prevent acceptance at a top venue. Accuracy improvements are modest and sometimes within variance. The overclaiming further weakens the submission.

**Axes assessment:**
- *Originality*: Low-to-moderate. The greedy proximity ordering is intuitive but ad hoc; the PE analysis is novel and valuable.
- *Importance of research question*: High — how to serialize unordered 3D data for SSMs is genuinely important.
- *Claims well supported*: Partially — PE ablation is well-supported; SOTA claims and robustness claims are not.
- *Soundness of experiments*: Moderate — fair reproduction, but missing critical baselines and ablations.
- *Clarity of writing*: Good — well-structured and readable.
- *Value to research community*: Low-to-moderate — the PE insight is valuable, but the core method needs more rigorous validation.

**Final Score: 4.5**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>