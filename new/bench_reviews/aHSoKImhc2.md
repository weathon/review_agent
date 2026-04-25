Now let me do calibration searches to properly score the paper.Now I have enough information to write the final review. Let me synthesize everything.

---

## Summary
NIMBA introduces a proximity-preserving 3D-to-1D reordering strategy for applying Mamba-based state space models to point clouds. By ensuring consecutive tokens in the sequence are spatially adjacent, the method eliminates the need for positional embeddings and sequence replication (used in prior work to achieve 3N-length sequences), achieving shorter sequences and improved robustness. The central empirical finding is in Table 5: NIMBA's accuracy drops only 1.68% without positional embeddings versus 4.11–6.53% for competing methods, supporting the claim that spatial ordering alone encodes sufficient positional structure.

---

## Strengths

- **PE ablation (Table 5) is the paper's strongest and most novel evidence**: NIMBA suffers only a 1.68% accuracy drop without positional embeddings, vs. 4.11% (PointMamba), 5.96% (PointTramba), and 6.53% (Point-MAE). This directly validates the core claim that a proximity-preserving ordering can replace the redundant PE mechanism used in all prior Mamba-based point cloud methods.

- **Consistent accuracy gains over PointMamba across all benchmarks (Table 2)**: NIMBA exceeds PointMamba at both 12.3M and 23.86M parameter scales on all four classification benchmarks and on ShapeNetPart segmentation (Table 4), with statistically reported mean ± std over three runs — a higher standard than most prior work in this area.

- **Robustness experiments (Figure 3) are well-structured**: Testing under rotation, jittering, random input dropout, and combinations thereof, with train-only, test-only, and train+test variants is systematic. NIMBA consistently outperforms PointMamba, with the rotation result particularly principled — rotation is a distance-preserving transform and thus should not disrupt the ordering's proximity property.

- **Efficient sequence compression (Table 3)**: Eliminating sequence replication reduces training time ~14% on ModelNet and ~17% on ScanObjectNN. While modest in absolute speedup (less than the theoretical 3× from sequence length reduction), this is a concrete and reproducible gain.

- **Clean theoretical framing of the attention–Mamba distinction**: Propositions 1–2 and the $\Phi_\text{SDPA}$ vs. $\Phi_\text{S6}$ representation (Eqs. 3 and 7) provide a clean pedagogical motivation for why ordering matters for Mamba but not attention. This framing is useful to the community even if the propositions themselves are not novel.

- **Reproducible experimental setup**: Grid searches for learning rates across all reproduced methods, three independent runs per method, and a unified codebase (PointMamba/Point-MAE framework) make comparisons credible.

---

## Weaknesses

### Fatal
None.

### Major

- **PointTramba is absent from the main comparison table (Table 2).** PointTramba achieves 92.42 ± 0.48% on OBJ-BG (Table 5), compared to NIMBA's 89.80 ± 0.36% — a ~2.6 percentage point gap. The paper correctly includes it in Table 5 and acknowledges "PointTramba, which, despite outperforming NIMBA under normal conditions, relies heavily on PE." However, excluding PointTramba from Table 2 means the main accuracy comparison table does not contain the strongest competitor. The abstract's claim of "surpassing Transformer-based models in both accuracy and efficiency" is most naturally read to include hybrid models like PointTramba. The paper's actual strongest claim — that NIMBA without PE outperforms all methods without PE (Table 5) — is more defensible than the framing in the abstract, and all claims should be reconciled with PointTramba's inclusion in the main table.

- **The "almost permutation-invariant" claim is asserted but never defined or substantiated.** The abstract states NIMBA allows Mamba "to be applied effectively in an almost permutation-invariant manner." Proposition 2 establishes that Mamba is not permutation-invariant; the paper provides no theorem, bound, or even informal argument for how sensitive NIMBA's output is to permutations of the input relative to the baseline strategies. The reordering reduces sensitivity in practice, but the phrase "almost permutation-invariant" without any formal definition weakens the "principled" framing of the paper. Either a formal argument or a retraction/qualification of this phrasing is needed.

### Minor

- **The reordering algorithm is underspecified.** Section 3.3.2 describes the algorithm in two informal prose bullets. Specifically, when the distance to the next center exceeds threshold *r*, the algorithm looks "for a center along the sequence that is near enough" — but neither the search procedure (linear scan? nearest-neighbor?) nor the tie-breaking rule when multiple candidates qualify is specified. Without pseudocode, two implementations could behave differently on edge cases, which affects reproducibility.

- **No ablation over the threshold *r*.** The method has one core hyperparameter. The paper discusses the qualitative behavior at the extremes (*r* = 0 and *r* ≥ 2√3) but provides no quantitative sweep over intermediate values. This makes it impossible to assess how sensitive results are to this choice, or whether *r* = 0.8 is optimal or merely sufficient.

- **ModelNet accuracy is identical across parameter scales.** Both 12.3M and 23.86M NIMBA variants report exactly 92.10 ± 0.14% on ModelNet40. Identical means and standard deviations across independent configurations warrants at least a brief explanation (saturation of the dataset is plausible but should be stated).

- **Training from scratch setting is not qualified in the abstract.** The dominant paradigm in point cloud classification uses ShapeNet pre-training followed by ScanObjectNN fine-tuning, where absolute numbers are substantially higher. The paper explicitly states (Section 4) it trains from scratch, but the abstract's "state-of-the-art" claim is not qualified to the from-scratch sub-setting. This is a framing issue, not a methodological flaw.

- **The claimed rotation improvement (Section 4.3.2) is not statistically substantiated.** The paper states "we even observe an improvement in performance" under rotation, but Figure 3 shows the difference between Baseline (~89.5%) and Rotation Train & Test (~89.5%) is within digitization noise, and no confidence intervals are reported for these robustness experiments.

- **Efficiency gain is unexpectedly modest.** A 3× sequence length reduction should approach 3× speedup in Mamba layers (linear complexity). The actual 14–17% gain implies that the bottleneck lies in shared components (FPS, kNN, patch embedding). This is worth explaining because it clarifies the practical scope of the efficiency advantage.

### Trivial

- The Hydra experiment (Section 4.3.3) shows a performance drop when replacing Mamba with Mamba2/Hydra but attributes it vaguely to "optimization challenges." Since this is a limitation of the work, a more honest discussion than "we recommend future research" would strengthen the paper.

---

## Nice-to-Haves

- A visualization of the constructed sequences for representative ScanObjectNN objects would help readers verify that the proximity-preserving property holds in practice and identify whether pathological orderings arise for specific object geometries.
- A pre-training experiment (e.g., ShapeNet pre-training + ScanObjectNN fine-tuning) would let the paper make a broader "state-of-the-art" claim and connect to the dominant evaluation paradigm in the field.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic — "hand-picked comparison set" / result integrity concern**: The paper explicitly acknowledges PointTramba outperforms NIMBA in Section 4.3.1 with direct quotation. This is not a case of hidden data; it is a known limitation openly discussed. The framing issue (PointTramba's absence from Table 2) is kept as a Major weakness for different reasons, but the stronger charge of result integrity manipulation is removed.
- **Strength Finder — "principled threshold r=0.8 justification"**: The paper's justification is that 0.8 ≈ 40% of the scene radius in a unit cube. This is heuristic reasoning, not a principled derivation. Kept only as a minor weak point (no dedicated ablation), not retained as a strength.
- **Strength Finder — "Propositions 1 and 2 as genuine theoretical contribution"**: Permutation invariance of attention is standard background material (Vaswani et al., 2017). The propositions are clearly labeled as "easy to prove" by the authors themselves. Not a novel contribution; moved to removed.

---

## Novel Insights

The most genuinely insightful observation — supported by Table 5 — is that **proximity-preserving ordering is a sufficient substitute for positional embeddings in Mamba-based point cloud models**. Prior work treated PE as a necessary component transplanted from transformers; NIMBA shows empirically that PE sensitivity in models like PointMamba is partly a symptom of poor sequence construction, not an inherent requirement of the architecture. This insight reframes PE removal from an engineering simplification to a diagnostic signal about ordering quality, and could guide future design of SSM-based 3D architectures.

---

## Suggestions

1. **Add PointTramba (and OctreeMamba, Mamba3D if reproducible) to Table 2**, alongside honest framing that NIMBA's primary advantage is PE-free operation, not raw accuracy.
2. **Qualify the "state-of-the-art" claim** throughout to "within the from-scratch training setting."
3. **Provide pseudocode** for the NIMBA reordering algorithm with tie-breaking rules and complexity analysis.
4. **Provide a quantitative sweep over *r***, even if brief, to justify the choice of 0.8 and show robustness to this hyperparameter.
5. **Either formalize or remove the "almost permutation-invariant" phrase.** A simple empirical measure (e.g., variance of output over random permutations of the same input) would suffice.
6. **Add confidence intervals or error bars to the robustness bar chart** (Figure 3) to support claims of performance improvement vs. within-noise differences.

---

## Score and Decision

**Calibration anchors consulted:**

| Path | Avg score | Comparison to NIMBA |
|------|-----------|---------------------|
| SU3lZ8jrRD | 4.75 (Rejected) | Most topically similar: Mamba ordering for point clouds, similar claims of improvement over PointMamba. Weaker experimental setup, more complex/underspecified method; comparable issues with overclaiming. |
| XKQ2qzajbU (GlobalMamba) | 5.00 (Rejected) | Mamba serialization for vision, similar framing, 5/5/5/5/5 uniform scores — a "technically fine but limited contribution" paper. |
| imZcqOrbig | 6.00 (Accepted) | 3D point cloud pre-training, stronger positive contribution with clearer novelty, clean ablations, more complete baselines. |
| 62Ff8LDAJZ | 6.80 (Accepted) | 3D point cloud generation, strong theoretical + empirical package with no major baseline gaps; clearly above NIMBA in completeness. |
| vOfDGYGVyj | 2.50 (Rejected) | Mamba with controllability theory; fundamentally flawed method, far weaker than NIMBA. |
| Sw10tbj0gM | 3.50 (Rejected) | Mamba for time series imputation, incremental and poorly validated; weaker than NIMBA but same general issue of limited contribution. |

**Assessment against anchors**: NIMBA sits between SU3lZ8jrRD (4.75) and XKQ2qzajbU (5.00) as the most direct comparators. NIMBA has a marginally cleaner contribution (single ordering strategy, convincing PE ablation) and more rigorous experimental practice (3-run variance). However, the missing PointTramba in the main table, the undefined "almost permutation-invariant" claim, and the underspecified algorithm mean it does not clearly surpass the borderline tier. The paper's best result (Table 5 PE ablation) is genuinely compelling and distinguishes it from the 3.5 band. Score: **5.0**.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>