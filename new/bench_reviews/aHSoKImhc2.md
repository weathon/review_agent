## Summary

NIMBA proposes a proximity-preserving 1D reordering strategy for point clouds that eliminates the 3× sequence replication and positional embeddings used in prior Mamba-based methods such as PointMamba. The paper shows that NIMBA can operate without positional embeddings while maintaining or improving accuracy on ModelNet40, ScanObjectNN, and ShapeNetPart, and reduces training time by 14–17%.

## Strengths

- **Identification of real inefficiencies in prior work.** The paper correctly notes that prior Mamba point-cloud methods replicate sequences along multiple axes and use positional embeddings redundantly, and it proposes a concrete alternative (Table 1).
- **Compelling positional-embedding ablation.** Table 5 shows that removing positional embeddings hurts NIMBA far less (–1.68 points on ScanObjectNN OBJ-BG) than it hurts PointMamba (–4.11) or Point-MAE (–6.53), and NIMBA without PE still outperforms PointMamba with PE. This is the strongest empirical evidence in the paper.
- **Efficiency improvement.** Avoiding 3× sequence replication yields a measurable training-time reduction (Table 3) and shorter sequences, which is practically useful.

## Weaknesses

### Fatal

- **Suspicious duplicate statistics in Table 2 undermine experimental trust.** NIMBA reports *exactly* identical mean and standard deviation (92.10 ± 0.14) on ModelNet40 for both the 12.3M and 23.86M parameter configurations. Obtaining the same mean *and* variance across independent training runs at two different scales is statistically implausible. The reproduced PointMamba baseline also shows identical means (92.08) across scales (with different standard deviations). This pattern strongly suggests a reporting error or duplication of results and severely erodes confidence in the experimental record. Without a corrected and verifiable set of experiments, the empirical claims cannot be relied upon.

### Major

- **Unsupported “state-of-the-art” claim.** The abstract claims “state-of-the-art results in ModelNet40,” yet on ModelNet40 NIMBA (92.10%) underperforms the reproduced Point-MAE baseline (92.30%) and omits comparisons to stronger methods discussed in the related work (e.g., Point-BERT, PointGPT). The headline claim is therefore inaccurate or at best unsubstantiated.
- **Core reordering algorithm is under-specified and contradicts robustness framing.** Section 3.3.2 describes the proximity check with imprecise language (“we look for a center along the sequence that is near enough to the starting center and place it next to it”), without pseudocode, tie-breaking rules, search procedure, or complexity analysis. Because the algorithm is initialized by sorting along the y-axis, the sequence construction is explicitly tied to the global coordinate frame and is *not* robust to rotations by construction. Any observed robustness (Figure 3) is a learned property of data augmentation, not an algorithmic invariant, which contradicts the “principled” and “robust” framing in the title and abstract.
- **Missing ablations isolate the proximity heuristic.** The paper attributes accuracy gains and the ability to remove positional embeddings to the proposed proximity check, but there is no ablation comparing NIMBA against (1) simple y-axis ordering at sequence length *N* without 3× replication, or (2) random ordering, with and without positional embeddings. Without these controls, it is impossible to determine whether the gains come from the proximity heuristic itself or merely from avoiding sequence duplication and multiple scans.

### Minor

- **Modest speedups are unexplained.** A 3× reduction in sequence length yields only ~14–17% end-to-end training-time reduction (Table 3). The paper does not provide a layer-wise breakdown or discuss why the reduction is smaller than the sequence-length reduction would suggest.
- **Trivial theoretical propositions.** Propositions 1 and 2 in Section 3.3 restate well-known properties of attention and RNNs; they add little novel insight, though they do no harm.

### Trivial

- None.

## Nice-to-Haves

- Replace the y-axis initialization with a geometrically motivated anchor (e.g., centroid-nearest point) to make the sequence construction itself coordinate-frame-independent rather than relying solely on data augmentation for rotation robustness.
- Visualizations of the constructed 1D traversal paths on diverse shapes to verify spatial coherence and reveal failure modes.
- Benchmark the latency and FLOPs of the reordering step itself, as its cost is omitted from efficiency comparisons.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Criticism that Table 5 “mixes fundamentally different architectures.”** The paper explicitly designs Table 5 as a cross-architecture ablation on the effect of removing positional embeddings; comparing Transformers, Mamba, and Hybrid models is the intended purpose of the table, not a confound.
- **Criticism that PointTramba outperforms NIMBA.** The paper itself reports PointTramba’s higher score (92.42% vs. 89.80%) in Table 5 and discusses it in Section 4.3.1, so this is not a hidden weakness.
- **Criticism about the y-axis sort being order-dependent.** While the harsh critic correctly identifies the y-axis initialization as frame-dependent, the paper does acknowledge that Mamba is order-sensitive (Sec. 3.3) and does not claim full permutation invariance for the algorithm itself, only an “almost permutation-invariant manner” in the abstract. The main issue is the mismatch between this framing and the rotation-robustness claims, not the existence of the y-axis sort per se.
- **Formatting and grammar nitpicks.** These are parser artifacts, not author errors.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

1. **Correct and re-run all experiments.** Table 2 must report independent statistics for each model scale. If the identical numbers are a copy-paste error, the authors should rerun and report distinct means and standard deviations; if they are not an error, the authors must explain why two independent model sizes produced identical results.
2. **Add the missing ablations.** Include a simple y-axis-ordered baseline at length *N* (no replication) and a random-ordering baseline, both with and without positional embeddings, to isolate the contribution of the proximity heuristic.
3. **Provide a formal algorithmic specification.** Include pseudocode and complexity analysis for the reordering procedure, and clarify tie-breaking and search rules.
4. **Retract or qualify the SOTA claim.** Either add comparisons to stronger recent methods or reframe the claim as improvements over specific baselines rather than state-of-the-art.

## Score and Decision

**Calibration comparison:**
- **High anchor:** *Differentiable Euler Characteristic Transforms* (avg 6.25, Accept poster) — solid method, competitive results, well-written, minor missing ablations. NIMBA is below this because it has suspicious duplicate statistics and an unsupported SOTA claim.
- **Medium anchor:** *PGV* (avg 5.50, Reject) — decent idea but limited in scope and experimental validation. NIMBA has a clearer core idea but more serious experimental integrity issues.
- **Low anchor:** *CFBD* (avg 4.00, Reject) — overclaimed SOTA and insufficient comparisons, plus methodological concerns. NIMBA shares the overclaiming but adds statistically implausible duplicate results, which is worse. Another low anchor, *Is Memorization Actually Necessary* (avg 3.75, Reject), had methodological errors and data leakage but an interesting observation; NIMBA’s duplicate results are comparably damaging to trust.

NIMBA presents an interesting direction and a compelling positional-embedding ablation, but the suspicious duplicate statistics in Table 2 are a severe red flag that undermines the entire experimental record. Coupled with an unsubstantiated SOTA claim, an under-specified core algorithm, and missing ablations, the paper is not currently acceptable.

**Score: 3.5**

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>