Now I have a thorough understanding of the paper and calibration anchors. Let me compile my final review.

## Summary

NIMBA proposes a point cloud reordering strategy for Mamba-based models that preserves 3D spatial proximity in the 1D sequence, enabling removal of positional embeddings and reducing sequence length from 3N to N. The core idea is a greedy nearest-neighbor chain heuristic that rearranges centers after an initial y-axis sort, ensuring adjacent entries in the sequence are close in 3D space. Experiments on ModelNet40, ScanObjectNN, and ShapeNetPart show consistent improvements over PointMamba, with the strongest finding being that NIMBA's ordering reduces positional embedding dependence from a 4.11% accuracy gap to just 1.68%.

## Strengths

- **The PE ablation (Table 5) is the paper's most convincing contribution.** Removing positional embeddings causes only a 1.68% accuracy drop for NIMBA versus 4.11% for PointMamba, 5.96% for PointTramba, and 6.53% for Point-MAE. This directly demonstrates that NIMBA's ordering encodes spatial structure into the sequence itself, reducing reliance on explicit positional information — a property no prior Mamba-based point cloud method achieves (Section 4.3.1, Table 5).

- **Consistent accuracy improvements across all benchmarks.** NIMBA outperforms PointMamba on every dataset at both 12.3M and 23.86M parameter scales, with notably large gains on the harder ScanObjectNN variants (+1.7% on OBJ-BG, +1.2% on PB-T50-RS at 23.86M) and +1.0% Cls. mIoU on ShapeNetPart (Tables 2, 4).

- **Concrete efficiency gains from shorter sequences.** By avoiding 3× sequence replication, NIMBA achieves ~14% and ~17% training time reductions on ModelNet and ScanObjectNN respectively, at identical parameter counts of 17.4M (Table 3).

- **The formal framing via Propositions 1 and 2** correctly identifies why ordering matters for SSMs (lower-triangular, order-dependent Φ_S6) but not for attention (isotropic, permutation-invariant Φ_SDPA), even though the propositions themselves are straightforward (Section 3.3, Eqs. 3 and 7).

## Weaknesses

### Fatal
None.

### Major

- **The "almost permutation-invariant" claim in the abstract is unsupported and misleading.** The abstract states NIMBA processes point clouds "in an almost permutation-invariant manner," but the paper provides neither a formal nor empirical definition of "almost permutation-invariant." Proposition 2 proves Mamba is *not* permutation-invariant, and NIMBA's own pipeline (FPS → y-axis sort → proximity correction) is deterministic given coordinates but coordinate-system-dependent (rotation changes the y-axis sort). No experiment measures output variance under input permutation or rotation of the initial point order. The paper demonstrates reduced PE dependence and improved rotation robustness, which are real findings, but these are distinct from "almost permutation-invariant" processing.

- **Limited experimental comparisons weaken the claimed contribution.** Table 1 lists seven Mamba-based/hybrid architectures (PointMamba, Point Cloud Mamba, OctreeMamba, Point Tramba, PointABM) plus two transformer baselines (PCT, Point-MAE), but experiments only compare against PointMamba (and PointTramba briefly in the PE ablation). OctreeMamba in particular uses z-order curve-based serialization for exactly the locality-preservation purpose NIMBA addresses — this is the most natural baseline comparison and its absence is conspicuous. The "state-of-the-art results" claim in the abstract is further undermined by ModelNet40 accuracy (92.10%) being below Point-MAE's 92.30% in the paper's own Table 2, even within the limited comparison set.

### Minor

- **The reordering algorithm description in Section 3.3.2 is ambiguous.** The phrase "look for a center along the sequence that is near enough to the starting center and place it next to it" does not specify the search direction, whether this is a swap or insertion, or what happens to the displaced element. This raises reproducibility concerns and makes it hard to assess the claim of "no data replication."

- **The threshold r = 0.8 is justified heuristically with no sensitivity analysis.** The rationale (40% of the distance from center to border of a [−1,1]³ cube) is dataset-normalization-specific, undermining the "principled" framing. A sweep over r values on different normalization conventions would establish whether this is principled or well-tuned.

- **Rotation robustness claims are partially overstated.** Section 4.3.2 states NIMBA is robust to rotations "because the reordering preserves pairwise distances," but the ordering itself starts from y-axis sort, which is rotation-dependent, and FPS is coordinate-dependent. The improved rotation robustness likely stems from the Euclidean-distance-based proximity correction partially mitigating axis-alignment effects, rather than true rotation-equivariance.

- **The title claims "no need for positional embeddings" but removing PE still costs 1.68% accuracy** (Table 5, Section 4.3.1). The gap is substantially smaller than other methods, which is the real contribution, but "no need" is an overstatement.

### Trivial
None.

## Nice-to-Haves

- A permutation sensitivity test: randomizing initial point order before FPS+kNN+reordering and measuring output variance would directly test the "almost permutation-invariant" framing.
- Comparison with OctreeMamba and Point Cloud Mamba as natural baselines in the same domain.
- Pseudocode for the reordering algorithm to resolve ambiguity.
- Sensitivity analysis over r on datasets with different normalization conventions.
- Analysis of why bidirectional (Hydra) processing hurts performance, rather than dismissing it as an optimization challenge.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Propositions 1 and 2 are "trivial"**: The harsh critic argues these are well-known properties, but the paper uses them as motivational background, not as claimed novel contributions. This framing is appropriate and standard. Removed because the critic mischaracterizes the role of these propositions in the paper.

- **Demand for comparison with modern point cloud transformers (PointNeXt, RepSurf)**: The paper explicitly builds on and compares within the Point-MAE/Mamba framework. Requesting baselines outside the paper's stated scope is scope creep, especially since the focus is on Mamba-based architectures. Removed as scope creep.

- **Demand for formal characterization of spatial locality preservation** (quantifying NIMBA's ordering vs. z-order/Hilbert curves): While valuable, this would strengthen rather than validate the core claim. The PE ablation already empirically demonstrates that the ordering captures positional information. Removed as nice-to-have, not a core flaw.

- **Formatting/stylistic issues**: Removed per rules against formatting nitpicks.

- **Missing appendix proofs**: Removed per rules — the parser strips appendices.

## Novel Insights

The most interesting insight that emerges from combining the reviews is that NIMBA's real contribution is NOT about permutation invariance (which it does not achieve) but about *positional embedding redundancy*: by constructing a 1D ordering that already encodes 3D spatial proximity, the model needs less explicit positional information injected via embeddings. This reframes the contribution from a claim about invariance (which is flawed) to a practical finding about information redundancy in sequential models for point clouds. The PE ablation gap (1.68% vs. 4–6%) is the paper's strongest evidence and deserves to be the centerpiece rather than the "almost permutation-invariant" framing.

## Suggestions

- Rewrite the abstract and introduction to replace "almost permutation-invariant manner" with the more accurate and empirically supported "in a manner that reduces dependence on positional embeddings" or "that preserves spatial proximity in the 1D ordering."
- Add comparison with at least OctreeMamba (z-order-based, N-length sequence, N in Table 1) as the most directly competing approach.
- Provide pseudocode for the reordering algorithm.
- Qualify the "state-of-the-art" claim to reflect the limited comparison set, or expand comparisons.
- Add a sweep over the threshold r to establish its robustness.

## Score and Decision

**Calibration anchors considered:**

| Paper | Score | Comparison |
|-------|-------|-----------|
| Spectral Spatial Traversing (SU3lZ8jrRD) | 4.75 | Very similar topic (Mamba ordering for point clouds via spectral methods). More theoretically grounded but with overclaims and incomplete ablations. NIMBA is less principled (greedy heuristic vs. spectral analysis) but has a cleaner empirical finding (PE independence). |
| DSConv (XfWJT3BUmX) | 4.40 | Point cloud serialization for convolutions. Incremental, multiple engineered modules. NIMBA has a clearer conceptual contribution but similar baseline coverage issues. |
| GlobalMamba (XKQ2qzajbU) | 5.00 | Mamba spatial ordering for vision. NIMBA is comparable in contribution but with more focused ablations. |
| 3ZdGSTxKuy (Harry Potter OOD) | 2.0 | Severely overclaimed, minimal novelty. NIMBA has more substance. |
| DiffMatch (Zsfiqpft6K) | 8.0 | Strong novelty, thorough ablations, clear methodology. NIMBA is far from this level. |
| Simplifying Transformer Blocks (RtDok9E3s) | 7.33 | Clean ablations, well-motivated. NIMBA lacks this level of rigor. |

NIMBA sits between the 4.4 (DSConv) and 5.0 (GlobalMamba) anchors. It has a genuine empirical insight (PE independence) that is novel and useful, but is hampered by overclaimed scope ("almost permutation-invariant," "state-of-the-art"), a limited experimental comparison (only PointMamba as the main Mamba baseline), and an underspecified core algorithm. The Spectral Spatial Traversing paper (4.75) offers a more theoretically grounded approach to the same problem but with similar overclaim issues. NIMBA's PE finding is arguably more practical but the paper overreaches in its framing.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>