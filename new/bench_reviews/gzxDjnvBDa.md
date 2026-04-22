Now I have a thorough understanding of the paper and the reviews. Let me synthesize the final review.

## Summary

The paper introduces "dynamic frames" for SE(3)-invariant crystal structure modeling, proposing that local coordinate systems for each atom should be constructed based on learned attention weights rather than static structural information. The authors instantiate this concept via two variants—weighted PCA frames and max frames—integrated into the Crystalformer architecture, creating "CrystalFramer." Extensive experiments on JARVIS, MP, and OQMD datasets demonstrate consistent improvements, with max frames achieving state-of-the-art results on most tasks.

## Strengths

- **Novel conceptual contribution**: The idea that frames should reflect interatomic interaction patterns rather than purely structural alignment is a genuine insight. The argument in Section 3 (Eq. 5) that static frames can be influenced by atoms with zero interaction weight ($w_{ij(n)}=0$) is well-motivated and challenges a real assumption in prior work. (Section 3, Eq. 5 discussion)

- **Substantial empirical improvements**: CrystalFramer with max frames achieves meaningful improvements over Crystalformer across all three datasets and most tasks—e.g., formation energy on JARVIS improves from 0.0306 to 0.0263 (~14% relative), formation energy on MP from 0.0186 to 0.0172, and consistent gains on OQMD. These are nontrivial improvements over an already competitive baseline. (Tables 1–3)

- **Effective parameter efficiency**: As shown in Table 4, CrystalFramer adds only ~100K parameters (952K vs 853K) while outperforming models with 2–6× more parameters (iComFormer: 5.0M, PotNet: 1.8M). (Section 5.2, Table 4)

- **Thorough ablation across frame variants**: Comparing PCA frames, lattice frames, static local frames, weighted PCA frames, and max frames provides meaningful insight into what frame constructions work and why. The finding that max frames outperform both static and PCA-based alternatives is informative. (Tables 1–2)

- **Honest discussion of limitations**: The paper acknowledges the discontinuity issue with max frames (Section 6), the gradient detachment decision (footnote 2, Section 3.1), and the limited improvements of weighted PCA frames (Section 5.1).

## Weaknesses

### Fatal
None

### Major

- **The "dynamic" framing of the contribution is only partially supported by the evidence**: The paper's central narrative is that *dynamic* frames—constructed from learned attention weights per atom per layer—are superior to static frames aligned only with structure. However, the data tells a more nuanced story. The weighted PCA frame variant (the most natural instantiation of the "dynamic" concept, using attention weights continuously) performs *worse than or equal to* static local frames on most metrics (JARVIS formation energy: 0.0287 vs 0.0285; MP formation energy: 0.0197 vs 0.0178; MP bandgap: 0.214 vs 0.191). Only max frames—which use an argmax construction procedure that may benefit from its discrete, simpler selection mechanism rather than from the "dynamic" nature of the weights—consistently outperform static local frames. The paper does not include a "static max frames" ablation (i.e., argmax selection with fixed distance-based weights) that would isolate the contribution of dynamism from the contribution of the frame construction method. This matters because the paper frames the entire contribution around "dynamic" frames, when the evidence suggests the frame *construction procedure* (argmax + Gram-Schmidt) may be the primary driver. (Tables 1–2, Section 3.1, Section 5.1)

- **No experimental verification of SE(3) invariance**: The paper's title and framing center on SE(3)-invariant modeling. Both frame construction methods provide only *approximate* invariance via stochastic frame averaging (random sign flips for PCA; random perturbation for max frames). While stochastic FA is an established technique, the paper's specific construction—with gradient-detached attention weights and perturbation-based tie-breaking—has not been empirically verified to produce rotation-invariant predictions. A simple test (e.g., measuring prediction variance under random SO(3) rotations of test structures) would address this. Without it, the claim that the model achieves SE(3) invariance remains unverified, especially given the acknowledged discontinuities in max frame construction that "may limit generalization to out-of-domain data." (Title, Abstract, Section 2.3, Section 6)

### Minor

- **Gradient detachment limits interpretability and may affect frame quality**: The paper acknowledges (Section 3.1, footnote 2) that gradients from frames to attention weights are omitted, meaning the attention weights are optimized only for message-passing, not for frame quality. This design choice means the model has no mechanism to adjust attention weights that produce degenerate frames. The reported 10% eigenvalue degeneration rate for weighted PCA frames and the underperformance of this variant relative to max frames may be partially attributable to this. The authors tried alternatives (straight-through estimator, temperature annealing) but report only that the current approach "gave the best results" without detailed comparison. (Section 3.1, footnote 2)

- **Test-time determinism is unspecified**: The paper does not clarify whether the perturbation noise used for max frame tie-breaking is applied at inference time (making predictions stochastic) or only during training. If stochastic at inference, predicted properties could vary across forward passes, which is important for practical deployment. (Section 3.1)

### Trivial
None

## Nice-to-Haves

- A "static max frames" ablation (using argmax + Gram-Schmidt with fixed distance-based weights $w_{ij(n)} = \exp(-r_{ij(n)}^2)$ rather than attention weights) would directly isolate the contribution of "dynamism" and significantly strengthen the paper's narrative.

- An SE(3) invariance test measuring prediction consistency under random rotations would address the core motivation and require minimal additional computation.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "The comparison against iComFormer and eComFormer conflates architecture differences"**: The paper itself explicitly notes (Section 5.1) that ComFormer uses per-task hyperparameter tuning while CrystalFramer uses only dataset-level adjustments. This is acknowledged, not hidden. Additionally, the most meaningful comparison (Crystalformer vs. CrystalFramer) is clearly presented. This criticism overstates the issue—fair comparison concerns are noted but not a methodological flaw.

- **Harsh critic: "OQMD results only compare against Crystalformer"**: The OQMD dataset is 817K materials, far larger than JARVIS/MP. Many prior methods have not been evaluated on OQMD, so the comparison limitation reflects the field rather than an author oversight. This is a scope creep criticism.

- **Strength finder: "Effective handling of eigenvalue degeneracy"**: While the max frame construction bypasses PCA degeneracy, this is better characterized as a *design choice* that avoids a known problem rather than a *strength* of a solution—weighted PCA frames still suffer from degeneracy issues (10% rate), and the max frame approach introduces its own discontinuity concerns. Moved because it conflates a workaround with a strength.

- **Strength finder: "Scalability demonstration on OQMD"**: While evaluating on a larger dataset is welcome, the OQMD comparison only includes Crystalformer as a baseline. This limits what can be concluded about scalability relative to other methods.

## Novel Insights

The paper raises an interesting tension that it does not fully resolve: the best-performing variant (max frames) may succeed primarily due to the *discrete selection* of frame axes (argmax) rather than the *dynamism* of the weights. This is suggested by (1) the underperformance of continuously-weighted PCA frames relative to static local frames, (2) the authors' own observation that max frames converge faster "due to the discrete nature of their construction," and (3) the lack of a static-max-frame control. This suggests that the expressiveness of per-atom-per-layer coordinate frames—which undeniably improves performance—may be distinct from whether those frames are "dynamic" or "static," and the axis selection algorithm matters more than the weight adaptation mechanism.

## Suggestions

- Add a "static max frames" ablation variant (same argmax frame construction but with fixed Gaussian distance weights) to cleanly attribute performance gains to dynamism vs. construction method.

- Add a rotation-invariance test: apply random SO(3) rotations to test structures and measure prediction variance to empirically verify approximate SE(3) invariance.

- Clarify in the paper whether perturbation noise is applied at inference time and, if so, report prediction stability statistics.

- Temper the "dynamic frames" narrative in light of the weighted PCA vs. static local comparison; acknowledge that the max frame construction procedure itself may be a key factor.

## Score and Decision Calibration

**Anchors compared against:**

| Paper | Avg Score | Comparison |
|-------|----------|------------|
| Crystalformer (baseline paper this work extends) | 7.25 | This paper meaningfully extends Crystalformer with a novel concept and consistent improvements; comparable quality but with a partially overclaimed narrative |
| DiffCSP++ (crystal generation, SE(3)-equivariant) | 7.33 | Lower—DiffCSP++ had cleaner claims for its domain |
| PDDFormer (crystal property prediction) | 5.0 | Higher—PDDFormer had fundamental completeness claims that were wrong, while this paper's empirical gains are real |
| Learning Symmetries through Loss Landscape (equivariance claims, no verification) | 5.25 | Higher—this paper has stronger empirical grounding and builds on established stochastic FA theory, but shares the equivariance verification gap |
| CGIL (overclaimed conceptual contribution) | 3.0 | Much higher—this paper has genuine, substantial empirical improvements |

This paper shares characteristics with papers in the 5.5–7 range: real empirical contributions but a notable gap between the narrative ("dynamic frames are superior") and the evidence (only argmax-based max frames are clearly superior, while the genuinely dynamic weighted PCA variant is not). The invariance verification gap, while not fatal (the theoretical basis via stochastic FA exists), is a notable omission given the title/abstract emphasis. The paper is clearly above the 4–5 range (where papers have fundamental methodological issues) but below the 7+ range (where contributions are clean and well-supported). The strongest comparable anchor is Crystalformer itself at 7.25—this paper extends it meaningfully but with somewhat overclaimed framing.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>