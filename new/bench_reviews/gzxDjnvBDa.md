## Summary

CrystalFramer introduces "dynamic frames" for SE(3)-invariant crystal structure modeling. Rather than using static, structure-aligned coordinate systems (PCA or lattice-based), the method defines local coordinate systems per atom per layer, constructed using attention weights learned from interatomic interactions. These dynamic frames are integrated into the Crystalformer architecture by using softmax attention weights to select frame axes and by adding angular edge features alongside existing distance features. Experiments on JARVIS, MP, and OQMD datasets show systematic improvements over Crystalformer and competitive or superior results compared to other reported SOTA methods.

---

## Claims and Support

**Claim 1: Dynamic frames are a better principle than static frames for crystal modeling.**
*Partially supported.* The ablation (Tables 1–2) compares max frames against static global frames (PCA, lattice) and one static local baseline (exp(−r²) weighting). Max frames outperform all three in most tasks. However, the architecture also adds angular edge features not present in the baseline, so "dynamism" cannot be fully isolated from "angular information." The static local frame ablation partially addresses this. The conclusion that dynamic frames are better *within this Crystalformer-based instantiation* is well-supported; the general principle is plausible but not fully isolated.

**Claim 2: CrystalFramer outperforms existing crystal encoders.**
*Partially supported.* Internal comparisons against Crystalformer and frame variants are well-controlled. External comparisons against Matformer, PotNet, iComFormer, eComFormer explicitly use "cited reported scores to reduce computational burden" (Sec. 5.1) while simultaneously modifying training epochs. This is acknowledged but limits the strength of a broad SOTA claim. On OQMD, only Crystalformer is compared, which is fair.

**Claim 3: Dynamic frames validate interaction-alignment hypothesis.**
*Partially supported.* The comparison of max frames to static local frames (distance-based weights vs. attention-based weights) provides meaningful evidence. The visualization (Figure 3) is illustrative. But the entanglement of frame construction with the attention mechanism prevents definitive causal attribution.

**Claim 4: Method is invariant to unit-cell variations.**
*Asserted but not formally verified.* Sec. 3 states: *"These frames are defined with the entire crystal structure, P̃, reconstructed from (P, L). This fact highlights an advantage of our frames being invariant to unit cell variations within the same structure."* The conceptual argument is plausible (frames use the full infinite crystal, not a specific unit cell), but the practical implementation uses stochastic tie-breaking (perturbation noise), argmax operations, and truncated periodic sums. No formal proof and no empirical test using alternative unit-cell representations are provided.

**Claim 5: Small parameter overhead with favorable efficiency tradeoff.**
*Well-supported.* Table 4 confirms ~100K additional parameters (853K → 952K). The paper honestly acknowledges training time more than doubles (32s → 74s/epoch).

**Claim 6: Weighted PCA frames outperform conventional PCA frames.**
*Mostly supported, with caveats.* In JARVIS (Table 1): weighted PCA (0.0287) < PCA frames (0.0325) for formation energy. In MP (Table 2): weighted PCA (0.0197) is identical to PCA frames (0.0197) for formation energy, and worse than baseline Crystalformer (0.0186). The improvement over conventional PCA is not consistent across all datasets and tasks.

---

## Strengths

- **Novel conceptual contribution**: The reframing of frame construction as interaction-aligned rather than structure-aligned is original and well-motivated. Dynamic, per-atom, per-layer frames based on learned attention weights have not been applied to crystals before.
- **Clean ablation within one architecture**: The paper compares PCA, lattice, static local, weighted PCA, and max frames all within the same Crystalformer backbone. This controlled internal comparison makes the relative contribution of each design choice traceable.
- **Static local frame ablation is genuinely informative**: Using exp(−r²)-weighted local frames as a control isolates some effect of the "dynamic" (attention-derived) aspect. Max frames outperform this baseline on most tasks (Table 1 and 2), which is non-trivial evidence.
- **Strong empirical results on a large benchmark**: Consistent improvement over Crystalformer on 5/5 JARVIS, 4/4 MP (except shear modulus), and 3/3 OQMD tasks. The OQMD results (817K materials) are particularly compelling as a scalability demonstration.
- **Parameter efficiency**: Only ~100K additional parameters over Crystalformer despite substantial accuracy gains—a favorable cost-performance ratio compared to larger models like iComFormer (5M params).
- **Honest acknowledgment of limitations**: The paper openly discusses discontinuity in max frames, gradient non-differentiability, weighted PCA degeneration, and OOD concerns (Sec. 6, Appendix I), which is above average in candor.

---

## Weaknesses

### Fatal
*None.*

### Major

**1. Cross-paper SOTA comparisons are not conducted under controlled conditions — why it matters:**
Sec. 5.1 explicitly states: *"we cite their reported scores to reduce computational burden"* while simultaneously increasing training epochs for CrystalFramer relative to Crystalformer. Numbers from prior works use different optimization budgets, early stopping criteria, and data preprocessing practices even on the same nominal splits. The *within-paper* comparisons (vs. Crystalformer and frame variants) are credible and well-controlled. However, the headline claims of "outperforming existing crystal encoders" over Matformer, PotNet, and iComFormer rest on uncontrolled comparisons. On JARVIS E-hull, iComFormer (0.044) ties with static local frames and outperforms max frames (0.0471). These results should be presented as contextual benchmarking against reported numbers, not as clean controlled superiority.

**2. Unit-cell invariance is asserted but not established — why it matters:**
Unit-cell invariance is introduced as a *central motivation* (Sec. 1: *"invariance to unit-cell variations within the same crystal structure"* is listed as a core requirement for crystal encoders) and repeated as a key advantage of dynamic frames (Sec. 3: *"This fact highlights an advantage of our frames being invariant to unit cell variations"*). However, the implementation uses stochastic perturbation tie-breaking (argmax with noise), sign-randomized PCA, and truncated periodic sums. None of these clearly preserve invariance under re-parameterization of the unit cell. The paper provides no formal argument and no empirical test (e.g., predicting the same property under multiple valid unit-cell representations of the same crystal). A central architectural claim is treated as self-evident when it requires verification.

**3. Isolation of "dynamism" from "angular features" is incomplete — why it matters:**
CrystalFramer adds both dynamic frames AND angular edge features (Eq. 7) over the Crystalformer baseline, which uses only distance features. The ablations all include angular features (even the static local frame variant uses angular GBFs). There is no ablation showing what happens when angular features are added with a fixed global frame (PCA or lattice) using the same GBF parameterization for angles. This would clarify whether the gains come from angular information broadly or specifically from dynamic frame construction. As currently structured, the experiments cannot fully attribute the improvement to the "dynamic" aspect rather than the richer directionality encoding in general.

### Minor

**4. No variance/error reporting despite stochastic frame construction:**
Both weighted PCA and max frames use stochastic perturbations at training and test time. No multiple-seed results or confidence intervals are reported anywhere in the paper. On JARVIS formation energy, the improvement over iComFormer is from 0.0272 to 0.0263 — a gap that may be within run-to-run variation. This makes it difficult to assess which marginal improvements are robust, particularly for cross-method comparisons.

**5. Weighted PCA frames underperform in ways not fully explained:**
In MP (Table 2), weighted PCA (0.0197) is no better than conventional PCA frames (0.0197) and worse than the baseline Crystalformer (0.0186). Appendix F is referenced but the core paper does not adequately explain why weighted PCA fails when max frames succeed, given both leverage the same attention weights. This gap weakens the claim that dynamism per se is the key — it may be the argmax selection heuristic that matters, not the dynamic alignment principle broadly.

**6. OOD generalization concern is relegated to appendix:**
Section 6 acknowledges: *"[max frames] introduce noticeable discontinuities to the model and may limit generalization to out-of-domain data, as discussed in Appendix I."* Given that max frames are the headline method and OOD generalization is critical for materials discovery applications, the extent of this limitation deserves more quantitative treatment in the main body.

### Trivial

**7. Training time doubles:** Table 4 documents 32s → 74s per epoch, acknowledged and partially addressed with lightweight GBF configurations in Appendix G. This is an honest limitation; its impact depends on the application.

---

## Nice-to-Haves

- **Multiple random seeds with reported variance**: Given stochastic frame construction, 3–5 seed runs for at least CrystalFramer and Crystalformer would significantly strengthen the empirical claims on datasets where margins are narrow.
- **Ablation: angular features with static global frames**: Adding a row to Tables 1–2 for Crystalformer + angular edge features (Eq. 7) with PCA or lattice frames (but no dynamic frame) would cleanly isolate the contribution of dynamism from that of angular features.
- **Out-of-distribution generalization test**: An experiment training on MP and testing on structurally dissimilar materials (e.g., from JARVIS or novel crystal families) would empirically assess the discontinuity concern flagged in Sec. 6 and Appendix I.
- **Formal argument or empirical test for unit-cell invariance**: Even a short proof-sketch showing that the infinite crystal construction and frame definition are invariant to unit-cell reparameterization would significantly strengthen the paper's theoretical grounding.
- **Force/equivariant property evaluation (preliminary)**: The equivariant extension is discussed qualitatively (Sec. 6). Even a small experiment on force prediction would substantially broaden the impact claim and validate the framing of dynamic frames as a general mechanism.
- **Why weighted PCA fails but max frames succeed**: Deeper analysis of the failure mode of continuous weighted PCA (eigenvalue dynamics, effective interaction coverage) would help the community understand when the dynamic frame principle generalizes beyond the argmax implementation.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**[R1] Criticism that MACE/NequIP/Allegro equivariant baselines are missing (Spark reviewer):** These architectures are primarily developed for molecular simulation and only recently applied to periodic crystal benchmarks in specialized settings. Their omission does not represent a methodological gap in a paper explicitly scoped to invariant crystal transformers. The compared baselines (eComFormer, iComFormer, PotNet, Matformer, Crystalformer) are the direct community standards for this exact task and dataset split. This is scope creep.

**[R2] Criticism questioning the existence, availability, or reproducibility of cited baselines:** No reviewer explicitly raised this, but any concern of the form "iComFormer scores cannot be verified" falls under the hard removal rule — cited results exist.

**[R3] "Non-differentiability of max frames as a fundamental flaw" (neutral reviewer):** The paper explicitly addresses this in Sec. 3.1, footnote 2: *"simply ignoring the frame gradients gave the best results."* The attention weights still receive gradients through the message-passing path (Eq. 5). This is acknowledged and handled; it is a limitation but not a fatal flaw. Retained only as a minor point in the context of the broader mechanistic interpretation concern.

**[R4] Demanding force prediction / equivariant evaluation as a core missing experiment:** The paper explicitly scopes the contribution to invariant property prediction and positions equivariant extension as future work (Sec. 6). Evaluating on forces would require a different dataset, training objective, and architectural extension. This is beyond the paper's stated scope; retained only as a nice-to-have.

**[R5] Demanding a broader backbone comparison (dynamic frames on ComFormer or other architectures):** The paper explicitly motivates the choice of Crystalformer over ComFormer-family models because of the difference between standard softmax and channel-wise sigmoid attention (Sec. 2.2, Sec. 6). The backbone choice is reasoned, not arbitrary. This is a reasonable future direction, not a current weakness.

---

## Novel Insights

The paper's most genuinely novel intellectual contribution is the reconceptualization of frames not as canonical structural representations but as interaction masks — the observation that each atom in a message-passing layer has its own "partial view" of the crystal (Eq. 5, Sec. 3) and that frames should reflect *that* partial view rather than the whole structure. This inverts the conventional frame-averaging intuition: instead of finding a canonical coordinate system for the full structure, one finds a coordinate system for each atom's learned interaction neighborhood. The static local frame ablation — which uses the same locality principle but with fixed distance-based weights — provides non-trivial evidence that it is specifically the learned interaction weights (and not just locality) that drives the performance gain for the max-frame variant. The empirical finding that max frames (discrete, stochastic, non-differentiable) consistently outperform weighted PCA frames (continuous, smooth, differentiable) is a counterintuitive and underexplained result that warrants further investigation; it suggests that sharp, discrete selection of primary interaction axes may be more useful for SE(3)-invariant crystal encoding than smooth averaging, possibly because it more reliably aligns with dominant coordination motifs.

---

## Suggestions

1. **Formally justify unit-cell invariance** or provide an empirical test: generate multiple valid unit-cell representations of the same 10–20 JARVIS materials and verify that CrystalFramer produces consistent predictions and similar frames.
2. **Add an "angular features only, static frame" row to Tables 1–2** to isolate the contribution of dynamic frame construction from the angular edge feature augmentation.
3. **Report mean ± std over 3–5 seeds** for CrystalFramer and Crystalformer (at minimum) to enable statistical assessment of margins on close tasks.
4. **Narrow or qualify the SOTA comparison language** in the abstract and Sec. 5.1 to distinguish between (a) controlled improvements over Crystalformer and (b) contextual comparison to reported prior results under potentially different training conditions.
5. **Explain the weighted PCA failure mode more thoroughly** in the main paper rather than appendix — this is key to understanding the scope of the dynamic frames principle.

---

## Score and Decision

**Originality:** Good. The dynamic frames concept for crystals is novel; applying attention weights as interaction masks for frame construction is a new and principled idea.

**Importance of research question:** High. Crystal property prediction is a practically important problem and the question of how to encode directional geometry in periodic systems is fundamental.

**Claims support:** Moderate. The within-paper ablations strongly support the core contribution. The cross-paper SOTA claims are understated by the comparison methodology. The unit-cell invariance claim is unverified.

**Soundness of experiments:** Good, with gaps. The ablation design is thoughtful; the lack of variance reporting and the absence of one key ablation (angular features with static frames) are meaningful gaps.

**Clarity of writing:** Good. Well-organized, acknowledges limitations honestly, clear motivation.

**Value to research community:** Solid. Achieves state-of-the-art on large benchmarks with minimal parameter overhead. Code is released. The conceptual contribution could influence frame-based methods broadly.

**Overall:** A solid empirical paper advancing the state of the art in crystal property prediction with a genuine conceptual contribution. The main weaknesses are methodological gaps in validating the unit-cell invariance claim and isolating the role of dynamism from angular features, plus uncontrolled cross-paper comparisons. These do not invalidate the core contribution but should be addressed.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>