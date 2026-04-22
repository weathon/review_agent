## Summary
The paper proposes CrystalFramer, a transformer-based crystal encoder that uses per-atom, per-layer “dynamic frames” constructed from attention weights to define SE(3)-invariant angular features, extending the distance-only Crystallformer. Experiments on JARVIS, Materials Project, and OQMD show consistent accuracy gains over Crystallformer and over static frame methods such as PCA and lattice frames, with moderate parameter overhead but higher training cost.

## Strengths
- **Clear, novel framing of interaction-based dynamic frames.** Section 3 defines frames \(F_i\) per atom, per layer, constructed from attention weights \(w_{ij(n)}\) (Eq. 5, Sec. 3.1), conceptually shifting from globally structure-aligned frames to interaction-aligned local frames.
- **Strong, consistent empirical gains over a solid baseline and prior frame methods.** Tables 1–3 show that max frames substantially improve over Crystallformer and that dynamic frames outperform PCA and lattice frames across multiple JARVIS/MP tasks and OQMD; e.g., JARVIS formation energy MAE improves from 0.0306 (Crystallformer) to 0.0263 (max frames), while PCA/lattice frames worsen or barely match the baseline.
- **Ablations within the frame family are informative.** Static local frames (distance-based, no attention) and weighted PCA frames are included (Tables 1–2), demonstrating that (i) naive static local frames are already strong, and (ii) max frames leveraging dynamic attention generally perform best.
- **Method is architecturally simple and parameter-efficient.** Sec. 3.2 and Fig. 2 show a minimal extension of Crystallformer: add frame construction and angular GBF encodings (Eq. 7). Table 4 shows only ~12% parameter increase (853K → 952K) and still substantially fewer parameters than PotNet/Matformer/iConformer.
- **Thoughtful handling of numerical issues and qualitative analysis.** Sec. 3.1 and Footnotes 1–2 explicitly discuss degeneracies, perturbation-based symmetry breaking, and the decision to stop gradients through frames; Sec. 6 and Fig. 3 provide qualitative evidence that dynamic frames align with meaningful local motifs (octahedra/tetrahedra).

## Weaknesses

### Fatal
None.

### Major
- **Core invariance / equivariance guarantee for dynamic frames is under-argued and under-tested.**  
  The paper relies heavily on FA/Stochastic FA intuition (Sec. 2.3, 3.1) and states that perturbations “are considered a type of stochastic FA,” but never clearly argues that the combination of (i) attention weights depending on learned features, (ii) argmax-based max-frame selection with random perturbations (Sec. 3.1), and (iii) non-differentiable, per-head, per-layer frame construction yields periodic SE(3)-invariance in any precise sense. The model is certainly rotation/translation invariant at the *input* feature level (distances and cosines in Eq. 7), but because the frames are stochastically and discontinuously selected from attention-defined neighborhoods, small rotations or unit-cell changes could induce qualitatively different frames and therefore different outputs. The paper does not provide:
  - a constructive argument that the distribution over \(F_i\) is consistent under SE(3) and unit-cell changes, nor  
  - empirical tests (e.g., applying random rotations or alternative unit cells at test time) to demonstrate approximate invariance.  
  Given that “rethinking the role of frames for SE(3)-invariant crystal structure modeling” is the central motivation, this is a substantial conceptual gap.

- **Causal attribution to “dynamic, interaction-based” frames vs. richer angular features is not fully isolated.**  
  The architecture change from Crystallformer to CrystalFramer combines two ingredients: (i) adding richer angle-based GBF encodings (Eq. 7) and (ii) choosing frames based on interaction weights. The paper compares several frame constructions (PCA, lattice, static local, weighted PCA, max), but *does not* include a key control: Crystallformer + the same distance+angle GBFs in Eq. 7 computed in a trivial, non-interaction-based global frame (e.g., fixed orthonormal basis or global PCA without attention dependence). As a result:
  - The strong gains of static local frames over Crystallformer (e.g., JARVIS formation energy 0.0306 → 0.0285; MP 0.0186 → 0.0178 in Table 2) show that locality + angles are already very powerful.
  - The incremental benefit of max frames over static local frames, while real (e.g., JARVIS 0.0285 → 0.0263; MP 0.0178 → 0.0172), is not disentangled from the general effect of additional angular features versus the specific interaction-based dynamic construction.  
  Thus, the strong narrative that “dynamic, interaction-aligned frames” are the main driver of performance is somewhat ahead of the experimental evidence.

- **Baseline comparisons are not strictly controlled in terms of training budget and scope.**  
  Sec. 5.1 states that the authors “precisely follow the training settings of the baseline method, Crystalformer … with only one modification. We have increased the number of training epochs…”, and for JARVIS they train CrystalFramer for 2000 epochs. However, the Crystallformer numbers in Tables 1–3 are cited from prior work and not reported under the same extended schedule, so it remains unclear whether Crystallformer would close part of the gap with longer training under otherwise identical hyperparameters. Similarly, for OQMD the only baseline is Crystallformer, and other strong architectures (PotNet, i/eConformer) are absent, yet the text still frames results as outperforming “other state-of-the-art networks.” Overall, this leaves some ambiguity as to how much of the performance gain is due to the new inductive bias vs. more training or different hyperparameter choices and a limited baseline set on the largest dataset.

### Minor
- **Stochasticity and non-differentiable frame selection are not empirically stress-tested.**  
  Sec. 3.1 acknowledges that both PCA and max frames use random perturbations for symmetry breaking, and gradients through frames are stopped; Sec. 6 notes that max frames introduce discontinuities and “may limit generalization to out-of-domain data.” Yet the main experiments (Tables 1–3) report only single-point MAEs without:
  - seed variance for the stochastic frame construction, or  
  - sensitivity tests to small structural perturbations (slight distortions, rotations, or alternative cells).  
  This makes it harder to assess robustness and reproducibility of the max-frame variant, which is the main recommended configuration.

- **Interpretability/physical alignment story remains mostly qualitative.**  
  The paper argues that dynamic frames better capture “actively interacting atoms” and physically meaningful local motifs (Sec. 3, 6), and Fig. 3 provides a nice qualitative visualization. However, there is no quantitative analysis linking frame axes to known coordination geometries or to property-specific sensitivity (e.g., cases where angle distortions are critical). This does not undermine the correctness of the method but weakens the strength of the interpretability claim.

- **Limited exploration of alternative differentiable frame constructions.**  
  Footnote 2 briefly mentions attempts at straight-through estimators and temperature annealing, but no results are shown. A small ablation comparing these more continuous relaxations to the non-differentiable max frame would help substantiate the claim that simply stopping gradients “gave the best results” and that the chosen design is not a fragile local optimum.

- **No explicit quantitative test of periodic SE(3) invariance / unit-cell invariance.**  
  While Sec. 2.2 and 3 emphasize periodic SE(3) invariance and claim invariance to unit-cell variations via the use of \(\tilde P\), there is no experiment that demonstrates consistency of predictions across equivalent unit-cell parameterizations for the same crystal. This is non-fatal but would substantially strengthen the narrative.

### Trivial
- Minor clarity gaps, e.g., the exact truncation strategy for the “infinite” neighbor sums in frame construction is only implicitly tied to the 3.5\(\sigma_i\) radius used in Crystallformer (Sec. 3.2), and the discussion of “approximately achieving SE(3) invariance” (Sec. 2.3, 3.1) could be more precise about what is guaranteed vs. empirically observed.

## Nice-to-Haves
- A dedicated experiment that applies random rotations and alternative unit cells at test time (for fixed physical crystals), reporting the distribution of absolute prediction differences for Crystallformer vs. CrystalFramer, would concretely illustrate any invariance advantages or potential brittleness introduced by dynamic frames.
- Quantitative analysis of how often max-frame axes align with known coordination directions (e.g., octahedral/tetrahedral environments) across a large test set, and whether such alignment correlates with lower errors on specific subsets of materials.
- Efficiency-oriented baselines where Crystallformer is given a comparable FLOP or wall-clock budget (e.g., more layers or wider channels) to see whether the same compute can partially or fully close the gap.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Existing crystal frame methods are fundamentally misaligned with what matters physically” as a strong empirical criticism.**  
  The harsh review questioned that the paper overstates weaknesses of PCA/lattice frames without showing that those weaknesses matter. The actual paper’s wording is more measured (Sec. 2.3, 3) and includes explicit empirical comparisons where PCA/lattice frames underperform; thus, a complaint that this is purely rhetoric is overstated and has been dropped as a weakness.
- **Demand for comparisons to every contemporaneous large-scale crystal model on OQMD.**  
  While broader baselines would help, requiring inclusion of all known models goes beyond what is standard and is partially mitigated by the strong Crystallformer baseline; this is better framed as a nice-to-have than a core flaw, so the stricter formulation is removed.
- **Claims that the method is not SE(3)-invariant at all.**  
  The paper does rely on distances and cosines (Eq. 7) and uses FA-inspired stochastic framing; it never claims exact invariance guarantees beyond the FA analogy. The harsher suggestion that the model is outright non-invariant is too strong without constructed counterexamples and is softened to a “under-argued invariance story” in the Major weaknesses.

## Novel Insights
None beyond the paper’s own contributions; the main critical insights center on the need for more direct evidence on invariance properties and more fine-grained ablations to separate the roles of angular features and interaction-based dynamic frames.

## Suggestions
- Provide a clear, self-contained argument or at least a careful discussion of how (or whether) dynamic frames preserve periodic SE(3) invariance in distribution, given the stochastic, argmax-based construction and dependence on learned attention weights.
- Add an ablation where Crystallformer is augmented with the same angular GBFs (Eq. 7) but using a trivial, non-dynamic global frame; this will isolate the incremental benefit of interaction-based dynamic frames beyond “just adding angles.”
- Retrain Crystallformer under the same epoch counts and key hyperparameters used for CrystalFramer (at least on JARVIS and MP) and report side-by-side results, clarifying how much of the improvement stems from architecture vs. training schedule.
- Report results across multiple random seeds, and, if feasible, rotation/unit-cell perturbation tests, to quantify robustness to the stochastic frame mechanism and to structural transformations.
- If space allows, include a small study of differentiable or smoothed frame variants versus the chosen non-differentiable max frame to justify that ignoring frame gradients is a principled choice and not merely an implementation convenience.

### Overall assessment (originality, importance, soundness, clarity, value)
- **Originality:** High; the interaction-based, per-atom dynamic frame construction within a standard transformer is a novel and conceptually interesting direction.
- **Importance:** Moderate to high; crystal property prediction is an active area and methods that improve accuracy with modest overhead are valuable, but the lack of stronger invariance evidence and broader baselines tempers the impact.
- **Support for claims / soundness of experiments:** Empirical gains over baselines are solid, but causal claims about the specific role of dynamic frames and about invariance properties are only partially supported by the current experiments.
- **Clarity:** Generally high; the method and experiments are clearly described, with useful figures and explicit architectural detail.
- **Value to community:** Good; the approach is practical, integrates easily into existing transformer architectures, and the empirical results are strong, but the above gaps prevent it from being clearly top-tier in its present form.

## Score and Decision

### Calibration anchors used
- **Medium-score, topic-related anchors (4–6 range):**
  - `/home/wg25r/review_agent/human_reviews/ewjN1MAnJi.md` (avg 5.00, Withdrawn/Reject): crystal transformer (PDDFormer) with some invariance ideas but mixed empirical strength. The current paper has clearer conceptual framing and stronger, more consistent empirical gains.
  - `/home/wg25r/review_agent/human_reviews/rcdR97P2Mp.md` (avg 4.50, Reject): periodic crystal invariance paper with concerns about overclaiming invariance; our paper is somewhat stronger empirically but has a similar pattern of under-argued invariance claims.
  - `/home/wg25r/review_agent/human_reviews/NVKwjCIAAX.md` (avg 4.75, Reject): crystal-structure optimization with decent experiments but concerns about baselines and clarity; qualitatively comparable but CrystalFramer seems more solid experimentally.
  - `/home/wg25r/review_agent/human_reviews/z3mPLBLfGY.md` (avg 6.00, Reject): SE(3)-equivariant molecular interaction transformer with strong experiments but some scope/claim issues; this is a bit stronger in theoretical grounding than the current paper, suggesting a slightly lower or similar score for CrystalFramer.
- **High-score, equivariant/invariance anchors (>7):**
  - `/home/wg25r/review_agent/human_reviews/A4eCzSohhx.md` (avg 7.00, Accept): Equivariant Neural Fields, strong theory and experiments; more thoroughly justified invariance than the current paper.
  - `/home/wg25r/review_agent/human_reviews/BBD6KXIGJL.md` (avg 7.33, Spotlight): Hybrid Directional GNN with carefully validated equivariance; clearly stronger in rigor.
  - `/home/wg25r/review_agent/human_reviews/9UIGyJJpay.md` (avg 7.33, Spotlight): GVF-based protein design using frames; deeper integration of frames and well-tested; stronger than the present work overall.
- **Medium-score, “strong experiments but overclaiming invariance” anchors:**
  - `/home/wg25r/review_agent/human_reviews/K3SviXqDcj.md` (avg 4.67, Reject): good results but overstates necessity of invariance; similar pattern to this paper but with weaker empirical story.
  - `/home/wg25r/review_agent/human_reviews/MtbelAMXJg.md` (avg 4.33, Reject): strong experiments on learned invariances via pruning but overclaiming; current paper is similar in overclaiming but somewhat more careful.
- **Low-score anchors (<3):**
  - `/home/wg25r/review_agent/human_reviews/xVbke7yC07.md` (avg 2.33, Reject): GNN for cyclone prediction with serious methodological flaws; clearly weaker than the paper under review.
  - `/home/wg25r/review_agent/human_reviews/eJhgguibXu.md` (avg 2.50, Reject): low-quality GNN paper with poor baselines and unclear contributions; far below the current paper.
  - `/home/wg25r/review_agent/human_reviews/AxYTFpdlvj.md` (avg 2.00, Reject): very weak GNN work; serves as a lower bound anchor.
  - `/home/wg25r/review_agent/human_reviews/EmrbRRworT.md` (avg 2.33, Withdrawn/Reject): another clearly weak GNN paper.

Relative to these:
- The current paper is substantially stronger than the <3 anchors and clearly above the 4.3–5.0 medium anchors that had more serious experimental and clarity deficiencies.
- It is somewhat below the 7–7.3 high anchors, which offer more rigorous theoretical guarantees or more exhaustive experimental validation.
- It is close to the 6.0 equivariant-transformer anchor (`z3mPLBLfGY.md`), but with slightly weaker theoretical footing and somewhat narrower evaluation.

Positioning relative to these anchors, a calibrated score around 6.0–6.5 seems appropriate. Given the strong and consistent empirical results and solid originality, but notable gaps in invariance justification and causal attribution, I lean toward the middle of that range.

**Final calibrated score:** 6.0  
**Provisional decision:** Borderline accept/reject; given typical selectivity and the unresolved major issues, I lean slightly toward **Reject**.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>