## Summary

This paper proposes *dynamic frames* for SE(3)-invariant crystal property prediction. Rather than using a single static frame per crystal (e.g., PCA or lattice frames), the authors construct per-atom, per-layer local coordinate systems whose axes are determined by learned self-attention weights. They instantiate this idea as *CrystalFramer*, built on the Crystalformer transformer, and evaluate it on JARVIS, Materials Project (MP), and OQMD benchmarks. The *max frame* variant achieves strong empirical results, outperforming the baseline and several competing methods on most tasks while adding only ~100K parameters.

## Strengths

- **Strong empirical performance with minimal parameter overhead.** CrystalFramer with max frames achieves the best or near-best MAE on the majority of JARVIS and MP tasks (Tables 1–2) and improves over the Crystalformer baseline on all three OQMD tasks (Table 3), while using only 952K parameters—far fewer than PotNet, Matformer, or iComFormer (Table 4).
- **Directly addresses concrete pathologies of existing frames.** The paper explicitly tackles eigenvalue degeneration in PCA frames (Sec. 2.3) and unit-cell sensitivity (Sec. 3) via local, structure-reconstructed frame construction, which is a genuine and well-motivated improvement.
- **Clean architectural integration.** The frame construction plugs neatly into Crystalformer’s existing attention mechanism without circular dependencies (Sec. 3.2), and the handling of non-differentiable frame operations is pragmatic and clearly explained (Sec. 3.1, footnote 2).

## Weaknesses

### Fatal
None.

### Major
- **The central conceptual claim conflates locality with dynamic learned alignment, and the ablations do not robustly isolate the latter.** The paper’s framing (title, abstract, Sec. 3) presents *dynamic* attention-based re-weighting as the essential insight. However, the **static local frame** ablation—which masks frame construction by distance rather than learned attention—achieves remarkably similar performance to max frames across most tasks. On JARVIS E hull, static local frames (0.0444) actually outperform max frames (0.0471), and on MP formation energy the gap is only 0.0178 vs. 0.0172 (Tables 1–2). While max frames do outperform static local frames on 8 of 9 tasks, the paper never quantitatively analyzes *when* or *why* learned attention weights improve upon simple distance-based locality. Because the stated motivation—excluding zero-weight atoms from frame construction—is already satisfied by static local masking, the evidence better supports the narrower conclusion that **local per-atom frames** are beneficial, with dynamic weighting providing inconsistent marginal gains. This fracture between the paper’s central conceptual thesis and the experimental evidence is consequential.

### Minor
- **Weighted PCA frames frequently underperform the baseline, undermining the “family of dynamic frames” framing.** The paper introduces dynamic frames as a broad conceptual family encompassing both weighted PCA and max frames. Yet weighted PCA regresses relative to the plain Crystalformer baseline on MP formation energy (0.0197 vs. 0.0186), MP bandgap (0.214 vs. 0.198), and shows only modest or mixed results elsewhere (Tables 1–2). The main text briefly notes “relatively limited improvements,” but because only max frames yield reliable gains, the evidence supports a specific engineering choice rather than a validated general concept. The paper should more directly own this asymmetry in the main text.
- **No variance estimates or statistical significance testing.** The tables report single-run MAEs with no standard deviations across seeds. Several differences between static local and max frames are small (e.g., MP shear modulus 0.0708 vs. 0.0677), making it impossible to assess whether the “dynamic” advantage is robust or within training noise.
- **Ambiguity in training protocol introduces a potential confound.** The authors state they increased training epochs relative to the baseline to accommodate “increased complexity” (Sec. 5). The Crystalformer baseline rows in Tables 1–2 are attributed to Taniai et al. (2024), while the frame variants are indented beneath it. It is unclear whether the baseline itself—and the intermediate frame ablations (PCA, lattice, static local)—were all re-trained with this identical extended budget. If the proposed max-frame variant received substantially more optimization than the cited baseline, the comparison is confounded.

### Trivial
None.

## Nice-to-Haves
- A direct ablation that fixes locality (e.g., static local masking) and incrementally adds dynamic attention-based re-weighting, with statistical testing, to isolate the marginal contribution of “dynamic” alignment.
- A mechanistic explanation for why continuous weighted PCA fails where discrete max frames succeed (e.g., gradient instability through eigendecomposition, sensitivity to attention magnitude), given that both use the same underlying attention weights.
- A quantitative invariance audit (e.g., prediction variance under randomized unit-cell choices and rotations at test time) to complement the theoretical claims about unit-cell invariance.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **“The paper should not be accepted in its current form.”** This recommendation is too severe. Despite the conceptual overselling, the paper presents a novel architectural idea, strong benchmark results, and a well-executed implementation. These are genuine contributions that warrant acceptance with revisions, not outright rejection.
- **Missing appendix, missing proofs, or absent references.** The parser strips appendix sections; they exist in the original submission.
- **Formatting/style nitpicks, typos, or grammatical issues.** These are parser artifacts, not author errors.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Restructure the abstract and introduction to acknowledge that locality is a major driver of improvement, and frame dynamic weighting as a further refinement that provides additional (if modest and task-dependent) gains, rather than the sole essential insight.
- Report standard deviations across multiple seeds for all main results, or at minimum for the critical static-local-vs-max-frame comparisons.
- Clarify explicitly in Sec. 5 whether the baseline Crystalformer and all ablated frame variants were trained with the identical epoch budget and optimizer settings.

## Score and Decision

**Calibration comparison:**
- *High anchor:* `fxQiecl9HB.md` (Crystalformer, avg 7.25, Accept poster). Strong empirical results, clean presentation, but had concerns about similarity to PotNet. CrystalFramer is topically similar and builds on Crystalformer. It has strong empirical results and clean presentation, but its central conceptual claim is less robustly isolated than Crystalformer’s infinite-attention formulation. It is therefore below this anchor.
- *Medium anchor:* `jckKNzYYA6.md` (AssembleFlow, avg 6.50, Accept poster). Good empirical results and clear writing, but reviewers raised concerns about whether the core decomposition was sufficiently novel. CrystalFramer is comparable: good results and clarity, but the “dynamic” aspect is not cleanly separated from locality. I place CrystalFramer slightly below this anchor because the ablation issue more directly undermines the central thesis.
- *Low anchor:* `ewjN1MAnJi.md` (PDDFormer, avg 5.00, Withdrawn). Had serious concerns about completeness claims and limited novelty. CrystalFramer is clearly above this—it has stronger empirical gains, fewer correctness issues, and a more clearly defined contribution.
- *Low anchor:* `zUDbPgskDS.md` (CrysToGraph, avg 3.25, Reject). Weak contribution, messy presentation. CrystalFramer is far above this.

**Final assessment:** The paper is a solid empirical contribution with an interesting architectural direction, but the central conceptual claim is ahead of the ablation evidence. The max-frame results are genuinely strong, the implementation is clean, and the efficiency is commendable. However, the conflation of locality with dynamic learning, the failure of weighted PCA to validate the “family” concept, and the lack of variance estimates prevent a higher score. I score it at the lower end of the acceptable range: a clear empirical advance, but one whose conceptual framing needs honest recalibration.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>