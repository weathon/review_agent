=== CALIBRATION EXAMPLE 21 ===

# Final Consolidated Review
## Summary

This paper introduces Multi-View Probabilistic Slot Attention (MVPSA), a framework for learning identifiable object-centric representations from multiple views without requiring explicit camera/viewpoint conditioning. Building on Probabilistic Slot Attention (Kori et al., 2024), MVPSA aggregates view-specific slot distributions (each a GMM) into a viewpoint-invariant content representation via a mixing-coefficient-weighted convex combination. The authors prove four theoretical results — identifiability (up to affine equivalence), viewpoint invariance, and approximate representational equivariance of the aggregate content — and validate empirically on CLEVR, GQN, and two newly proposed large-scale multi-view datasets (MV-MoViC/D).

---

## Strengths

- **Viewpoint-agnostic multi-view OCL with identifiability guarantees.** Prior multi-view OCL work (e.g., MULMON/Li et al. 2020) requires ground-truth camera conditioning at inference. MVPSA infers viewpoint information from the data itself and provides formal identifiability guarantees — a combination that has not been achieved before in multi-view OCL, as the related work survey confirms.

- **Principled occlusion handling via mixing coefficients.** The content aggregator (Eq. 5–6) uses view-specific mixing coefficients π̃ to weight slots; occluded objects receive near-zero weight in views where they are absent, so their representation is driven by views where they appear. This mechanism directly addresses the viewpoint-sufficiency assumption and is cleanly motivated by Example 1.

- **Compelling 2D synthetic identifiability verification.** Figure 3 (SMCC = 0.95 ± 0.01 across four independent training runs) and Figure 4 (view-invariant q(c) distributions recoverable up to affine transformations across three view pairs) provide unusually direct visual evidence for theoretical claims — stronger than the purely quantitative validation most OCL papers offer.

- **Release of MV-MoViC/D datasets.** The two new large-scale multi-view datasets, one of which (MV-MoViD) deliberately violates viewpoint sufficiency, are a tangible contribution for the community regardless of this paper's acceptance.

- **Strong identifiability metric gains over baselines.** On CLEVR-MV, MVPSA-SMCC = 0.67 vs. SA = 0.47 and MULMON = 0.61 (with MULMON having access to ground-truth cameras — an advantage MVPSA does not). INV-SMCC (unique to MVPSA) = 0.66. These gains are substantial and consistent across multiple benchmarks.

---

## Weaknesses

### Fatal
None identified.

### Major

- **No improvement in object segmentation (mBO) despite identifiability gains.** Table 2 is striking: MVPSA-MLP achieves mBO = 0.28 ± 0.021, identical to SA-MLP (0.28 ± 0.091). MVPSA-Transformer (0.38 ± 0.008) is only marginally above PSA-Transformer (0.37 ± 0.021) and within error bars of SA-Transformer (0.34). Despite identifiability metrics improving dramatically, downstream segmentation quality does not follow. The paper does not address this disconnect, which calls into question the practical utility of the identifiability guarantees and is a fundamental omission.

- **The specific theoretical role of multi-view aggregation in enabling identifiability is never articulated.** Theorem 2 (affine equivalence of aggregate content) follows the structure of Kori et al. (2024) (Kivva et al., 2022) — GMM prior → non-degenerate posterior → invertibility → affine constraint. The proof sketch gives no indication of how or whether multi-view aggregation is *theoretically necessary* for identifiability, vs. whether the same result follows from the GMM prior structure alone (as in single-view PSA). If the multi-view setting is not essential for the theoretical guarantee, the framing of the paper's core contribution requires reconsideration.

- **Absence of standard segmentation metrics (ARI, FG-ARI) on benchmark datasets.** Table 1 reports only SMCC, INV-SMCC, and MCC. Standard OCL benchmarks use ARI or FG-ARI to assess object discovery quality, and many downstream readers will wish to compare MVPSA against the broader OCL literature on those metrics. Without them, it is impossible to situate this work in the standard OCL evaluation ecosystem.

- **The ELBO derivation (Eq. 9–10) underexplains the treatment of content c.** The bound integrates over both q(v|x) and p(c), but p(c) is the aggregate posterior q(c) which is itself constructed from the same data. The relationship between "sampling c from p(c)" at training time and the variational inference over c is never made precise. It appears there is no KL term for c because p(c) = q(c) by construction (optimal prior design), but this is not stated explicitly in the ELBO derivation. The resulting ELBO (Eq. 10) has only a reconstruction term and a KL on v, which looks unusual without explanation.

### Minor

- **High variance in Theorem 3 empirical evidence.** The SMCC for viewpoint invariance (Theorem 3) is 0.87 ± 0.11, with a standard deviation that is an order of magnitude larger than the identifiability result (0.95 ± 0.01). This inconsistency across different view pair choices is not investigated, yet robustness to the choice of viewpoint subset is a core theoretical claim.

- **GQN SMCC is lower than MULMON.** On GQN, MVPSA achieves SMCC = 0.59 ± 0.01 vs. MULMON = 0.61 ± 0.03. The paper does not remark on this reversal. Given that MULMON uses ground-truth camera conditioning (an advantage), this is worth noting, but the unanswered question is why the invariant content representation appears to help less on GQN.

- **Hungarian matching introduces view-ordering asymmetry.** Using viewpoint 1 as the base representation for Hungarian matching is arbitrary. In principle, results could depend on which view is designated first. This is never discussed, and no ablation varies the base choice. Standard practice would at least note that results were validated to be stable across different base-view selections.

- **Proof sketches are too terse to assess correctness.** Theorem 4's proof sketch ("we follow the steps in Theorem 3, over view distribution p(v) but for fixed content vector c") contains essentially no insight. While full proofs in an appendix are standard, the sketches should at minimum convey the key novel step. Reviewers cannot assess correctness from these sketches.

- **Figure 5 x-axis is mislabeled or ambiguous.** The caption reads "Influence of Number of viewpoints" and the x-axis shows "100, 200, 300." Having 100–300 distinct camera viewpoints per scene is implausible; these almost certainly refer to the number of training scenes. This should be corrected.

- **Theorem 3 "invariance" language is imprecise.** The theorem establishes f_A ~_s f_B, i.e., invariance up to permutation + affine. The intuition section says "content is invariant to viewpoints," which overstates the result. This distinction matters: the representation is invariant only within an equivalence class, not in an absolute sense.

### Tiny

- **Algorithm 1 is referenced in the main text but only present in the appendix.** A brief inline pseudocode or reference to Section/App. X would help readers.
- **Notation inconsistency.** The paper switches between |V| (as a cardinality) and V (as a count), and between set-notation A = {1, 2, 3} and interval notation A ⊆ [V] in a few places. A notation table (which the paper says is in App. A) should be referenced earlier.

---

## Nice-to-Haves

- **Ablation on number of views V.** Figure 5 shows the effect of viewpoint count (though the x-axis label is ambiguous), but does not systematically ablate V = 1, 2, 3, 4 views to show where the multi-view benefit saturates.
- **Reconstruction quality metrics (MSE/LPIPS/FID).** Since MVPSA is a generative model, a brief report of perceptual quality alongside identifiability would help characterize the trade-offs.
- **Visualization of failure cases under Assumption 1 violations.** The MV-MoViD results are deferred to the appendix; a brief in-paper discussion with at least one qualitative failure case (e.g., slot collapse when an object is never visible) would strengthen the limitations section considerably.
- **Computational overhead relative to PSA.** A wall-clock or FLOP comparison of MVPSA vs. single-view PSA would help practitioners assess scalability.
- **Comparison to newer multi-view contrastive/self-supervised OCL baselines.** MULMON (2020) is the only multi-view OCL baseline; more recent self-supervised multi-view methods, if they exist, would strengthen the comparison.
- **Clarification of inference procedure.** It would be helpful to explicitly state whether, at test time on a new scene, the model requires multiple views simultaneously or whether it can operate on a single view using the learned prior p(c).

---

## Removed Points

*These points were flagged for removal — treat them with caution.*

- **Critic: Eq. 6 variance formula is incorrect / a lower bound.** Eq. 6 computes Var(c_k) = Σ_v (w_v)² · Var(s̃_k^v), where w_v = π̃_k^v / Σ_v π̃_k^v. This is the correct variance formula for a *weighted linear combination* of independent Gaussian random variables (not a Gaussian mixture). The aggregator g defines c_k as a weighted sum of slot random variables, not a mixture thereof. The critic appears to conflate mixture-variance with linear-combination-variance. **Removed as factually incorrect criticism.**

- **Critic: MULMON comparison is unfair.** MULMON requires ground-truth camera conditioning while MVPSA does not. This asymmetry favors MULMON, not the authors' method — making any near-parity result stronger evidence for MVPSA, not weaker. Removed per rules on comparisons that are asymmetric in favor of the baseline.

- **Critic: Missing baselines (SIMONe, ObSuRF, uORF).** The paper's primary evaluation is on identifiability metrics (SMCC/MCC) for which these methods are not designed. Comparing on segmentation or reconstruction metrics against these baselines without identifiability measures would be an apples-to-oranges comparison. The paper explicitly situates itself as a theoretical-identifiability contribution. Removed as scope-creep.

- **Critic: Abstract overstates novelty relative to Brady et al., Lachapelle et al., Kori et al.** The abstract says "setting it apart from prior work focusing on single-view settings and lacking theoretical foundations." Reading in context, this refers to the multi-view OCL literature (MULMON et al.), not the identifiability OCL literature. The single-view identifiability papers are properly cited in the introduction as related work. The claim is accurate: no prior work provides identifiability guarantees specifically for multi-view OCL. Removed as misreading.

- **Critic: Theorem 4 dimensionality constraint is geometrically odd.** Remark 1 clarifies that the paper is not claiming viewpoint equivariance but that the transformation H_v lies in the same subspace as input transformations H_x. This is further supported by Remark 2's homography motivation. The concern is addressed directly in the paper. Removed.

---

## Novel Insights

The most genuinely interesting theoretical observation — worth the community's attention — is the combination of the "optimal prior by design" principle (p(c) = q(c) through aggregate posterior, following Hoffman & Johnson 2016) with the multi-view setting: unlike VAEs that impose a prior as a regularizer, MVPSA's content prior *emerges* from the data through aggregation and is guaranteed to be a non-degenerate GMM regardless of decoder architecture. This architecture-agnostic identifiability is notable. However, the reviews collectively expose a gap that the authors should address: whether the multi-view aggregation is *theoretically load-bearing* for identifiability, or whether identifiability already follows from the GMM prior structure in the single-view setting. If the latter, the paper's core claim needs sharpening. The empirical observation that identifiability metrics and segmentation metrics appear to be decoupled (large SMCC gains, near-zero mBO gains) is also a substantive finding that the paper leaves unexplained.

---

## Suggestions

1. **Demonstrate why multi-view aggregation is theoretically necessary for identifiability** (or not). A brief theoretical comparison to single-view PSA — explaining what specifically breaks without aggregation — is essential to establish the paper's theoretical novelty on solid ground.

2. **Add ARI/FG-ARI to Table 1** so that MVPSA can be evaluated on the same metrics as the broader OCL literature.

3. **Investigate and explain the mBO–SMCC disconnect.** Either show conditions under which identifiable representations do improve segmentation, or explicitly discuss why the improvement in SMCC does not propagate to mBO. This is the most important empirical gap.

4. **Clarify the ELBO derivation.** Explicitly state that no KL term appears for c because p(c) = q(c) by the optimal prior construction, and explain how c is sampled at training time (i.e., from the stored aggregate GMM from previous iterations or the current batch).

5. **Include at least a brief in-paper treatment of the MV-MoViD (Assumption 1 violation) results** — a single-row or one-line qualitative observation — rather than deferring entirely to the appendix, given this directly tests the paper's main practical limitation.

6. **Fix Figure 5's axis label** to unambiguously describe whether "100, 200, 300" refers to training scenes or camera viewpoints.

---

**Overall assessment:** The paper addresses a genuine and under-explored gap — multi-view identifiable OCL without camera supervision — with a clean and well-motivated model. The theoretical framework is novel and the empirical verification of identifiability claims on 2D synthetic data is convincing. However, the paper has two significant unresolved issues: (1) the lack of clarity on whether multi-view aggregation is theoretically necessary for identifiability (the proof sketches do not illuminate this), and (2) the failure of identifiability gains to translate into segmentation quality improvements. These issues do not invalidate the contribution but leave the paper's significance uncertain until addressed. The current submission is below the ICLR acceptance bar, but the gap is closable with targeted revisions.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0, 5.0]
Average score: 3.5
Binary outcome: Reject
