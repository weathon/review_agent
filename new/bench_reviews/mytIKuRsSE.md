Now I have all the information I need. Let me compose the final review.

## Summary

The paper introduces the Dual-level Noisy Correspondence (DNC) problem for Multi-Modal Entity Alignment (MMEA), which captures misalignments in both intra-entity (entity-attribute) and inter-graph (entity-entity and attribute-attribute) correspondences. To address DNC, the proposed RULE framework estimates correspondence reliability via a two-fold principle (uncertainty + consensus), divides pairs into three subsets with tailored training strategies, and leverages inter-graph reliability for robust intra-entity attribute fusion. A test-time correspondence reasoning (TTR) module using an MLLM further enhances inference.

## Strengths

- **Novel and well-motivated problem formulation (DNC)**: The identification of dual-level noisy correspondence in MMEA is genuinely new and practically grounded. The paper reports that real-world ICEWS benchmarks contain over 50% noisy correspondence (Appendix B), and Tables 1–2 systematically show degradation of all prior methods under increasing noise, confirming the problem's significance.

- **Principled two-fold reliability estimation with theoretical justification**: The combination of uncertainty (Dempster-Shafer Theory, Eq. 2–3) and consensus (Eq. 5) is well-justified by Theorem 1, which proves that low uncertainty alone does not guarantee correct correspondence. This is not merely heuristic—it provides a formal motivation for the dual principle. Fig. 3(b) confirms the combined metric effectively separates clean and noisy pairs.

- **Tailored tri-partite treatment of inter-graph correspondences**: Dividing pairs into S_U (excluded), S_I (soft-labeled via Eq. 12), and S_C (standard loss, Eq. 11) is a principled adaptation of noisy-label techniques to this structured setting. Table 3 ablation confirms both uncertainty-only and consensus-only variants underperform the full DRL, validating the complementary design.

- **Consistent and substantial empirical improvements in Non-name settings**: Even accounting for TTR's modest contribution (~1.7 H@1 under 50% DNC Non-name), the core RULE framework achieves large margins over baselines. Under 50% DNC Non-name, w/o TTR (56.5 H@1) still outperforms MEAformer (42.4) by 14.1 points on ICEWS-WIKI. Under Inherent DNC Non-name, the margin is 10.7 points (64.2 vs 53.5).

- **Elegant cross-level synergy (Dually Robust Fusion)**: Eq. 14 re-uses inter-graph reliability estimates to weight attribute contributions during fusion, exploiting the formal insight from Section 2.1 that incorrect entity-attribute pairs necessarily produce incorrect attribute-attribute pairs. Fig. 5 qualitatively confirms noisy attributes receive low reliability scores.

- **Reproducibility**: Code is available, and key hyperparameters (λ = 1e-4, β = 0.3, τ = 0.07) are fixed across all experiments, reducing overfitting risk.

## Weaknesses

### Fatal
None.

### Major

- **TTR module introduces an unshared test-time advantage that partially confounds headline improvements**: The TTR module (Section 2.5) uses Qwen2.5-VL-72B-Instruct at inference time, giving RULE a capacity advantage no baseline enjoys. The ablation (Table 3) is conducted only under 50% DNC on ICEWS-WIKI. Under 50% DNC All-attributes, TTR contributes ~3.7 H@1 points (94.0 → 97.7), meaning w/o TTR (94.0) still beats MEAformer (91.9) by only 2.1 points—a much smaller margin than the headline 5.8 points. Crucially, the paper does not provide the w/o TTR ablation under Inherent DNC (the most commonly reported setting), where RULE leads MEAformer by 3.0 H@1 (98.9 vs 95.9) in All-attributes. If TTR adds a similar ~3.7 points under Inherent DNC, the core method might slightly underperform MEAformer in this setting. The absence of this ablation is a significant gap. While the Non-name results are robust even without TTR, the All-attributes improvements are substantially conflated with MLLM capacity.

- **Attribute-attribute noise injection conflates feature corruption with correspondence noise**: The paper defines DNC as *misaligned correspondences* (wrong pairings), but the attribute-attribute noise injection adds Gaussian noise to images and random character replacements to text (Section 3.1). This is feature-level corruption, not correspondence swapping. A genuine attribute-attribute NC would swap which entity an attribute belongs to across graphs, matching the paper's own formal definition (y^m_{ij} = 1 iff h^m_i = 1 & h̃^m_j = 1 & y_{ij} = 1). Feature corruption and correspondence noise are fundamentally different failure modes: Gaussian noise uniformly reduces evidence without creating false high-similarity pairs, making it potentially easier for the reliability estimator to detect. The entity-entity and entity-attribute noise injections *are* genuine correspondence noise, so the core method is partially validated, but the claim of robustness to "dual-level" NC is not cleanly established for the attribute-attribute level.

### Minor

- **Potential confirmation bias in S_TP-based threshold estimation**: Equation 8 computes thresholds using S_TP = {i | argmax(s_i) = argmax(y_i)}, where y_i is the potentially noisy label. If the model begins fitting to a noisy pair, it enters S_TP and shifts thresholds to treat it as clean, creating a feedback loop. The paper does not discuss this risk or provide training dynamics analysis. Fig. 3(b) shows clean separation at one training point but does not demonstrate stability across training. However, the β margin in Eq. 8 provides some protection, and this is a standard concern in self-training methods.

- **Ablation only under 50% DNC**: Table 3 reports ablations only under the highest noise setting (50% DNC) on a single dataset (ICEWS-WIKI). Providing the ablation under Inherent DNC and across more datasets would strengthen the analysis, particularly for understanding TTR's contribution at lower noise levels.

### Trivial
None.

## Nice-to-Haves

- Providing w/o TTR results under Inherent DNC and on additional datasets would clarify how much of the All-attributes improvement comes from the core method vs. the MLLM.
- Replacing the attribute-attribute feature corruption with genuine attribute-attribute correspondence swapping (e.g., exchanging which entity an attribute is associated with across graphs) would provide a cleaner evaluation of DNC robustness.
- Tracking how S_TP composition and thresholds evolve across training epochs would address the confirmation bias concern empirically.
- Reporting inference cost/latency of the 72B TTR module would inform practical applicability.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"RULE without TTR achieves 94.0 H@1 on ICEWS-WIKI All-attributes (Inherent DNC), which underperforms MEAformer's 95.9"** (Harsh Critic Critical Issue 1): This is factually wrong. The ablation in Table 3 is under 50% DNC, not Inherent DNC. Under 50% DNC All-attributes, RULE w/o TTR (94.0) still beats MEAformer (91.9). The critic misidentified the noise level of the ablation.

- **"The comparison is structurally unfair; baselines should receive the same MLLM enhancement"** (Harsh Critic): While the broader TTR fairness concern is valid (and kept above as a Major weakness), demanding baselines receive the *same* MLLM enhancement goes too far. The paper provides the w/o TTR ablation so readers can isolate the core method's contribution. The issue is that the Inherent DNC w/o TTR ablation is missing, not that baselines need MLLM augmentation.

- **"No standard deviations or multiple runs reported"** (Harsh Critic Section Notes): This is a common practice in this research area and is not a meaningful weakness for the evaluation.

- **"Computational cost of TTR never discussed"** (Harsh Critic Section Notes): Moved to Nice-to-Have. Practical concern but not a core methodological flaw.

- **"Assumption 1 is unproven / no empirical validation"** (Harsh Critic Section 2.2.2): Assumption 1 (correct attributes have non-negative marginal contribution) is a reasonable inductive bias that underpins the greedy strategy for estimating correspondence during inference. While empirical validation would strengthen the paper, the assumption is plausible and the method works in practice.

- **"Figure 1(b) provides only illustrative evidence, not controlled experiments"** (Harsh Critic): The systematic degradation across all baselines in Tables 1–2 under increasing noise IS the controlled experiment demonstrating DNC's impact. Figure 1(b) is motivation, not the sole evidence.

- **"The paper claims TTR is 'one of the first methods to enhance test-time robustness for MMEA' but the CoT formulation in Eq. 16 is essentially pseudocode"**: Eq. 16 formally specifies the computation; the CoT notation indicates the reasoning process. The formulation is adequate for a methods section, with details in appendices.

- **Strength Finder's "Self-adaptive threshold mechanism" as a separate strength**: This is subsumed by the tri-partite treatment strength. Removed to avoid redundancy.

- **Strength Finder's "Test-time Correspondence Reasoning (TTR) for enhanced inference" as a separate strength**: Given that TTR is identified as a Major weakness (unfair advantage), listing it as an independent strength would conflict. The TTR *idea* is interesting but its evaluation confounds the results.

## Novel Insights

The most insightful observation across the reviews is the asymmetry in TTR's contribution across settings: TTR adds ~3.7 H@1 in All-attributes but only ~1.7 H@1 in Non-name (Table 3). This suggests the MLLM primarily helps when entity names are available (serving as strong textual anchors for the MLLM's reasoning), but provides limited benefit in the more challenging Non-name setting where the core method's contribution is clearest. This asymmetry actually strengthens the case for the core DRL+DRF framework while raising questions about TTR's marginal value-cost tradeoff.

## Suggestions

- Provide the w/o TTR ablation under Inherent DNC on at least ICEWS-WIKI and one additional dataset. This single number would resolve the most significant ambiguity about how much of the All-attributes improvement is due to the core method versus the 72B MLLM.
- Replace or supplement the Gaussian noise / character replacement for attribute-attribute NC with genuine correspondence swapping (randomly reassigning attributes across entities in the paired graph), which directly matches the DNC definition in Section 2.1.
- Report the Non-name results more prominently, as they provide the cleanest evidence of the core method's contribution and are arguably the more challenging and informative setting.

## Score and Decision

**Calibration anchors used:**

| Paper | Avg Score | Decision | Comparison |
|-------|-----------|----------|------------|
| Norton (noisy correspondence, video-language) | 8.0 | Accept Oral | Stronger: cleaner methodology, no fairness concerns |
| Test-time Adaptation for Cross-modal Retrieval | 7.5 | Accept Spotlight | Stronger: test-time method with fair evaluation |
| PCME++ (probabilistic cross-modal, noisy correspondence) | 6.67 | Accept Poster | Comparable: probabilistic approach to noisy correspondence, but more incremental |
| SGN (unified noisy label learning) | 6.0 | Accept Poster | Comparable: robust learning under noise, similar novelty level |
| CoTnPoT (inference compute unfair comparison) | 5.0 | Reject | Weaker than this paper: TTR is a smaller portion of total contribution, and core method works without it |
| Unified Robust Contrastive Loss (noisy labels) | 4.4 | Reject | Weaker: limited empirical support |
| FreeLM (unfair comparison to GPT-3) | 2.0 | Reject | Much weaker: fundamentally misleading comparison |

This paper is clearly above the low-scoring anchors (FreeLM, FV-NeRV) because the core method (DRL+DRF) genuinely works even without TTR, especially in Non-name settings. It is below the high-scoring anchors (Norton, Test-time Adaptation) due to the TTR fairness concern and the attribute-attribute noise injection conflation. It sits in the range of the medium anchors: comparable to SGN (6.0) and slightly below PCME++ (6.67). The DNC problem formulation is more novel than SGN's contribution, but the TTR fairness concern is a real drag. A score of 6.0 reflects genuine contributions with meaningful but non-fatal weaknesses.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>