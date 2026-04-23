## Summary

The paper proposes CMO, a circuit symbolic learning framework for efficient Logic Optimization (LO) in chip design. CMO addresses the "circuit symbolic generalization problem" — that symbolic regression functions fail to generalize to unseen circuits — by introducing the Graph Enhanced Symbolic Discovery (GESD) framework, which distills knowledge from a GNN teacher into MCTS-based symbolic tree search. A structural-semantic feature decomposition reduces 69 circuit features to 5 structural features plus boolean semantic features, making symbolic search tractable. Experiments demonstrate that CMO achieves generalization comparable to the GNN teacher while being hundreds of times faster on CPUs.

## Strengths

- **GESD effectively transfers GNN generalization to symbolic functions**: The ablation in Table 3 shows that removing GESD causes severe recall drops (e.g., Hyp: 0.99→0.67; Ethernet: 0.72→0.44), while CMO with GESD achieves recall comparable to or exceeding the GNN teacher on most circuits in Table 1. This directly validates the core technical contribution.

- **Massive inference efficiency gains with preserved generalization**: Table 4 demonstrates speedups of several hundred times over COG on CPU machines (e.g., Sixteen: 4.16s vs. 1377.66s; Hyp: 0.06s vs. 28.28s), while Table 1 shows comparable prediction recall. This is the key practical advantage for CPU-based industrial LO tools.

- **Structural-semantic feature decomposition is well-motivated**: Figure 1c shows structural (92.33%), semantic (90.90%), and default (91.93%) features yield comparable GNN accuracy, confirming separability. Table 3 ablation confirms the decomposition's contribution (e.g., Multiplier: 0.96 with SFD vs. 0.52 without SFD and GESD). Reducing 69 features to 5 structural features is a clever engineering choice that makes symbolic search tractable.

- **Rigorous leave-one-domain-out evaluation**: Section 5 describes 12 leave-one-out datasets where each circuit is held out as the test set, a challenging and realistic protocol for evaluating generalization to unseen circuits.

- **Practical deployment pipeline**: Section 4.3 describes compiling learned symbolic functions into shared objects for direct integration into LO tools, requiring no GPU at inference — critical for the CPU-based industrial setting.

## Weaknesses

### Fatal

None.

### Major

- **Factually incorrect 2.5× speedup claim attributed to wrong configuration and circuit**: The paper states in Section 5: "In particular, our CMO achieves 2.5× faster runtime on the very large-scale circuit Sixteen (about 13 hours)." This is incorrect on both counts. From Table 2: CMO-Mfs2(0.5) on Sixteen gives 78784.04/52001.27 ≈ **1.5×** speedup, not 2.5×. The actual 2.5× figure comes from **2CMO-Mfs2(0.3)** on **Hyp** (319.33/127.51 ≈ 2.5×), which is a different method variant running twice on a small circuit. Even 2CMO-Mfs2(0.3) on Sixteen gives only ~2.16×. This error is not a rounding issue — it is a nearly two-fold overstatement of the headline result, and it is repeated in the abstract ("achieving up to 2.5× faster runtime") and conclusion. The abstract's "up to" qualifier is borderline defensible if interpreted as the maximum across all configurations, but the body text's specific attribution to CMO on Sixteen is factually wrong. This undermines trust in the paper's primary practical claim.

- **Incomplete ablation does not fully isolate GESD from SFD**: Table 3 compares CMO vs CMO w/o GESD (both with SFD) and CMO w/o GESD vs CMO w/o GESD&SFD. The first comparison does demonstrate GESD's contribution when SFD is present, but there is no "GESD without SFD" condition. Additionally, DSR and SPL are shown in Figure 1b performing poorly on the *default* 69 features, but never evaluated on the same 5 decomposed features CMO uses. If a standard SR method given the same decomposed features performs comparably, the claimed contribution of GESD (versus feature engineering) would be significantly weakened. The paper mentions comparisons with five lightweight baselines in Appendix C.3, but these results are not in the main text.

### Minor

- **Overstated generalization claim over COG**: The abstract claims CMO "outperform[s] previous SOTA GPU-based...approaches in terms of...generalization capability." Table 1 shows CMO and COG are roughly comparable: CMO wins or ties on ~7/12 circuits, COG wins or ties on ~7/12 (with ties on Hyp, Multiplier, Sixteen). The paper would be more accurate to say "comparable generalization" rather than "outperform."

- **No variance reporting for stochastic MCTS**: MCTS is stochastic, and no standard deviations or multiple-run results are reported. While the large performance differences in Table 3 make this unlikely to change conclusions, reporting variance would strengthen reliability claims.

- **Asymmetric k values in online comparison (Figure 4) conflate prediction quality with inference speed**: The paper uses k=50% for CMO/COG and k=70% for Effisyn, justified by Effisyn needing more nodes for comparable QoR. While this is a fair practical deployment comparison, it makes Figure 4's "efficiency" comparison difficult to interpret in isolation. Table 4's raw inference times (which clearly favor CMO) provide a cleaner comparison.

### Trivial

None.

## Nice-to-Haves

- Include a "GESD without SFD" ablation condition to fully isolate each component's contribution.
- Evaluate standard SR baselines (DSR, SPL) on the same 5 decomposed features to test whether feature engineering alone suffices.
- Show 1–2 representative learned symbolic functions in the main text rather than deferring all to Table 16 in the appendix, to substantiate the interpretability claim directly.
- Correct the 2.5× speedup claim to accurately reflect which configuration and circuit it corresponds to.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Not yet released" / reproducibility concerns about COG**: The paper cites COG as an existing method; per the hard rules, if the paper cites it, it exists. Removed.

- **Missing appendix results (Table 16, Figure 8, Appendix C.3, etc.)**: The parser strips appendices from all papers; these sections exist in the original submission. Removed.

- **Missing hyperparameter η value in main text**: The penalty constant is described in Section 4.2 with its range (0,1), and detailed implementation is deferred to the appendix. This is a minor presentation choice, not a substantive gap. Removed as trivial nitpick.

- **Demand for confidence intervals / statistical significance for large-scale benchmarks**: Single-run evaluation is the norm in this domain; requesting it is a nice-to-have, not a weakness. Removed to Nice-to-Haves.

- **Concern about Ci1–Ci4 being from the same design family**: This is speculative and the paper explicitly identifies them as an industrial benchmark; there is no evidence they share an easy generalization structure. Removed as unsupported speculation.

- **Requests for k=30% and k=40% results for CMO-Mfs2 specifically**: The paper already provides 2CMO-Mfs2 results at these k values in Table 2. Removed as already partially addressed.

- **Claim that the paper doesn't discuss trade-offs between 2CMO-Mfs2 runtime vs. CMO-Mfs2**: The paper does discuss this (Section 5, Experiment 2, noting that 2CMO-Mfs2 with k=30% trades off some runtime for better QoR). Removed as already addressed.

## Novel Insights

The GESD framework's insight — that GNN outputs encode domain-invariant generalization information that symbolic functions cannot easily capture from node features alone — is compelling and well-validated. The choice of MSE over KL-divergence for distillation, justified by the observation of a simple nonlinear mapping between features and GNN output, is a non-obvious design choice that deserves more prominence in the main text. The paper's approach of treating the symbolic function discovery as a distillation problem rather than a pure regression problem is the key conceptual advance.

## Suggestions

- Correct the 2.5× speedup claim: either attribute it correctly to 2CMO-Mfs2(0.3) on Hyp, or report the actual CMO-Mfs2 speedup on Sixteen (~1.5×). The current formulation is a factual error that significantly undermines trust.
- Add a GESD-without-SFD ablation condition to complete the ablation study.
- Clarify the generalization claim: use "comparable generalization" instead of "outperform" when comparing to COG, since Table 1 shows roughly equal performance.

## Score and Decision

**Calibration anchors compared:**

| Anchor | Path | Avg Score | Comparison |
|--------|------|-----------|------------|
| LLM-SR (SR+LLM, strong experiments) | m2nmp8P5in | 8.0 | This paper has weaker novelty and more overclaimed results |
| MDLformer-guided SR (new SR objective) | ljAS7cPAU0 | 5.67 | This paper has stronger empirical results but a factual error in its headline claim |
| Physics-constrained Graph SR (GNN+MCTS) | Ia17iAtr0P | 5.33 | Very similar technical approach; this paper has better validation of GNN component but a worse overclaim |
| LightSeq (overclaimed speedup) | kC5i5X9xrn | 5.00 | Most similar pattern: novel technique with inflated speedup claim; this paper's overclaim is more severe (wrong circuit + wrong method variant) |
| V:N:M sparsity (speedup falls short) | gWHQQagPbN | 5.80 | Similar pattern of observed vs. claimed speedup gap; this paper's gap is larger |
| FlashSampling (questionable speedup) | V4Xs283LHH | 2.50 | Much weaker paper overall; this paper has far more substance |
| APPD minimax (unfounded speedup) | bEgDEyy2Yk | 1.0 | Fundamentally flawed; this paper is clearly above this |

The paper has genuine and substantial contributions: GESD is a well-validated technique, the inference efficiency gains are dramatic (Table 4), and the generalization results are strong. However, the factually incorrect 2.5× speedup claim — attributed to the wrong method variant on the wrong circuit — is a significant overstatement that undermines trust in the paper's headline result. The incomplete ablation is a secondary concern. Compared to LightSeq (5.0, similar overclaim pattern) and Physics-constrained Graph SR (5.33, similar technical approach), this paper sits in the same range. The overclaim is worse than LightSeq's, but the empirical validation of the core contribution (GESD) is stronger.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>