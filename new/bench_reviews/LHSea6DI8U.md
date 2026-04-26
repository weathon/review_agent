Now I have enough information. Let me write the final consolidated review.

## Summary

STBP proposes a continual spatio-temporal forecasting framework that pairs a frequency-domain backbone (FreNet) with a dual-stream linear graph attention (DLGA) module and a scalable contextual pattern bank. The backbone learns general spatio-temporal representations from the first incremental period and is then frozen, while the pattern bank expands incrementally via parameter addition and guides the frozen backbone through prompt-based gating and attention, addressing catastrophic forgetting and distributional drift in streaming graph scenarios.

## Strengths

- **Strong empirical improvements on traffic datasets**: STBP reduces average MAE by 21.44% and 21.93% over the best baseline (EAC) on PEMS-Stream and CA-Stream respectively (Table 1), with consistent improvements across all metrics and horizons. Few-shot results (Table 2) further confirm robustness under data scarcity.
- **Well-motivated and coherent architecture**: The separation of concerns—backbone for stable general knowledge, pattern bank for adaptive context—is a clean design that aligns with prompt-tuning literature and the continual learning stability-plasticity tradeoff. The dual-stream attention mechanism (Eq. 7–9) that incorporates pattern bank entries as additional keys is a conceptually clear way to unify spatial interaction with pattern-bank-guided adaptation.
- **Comprehensive baseline comparisons**: The paper compares against 5 dedicated CSTF baselines plus 3 conventional STGNN methods, making the competitive landscape thorough and informative.
- **Pattern bank interpretability**: The t-SNE visualizations (Figures 3, 6) demonstrate that the pattern bank autonomously discovers meaningful clusters without explicit clustering objectives, and new nodes from later periods are correctly assigned to existing clusters.

## Weaknesses

### Fatal

None.

### Major

- **Unclear specification of the linear attention mechanism undermines the O(N) complexity claim**: The paper states that φ(·) is "a random feature mapping, with Softmax used for approximation in our implementation" (Sec. 4.3, after Eq. 9). This is ambiguous: softmax is not a valid random feature mapping, and no standard linear attention method uses softmax as φ. If the implementation computes the full softmax attention in Eq. 8 (O(N²)), the claimed linear complexity of DLGA (a key selling point) does not hold. If some other φ is used, the approximation quality relative to softmax must be verified. The paper references Appendix A.3.1 for derivation details, but the main text must specify the actual feature map used and demonstrate the approximation is reasonable. Without this clarity, both the scalability narrative (Section 5.5) and the architectural novelty of DLGA are weakened.

- **No isolated ablation for FreNet, a core claimed contribution**: The paper claims FreNet "extracts stable low-frequency components...resilient to distributional drift" as a key contribution. Yet the ablation study only removes DLGA ("w/o DLGA") and replaces the entire backbone ("w/o Backbone"), with no variant that replaces FreNet alone with a standard temporal encoder (e.g., linear projection, 1D CNN) while keeping DLGA. Without this ablation, it is impossible to determine whether FreNet's frequency-domain processing specifically addresses distribution drift as claimed, or whether the backbone's strong performance comes primarily from DLGA.

### Minor

- **Performance gains are much smaller on AIR-Stream than traffic datasets**: STBP achieves 21%+ MAE reductions on PEMS-Stream and CA-Stream but only 2.35% on AIR-Stream, where most CSTF baselines cluster closely together. The paper attributes gains to "capturing dynamic spatio-temporal correlations" and "mitigating distributional drift," but does not analyze why these mechanisms are less effective for air quality data. Discussing how dataset characteristics (graph expansion rate, shift magnitude, signal characteristics) interact with the method's assumptions would strengthen the generality claim.

- **The backbone is frozen after only the first incremental period with no ablation of this strategy**: While this is consistent with prompt-based continual learning (EAC and STRAP also freeze backbones), the paper does not study how sensitive performance is to this design choice. An ablation varying the number of periods before freezing would help validate whether the method is robust to the quality of the first period's data.

- **Definition 1 assumes purely additive graph evolution (G_τ = G_{τ-1} + ΔG_τ)**: This excludes node removal or edge deletion that occurs in practice (e.g., sensor failures). While this is a reasonable scope limitation, it narrows the applicability claim.

### Trivial

- The t-SNE visualization in Figure 3 is presented as motivation for the pattern bank design, but since it uses post-hoc trained embeddings, it functions more as validation than motivation. This is a minor presentation issue.

## Nice-to-Haves

- An ablation that replaces FreNet with a simple temporal encoder (e.g., linear projection or 1D CNN) while keeping DLGA would directly test the claimed benefit of frequency-domain processing for distribution drift.
- A scaling experiment varying N (number of nodes) with wall-clock time profiling for DLGA would empirically verify the O(N) vs O(N²) claim and clarify the φ implementation.
- Analysis of what makes AIR-Stream different from traffic datasets (e.g., smaller graph expansion, different shift dynamics) would help delineate STBP's effective operating regime.

## Removed Points

- **Questioning the availability of cited models/benchmarks**: The reviewer raised concerns about verifying cited methods; however, all cited baselines (EAC, STRAP, PECPM, etc.) are assumed to exist per our guidelines.
- **Conventional methods are unfairly compared**: The critic noted that retraining from scratch per period (GWNet, STID) is a "severe handicap." This is the standard evaluation protocol in CSTF literature and is explicitly noted in the paper as following prior work. The asymmetry favors the baselines (it gives them a clean-slate advantage per period), not STBP, so per our rules this is not a valid criticism.
- **The t-SNE visualization constitutes "circular reasoning"**: Using post-hoc embeddings to illustrate that a design works is standard practice in visualization, not circular reasoning. It would only be circular if the visualization were used in lieu of quantitative evaluation, which it is not.
- **FreNet's claim of mitigating distribution drift is "unverifiable"**: While there's no isolated FreNet ablation (noted as a Major weakness above), the paper does provide theoretical motivation and cites prior work on frequency-domain stability. The claim is partially supported, just not fully isolated experimentally.
- **The "general" backbone cannot exploit known adjacency**: The paper explicitly defines "general" as adjacency-free and node-count-agnostic, which is a design choice for scalability, not a flaw. Ignoring known adjacency is the stated scope, not an oversight.
- **Formatting nitpicks and typos**: Removed per guidelines—these are parser artifacts.

## Novel Insights

The key insight this review surfaces is that STBP's architecture follows a "freeze backbone, adapt prompts" paradigm that is now well-established in continual learning (cf. EAC, STRAP, prompt-tuning literature). The genuine novelties are (1) the FreNet module for frequency-domain processing to handle distribution drift in a backbone designed for continual learning, and (2) the dual-stream linear attention that integrates pattern bank entries as keys. However, both of these contributions have significant verification gaps: the φ function in the claimed linear attention is unspecified, and FreNet lacks an isolated ablation. The 21%+ gains on traffic datasets are impressive, but the near-parity on AIR-Stream tempers the generality claim.

## Suggestions

- Explicitly state what φ(·) is in the implementation (e.g., is it ELU+1, ReLU, or an actual random feature map?) and verify its approximation quality relative to_softmax. If softmax is computed exactly, acknowledge the quadratic cost.
- Add a "w/o FreNet" ablation that replaces FreNet with a simple temporal encoder while keeping DLGA to isolate its contribution.
- Add a discussion section analyzing the dataset-specific factors that explain the performance gap between traffic and air quality datasets.

## Score and Decision

**Calibration reasoning**: I compared this paper against several anchors:

- **MISA** (avg 6.5): Prompt-based continual learning with 18-22% empirical improvements. STBP has similar or larger gains on traffic data but smaller gains overall (especially AIR-Stream). MISA's methodology is more rigorously verified. STBP is comparable but with more verification gaps.
- **TimeMixer++** (avg 8.0): Pattern-based time series forecasting. Much broader evaluation (8 tasks), but incremental novelty. STBP has a narrower scope but addresses a more challenging setting (continual learning with graph expansion).
- **Latte** (avg 3.5): Linear attention with mathematical inconsistencies. STBP has a similar but less severe issue (unclear φ, not provably wrong). STBP's empirical results are much stronger.
- **SKI-CL** (avg 5.33): Continual time series forecasting. Similar scope, weaker novelty, weaker experiments. STBP has stronger empirical results but similar methodology gaps.
- **PN-Train** (avg 6.0): Pattern neurons in urban forecasting, strong but narrow contribution.

STBP sits between a 5 and 6. The empirical results are strong on traffic, and the architecture design is coherent. However, the two major weaknesses (unclear linear attention implementation undermining the O(N) claim, and no isolated FreNet ablation) are significant verification gaps for a paper that claims both as core contributions. The unclear φ function is particularly concerning because the linear complexity claim is a central selling point. These issues are addressable in a rebuttal but currently weaken confidence in the claims.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>