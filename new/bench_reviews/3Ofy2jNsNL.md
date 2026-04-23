Now I have all the information I need. Let me compose the final review.

## Summary

ACT-IN-LLM proposes adaptively compressing vision tokens within LLM layers by retaining all query tokens while compressing key/value tokens in self-attention, addressing the information loss problem of Pre-LLM and Early-LLM compression methods. The paper provides strong motivating evidence that early token compression is harmful (especially for high-resolution tasks), a clean unified formulation for comparing compression strategies, and demonstrates consistent improvements (+5.5% over FastV on high-resolution benchmarks) across multiple model scales.

## Strengths

- **Strong motivational analysis (Figure 2):** The layer-wise token dropping experiment (Fig. 2a) shows up to ~15% degradation on high-resolution benchmarks when dropping tokens in early layers, and the attention heatmap visualization (Fig. 2b) demonstrates that token importance shifts across layers. This provides clear, direct evidence that early compression is problematic — a genuine empirical contribution.

- **Clean unified formulation (Section 4.1, Eq. 9):** The Com(C^Q, C^K, C^V) framework elegantly encompasses Full Token, Pre-LLM/Early-LLM, FlexAttention, and ACT-IN-LLM within a single notation, making the structural comparison transparent.

- **Consistent empirical improvements across scales:** Table 2 shows ACT-IN-LLM achieves 45.4% average on high-resolution benchmarks vs. 39.9% for FastV (+5.5% absolute), and even without training (43.5%) it outperforms all trained compression baselines. Figure 7 shows gains are consistent across 0.5B→3B→7B model sizes and varying SFT data.

- **The in-LLM K/V compression strategy is well-motivated and effective:** Table 4b shows that all in-LLM methods (including simple AvgPool-1D at 45.08%) dramatically outperform Pre-LLM approaches (39.15%), confirming that the compression location matters more than the specific mechanism. The hierarchical ratio design is well-supported by the ablation in Table 4a.

- **The "Ours w/o train" result (Table 2):** Achieving 43.5% without any training (vs. 38.7% for FastV w/o train and 39.9% for trained FastV) is a strong result, showing the architectural advantage is genuine and not just an artifact of training.

## Weaknesses

### Fatal
None.

### Major

- **The headline "6.3% improvement" claim is numerically inaccurate.** The abstract claims "a 6.3% improvement over existing token compression techniques," the introduction states "6.2%," but the main comparison (Table 2) shows only a +5.5% absolute gap (45.4% vs. 39.9% on high-resolution benchmarks). No computation in the paper yields 6.3%. This is the paper's central empirical claim and it does not match the reported results. (Abstract, line 15; Introduction, line 47; Table 2)

- **The "text-guided" selection mechanism — the paper's primary algorithmic novelty — provides negligible improvement over simple average pooling.** Table 4b shows attention-weight selection achieves 45.35% vs. AvgPool-1D at 45.08% on high-resolution benchmarks (0.27% gap) and 75.04 vs. 75.06 on general benchmarks (AvgPool-1D actually wins). This means the core algorithmic contribution of ACM is essentially a rounding error. The paper's narrative centers on "text-guided information extraction" (Eq. 4, Section 3.2) as the key innovation, but the ablation undermines this. The real contribution is the in-LLM K/V compression *strategy* (where to compress), not the *mechanism* (how to compress). The paper should explicitly acknowledge this and reframe its claims accordingly.

- **The theoretical analysis (Theorems 2–3) does not connect to the actual method.** The theorems provide existence proofs using random sampling constructions from randomized linear algebra (Theorem 2: "there exists matrices C^K and C^V..."). However, the actual ACM implementation uses deterministic top-k selection based on attention weights from the previous layer (Eq. 5). The theorems establish that *some* K/V compression can approximate full attention well, but not that *this specific top-k selection* does. The gap between existential random-sampling bounds and the greedy deterministic selection used in practice is neither discussed nor bounded. Theorem 3's structural comparison (Com(I,C^K,C^V) vs. Com(C^Q,C^K,C^V)) remains valid in principle, but the specific approximation quality of the method used is unproven. (Sections 4.2–4.3)

### Minor

- **The "error correction mechanism" framing is imprecise and unsupported.** The paper claims (Section 3.2, Conclusion) that retaining all query tokens provides "an inherent error correction mechanism that mitigates the permanent loss of valuable information." The valid architectural insight is that all tokens are preserved in the residual stream (H_{i+1} = H_i + ...), so subsequent layers can access different K/V subsets. However, calling this "error correction" implies an active recovery process, when it is simply information preservation through residual connections. No experiment isolates this mechanism (e.g., by comparing Q-retention vs. full Q/K/V compression at matched token counts). The claim should be rephrased as "information preservation" rather than "error correction." (Section 3.2, line 97; Conclusion, line 420)

- **No justification for using the last token's attention row (Eq. 4).** The text-guided compression uses A_{i-1}[N+L, :], the attention weights of the last token, as the proxy for "text-guided importance." No justification or ablation is provided for why the last token specifically, nor are alternatives (e.g., averaging over all text tokens, or using a task-specific token) considered. Given that the last token's attention pattern may not reflect task-relevant visual importance, this choice deserves discussion. (Section 3.2, Eq. 4)

- **Efficiency gains are modest and slightly overstated.** Memory reduction is only 5.5% (19.9→18.8 GB) and time reduction is ~17% (621→515 ms), while the abstract claims "~20%" time reduction. Since all queries are retained, FFN layers still operate on the full token sequence; savings come only from reduced attention computation. A breakdown of attention vs. FFN costs would clarify the practical efficiency profile. (Table 2, Abstract)

- **Scaling experiments (Figure 7) have only 3 data points per curve with no error bars or variance**, making it impossible to assess statistical significance of the scaling trends.

### Trivial
None.

## Nice-to-Haves

- An ablation comparing Q-retention (ACT-IN-LLM's approach) vs. compressing Q alongside K/V within the LLM at matched token counts would validate the "information preservation" claim and clarify the contribution of the Q-retention design choice.
- Qualitative examples showing what information ACT-IN-LLM preserves vs. what Pre-LLM methods lose on specific high-resolution tasks would make the information-preservation argument more concrete.
- Analysis of when/why AvgPool-1D matches attention-weight selection, and whether the gap widens on more complex tasks, would strengthen understanding of the method's operating regime.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "competitive performance with non-compression methods" is overclaimed.** The paper says "competitive performance" and Table 2 shows 45.4% vs. 48.0% (94.6% of full performance). The word "competitive" is reasonable for retaining ~95% of performance while saving significant compute. This is not overclaiming.

- **Harsh critic: The SOTA comparison (Table 3) is misleading because different models use different encoders/training.** The paper explicitly positions this as showing ACT-IN-LLM "achieves competitive performance compared with SOTA MLLMs" — this is a different claim than "our compression method beats their compression method." The comparison is meant to show efficiency, not a controlled compression experiment. The paper clearly reports token counts and data sizes for transparency. This is a scope-creep criticism.

- **Harsh critic: Assumption 1 may not hold for document/text-rich images.** The paper verifies Assumption 1 empirically in Fig. 5(b). While the assumption could be weaker for document images, the method still works well on DocVQA (45.2% vs. 38.6% for FastV), suggesting the assumption is not critically violated in practice.

- **Strength finder: "Text-guided compression using inter-layer attention signals" as a core strength.** This is removed because Table 4b shows this mechanism provides negligible improvement over AvgPool-1D. A mechanism that performs identically to simple pooling cannot be listed as a core strength.

- **Strength finder: "Theoretical justification that K/V compression yields a better low-rank approximation" as a core strength.** The theorems use random sampling constructions that don't match the actual top-k selection method. While Theorem 3's structural comparison is valid in principle, the specific approximation guarantee does not apply to the method as implemented, so calling this a "theoretical justification" overstates what the theory actually establishes.

## Novel Insights

The ablation results (Table 4b) reveal an important but underappreciated insight for the field: for in-LLM K/V compression, *where* you compress matters far more than *how* you compress. Simple average pooling within later LLM layers nearly matches attention-weight selection, both dramatically outperforming sophisticated Pre-LLM approaches. This suggests the field may be over-investing in clever token selection mechanisms when the primary gains come from the architectural decision of compressing K/V within the LLM rather than before it.

## Suggestions

- Reframe the contribution: lead with the in-LLM K/V compression *strategy* (which the evidence strongly supports) rather than the text-guided selection mechanism (which the evidence does not support). The honest framing is "compressing K/V within LLM layers is more effective than compressing before the LLM, and even simple pooling suffices for selection."
- Correct the headline improvement number to match Table 2 (+5.5% absolute on high-resolution benchmarks).
- Replace "error correction mechanism" with "information preservation through residual connections" to accurately describe the architectural property.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| FastGen (KV cache compression) | 8.0 | Much stronger: tight theory-experiment connection, clean method, plug-and-play. ACT-IN-LLM is clearly below this. |
| Transfusion (multi-modal model) | 8.0 | Far more novel paradigm, extensive controlled experiments. Not comparable on novelty. |
| Matryoshka Multimodal Models | 6.0 | Similar simplicity, similar "adaptive" overclaiming concern. ACT-IN-LLM has stronger motivation and more comprehensive experiments, but also more overclaiming. |
| Visual Attention Sink | 5.25 | Interesting finding but simpler method. ACT-IN-LLM has more substance. |
| MiKV (mixed-precision KV) | 5.0 | Limited novelty, marginal improvements, rejected. ACT-IN-LLM is clearly better. |
| PyramidDrop (layer-wise LVLM dropping) | 3.0 | Very similar motivation but drops tokens entirely, incomplete experiments. ACT-IN-LLM is significantly better. |
| EfficientSkip | 2.5 | Limited novelty, limited experiments. ACT-IN-LLM is much stronger. |

ACT-IN-LLM sits above the borderline papers (Matryoshka at 6.0, Visual Attention Sink at 5.25) in terms of empirical substance and motivation, but below them in honesty of claims. The paper makes a real contribution (in-LLM K/V compression strategy), but overclaims on the 6.3% number, the "error correction" mechanism, and the importance of text-guided selection. The most damaging issue is that the core algorithmic novelty (attention-weight selection) is empirically shown to be negligible in the paper's own ablation. This significantly reduces the contribution's depth. The paper is comparable to Matryoshka (6.0) but with more overclaiming, placing it slightly below.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>