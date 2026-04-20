## Summary

This paper proposes injecting additive uniform noise into intermediate layer activations of LLMs as a complementary source of randomness to prediction-layer sampling for hallucination detection. The core insight—that intermediate perturbations can reorder token probabilities while temperature sampling preserves their ordering—is conceptually motivated and validated through a scatter-plot analysis (Figure 3) showing moderate correlation (r=0.67) between the two uncertainty signals. Experiments across 4 datasets, 3 uncertainty metrics, and 3 model architectures show consistent AUROC improvements, with the largest gains observed for the Answer Entropy metric on GSM8K (+5.40).

## Strengths

- **Genuine conceptual insight with direct empirical validation.** The paper identifies a theoretically grounded distinction between output-layer sampling (which preserves token likelihood ordering at any temperature) and activation-space perturbation (which can reverse it), and directly validates this complementarity with Figure 3's scatter plot showing r=0.67 between the two uncertainty sources. This is the paper's most important contribution.

- **Consistent AUROC improvements across diverse settings.** Table 3 shows that noise injection improves Answer Entropy AUROC on all four datasets (GSM8K: +5.40, CSQA: +1.76, TriviaQA: +1.26, ProntoQA: +1.59). Table 6 generalizes this to Llama2-7B (+1.71) and Mistral (+5.92). Table 7 confirms gains under Lexical Similarity and Semantic Entropy on TriviaQA.

- **Noise injection improves hallucination detection without degrading generation accuracy.** Table 2 shows accuracy on GSM8K increases from 34.95 to 36.32 with noise injection, and Table 5 similarly shows accuracy improvements across injection layers (35.07→36.65 for upper layers). This is a practically valuable property.

- **Answer Entropy metric is well-tailored for reasoning tasks.** Equation 4 targets the extracted final answer string rather than averaging over all tokens, addressing length-biased entropy on tasks with lengthy reasoning chains. This metric shows the largest gains from noise injection (Table 3).

- **Simple, inference-time-only method.** Algorithm 1 provides a clear, implementable procedure requiring no training or model modification (Section 3.4).

## Weaknesses

### Fatal
None.

### Major

- **Hyperparameter tuning conflates with evaluation on the GSM8K test set.** Section 3.1 states temperature T=0.8 is chosen because it "optimizes GSM8K accuracy within the set T={0.2, 0.5, 0.8, 1.0}" and Section 4.1 notes "we follow the setup of Section 3.1 and select the noise magnitude as α=0.05 based on GSM8K performance." Both the temperature and noise magnitude are chosen on GSM8K test set performance. While the paper acknowledges "In practice, the noise magnitude can be selected based on the validation set" (Section 3.4), the reported results use test-set-tuned hyperparameters, inflating the GSM8K gains and making cross-dataset generalizability claims difficult to fully assess.

- **Modest gains for most metrics and datasets undermine headline claims.** The abstract and introduction claim noise injection "significantly improve[s] detection accuracy," but Table 3 shows gains of only +0.20 to +1.08 AUROC points for Predictive and Normalized Entropy across CSQA, TriviaQA, and ProntoQA. The only substantial gain (+5.40 on GSM8K Answer Entropy) occurs where hyperparameters were directly tuned. The evidence supports incremental improvement in many settings rather than the strong claims made in the framing.

### Minor

- **No inference cost or latency analysis.** The paper shows AUROC improvements but provides no wall-clock time, memory footprint, or token throughput comparison between noise-enhanced and standard sampling. Given that activation perturbation requires custom inference kernels that may disrupt KV-caching optimizations, the practical viability for production use is unclear.

- **Layer-wise activation magnitude not characterized for noise justification.** While the ablation in Table 5 shows different optimal noise magnitudes for different layer ranges (α=0.05 for upper, 0.02 for middle, 0.01 for lower), and Table 6 shows model-specific tuning (Llama α=0.05 vs Mistral α=0.02), the paper does not report the mean/variance of activation norms per layer to justify why these fixed magnitudes are appropriate. A noise formulation scaled to local activation statistics would strengthen the claim of generalizability.

### Trivial
None.

## Nice-to-Haves

- **Provide concrete examples showing token distribution shifts pre/post perturbation.** 1–2 GSM8K examples demonstrating the claimed "order reversal" effect (Section 3.3) would empirically ground the mechanistic intuition rather than relying on correlation alone.

- **Report mean/standard deviation over multiple random seeds for AUROC deltas.** All primary results in Tables 3, 6, and 7 appear to be single-run. Adding seeds would strengthen confidence in the reported gains.

## Removed Points

These points are flagged to be removed, treat them with caution:

1. **Criticism: "Absence of Activation-Scale Normalization renders the method theoretically unsound and unreproducible."** The paper is empirical, not theoretical. The ablation studies in Tables 4–6 already demonstrate that α needs tuning per model, per layer range, and per temperature. This is standard for empirical hyperparameters, not a fundamental flaw. The claim of "unreproducibility" is incorrect — the method can be reproduced exactly as described for Llama2-13B with α=0.05 on layers 25–40.

2. **Criticism: "Hallucination labeling protocol is circular because labels are derived from generation agreement rather than external ground truth."** This misreads Section 3.2: "model responses to a question are considered as hallucinating if the majority of the K=5 generated answers are incorrect." "Incorrect" here refers to incorrectness relative to the dataset's ground-truth answers (GSM8K provides verifiable solutions). This is a standard majority-vote evaluation protocol, not circular reasoning.

3. **Criticism: "KV-cache disruption makes the method computationally unjustified for marginal AUROC gains."** This is speculative implementation concern. The paper does not make claims about KV-cache compatibility, and inference cost analysis is not standard practice in hallucination detection papers (cf. XJiN1VkgA0). Moved to Nice-to-Have.

4. **Criticism: "Upper-layer noise tolerance contradicts standard findings about lower layers capturing syntax and upper layers semantics."** The paper provides a reasonable empirical explanation — lower-layer perturbations propagate as errors through subsequent layers, while upper-layer perturbations have shorter propagation paths. This is not contradictory; it's a distinct, empirically supported observation.

5. **Criticism: "LayerNorm/RMSNorm dampens or amplifies noise unpredictably."** Speculative analysis not grounded in the paper's empirical findings. The results in Tables 4–6 already empirically map the effective noise ranges per model and layer.

6. **Criticism: "Moderate correlation (r=0.67) does not prove complementarity."** This misunderstands what complementarity means in this context. Two uncertainty signals are complementary precisely because they are correlated but not identical (r<1.0). The downstream AUROC gains (Table 3) demonstrate the practical benefit.

## Novel Insights

The paper's most novel contribution is the conceptual framing of hallucination detection randomness as operating along two distinct axes: prediction-layer sampling (which preserves the model's learned token likelihood ordering at any temperature) and intermediate-layer activation perturbation (which can fundamentally reorder token probabilities). This distinction is well-motivated in the introduction and validated in Figure 3, providing a principled reason why combining the two sources should yield better uncertainty signals than either alone. The observation that noise injection improves majority-vote accuracy alongside AUROC is also interesting — it suggests that wrong answers are less robust to activation perturbation, making them more detectable while also making correct answers more likely under majority vote. This observation is underdeveloped but worth pursuing.

## Suggestions

- **Clarify the hyperparameter tuning protocol.** Explicitly state that T and α are tuned on the test set for the reported results, and either (a) run a proper train/validation/test split or (b) acknowledge this as a limitation and frame GSM8K results as an upper bound.

- **Soften language regarding gain magnitude.** Replace "significantly improve[s] detection accuracy" in the abstract with more precise language acknowledging that gains range from marginal (+0.20–+1.08) for token-level metrics to substantial (+5.40) for the answer-level metric on the tuned dataset.

- **Add an inference cost analysis.** Even a single-table comparison of wall-clock time per generation with and without noise injection would help assess practical viability.

- **Characterize activation magnitudes.** Report the mean and standard deviation of MLP output norms for the relevant layers and models, to contextualize why α=0.05 is appropriate for Llama but α=0.02 is needed for Mistral.

## Score and Decision

**Calibration anchors:**

- **Low-scoring papers (<4):** GXzwq6waYb (3,3,3,8 — withdrawn) proposed semantic clustering for hallucination detection with stronger AUROC gains (up to 9.3%) but received low scores due to presentation and incremental contribution. ByLO7p0oCF (3,3,3 — withdrawn) had marginal 1–3% AUROC gains and weak trends. The paper under review has somewhat weaker empirical gains than GXzwq6waYb but is cleaner in methodology.

- **Medium-scoring papers (5–6):** XJiN1VkgA0 (6,6,6,6 — reject) was a well-written uncertainty quantification paper good empirical results across multiple models. tZk3LnvVtK (5,5,6,6,6 — reject) had moderate AUROC improvements but clarity issues. The paper under review is comparable to XJiN1VkgA0 in quality — clean experiments but modest gains that don't quite break acceptance.

- **High-scoring papers (7+):** zMoNrajk2X (8,8,8 — accept spotlight) had strong theoretical backing, comprehensive experiments, and new SOTA results. 3ogIALgghF (8,6,6,8 — accept poster) had novel framing and well-designed experiments. The paper under review does not reach this level — the gains are too incremental and the conceptual contribution, while interesting, is not deeply developed.

The paper under review sits between the 6-level (XJiN1VkgA0) and 5-level. The complementarity insight is genuinely interesting and the experiments are methodologically clean, but the AUROC gains for most settings are too marginal to strongly support the headline claims. The test-set tuning concern further weakens confidence. It is a solid, useful empirical study that would be stronger with proper validation splits and less overclaiming.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>