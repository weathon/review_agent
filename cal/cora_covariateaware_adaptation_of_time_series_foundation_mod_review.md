=== CALIBRATION EXAMPLE 35 ===

# Final Consolidated Review
## Summary

CoRA proposes a framework for adapting univariate-pretrained Time Series Foundation Models (TSFMs) to covariate-aware forecasting by freezing foundation model backbones as embedding extractors, learning a Causality Embedding to weight covariates by their predictive significance, and injecting this information via a zero-initialized adaLN mechanism. The method handles multi-modal covariates (time series, text, images) and demonstrates strong empirical results across uni-modal, multi-modal, few-shot, and multivariate forecasting benchmarks.

## Strengths

- **Principled adaptation design with zero-initialization guarantee**: CoRA ensures the adapted model starts identically to the pretrained TSFM by zero-initializing all adaptation parameters (W_{m,i}, b_{m,i}, MLP). The ablation (Table 5) confirms that replacing zero-init with Xavier initialization degrades performance, providing empirical support that this design choice matters for stable TSFM adaptation—a contribution that goes beyond simply borrowing from the LoRA/DiT literature.

- **General covariate integration across modalities and backbones**: Unlike prior adaptation methods (ChronosX, AdaPTS, UniCA) that modify the TSFM input structure, CoRA operates on the output embedding space, enabling clean integration of multi-modal covariates. The generality experiments (Figure 6, Table 11) show consistent improvements across four distinct TSFMs (Sundial, TimesFM, Chronos-Bolt, FlowState), supporting the claim of backbone-agnostic applicability.

- **Emergent interpretability of Causality Embedding**: Figure 7 demonstrates that the learned Softmax(W_CE) weights correlate with statistical Granger-Geweke causality tests across 1000 windows on ETTh1, providing interpretable insight into covariate relevance. This is a practically useful property not commonly offered by deep forecasting methods.

- **Comprehensive empirical coverage**: The paper evaluates on long-term forecasting (7 datasets), short-term EPF (5 datasets), multi-modal (RT-1, Time-MMD), few-shot, and multivariate settings, with consistent improvements over both supervised models and adaptation baselines.

## Weaknesses

### Major:

- **Causality Embedding nomenclature and framing risk overstating the contribution**: The term "Causality Embedding" implies the module discovers causal relationships, but it is trained purely on prediction loss. The correlation with Granger-Geweke statistics (Figure 7) is an emergent property, not a design guarantee. Granger causality itself captures predictive usefulness, not structural causation (which the paper acknowledges in Section 3.2, but only briefly). More critically, no quantitative measure of the correlation strength (e.g., mean Pearson coefficient, p-value) is reported, and the alignment is shown on only one dataset. Without demonstrating that the learned weights are *functionally* important (e.g., masking high-weight covariates hurts more than masking low-weight ones), the interpretability claim remains correlational rather than causal. This matters because practitioners may misuse these weights as evidence of true causal drivers.

- **No computational cost analysis**: CoRA requires loading and running inference through multiple frozen foundation models simultaneously (e.g., Sundial for the target, ViT for image covariates, Qwen3-Embedding for text covariates). For datasets like ECL with 320 covariates, each covariate requires a forward pass through the TSFM backbone. The paper provides no training time, inference latency, memory footprint, or parameter count comparisons. For an adaptation method where practical deployment feasibility is critical, this omission is significant. The efficiency trade-off between CoRA's performance gains and its multi-backbone overhead is uncharacterized.

- **Limited analysis of failure modes and covariate robustness**: The paper does not evaluate how CoRA performs when covariates are noisy, irrelevant, or adversarial. If the Causality Embedding claims to filter irrelevant information, this should be empirically validated by injecting random noise covariates and showing performance is maintained. Similarly, there is no analysis of when CoRA underperforms or when covariate inclusion provides negligible benefit (e.g., datasets where the target is largely autoregressive). This makes it difficult for practitioners to assess applicability.

### Minor:

- **Mean aggregation discards temporal structure in multi-modal covariates**: For text and image covariates, Equation 3 averages embeddings across all time steps. The paper acknowledges this in Section D, but the design choice is not motivated in the method section. For datasets like RT-1 where visual dynamics are important, this could limit effectiveness.

- **Full results limited to Sundial backbone**: While Section 4.2 and Table 11 demonstrate generality across TSFMs on the EPF dataset, the main results (Tables 1, 2, 4) rely exclusively on Sundial. Full results on additional backbones would strengthen the generality claim.

- **Some percentage claims do not cleanly match reported table values**: The abstract and Section 4.1.1 claim "31.1% MSE reduction" over TimeXer. Computing from Table 1 averages across all 7 datasets, the reduction is approximately 29-30% depending on the averaging method. Similarly, the claim of "1.9% MSE improvement" on Time-MMD (Section 4.1.2) is difficult to verify from Table 3, where CoRA achieves 0.641 vs UniCA's 0.661 (≈3% reduction). These discrepancies, while small, undermine precision.

### Trivial:

- The first contribution bullet ("We emphasize that an important paradigm...") is a positioning statement rather than a concrete technical contribution.

## Nice-to-Haves

- Compare learned Causality Embedding weights against fixed statistical Granger causality scores used as weights. If the statistical test performs similarly, the learned module's added complexity would need further justification.

- Quantify catastrophic forgetting by evaluating the adapted model's performance on the TSFM's original zero-shot benchmark tasks post-adaptation.

- Add a simple linear probe baseline (frozen TSFM embeddings + linear projection with covariates) to isolate whether the architectural innovations (adaLN, zero-init) contribute beyond simple covariate inclusion.

- Report standard deviations across multiple runs for main results; while single-run reporting is common in the TS community, the margins on some comparisons are narrow enough that variance information would be informative.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Table formatting issues**: The harsh critic raised concerns about table readability, but these are explicitly parser extraction artifacts, not issues with the paper itself. Removed as formatting nitpick.

- **Missing related works (prompt tuning, prefix tuning, adapter layers)**: The spark finder suggested comparisons with additional adaptation methods. Per rules, I cannot confirm these specific methods exist in the TS adaptation literature with the exact configurations imagined, and the paper already compares with the most directly relevant TS adaptation methods (ChronosX, AdaPTS, UniCA, Gen-P-Tuning).

- **Reproducibility concerns about undisclosed hyperparameters**: The paper specifies learning rates, batch size, optimizer, epochs, and early stopping. Removed per rules against nitpicking trivial implementation details.

- **Broader impact / negative societal consequences discussion**: Not standard for ICLR submissions in this area. Removed as scope creep.

- **Concern that zero-initialization claim lacks direct TS domain support**: The paper cites Hu et al. (2021) and Peebles & Xie (2023) for zero-initialization principles, and provides ablation evidence (Table 5) showing degradation without it. The criticism that these citations are from other domains is valid but weakened by the direct ablation evidence.

- **Demand for statistical significance tests as a major weakness**: While desirable, single-run deterministic evaluation is the established norm in the time series forecasting community (all baseline papers in Table 1 follow this convention). Moved to nice-to-have as it demands practices not standard in the paper's field.

- **Concern that the TSFM backbone might not be truly frozen**: The paper is clear that "CoRA treats pre-trained foundation models of different modalities as frozen embedding extractors" (Section 3.1) and only trains the adaptation modules. This concern reflects a misreading.

## Novel Insights

The zero-initialized adaLN injection mechanism represents an underexplored design point at the intersection of foundation model adaptation and time series forecasting. While adaLN is borrowed from diffusion models (DiT), its application here is meaningfully different: in DiT, the conditioning signal (class label/timestep) is always informative, whereas in CoRA, the conditioning signal (covariate embeddings) may be noisy or irrelevant, making the Causality Embedding's gating function essential for preventing harmful injection. This creates an interesting interplay where the zero-init ensures safe starting conditions, the adaLN provides expressive modulation, and the Causality Embedding acts as a quality filter—each component addressing a distinct failure mode. The empirical finding that these learned weights correlate with Granger causality suggests that prediction-loss optimization in this architecture implicitly recovers a notion of predictive relevance, which raises a broader question: to what extent do well-designed adaptation architectures naturally discover interpretable feature importance without explicit regularization?

## Suggestions

- Add a noise robustness experiment: inject K random noise covariates into existing datasets and show that Causality Embedding weights for noise covariates remain near zero while performance is preserved.

- Report computational overhead (training time, GPU memory, inference latency) for CoRA vs. baselines, particularly for high-covariate-count datasets like ECL (320 covariates) and Traffic (861 covariates).

- Provide the mean and standard deviation of the Pearson correlation coefficient in Figure 7, and test on at least one additional dataset beyond ETTh1 to support the interpretability claim.

- Add a functional validation of Causality Embedding: ablate (zero out) the top-K vs. bottom-K weighted covariates and show that top-K ablation hurts significantly more.

- Include one full result table (e.g., on EPF) for at least one additional backbone (TimesFM or Chronos-Bolt) to complement the Sundial-only main tables.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 8.0, 4.0]
Average score: 4.5
Binary outcome: Reject
