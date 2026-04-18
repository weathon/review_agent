## Summary

This paper proposes forecasting whole-brain neuronal activity from volumetric calcium imaging video (3D+time) directly, rather than first reducing data to 1D neuron traces. The authors adapt a 4D UNet with temporal-context-as-channels, lead-time conditioning, and spatial sharding to handle the massive input volumes (~1.5 trillion voxels) from the ZAPBench zebrafish dataset. Through extensive ablations, they find that spatial context helps most when temporal context is short, that higher input resolution and cross-specimen pre-training do not improve performance, and that their video model outperforms trace-based methods particularly for short temporal contexts (C=4).

## Strengths

- **Novel and well-motivated problem formulation**: Bypassing ROI-segmentation to forecast directly on volumetric video is a logical and innovative approach that avoids the information loss inherent in trace extraction. The paper clearly formalizes the problem and the relationship between video and trace domains (Eq. 1–3).

- **Comprehensive and honest experimental evaluation**: The systematic ablations of spatial vs. temporal context (Fig. 5), input resolution (Table 2), and pre-training (Table 1) provide genuinely useful engineering insights. The negative findings—that higher resolution hurts and cross-specimen pre-training fails—are clearly reported rather than hidden, which is valuable for the community.

- **Scalable architecture and engineering**: The adaptation of the UNet to 4D volumetric data with spatial sharding across accelerators (Sec. 3.4) and the lead-time conditioning scheme (Sec. 3.3) represent solid engineering contributions that could benefit others working with similar data scales.

- **Effective short-context performance**: At C=4, the video model achieves a clear ~10pp reduction in MAE for 1-step-ahead prediction over the best trace-based model, and Figure 5's spatial-temporal trade-off is an informative empirical finding.

## Weaknesses

### Fatal
None.

### Major

- **The claim that performance gains come from "correlations between cells" is under-supported by the experiments**: The paper concludes that "it is specifically the correlations between cells in the recorded fluorescence signals, rather than the distribution of signals within individual cells, that drives these improvements" (Sec. 4.2). However, the masking experiment (App. A.2) only rules out unsegmented voxels as a driver; it does not distinguish between (a) true inter-neuronal spatial correlations, (b) within-cell spatial averaging/smoothing effects improving SNR, or (c) spatial smoothness priors implicitly imposed by the convolutional architecture. No controlled ablation isolates within-cell vs. between-cell contributions—for instance, by restricting the video model to only use voxels belonging to their own segmented cell, or by shuffling spatial positions of neurons while preserving temporal traces. This matters because the claim that video models succeed because they exploit "multivariate" neural relationships is the central interpretive conclusion, and multiple alternative explanations remain viable.

- **Performance gains are modest and horizon-limited, yet framed as "leading performance" and "consistent outperformance"**: At C=4, the video model is better only for early horizons (overtaken by trace models around step ~10 on the test set). At C=256, the paper itself states "there is no significant difference...when evaluated with MAE." Per-condition results show the video model wins on 6/9 conditions for C=4 and is worse on 2. The abstract's claim that the model "outperforms trace-based forecasting approaches" is true only in a narrow sense (short horizons at C=4), and the claim of being "the only approach that consistently benefits from multivariate information" overstates the consistency and magnitude of the gains given the 2–3 orders of magnitude increase in computational cost (acknowledged in the conclusion). These limitations should be reflected in the framing.

- **Single-animal evaluation with poor cross-specimen generalization**: All results are from one zebrafish. The pre-training experiments (Table 1) show that models trained on other specimens fail to transfer, and even reduce performance. While the paper mentions this, the broader claims about "predictive models of brain function" and "whole-brain activity prediction in vertebrates" (Intro, Abstract) overgeneralize from a single animal and recording session. The model's apparent strengths may be heavily tied to idiosyncratic properties of this particular specimen and preprocessing pipeline.

### Minor

- **Full-resolution results are from a single seed**: Table 2 acknowledges that the full-resolution result has no error bars "because of their compute requirements." Given that this is the only data point suggesting higher resolution hurts performance—a counterintuitive finding that the paper highlights—having only a single seed weakens confidence in this conclusion. Optimization instability at high dimensionality could be an alternative explanation.

- **The narrative tension between "preserving spatial information is key" and "lower resolution works better" is not well resolved**: The paper motivates the video approach by arguing that spatial structure is crucial, but then finds that 4× downsampled inputs perform best (Table 2). The authors speculate about voxel-to-parameter ratio, but this is not tested (e.g., by proportionally scaling up model parameters at full resolution). A more thorough discussion of what this implies for the role of spatial information would strengthen the paper.

- **Correlation metrics relegated to appendix**: Given that MAE alone can conflate deterministic error with intrinsic variability and noise, and that correlation is more interpretable in neuroscience contexts, the main text would benefit from including correlation results alongside MAE, especially since the video model shows stronger correlation improvements at C=256.

## Nice-to-Haves

- **Spatial permutation experiment**: Shuffling the spatial positions of neurons while preserving temporal traces would be the most direct test of whether spatial structure per se drives improvements, and would substantially strengthen (or weaken) the "correlations between cells" claim.
- **Trace model augmented with spatial coordinates**: Adding 3D neuron positions or adjacency information as features to the best trace-based model would help disentangle how much of the video model's advantage comes from spatial structure vs. the vastly richer input representation.
- **Cost-benefit analysis**: Given the acknowledged 2–3 orders of magnitude increase in compute, a brief analysis of how performance scales with compute for both approaches would help readers assess practical viability.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Unfair comparison because the video model has more input information"**: The paper's central thesis IS that preserving spatial structure (which traces discard) provides useful information. This comparison is the point of the paper, not a methodological flaw. The video model having richer input is the feature being tested. However, I kept the narrowed concern that the specific mechanism ("correlations between cells") is insufficiently demonstrated—this is about interpretation, not fairness of comparison.

- **"Missing baselines like neural data transformers (NDT), LFADS, etc."**: Per instructions, I cannot confirm the existence or relevance of specific baselines not discussed in the paper. The paper compares against the ZAPBench benchmark baselines, which include univariate and multivariate trace models. The absence of a GNN or coordinate-augmented trace model is a nice-to-have suggestion (moved above), not a fatal omission.

- **"Reproducibility concerns about code/data availability"**: The paper states code will be released post-review. This is standard practice and not a meaningful weakness.

- **The harsh critic's claim that the comparison "structurally biases" toward the video model**: This is the comparison the paper is designed to make. The paper does not claim the video model is a better architecture *given the same input*; it claims that operating in the video domain provides advantages. This is a valid experimental question, not a biased comparison. Removed as a structural concern, but the narrower point about overclaiming the specific mechanism is retained above.

- **"Metric choice is insufficient—MAE alone can't assess biological significance"**: The paper uses MAE because it is the ZAPBench benchmark metric and notes that alternative trial-to-trial metrics are inapplicable. This is a reasonable choice given the benchmark constraints, and correlation metrics are reported in the appendix.

## Novel Insights

The most interesting finding is the spatial-temporal trade-off (Fig. 5): spatial context helps when temporal context is short but hurts (via overfitting) when temporal context is long. Combined with the pre-training failure (Table 1) and the resolution result (Table 2), this paints a coherent picture suggesting that the video model's advantage comes primarily from spatial aggregation/smoothing acting as an implicit regularizer when temporal information is scarce, rather than from learning rich distributed neural dynamics. This read of the evidence would suggest that simpler spatial augmentations to trace models might achieve comparable gains at far lower cost—an important practical consideration the paper does not explore.

## Suggestions

- Re-frame the abstract and introduction to reflect that gains are primarily for short-horizon prediction at short temporal context, with parity at longer contexts, rather than claiming general "outperformance."
- Add a spatial permutation experiment (shuffle neuron positions, preserve traces) to directly test whether spatial structure per se drives improvements, which would either validate or invalidate the "cell–cell correlation" interpretation.
- Include correlation results in the main text rather than relegating them to an appendix, particularly for C=256 where MAE differences are negligible but correlation differences may be meaningful.

## Score and Decision

**Calibration**: I compared against papers with similar profiles: ZAPBench (scores 8/8/6/8, spotlight) is a benchmark resource paper with broader impact; Neuroformer (6/8/6/5, poster) applies transformers to neural data with moderate empirical gains; NDT3 (6/5/6/6, reject) found scaling limitations in a neural foundation model; BrainLM (6/6/6/6, poster) overclaimed generalization from limited data. This paper has solid engineering and an interesting spatial-temporal trade-off finding, but its empirical gains are modest and horizon-limited, and the central mechanistic interpretation is under-supported. It is comparable to BrainLM/NDT3-level contributions with somewhat better ablations, but weaker empirical wins than ZAPBench-level work.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>