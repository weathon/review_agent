The paper content is already in the user message. Let me now verify the specific claims made by the harsh critic and neutral reviewer against the actual paper content.

Key points to verify:

1. **Table 2 "+NoTS" rows**: Looking at the paper, PatchTST row shows classification accuracy ~80s, but +NoTS shows numbers like 11.71, 11.65. These are clearly not classification accuracies if classification is (↑). Similarly for imputation: PatchTST ETTm1 = 0.181, but +NoTS = 1.003. This is indeed confusing/misformatted as presented. The "Avg. error rate" column does show improvement (21.78→18.33, 16.07→15.70), so there is evidence of benefit, but the per-dataset numbers in the +NoTS rows are not interpretable as stated metrics.

2. **Theorem 1 applicability**: The harsh critic claims Theorem 1 is not specific to "next-period" tokenization. Let me re-read Section 4. The theorem says given a differential operator A that forms a sequence-to-sequence function f_{(A)} under sampling, there exists an input where any transformer network has lower bounded error. The claim is that sampling creates discontinuous maps. The critic argues this applies to ANY tokenization scheme. But the paper's argument is specifically about treating the signal S as direct input X to the transformer (i.e., period-based or point-based treatment), while the functional sequence approach creates a different input construction. The paper's proposition 1 then argues that creating functional sequences can make the mapping continuous. The critic correctly notes that Proposition 1's sufficient conditions are generic (existence of continuous mapping or expressive tokenizer) and could be satisfied by other approaches too.

3. **Forecasting absence**: The paper indeed does not include forecasting experiments. It evaluates classification, anomaly detection, and imputation.

4. **Pre-training on synthetic data only**: Confirmed - Section 5.2 states all pre-trained models are trained on synthetic datasets.

Now let me write the final consolidated review.

## Summary

NoTS (Narratives of Time Series) proposes re-interpreting time series as functions of time rather than concatenations of time periods, constructing degradation operators (convolution-based smoothing at multiple scales) to create progressively simplified variants, and training an autoregressive transformer to recover the original signal from most degraded to least—a "next-function prediction" objective. The paper provides theoretical analysis under the universal approximation framework claiming this functional perspective broadens the class of approximable functions, and validates on 2 synthetic and 22 real-world datasets across classification, anomaly detection, and imputation.

## Strengths

1. **Novel conceptual framing**: Re-interpreting time series as temporal functions with coarse-to-fine autoregressive recovery is a creative and well-motivated alternative to next-period prediction, offering a principled new axis for pre-training design.

2. **Attempted theoretical grounding**: Section 4 provides formal analysis (Theorem 1, Proposition 1) linking the method to universal approximation theory, which goes beyond purely empirical contributions common in time series pre-training papers.

3. **Targeted synthetic validation**: The feature regression experiment (Table 1) directly tests approximation ability on synthetically generated signals with known ground truth features (SSC, WAMP, H-index), showing substantial improvements over baselines.

4. **Multi-task real-world evaluation**: Testing on 22 datasets spanning classification, anomaly detection, and imputation provides breadth of evidence for the method's generality.

5. **Parameter-efficient adaptation**: The demonstration that NoTS-lw achieves 82% of full performance with <1% parameters trained is a practically valuable finding.

6. **Thoughtful ablations**: Table 3 disentangles the contributions of the latent consistency term, AR masking, and connected augmentations, providing insight into what drives performance.

## Weaknesses

### Fatal
None.

### Major

- **Gap between theoretical claims and the practical method** — The paper's central claim is that "constructing sequences of temporal functions allows for a broader class of approximable functions compared to sequences of time periods." However, Theorem 1 shows a universal approximation failure for the differential operator when input approaches zero (a pathological M→∞ construction) that is about discontinuity of the sampled operator map, not specifically about "period-based" vs "function-based" tokenization. Proposition 1 states that if either (i) the constructed sequence is expressive enough, or (ii) the encoder creates a continuous mapping to the target, then transformers can approximate—these are existence conditions that could equally be satisfied by a strong period-based tokenizer. The paper does not demonstrate that NoTS's specific degradation sequence satisfies these conditions, nor that standard period-based tokenization fails them. The paper itself acknowledges this gap (Section 5: "the approximation analysis posts strong assumptions on the solution including the minimal length T of the constructed sequence and the use of specific encoder E"), but the abstract and introduction present the theoretical result as if it directly justifies the method. This matters because the framing elevates what is essentially a well-regularized multi-scale smoothing pre-training objective into a theoretically justified paradigm shift.

- **The "+NoTS" rows in Table 2 are uninterpretable as presented** — The rows for PatchTST+NoTS and iTransformer+NoTS show classification values of ~11 (under a ↑ metric where baselines are ~80–88) and imputation values of ~1.0 (under a ↓ metric where baselines are ~0.1–0.3). These numbers are clearly not raw performance metrics in the same scale as the baselines. The "Avg. error rate" column does show improvement (21.78→18.33, 16.07→15.70), which suggests the method helps, but the per-dataset values cannot be directly compared, undermining the paper's claim that "NoTS improves their performance without specific backbone or adaptors, showing the versatility of the pre-training method." The paper does not explain what these numbers represent or how they should be read.

- **Missing forecasting task** — The paper positions itself as providing "a viable, theoretically justified alternative for building foundation models for time series" and directly compares against next-period prediction (which is designed for forecasting), yet evaluates only classification, anomaly detection, and imputation. Forecasting is the primary evaluation task in nearly all time series foundation model literature. Its absence is a significant gap in evaluating whether NoTS truly serves as a general-purpose foundation model alternative.

- **Experimental comparisons conflate pre-training objectives with architecture and regularization effects** — The synthetic feature regression experiment (Table 1) compares methods with different pre-training objectives, architectures, and loss functions (VQVAE vs MAE vs FAMAE vs next-period vs NoTS). The real-world experiments similarly compare fully different pre-training pipelines. The gains could plausibly arise from NoTS's multi-scale smoothing augmentation acting as a strong regularizer, rather than from the "functional narrative" per se. The ablation in Table 3 partially addresses this (showing that non-AR connected augmentations still help, variant 2), but only on a single synthetic metric (H-index), leaving open whether the AR structure specifically matters for real-world tasks.

### Minor

- **Pre-training exclusively on synthetic data** — NoTS-lw is pre-trained only on fBm and autocorrelated sinusoids, then adapted to real-world tasks. While the <1% parameter result is interesting, the generalizability of representations learned from two specific synthetic distributions to diverse real-world domains is not analyzed, and synthetic-only pre-training may not scale to truly diverse time series corpora.

- **Limited ablations on degradation design** — The choice of local averaging and global sinc filters, the number of degradation levels K, and kernel sizes {p_k} are central to NoTS, yet there is no sensitivity analysis. The ablation only tests Gaussian noise as an alternative degradation on one synthetic feature (Table 3, variant 4).

- **"Average error rate" metric lacks specification** — The paper aggregates across tasks with fundamentally different metrics (classification accuracy ↑, anomaly detection F1 ↑, imputation MSE/MAE ↓) into a single number without clearly specifying the normalization procedure in the main text.

- **Scalability claim is premature** — The power-law scaling claim (Figure 3C) is based on only 4 model sizes (127K–2.1M parameters), far smaller than modern foundation models. Concluding "power law behaviour" from 4 points is overclaiming.

- **Information monotonicity of degradation operators is not verified** — The paper states g_{k+1}(t) "contains strictly more or an equal amount of information than g_k(t)," but does not prove or empirically verify that convolution-based degradation with the chosen hyperparameters yields nested information.

### Trivial

- Minor notation inconsistencies (e.g., the abstract mentions "22 real-world datasets" but only ~10 are shown in Table 2; the count presumably includes appendix datasets).

## Nice-to-Haves

- Forecasting experiments would substantially strengthen the claim of NoTS as a general foundation model alternative.
- Ablation of K (number of degradation levels) and sensitivity to kernel sizes on real-world tasks.
- Comparison with more recent time series foundation models (Chronos, MOMENT, Moirai, TimesFM).
- Analysis of computational cost (training time, inference latency, memory) relative to baselines.

## Removed Points

These points are flagged to be removed, treat them with caution.

1. **"The theoretical results are essentially tautological" (Harsh Critic)** — While Proposition 1's conditions are general, calling them "tautological" overstates the case. They are standard sufficient conditions in the universal approximation framework. The real issue is the gap between theory and practice, which is already captured in the Major weakness above.

2. **"Missing comparison with modern foundation models like Chronos, MOMENT, Moirai" (Spark)** — While this would strengthen the paper, the paper compares against 3 established pre-training methods (SimMTM, bioFAME, next-period prediction) and 2 backbone architectures. Demanding additional baselines beyond the already reasonable comparison set is scope creep for a methods paper, though it would be nice to have.

3. **"Computational cost not discussed" (Neutral Reviewer)** — This is standard for methods papers in this venue. Not discussing FLOPs does not undermine the core claims.

4. **"The 'narrative' framing is just multiresolution smoothing" (Harsh Critic, conceptual)** — The conceptual reframing, even if practically implemented as multi-scale convolution smoothing, is a legitimate contribution if it provides new insight and better empirical results. The concern is about overclaiming, not about the method itself.

5. **"All pre-training on synthetic data without domain gap analysis" (Spark)** — While a valid concern, this is already partially addressed in the Minor weakness about synthetic pre-training. Spark's demand for "what dynamics transfer successfully" is scope creep.

6. **"Theorem 1 uses a pathological M→∞ example" (Harsh Critic)** — While the example is extreme, this is standard practice in approximation theory to demonstrate failure modes. The example is valid for showing that the mapping can be discontinuous, which is what matters for universal approximation. However, the critic's point that this doesn't specifically indict "period-based" tokenization is valid and captured above.

7. **"No comparison where only the ordering changes" (Harsh Critic)** — This is a valid experimental concern but asking for a complete disentangling of all factors is an unreasonable standard for a single paper. The partial ablation in Table 3 addresses this to some extent.

8. **"Formatting concerns about Table 2 +NoTS rows"** — Multiple reviewers flagged the same issue, which IS a real problem. However, some of the specific numerical concerns (e.g., claiming the numbers show "performance degradation") may be incorrect if these numbers are on a different scale. The issue is insufficient explanation, not necessarily incorrect results. This is captured in the Major weakness above.

## Novel Insights

The most insightful observation across reviews is that NoTS's empirical gains may stem more from the multi-scale smoothing augmentation acting as an effective regularizer than from the "functional narrative" autoregressive structure. This is directly supported by Table 3 variant (2), which shows that non-AR connected augmentations (without AR masking) achieve error 1.48 vs. NoTS's 1.27 on H-index—still better than next-period prediction's 1.75, but with much of the gain coming from augmentation rather than AR ordering. This suggests the practical contribution may be more about data augmentation than representation learning theory, even though the framing emphasizes the latter.

## Suggestions

- Clearly explain what the "+NoTS" rows in Table 2 represent (absolute metrics or relative improvements) and provide raw performance numbers for all entries, or move the current presentation to an appendix with a clear legend.
- Weaken the theoretical claims in the abstract and introduction to match what Section 4 actually demonstrates (existence of discontinuous maps under sampling, and sufficient conditions for continuity, rather than a proven advantage of functional sequences over period sequences).
- Add at least one forecasting benchmark to validate the claim of being a foundation model alternative.
- Provide sensitivity analysis on K and the degradation hyperparameters alongside the ablation on one real-world task (not just H-index on synthetic data).

## Evaluation on Key Axes

- **Originality**: The conceptual reframing from next-period to next-function prediction is novel and interesting. However, the practical implementation (multi-scale conv smoothing + AR transformer) is relatively incremental, and the theoretical analysis does not tightly support the specific claims made.
- **Importance of research question**: Building general-purpose pre-training for time series is an important open problem. NoTS offers a promising direction.
- **Whether claims are well supported**: This is the main weakness. The theoretical claims overreach what is proven, the key experimental table has uninterpretable entries, and the empirical gains are not cleanly attributable to the "functional narrative" mechanism.
- **Soundness of experiments**: Synthetic experiments are well-designed; real-world experiments cover breadth but lack forecasting and have the Table 2 formatting issue; ablations are limited to one synthetic metric.
- **Clarity of writing**: The paper is generally well-written with clear illustrations (Figure 1 is excellent), but has significant clarity issues in Table 2 presentation.
- **Value to research community**: Moderate — the direction is promising but the overclaimed theory and confusing experimental presentation reduce immediate impact.

## Score and Decision

**Calibration**: I compared against:
- PTE4TS (human scores 3,3,5,3 — rejected): A time series pre-training paper with limited novelty, weak theoretical grounding, and insufficient ablations. NoTS is stronger than PTE4TS in experimental breadth and conceptual novelty but shares some weaknesses (limited theory-experiment connection).
- bioFAME (human scores 6,5,8,3 — rejected): A frequency-aware pre-training method for biosignals. NoTS has a similarly interesting conceptual framing but similar concerns about novelty of practical components (convolution smoothing vs. frequency filters) and overclaimed theory.
- OccVAR (human scores 3,3,6,3 — rejected): A coarse-to-fine AR model for occupancy prediction. NoTS shares the "coarse-to-fine AR" paradigm and has similar novelty concerns — the incremental AR structure on top of multi-scale decomposition, similar to what OccVAR does for 3D occupancy.
- CARD (human scores 6,6,5,8 — accepted poster): A time series transformer with channel alignment. NoTS has a comparable level of empirical contribution but weaker experimental clarity.

NoTS presents a genuinely novel conceptual direction but overclaims its theoretical contribution and has significant presentation issues in the main results table. The absence of forecasting experiments and the conflation of augmentation effects with functional narrative effects weaken the empirical case. This is a promising preliminary contribution that needs significant revision to back its theoretical framing and to validate on the most natural evaluation task.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>