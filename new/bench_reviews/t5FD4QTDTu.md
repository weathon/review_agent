Let me re-read Table 2 from the paper content provided in the prompt. Looking carefully at the bottom two rows:

For PatchTST alone: UCR-9 = 83.57, UEA-5 = 63.31, SMD = 78.96, MSL = 78.81, SWaT = 83.75, PSM = 78.07, ETTm1 = 0.181, ETTm2 = 0.126, ETTh1 = 0.347, ETTh2 = 0.187

For "+NoTS (Ours)": UCR-9 = 11.71, UEA-5 = 11.65, SMD = 12.20, MSL = 13.96, SWaT = 15.97, PSM = 11.25, ETTm1 = 1.003, ETTm2 = 1.003, ETTh1 = 1.064, ETTh2 = 1.006

These numbers are clearly on a completely different scale. For classification (UCR, UEA), the values drop from ~83 to ~11; for anomaly detection, they drop from ~78-83 to ~11-16; for imputation, error increases from ~0.18 to ~1.0. This looks like dramatic performance *degradation*, not improvement. Yet the text says "NoTS improves their performance without specific backbone or adaptors." This is a major contradiction.

The "Avg. error rate" column also drops from 21.78 to 18.33 for PatchTST, which seems to suggest improvement on this aggregate metric, but the individual numbers clearly show degradation. This is very confusing.

Let me also check the iTransformer rows:
- iTransformer: UCR-9 = 82.67, UEA-5 = 67.62, SMD = 85.18, etc.
- +NoTS: UCR-9 = 11.26, UEA-5 = 10.65, SMD = 10.17, etc.

Again, dramatic degradation on individual metrics, yet the "Avg. error rate" claims improvement.

Wait - let me reconsider. The error rate metric might inverse the classification and anomaly detection numbers. If error rate = 1 - accuracy for classification (so 1 - 83.57 = 16.43), and NoTS gives 11.71, that would actually be better. But the raw numbers show classification accuracy dropping from 83.57 to 11.71, which can't be right under any reasonable interpretation.

Actually, looking more carefully at Table 2 headers: for "+NoTS" rows under classification columns, the numbers (11.71, 11.65) are labeled under "Classification (↑)". If lower is worse for classification accuracy, then 11.71 < 83.57 is terrible. Under anomaly detection (↑), 12.20 < 78.96 is terrible. Under imputation (↓), 1.003 > 0.181 is also worse.

So combining NoTS with PatchTST/iTransformer **without adaptors** dramatically hurts performance on every metric. Yet the text says it "improves their performance" and the "Avg. error rate" shows improvement. This is a fundamental contradiction.

Unless there's some interpretation issue... Perhaps the error rate formula somehow inverts things? But even so, the individual metrics unambiguously show catastrophic degradation.

Actually, wait - I need to re-read the caption more carefully. "We compute an average error rate (↓) to compare the final performance." So the avg error rate is computed differently than the individual columns. For classification and anomaly detection, the individual columns show accuracy (↑), while for imputation, they show error (↓). So the "error rate" inverts accuracies and normalizes imputation errors somehow. With NoTS, individual classification accuracy drops from 83.57 to 11.71 - this is horrific in any metric. The average "error rate" being lower despite terrible individual numbers seems mathematically impossible unless there's some very strange normalization, or the table contains errors.

This is a genuine major issue - either the Table 2 numbers for +NoTS rows are wrong/misreported, or NoTS actually catastrophically degrades PatchTST/iTransformer when used without adaptors, and the "error rate" aggregation is misleading.

Now let me synthesize the review.

Let me also check: the paper says "NoTS improves their performance without specific backbone or adaptors, showing the versatility of the pre-training method." This directly contradicts the numbers in the table.

Looking again at the table structure - for the top section (✓ ✓ = adaptors used, frozen), NoTS-lw vs SimMTM/bioFAME/Next-pred shows improvements. For the middle section (✓ ✗ = adaptors used, unfrozen), NoTS-lw also shows improvements. The bottom section (✗ ✗ = no adaptors, unfrozen) shows PatchTST + NoTS and iTransformer + NoTS with strange numbers.

I think the bottom section numbers might actually represent something like a "relative performance metric" or be garbled in the PDF extraction. But the text claims improvement. This needs clarification.

Actually, I realize the table says `(a) (b)` columns where (a) = whether adaptors are used and (b) = whether pre-trained weights are frozen. For the bottom rows: ✗ ✗ means no adaptors, unfrozen. The paper says "NoTS improves their performance without specific backbone or adaptors." 

If the +NoTS numbers are accuracy dropping from 83.57 to 11.71, this is catastrophic, not an improvement. The "Avg. error rate" could be computed differently (e.g., using a relative metric), but the per-dataset numbers are unambiguous.

This is a significant issue with the paper - either a table error or a fundamental problem with the claim.

Let me now write the consolidated review.</think>Now I have a thorough understanding of the paper and calibration papers. Let me verify the Table 2 anomaly more carefully by re-reading the relevant section of the paper.

Looking at Table 2 closely:
- Row "PatchTST" (✗ ✗): UCR-9=83.57, SMD=78.96, ETTm1=0.181
- Row "+NoTS (Ours)" (✗ ✗): UCR-9=11.71, SMD=12.20, ETTm1=1.003

The Classification and Anomaly Detection columns use (↑) = higher is better, while Imputation uses (↓) = lower is better. For every single entry in the "+NoTS" rows (PatchTST and iTransformer), the numbers dramatically worsen. Yet the paper states: "NoTS improves their performance without specific backbone or adaptors." The "Avg. error rate" column claims improvement (21.78→18.33 for PatchTST), which appears inconsistent with the per-dataset numbers.

This is a critical inconsistency that must be flagged.

## Summary

The paper proposes NoTS (Narratives of Time Series), an autoregressive pre-training objective for time series transformers that constructs coarse-to-fine sequences by applying degradation operators (convolution-based smoothing) and trains a transformer to progressively recover the original signal. A theoretical section argues that treating time series as functions rather than sequences of time periods enables approximation of a broader class of operators. Experiments on synthetic feature regression and 22 real-world datasets across classification, anomaly detection, and imputation tasks show improvements over existing pre-training methods.

## Strengths

- **Novel conceptual framing**: The shift from next-period prediction to constructing functional sequences via degradation operators is a genuine conceptual contribution. It reframes time series pre-training in a way that preserves nonlocal functional properties, addressing a real limitation of patch-based approaches.
- **Theoretically-motivated design**: The attempt to justify the approach through approximation theory (Theorem 1, Proposition 1), even if limited, goes beyond purely empirical design and grounds the method in a formal framework.
- **Comprehensive multi-task evaluation**: Testing across classification, anomaly detection, and imputation on 22 real-world datasets provides broader coverage than many time series pre-training papers that focus on forecasting alone.
- **Parameter-efficient adaptation**: The demonstration that <1% parameter training achieves 82% of full performance (frozen-weights setting) is a practically interesting result for transfer learning.
- **Well-designed synthetic benchmark**: The fBm and autocorrelated sinusoid datasets with feature regression (SSC, WAMP, Hurst index) provide a principled testbed for evaluating function approximation capabilities.

## Weaknesses

### Major:

- **The Table 2 "+NoTS" rows contain irreconcilable contradictions with claimed improvements.** The bottom section of Table 2 (no adaptors, unfrozen weights) shows catastrophic degradation on every individual metric when NoTS is added to PatchTST and iTransformer: e.g., PatchTST UCR-9 accuracy drops from 83.57→11.71, anomaly detection SMD drops from 78.96→12.20, and imputation ETTm1 error increases from 0.181→1.003. Yet the text claims "NoTS improves their performance without specific backbone or adaptors." The "Avg. error rate" column shows improvement (21.78→18.33), but this is inconsistent with the per-dataset numbers, suggesting either a table error, a misleading aggregation formula, or that NoTS actually degrades existing architectures without adaptors. This fundamentally undermines the claim of versatility and must be resolved.

- **The theoretical claims significantly overstate what is actually proved.** The abstract claims "constructing sequences of temporal functions allows for a broader class of approximable functions (e.g., differentiation)," but Theorem 1 only shows a single adversarial example (a differential operator applied to g_M(t) = sin(Mt)/M) where the discrete mapping becomes discontinuous, causing approximation difficulties for any continuous model. This does not uniquely implicate next-period prediction—any continuous parametrization faces similar issues. Proposition 1 then states two tautological sufficient conditions: (1) the constructed sequence admits a continuous mapping to the target, or (2) the encoder creates one. Neither condition is shown to hold for the specific smoothing operators used in NoTS, nor is it shown that patch-based tokenizers cannot satisfy similar conditions. The strong claims about "broader class of approximable functions" are not substantiated by the analysis provided.

- **The experimental evidence does not isolate the contribution of the "next-function" conceptual idea from confounding design choices.** The ablation study (Table 3) covers only one 1D feature (Hurst index) on one synthetic dataset. Missing critical ablations include: (1) using the same degradation operators in a non-AR reconstruction objective to test whether the AR structure matters; (2) randomizing the ordering of degradation levels to test whether coarse-to-fine matters vs. just having multi-scale augmentations; (3) applying the same encoder/decoder architecture to next-period prediction for a matched comparison. Without these, the gains cannot be attributed to the "functional narrative" idea rather than to the bundle of architectural and training-design choices.

- **Missing forecasting task evaluation.** The paper's central analogy is to next-word prediction in language, and its primary comparator is next-period prediction (designed for forecasting). Yet no forecasting experiments are included. Without demonstrating competitive performance on the task that the competing paradigm was designed for, the claim that NoTS is a "viable alternative for building foundation models" is unsupported—forecasting is arguably the most natural and important task for AR time series models.

### Minor:

- **The "function vs. period" dichotomy is somewhat misleading in practice.** The degradation operators are simply local averaging and global low-pass filtering applied to discrete samples—standard discrete-time signal processing operations, not modeling in a genuine function space. The method is effectively a multi-resolution smoothing ladder plus AR reconstruction in latent space, which is an interesting design but not fundamentally different from multi-resolution approaches in prior work (e.g., multi-resolution masking in Das et al., 2023; Woo et al., 2024).

- **Missing comparison with recent time series foundation models.** The paper positions NoTS as a "viable alternative for building foundation models for time series" but only compares against small-scale pre-training methods (SimMTM, bioFAME, next-period prediction). It does not compare with more recent foundation models like Chronos, TimesFM, MOMENT, or MOIRAI, which are the most relevant baselines for this claim.

- **No error bars or significance tests on real-world experiments.** Table 2 reports single numbers with no variance information, making it difficult to assess whether the observed improvements (5-7% relative improvement in average error rate) are statistically meaningful given the heterogeneity of tasks and datasets.

- **The degradation operators embed a strong inductive bias that "signal" is low-frequency and "detail" is high-frequency.** For domains where key information resides in high-frequency transients (e.g., spike detection in neuroscience, seismic events), this bias could be counterproductive. The paper does not discuss or evaluate this limitation.

- **The information monotonicity assumption** (that g_{k+1} contains "strictly more or an equal amount of information" than g_k) **does not hold in general** for arbitrary smoothing hierarchies. Lossy smoothing operations irreversibly remove information, and the combination of local and global filters with different bandwidths does not necessarily impose a monotone information ordering without careful design.

## Nice-to-Haves

- **Sensitivity analysis on the number of degradation levels K and kernel hyperparameters** would strengthen confidence in the method's robustness.
- **Comparison on forecasting benchmarks** would make the case for NoTS as a general pre-training framework much more compelling.
- **Failure-mode analysis** showing when smoothing-based degradation is counterproductive (e.g., spike-heavy signals) would add important practical guidance.
- **Analysis of autoregressive error accumulation** during inference, particularly when recovering fine details from degraded variants.

## Removed Points

- **"The paper claims results on 22 datasets but only shows 10"**: The paper explicitly states the remaining 12 datasets and additional experiments are in the appendix. This is a formatting choice, not a missing evaluation. *Treat with caution.*
- **"Table 2 +NoTS rows are garbled/misreported"**: While the numbers appear contradictory to the claimed improvements, this could be a PDF extraction artifact. However, the fundamental inconsistency between per-dataset numbers and the "Avg. error rate" column remains a genuine concern that cannot be dismissed as a formatting issue, given the individual metrics unambiguously show degradation. *Keeping the concern but noting extraction uncertainty.*
- **"Missing related work on multi-resolution masked modeling"**: The paper (Section 2.2) does cite Das et al. (2023) and Woo et al. (2024) as multi-resolution approaches. The reviewer's claim that these are underplayed is subjective but has some merit—no separate comparison is made. *Treat with caution.*
- **"NoTS pre-trained on synthetic data only"**: While true, the paper explicitly acknowledges this limitation and shows positive transfer results. Criticizing the specific pre-training data choice goes beyond the paper's stated scope of showing that the method works at all for transfer. *Treat with caution: legitimate concern about foundation model claims but not a methodological flaw per se.*
- **"AR error accumulation is a well-known limitation"**: This is a generic concern applicable to any AR model. The paper shows successful reconstruction in Figure 3(A) but does not explicitly analyze error propagation. This is a fair suggestion but not a fundamental flaw. *Treat with caution.*
- **"Formatting issues with Table 2 units/metrics"**: The "Avg. error rate" formula is not precisely defined in the main text but is referenced. The lack of clarity is a real concern, but this borders on a presentation nitpick. *Treat with caution.*

## Novel Insights

None beyond the paper's own contributions. The paper introduces an interesting conceptual reframing (functional sequences vs. time periods), but the gap between the theoretical claims and what is actually implemented/proved is notable: the "functions" are smoothed discrete samples, the theory addresses only one adversarial example, and the core method is effectively a multi-scale AR reconstruction approach. The tension between the ambitious theoretical framing and the relatively standard implementation is the most insightful observation that emerges from a careful reading.

## Suggestions

1. **Resolve the Table 2 contradiction immediately**—either the "+NoTS" rows for PatchTST and iTransformer contain errors (likely given the numbers), or the claim of "improvement without adaptors" is incorrect and must be revised. Provide clear, consistent per-dataset numbers and define the aggregation metric precisely.
2. **Add the missing ablations**: (a) same degradation operators in a non-AR (e.g., masked reconstruction) objective; (b) randomized ordering of degradation levels; (c) matched architecture next-period baseline.
3. **Tone down the theoretical claims** to match what is actually proved: Theorem 1 shows that a specific sampling-induced discontinuity can cause approximation problems, and Proposition 1 provides sufficient conditions that are not instantiated for NoTS's specific design. The "broader class of approximable functions" claim should be presented as a motivating intuition rather than a proven result.
4. **Add at least one forecasting benchmark** to establish that NoTS is competitive on the task that next-period prediction was designed for.

## Score and Decision

**Calibration**: I compared against several time series pre-training papers: TimeDART (AR + diffusion, rejected, scores 3-8, avg ~5), PTE4TS (TS pre-training, rejected, scores 3-5, avg ~3.5), DAM (foundation model for TS forecasting, accepted poster, scores 6-8, avg ~7), mr-Diff (multi-resolution diffusion for TS, accepted poster, scores 6-8, avg ~6.5), WaveToken (wavelet tokenization, rejected, scores 3-8, avg ~5.5), LPTM (large pre-trained TS model, rejected, scores 3-5, avg ~3.5), OTiS (TS foundation model, rejected, scores 5-6, avg ~5.2).

This paper shares strengths with DAM (foundation model ambitions, novel pre-training) and mr-Diff (multi-resolution for TS), but has more significant weaknesses than accepted papers: the theoretical claims are overstated, the Table 2 contradiction undermines key claims, critical ablations are missing, and forecasting evaluation is absent. These are similar to weaknesses seen in rejected papers like TimeDART (AR error accumulation, missing baselines) and WaveToken (marginal improvements, overclaiming). However, the conceptual novelty (functional narratives) and the synthetic evaluation are stronger than typical rejected TS pre-training papers. The paper is weakened most by the Table 2 inconsistency and the gap between theoretical ambition and actual proof, but not fundamentally broken—it presents a genuine idea with promising preliminary results.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>