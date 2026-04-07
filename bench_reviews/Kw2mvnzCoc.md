## Summary

TSPulse proposes ultra-lightweight (1M parameter) pre-trained models for time-series diagnostic tasks with disentangled representations across temporal, spectral, and semantic views. The paper introduces hybrid masking for training, task-specific post-hoc fusers (TSLens, MHT), and demonstrates competitive or superior performance on anomaly detection, classification, imputation, and similarity search compared to models 10–100× larger.

## Strengths

- **Compelling efficiency-performance trade-off:** Achieving state-of-the-art results on TSB-AD (both univariate and multivariate) and substantial gains on UEA classification (5–16% over VQShape, MOMENT, UniTS) with only 1M parameters is empirically impressive. The 10–100× parameter reduction while matching or exceeding larger models addresses a practical deployment need.

- **Hybrid masking with demonstrated impact:** The 79% performance drop on irregular-mask imputation when ablating hybrid pre-training (Table 1c) provides strong empirical justification. This addresses a real issue—prior models pre-trained on block masking underperform when tested on irregular missingness patterns common in practice.

- **Identity-initialized channel mixing:** The 9% accuracy drop from random initialization (Table 1b) demonstrates this is a non-trivial engineering contribution for stable fine-tuning, particularly valuable for multivariate transfer from univariate pre-training.

- **Comprehensive empirical evaluation:** Results span 75+ datasets across four distinct tasks with consistent methodology. The inclusion of both zero-shot and fine-tuned variants, plus detailed ablations on synthetic perturbations, provides useful characterization of the learned representations.

- **Clear practical motivation:** The paper explicitly targets diagnostic tasks where real-time inference on CPU-only hardware matters, and provides concrete inference time measurements (Table 3 in Appendix).

## Weaknesses

- **"Zero-shot" terminology inconsistency:** The TSPulse(ZS) anomaly detection results use the official TSB-AD tuning set with labeled data to select the best-performing head per dataset. While this follows benchmark rules and is disclosed in Appendix A.11, calling this "zero-shot" is misleading when a labeled validation set informs model selection. The paper should use "unsupervised with head selection on tuning data" or similar terminology for transparency.

- **Disentanglement claims exceed evidence:** The paper achieves "disentanglement" through partitioned embeddings with different loss objectives—effectively multi-task learning with architectural inductive bias. No formal measure of disentanglement (e.g., mutual information between segments) is provided. The sensitivity analysis uses only synthetic sine waves (Section 6), which do not capture real-world complexity. Validation on real data would strengthen the claim that these embeddings capture genuinely complementary properties in practice.

- **Imputation gains partially from distribution alignment:** The +50% zero-shot imputation gains primarily compare against models pre-trained on block masking when evaluated on hybrid masking. TSPulse also outperforms on block masking (Appendix Figure 13), so the contribution is real, but the headline comparison advantages TSPulse's training-test alignment. The paper acknowledges this partially but could be clearer that hybrid masking both improves generalization and aligns with the evaluation regime.

- **No statistical significance testing:** Mean accuracy improvements (e.g., 0.733 vs 0.701 on 29 UEA datasets) are reported without confidence intervals or p-values. Given the variance common in time-series benchmarks, statistical significance of these improvements is unclear.

- **Multiple specialized checkpoints required:** Section 3.1 reveals different pre-training configurations per task (different masking strategies, head weightings). Users need separate 1M-parameter models for AD, classification, and imputation/retrieval—approximately 3 checkpoints, not one "versatile" model. Appendix A.15 shows a unified model works reasonably well, but this alternative is relegated to the appendix.

## Nice-to-Haves

- **Real-world disentanglement validation:** T-SNE or PCA visualizations of embeddings on actual UEA or TSB-AD data (colored by class/label) would validate that semantic embeddings capture meaningful structure beyond synthetic signals.

- **Forecasting results:** Even a brief evaluation on standard forecasting benchmarks would clarify whether disentangled representations sacrifice temporal predictive dynamics.

- **Non-pre-trained baseline comparisons:** Comparison against lightweight specialized models (e.g., TSMixer variants trained from scratch on each task) would isolate the value of pre-training from architecture.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"No explicit confirmation of no data leakage":** Appendix A.8 explicitly states all pre-training datasets are disjoint from evaluation sets with a clarifying example (australian electricity demand vs. ECL).

- **"Request for forecasting evaluation":** The paper explicitly scopes to diagnostic tasks; requesting forecasting is scope creep for this contribution.

- **"TSLens uses explicit attention":** The paper describes TSLens as having a "learned mechanism that adaptively extracts relevant features" but does not claim attention—this misreads the architecture (Section 3.3).

- **"No comparison against specialized retrieval baselines":** The paper compares against MOMENT and Chronos (the relevant pre-trained baselines for zero-shot embeddings). Requesting DTW or TS2Vec would be reasonable but is not a critical flaw.

- **"The 130% phase distortion is unsurprising":** While true that temporal embeddings should be phase-sensitive by design, the paper's contribution is demonstrating this quantitatively—criticizing it as "self-confirming" overstates the issue.

- **"Generic 'well-written' comment":** Removed as requested for being too generic.

## Novel Insights

The hybrid masking strategy's 79% performance drop on irregular imputation when ablated reveals a critical failure mode in prior foundation models: models pre-trained exclusively on fixed-length block masks may overfit to specific missingness patterns and fail to generalize to the irregular, point-level missingness common in real-world data. This observation—that training distribution alignment matters for practical robustness—is underappreciated in the TS pre-training literature and has broader implications for how future models should design masking curricula. Additionally, the identity-initialized channel mixer finding (9% accuracy difference) suggests that pre-trained weights carry useful univariate features that can be smoothly extended to multivariate settings if new layers are initialized to pass information through rather than randomly—this deserves attention as a transfer learning best practice.

## Suggestions

- Add confidence intervals or paired statistical tests for classification results across the 29 UEA datasets to strengthen claims of significance.

- Clarify terminology by renaming "TSPulse(ZS)" to "TSPulse (unsupervised, head-selected)" or similar in the main figures and text, reserving "zero-shot" for approaches that use no labeled data whatsoever.

- Include a paragraph in the limitations section acknowledging that the current approach requires separate task-specialized checkpoints for optimal performance, though a unified checkpoint remains competitive (cite Appendix A.15).

- Report imputation results on both block and hybrid masking in the main text (currently only hybrid is in main figures) to show performance holds across evaluation regimes.

- Consider adding one real-world dataset example to the sensitivity analysis (e.g., one anomaly type from TSB-AD) to validate disentanglement properties beyond synthetic sine waves.