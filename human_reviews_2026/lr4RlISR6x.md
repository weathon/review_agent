# PHAT: Modeling Period Heterogeneity for Multivariate Time Series Forecasting

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
While existing multivariate time series forecasting models have advanced significantly in modeling periodicity, they largely neglect the periodic heterogeneity common in real-world data, where variables exhibit distinct and dynamically changing periods. To effectively capture this periodic heterogeneity, we propose PHAT (Period Heterogeneity-Aware Transformer). Specifically, PHAT arranges multivariate inputs into a three-dimensional "periodic bucket" tensor, where the dimensions correspond to variable group characteristics with similar periodicity, time steps aligned by phase, and offsets within the period. By restricting interactions within buckets and masking cross-bucket connections, PHAT effectively avoids interference from inconsistent periods. We also propose a positive-negative attention mechanism, which captures periodic dependencies from two perspectives: periodic alignment and periodic deviation. Additionally, the periodic alignment attention scores are decomposed into positive and negative components, with a modulation term encoding periodic priors. This modulation constrains the attention mechanism to more faithfully reflect the underlying periodic trends. A mathematical explanation is provided to support this property. We evaluate PHAT comprehensively on 14 real-world datasets against 18 baselines, and the results show that it significantly outperforms existing methods, achieving highly competitive forecasting performance. Our sources is available at GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper provides an in-depth investigation into the frequency heterogeneity among variables in multivariate time series, highlighting that existing homogeneous modeling approaches fail to align with real-world data characteristics. To address this, the authors propose a bucket-based modeling framework that groups and models sequences according to their distinct periodic properties. Moreover, the paper explores the coexistence of positive and negative components within sequences, emphasizing that negative components are often suppressed by the softmax operation and thus require a redesigned attention mechanism to properly capture such “negative relations.” Experimental results demonstrate the effectiveness and robustness of the proposed method.

### Strengths
1. The paper is clearly written and easy to follow, with a logical and coherent structure, though certain experimental details could be further refined.
2. The experimental setup is fair and the comparisons are comprehensive.
3. The motivation is well grounded, and the problem addressed carries strong research significance.

### Weaknesses
1. Positive-Negative Attention Mechanism: The proposed mechanism is conceptually sound; however, the paper would benefit from multi-level visualizations of the positive and negative components — at the sequence level, feature (patch) level, and attention level. Without such analyses, it remains unclear whether the originally negative correlations in the raw sequence might become positively correlated after complex linear projections. This visualization is crucial for validating the effectiveness of the proposed mechanism.
2. It is recommended that Table 3 include results on more datasets to enhance the persuasiveness and comprehensiveness of the validation.
3. The results in Figure 5 and Table 3 could be presented in greater detail, with extended analyses provided in the appendix.
4. The paper could further discuss how to model sequences that share similar or overlapping periodicities. Since real-world time series often consist of multiple prominent periodic components, the strategy for bucket assignment and cross-bucket interaction among different periods remains an important open question deserving deeper exploration.

### Questions
see weeknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PHAT (Period Heterogeneity-Aware Transformer) for multivariate time-series forecasting. The key idea is to (i) detect per-variable dominant periods via FFT, (ii) group variables into periodic buckets that share a period, fold sequences into a 3-D tensor (bucket × period-offset × period-aligned), and (iii) apply a Positive-Negative Attention (PNA) with X-shaped receptive field that models phase-aligned vs. within-period relations and explicitly decomposes positive/negative correlations with a period-distance modulation. A frequency-weighted fusion produces the final forecast. Across 14 real-world datasets and 18 baselines, PHAT reports SOTA/top-2 results on most metrics, with strong complexity reductions.

### Strengths
This paper proposes PHAT (Period Heterogeneity-Aware Transformer) for multivariate time-series forecasting. The key idea is to (i) detect per-variable dominant periods via FFT, (ii) group variables into periodic buckets that share a period, fold sequences into a 3-D tensor (bucket × period-offset × period-aligned), and (iii) apply a Positive-Negative Attention (PNA) with X-shaped receptive field that models phase-aligned vs. within-period relations and explicitly decomposes positive/negative correlations with a period-distance modulation. A frequency-weighted fusion produces the final forecast. Across 14 real-world datasets and 18 baselines, PHAT reports SOTA/top-2 results on most metrics, with strong complexity reductions.

### Weaknesses
1. the main protocol tunes the look-back T per model and reports the best; a fixed-T comparison is deferred to the appendix. While both settings are shown, the primary table mixing tuned-T results across diverse baselines can blur fairness. Please foreground the fixed-T tables in the main paper (or add both side-by-side) and state the exact search ranges for T and other critical hparams per baseline. 
2. Sensitivity to period detection & K. Periods are extracted by FFT Top-K peaks and rounded to discrete lengths; buckets may overlap. The paper gives a small K-sweep but lacks robustness tests to mis-estimated periods, spectral noise, or drifting cycles. Please add stress tests varying K and perturbing detected periods ±{5–20%}, plus ablations on overlapping bucket policy and bucket cardinality.
3. The math mainly supports monotonic distance behavior / stick-breaking view of the modulated logits. A brief learning-theoretic argument (e.g., why decomposing positive/negative paths with masking improves bias/variance vs. vanilla attention under heterogeneity) would strengthen the theory section.

### Questions
As in Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge that heterogeneous variables in multivariate time series may exhibit different periodic patterns. The authors introduce a **periodic bucket structure** that groups variables based on their periodic lengths, and then model each periodic group separately. To address the lack of “negative correlation” modeling in attention mechanisms, the paper proposes a **positive-negative period-aware attention mechanism**. Experiments across numerous datasets and baselines demonstrate strong performance.

### Strengths
1. Simple and efficient method, easy to understand.
2. Clear motivation and well-organized structure.
3. Extensive experiments with diverse datasets and baselines, offering strong empirical support.

### Weaknesses
1. The experimental validation regarding “attention ignoring negative correlations” is not convincing; raw data analysis alone is insufficient to justify modeling implications at the feature level.
2. The paper should include an ablation that isolates the Frequency-based Multi-period Prediction component to clarify the exact gain contributed by the core modules, especially since prediction head size can directly affect performance in many settings.
3. Table 4 only reports FLOPs, but **actual inference latency** is not provided.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces PHAT (Period Heterogeneity-Aware Transformer), a novel architecture designed to handle temporal heterogeneity in video action recognition. Unlike existing temporal models that assume uniform periodicity, PHAT explicitly models multi-scale and non-uniform temporal patterns by decomposing video sequences into adaptive period components. Evaluations on benchmarks show consistent performance gains over recent baselines.

### Strengths
- The proposed PHAT design is principled and intuitive, integrating adaptive temporal decomposition into the Transformer framework without major architectural overhead. 

- The method is generalizable and can be incorporated into existing video backbones. 

- Experimental results are strong and consistent across multiple datasets, showing both improved accuracy and efficiency.

### Weaknesses
- The novelty is moderate, as the idea of handling multi-frequency or periodic dynamics has appeared in previous works on temporal Fourier attention and spectral modeling. 

- The mathematical formulation of heterogeneity modeling could be more rigorous; the “adaptive period tokens” are primarily empirical and not theoretically justified. 

- The comparisons focus mainly on uniform-period baselines but omit stronger contemporaneous temporal adaptation models. 

- The ablation studies are limited—particularly lacking analysis of how many period components are optimal or how PHAT behaves for short versus long actions.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
