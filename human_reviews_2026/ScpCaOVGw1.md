# EVEREST: A Transformer for Probabilistic Rare-Event Anomaly Detection with Evidential and Tail-Aware Uncertainty

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4

## Abstract
Forecasting rare events in multivariate time-series data is a central challenge in machine learning, complicated by severe class imbalance, long-range dependencies, and distributional uncertainty. We introduce EVEREST, a transformer-based architecture for probabilistic rare-event forecasting that delivers calibrated predictions and tail-aware risk estimation, with auxiliary interpretability through attention-based signal attribution. EVEREST integrates four key components: (i) a learnable attention bottleneck for soft aggregation of temporal dynamics; (ii) an evidential head for estimating aleatoric and epistemic uncertainty via a Normal–Inverse–Gamma distribution; (iii) an extreme-value head that models tail risk using a Generalized Pareto Distribution; and (iv) a lightweight precursor head for early-event detection. These modules are jointly optimised with a composite loss combining focal loss, evidential negative log-likelihood, and a tail-sensitive EVT penalty, and act only at training time; deployment uses a single classification head with no inference overhead. We evaluate EVEREST on a real-world benchmark spanning a decade of space-weather data and demonstrate state-of-the-art performance, including True Skill Statistic (TSS) scores of 0.973, 0.970, and 0.966 at 24, 48, and 72-hour horizons for C-class flares. The model is compact (≈0.81M parameters), efficient to train on commodity hardware, and applicable to other high-stakes domains such as industrial monitoring, weather, and satellite diagnostics. Limitations include reliance on fixed-length inputs and exclusion of image-based modalities, motivating future extensions to streaming and multimodal forecasting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes EVEREST, a compact rare-event time-series forecaster that co-optimizes discrimination, calibration, and tail-risk. The backbone is a 6-layer Transformer with a single-query attention bottleneck. Three training-only auxiliaries, including an evidential NIG head on the logit, an EVT (GPD) exceedance loss, and a lightweight “precursor” head, regularize the shared representation, while deployment uses a single classification head (≈0.81M params). On SHARP–GOES solar-flare benchmarks, EVEREST reports SOTA TSS across 9 tasks (e.g., ≥C: 0.973/0.970/0.966 at 24/48/72h) with strong calibration; it also transfers to SKAB with F1≈98%.

### Strengths
- Tail-aware + calibrated: EVT loss on logits and evidential NIG improve both sensitivity to extremes and reliability (e.g., ECE≈0.016 on the hardest task). 
- Clear ablations & diagnostics tying gains to modules; decision-theoretic thresholding (cost–loss) is practical.

### Weaknesses
- Novelty is integrative rather than fundamentally new—readers may view EVEREST as a well-engineered recipe.
- Data scarcity at the extreme tail (e.g., M5 with 104 positives/test) risks optimistic estimates; consider additional robustness tests (temporal hold-outs, cycle shift). 
- Sensitivity analysis limited: fixed EVT quantile ($\mu$=0.9), $\lambda$ weights, and focal-$\gamma$ schedule. More systematic sweeps (or conformal variants) would strengthen claims.

### Questions
- How sensitive are results to the exceedance quantile and to the stability regularizer? 
- Generalization under distribution shift: Any evaluation across solar cycles or on a temporally forward-held test period (e.g., 2024–2025 only) to assess drift? 
- Calibration granularity: Do you have class-conditional and operating-threshold–conditional calibration (e.g., reliability among high-risk alerts) beyond global ECE? 
- SKAB protocol alignment: Please clarify whether your SKAB setup strictly matches TranAD and others (windowing, labels, early-event detection vs point detection) and provide per-valve confusion matrices. 
- Ablation rigor: Could you report per-seed distributions for the large ∆TSS (e.g., +0.427 from the bottleneck) and include a joint ablation (evidential+EVT removed together)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces EVEREST, a compact Transformer-based architecture for probabilistic rare-event forecasting in multivariate time series, addressing core challenges of class imbalance, long-range dependencies, and distributional uncertainty. Targeting high-stakes domains (space weather, industrial monitoring), it delivers calibrated predictions, tail-risk estimation, and interpretability—with no inference overhead (814k parameters) as auxiliary modules act only at training.

Key contents.

1. EVEREST integrates four critical components:  a learnable attention bottleneck, an evidential (Normal-Inverse-Gamma) head, an EVT (Generalized Pareto) head, a precursor head.

2. A unified objective balances discrimination (focal loss for imbalance), calibration (evidential NLL), tail awareness (EVT penalty), and anticipatory learning (precursor BCE).

3.  The proposed method has the practical utility, compact (16.6M FLOPs), efficient to train (24s/epoch on RTX A6000), and interpretable (attention aligns with physical precursors like solar flux).

### Strengths
The paper has its own originality by addressing a longstanding gap in rare-event time-series forecasting: joint optimization of discrimination, calibration, and tail-risk within a compact, deployment-efficient framework—an unmet need in prior work that treated these goals in isolation. Unlike existing Transformer-based methods that prioritize discrimination over uncertainty or tail behavior, EVEREST innovates via four integrated, training-only auxiliary modules: a single-query attention bottleneck, an evidential Normal-Inverse-Gamma (NIG) head, an Extreme Value Theory (EVT) head, and a precursor head (anticipatory supervision). The proposed method is shown to achieve reasonably good experimental results.

### Weaknesses
The biggest weakness of this paper is its presentation. See detailed comments in "Questions" section.

### Questions
1. I feel like the presentation is very poor. The organization of this article is really weird. All sections are structured as fragmented short paragraphs. This makes it more like an outline document rather than an academic paper.

2. The presentation of Section 3 is not clear enough. For example, there is no explanation of symbols $T$, $F$. There are no formula definition of  $\mathcal L_{evid}, \mathcal L_{evt}, \mathcal L_{prec}$.

3. In the loss function, there are four tuning parameters, namely, $\lambda_f, \lambda_e, \lambda_t, \lambda_p$. Actually, only three of them are needed due to the scale invariance.

4. Baseline methods like Liu et al 2019, Sun et al 2022, Abduallah et al 2023 are not well-explained.

5. Overall, the poor writing prevents me to fully judge the quality of the paper.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces EVEREST, a Transformer-based architecture designed for forecasting rare events in multivariate time-series data. 

The model's main proposition is to integrate four components: 

- a learnable attention bottleneck for temporal aggregation
- an evidential head for uncertainty quantification 
- an extreme-value head to model tail risk 
- a precursor head for early detection. 

These components are jointly optimized during training via a composite loss function, but only the primary classification head is used for inference, resulting in no runtime overhead. 

The authors claim SoTA performance on the SHARP dataset and test cross-domain transfer to the SKAB dataset. Both datasets known for presence of rare events in time series.

### Strengths
The work presents some novelty of architecture and a combination loss to train it well. This is a valuable combination of many existing concepts often considered in the field. The strong performance of the trained models on the tested datasets shows the effectiveness of the approach. 

Some of the work in the appendix such as appendix J shows good real world utility of the methods.

The paper is well presented and the code and instructions made for reproducibility are super helpful. They help validate the information shared in the work.

### Weaknesses
The hardest to buy assumptions are about the multi-component loss. Section 5.4 very briefly talks about ablations to identify the value of each of these and there is explanation presented in Section 3.5 however it does not feel enough to motivate the key contribution of the work. It would be significantly more compelling to run better designed loss component ablations. Some analysis on whether each loss component and head are actually showing improvement on the kind of time series one would expect. Just looking at overall numbers in the case of evaluating such specific choices tends to obfuscate the true value of the contributions. This is the key reason behind my weak accept rating and not an outright accept.

Such analysis might also allows readers to gain insights on adapting these methods to other datasets and settings. 

The complexity of navigating the hyperparameter landscape that this loss function introduces is also something that more discussion and analysis would help.

### Questions
1. A more detailed ablation analysis and if possible correlating the components to the kind of samples they improve would be the one thing that would make the biggest difference to the quality of this work.

2. A couple of relevant references that seem to be missing here are

https://arxiv.org/abs/2202.13418: This work uses the pareto distribution to model extreme events in time series.
https://arxiv.org/abs/2103.12474: Work exploring contrastive loss, could be an interesting direction to check as well

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors 

The authors claim the following contributions
1. A recipe that focuses on long contexts via a single-query attention bottleneck, 
2. A recipe that learns calibrated, closed-form uncertainty (evidential NIG on the logit), and 
3. A recipe that emphasises extremes through an EVT exceedance penalty

### Strengths
# originality

Tests on two benchmark datasets

# significance

Results show the method performs well on the benchmarks

### Weaknesses
# clarity

The paper is quite challenging to evaluate. It reads in a stop-start fashion, lacking flow.

It feels like the paper presents a Rubik's Cube solution that has been heavily tuned to the specific datasets. For instance, the method introduced four new hypers in the loss.

I find it difficult, in its current form, to determine whether the method presents an approach that is generalizable beyond these specific case studies.

### Questions
None

### Soundness
2

### Presentation
1

### Contribution
2
