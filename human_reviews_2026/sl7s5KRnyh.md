# VisionTS++: Cross-Modal Time Series Foundation Model with Continual Pre-trained Vision Backbones

- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
Recent studies have indicated that vision models pre-trained on images can serve as time series foundation models (TSFMs) by reformulating time series forecasting (TSF) as image reconstruction.  However, effective cross-modal transfer from vision to time series remains challenging due to three discrepancies:  (1) the data-modality gap between structured, bounded image data and unbounded, heterogeneous time series;  (2) the multivariate-forecasting gap between fixed RGB-three-channel vision models and time series with arbitrary numbers of variates; and (3) the probabilistic-forecasting gap between the deterministic outputs of vision models and the requirement for uncertainty-aware probabilistic predictions. 
To bridge these gaps, we propose VisonTS++, a TSFM based on continual pre-training of a vision model on large-scale time series.  Our approach introduces three key innovations:  (1) vision-model-based filtering to identify high-quality sequences to stabilize pre-training and mitigate modality gap;  (2) colorized multivariate conversion, encoding multivariate series as multi-subfigure RGB images to enhance cross-variate modeling;  (3) multi-quantile forecasting, using parallel reconstruction heads to generate quantile forecasts without parametric assumptions. 
Experiments show that VisionTS++ achieves state-of-the-art performance in both in-distribution and out-of-distribution forecasting, outperforming specialized TSFMs by 6%–44% in MSE reduction and ranking first in GIFT-Eval benchmark which comprises 23 datasets across 7 domains.  Our work demonstrates that with appropriate adaptation, vision models can effectively generalize to TSF, thus advancing the pursuit of universal TSFMs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work explores the application potential of vision pre-trained model as time series foundation model. Addressing the challenges in transferring vision models to time series tasks—including 1) the data-modality gap between image data and time series, 2) the multivariate-forecasting gap between RGB channels in vision models and time series structure, and 3) the probabilistic-forecasting gap between vision models and time series predictions—this research proposes the ​VisionTS++​​ model. By leveraging vision models, VisionTS++ represents multivariate time series as multiple RGB sub-images to enhance cross-variable dependency modeling and identify salient temporal features. Additionally, it incorporates multi-quantile probabilistic forecasting. This work further investigates the effectiveness of adapting vision models as temporal foundation models.

### Strengths
1.The proposed multi-quantile forecasting provides a solution for converting images into probabilistic predictions that reflect the uncertainty in time series forecasting.

2.The "Vision-Model-Based Filtering" applied to time series appears to be a reasonable and necessary processing step, ensuring that the unbounded nature of time series values can adapt to the bounded pattern of raw image pixels.
3.Extensive experiments on multiple datasets (including the Monash Benchmark, LTSF, and probabilistic forecasting datasets) and competitive results in both in-distribution forecasting and out-of-distribution forecasting validate the model's strong performance.

4.The paper is well-written, complete, and easy to follow.

### Weaknesses
1.While the Color-as-boundary strategy appears reasonable, it may exhibit inherent limitations when the number of time series variables significantly exceeds the available RGB channels: On one hand, the intrinsic properties of RGB channels may introduce implicit color bias, causing the model to overemphasize variables associated with certain channels and impair the balance of feature extraction. On the other hand, color reuse may lead to erroneous associations between non-adjacent variables, resulting in the model capturing false dependencies and interfering with the modeling of genuine variable relationships. Moreover, the randomness in color assignment lacks consideration of variable semantic correlations, which can easily lead to scenarios where highly correlated variables are assigned vastly different colors or weakly correlated variables share similar colors, thereby undermining the model's stability and robustness in high-dimensional settings.

2.Although the "Vision-Model-Based Filtering" serves as a reasonable truncation method to reconcile the numerical characteristics of time series with the input compatibility of vision models, it remains questionable whether such normalization adequately accounts for the diversity of time series distributions. For instance, in domains such as healthcare and finance, time series may contain extreme values of significant importance or exhibit upward or downward trends. It is unclear whether the filtering approach remains compatible in such cases. Additionally, other filtering methods worthy of consideration could be explored to accommodate a broader range of time series types.

3.While VisionTS++ explores more suitable input representations for time series data and improvements to the vision model's output head, the MAE backbone remains largely unmodified. Investigating the potential of more diverse vision backbones for time series applications may help enhance forecasting performance. Furthermore, incorporating comparisons with recent vision-based related work [1] would strengthen the rigor and comprehensiveness of the study.

[1] Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting.  ICML2025

### Questions
In Weakness.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes VisionTS++, a new time-series foundation model that adapts pretrained vision backbones through continual pre-training on large-scale time-series data. The authors identify three critical challenges—data-modality gap, multivariate-forecasting gap, and probabilistic-forecasting gap—when transferring vision models to time-series forecasting. VisionTS++ addresses these issues via three key components: Vision-Model-Based Filtering, Colorized Multivariate Conversion, and Multi-Quantile Forecasting. The proposed method achieves strong results on 23 datasets across multiple benchmarks, showing consistent improvement over other TSFMs.

### Strengths
1. The paper clearly identifies major limitations when applying vision models to time-series forecasting and systematically attempts to address them.
2. Incorporating probabilistic forecasting into the framework is novel and refreshing, extending beyond conventional deterministic designs.

### Weaknesses
1. The study only evaluates MAE-based VisionTS++, without validating other vision backbones(SimMIM, BootMAE, etc.). This limits the generality of the proposed framework.
2. The paper does not discuss the computational cost of continual pre-training, raising concerns about training efficiency and scalability.

### Questions
1. VisionTS++ appears to use longer or variable look-back windows compared to baselines. Does this lead to unfair comparisons? Moreover, how does the model’s performance vary with respect to input window length?

2. The Vision-Model-Based Filtering step discards samples incompatible with vision model constraints. Does this imply that VisionTS++ cannot handle such series at all? The normalization parameter γ seems manually set—could adjusting it recover those filtered samples?

3. The Colorized Multivariate Conversion introduces several potential issues:

   a) The artificial spatial adjacency between variables might induce bias, as neighboring pixels are assumed to be correlated in image models.

   b) The color assignment scheme lacks theoretical justification and could introduce additional bias.

   c) For high-dimensional series (M > 224), some variables cannot be represented within a fixed image size. How is this issue handled?

4. Using multiple heads in Multi-Quantile Forecasting could exacerbate the computational burden of VisionTS++. Has the efficiency–accuracy trade-off been analyzed?

5. As shown in Figure 2 and Appendix E, the model seems to rely heavily on periodic horizontal replication patterns. If the image conversion were not aligned with the true periodicity, would VisionTS++ still perform effectively?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To address the data–modality gap, the multivariate–forecasting gap, and the probabilistic–forecasting gap, this paper proposes VisionTS++, a vision-model-based time series foundation model trained through large-scale pretraining. VisionTS++ consists of three key components: vision-model-based filtering, colorized multivariate time series conversion, and multi-quantile forecasting. After pretraining, VisionTS++ achieves strong forecasting performance in both in-distribution and out-of-distribution scenarios.

### Strengths
1. The paper proposes a new vision-model-based time series foundation model, which achieves good forecasting performance across multiple benchmarks.
2. The paper clearly identifies the key challenges of applying vision models to time series analysis, including the data–modality gap and the multivariate–forecasting gap.
3. The overall writing of the paper is clear and well-organized.

### Weaknesses
1. The paper presents an incremental improvement over VisionTS, with the proposed modules—vision-model-based filtering, colorized multivariate conversion, and multi-quantile forecasting—being relatively straightforward. The filtering module performs simple threshold-based filtering; the multivariate conversion resembles prior vision-based time series models such as ViTST (NeurIPS 2023); and the multi-quantile forecasting capability has already been incorporated in most recent TSFMs.
2. Regarding vision-model-based filtering, the method relies on a simple threshold mechanism, which may inadvertently remove large but legitimate variations (e.g., those caused by major events or holidays), potentially leading to inaccurate predictions in such cases.
3. For multivariate conversion, the method concatenates different time series variables into an image representation. While the vision model may implicitly capture variable dependencies, the approach does not explicitly model variable correlations, which could lead to suboptimal channel correlation modeling. It would be helpful if the authors could provide further evidence or analysis to support the model’s effectiveness in this regard.
4. Table 1 and 2 lacks comparisons with more recent baselines, such as Sundial (ICML 2025).
5. The paper also lacks an efficiency analysis compared with other TSFMs.

### Questions
See Weaknesses.

### Soundness
3

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
VisionTS++ adapts a Masked Auto-Encoder (MAE) pre-trained on ImageNet to universal time-series forecasting via three tweaks: (i) vision-based filtering that discards sequences whose normalised values fall outside the pixel range expected by the MAE, (ii) “colourised” multi-subfigure images that stack an arbitrary number of variates into one 3-channel picture, and (iii) some parallel reconstruction heads trained with quantile loss to deliver probabilistic forecasts without parametric assumptions. After continual pre-training on 231 B points (LOTSA) the model is evaluated on Monash (29 data-sets), LTSF, PF and GIFT-Eval benchmarks, beating recent TSFMs by 6–44 % MSE and ranking first on most tasks.

### Strengths
(1) first paper that systematically closes the data-range, multivariate and probabilistic gaps when turning an off-the-shelf vision backbone into a competitive TSFM.
(2) no new attention layers, no patch re-design; only lightweight heads and input/output converters—easy to reproduce.
(3) SOTA on 4 widely used benchmarks (31/62 first places on LTSF, best nMAE on Monash, top CRPS on PF, 1st on GIFT-Eval) with both base and large variants.
(4) removing filtering (−7 %), colourisation (−12 %) or multi-quantile heads (−10 %) consistently hurts, validating each gimmick.

### Weaknesses
(1) the core idea (TS ➔ image ➔ MAE) is identical; improvements come from three engineering accessories rather than a new modelling principle.
(2) no analysis of why pixel-range filtering or random RGB boundaries should be optimal; no guarantee that vision inductive biases align with temporal dynamics.
(3) only forecasting; classification, anomaly detection or irregular sampling not tested.
(4) no study on (i) #quantile heads h, (ii) alternative change-point or range-based filters, (iii) image size W or subfigure layout, (iv) datasets outside LOTSA/GIFT.
(5) several typos (“appedix”, “quarntie”), some tables missing std-dev, captions duplicated.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
