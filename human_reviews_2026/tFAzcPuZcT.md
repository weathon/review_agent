# RainPro-8: An Efficient Deep Learning Model to Estimate Rainfall Probabilities Over 8 Hours

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
We present a deep learning model for high-resolution probabilistic precipitation forecasting over an 8-hour horizon in Europe, overcoming the limitations of radar-only deep learning models with short forecast lead times. Our model efficiently integrates multiple data sources - including radar, satellite, and physics-based numerical weather prediction (NWP) - while capturing long-range interactions, resulting in accurate forecasts with robust uncertainty quantification through consistent probabilistic maps. Featuring a compact architecture, it enables more efficient training and faster inference than existing models. Extensive experiments demonstrate that our model surpasses current operational NWP systems, extrapolation-based methods, and deep-learning nowcasting models, setting a new standard for high-resolution precipitation forecasting in Europe, ensuring a balance between accuracy, interpretability, and computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces RainPro-8, a deep-learning model for probabilistic precipitation forecasting. It proposes an Ordinal-Consistent Loss that maintains the natural ordering of rainfall intensity classes and a Single-Pass Prediction method that predicts all future timesteps at once for greater efficiency and temporal consistency. Built on a U-Net with MaxViT blocks, the architecture efficiently processes multi-resolution atmospheric inputs with significantly fewer parameters than prior MetNet models.

### Strengths
- The paper identifies the limitations of radar-only forecasting methods and proposes a multi-source approach to improve prediction accuracy.
- It provides a detailed analysis of recent Climate AI research, highlighting their characteristics and limitations, and conducts experiments following established protocols and evaluation metrics.
- To address the ordinal nature of precipitation classes and the difficulty of long lead-time forecasts, it introduces a specialized loss function and a single-pass prediction method.
- The appendix provides comprehensive details on the variables, data assimilation, and preprocessing steps for the diverse data sources used.

### Weaknesses
- Including the baseline model MetNet, RainPro-8 relies on a variety of data sources, which limits its applicability to regions—such as developed countries—where high-resolution data from multiple sensors, satellites, and NWP outputs are readily available.

- The model lacks significant technical novelty. Moreover, its performance improvement over the existing baseline, MetNet-3, is marginal.

### Questions
- Many models that perform 6-hour nowcasting are discussed in the Related Works section. What is the logical reasoning and supporting justification for forecasting 8 hours instead of 6, as done in those prior works?
- Besides SEVIR, is it possible to evaluate performance on other relevant benchmark datasets such as the Shanghai dataset or CIKM?
- In Line 109, the paper states that “its training requires significant time and resources, involving hundreds of Tensor Processing Units (TPUs) for multiple days.” However, if this method uses an “NVIDIA H100 SXM5 GPU,” isn’t that also an extremely powerful computational setup? Please summarize the computational power and training duration for the compared baseline methods.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents RainPro-8, an efficient and compact deep learning model for 8-hour high-resolution probabilistic precipitation forecasting in Europe. RainPro-8 integrates heterogeneous multi-source data (radar, satellite, and numerical weather prediction, NWP), and adopts an optimized U-Net + MaxViT architecture. The model generates probabilistic maps for all time steps and precipitation classes in a single forward pass. The authors introduce an "ordinal consistent loss" to explicitly encode the ordinal nature of precipitation bins, improving interpretability and consistency of the probabilities. RainPro-8 uses less than 20% of MetNet-3's parameters but achieves superior performance across lead times and precipitation intensities, outperforming NWP baselines, extrapolation methods, and the latest deep learning models (e.g., Earthformer, SimVP, MetNet-3*). Extensive experiments, including ablations, attribution, and efficiency analysis, are provided, as well as competitive results on the short-term SEVIR benchmark. Code will be public, and datasets and experimental details are thorough.

### Strengths
RainPro-8 efficiently combines heterogeneous multi-source data, achieves multi-step probabilistic forecasting in one pass with much fewer parameters than strong baselines, and its ordinal consistent loss is specifically designed for the nature of precipitation bins. The model consistently outperforms state-of-the-art competitors on accuracy, efficiency, and uncertainty quantification.

### Weaknesses
The “ordinal consistent loss” (Section 3.2) is a core novelty, but its theoretical justification is limited. The paper gives only a high-level description and refers to another field (semantic segmentation) for motivation. There is no theoretical or empirical exploration of why ordinal modeling is essential for probabilistic precipitation. Ablation is very basic, and critical aspects such as loss stability, parameter sensitivity, and generalization are not deeply examined.

While RainPro-8 claims to handle varying precipitation intensities, most experiments focus on average metrics. The model’s ability to capture extreme/rare strong precipitation events (e.g., ≥25 mm/h) is not thoroughly assessed. Only the class coverage is reported, but there are no case studies or targeted metrics for extreme events, missing a rigorous quantification of robustness or uncertainty calibration for outlier scenarios.

### Questions
does the ordinal consistent loss have any theoretical advantage (e.g., calibration, capturing uncertainty, or predicting extremes)? Could you provide more empirical or theoretical analysis on loss stability and generalization?

In known European extreme rainfall cases or strong out-of-distribution samples, how does RainPro-8's predicted probability align with actual precipitation? Are confidence intervals and rare event probabilities calibrated and unbiased?

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
The paper describes a  deep-learning model for high-resolution probabilistic precipitation
forecasting over an 8-hour horizon. The main advantage of the proposed model is in its ability to predict precipitation for a longer time frame than the standard nowcasting, up to 2 hours. To achieve this goal, the authors fuse radar, satellite, and physics-based nu-
merical weather prediction (NWP) data.

### Strengths
The paper covers an important topic, precipitation forecasting. 

Originality: the paper leverages multi-source prediction and claims prediction of up to 8 hours (however, see the weaknesses below)

Quality: the paper is well-written in general (however, there are caveats as some parts of the paper are difficult to follow). The equations, as I checked, are correct

### Weaknesses
Clarity and quality: see questions below

Significance: I think the authors need to clarify on this point.  The contributions cite the following:
-  *Efficient architecture and training strategy*: however, the architecture seems to be a well-parameterised UNet-based model (Figure 1). The authors state: "Key differences include single-pass predictions without lead time conditioning (Section 3.3), early downsampling in the encoder, halving internal channels, and removing topographical embeddings, all contributing to a reduced parameter count of 36.7M from the original 227M." Would this be the contribution? I think it could be better to have some sort of takeaway message justifying these architectural solutions, and why it could help develop better new precipitation forecasting architectures
- "significant gains over deep-learning nowcasting models" See Q1
- "Demonstration of RainPro’s versatility for radar-only 2-hour predictions on the SEVIR benchmark, achieving state-of-the-art performance compared to both deterministic and generative nowcasting models" I am not entirely sure I get it. If the claim is to get 8 hour prediction, is achieving state-of-the-art results on 2-hour radar predictions of the contribution?

### Questions
1. "Extensive empirical evaluation demonstrating that RainPro-8 outperforms existing operational methods by 65% " I am not sure I can follow where this is described, it only appears in the introduction
2. I am not sure I can follow Figure 2, it is very small. From what I can follow, does the proposed method essentially follow very closely MetNet-3*, is that correct? In that case, the main claim could be that you achieve slightly higher performance, but reduce the computational costs  48 times.
3. For Table 3, the ablation study results show only small differences in performance. Could the authors give confidence intervals, if possible?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose RainPro-8, a deep learning model for high-resolution, 8-hour probabilistic precipitation forecasting over Europe. The model is based on the MetNet-3 architecture but is modified to have significantly fewer parameters. It introduces a *single-pass prediction* strategy to improve inference efficiency and an Ordinal Consistent Loss to handle probabilistic bins. The authors claim the model surpasses existing deep learning and numerical weather prediction methods on several metrics.

### Strengths
1. The paper addresses the challenge of 8-hour, high-resolution probabilistic precipitation forecasting. This is a critical and difficult task that bridges the gap between traditional nowcasting and medium-range forecasting.
2. The usage of Ordinal Consistent Loss models the conditional probability of exceeding intensity thresholds, is designed to explicitly account for the ordinal structure of precipitation classes , which is a more principled approach than using a standard cross-entropy loss that treats classes as independent.

### Weaknesses
1. **Limited Methodological Novelty (*Main Weakness*)**: The primary weakness of this paper lies in its limited methodological novelty relative to ICLR standards, which emphasize fundamental advances in learning representations.
- Aside from the new loss function, the work's primary contribution is an application of existing techniques to create an efficient system.
- The model architecture is just based on MetNet-3. The main architectural changes include *early downsampling* and *halving internal channels*. These are standard engineering practices for model compression and efficiency, not **novel architectural designs or new methods for learning representations**.
2. **Lack of comparison on GAN-based models**: The paper's experimental validation lacks a crucial component in its discussion of *GAN-based generative models for precipitation*. In the related work section (Section 2), the authors correctly identify the importance of deep generative models, citing high-impact work such as DGMR (Ravuri et al., 2021) and NowcastNet (Zhang et al., 2023). These models, both published in *Nature* and recognized for their strong performance, are a key pillar of the state of the art in this field.
Despite acknowledging this work, the main experimental comparison in Table 1 completely **omits comparisons against these (or any other) GAN-based generative methods**. This is a significant gap in the evaluation. Without benchmarking against this entire class of SOTA models, it is impossible for the reader to assess the paper's true performance, and the claim to "surpass... deep-learning nowcasting models"  remains unsubstantiated.
3. **Contradiction Between Quantitative Metrics and Qualitative Results**: A significant weakness undermines the paper's experimental conclusions: the quantitative metrics and the qualitative case studies appear to contradict each other directly.
- Quantitative Metrics: The paper's aggregated metrics, specifically the Frequency Bias Index (FBI) in Table 7 and Figure 8, suggest a key advantage for RainPro-8 in forecasting heavy precipitation. At the 10.0 mm/h threshold, RainPro-8 reports an FBI of 1.636, which is notably lower (i.e., less over-forecasting) than the 1.821 reported for MetNet-3*. **This metric suggests the author's model is more balanced**.
- Case Study Results: The paper's own qualitative visualizations repeatedly show the opposite. In the case study for the +8h forecast in Figure 19, the Ground Truth shows only a small area of precipitation at the >10.0 mm/h (yellow) level. However, **RainPro-8 predicts a large, distinct area of heavy rain, while the MetNet-3 forecast for the same event shows a substantially smaller overforecasted area**. This same pattern, where RainPro-8 visually over-forecasts heavy rain more severely than MetNet-3, is apparent in the other examples provided (Figure 17).
This fundamental discrepancy, in which all qualitative examples in the paper contradict the aggregate metrics for heavy rain, is neither acknowledged nor explained. This casts serious doubt on the reliability of the reported results and the validity of the paper's evaluation.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1
