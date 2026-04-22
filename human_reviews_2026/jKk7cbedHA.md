# Order-Preserving Pattern Mining Enhances Structure-Aware Time Series Forecasting

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 0

## Abstract
Traditional time series forecasting models tend to focus on numerical fitting, mak-
ing it difficult to explicitly model and leverage the relative ordering patterns in-
herent in time series. This often results in suboptimal predictions when dealing
with data segments that exhibit clear pattern regularities. To address this gap, this
paper introduces Order-Preserving Patterns (OPPs) into time series forecasting for
the first time and proposes a novel model that explicitly incorporates prior pattern
knowledge by leveraging frequent OPPs as explicit priors. The proposed model
utilizes a convolutional neural network to perform feature dimensionality reduc-
tion on high-dimensional labeled time series, extracting one-dimensional repre-
sentations suitable for pattern mining. It then applies a sliding window and sup-
port counting strategy to discover frequent OPPs. An OPP matching mechanism
is proposed to distinguish between OPP and non-OPP training samples. Addition-
ally, a pattern constrained loss function is designed to guide the predicted values
toward consistency with the prior pattern logic. This constraint is imposed from
three perspectives—right boundary, left boundary, and intermediate positions—to
ensure order alignment with the tail elements of the OPPs. Experimental results
show that under the 'Perturbation Boundary' window sizes across ten real-world
and public benchmark datasets, the proposed OPPCL model consistently achieves
substantially lower MSE compared with state-of-the-art methods. In particular,
it yields at least 31.45% and 37.30% reductions on the SWaT and Electricity
datasets, respectively. The improvement becomes more pronounced when the
window size exceeds the 'Perturbation Boundary'. Code is available at this repos-
itory: https://anonymous.4open.science/r/OPPCL-B070/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Order-Preserving Patterns (OPPs) into time series forecasting and proposes the OPPCL (Order-Preserving Pattern Constrained Learning) model. The model employs a CNN for dimensionality reduction of high-dimensional labeled time series, uses a sliding window and support counting strategy to mine frequent OPPs, designs an OPP matching mechanism to distinguish between OPP and non-OPP training samples, and develops a pattern-constrained loss function (incorporating right-boundary, left-boundary, and intermediate position constraints) to align predictions with prior pattern logic.

### Strengths
1.This paper propose a pattern-aware forecasting model OPPCL, which explicitly introduces frequent OPPs as structural priors in the time series forecasting task, this paper design a position-sensitive pattern-constrained loss function, which explicitly supervises the relative order of the model outputs；
2.The method proposed in the paper has proven its effectiveness on SWAT and ETTh1 datasets；
3.Most parts of the paper are comprehensible.

### Weaknesses
1.The method proposed in the paper is only tested on two datasets. Current mainstream time series research typically validates models on more than 10 datasets, so testing on merely two datasets is far from sufficient;
2.Using only MSE as the evaluation metric is also insufficient. In Table 2, the optimal results under certain settings are not highlighted in bold;
3.The figure quality of the paper needs improvement: the text in many figures is too small to read clearly; Figure 1 lacks strong persuasiveness; much text in Figure 2 is outside the text boxes; many numerical labels in Figure 3 are also outside the graph, and the x-axis and y-axis labels are not marked;
4.The paper contains an excessive number of formulas. Many general formulas (such as the calculation of attention) do not need to be listed, and many formulas lack punctuation marks;
5.The dimensionality reduction strategy lacks consistency and detailed validation. The paper uses CNN for SWaT (51-dimensional data) and PCA for ETTh1 (7-dimensional data) but does not justify this choice (e.g., why not use CNN for ETTh1 or PCA for SWaT?);
6.The pattern constraint loss function’s hyperparameter setting lacks sensitivity analysis. The total loss uses identical weights for both datasets, but the paper does not explore how varying these hyperparameters influences model performance;
7.Please explain the sentence in the Conclusion section: "Furthermore, OPPCL enhances the attention focus of baseline Transformer-based models, indicating improved interpretability"; 
8.The references and notation consistency require refinement. Some references lack complete publication details (e.g., "Wang & Chen" for LSTNet is not fully cited in the References section).

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces OPPCL, a time series forecasting model designed to be "structure-aware" by explicitly incorporating Order-Preserving Patterns (OPPs). The core idea is to mine frequent relative orderings from the training data and then use a novel pattern-constrained loss function to penalize predictions that violate these learned structures.

### Strengths
The main strength of this work lies in its conceptual novelty. Moving beyond implicit structure learning to an explicit, pattern-based constraint is a significant and promising direction, especially for industrial time series where underlying processes often dictate such ordinal logic. The proposed constraint loss is an intuitive and direct implementation of this idea. The authors support their claims with a comprehensive set of experiments against numerous baselines, and the ablation study effectively demonstrates the utility of the proposed constraint mechanism.

### Weaknesses
However, the paper suffers from several critical weaknesses that currently undermine the validity and generality of its contributions. The most significant issue is a lack of clarity and consistency in the core methodology. The dimensionality reduction step—a crucial prerequisite for pattern mining—is described as a CNN trained on unexplained "timestamp-level labels" for one dataset, while a simple PCA is used for another. This inconsistency feels ad-hoc and makes the framework seem less like a generalizable solution.

The reported performance improvements on the SWaT dataset are so extraordinary (orders of magnitude better than strong baselines) that they raise concerns about the experimental protocol. It is essential to confirm that the OPP mining process was conducted strictly on the training set, with no leakage from the test data. Without this confirmation, the results are difficult to trust. Finally, the "Perturbation Boundary" concept, invoked to explain why the model's advantage is limited to specific input lengths, feels like a post-hoc rationalization for a potential lack of robustness rather than a deeply analyzed phenomenon.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Order-Preserving Patterns (OPPs) for time series forecasting and leverages frequent OPPs as explicit priors. Experiments on 2 datasets show improvements.

### Strengths
1. This paper is easy to follow.
2. Visualization and showcases. 
3. Code are provided for reproducibility.

### Weaknesses
1. Many related works are missing, such as [1-2] which also consider similar Order, Shapelet or Shape for time series modeling. The claimed contributions and novelty are limited.

[1] TimeCSL: Unsupervised Contrastive Learning of General Shapelets for Explorable Time Series Analysis. PVLDB 2024.

[2] Shape analysis for time series. NeurIPS 2024.

[3] Revitalizing multivariate time series forecasting:Learnable decomposition with inter-series dependencies and intra-series variations modeling. ICML 2024.

2. Key experimental comparison for baselines [1-4] is missing. 

[4] Deep Time Series Forecasting With Shape and Temporal Criteria.

3. More benchmarks, such as Traffic / Electricity / Weather, and metrics, such as MAE, should be considered. The authors should justify their choice of benchmarks and metrics, and explain why they believe their current selections are sufficient or how they plan to expand their evaluation.

4. The author should also conduct experiments and compare results on long-term forecasting. And the used baselines are out-of-date. Please compare state-of-the-art baselines such as [3].

### Questions
see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The authors used an incorrect template (wrong page margin)?

### Strengths
N/A

### Weaknesses
N/A

### Questions
N/A

### Soundness
3

### Presentation
1

### Contribution
2
