# ReST: Remarkably Simple Transferability Estimation

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Existing transferability estimation methods for pre-trained neural networks suffer from method complexity, requiring extensive target data and labeled samples to predict transfer performance. We introduce ReST, a remarkably simple yet effective approach: It only requires a small subset of unlabeled samples from target data and analyzes the stable rank—a robust measure of matrix effective dimensionality—of the final layer representations. We demonstrate that this simple metric strongly correlates with transfer learning success across diverse tasks and architectures.  Through comprehensive experiments on vision transformers and CNNs across multiple downstream tasks, we show that this remarkably simple approach not only matches but often exceeds the performance of sophisticated existing methods. ReST achieves 4.6\% improvement over state-of-the-art methods, establishing stable rank as a powerful predictor for transferability assessment and fundamentally challenging the need for complex analysis in transfer learning evaluation. The code is made anonymously available at https://anonymous.4open.science/r/random-07C2 to ensure reproducibility of our results.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ReST (Remarkably Simple Transferability Estimation), a novel method designed to predict the fine-tuning performance of pre-trained neural networks on target tasks without requiring extensive experimentation or labeled target data. ReST is positioned as a simple yet effective approach that analyzes the geometric structure of representations, specifically using stable rank—a measure of matrix effective dimensionality. The ReST score combines two components: the Generalization (G-Score), based on the stable rank of the penultimate ($W_p$) and classifier ($W_c$) weight matrices to measure intrinsic capacity, and the Learnability (L-Score), which quantifies adaptation flexibility by measuring the relative change in stable rank of activation features between source and target domains. The method is computationally efficient, reducing to simple matrix norm calculations.

### Strengths
**Geometric Insight**: The approach is fundamentally insightful, linking transfer success to two geometric properties captured by stable rank: intrinsic generalization capacity (G) (distributed weight spectra) and adaptation flexibility (L) (moderate activation shifts). This provides valuable insight into why certain representations transfer well.

**Remarkable Data Efficiency**: ReST is a label-free estimator requiring only a small, unlabeled sample subset (effective size equivalent to $\approx 2$ samples per class). This addresses a key practical limitation of prior methods like LEEP and LogME, which require substantial labeled data.

### Weaknesses
**Hyperparameter Sensitivity and Complexity Trade-off**: The "Remarkably Simple" title is somewhat undermined by the reliance on architecture-specific tuning of the layer balance parameter ($\alpha$) and adaptation weight ($\gamma$) for optimal performance. For instance, optimal parameters shift significantly between supervised CNNs ($\alpha=0.51, \gamma=0.21$) and self-supervised Vision Transformers ($\alpha=0.885, \gamma=0.650$). Furthermore, zero-shot performance requires changing $\alpha$ from 0.51 to 0.43 to achieve the highest correlation (0.687). This suggests that a user needs to pre-determine the model class or data regime to select the optimal configuration, adding a layer of hidden complexity to the "simple" methodology. It’s also unclear how these values are chosen, if these values were chosen based on correlation with downstream test accuracy, this is unjustified. Please shed light on the exact procedure for selecting these hyperparameters in the particular problem setting.

**Ranking Inconsistencies**: The heatmap analysis (Figure 4) explicitly reveals that ReST shows a larger mismatch (ranking inconsistency) on specific datasets like Flowers-102, and consistently underestimates the DenseNet family while overestimating the ResNet family. While overall performance is SOTA, these systematic biases warrant deeper investigation.

**Relevant Literature missing**: Some relevant work on transferability measures is missing from related work. E.g., H-Score [Bao 2019, Ibrahim 2022]. Please include these in the related work. If possible, include these in the comparisons as well.

**Typos and Formatting Issues**
- **Source Labeling/Numbering**: The text references "Table 5(b)", but the corresponding comparison table is labeled "Figure 5: (b) Runtime comparison...". This confusion between table and figure numbering should be corrected for consistency (e.g., Table 5).
- **Reference to Figure 5(a)**: In Section 4.3.2, the text discusses "Figure 5(a) shows the effect..." but Figure 5 is located much later. This placement or numbering should be improved.

*References*:
- Yajie Bao et al. “An Information-Theoretic Approach to Transferability in Task Transfer Learning.” 2019 IEEE International Conference on Image Processing (ICIP) (2019): 2309-2313.
- Shibal Ibrahim, Natalia Ponomareva, and Rahul Mazumder. 2022. Newer is Not Always Better: Rethinking Transferability Metrics, Their Peculiarities, Stability and Performance. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2022.

### Questions
**Hyperparameter Generalization**: The optimal parameters ($\alpha, \gamma$) vary significantly based on the model type (CNN vs. ViT) and data availability (zero-shot vs. few-shot). Can the authors propose a robust, default setting for $\alpha$ and $\gamma$ that performs consistently well across all architectures and regimes (perhaps slightly below the peak optima) to truly deliver a "Remarkably Simple" out-of-the-box solution without model-specific tuning?

**Normalization of L-Score**: The normalization of scores across the models seems very hacky to me. Isn’t there a better way of normalization e.g., for $L$ relative change divided by source stable rank, especially given that activation stable rank values might vary widely depending on the model's depth or architecture? Or some other alternatives?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a transferability estimation (TE) method named ReST for evaluating the downstream task transferability of pre-trained models. The core idea of ReST is to analyze the generalization score (G-Score) and learnability score (L-Score) of the final two layers of the model, achieving SOTA results with a small computational cost using only a small amount of unlabeled data.

### Strengths
ReST is simple and does not use annotations.

### Weaknesses
-- ReST requires the use of a classifier layer, which tends to limit its wider application, such as in self-supervised models like MAE and DINO that do not have a classifier layer.

-- The method presented in this paper is relatively innovative. It mainly focuses on the combination and extension of existing methods (Stable rank).

-- The figures are blurry, the font size within the images is too small, and the layout of elements within the figures is unclear.

-- The ablation studies seem to show that ReST is not stable (i.e., it is quite sensitive to hyperparameters). Although Figure 5(a) illustrates the impact of the two hyperparameters, Table 2 shows that a small change in $\alpha$ can cause large fluctuations in results for specific datasets. For instance, when $\alpha=0.43$, ReST performs significantly better on the Aircraft and Cars datasets, but its performance drops markedly on the Flowers dataset.

-- Although most TE methods require using a large amount of downstream data to input and extract features while ReST only needs two unlabeled samples, other TE methods achieve plug-and-play based on features. In contrast, ReST needs to access the model's internal weights and activation values, which might be a disadvantage.

-- The paper's intuition for the L-Score appears to contradict its final formulation. In Section 3.2.2, the authors state that moderate representational changes indicate healthy adaptation, whilea very large L-Score could signal catastrophic forgetting. However, the final ReST score (Eq. 4) scales the L-Score with a simple positive coefficient ($\gamma$).

### Questions
-- Although the experimental setup of this paper is quite similar to previous methods, I believe that CNN and ViT should not be evaluated separately. How about combining the experiments for CNN and ViT? This is closer to the actual usage scenario. No additional experiments are required. It is sufficient to combine the experimental results and recalculate the Kendall coefficient.

-- Existing self-supervised ViT models have been widely applied. How can ReST be utilized in these models? Have you considered evaluating on these newer and better models?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose a metric for source-independent transferability estimation, REST. REST allows for label-free computations, and authors also show the performance of REST over other metrics in their experiments.

### Strengths
The authors suggest a very simple score that predicts the transferability of selected models with very high confidence. The results demonstrate good modeling and robustness, covering a range of metrics. The ablation studies demonstrate the effectiveness of rest and present it interactively. The time to compute it is also low. The idea of using stable rank as a basis for transferability estimation is interesting and simple.
* The approach works in a label-free and few-shot setting, which is practically valuable.
* Ablation studies and hyperparameter analyses are comprehensive and highlight stability across sample sizes.

### Weaknesses
* Missing comparison with TransRate and EMMS, both established baselines in recent literature.
* The evaluation lacks Pearson correlation results, which would clarify the metric’s linear consistency.
* I also think the model and dataset space use is very outdated, though used in recent papers this experimental space does not allow for fair comparison as ResNet151 is always the best performing network and the static ranking of the models space is indeed an issue to judge the effectiveness of any metric on this benchmark, can authors also show the effectiveness in two other ways to show confidence in the metric, use setup similar to [1] for broader evaluation, use the ablations similar to [2] for robustness evaluation, Authors can also provide ablation studies as shown in [2] for a better robustness of their metric.  Though it has become a standard practice to use older finetuned scores from other papers i would advise against it and retrain pretrained models multiple times to account for variance. 
* The paper’s experimental scope is limited to supervised image classification; no results are shown for regression, detection, or multimodal settings (e.g., GLUE for NLP).
* It remains unclear whether ReST can meaningfully capture absolute versus relative performance differences.
* The analysis of failure cases (e.g., Flowers, Food datasets) is superficial understanding why the metric underperforms could strengthen the argument.
* The claim of handling unlabeled data is overstated, since training and evaluation still involve supervised datasets.
* The writing lacks intuitive reasoning behind why richer representational geometry (higher stable rank) enhances transfer.
* The paper does not explore how ReST behaves when combining CNNs and ViTs within a unified model selection scenario.

[1] k-NN as a Simple and Effective Estimator of Transferability, TMLR

[2] How NOT to benchmark your SITE metric: Beyond Static Leaderboards and Towards Realistic Evaluation.  https://arxiv.org/pdf/2510.06448

### Questions
1. Include TransRate, EMMS, and Pearson correlation in comparisons.

2. Explore scenarios where only unlabeled datasets are available during both training and transferability estimation.

3. Evaluate on non-vision domains or multi-task settings.

Some sentences that should be rewritten to avoid vagueness: This component captures the geometric reorganization of representations across domains, this sentence is too dense and does not really help the reader.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ReST, a simple yet effective method for estimating transferability using the stable rank of final-layer representations. It requires only a few unlabeled target samples and avoids complex optimization. The approach captures both a model’s generalization capacity and adaptation flexibility through basic matrix operations. Overall, I think this is a good paper.

### Strengths
- The method is conceptually clean, relying only on stable rank computation, and requires minimal unlabeled target data.

- This method can estimate transferability in both few-shot and even zero-shot settings, which is highly practical, and has received little attention in existing methods.

- The connection between stable rank, generalization bounds, and representation geometry provides a theoretical foundation.

### Weaknesses
- The idea of using spectral or rank-based measures has been explored; the contribution mainly simplifies existing approaches. Although this method is effective, the novelty and motivation require further explanation.

- Experiments are confined to image classification; generalization to other modalities or tasks remains unexplored.

- The experiments could be conducted on more model selection tasks, such as self-supervised algorithm selection and more metrics such as Pearson correlation, Top-1 accuracy, etc.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
