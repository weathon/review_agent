# Toward Robust Feature Space in Long-Tailed Time Series Classification: A Multi-Scale Perspective

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
In recent years, time-series classification (TSC) has seen significant progress. Nevertheless, research on long-tailed TSC remains relatively limited. A key issue in long-tailed scenarios is that high inter-class similarity often leads models to learn overlapping features, making tail classes particularly difficult to distinguish. This phenomenon gives rise to three specific challenges: (1) Conventional approaches based on oversampling or uniform-intensity data augmentation may overfit or fail to learn robust features for tail classes. (2) Limited model representation capacity can lead to aligned temporal features across classes, further exacerbating class confusion. (3) Such class overlap makes it challenging to establish discriminative decision boundaries, particularly in highly imbalanced scenarios. To address these challenges, we propose TimeLT, a novel framework designed to learn a robust and discriminative feature space from long-tailed time-series data. First, we introduce a personalized augmentation strategy that generates tailored perturbations for scarce tail samples, preventing overfitting while increasing sample diversity. Second, we employ a multi-scale temporal encoder to capture patterns at different temporal resolutions, enabling the model to extract informative and discriminative features for both head and tail classes. Third, we propose a boundary-repelling regularization term that encourages embeddings to move closer to their respective class centroids while being repelled from inter-class boundaries, promoting compact and well-separated feature representations. To promote comprehensive research in this area, we consolidate a dedicated benchmark comprising several long-tailed datasets and over 16 advanced baselines. Extensive experiments across all datasets demonstrate that TimeLT significantly outperforms the strongest baselines, achieving accuracy improvements ranging from 0.55\% to 12.27\%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes TimeLT to tackle long-tail time series classification by considering multi-scal temporal encoding, data augmentation, and variant representation learning strategies. TimeLT demonstrates effectiveness across 4 selected datasets with ablation study and parameter analysis.

### Strengths
1. The idea of boundary-repelling regularization $L_b$ sounds interesting and novel to me, which encourages samples to be away to other classes, with the corresponding ablation study and parameter study to support this claim.
2. The paper details the parameters and the training configurations, especially that the authors include the codebase as the supplementary, making this work reproducible.
3. The ablation study and parameter analysis are extensive and comprehensive, with testing variants of each module, hyper-parameters, and the loss designs.

### Weaknesses
1. The motivation about higher inter-class similarity for time series data is not convincing (Fig. 2 with L65-67). While dogs and pandas are different that can be easily separated, if we consider more fine-grained classes (e.g., black bear vs. brown bear), they probably may not be highly separable. Note that this concern is raised since the authors use "walking" and "running" as the time series examples, which is unfair to compare to coarse categories in the image domain. As this is the main (and only) motivation, this work does not sound necessary, even though it may demonstrate performance improvements.
2. While TimeLT demonstrates effectiveness over existing baselines, the experiments are only conducted with 4 datasets; however, CFAMG used 53 time-series datasets for evaluations, causing the experiments in this paper lacking insufficient validation. Additionally, the used 4 datasets are different from the CFAMG paper, which is difficult to evaluate its validity from the tables.
3. The third contribution of releasing a new benchmark remains unclear and questionable since the authors do not describe any motivations about it as well as why existing benchmarks are insufficient (e.g., UCR and UEA in the CFAMG paper)
4. The multi-scale temporal encoding has been explored by existing works, e.g., [1, 2]. However, the authors do not include any related works for comparisons and discussions.
5. The setting for head and tail classes (Table 2) is a bit confusing. Since it is an imbalance task, dividing with 50-50 to represent head and tail could largely cover classes that have similar portions. This raises the validity of interpreting Table 2.
6. [Minor]: Missing reference in L676.

[1] LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters

[2] PromptTSS: A Prompting-Based Approach for Interactive Multi-Granularity Time Series Segmentation

### Questions
Q1: Regarding the claim "representations of tail classes tend to cluster near or even overlap with those of head classes, leading to blurred decision boundaries", could the authors prove or provide references to support this claim?
Q2: What is the reason that many methods suffer from very worse performance for Epilepsy-LT?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes TimeLT, a method designed to address the challenges of imbalanced long-tailed time series classification (TSC).
TimeLT is composed of three main steps. Firstly, data augmentation via oversampling and perturbation is employed to enhance data diversity. Next, a temporal encoder is implemented for time series representation, consisting of a 1D-CNN embedding on multi-scale down-sampled series and a GRU as the backbone encoder. Finally, two regularization terms are applied to refine the decision boundary, with one pushing samples away from the boundary and the other pulling embeddings closer to the corresponding centroid.
Experimental results across multiple datasets, in comparison with 16 baseline methods, demonstrate TimeLT's leading performance on long-tailed TSC. These results are further verified through analysis including ablation, visualization, and sensitivity test.

### Strengths
(1) The problem of long-tailed time series classification is important and timely within the community. The proposed method provides a well-motivated solution. \
(2) The experimental results are compelling and demonstrate the efficacy of the method. \
(3) The structure of the article is clear, and the paper is easy to follow.

### Weaknesses
(1) The ablation is not rigorous. Specifically, in the preprocessing stage, oversampling is a commonplace method when dealing with imbalanced classes. The current "**w/o O&A**" ablation does not clarify whether the improvements stem from oversampling or perturbation. A more thorough ablation should include "**w/o O**", "**w/o A**", and "**w/o O&A**" to ascertain the contribution of each component. \
(2) The novelty of the proposed framework appears to be more of an engineering integration rather than an algorithmic breakthrough, given its reliance on established techniques. \
(3) There are several problems w.r.t. the writing. For example, grammatical faults like "This is can be" and format faults like "**Analysis of** $\beta$" ($\beta$ should be bolded) \
(4) Please refer to the questions for further potential weaknesses.

### Questions
(1) The Imbalance Ratio (IR) is defined as the ratio between the largest and smallest class sample counts. This only considers the two extreme classes. For instance, compare two scenarios: one where class sizes decay linearly from $N_1$ to $N_C$, and another where only $N_C$ drops sharply with the others being similar. Under your IR definition and its application in the method, would there be different effects in these cases? \
(2)  In Eq.(3), the use of c/C presumes a linear decay of sample counts across classes. Is this assumption valid? Or would it be more appropriate to use $N_c$ / $N_C$? \
(3) The author suggests GRU is a better choice in comparison with the Transformer because of the sequential modelling capability. Two questions have arisen w.r.t. this argument. Firstly, there are certain variations of the Transformer models that can model temporal dependencies better than the vanilla Transformer, but these variations are not included in the analysis or the comparable experiments. Second, the author suggests that information leakage caused by the Transformer will cause less accurate modelling. However, variations like causal attention enable the model only having access to the past time steps. At the same time, the whole series is available for TSC, which differentiates it from time series forecasting, so I do not consider that there is an information leakage problem. \
(4) Can Eq.(8) and Eq.(9) be unified in one contrastive loss function that simultaneously pushes away from the boundary (negative sample pair) and pulls closer to the centroid (positive sample pair)? \
(5) The Related Work in Appendix A has only one subsection A.1, and only the long-tail learning is included. Is this part left to be unfinished?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper propose a framework, named TimeLT, to learn a robust feature space from long-tailed data, thereby improving the overall accuracy for long-tailed TSC. Additionally, a benchmark is released, which includes data processing protocols, diverse datasets, and multiple baselines.

### Strengths
1.	The problem addressed in this paper is common and significant within the field of time-series classification.
2.	The authors publish a standardized benchmark that holds significant practical value for promoting fair comparisons and future development within the long-tail TSC domain.
3.	The experiments are comprehensive, demonstrating the proposed method's effectiveness through comparisons with multiple baselines.

### Weaknesses
1.	The components of TimeLT, including perturbation-aware data augmentation, multi-scale temporal encoding, and oundary-repulsion regularization, are well-established techniques that have been extensively studied, which to some extent limits the novelty of this paper. Authors should provide a more detailed explanation of why this combination can address the challenges posed by high inter-class similarity.
2.	This paper lacks discussion of the computational overhead and training/inference time of the TimeLT framework, as multi-scale encoding will inevitably increase computational costs.
3.	The description of the decision boundary embedding set B in Eq.(7) is vague, a more detailed computational process and explanation should be provided.

### Questions
The authors only illustrate inter-class similarity through Figure 2, lacking quantitative analysis and description (such as DTW distance) to demonstrate that it is indeed higher than typical image datasets.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles the problem of long-tailed time series classification, where models often fail under severe class imbalance. The authors propose TimeLT, which integrates (1) a perturbation-aware augmentation to enhance tail-class diversity, (2) a multi-scale temporal encoder for rich feature extraction, and (3) a boundary-repulsion regularization to improve class separability. A new benchmark with 16 baselines is introduced, and experiments show consistent gains, especially on tail classes.

### Strengths
1. The paper tackles an underexplored yet practically relevant problem, long-tailed time series classification, and provides a unified benchmark that may serve as a foundation for future research.

2. Extensive experiments across 16 baselines and multiple datasets convincingly show the effectiveness and robustness of TimeLT.

### Weaknesses
1. The augmentation and regularization strategies appear to reuse existing ideas with limited novelty or justification specific to time-series data.

2. The motivation-to-method alignment is weak and some design choices are introduced abruptly without sufficient reasoning.

### Questions
1. The abstract lists three components of TimeLT, but the unsolved problems they connected to are unclear. 

2.  How does Gaussian noise outperform more structured augmentations such as frequency- or context-based methods?

3. What role does the multi-scale, channel-independent design play in handling class imbalance?

4.  The boundary-repulsion loss resembles margin-based or supervised contrastive objectives. Could the authors clarify its unique contribution and explain how it improves discrimination in long-tailed TSC?

### Soundness
3

### Presentation
3

### Contribution
3
