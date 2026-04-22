# SeBA: Semi-supervised few-shot learning via Separated-at-Birth Alignment for tabular data

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Learning from scarce labeled data with a larger pool of unlabeled samples, known as semi-supervised few-shot learning (SS-FSL), remains critical for applications involving tabular data in domains like medicine, finance, and science.
The existing SS-FSL methods often rely on self-supervised learning (SSL) frameworks developed for vision or language, which assume the availability of a natural form of data augmentations. For tabular data, defining meaningful augmentations is non-trivial and can easily distort semantics, limiting the effectiveness of conventional SSL. In this work, we rethink SSL for tabular data and propose Separated-at-Birth Alignment (SeBA), a joint-embedding framework for SS-FSL that eliminates the dependence on augmentations. 
Our core idea is to separate the data into two independent, but complementary views and align the representations of one view to mirror the nearest-neighbor correspondence of the data in the second view. A type-aware separation scheme ensures robust handling of mixed categorical and numerical attributes, while a lightweight architecture with ensemble aggregation improves generalization and reduces sensitivity to misselection of model parameters. An experimental study conducted in various benchmark datasets demonstrates that SeBA often achieves state-of-the-art performance in tabular SS-FSL, opening a new avenue for SSL paradigm in the domain of tabular data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents SeBA, a novel augmentation-free SSL framework for tabular data. It splits each sample into two complementary views — feature and target — and learns to align the nearest neighbor relationships identified in the target view within the representation space of the feature view, under the assumption that tabular features are redundant and semantically correlated, allowing partial views to preserve the same underlying structure.

### Strengths
- The paper is easy to follow.
- The paper is well motivated, addressing the challenge of meaningful augmentation in tabular learning.
- The paper introduces a novel augmentation-free SSL framework, which based on JEAs, for tabular data. 
- The framework is conceptually simple yet generalizable.

### Weaknesses
- The paper lacks evaluation fairness across datasets. While SeBA adds new datasets, it remains unclear under what criteria these datasets were added and how representative they are of broader tabular domains. Several entries in Tables 1–2 remain blank, leaving unclear whether SeBA’s advantage stems from better representation learning or from dataset selection bias.
- The paper lacks organization and comprehensive baseline coverage. The evaluation does not clearly separate supervised, semi-supervised, and self-supervised baselines, resulting in a fragmented comparison. Several widely used methods are omitted: classical semi-supervised algorithms such as Pseudo-Labeling [1], Mean Teacher [2], and ICT [3], as well as strong tabular SSL frameworks like SAINT [4]. Moreover, recent contrastive and range-limited augmentation approaches (e.g., FESTA [5]) directly address the same challenge of constructing semantically valid augmentations for tabular data. Including and organizing these baselines would provide a fairer and more interpretable evaluation of SeBA’s contribution within the broader SSL landscape.
- The paper shows an unreliable alignment assumption on certain datasets, as the core premise—that samples close in the target-view space are semantically similar—does not hold for datasets such as CMC and GES, where feature redundancy is low. This suggests that the alignment mechanism is not reliable and becomes unstable when the underlying feature correlation is weak or when the domain structure is heterogeneous.
- The paper shows limited performance without ratio ensemble, as training with a single separation ratio 𝑇 leads to highly variable results across datasets. This suggests that the learned representations are mask-dependent rather than semantically invariant. The ratio ensemble compensates for this instability but requires training multiple encoders (≈5× compute), reducing scalability and reproducibility. More efficient alternatives—such as Dynamic Ratio Sampling or a ratio-conditioned encoder—could replace the multi-encoder ensemble by learning to handle varying separation ratios within a single network.

*Reference*

[1] Dong-Hyun Lee, Pseudo-label: The simple and efficient semi-supervised learning method
for deep neural networks. (ICML 13)

[2] Antti Tarvainen and Harri Valpola, Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. (NeurIPS 17)

[3] Vikas Verma et al., Interpolation consistency training for semi-supervised learning. (Neural Networks 22)

[4] Gowthami Somepalli et al., Saint: Improved neural networks for tabular data via row attention and contrastive pre-training. (TRL workshop at NeurIPS 22)

[5] K Lee et al., Range-limited Augmentation for Few-shot Learning in Tabular Data with Comprehensive Benchmark. (KDD 25)

### Questions
[Q1] Dataset selection criteria

What were the criteria for adding the new datasets from OpenML-CC18 (e.g., domain diversity, feature-correlation profiles, class balance)?
Additionally, could the authors clarify why the results (“–” entries in Tables 1–2) for these datasets were left unreported, and whether they could be provided to ensure evaluation fairness and reproducibility?

[Q2] Baseline coverage

Would additional comparisons with well-known semi-supervised methods (Pseudo-Labeling [1], Mean Teacher [2], ICT [3]), strong tabular SSL models such as SAINT [4], and recent contrastive or range-limited augmentation approaches like FESTA [5] provide further insight into SeBA’s strengths? Such comparisons could better demonstrate how SeBA differs from and potentially improves upon existing paradigms that also address the challenge of defining semantically valid augmentations in tabular data.

[Q3] Alignment stability and assumption validity

In datasets like CMC and GES, where feature redundancy is low, the alignment assumption seems unreliable.
Have the authors analyzed whether the instability arises from the neighbor selection process or from dataset-specific feature sparsity?
Would a feature-group-aware masking strategy mitigate this issue?

[Q4] Dependence on ratio ensemble

Could the authors clarify whether the ratio ensemble is a core design choice that contributes to SeBA’s effectiveness, or merely a compensatory mechanism to mitigate instability in single-ratio training?
Have the authors explored a single-model variant, for example by sampling different separation ratios 𝑇  for each batch (Dynamic Ratio Sampling) or by explicitly conditioning the encoder on the current ratio (ratio-conditioned encoder), so that one model learns to handle multiple ratio distributions without ensembling?

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
4

### Summary
This paper proposes Separated-at-Birth Alignment (SeBA), a semi-supervised few-shot learning (SS-FSL) framework specifically designed for tabular data. The SeBA addresses the challenge of learning representations from scarce labeled samples with access to many unlabeled ones.

The key innovation is to remove the augmentation dependency that is typical in self-supervised learning (SSL). Instead of generating positive pairs through data augmentations, SeBA divides the input features into two complementary “views” (a feature view and a target view) and aligns the learned representations of the feature view with the nearest-neighbor graph derived from the target view.
The method introduces a type-aware separation scheme to handle mixed categorical and numerical attributes, ensuring semantic consistency when creating the two views.
SeBA uses a lightweight multi-layer perceptron (MLP) encoder and a conditioned projector, combined with an ensemble strategy across multiple separation ratios to improve generalization and avoid overfitting on small datasets.

### Strengths
1. The paper addresses an important gap in the literature. Few-shot and semi-supervised learning have been well explored in vision and NLP, but the tabular domain remains underdeveloped. SeBA offers a principled approach to this problem.

2. The method eliminates the need for artificial augmentations. This is a substantial conceptual improvement, as data augmentations are ill-defined or even harmful in tabular data, and SeBA provides a clear alternative.

3. The design is elegant and simple. By constructing “separated-at-birth” views and aligning representations via nearest neighbors, the framework remains lightweight yet effective.

### Weaknesses
1. The paper’s scope is somewhat narrow. Although the title and framing emphasize “semi-supervised few-shot learning,” all experiments are limited to tabular data. The contribution might not generalize to other modalities such as vision or multimodal data.

2. The novelty is incremental within the self-supervised learning paradigm. The core idea is constructing paired views without augmentation. This idea is conceptually related to existing joint-embedding methods (e.g., BYOL, SimCLR) with different pairing mechanisms.

3. The method relies on the assumption of meaningful nearest neighbors. The nearest-neighbor graph in the target view may not always reflect true semantic relationships, especially in high-noise or high-dimensional data.

4. The evaluation lacks direct comparison to transformer-based tabular models. Models like TabPFN and UniTabE are mentioned but not deeply analyzed in the few-shot regime, which could provide a more rigorous benchmark.

5. In recent years, Multi-modal Large Language Models (MLLMs) have demonstrated strong capabilities, featuring large parameter sizes and excellent generalization. In contrast, the model proposed in this paper appears to have a relatively small number of parameters. I believe that the proposed method could potentially be integrated into MLLM frameworks, but the authors have not explored this direction.

### Questions
Please address the concerns I raised in the Weaknesses section.

In addition, could the authors include qualitative examples of the datasets and model outputs in the main text (or in the supplementary material)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors address the challenge of semi-supervised few-shot learning (SS-FSL) for tabular data, where models must classify data with very limited labeled examples while leveraging a large pool of unlabeled data. This problem is particularly relevant in domains like medical diagnosis, credit risk prediction, and cognitive sciences where obtaining labeled data is expensive but unlabeled data is readily available. The research aims to develop an effective pretraining method specifically tailored for tabular data that can learn meaningful representations from unlabeled data and then be fine-tuned with just a few labeled examples.

The paper identifies critical limitations of existing self-supervised learning (SSL) methods when applied to tabular data. Traditional SSL approaches rely heavily on data augmentations to create semantically similar positive pairs for contrastive learning. While augmentations like cropping, rotation, or color jittering work naturally for images, defining meaningful augmentations for tabular data is problematic. Poorly chosen transformations such as zero masking, Gaussian noise, or sampling from marginal distributions can distort semantic meaning, generate out-of-distribution samples, or create invalid data points. For instance, decreasing a car's age while increasing its mileage would be semantically inconsistent, or assigning non-integer values to discrete features like number of car seats would be invalid. This fundamental challenge has led recent work to largely abandon SSL for tabular data in favor of alternative approaches like cluster detection or diffusion-based methods.

SeBA introduces a novel approach that eliminates the need for augmentations entirely. The core idea involves separating tabular data "at birth" into two complementary and independent views: feature views and target views. For each minibatch, a random binary mask determines which columns belong to each view. The method then identifies nearest-neighbor relationships in the target view space and trains an encoder to align the representations of feature views according to these nearest-neighbor correspondences.
The authors argue this works well for several reasons. First, it avoids the problematic task of designing augmentations for tabular data. Second, the nearest-neighbor relationships provide semantically meaningful positive pairs based on actual data similarity rather than artificial transformations. Third, the method employs a conditioned projector that takes both the encoder representation and the separation mask as inputs, allowing the model to adapt to different separation schemes. Fourth, type-aware separation ensures categorical variables are handled properly without splitting their one-hot encodings. Finally, an ensemble strategy using multiple separation ratios eliminates the need for careful hyperparameter tuning.

The authors conduct extensive experiments across twelve tabular datasets in 1-shot, 5-shot, and 10-shot classification settings. They compare SeBA against multiple baseline categories including supervised methods like CatBoost and k-NN, self-supervised methods like VIME and SCARF, meta-learning approaches, and state-of-the-art SS-FSL methods STUNT and D2R2. Performance is measured through classification accuracy with multiple random seeds and support/query set selections to ensure statistical reliability.
Additionally, the authors provide detailed ablation studies examining the impact of data normalization, missing data imputation strategies, separation ratios, and classifier choices. They also analyze the alignment between the pretraining objective and downstream tasks by measuring the proportion of nearest neighbors that share the same class label and examining the stability of neighbor assignments across different random separations.

The authors conclude that SeBA successfully demonstrates that self-supervised learning can be effective for tabular data when properly designed. The method achieves state-of-the-art performance on tabular few-shot learning benchmarks, obtaining the best accuracy in 29 out of 36 experimental instances. The main contributions include introducing the Separated-at-Birth Alignment framework that eliminates augmentation requirements, instantiating it as a lightweight model with ensemble strategies to prevent overfitting, providing thorough empirical validation, and demonstrating consistent generalization across diverse tabular datasets. The work opens new avenues for SSL paradigms in tabular data and provides a practical foundation for data-constrained applications.

### Strengths
SeBA addresses the fundamental incompatibility between traditional SSL and tabular data by completely reimagining how positive pairs are constructed. Rather than forcing unnatural augmentations onto tabular data, it leverages the inherent structure of the data itself through nearest-neighbor relationships. This approach is particularly well-suited because tabular data naturally contains meaningful similarity structures that can be discovered through partial feature comparisons.

The key contribution lies in demonstrating that SSL principles can be successfully adapted to tabular data without relying on augmentations. By introducing the separated-at-birth concept, the authors provide a principled alternative to augmentation that maintains the benefits of contrastive learning while respecting the unique characteristics of tabular data. The type-aware separation scheme for handling mixed categorical and numerical features represents an important technical innovation that ensures semantic validity.

The method builds on solid theoretical foundations from contrastive learning while addressing specific tabular data challenges systematically. The use of InfoNCE loss provides a well-understood optimization objective, while the conditioned projector allows the model to handle varying separation schemes coherently. The ensemble approach addresses the practical challenge of hyperparameter selection in few-shot scenarios where validation data is scarce. Each design choice, from zero imputation to type-aware separation, is motivated by specific tabular data characteristics and supported by ablation studies.

### Weaknesses
The paper lacks theoretical justification for why nearest-neighbor relationships in partial feature spaces should consistently produce semantically meaningful positive pairs. While empirical results show high same-class neighbor rates, the conditions under which this assumption holds or might fail are not thoroughly analyzed. The relationship between separation ratio and dataset characteristics remains underexplored, leaving practitioners without clear guidance on when certain ratios might be preferred.

While the experiments cover multiple datasets and shot settings, certain aspects lack depth. The comparison with D2R2 uses only the inductive variant rather than the full transductive version that achieves better performance. The paper does not explore performance on datasets with very high dimensionality or extreme class imbalance, both common in real-world tabular applications. Additionally, computational efficiency comparisons are absent, which is important given the ensemble strategy requires training multiple models.

The ensemble approach, while eliminating hyperparameter tuning, significantly increases computational cost during both training and inference. The method's reliance on nearest-neighbor relationships may be problematic for datasets where local similarity doesn't align well with class structure, such as data with multimodal class distributions. The fixed separation ratios used in the ensemble might not be optimal for all datasets, particularly those with very few or very many features. Finally, the approach assumes that partial views contain sufficient information for meaningful nearest-neighbor matching, which might not hold for datasets with complex feature dependencies.

### Questions
Regarding the architectural choice, what is the theoretical advantage of using a lightweight MLP encoder over a transformer-based model for representation learning in the tabular SS-FSL setting, aside from the general benefit of avoiding overfitting on small datasets?

Can the authors clarify the underlying logical reason why zero imputation for masked features performs best in the SeBA framework compared to previous methods like sampling from empirical distribution, especially considering how this choice impacts the subsequent nearest-neighbor calculation in the target view?

Given the acknowledged weakness of SeBA's performance on the CMC and GES datasets, what specific characteristics of the data in these two datasets, such as dimensionality, feature distribution, or type complexity, might be hypothesized as the cause of the reduced efficacy of the Separated-at-Birth Alignment mechanism?

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
4

### Summary
In the Tabular Semi-supervised Few-shot Learning (SS-FSL) setting, to address the fundamental difficulty that conventional Self-Supervised Learning (SSL) methods face in defining data augmentations, the authors propose a new joint-embedding architecture called Separated-at-Birth Alignment (SeBA). The method splits the data into a complementary Feature View and Target View, and achieves augmentation-free construction of positive pairs by aligning the representation of the Feature View to the k-nearest-neighbor relationships defined in the Target View. Combining a type-aware separation scheme with an ensemble strategy over various separation ratios ($T$), SeBA is reported to outperform existing state-of-the-art methods across diverse benchmark datasets.

### Strengths
**Augmentation-Free SSL Paradigm**: By sidestepping the long-standing challenge of designing semantically meaningful augmentations for tabular data, the authors introduce a new SSL pretext task that combines feature separation with nearest-neighbor matching. This has clear potential to steer research directions in the area.

**Robustness-Oriented Design**: Instead of manually tuning the optimal feature separation ratio ($T$), the approach ensembles encoders trained with multiple $T$ values to secure generalization and reliability. This is a practical strategy under few-shot constraints.

**Effective Ablations**: The analyses show that, for missing-data imputation, zero imputation outperforms marginal sampling by about 3–5 percentage points. They also confirm that linear probing is the most effective few-shot classifier, providing concrete justification for key design choices.

### Weaknesses
(Novelty)


The proposed idea is theoretically very similar to existing augmentation-free, feature-separation-based–based SSL methods (e.g., T-JEPA, Thimonier et al., 2025). In particular, generating subsets via random masking of tabular data and learning structure based on another subset overlaps with the core concept of T-JEPA.


The theoretical and empirical distinctions from the most closely related prior work, T-JEPA, are not clearly articulated .


(Technical Quality)


Lack of reproducibility and stability verification: Standard deviations are missing for all methods in the key results (Tables 1 and 2). Without them, one cannot assess statistical significance—crucial for evaluating reproducibility and performance stability in few-shot settings—representing a serious deficiency in technical rigor.


Incomplete and potentially unfair baseline comparisons: Results on datasets where important baselines were added (MAR, SAT, TEX, etc.) omit strong methods such as TabPFN, SCARF, and UMTRA, making it difficult to judge whether the proposed method generalizes broadly against up-to-date baselines (Tables 1 and 2). No clear technical rationale is provided for these omissions.

Questionable justification for random masking / NN alignment: In tabular data, random masking can discard information from critical features, and defining positive pairs via nearest neighbors assumes that local manifold similarity in high-dimensional/sparse spaces reflects global semantic similarity. This strong assumption (Section 3.1) lacks rigorous theoretical or empirical support.


Insufficient justification for the Conditional Projector: The design of the conditional projector π(h,m)\pi(h, m)π(h,m), which re-conditions the Feature-View encoding on the mask (Equation 5), appears logically unnecessary, and the paper provides limited analysis of its added value.


(Significance)


While the work aims to offer methodological insights toward addressing long-standing challenges in tabular SSL, the absence of statistical stability (missing STDs) and incomplete baseline coverage prevent an objective assessment of whether the paper meaningfully advances the field. (Insufficient evidence; reason: no statistical significance testing possible.)


(Writing & Presentation)


The overall structure and methodological exposition are clear. However, the omission of standard deviation information for the key experimental results (Tables 1 and 2) substantially limits readers’ ability to judge the reliability of the findings. Details necessary for reproducibility—especially those concerning statistical stability—are insufficient.

### Questions
Request for sensitivity analysis of NN alignment: To validate the methodology of using Nearest Neighbors (NN) as surrogates for positive pairs, please present an analysis of the final performance and the structural changes in the embedding space as the value of 
𝑘 varies. In particular, could you quantitatively analyze how frequently semantic mismatch occurs when the definition of NN does not align semantically?

Request for explanation of missing baseline experiments: Please explain the specific reasons why the results for certain datasets (e.g., MAR, SAT, TEX) are missing for major baselines such as TabPFN, SCARF, and UMTRA in Tables 1 and 2. If possible, please provide additional experimental results on the missing datasets using the codebases of those baselines to ensure fairness in comparison.

Strengthening the distinction from T-JEPA: Please provide a detailed analysis of the theoretical and experimental differences between T-JEPA (Thimonier et al., 2025) and SeBA’s nearest-neighbor alignment–based approach. Clearly explain whether the two approaches pursue fundamentally different learning objectives rather than being simple variations of each other.

### Soundness
2

### Presentation
2

### Contribution
2
