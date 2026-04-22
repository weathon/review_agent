# ShapeMatch: Shapelet-Guided Semi-Supervised Learning for Multivariate Time Series Classification

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2, 4

## Abstract
Multivariate Time Series Classification (MTSC) is crucial for many real-world applications and deep learning models such as Transformer have become the state-of-the-art (SOTA) for MTSC due to their ability to capture complex temporal and spatial dependencies. However, they struggle to perform well without sufficient labelled data, limiting their effectiveness in label-scarce scenarios. Furthermore, the absence of effective augmentation methods for time series data that can enhance generalisation whilst preserving essential temporal structures poses a significant challenge. As a result, despite the success of semi-supervised learning in other domains, these limitations have left its integration with deep learning-based MTSC largely unexplored. To bridge this gap, we propose ShapeMatch, a novel flexible semi-supervised framework designed to enhance MTSC in label-constrained environments. ShapeMatch introduces two key innovations: (1) a hybrid training approach that leverages the classic Shapelet Model to guide the deep learning model in the early stages, capitalising on Shapelets' robustness for low-label regimes, and (2) ShapeAug, a tailored augmentation strategy for multivariate time series that preserves critical structural patterns whilst introducing meaningful variations. Extensive experiments on benchmark datasets demonstrate that ShapeMatch surpasses existing SOTA methods for scenarios with limited labelled data, making it a powerful solution for real-world MTSC applications. Our code is available at http://anonymous.4open.science/r/Shape-Match-MTSC/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces ShapeMatch, a shapelet-based framework designed for multivariate semi-supervised time series classification. ShapeMatch incorporates a discriminative shapelet model alongside a shapelet-guided masking augmentation strategy to effectively leverage unlabeled data in semi-supervised learning settings. Experimental evaluations conducted on five healthcare and seven UEA benchmark time series datasets demonstrate that ShapeMatch consistently outperforms four traditional semi-supervised classification baselines and exhibits good model-agnostic adaptability across different backbone architectures.

### Strengths
1. The paper presents an interesting application of the shapelet concept to multivariate semi-supervised time series classification, with a clear and coherent overall structure.
2. The figures and experimental tables are well-organized and easy to interpret, and the paper provides a comprehensive review of prior work on fully supervised multivariate time series classification.
3. The experimental evaluation across different backbone architectures is thorough, offering valuable insights for future researchers in selecting state-of-the-art deep learning models for fully supervised classification tasks.

### Weaknesses
1. Figure 1, intended as the motivation figure, is unconvincing. The comparison between ShapeMatch (using both labeled and unlabeled data) and models (Transformer and a shapelet-only model) trained solely on limited labeled data is unfair. The paper should instead compare ShapeMatch with existing semi-supervised classification methods to demonstrate its true advantage. Moreover, at a 100% labeling rate, ShapeMatch performs similarly to the Transformer, suggesting that shapelets offer limited benefit without unlabeled data.
2. The paper’s novelty is incremental. ShapeMatch adopts the ShapeletDistance strategy from Le et al. (2022) for shapelet search. While the proposed shapelet-guided masking (Eq. 3) is commendable, the augmentation methods (jittering, masking, cropping, shifting) are well-established and do not warrant detailed discussion.
3. The shapelet discovery process is time-consuming. As shown in Table 8, searching with all training samples requires 10.5 hours, compared to only 0.72 hours for supervised training (Table 6), limiting real-world applicability. Although the authors acknowledge this in the appendix, prior works have already proposed learnable shapelet methods [1,2] to significantly reduce search time.
4. The paper lacks a review of shapelet-based methods, particularly those employing learnable shapelet approaches [1,2], which substantially reduce the computational burden compared to the search-based approach described in Equation 1.
5. The experimental baselines include only four semi-supervised methods, none of which are time-series–specific. Moreover, related works [3,4] have explored shapelet-based semi-supervised time series classification but are not discussed.
6. The anonymous code repository is inaccessible, raising concerns about reproducibility.

[1] Grabocka, Josif, et al. "Learning time-series shapelets." Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.

[2] Qu, Eric, et al. "CNN kernels can be the best shapelets." The Twelfth International Conference on Learning Representations. 2024.

[3] Du, Mingsen, et al. "Multivariate Time Series Classification via Heterogeneous Graph Representation." IEEE Transactions on Industrial Informatics (2025).

[4] Wang, Zhicheng, et al. "Multiview Contrastive Shapelet Learning: A Novel Semisupervised Approach for Explainable Machine Fault Diagnosis With Insufficient Annotated Data." IEEE Transactions on Instrumentation and Measurement (2025).

### Questions
1. The authors selected seven UEA time series datasets for semi-supervised evaluation. Since UEA datasets provide only default training and test splits, how was the testing procedure for ShapeMatch conducted?
2. In Table 3, were the reported results obtained from a single dataset? If so, which dataset was used?
3. As noted in [1,2], selecting an appropriate subsequence length is crucial for shapelet discovery. Based on Table 9, how were the shapelet lengths determined for the five healthcare and seven UEA datasets?
4. Since shapelets are designed to enhance interpretability through discriminative subsequences, does the paper provide visualizations showing any notable patterns in the learned shapelets, or any quantitative metrics assessing their quality?

### Soundness
3

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
5

### Summary
This paper proposes a shapelet-guided semi-supervised framework, termed ShapeMatch, for multivariate time series classification. The framework aims to guide deep learning models in aligning their predictions with those of a shapelet-based model during the initial training phase. Specifically, ShapeMatch introduces a shape mask augmentation strategy designed for multivariate time series, enabling the model to extract more informative representations from unlabeled data in a semi-supervised setting. Experimental evaluations conducted on 12 multivariate time series datasets demonstrate that ShapeMatch consistently outperforms selected baseline methods.

### Strengths
1.	The paper is clearly structured, and the figures and tables are well-organized, which facilitates readers’ understanding of the content.
2.	The study’s introduction of time series shapelets into the problem of semi-supervised multivariate time series classification represents a valuable approach, providing a useful reference for future research on shapelet-based time series classification.

### Weaknesses
1.	The paper offers limited novelty in time series shapelets modeling and overlooks much prior work. Specifically, the authors adopt the ShapeletDistance method (Equation 1) for shapelet search, which has been established in prior studies. Furthermore, the runtime analysis in the appendix indicates that this procedure is computationally expensive. In contrast, studies [1,2,3] have demonstrated that learning shapelets via neural networks can significantly reduce the time required to discover shapelets compared to distance-based search methods (Equation 1).

2.	The study lacks innovation in modeling relationships between variables, which is critical for multivariate time series semi-supervised classification. While the authors note in the related work that existing semi-supervised time series methods do not consider inter-variable relationships, their model only mentions in lines 194–197 that the input is multivariate, and the experimental analysis lacks discussion on inter-variable relationship modeling. In comparison, studies [4] and [5] employ clustering and graph networks, respectively, to capture relationships among variables.

3.	The semi-supervised classification baselines selected in this work are primarily designed for image data and do not account for methods specifically developed for time series. For example, studies [6,7] apply shapelets to semi-supervised classification of multivariate time series, while studies [8,9,10] focus on semi-supervised classification of univariate time series, but their semi-supervised learning paradigms can also be effectively applied to multivariate time series.

[1] Learning time-series shapelets. KDD, 2014.

[2] Shapenet: A shapelet-neural network approach for multivariate time series classification. AAAI, 2021.

[3] Multiview unsupervised shapelet learning for multivariate time series clustering. TPAMI, 2022.

[4] From similarity to superiority: Channel clustering for time series forecasting. NeurIPS, 2024.

[5] Fully-Connected Spatial-Temporal Graph for Multivariate Time Series Data. AAAI, 2024.

[6] Heterogeneous Relationships of Subjects and Shapelets for Semi-supervised Multivariate Series Classification. arXiv, 2024.

[7] Multiview Contrastive Shapelet Learning: A Novel Semisupervised Approach for Explainable Machine Fault Diagnosis With Insufficient Annotated Data. IEEE Transactions on Instrumentation and Measurement, 2025.

[8] Self-supervised learning for semi-supervised time series classification. PAKDD, 2020.

[9] Semi-supervised time series classification by temporal relation prediction. ICASSP, 2021.

[10] Self-supervised contrastive representation learning for semi-supervised time-series classification. TPAMI, 2023.

### Questions
1.	In the context of using shapelets for semi-supervised multivariate time series classification, what are the core differences between the proposed method and studies [6,7]? Additionally, regarding the runtime for shapelet discovery, what are the relative advantages and disadvantages of the proposed approach compared to [6,7]?

2.	Compared to studies [8,9,10], what specific advantages does the proposed method demonstrate in semi-supervised classification performance across the selected 11 multivariate time series datasets?

3.	Studies [9,10] also employ data augmentation techniques discussed in their proposed method (e.g., jittering) for semi-supervised time series classification. Without the shapelet-guided mask, how does the proposed ShapeAug augmentation differ from these existing methods?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the problem of Multivariate Time Series Classification under label-scarce conditions. The authors propose ShapeMatch, a semi-supervised framework designed to enhance model generalization with limited labeled data. The framework combines two main ideas: (1) a hybrid training paradigm that integrates classical Shapelet models to guide deep learning backbones during early training stages, and (2) a tailored ShapeAug strategy that introduces meaningful temporal variations while preserving key structural patterns. Experiments on multiple benchmark datasets show that ShapeMatch outperforms existing state-of-the-art semi-supervised and fully supervised baselines, especially in low-label regimes.

### Strengths
1. The paper addresses a practically relevant and underexplored problem and offers clear potential for real-world applications.

2. The framework and model pipeline are clearly illustrated, improving readability and conceptual understanding.

### Weaknesses
1. Some claims are overstated; most shapelet-based methods were originally developed for univariate time series and may not generalize directly to multivariate contexts.

2. The experimental presentation lacks clarity as key results are fragmented, and the overall comparison could be made more intuitive and systematic.

### Questions
1. Could the authors provide a concise summary (e.g., a table) showing ShapeMatch’s performance across different deep learning backbones, to clearly support its claimed generality?

2. Given that only r = 50 samples per class are used, how stable are the reported results? Including standard deviations or comparisons with larger r (e.g., 100) would clarify robustness.

3. How are multivariate shapelets learned and integrated with univariate ones? A more detailed explanation would help clarify the modeling of inter-variable dependencies.

4. Is there theoretical or empirical evidence that shapelet priors reliably improve robustness under label scarcity, especially in high-dimensional MTSC settings?

5. Since the framework shares conceptual similarities with FixMatch, could the authors elaborate on the key methodological differences and provide a direct comparison to emphasize ShapeMatch’s unique contributions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
1.Proposes ShapeMatch: a novel semi-supervised framework that integrates classic Shapelet models with modern deep learning backbones.
2.Introduces Shapelet-guided training: leverages the robustness of Shapelet models in early training to guide deep models, providing strong inductive bias under low-label conditions.
3.Designs ShapeAug: a tailored augmentation strategy that identifies and preserves class-discriminative subsequences (shapelets) while applying controlled noise, masking, scaling, and shifting elsewhere.

### Strengths
1. The paper integrates two major MTSC techniques—shapelet and deep learning (DL) methods—by fusing them together, where the shapelet method essentially serves as a preprocessing or augmentation step for time series, followed by training with various DL backbones.  
2. The experimental results appear strong, especially in demonstrating that ShapeMatch can be effectively applied to other DL methods as well.

### Weaknesses
1. Shapelet discovery is used as a preprocessing step because directly applying the shapelet method is computationally expensive despite its high accuracy, while deep learning methods offer a faster but less accurate alternative. However, this design does not truly combine the strengths of both approaches.
2. The approaches for shapelet discovery, shapelet distance feature definition, and training are not novel, resulting in a lack of innovation.  
3. Weak and strong augmentation strategies have already been employed in contrastive learning for time series. The paper should cite relevant prior work, clearly highlight the differences, and include comparisons with similar methods in the experiments.  
4. The decision not to augment the key subsequences is debatable; it may negatively impact the model’s generalization ability on most datasets.

### Questions
1. Using shapelet discovery as preprocessing—why not directly use the shapelet method? The main drawback of shapelet methods is their high computational cost despite high accuracy, whereas deep learning (DL) methods exhibit the opposite trade-off.  
2. If it is useful, it should not be limited to semi-supervised scenarios—have you tried it in other settings?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
ShapeMatch tackles semi-supervised multivariate time-series classification (MTSC) when labels are scarce. It first extracts class-specific shapelets from the labelled subset with a perceptually-aware algorithm (PPSD), trains a lightweight Shapelet Model and any deep backbone on labelled data, then continues semi-supervised training with “ShapeAug” augmentations and pseudo-labels that fuse frozen shapelet and deep-model predictions. Across 12 healthcare/UEA benchmarks and five backbones the framework improves 1 %–20 % label-regime accuracy over FixMatch, Semiformer, etc., while adding <1 h CPU shapelet discovery and 0.5 GB GPU memory.

### Strengths
1) Novel hybrid guidance mechanism: Combines shapelet distance features with deep logits for pseudo-labeling via epoch-scheduled and class-distance biases, yielding consistent +2–6 % accuracy over FixMatch on 12 datasets. Demonstrates complementarity: shapelets excel early when labels are extremely scarce, deep model dominates later, justifying the proposed curriculum.

2) ShapeAug augmentation strategy: Introduces shapelet-masked jitter/mask/scale/shift that preserves discriminative subsequences; ablation shows joint techniques outperform individual ones by up to 2 %.

3) Broad backbone compatibility: Evaluated with five architectures: TSLANet, iTransformer, ShapeFormer, MedFormer, PatchTST and a small CNN; ShapeMatch beats respective SSL baselines in every case, indicating framework generality.

### Weaknesses
1) Incomplete baseline coverage: Omits recent time-series contrastive or self-supervised methods that also work in label-scarce regimes; comparisons are limited to FixMatch, Pseudo-Label, Semiformer.

2) Scalability & computational bottleneck: Shapelet discovery remains CPU-bound (36 min on 8 cores for moderate data, Table 7); no GPU acceleration or complexity analysis with respect to V, T, or pool size g.

3) Limited analysis of failure modes: No dataset is identified where ShapeMatch underperforms supervised or SSL baselines; absence may indicate selective reporting.

4) Expand baseline coverage: Include at least two recent time-series SSL baselines under identical 1 %/5 %/20 % splits and report mean ± std to situate ShapeMatch novelty.

### Questions
See above weakness.

### Soundness
2

### Presentation
2

### Contribution
2
