# UniOD: A Universal Model for Outlier Detection across Diverse Domains

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Outlier detection (OD), distinguishing inliers and outliers in completely unlabeled datasets, plays a vital role in science and engineering. Although there have been many insightful OD methods, most of them require troublesome hyperparameter tuning (a challenge in unsupervised learning) and costly model training for every task or dataset. In this work, we propose UniOD, a universal OD framework that leverages labeled datasets to train a single model capable of detecting outliers of datasets with different feature dimensions and heterogeneous feature spaces from diverse domains. Specifically, UniOD extracts uniform and comparable features across different datasets by constructing and factorizing multi-scale point-wise similarity matrices. It then employs graph neural networks to capture comprehensive within-dataset and between-dataset information simultaneously, and formulates outlier detection tasks as node classification tasks. 
As a result, once the training is complete, UniOD can identify outliers in datasets from diverse domains without any further model/hyperparameter selection and parameter optimization, which greatly improves convenience and accuracy in real applications. More importantly, we provide theoretical guarantees for the effectiveness of UniOD, consistent with our numerical results. We evaluate UniOD on 30 benchmark OD datasets against 17 baselines, demonstrating its effectiveness and superiority.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes UniOD, a universal outlier detection framework that trains a single model on a collection of labeled historical datasets to detect outliers in new, unseen tabular datasets without retraining or hyperparameter tuning. To achieve cross-domain generalization, UniOD constructs multi-scale similarity matrices from each dataset, factorizes them via SVD to obtain uniform-dimensional node features, and then employs a hybrid architecture of GINs and graph transformers to perform node-level binary classification (inlier vs. outlier). The method is evaluated on 15 datasets from ADBench.

### Strengths
1.	The proposed method introduces a novel paradigm shift from dataset-specific to universal outlier detection, which is underexplored in the literature. 
2.	The methodology is well-motivated and technically sound. The integration of SVD-based feature unification, graph construction, and GNNs is carefully designed to handle varying feature dimensions and semantics. The theoretical analysis provides a nontrivial generalization bound that aligns with empirical findings.
3.	The paper is generally well-written, with clear figures and a logical flow from problem statement to evaluation.

### Weaknesses
1.	The authors state that experiments are conducted on 30 datasets from ADBench, split into two groups of 15 for cross-validation. However, ADBench actually contains 57 datasets, not just 30. The paper does not justify why only a subset was selected, nor does it clarify the criteria for partitioning. This raises concerns about potential cherry-picking. The authors should either (a) use the full ADBench tabular benchmark, or (b) explicitly state the selection rationale and provide results on a broader set to demonstrate robustness.
2.	The paper partitions the 30 ADBench datasets into two fixed groups of 15 for training and testing, and performs a single cross-validation swap. While this demonstrates basic robustness, it does not adequately address how historical datasets should be chosen in practice or how performance varies under different selection strategies.
3.	The claim of universality is compelling but narrowly validated. All experiments are on tabular data; it remains unclear whether UniOD can generalize to other modalities (e.g., images, time series) without significant architectural changes. Clarifying the scope of “universal” (i.e., universal across tabular domains only) would temper overstatement.

### Questions
Could the authors clarify the criteria used to select these 30 datasets?
Can the authors provide qualitative or quantitative analysis of when and why UniOD fails?

### Soundness
2

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
This paper proposes UniOD, a universal outlier detection model designed to address the core limitations of conventional methods, including the significant efforts required for hyperparameter tuning, the need for repetitive model training on new datasets, and the inability to leverage knowledge from historical datasets. The key contributions of this work are fourfold:

1. **Problem Formulation**: It is the first to explicitly define and formalize the problem of universal outlier detection, which aims to "train a single model that can directly generalize to any unseen tabular datasets from diverse domains without any retraining or hyperparameter tuning".

2. **Technical Framework**: It introduces a novel technical framework based on a "graph reformulation" process, which unifies heterogeneous datasets by constructing multi-scale similarity matrices, decomposing them into uniformly dimensioned features via SVD, and subsequently employing graph neural networks for node classification.

3. **Theoretical Analysis**: It provides a theoretical generalization bound, offering mathematical guarantees for the effectiveness of the proposed method.

4. **Experimental Validation**: Extensive experiments demonstrate that UniOD outperforms 17 baseline methods on 30 datasets while achieving faster inference speed.

### Strengths
This paper presents several noteworthy strengths:

1. **Paradigm-Shifting Contribution**
   - First explicit proposal and systematic formalization of "universal outlier detection"
   - Core innovation: single model generalizing across diverse domains without retraining or hyperparameter tuning
   - Represents fundamental reformulation of conventional outlier detection paradigm

2. **Technical Innovation**
   - Elegant graph reformulation process transforms heterogeneous tabular data
   - Creates unified, structurally learnable representations from disparate datasets
   - Overcomes limitations of existing transfer learning approaches that require:
     - Extensive hyperparameter evaluation
     - Strong domain similarity assumptions

3. **Theoretical Rigor**
   - Provides generalization error analysis beyond empirical demonstrations
   - Offers mathematical guarantees for method effectiveness
   - Enhances academic credibility and theoretical foundation

### Weaknesses
While the proposed UniOD framework demonstrates compelling performance, several limitations warrant discussion for future improvement:

1. **Scalability Challenges in Preprocessing**  
The O(n²) computational and memory requirements for similarity matrix construction present practical constraints, as evidenced by the needed subsampling for datasets beyond 6,000 samples. Future work could explore approximate nearest neighbor techniques or sparse graph construction to enhance applicability to larger-scale datasets.

2. **Dependence on Euclidean-Based Similarity**  
The reliance on Gaussian kernel similarity (and consequently Euclidean distance) may limit performance in scenarios where this metric is suboptimal. Incorporating learnable or adaptive distance metrics could strengthen robustness across diverse data distributions.

3. **Sensitivity to Historical Data Availability**  
While leveraging multiple labeled historical datasets is a strength, the framework's effectiveness in domains with extreme scarcity of labeled anomalies remains unverified. Investigating few-shot or semi-supervised adaptations would valuablely expand its applicability.

### Questions
1. **Scalability and Computational Efficiency**  
The paper indicates that datasets exceeding 6,000 samples required subsampling due to computational constraints. Have the authors explored more scalable graph construction alternatives, such as k-nearest neighbor sparse graphs or Nyström approximation methods, to avoid full similarity matrix computation? Could experimental results demonstrate whether these approaches maintain performance while handling datasets at larger scales (e.g., tens of thousands of samples), thereby improving practical applicability in big data scenarios?
2. **Analysis of Performance Boundaries**  
While UniOD shows strong average performance, simpler methods outperform it on specific datasets (e.g., Cardiotocography, Pima). Could the authors provide further analysis identifying dataset characteristics where UniOD may underperform? For instance, are there correlations with meta-features like anomaly ratio, dimensionality, or cluster structure clarity? Defining such boundaries would offer valuable guidance for practical application.

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
2

### Summary
This paper proposed UniOD, a universal pretrained model for outlier detection. UniOD learns from various historical tabular datasets and can directly score the outliers for new datasets in a zero-shot manner, without re-training the model or hyperparameter tuning. The datasets are converted into point-wise kernel matrices at different scales and a GNN is trained on graph-structured datasets.

### Strengths
1. Plug-and-play: this paper proposes the UniOD framework by pretraining on various datasets and conduct inference in a zero-shot manner, saving the deployment cost for OD tasks.
2. Unified dataset representation: this paper utilizes the multi-scale similarity and SVD to produce unified node features, enforcing the generalizability of the model.
3. The authors provides both comprehensive theoretical justification and extensive empirical analysis of the method. UniOD is well theoretical grounded.

### Weaknesses
1. Dependence on historical datasets. UniOD requires labeled historical datasets, which can be unavailable in real-world applications. The limited historical datasets may impair the performance of UniOD on new datasets, especially when the historical datasets is limited to few domains.
2. Generality concern: the effect of dataset variability is not fully discussed in the paper, as this will potentially influence the model's generality if the model hasn't encountered datasets from similar distributions.

### Questions
See the weaknesses above.

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
4

### Summary
This paper introduces UniOD, a novel framework for outlier detection that is generalizable to diverse datasets. The authors address the inefficiency of conventional OD methods, which typically require per-dataset retraining and hyperparameter tuning. UniOD tackles this by leveraging a collection of historical labeled datasets to train a single, universal model. The framework's key innovation is its data unification pipeline, which transforms each dataset into a set of multi-scale similarity matrices to create graph-structured representations. It then employs Singular Value Decomposition (SVD) to generate uniformly dimensioned features. Outlier detection is thus reformulated as a node classification task on these graphs, tackled by a GNN-based model. Once trained, UniOD can be directly applied to new, unseen datasets without any further training, demonstrating superior performance against 17 baseline methods on a benchmark of 30 datasets.

### Strengths
•	A Novel Paradigm for OD: The primary strength is the innovative concept of a universal framework that eliminates the need for per-dataset retraining. This directly addresses a major bottleneck in the practical application of outlier detection.
•	Elegant Unification of Heterogeneous Data: The use of multi-scale similarity matrices combined with SVD is a powerful and clever technique for creating a unified feature space from datasets with diverse dimensionalities and semantics.
•	Strong Empirical and Theoretical Backing: The claims are convincingly supported by comprehensive experiments on a large benchmark, which is further bolstered by a theoretical analysis of the model's generalization ability.
•	High Practicality and Efficiency: By decoupling training from testing, the framework is highly practical and computationally efficient at inference time, making it well-suited for real-world scenarios requiring rapid analysis of new data.

### Weaknesses
•	Heavy Reliance on Historical Data Composition: The model's success is fundamentally tied to the quality, scale, and diversity of the historical datasets. The paper lacks an investigation into the sensitivity of the model to the composition of this training pool.
•	Potential Scalability Bottlenecks: The methodology relies on constructing an n*n similarity matrix, which has a quadratic complexity (O(n²)) with respect to the number of samples. This could be computationally prohibitive for very large datasets.
•	Limited Exploration of Graph Construction: The framework exclusively uses a Gaussian kernel to build the similarity matrices. An investigation into different kernel functions or alternative graph construction techniques would have strengthened the paper's claims of robustness.

### Questions
1.	It will strengthen the paper if providing an analysis of the model's sensitivity to the composition of the historical training data. For example, how does performance on a specific target domain (e.g., finance) change when all finance-related datasets are deliberately excluded from the training pool? This would clarify the practical requirements for curating the training set.
2.	The O(n²) complexity for similarity matrix construction is a potential bottleneck. It will be helpful to explore more scalable graph construction techniques, such as those based on approximate nearest neighbors, to enhance the framework's applicability to datasets with millions of samples.
3.	It will be helpful to provide some qualitative analysis on the structural patterns the GNN model learns to distinguish outliers. For instance, in the graph representation, are outliers typically identified as isolated nodes, or do they belong to small, dense clusters disconnected from the main graph component? This would provide valuable insight into the model's decision-making process.

### Soundness
3

### Presentation
4

### Contribution
3
