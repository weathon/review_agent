# FSTS: A Feature Space Transfer Selection Method for Data

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
The performance of deep learning image models is often largely determined by data quality. However, high-quality data is often scarce and difficult to collect. The data exchange platform provides a promising solution for obtaining image training samples, but the actual data exchange must consider the costs associated with data collection, storage, and training. This article proposes a Feature Space Transfer Selection (FSTS) method for identifying core data subsets that are crucial for model training. The proposed method extracts feature vectors from the source and target datasets, calculates class centroids from the reference (source) dataset, and ranks the target samples based on their similarity to these class centroids. Then select the core dataset from the target data based on the similarity ranking. The experimental results show that FSTS outperforms prior state-of-the-art approaches and effectively helps users select the core set of training data, which helps improve the efficiency of model training and overall performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper focuses on the data quality of the training set. The goal is to select the most important samples for training. To achieve this, this submission proposes a simple method called FSTS, which selects the core dataset from the target data based on the similarity ranking. Specifically, FSTS extracts feature vectors from the source and target datasets, calculates class centroids from the reference (source) dataset, and ranks the target samples based on their similarity to these class centroids. The experiments on CIFAR and KITTI show that FSTS is able to achieve improved accuracy compared with other baselines.

### Strengths
+ This paper is clear and easy to read. The method is well-illustrated, and the experimental settings and details are well included

+ The evaluation includes the robustness to Noise: The method shows strong performance under label noise and image corruption, indicating its reliability in noisy, real-world datasets.

### Weaknesses
- [**Limited Contribution**] The major concern is that the proposed method FSTS has limited contribution. ***First***, FSTS uses the standard coreset selection method, which is based on distance in feature space or diversity measures [a, b]. The idea of using class centroids for distance calculation is a standard approach and is widely used in the same problem; I did not see any significant difference from these techniques. ***Second***, recent methods of coreset selection are not included for comparison. For example, the methods of [a, b, c]. A comparison and discussion should be provided. Again, the proposed method is a simplified version of them; the advantage is limited.


    [a] Moderate coreset: A universal method of data selection for real-world data-efficient deep learning

    [b] ELFS: Label-Free Coreset Selection with Proxy Training Dynamics

    [c] Zero‑Shot Coreset Selection: Efficient Pruning for Unlabeled Data 

- [**Evalution is somewhat weak**] ***First***, experiments are only conducted on KITTI Road and CIFAR-10. Both tasks are small-scale and limited in data diversity. Other diverse datasets should be considered for validation, such as ImageNet and DomainNet. ***Second***, the class centroid is based on the feature extractor, but the paper does not study how the choice of model (e.g., VGG-16 vs. ResNet) affects the results. ***Third***, I noticed that all experiments are in-domain, where the source and target come from the same distribution (CIFAR-10 split, KITTI split). How about selecting data from different domains (e.g., DomainNet)?

- [**Intra-Class Variability**] FSTS ranks samples by their similarity to class centroids, but this inherently favors samples near the center of the distribution. I noticed the experiments compare with the other three selections (Cloest, Farthest and Two-ends). But it is still unclear why excluding hard or diverse examples helps. Please clasirfy this point.

### Questions
- [**Potential Failure cases**] While the evaluation includes noisy data, it is still unclear whether the method works under negative transfer when the domain shift is large. Also, what if the selected source dataset is biased? How can this be mitigated? What if the feature representations for class centroids are poor?

- Please clarify how the method performs when there is a significant domain gap between source and target datasets? For example, the cross-domain benchmarks like DomainNet?
- The paper adopts a fixed selection policy (e.g., top 40%–60% cosine similarity). Why was this particular selection window chosen? Have other ranges or adaptive thresholds been tested?

### Soundness
2

### Presentation
2

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
This paper proposes FSTS (Feature Space Transfer Selection), a method for selecting core subsets from target datasets based on their similarity to source dataset class centroids in feature space. The method is designed for data exchange platforms where users need to efficiently select high-quality training data while considering acquisition, storage, and computational costs. FSTS extracts features from both source and target datasets, computes class centroids from the source, ranks target samples by cosine similarity to these centroids, and selects samples from the middle of the ranking distribution. Experiments on KITTI Road and CIFAR-10 datasets show that FSTS outperforms several baseline methods, particularly at low selection ratios, and demonstrates robustness to label and image noise.

### Strengths
1. The paper clearly frames the problem within the practical context of data exchange platforms, addressing real-world concerns about data acquisition costs, storage limitations, and computational efficiency. 
2. The paper provides thorough experiments across multiple datasets (KITTI Road, CIFAR-10), multiple selection ratios, and under various noise conditions (label noise, image corruption). The comparison with multiple baselines is systematic.
3. The inclusion of robustness tests under label noise and image corruption adds value, showing that FSTS performs competitively, especially under image corruption scenarios.

### Weaknesses
1. The core idea of selecting data based on similarity to source class centroids is straightforward and builds heavily on established concepts in metric learning and data selection. The methodological innovation is incremental compared to existing centroid-based approaches like Herding or moderate coreset selection.
2. The paper lacks a theoretical analysis of why selecting samples from the middle of the similarity distribution should be optimal. 
3. The experiments are limited to only two datasets, one of which (KITTI Road) is very small (289 images).
4. The choice of the middle 40%-60% range is adopted without sufficient ablation or justification specific to FSTS. Why this range is optimal for this method is not explored.

### Questions
1. Can the authors provide a theoretical intuition or justification for why selecting samples from the middle of the similarity ranking (rather than the closest or farthest) leads to better generalization? Is there a connection to hard example mining or diversity-accuracy trade-offs?
2. How does FSTS perform when the source and target datasets come from different domains (e.g., different visual domains or significant distribution shift)?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Feature Space Transfer Selection (FSTS), a method for selecting high-quality core datasets in deep learning. FSTS leverages transfer learning principles by extracting features from a source dataset, computing class centroids, and ranking target samples based on their cosine similarity to these centroids. This approach efficiently identifies representative and semantically relevant samples without requiring model training or gradient computation. Evaluated on KITTI Road and CIFAR-10 datasets, FSTS outperforms existing methods.

### Strengths
- This work proposes the FSTS method, which addresses the problem of efficiently selecting high-quality core data in data exchange scenarios, demonstrating practical applicability.  
- The proposed method reverses the conventional use of transfer learning for data selection by leveraging class centers from the source domain to guide sampling in the target domain, thereby avoiding the inclusion of irrelevant samples that can occur when relying solely on inter-sample distances.  
- The method outperforms baseline approaches on both KITTI Road and CIFAR-10, and exhibits strong robustness under label noise and image corruption conditions.  
- The approach requires no model training or gradient backpropagation, resulting in low computational complexity and high efficiency.

### Weaknesses
1. The method proposed in this paper selects samples based on a transfer learning paradigm—implying that the source data and target data should originate from different data distributions. However, in the experimental evaluation, both the source and target data are drawn from the same dataset. This setup contradicts the fundamental logic and definition of transfer learning. Consequently, the experimental design is seriously flawed, making it difficult to substantiate the effectiveness of the proposed method.
2. The authors claim that their proposed method is flexible and efficient. However, the evaluated datasets are relatively small, which makes it difficult to substantiate this claim. Moreover, the experiments still employ older backbone architectures, failing to demonstrate the method's effectiveness when applied to training modern, mainstream vision models.
3. Equation (4) is difficult to understand. For instance, why is the number of selected samples defined as $ k = ||z_j|| \times R $, and how is it ensured that k is an integer?

### Questions
See weeknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes FSTS (Feature Space Transfer Selection), a data subset selection method intended for data-exchange scenarios. Given a small, high-quality source set and a larger candidate target set, the method extracts features for both with a shared backbone, builds class centroids on the source, ranks target samples by cosine similarity to the corresponding source-class centroid, and selects a band of samples per class to form a coreset. Experiments on KITTI Road (segmentation) and CIFAR-10 (classification), as well as corruption/label-noise variants, show FSTS is competitive or slightly better than Random, Craig, Glister, GraphCut, CAL, and Forgetting at low selection ratios, while being cheaper than the gradient-based methods.

### Strengths
- The paper addresses a real challenge: selecting training data under resource constraints (cost, storage, compute), especially in the context of data exchange platforms.
- FSTS is stated to be computationally lightweight, no need to train models, compute gradients, or use complex optimization. This makes it appealing in low-resource settings.
- FSTS shows improved performance especially when using small core subsets (10–30%), often outperforming more complex baselines in those cases.

### Weaknesses
- The robustness experiments are valuable but the claim of "superior performance" in image corruption needs to be backed by
specific, quantitative data

- Selecting only high-similarity samples might lead to redundant or overly safe coresets, especially for high-variance or long-tailed distributions.

- The ranking purely by similarity to centroids may bias selection toward "average" examples, reducing data diversity.

- Only two standard image datasets are used (CIFAR-10, KITTI Road), both small-scale and with limited domains and diversity.

- The entire method depends on having a labeled, relevant source dataset. This is rarely the case in real-world data marketplaces or in transfer scenarios with domain shift.

- Assumes strong source-target alignment which decreases practicability

- No strategy is proposed for out-of-distribution detection or discovering novel classes. No implication on how to account for unknown classes, rare examples, or domain drift.

- Feature extractor architecture, pretraining source, and impact of hyperparameters (e.g., cosine vs. euclidean, mean vs. median centroids) are not sufficiently discussed and ablations are missing.

- The authors claim that the method is computationally lightweight, this should be backed by comparative quantitative experiments and measurements for different coreset scales and methods, instead of only complexity analysis ("medium speed" is not quantifiable) 

- The core idea (selecting target samples based on similarity to source class centroids in feature space), from the current perspective of the reviewer with the state of the art analysis and argumentation of the authors, combines basic established ideas from prototype-based methods and transfer learning as well as an established data pruning method. Please point that out in the paper such that a reader can directly understand what differentiates the given method substantially from the state of the art.

### Questions
1. How does the presented approach methodologically differ from existing approaches ? What are the core contributions related to prior work? 

2. What parts of the approach mostly contribute to the improvements on data selection ? (Ablations)

3. How does the method perform for substantial domain shifts in the target and core dataset?

4. How does the method deal with outliers / out of distribution samples?

5. What effects does feature / label diversity have on the method? (Generalizability)

And potentially: 6. How does the method deal with the exploration of new classes?

### Soundness
3

### Presentation
2

### Contribution
2
