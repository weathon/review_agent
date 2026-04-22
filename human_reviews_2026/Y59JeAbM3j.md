# Adaptive Gaussian Expansion for On-the-fly Category Discovery

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
On-the-Fly Category Discovery (OCD) aims to address the limitations of transductive learning and closed-set prediction in category discovery tasks by enabling real-time classification of potential future categories using prior knowledge. Existing OCD approaches typically rely on hash-based encodings that map features into low-dimensional hash spaces and directly classify test samples using these encodings. Despite efforts to mitigate the sensitivity of hash functions during testing, these methods still suffer from severe overestimation of the number of categories. In this work, we thoroughly analyze the practical limitations of current OCD methods and formally identify a performance lower bound for the task. Based on this insight, we reformulate OCD into two sub-tasks: Open-Set Recognition and an Fully Novel OCD setting. For all samples, we employ a soft class thresholding strategy to directly detect known classes, which significantly enhances the deployment feasibility of OCD to downstream tasks. For outlier samples, we propose Adaptive Gaussian Expansion (AGE), a dynamic category discovery method that models the Probability Density Functions (PDF) of different classes to uncover potential novel categories in real time. Extensive experiments across multiple datasets demonstrate that our method achieves state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper studies the novel task of On-the-Fly Category Discovery (OCD), in which the model needs to perform real-time classification of potential future categories using prior knowledge. Previous methods suffer from severe overestimation of the number of categories. This paper first formally identifies a performance lower bound for the task, then decomposes OCD into two sub-tasks, i.e., Open-Set Recognition and a Fully Novel OCD setting. Specifically, this work proposes soft class thresholding and Adaptive Gaussian Expansion (AGE) for two tasks, respectively. Extensive experiments show the proposed AGE achieves remarkable performance.

### Strengths
1. This paper is well-motivated and easy to follow.
2. Some theoretical analyses are provided to validate the proposed insights.
3. The proposed method outperforms previous SOTA by a large margin on various datasets.
4. Detailed ablations are conducted to validate the validity of each component.

### Weaknesses
1. How to guarantee the estimated threshold using the validation set is precise and applicable to detect outliers/new classes in test samples? The authors should provide some explanations and experimental results of the first task, i.e., open-set recognition.
2. The writing and logic in Section 3.4 should be improved. For example, the lemma and proposition should be rearranged and interspersed with the viewpoint statement, rather than putting them all together at the end of Section 3.4.
3. According to my understanding, the novelty of AGE mainly lies in the inference time. I was wondering whether the method consumes more computations, so the comparison of inference time between the proposed method and previous works should be provided.

### Questions
See weakness. How to guarantee the estimated threshold could generalize to test-time samples?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on the On-the-Fly Category Discovery (OCD) task. The authors point out that existing OCD methods rely on hash-based encodings (e.g., PHE, SMILE), which lead to a serious overestimation of the number of categories and poor adaptability to downstream tasks. To address these issues, the authors propose: Establishing a theoretical lower bound, revealing that a closed-set classifier alone can achieve a certain level of performance; Reformulating the OCD task into two subtasks — Known Category Recognition (Open-Set Recognition, OSR) and Real-time Novel Category Discovery; Introducing the Adaptive Gaussian Expansion (AGE) framework, which enables adaptive expansion of class distributions through dynamic Gaussian modeling; Achieving significant performance improvements on multiple benchmark datasets.

### Strengths
1.	The paper is clearly written, well-structured, and effectively supported with figures and tables.
2.	The ablation study is relatively comprehensive, verifying the effects of factors such as soft covariance, threshold β, and PCA dimensionality on the model’s performance.

### Weaknesses
1.	The core characteristics of OCD lie in real-time streaming input and no requirement for global access to historical data. Although the AGE method claims to be “on-the-fly,” it actually depends on a pre-trained encoder, a validation set for estimating class-specific thresholds, and the need to update means and covariance matrices whenever a new sample arrives. Does such a design—requiring continual updates of historical statistics—truly satisfy the assumption of “real-time streaming discovery”? In a continuous data stream scenario, how can its time and memory complexity remain feasible as data grows linearly?
2.	Does AGE merely transfer the thresholding mechanism from the hash space to the Gaussian density space? Is there any theoretical or empirical evidence that it can control the number of discovered categories more accurately? If the threshold is chosen improperly, could it also lead to category fragmentation or under-segmentation?
3.	The paper claims to distinguish known and novel samples through soft-thresholding, yet this decision relies on fixed β-based confidence statistics. If the class distribution of the validation set differs significantly from that of the test set, can the threshold remain reliable? Does this mechanism truly embody the dynamic adaptivity required by OCD, or does it only work under static distributions?
4.	The paper lacks intuitive experimental visualizations to demonstrate how categories dynamically emerge during the on-the-fly discovery process.
5.	The literature review is incomplete, missing several recently published related works.

### Questions
Please see the weakness.

### Soundness
2

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
3

### Summary
The paper presents a novel framework called Adaptive Gaussian Expansion (AGE) for On-the-fly Category Discovery (OCD). The paper identify a performance lower bound for existing OCD methods and reformulate the task into two sub-problems: open-set recognition and fully novel OCD setting. AGE uses a probabilistic approach based on Gaussian distributions to model known and novel classes, enabling real-time classification and clustering without prior knowledge of class numbers. The method shows significant improvements over prior work across multiple benchmarks.

### Strengths
1. This paper establishes a theoretical lower bound for OCD, exposing the structural flaw of existing hashing methods that fail to fully exploit known-class information, and provides a rigorous benchmark for future work.  
2. The task is decomposed into open-set recognition and novel class discovery: a threshold first filters known samples, then Gaussian clustering handles anomalies, cutting off most easy decisions early and reducing noise for the later stage.  
3. AGE estimates per-class mean and covariance and explicitly computes membership probabilities via the multivariate Gaussian PDF, capturing intra-class shape and scatter and yielding finer uncertainty estimates along decision boundaries.

### Weaknesses
1.All samples falling below the confidence threshold are forwarded to AGE as novel candidates, yet the paper offers no mechanism to identify those that are merely low-confident members of known classes. Consequently, old-class noise is injected into the subsequent Gaussian estimation, biasing the covariance estimates and spawning spurious clusters.
2.Key parameters in the equations are reported only as bare values without physical interpretation or selection criteria, raising the barrier to comprehension. It is recommended that all parameters appearing in the formulas be thoroughly explained.
3.The paper lacks sufficient references to relevant research from the past three years. This undermines the persuasiveness of its argument regarding timeliness and innovation. It is recommended to supplement the study with comparative experiments and discussions involving SOTA methods published in recent years.

### Questions
1.Does the superior predictive performance on emerging categories contribute to the prediction of old categories?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper analyzes OCD’s practical limits and establish a formal performance lower bound, motivating a reformulation into two sub-tasks: Open-Set Recognition and Fully Novel OCD. It introduces soft class thresholding to directly detect known classes, improving deployability, and propose Adaptive Gaussian Expansion (AGE), which models class PDFs to dynamically discover novel categories on the fly. Across multiple datasets, the proposed approach achieves state-of-the-art results.

### Strengths
1. The problem reframing is clear. Separating known-class retention (OSR) from novel class discovery is pragmatic and improves deployability.
2. The paper is clearly written and easy to follow.
3. The proposed method achieves strong empirical results.

### Weaknesses
1. The lemma 1 assumes shared covariance and whitening, which may be restrictive. The link from these assumptions to the actual AGE decision process (with smoothed per-class covariances and priors) could be tightened.
2. The soft class-wise threshold is somewhat heuristic. There is limited analysis of calibration, class imbalance, or alternative OSR baselines (e.g., energy score, MSP+temperature) for the thresholding module.
3. The paper describes deciding “falls within existing cluster” by maximum posterior but does not specify a principled threshold or DP concentration parameter mapping for new cluster instantiation; the “CRP-like” notion is qualitative rather than a formal nonparametric Bayesian update.
4. There're quite a lot of hyperparameters and the proposed method seems to be sensitive to hyperparameters. While some ablations are provided, key hyperparameters materially impact performance. Guidance or automatic selection strategies are limited.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
3
