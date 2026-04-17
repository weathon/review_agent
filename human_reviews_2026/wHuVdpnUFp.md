# xRFM: Accurate, scalable, and interpretable feature learning models for tabular data

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Inference from tabular data, collections of continuous and categorical variables organized into matrices, is a foundation for modern technology and science. Yet, in contrast to the explosive changes in the rest of AI, the best practice for these predictive tasks has been relatively unchanged and is still primarily based on variations of Gradient Boosted Decision Trees (GBDTs). Very recently, there has been renewed interest in developing state-of-the-art methods for tabular data based on recent developments in neural networks and feature learning methods.  In this work, we introduce xRFM, an algorithm that combines feature learning kernel machines with a tree structure to both adapt to the local structure of the data and scale to essentially unlimited amounts of training data. We show that compared to $31$ other methods, including recently introduced tabular foundation models (TabPFN-v2) and GBDTs,  xRFM achieves best performance across $100$ regression datasets and is competitive to the best methods across $200$ classification datasets outperforming GBDTs. Additionally, xRFM provides  interpretability natively through the Average Gradient Outer Product.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new architecture for tabular prediction tasks called xRFM. Specifically, xRFM combines feature learning kernel machines with a tree structure to both adapt to the local structure of the data. Empirically, the authors showed that xRFM outperforms various baselines in tabular regression tasks and shows competitive results in tabular classification tasks.

### Strengths
1. The proposed method is fast, scalable, efficient, and also accurate.

2. The authors have done comparative experiments on a total of 300 datasets. This is a really big contribution to the tabular learning society and ICLR.

3. xRFM is a very efficient solution of tabular prediction tasks, while ensuring competitive performance.

4. The paper is well written.

5. Tabular prediction tasks are challenging and important problems in ML society. Moreover, I highly agree that this field is still dominated by simple GBDTs and some new architectures should be proposed progressively.

### Weaknesses
1. Can the authors provide the win rate of xRFM compared to competitive baselines like TabPFN-v2, RealMLP, etc?

2. Is xRFM also effective for few-shot tabular predictions tasks, where the number of class-per samples is just 1 or 5?

3. Are you going to open-source this project?

### Questions
See the above Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes xRFM, a model that uses the Average Gradient Outer Product (AGOP) to identify directions of maximal predictive variation, recursively splitting the data and fitting local kernel ridge regressors in each leaf. Overall, this is a solid paper that effectively demonstrates the usefulness of AGOP for tabular data.

### Strengths
This is a solid paper that effectively demonstrates the usefulness of AGOP for tabular data.

### Weaknesses
See Questions below.

### Questions
I have a few minor points for the authors to address:

1. In Equation (2), what happens if the gradients point in opposite directions or are orthogonal? Could this cause AGOP to fail or weaken its signal?

2. Please investigate or discuss why the proposed method performs well on TALENT, but not better than fine-tuned CatBoost on TabArena.

3. Could gradient boosting be combined with xRFM to further improve performance? A small-scale study or discussion on this would be helpful.

4. Figure 2 nicely illustrates how xRFM captures feature interactions. For this method, is it possible to derive a feature importance ranking similar to that in XGBoost? Please consider adding a short discussion.

5. Please consider performing a post-hoc meta-analysis to identify which types of datasets favor xRFM and which ones favor existing methods such as PFN or CatBoost.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces **xRFM**, a novel algorithm for learning from tabular data that integrates **feature-learning kernel machines** with **adaptive tree-based data partitioning**. The key idea is to build a binary tree that recursively splits the data along informative directions derived from the **Average Gradient Outer Product (AGOP)**, training a *Recursive Feature Machine (RFM)* at each leaf. xRFM achieves **log-linear training complexity** and **logarithmic inference time**, scaling effectively to very large datasets (up to 500K samples). It is evaluated extensively on **TALENT**, **TabArena-Lite**, and **Meta-Test** benchmarks, outperforming 31 baselines (including GBDTs and TabPFN-v2) on regression tasks and remaining competitive on classification tasks.

### Strengths
* The combination of **kernel-based feature learning** (via RFM) with **tree-based local partitioning** and echanism based on **AGOP decomposition** is conceptually novel and elegant.

* The paper demonstrates rigorous experimental validation across **three major benchmarks** and **hundreds of datasets**, establishing strong empirical credibility. The comparisons with baselinse are comprehensive, with fair baselines including recent foundation models like **TabPFN-v2** and **RealMLP**.
* The **scaling analysis** (O(n log n) training, O(log n) inference) is clearly supported by empirical runtime curves (Fig. 5).
* The method maintains theoretical ties to kernel learning and AGOP-based supervised PCA, providing a sound foundation for interpretability and feature relevance analysis.
* The paper is very well written, with clear motivation, detailed algorithmic descriptions (Algorithms A.1–A.5), and informative figures (especially Fig. 1,2).
* The paper addresses a long-standing gap: scalable, interpretable, and high-performing models for **tabular data**. The reported performance and efficiency make xRFM a viable candidate for large-scale deployment in applied ML contexts (e.g., finance, healthcare, industrial analytics).

### Weaknesses
* While the empirical results are compelling, the paper lacks a **formal analysis** of convergence or generalization bounds for the tree-partitioned RFM structure.

* The paper could better isolate the contributions of individual components:
    * How much performance gain is from **tree-based scaling**? For example a comparison with normal RFM can show it.
    * How much performance gain is from **AGOP**? If using normal tree splitting, how much will the performance drop? What if using other splitting method?

### Questions
See weakness

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
This paper addresses the limitations of traditional tabular data modeling in terms of scalability and feature learning capabilities and proposes a new model, xRFM. This model aims to simultaneously possess (1) local feature learning capabilities, (2) interpretability, and (3) computational efficiency that can be scaled to very large datasets, thereby surpassing existing gradient boosted decision trees (GBDTs) and the recent tabular data base model in both regression and classification tasks.

### Strengths
S1. Combining RFM with tree-based data partitioning and using AGOP main directions for supervised splitting, this combination of local feature learning and scalable kernel methods is novel in tabular ML.

S2. The method's detailed derivation is clear, and the theoretical background and splitting criteria for AGOP are clearly referenced and supported. Technical improvements, such as kernel parameter space exploration, categorical variable optimization, and bandwidth adaptation, are detailed.

S3. Experiments covering ultra-large datasets (>500,000 samples) demonstrate computational scalability. Multiple benchmarks validate its advantages in regression tasks.

S4. Feature interpretation is natively supported, eliminating the need for external tools. Locally relevant features can be analyzed directly from the AGOP of each leaf node.

### Weaknesses
W1. Based on experimental tables (such as the TALENT binary and multi-classification results), xRFM is only competitive in most classification tasks, rather than significantly leading. Significant improvements in a single regression domain do not fully demonstrate the model's versatility.

W2. The results lack fine-grained analysis comparing different model families. Although benchmark rankings are reported, there is a lack of grouped performance analysis for model families (Tree-based, Kernel-based, and NN-based), which fails to clearly demonstrate in which structural scenarios xRFM excels. The charts focus on overall rankings and do not show the correlation between different task attributes and performance.

### Questions
Q1. Is the AGOP splitting criterion stable in the case of high-dimensional sparse features? Has a comparison been conducted with splitting methods based on axis information gain?

Q2. What is the main bottleneck that causes xRFM's performance to lag behind in multi-classification tasks? Is it the tree splitting strategy, the Leaf RFM structure, or insufficient hyperparameter tuning?

Q3. If xRFM is combined with TabPFN-v2 (as a backend or feature extractor), can the gap in classification tasks be narrowed?

### Soundness
3

### Presentation
4

### Contribution
3
