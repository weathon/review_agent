# Effective Data Pruning through Score Extrapolation

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Training advanced machine learning models demands massive datasets, resulting in prohibitive computational costs. To address this challenge, data pruning techniques identify and remove redundant training samples while preserving model performance. Yet, existing pruning techniques predominantly require a full initial training pass to identify removable samples, negating any efficiency benefits for single training runs. To overcome this limitation, we introduce a novel importance score extrapolation framework that requires training on only a small subset of data. We present two initial approaches in this framework—k-nearest neighbors and graph neural networks—to accurately predict sample importance for the entire dataset using patterns learned from this minimal subset. We demonstrate the effectiveness of our approach for 2 state-of-the-art pruning methods (Dynamic Uncertainty and TDDS), 4 different datasets (CIFAR10, CIFAR100, Places-365, and ImageNet), and
training paradigms (supervised, unsupervised, adversarial). Our results indicate that score extrapolation is a promising direction to scale expensive score calculation methods, such as pruning, data attribution, or other tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel score extrapolation framework that eliminates the need for full training on the original dataset before pruning. The framework trains on a subset of the original data and extrapolates scores using k-nearest neighbors (KNN) and graph neural networks (GNN) based on the trained subset's scores. The method is validated across various datasets and training paradigms based on two different scoring methods. Their approach achieves Pareto-optimal time-accuracy trade-offs and significantly reduces computational effort.

### Strengths
1. The paper's main contribution addresses a key challenge in existing data pruning methods: ironically, these methods take longer to prune and train on a pruned dataset than to train fully on the original dataset. This paper mitigates this issue by calculating scores only on a subset of data points and extrapolating scores for the remaining data, rather than computing scores for every data point. This methodology arises from a strong motivation and solves the problem in a reasonable way.
2. The authors tested their score extrapolation methods on various datasets (Places365, ImageNet, Synthetic CIFAR-100, and CIFAR-10). They experimented across multiple settings—including supervised, unsupervised, and adversarial training—to investigate the applicability of score extrapolation methods. They also conducted various ablation studies, such as examining the correlation with original scores depending on initial sample size.
3. Importantly, the authors include failure cases in Appendix C.2 to highlight the current limitations of their framework, demonstrating transparency and a careful evaluation of their method.

### Weaknesses
1. Section 3 lacks the clarity necessary for comprehension and reproducibility. The main issues are as follows:
    1. What is $D$ in line 166 refer to? The notation should be clearly defined. Moreover, the symbol $D$ overlaps with the distance notation $D(\cdot, \cdot)$ used in line 210, which may cause confusion.
    2. In line 178, the authors defined the embedding function $\phi: \mathcal{X} \to \mathbb{R}^d$ that maps an input $x$ to embedding $z = \phi(x)$. 
        - First, $x \in \mathcal{X}$ should be explicitly stated.
        - Second, in lines 200~212 (*the Extrapolation with KNN paragraph*), the model is defined as $\mathcal{F}_s: \mathbb{R}^d \to \mathbb{R}^{d'}$. Here, $\mathbb{R}^d$ is used as the input dimension and $\mathbb{R}^{d'}$ as the output dimension, which differs from the earlier definition.
        - Moreover, $\mathcal{F}_s$ is described as an embedding function (outputting embeddings) rather than a neural network that outputs logits, which contradicts the definition in line 178. The authors should consistently use $\phi_s$ to denote the embedding function for clarity and consistency.
    3. In Equation (2), $y$ should be explicitly defined, including its dimensionality.
    4. In lines 171 and 175, the authors define $S_s$ and $S_r$ as vectors. However, in Equation (6), the same notation $S$ is used both as a function ($S_{knn}$), and as a scalar ($S_{\pi_i(x)}$). This inconsistency becomes more pronounced in lines 230-237, where $S_s$ and $S$ are again treated as functions. If $S$ is intended to represent a “score”, its form should be clearly and consistently defined to avoid confusion for readers. 
    5. The authors should explicitly state that $\pi_i(x) \in [m]$, indicating that the index of the $i$-th nearest neighbor of $x$ is selected from within $\mathbb{D}_s$.
    6. In line 221, $d(\cdot, \cdot)$ is used as a distance metric, which is inconsistent with the notation $D(\cdot, \cdot)$ defined in line 210-211.
    7. In Equation (7), the authors use $\sum_{x_i \in \mathbb{D}\_s}$, but the term $\mathcal{F}\_\mathcal{G}(\mathcal{A}, \mathcal{V};\theta)\_i$ explicitly depends on the index $i$. Therefore, I recommend replacing $\sum\_{x\_i \in \mathbb{D}\_s}$ with $\sum\_{i \in [m]}$, since the authors have already defined $\mathbb{D}\_s = \lbrace x_i \rbrace_{i \in [m]}$.
    
    Overall, the Section 3 should be rewritten for clarity, consistency, and reproducibility. The notations should be clearly defined and used consistently, and any ambiguous or overlapping definitions should be eliminated.
    
2. Comparison with recent SOTA baselines could further strengthen the proposed method:
    1. For example, the DUAL method (with its $\beta$-sampling scheme) explicitly targets reducing the score-computation time by jointly considering data uncertainty and difficulty [1].  Since your paper focuses on effective data pruning, this baseline appears directly relevant. It would be valuable either to compare your proposed method with DUAL, or to incorporate DUAL’s score as a base for your method, potentially improving overall pruning efficiency when combined with its sampling strategy.
    2. Also, methods such as Coverage-centric Coreset Selection (CCS) [2] and $\mathbb{D}^2$ [3] involve more expensive score-computation but achieve very competitive performance. These could serve as strong base scores or benchmarks for your method, especially if you aim to show improvement in computation vs. performance trade-off.
    
    To adopt the sampling methods from [1] and [2], one can first compute subset scores, extrapolate them to estimate the unevaluated samples, and then apply sampling based on these extrapolated scores.
    
3. Are there specific reasons for using the *Synthetic CIFAR-100* dataset instead of the standard CIFAR-100? Since the abstract (line 21) mentions CIFAR-100, readers may be confused when encountering Synthetic CIFAR-100 in the main text. As a reader unfamiliar with this dataset, I believe the paper should clearly explain:
    1. What this dataset is, including how it relates to the original CIFAR-100; and
    2. Why the authors chose to use this dataset instead of the standard CIFAR-100.
    
    My understanding is that the authors used the Synthetic CIFAR-100 dataset because the score extrapolation process requires computing scores on a subset first, and the absolute size of the dataset plays a critical role in this step. Nevertheless, I believe readers will still be interested in the model’s performance on the original CIFAR-100, especially since results on the original CIFAR-10 are already provided. Therefore, I recommend including the original CIFAR-100 results (at least in the appendix or as a stated limitation).
    
4. Why are there no data pruning results in the unsupervised setting? Currently, only Pearson and Spearman correlation results are reported for the CIFAR-10 dataset in Table 3. Since the scores have already been computed, it should not be difficult to perform dataset pruning based on these scores and then train the pruned subset in a supervised manner. Given that the paragraph is titled *“Unsupervised Data Pruning”,* readers (including myself) would naturally be interested in seeing the performance of a pruned subset whose scores were calculated in an unsupervised manner. I therefore recommend that the authors include the performance results of such a pruned subset.
5. As stated in Appendix C.2, there are certain datasets for which the score extrapolation method fails to accurately capture the original score distribution. Moreover, as shown in Figure 4 and Table 2, the correlations with the original scores are not particularly high. This issue is also reflected in the pruned dataset performance presented in Figure 2, where the pruning results are not significantly better than those of other pruning methods. Taken together, these observations suggest that the proposed approach behaves more like an interpolation between random pruning and full score computation, rather than a method that resolves the trade-off between computation and performance. In this regard, it would be helpful to include the results of random pruning in Figure 3 for comparison. Furthermore, I recommend that the authors to include the results of EL2N [4] and DUAL [1], which are both time-efficient data pruning methods.

---

**References:**

[1] Lightweight Dataset Pruning without Full Training via Example Difficulty and Prediction Uncertainty, ICML 2025.

[2] Coverage-centric Coreset Selection for High Pruning Rates, ICLR 2023.

[3] D2 Pruning: Message Passing for Balancing Diversity and Difficulty in Data Pruning, ICLR 2024.

[4] Deep Learning on a Data Diet: Finding Important Examples Early in Training, NeurIPS 2021.

### Questions
1. Why do the SSP and Zcoreset methods, which are training-free, take so much time to create a subset in Figure 3?
2. What does each point in Figure 3 represent? How were these points determined? If each point corresponds to a different pruning ratio, this should be clarified. Similarly, what does each point in Figure 4 represent? It seems likely proportional to the subset size used for score calculation, but this needs explanation.
3. When should KNN be used versus GNN? Which method is appropriate under which circumstances, and why? (Also, clarification is needed for lines 412–413. What does "This might be ~" mean exactly?)
4. In the *Adversarial Training* paragraph, as a reader unfamiliar with adversarial training, it is difficult to understand the metrics used in Table 4 and Table 5. The authors should clarify the definitions of “Robustness” and “Clean Accuracy.”
5. What if we train an independent model that maps embeddings to subset scores? Specifically, up to Step 3 in Figure 1, the process remains the same, but for the extrapolation step, we train a small neural network (e.g., an MLP) on the subset where the scores have been computed, and then evaluate it on the remaining data to predict their scores. How would such an approach perform? 
6. As far as I understand, the proposed method requires labeling only the initial subset when pruning the dataset, even in the supervised setting. Is that correct? Since only the embeddings of the remaining data are used when extrapolating the scores, if the pruning can indeed be performed without access to all labels, I think the authors should highlight this point. For example: “Our method requires labels only for the initial subset, thereby reducing labeling costs and enabling application to large-scale web datasets.”
7. There appears to be overlapping information among Table 1, Figure 2, and Figure 3. The authors might consider merging or simplifying these to avoid redundancy and improve readability.
8. Is there a specific reason for reporting *relative accuracy* in Table 1? It might be more intuitive to present the *actual accuracy* values while expressing *time efficiency* in relative terms, as this would make the comparison easier to interpret.

---
**Minor Corrections:**

1. Line 149: data sets → datasets
2. Figure 2(c) and Figure 6(c): Synthic → Synthetic
---
If the authors adequately address the issues raised in the Weaknesses and Questions sections, I would consider increasing my overall score.

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
Traditional pruning methods require training computing importance scores on all the samples of the dataset. Computing these scores often requires training a model for multiple epochs on the samples, which can become costly. This work proposes to compute the scores (and train an initial model) on only a small fraction of the samples and extrapolate to the rest of the dataset. To do so, they propose two different technics, one based on K-Nearest Neighbours and the other training a Graph Neural Network, both using the geometry induced by the embedding space.

### Strengths
In general I found this work interesting, it tackles a less explored direction in data pruning and can indeed bring valuable computational gains.
- The notion of extrapolating importance scores from a small subset is simple but original (to my knowledge), and it provides a new angle on making pruning efficient. 
- The KNN and GNN approaches effectively demonstrate that extrapolation can cut computation time with little performance loss and form a good proof of concept.
- The work evaluates multiple datasets, and training settings, which supports the generality of the approach.

### Weaknesses
Overall, the paper presents an interesting idea, though I think it could be strengthened with some further development and analysis.
- I did not find the theoretical justification very convincing. It seems that the main point is about the smooth interpolation of the samples influence that the authors use as a justification for their extrapolated scores (eq 6). But in the context of influence the extrapolated point is itself a convex interpolation of the reference points, and the weights are the associated factors, whereas in eq 6 we are assigning scores using an arbitrary formula depending on the distance to the reference points. Also the authors could introduce what is influence and why it would be relevant here.
- I think the paper could benefit from a comparison with other simple alternatives to reduce computations (see questions). 
- Dynamic data pruning methods are not considered in this work, though they are very popular and somehow connected to this since they face the problem of updating the scores and can do it on only a fraction of the samples every time to avoid prohibitive costs.

### Questions
Multiple "simple" alternatives could be compared to your proposed approach to strengthen the contribution
- Instead of training completely on the whole dataset, how would the proposed approach compare to training only a few epochs and use this intermediate model to score, then keep training the same model with only the selected fraction, which is often what is done in practice. 
- Could one use directly your model $\mathcal{F}_s$ trained on the selected subset and score the unseen points without using extrapolation ?
- In table 1, how would you compare to using the "ground truth scores" for the selected subset and random for the unseen, without needing extrapolation ?

- In line 152-153, the authors write that the trained model $\mathcal{F}_s$ depends on the selected extrapolation method, but from lines 167-172 it seems to be independent of the extrapolation method, could you clarify ?

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
This paper introduces a novel and efficient approach to data pruning. The method first estimates the importance of each training example by analyzing only a small subset of the data. It then extrapolates these importance scores to the entire dataset using one of two proposed strategies: one based on k-nearest neighbors (KNN) and the other on a graph neural network (GNN). Both strategies leverage embeddings from a model trained exclusively on the small subset. The authors evaluate their approach on four large-scale image datasets, covering supervised learning, unsupervised learning, and adversarial training scenarios. There are comprehensive experiments compare the method against state-of-the-art pruning techniques and simple baselines. Results demonstrate that the proposed approach significantly reduces computational cost while achieving accuracy that matches or closely approaches that of existing methods.

### Strengths
- The paper tackles a practically significant and under-addressed problem: how to make computationally expensive data pruning methods tractable for large-scale training by requiring only a small subset for direct score computation.
- The proposed score extrapolation framework is methodologically interesting and is instantiated with both a simple, transparent KNN approach and a more expressive, message-passing-based GNN, allowing for a clear analysis of trade-offs.
- Empirical validation is thorough across multiple datasets—CIFAR-10, CIFAR-100, Places-365, and ImageNet—and considers a range of pruning rates, tasks (supervised, unsupervised, adversarial), and multiple pruning methods (DU and TDDS).
- The efficiency benefit is systematically explored, with computing time and accuracy jointly visualized. As demonstrated in **Table 1**, the proposed KNN and GNN approaches recover a substantial portion of the original pruning benefit at a fraction of the computational cost, achieving notable speedups.

### Weaknesses
1. **Limited theoretical justification and over-reliance on local linearity assumptions:** 
   The primary mathematical support for extrapolation is drawn from influence function and local linearity arguments (Section 3). Yet, there is insufficient theoretical development or empirical diagnosis regarding the validity of these assumptions for highly nonlinear, high-dimensional representation spaces found in deep learning. As such, generalizability of the approach to broader architectures/tasks remains open.
2. **Score oversmoothing and limited modeling of complex distributions:** A main weakness shows up in **Figure 5** (Appendix C.2). Both the KNN and GNN methods tend to oversmooth the extrapolated importance scores. They miss the multiple peaks that exist in the original score distribution. This makes the approach less effective when data importance isn’t uniform—for example, when there are clearly two different groups of samples with different levels of importance. It can also lead to consistent ranking mistakes, especially for outliers or samples that are hard to classify. The paper notes these issues, but it doesn’t fix them with new experiments or method changes.
3. **Modest accuracy improvement and moderate correlation, especially at small subset sizes:** The extrapolation quality drops when the subset is too small or when a lot of data is pruned. This holds true both for how well the scores match the true importance and for the final model accuracy.  In many cases, even the best extrapolation method doesn’t fully match the performance of the original pruning approach. The results also change noticeably depending on the size of the subset used.  The method does offer real speedups. However, it doesn’t always give a better balance between accuracy and resource use. In settings with very tight resource limits, this could be a serious limitation.
4. **Mathematical construction and empirical transparency:** 
   Some mathematical steps and notation—in particular, the loss formulations for the GNN extrapolation and details of embedding construction—are not explained in enough depth (e.g., Section 3, Equation for GNN loss). The explanation of exactly how node features combine (what happens with missing labels in unsupervised, for instance), and the optimization details, are relegated to appendices and could benefit from greater clarity in the main paper.
5. **Potential failure cases insufficiently investigated:** 
   The brief exploration of failure modes (e.g., rank differences related to background/outliers in **Figure 5**) is insightful but limited. More systematic error analysis—including quantifying what data characteristics lead to the highest extrapolation mis-rankings or when the approach could harm downstream performance—is warranted.

### Questions
1. Could the authors elaborate on the limitations of the local linearity assumption in embedding space? Specifically, how does this assumption break down for highly heterogeneous data distributions, and are there diagnostics or empirical controls to quantify this risk in practice?
2. In GNN extrapolation, how are class labels handled for semi-supervised or unsupervised settings, especially when no reliable pseudo-labels are available? Would the approach degrade under severe class imbalance or label noise?
4. What strategies do the authors propose to mitigate oversmoothing and better capture multimodal or long-tailed importance score distributions in future work?
5. How does the extrapolation performance vary with increasing/heterogeneous dataset size (e.g., simulated “billion-sample” settings), or for modalities outside vision (e.g., text, multimodal tasks)?

### Soundness
3

### Presentation
2

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
This paper addresses the computational efficiency problem in data pruning methods by proposing a novel score extrapolation paradigm. The core idea is to compute importance scores on a small subset (10-20%) of the training data and then extrapolate these scores to the remaining unseen samples using KNN or GNN-based methods. The authors demonstrate that this approach achieves up to 4.9× speedup while maintaining competitive downstream task performance across supervised, unsupervised, and adversarial training settings on datasets including ImageNet, Places365, CIFAR-10, and synthetic CIFAR-100.

### Strengths
1. Using the Extrapolation method to refine the sample ranking is rather new.

### Weaknesses
1. The critical weakness is the overlooked training cost.  The method requires: 1) First training: Train on random 10-20% subset to compute initial scores. 2) Embedding + Extrapolation: Extract features and extrapolate scores for remaining 80-90%. 3) Second training: Train final model on the extrapolated-pruned subset.

1.1 This means training happens TWICE on similar-sized subsets, plus embedding the full dataset.

1.2 Did the authors account for the cost of BOTH training phases? The paper claims "4.9× speedup" but it's unclear if both training runs are included in the time measurement.

1.3 What is the cost of embedding 100% of the data? This requires forward passes through the entire dataset, which is not negligible.

1.4 What is the actual extrapolation cost? KNN search or GNN inference on large-scale datasets has computational overhead.

1.5 
The paper should clearly demonstrate that:
 ```
Cost(Training 20% + Embedding 100% + Extrapolation + Training 20%) 
< 
Cost(Training 100% once + Standard pruning)
```

2. The code provided by the authors cannot be opened.

3. On line 752, the authors mentioned "For the unsupervised setting, we employ DINOv2 (Oquab et al., 2023) as a foundation model to obtain fixed embeddings for all samples. ". Why not use this as a baseline and see how much the supervised method can improve?

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
