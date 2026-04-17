# Not All Imbalance Is Random: Cluster-Balanced Ensembling for Missing-Not-At-Random Class Imbalance

- Decision: Reject
- Scores: 6, 2, 0, 2

## Abstract
Class imbalance methods inherently assume that observed minority instances are representative of their class and Missing At Random (MAR). However, in many real-world settings, minority instances are Missing Not At Random (MNAR), with observability shaped by both class and feature values. This leads to structurally biased samples, introducing a deeper challenge that goes beyond class-count imbalance. We show that when MNAR affects high-impact features, popular imbalance methods overfit the observed minority and fail to generalize. To address this, we propose a simple yet effective cluster-balanced ensemble approach that constructs diverse, near-balanced training sets by pairing all minority instances with different clusters of the majority class. Extensive experiments identify MNAR conditions under which our approach improves F1 scores over existing methods, and when it does not. We also introduce an evaluation protocol using representative balanced test sets, demonstrating that standard hold-out testing on MNAR data can mislead performance assessments. Our findings underscore that the cause of imbalance is as critical as the correction method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In this paper, the authors study the problem of class imbalance. To be specific, the authors argue that the test distribution is often imbalanced not in a random way: Data is missing-not-at-random (MNAR), which is not taken into account in the class imbalance literature so far. The authors provide an analysis and introduce a method borrowed from the MNAR literature. The introduced ensembling based method provides strong results on two tabular benchmarks.

### Strengths
+ MNAR appears to be an important issue overlooked in the class imbalance literature.
+ The provided analysis is helpful.
+ The introduced method borrowed from the MNAR literature appears to be effective.

### Weaknesses
Weaknesses:

1. The paper makes many claims without providing justifying references. The whole Introduction section is written without any references at all. 

2. They generate MAR and MNAR from a balanced dataset. Although this is understandable for an analysis, it is not clear what the scale of the problem is for naturally-collected long-tailed datasets. Therefore, it is not clear whether this is a real concern in natural datasets. I strongly suggest the authors to perform experiment with LT datasets such as ImageNet-LT, Places-LT, iNaturalist.

2.1. The authors argue that "sampling test sets from imbalanced dataset that are potentially MNAR, can bias true model performance." => This potential should not prevent one from performing experiments with such datasets. 

3. The figures in the paper have major issues.

3.1. Figure 1 is not referred to in the text. As the figure doesn't explain how the samples are generated or what kind of dataset these are, the figure fails to support the paper.

3.2. Fig 2: Text too small to read.

3.3. Figure 3 is very critical for the main motivations of the paper. However, (i) many variables (measures, SHAP values, feature values) are drawn in a complex manner without sufficient guidance in the figure or the caption, and (ii) the figure is not sufficiently explained in the text. 

4. I find the experimental evaluation weak.

4.1. For starters, the datasets are not used in the imbalance literature and therefore, the experimental evaluation fails to be convincing. Even for a balanced starting dataset, the paper could have preferred datasets commonly used: E.g., CIFAR10 and CIFAR100, which are converted into their balanced settings in a controlled manner. 

4.2. "These metrics were aggregated across datasets (to enable a data-set agnostic analysis) to predict which features induce MNAR that leads to poor performance by existing methods and benefit most by cluster-balanced ensembling." => Is the impact of the missingness of a feature not dataset-dependent? Aggregation over datasets makes it difficult to perform a problem-dependent deductions.



Minor comments:

- "follows well-known hybrid methods of undersampling the majority class" => Please cite.
- "euclidean distance" => "Euclidean distance".
- "This is to capture local structure defined by original regions of the feature space, rather than the normalized subspace that may compress important structure. In particular, normalizing before clustering often results in less diverse majority subgroups." => It would be nice to visualize this.
- "sever MNA" => "severe MNA".
- "Implementation and details of each algorithm is given" => "Implementation and details of each algorithm are given".

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the overlooked Missing Not At Random (MNAR) class imbalance, where minority samples’ observability depends on both class and features—unlike traditional methods’ assumption of Missing At Random (MAR), which causes overfitting to biased observed minorities and misleading evaluations (testing on MNAR-damaged data). The Cluster-Balanced Ensemble (CBE) method proposed by the authors significantly improves multiple metrics when samples are MNAR by clustering majority class samples and then combining each cluster with minority class samples separately to train multiple classifiers. This paper identifies MNAR’s limitations on traditional methods; introduces CBE to mitigate MNAR bias; characterizes critical MNAR-triggering features; and provides a protocol to avoid evaluation distortion.

### Strengths
1. The paper novelly introduces class imbalance in the form of MNAR, and the proposed method achieves better performance than traditional methods focused on MAR.
2. The experiments compare multiple metrics, test various types of classifiers, and expose biased MNAR evaluation.
3. The paper is well organized and motivated, making it easy to follow.

### Weaknesses
1. Gold-standard protocol relies on rare originally balanced datasets, and crucially, no comparisons are done on imbalanced test sets (e.g., from KEEL’s imbalanced dataset repository, where balanced test sets do not exist). This leaves uncertainty about whether CBE still outperforms others on real-world imbalanced test beds.

2. CBE only uses K-means for majority clustering. This paper would be better if it showed how CBE can adapt methods like DBSCAN (for irregular clusters) or hierarchical clustering (for nested structures), and how such adaptations affect performance.

3. It uses "top-5 high-impact features" for MNAR simulation without justifying the number (3 vs. 5 vs. 7) and ignores feature interactions. This undermines MNAR realism—testing varying feature counts and including interactions is needed.

4. Scalability oversight: No complexity analysis for CBE’s K-means + multi-classifier training is provided. For large datasets, Mini-Batch K-means or distributed training is unmentioned, and scalability benchmarks (e.g., runtime vs. data size) are missing.

5. **Potentially Biased MNAR simulation**: MNAR is simulated using mechanisms that maximize CBE’s advantages (e.g., features where other methods struggle most). MNAR construction targets (via XGBoost) the weaknesses of traditional methods (overreliance on observed minority structure) while exploiting CBE’s strengths (majority cluster-based structural coverage). This creates a scenario in which CBE’s advantages are artificially amplified, rather than a fair test of its robustness across diverse MNAR mechanisms. Simulating diverse MNAR mechanisms would better validate CBE’s robustness.

6. On balanced test sets (where overall accuracy is more relevant), the paper overemphasizes F1. It lacks F1 comparisons on imbalanced test sets, where F1 is more appropriate, leaving gaps in understanding CBE’s performance in target real scenarios.

7. This method (CBE) is limited to imbalanced binary tabular data, which narrows its practical scope. Specifically, CBE cannot be extended to non-tabular data (e.g., images) nor to multi-classification tasks such as long-tailed learning.

8. Lack of theoretical analysis on how MNAR harms more than MAR and how CBE improves it.

### Questions
Same to weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper studies Missing-Not-at-Random (MNAR) class imbalance and proposes a cluster-balanced ensembling (CBE) scheme: k-means partitions the majority into ≈|majority|/|minority| clusters; each cluster is paired with all minority points to train base learners; predictions are PR-AUC–weighted. Using small, near-balanced tabular datasets (PMLB + one finance dataset), MNAR is simulated by deleting minority instances by feature value; evaluation is done on the original balanced test folds. CBE reports higher F1 than many baselines (SMOTE, ADASYN, Tomek, EasyEnsemble, etc.). While the MNAR emphasis and simple CBE are interesting, the heavy reliance on outdated baselines, limited novelty over prior cluster-ensemble ideas, and narrow evaluation setting put this below ICLR’s threshold for novelty and soundness in its current form.

### Strengths
+ The gold-standard evaluation idea (train on damaged MNAR data, test on the original balanced folds) is thoughtfully motivated and reveals pitfalls of conventional hold-out on biased test sets.
+ CBE is simple, reproducible, and competitively strong on the authors’ tabular MNAR simulations; tables/figures show consistent F1 gains over several baselines and classifiers.
+ The paper probes feature conditions that exacerbate MNAR (via SHAP over meta-features), which I find informative.

### Weaknesses
+ Baseline set is heavily outdated relative to ICLR expectations. Most “state-of-the-art” comparisons are classic reweighting, SMOTE variants, Tomek Links, Cluster Centroids, Easy Ensemble, and basic cost-sensitive losses—largely pre-deep-long-tail era and often with default imbalanced-learn/sklearn settings. There is no comparison to modern methods: margin-aware re-balancing (e.g., logit-adjustment with tuned τ), distributionally robust optimization, meta-reweighting, deferred reweighting, AUCPR-direct objectives, calibrated thresholding with risk control, or recent long-tail generalization techniques. This makes the claimed “consistent state-of-the-art” gains hard to accept for ICLR.
+ Method novelty is limited. CBE is essentially clustering-guided undersampling + ensembling. Similar ideas (e.g., EKR; clustering-centroid undersampling; balanced bagging) exist, with the main tweak here being that k is set by the imbalance ratio and all minority points are reused. The conceptual leap beyond known cluster-based ensembles is modest.
+ All evidence is on small tabular datasets with binary labels and simulated MNAR; there are no results on image/text/representation-learning regimes where feature geometry is non-Euclidean and k-means on raw features is questionable. The approach also fixes Euclidean k-means “on original unscaled features,” which can be brittle and scale-dependent.
+ he base classifiers are LR/SVM/RF/MLP/XGBoost with minimal tuning; no modern tabular SOTA (e.g., strong GBDT variants with tuned class weighting, TabTransformer, FT-Transformer) or representation learning is attempted. Reported superiority over such a baseline pool does not clear ICLR’s bar.
+ The paper centers F1; while PR-AUC appears in plots and for voting, there is no calibration or operating-point analysis under MNAR (sensitivity@specificity per subgroup), which is central to the motivation. F1 also has been criticized for usage in imbalanced datasets.

### Questions
+ Can you include modern baselines (e.g., DRO, meta-reweighting, AUCPR-direct losses, recent long-tail re-balancing with tuned priors/margins) and strong tabular SOTA (well-tuned LightGBM/CatBoost with class weights) to substantiate the “SOTA” claim? 
+ How sensitive is CBE to feature scaling and the choice of distance/representation for clustering? Have you tried clustering in a learned metric space (e.g., supervised embedding) rather than raw Euclidean space “on the original unscaled dataset”? 
+ Could you report calibration and thresholded metrics under MNAR (e.g., precision/recall at fixed costs) and include cost-weighted utilities to justify F1-centric conclusions?
+ Beyond simulated MNAR, can you provide semi-synthetic or cross-site evaluations, or at least stress tests where train/test MNAR mechanisms differ, to probe robustness of CBE?
+ What happens when k deviates from the imbalance ratio or when minority is extremely small (e.g., 100+:1)? Please include compute/time and memory costs vs. stronger baselines.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates data imbalance under the Missing Not At Random (MNAR) setting. The authors observe that commonly used imbalance-handling methods may fail when MNAR affects high-impact features. To address this issue, they propose a cluster-balanced ensemble approach that constructs diverse, near-balanced training sets by pairing each minority instance with different clusters of the majority class. The effectiveness of the proposed method is demonstrated through extensive experiments.

### Strengths
1. Methods that effectively handle data imbalance are important in practice.
2. The proposed approach is intuitive and easy to implement.
3. The method demonstrates improved performance in the experimental results.

### Weaknesses
1. The paper is not well written. The description of the methods is mostly textual and lacks a clear, organized structure. In addition, the discussion of the underlying intuition, advantages and limitations, and potential extensions of the proposed approach is quite limited.
2. The paper does not include any theoretical analysis or discussion to support the proposed method.
3. The novelty of the proposed approach is unclear.

### Questions
1. Is there any theoretical justification or analysis supporting the proposed method?
2. The number of clusters k is set as round($\frac{n^-}{n^+}$). Is this choice always optimal, or how sensitive is the performance to this parameter?

### Soundness
2

### Presentation
2

### Contribution
2
