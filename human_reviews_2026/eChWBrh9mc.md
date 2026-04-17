# EntryPrune: Neural Network Feature Selection using First Impressions

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
There is an ongoing effort to develop feature selection algorithms to improve interpretability, reduce computational resources, and minimize overfitting in predictive models. Neural networks stand out as architectures on which to build feature selection methods, and recently, neuron pruning and regrowth have emerged from the sparse neural network literature as promising new tools. We introduce EntryPrune, a novel supervised feature selection algorithm using a dense neural network with a dynamic sparse input layer. It employs entry-based pruning, a novel approach that compares neurons based on their relative change induced when they have entered the network. Extensive experiments on 13 different datasets show that our approach generally outperforms the current state-of-the-art methods, and in particular improves the average accuracy on low-dimensional datasets. Furthermore, we show that EntryPruning surpasses traditional techniques such as magnitude pruning within the EntryPrune framework and that EntryPrune achieves lower runtime than competing approaches. Our code is available in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose EntryPrune, an embedded feature-selection method built on a dense MLP with a dynamically sparse input layer. The key idea is an entry-based pruning criterion. Specifically, when a pruned input feature is (re)introduced as a candidate into the input layer, the algorithm measures the short-horizon relative change it induces. This is operationalized as the z-scored L1-norm of its first-layer gradient accumulations over the first n mini-batches post-entry. These accumulated scores are stored for all features. Periodically, the algorithm identifies the K features with the largest stored entry scores and prunes the rest. Then, it randomly regrows a batch of new candidate inputs whose incoming weights are reinitialized to tiny values. According to the authors, this strategy balances between established and freshly (re)introduced features and mitigates the bias of magnitude-based pruning toward long-tenured inputs. 
Empirically, across 13 datasets (tabular, vision, speech, genomics, and text), EntryPrune often outperforms or matches SotA feature selection baselines (NeuroFS, LassoNet, STG, QS, RFS, etc.). An ablation suggests that entry-based scoring (short-horizon gradient sums) beats magnitude‐based criteria inside the same framework. The proposed method often runs faster wall-clock than a sparse-training baseline (NeuroFS). The code and replication scripts are provided.

### Strengths
1. The algorithmic contribution is clear. The entry-based pruning idea, i.e., scoring features by their initial impact upon (re)entry and freezing that score to avoid tenure bias, is straightforward, well motivated, and easy to implement. The random regrowth and tiny reinitialization are thoughtfully chosen to avoid gradient suppression and encourage exploration.
2. The empirical evaluation is nice. The paper evaluates on 13 datasets spanning images, speech, sensor, and genomics, with reports of SVM accuracies and additional downstream models. Various ablations strengthen the analysis. 
3. The code is provided; many baselines are taken from public repositories. The paper documents settings and includes extensive appendix tables.

### Weaknesses
1. The related work discussion is incomplete. For example, recent approaches, such as "CancelOut: A Layer for Feature Selection in Deep Neural Networks" by Borisov et al., or "Leveraging model inherent variable importance for stable online feature selection" by Haug et al. are not discussed. In fact, I think they should be considered as competitors in the evaluation. 
2. Uncertainty quantification plays a critical role when it comes to feature selection. EntryPrune does not consider this aspect. Is there a way to measure/quantify the uncertainty of the selection process in EntryPrune?
3. It is not clear what the contribution of the entry metric from regrowth is. Is it possible to isolate this contribution?
4. The robustness analysis could be much deeper. Would a simple exponential decay or batch-normalized cumulative score improve robustness? In which scenarios does EntryPrune fail? What is the impact of concept shifts or other types of drifts in the data on EntryPrune's performance?

### Questions
See comments above.

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
3

### Summary
This paper proposes EntryPrune, a feature selection method that combines a dense neural network with a dynamically sparse input layer, where features are iteratively pruned and regrown. The main idea is an entry-based pruning metric that evaluates each new feature based on its early gradient-driven impact. The method is benchmarked on 10+ datasets, and (according to the results in the manuscript) it shows strong performance particularly on datasets with more samples than features.

### Strengths
- Novel method
- Good experimental coverage (13 datasets) 
- The paper is well written and easy to follow

### Weaknesses
- Although the paper motivates feature selection as a path to interpretability, it does not connect its contribution to established explainability methods such as SHAP, LIME, Integrated Gradients, or Grad-CAM. Given that the method relies on gradients and is applied to image data, this omission weakens the interpretability claim

- On wide datasets, the method offers little to no improvement over existing baselines

- The experimental setup relies mainly on SVMs, which are somewhat outdated, incorporating more modern models such as GBDTs or Random Forests would provide a fairer and more relevant comparison

- The evaluation also omits tree-based feature selection baselines (e.g., feature importance from Random Forest or GBDT), which are widely used in practice and should at least be discussed

### Questions
- How does EntryPrune compare to attribution-based interpretability methods (e.g., SHAP, LIME, Grad-CAM, ...)? Could the authors clarify whether EntryPrune should be viewed as a competing interpretability approach or a complementary one?
- Please see the "Weaknesses" section

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript proposes a supervised feature selection algorithm called EntryPrune. The algorithm is based on a dense neural network with a dynamically sparse input layer. The core mechanism is claimed to be "entry-based pruning", evaluating the importance of a neuron(feature) based on the relative change induced when it first enters the network, combined with a random regrowth strategy. The authors claim that this method outperforms (or matches) sota on 13 datasets, especially on "long" datasets.

### Strengths
1. The pruning approach proposed in this paper is an interesting heuristic, which attempt to address the issue of unfair evaluation time between new and old neurons in dynamic sparse training.
2. Compared to NeuroFS and LassoNet, the proposed method may have lower computation time while maintaining comparable performance.

### Weaknesses
1. Dynamic sparse training is a widely researched and used approach. The method proposed in this manuscript is more like an incremental improvement on the existing NeuroFS framework. The most creative part is the introduction of a new pruning metric strategy. In addition, this manuscript avoids any theoretical analysis of its effectiveness.

2. The manuscript seems to have ignored GBDT baselines e.g., xgboost and catboost, in the main text, and appendix B also seems to show GBDT's powerful ability to identify interactive features.

3. The manuscript argues that "random regrowth" is superior to "gradient-based regrowth" because the former can discover "interaction features." However, the only evidence for this claim comes from a toy example in Appendix B, lacking studies on more real-world datasets .

4. Fig.11 shows that the feature subset selected by the proposed method has low stability in multiple runs, which means that the reliability of this method may be highly dependent on hyperparameter changes. This is unacceptable in fields such as healthcare and finance.

5. While the method proposed in the manuscript performs well on homogeneous datasets such as images and speech, it performs poorly on many so-called "wide datasets" (where the number of features is greater than the number of samples, such as ARCENE and GLA-BRA-180). The paper's claim of "better overall performance" is inconsistent with the data.

### Questions
see weakness.

### Soundness
2

### Presentation
3

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
The paper introduces EntryPrune, a feature selection method for neural networks that uses a dynamically sparse input layer and entry-based pruning. The key idea is to evaluate features based on their initial impact when they first enter the network. The paper reports that EntryPrune outperforms established methods like NeuroFS and LassoNet in terms of accuracy and runtime efficiency.

### Strengths
The introduction of entry-based pruning; measuring the initial impact is reasonable and normalization method used in the techniques ensures fair comparison. Results show some advantage in runtime in long datasets and marginal improvement over the existing techniques.

### Weaknesses
1.	The core contribution of entry-based pruning is incremental at best. The main parts of the technique, gradient based regrowth and pruning largely borrows from prior works like NeuroFS and RigL. The entry-based pruning technique is simply a minor adaptation rather than a truly novel contribution to the field.
2.	The experimental setup with MLP of 1 hidden layer with 100 neurons and large network containing two layers, is too basic and fails to offer a convincing benchmark for current applicability of the method. 
3.	It is unclear how the pruning mechanism work in the case of multi layered networks and whether is it applicable to other than dense layers such as convolutional or residual.
4.	Baselines are too old (the latest is one from 2023). I suggest adding atleast GradEnFS (2024) and more to establish EntryPrune’s place in the current literature landscape. 
5.	The results are, at best, marginal and for the challenging dataset (Cifar-100), NeuroFS is performing better than the proposed consistently. Also, the reported accuracy ranges of these experiments (~40% for Cifar-10 and ~20% for Cifar-100) raises concerns about the practical significance of such techniques. 
6.	There is a significant lack of practical benefit demonstrated. The paper fails to provide clear real-world scenarios where EntryPrune would offer significant advantages over simpler, well-established feature selection methods other than special cases e.g. for interpretability.

### Questions
1.	Can you clearly highlight how EntryPrune differ from the existing works in a more substantial way, beyond the use of early batch scoring ?
2.	Could entry scores after some rotations drift over time? 
3.	The ablation in Figure 8 show sensitivity to hyperparameters. Can you provide a principled approach or a rule of thumb to set the hyperparameters?

### Soundness
2

### Presentation
2

### Contribution
1
