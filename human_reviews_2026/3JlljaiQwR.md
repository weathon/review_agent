# Mini-cluster Guided Long-tailed Deep Clustering

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
As an important branch of unsupervised learning, deep clustering has seen substantial progress in recent years. However, the majority of current deep clustering methods operate under the assumption of balanced or near-balanced cluster distributions. This assumption contradicts the common long-tailed class distributions in real-world data, leading to severe performance degradation in deep clustering. Although many long-tailed learning methods have been proposed, these approaches typically rely on label information to differentiate treatment across different classes, which renders them inapplicable to deep clustering scenarios. How to re-weight the training of deep clustering models in an unsupervised setting remains an open challenge. To address this, we propose a mini-cluster guided long-tailed deep clustering method, termed MiniClustering. We introduce a specialized clustering head that divide data into much more clusters than the target number of clusters. These predicted clusters are referred to as mini-clusters. The mini-cluster-level predictions serve as the guide for estimating the appropriate weights for classes with varying degrees of long-tailedness. The weights are then incorporated to re-weight the self-training loss in model training. In this way, we can mitigate model bias by re-weighting gradients from different classes. We evaluate our method on multiple benchmark datasets with different imbalance ratios to demonstrate its effectiveness. Further, our method can be readily applied to the downstream of existing unsupervised representation learning frameworks for long-tailed deep clustering. It can also adapt label-dependent long-tailed learning methods to unsupervised clustering tasks by leveraging the estimated weights. The code is available at https://github.com/LZX-001/MiniClustering.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel method to cluster longt-tailed datasets where the class distributions are imbalanced. The proposed method, called MiniClustering, enables class re-weighting in unsupervised learning(clustering). The network uses two heads 
* A target-cluster head ($f_t$) that predicts the desired number of clusters ($K$).
* A mini-cluster head ($f_m$) that predicts a much larger number of clusters ($M$).
By analyzing how many mini-clusters are associated with each target-cluster, the method estimates which target-clusters represent "head" vs. "tail" classes. It then calculates a weight for each target-cluster: clusters associated with many mini-clusters (head classes) get a smaller weight, and those with few (tail classes) get a larger weight. This weight is used in a re-weighted self-training loss ($L_r$) for the target-cluster head. The full model is trained with this loss, a self-training loss for the mini-cluster head ($L_m$), and a similarity alignment loss ($L_s$) to keep the two heads synchronized.

### Strengths
1. The papers primary contribution is a functional mechanism to perform class balancing in a unsupervised setting. Which is mostly limited to supervised learning until now.
2. The method is rather flexible in terms of representation learning framwork. The authors demonstrate this with BYOL,SimCLR and MoCO
3. The authors provide a comprehensive ablation study to justify each component of the method and providing some insides how these components influence the clustering quality

### Weaknesses
1. Hyperparameter Sensitivity, Minicluster introduces 4 new hyperparametes that lack a unsupervised tuning strategy:
* Number of mini-clusters
* threshold for assigning a mini-cluster to a target-cluster
* $\alpha$ and $\beta$ as trade-off weights for the losses. This parameters are quite significant as shown in figure 5.
2. The MaxiClustering variant proposed in the Appendix feels a post patch that contradicts some of the core logic of the method.
* MiniClustering ($M > K$) is based on the premise that head classes spread across many mini-clusters.
* MaxiClustering ($M < K$) is based on the inverted premise that tail classes spread across many maxi-clusters.
MaxiClustering inverts the re-weighting logic, hypothesizing that tail classes are associated with more maxi-clusters. This seems to be a direct contradiction of the "phenomenon" that motivated the entire paper.
3. Implicit Assumption of K. This is a common limitation/weakness of most SoTA e.g. Secu and SCAN (ref. in the paper) but a weakness nonetheless

### Questions
1. How can the hyperparameters be tuned in a real-world, fully unsupervised scenario?
2. How robust is the core assumption (Section 3, Phenomenon 3) across different types of data and encoders? 
3. Also related to 2., since there is a MiniClustering and MaxiClustering. What is the theoretical basis for this inversion, and does it imply the Phenomenon 3 is just a heuristic dependent on the $M/K$ ratio?
4. How much of the performance gain comes from the pre-training versus the re-weighting? How much would the baseline improve if it would continue the 200 epochs of clustering?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the long-tailed distributions in deep clustering. Since existing deep clustering methods assume class-balanced data, they struggle with real-world imbalanced scenarios. The authors propose MiniClustering that introduces an auxiliary mini-cluster head to generate finer-grained clusters than the final target number. These mini-clusters are used to estimate class-specific weights in an unsupervised manner, enabling re-weighted training analogous to supervised long-tailed learning. The framework includes three loss components: mini-cluster self-training loss, re-weighted target-cluster loss, and a similarity alignment loss to prevent desynchronization between clustering heads. Extensive experiments on CIFAR-10, CIFAR-20, STL-10, Tiny ImageNet, and ImageNet-LT, across varying imbalance ratios, show that MiniClustering consistently outperforms state-of-the-art baselines on multiple metrics.

### Strengths
1. The dual-head design is intuitive. It allows the model to approximate class-wise importance without needing labels, which is both elegant and practical in clustering scenarios.
2. The experimental results are strong and well-supported. Visualizations and benchmarks clearly show the method works better than existing ones across a variety of datasets and imbalance levels.

### Weaknesses
1. The main novelty of the paper lies in identifying the relationship between mini-clusters and target clusters, but these observations do not appear entirely original. The notion of mini-clusters has existed for years, with similar ideas explored in works like IIC and PICA. More specifically:

   a. Phenomena 1 and 3 would benefit from more conclusive, clearly stated takeaways rather than descriptive observations.

   b. Phenomenon 2 (“Mini-clusters can enhance purity”) is not very convincing, since purity naturally increases when clusters are subdivided. In the extreme case of infinite subdivisions, purity can trivially reach 1, which does not truly reflect clustering quality.

   c. Phenomenon 3 does not directly correspond to estimating class probabilities. The analysis is limited to comparing the relative proportions of head versus tail classes, without demonstrating whether this distinction can reliably separate them.

2. The results on balanced datasets are missing. It would be informative to see comparisons under standard settings—for example, relative to the results in Table 2 of ProPos. While this does not affect the core focus on long-tailed distributions, it would help situate the method’s overall competitiveness.
3. The similarity alignment loss resembles a second-order alignment loss introduced in *Graph Matching with Bi-level Noisy Correspondence*. A proper citation and discussion of this connection would strengthen the methodological grounding.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To address the problems of current deep clustering, where multiple assumptions about data distribution are unbalanced and existing long-tail learning methods rely on labels and do not apply to unsupervised scenarios, this paper proposes MiniClustering (a long-tail deep clustering method guided by mini-clustering). By introducing a dual clustering head (a target clustering head + a mini-clustering head, where the former outputs a prediction of the number of target clusters and the latter outputs a more fine-grained mini-cluster prediction), unsupervised category weight estimation is achieved based on the core phenomenon that "head class samples are associated with more mini-clusters and tail classes with fewer mini-clusters". Finally, its effectiveness is verified across multiple datasets, including CIFAR-10/20 and STL-10.

### Strengths
1. The paper proposes an architecture of "target clustering head + mini clustering head" that estimates weights through the collaboration of clustering heads with different prediction granularities. This differs from the traditional single clustering head design and provides a new technical path for unsupervised reweighting.
2. The algorithm still demonstrated reliable performance even in scenarios with extreme imbalances and large-scale data.

### Weaknesses
1. The proposed MiniClustering relies on several key hyperparameters that need to be manually tuned across different datasets and imbalance ratios. It lacks automated adaptation capabilities, which increases the threshold for practical application.
2. All experiments were based on image data and did not involve non-image modalities such as text, audio, and video. Due to significant differences in data distribution and embedding characteristics across modalities, it is impossible to determine whether the "mini-clustering mechanism" of MiniClustering applies to non-image scenarios.

### Questions
1. Are there any rules for designing the number of mini-clustering heads, and what impact does it have on the stability of the model?
2. How does batch size affect model performance? In particular, how does the model perform when the batch size is smaller than the total number of classes?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a novel method for unsupervised long-tailed deep clustering to address the challenges posed by imbalanced data distributions. The proposed method attempts to mitigate the bias towards majority classes by introducing an auxiliary "mini-clustering" head which over-clusters the data (into many more clusters than the target number) to estimate class-specific weights for a re-weighted self-training loss.

### Strengths
- The idea of estimating imbalance through over-clustering (mini-clusters) is creative and conceptually plausible.
- The loss formulation is clearly described and easy to follow.
- Comprehensive comparisons across multiple datasets and imbalance ratios, with also ablation studies to validate design choices. The addition of the maxi-clustering variant is also appreciated.
- The paper is generally well-written and easy to read.

### Weaknesses
- While the method is motivated by empirical observations (Figures 1–3), there is no theoretical justification connecting the number of mini-clusters a target cluster occupies to its degree of long-tailedness. The proposed weight computation (Eq. 2) is purely heuristic, and the paper offers no probabilistic or geometric reasoning explaining why these weights should correlate with true class frequencies.
- The method’s weight estimation depends on pseudo-labels from two heads — both trained simultaneously. Early in training, these pseudo-labels are noisy, which could cause unstable or biased weight updates. No mitigation (e.g., confidence annealing, temporal ensembling, or delayed weighting) is described, and the paper does not analyze the effect of pseudo-label noise on convergence.
- The method relies on a good quality of the mini-clustering head to estimate class frequencies, but there is no analysis of how sensitive the method is to errors in this estimation.
- The experimental results, while comprehensive, lack statistical significance analysis. It is unclear whether the observed improvements over baselines are statistically significant or could be due to random variation.
- Figure 2 and 3 are a bit too small to read comfortably.

### Questions
- Can you provide any formal argument or empirical correlation analysis linking the number of mini-clusters assigned to a target-cluster and the true class frequency?
- If some minority classes are not well separated in the mini-clustering, how does this affect the final performance?
- Could you provide standard deviations or confidence intervals for the results in Tables 1 and 2 to assess statistical significance? 
- Could you provide some visualizations of the learned embeddings (e.g., t-SNE or UMAP plots) to illustrate how well the method separates classes, especially minority ones?

### Soundness
2

### Presentation
3

### Contribution
3
