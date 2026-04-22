# GILA: Graph Structure Learning for Class-Imbalanced Tabular Data

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Graph-based learning has demonstrated strong performance across domains by capturing inter-sample dependencies, often surpassing traditional methods focused only on feature-level patterns. However, in tabular data, the relationships between instances are typically implicit or undefined, making it difficult to directly apply graph-based methods. Graph Structure Learning (GSL) addresses this by constructing task-relevant graph structures that explicitly capture the relationships from the data. Nonetheless, a critical yet underexplored challenge in tabular GSL is class imbalance, which is commonly encountered in real-world applications. In this paper, we propose $\textbf{G}$raph Structure Learning for Class-$\textbf{I}$mba$\textbf{l}$anced Dat$\textbf{a}$ ($\textbf{GILA}$) to overcome the difficulties of small-scale and imbalanced tabular datasets.  GILA introduces $\textit{helper nodes}$ to assist the minority class. These helper nodes are optimized using an $\textit{updater}$, and promote class separation among their connected neighbors by influencing their representations through message propagation. Our experiments on 44 imbalanced tabular datasets demonstrate that GILA outperforms existing tabular GSL models. Even under severe class overlap, it still achieves strong performance by leveraging helper nodes that facilitate class discrimination. Furthermore, visualization of the embedding space highlights how helper nodes guide the separation of minority-class samples from majority ones. Code for this submission is provided in an anonymous repository at: https://github.com/anon-gila/GILA

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes GILA, a graph structure learning approach for class-imbalanced tabular data that introduces helper nodes to assist minority class classification. Extensive experimental evaluation is conducted across 44 imbalanced datasets.

### Strengths
1. Extensive experimental evaluation across 44 imbalanced datasets

### Weaknesses
1. Limited Technical Novelty: The core contribution—helper nodes with learnable parameters—is essentially a form of synthetic minority augmentation in the imbalanced classification domain. While the updater mechanism (Eq. 5-6) adds learnable parameters p_i to zero-initialized helper nodes, this is conceptually similar to existing synthetic oversampling techniques (SMOTE). The paper doesn't sufficiently justify why this graph-based approach is superior to simpler alternatives.


2. Incomplete Baseline Comparison: Given that the evaluation focuses on tabular data, the absence of standard tabular ML baselines is an omission. The paper should include the following baselines: Gradient boosting methods (XGBoost, LightGBM); Ensemble methods (Random Forest, AdaBoost). These methods need to be equipped with SMOTE/class weighting for fair comparison. Without these baselines, it's unclear whether the graph structure learning paradigm actually provides benefits over established tabular methods.

3. Insufficient Technical Details: The graph regularization loss L_G (mentioned in Section 3.3) is referenced but not formally defined. Also, the complexity analysis is missing—both time and space complexity should be analyzed, especially given the addition of helper nodes. How does computational complexity scale with the number of helper nodes? Moreover, algorithm or pseudocode for the complete training procedure would help improve clarity.

4. Performance Degradation in Extreme Imbalance: The model completely fails on Abalone19 (IR=129.44), achieving F1=0.000 and ROC-AUC=0.5, which is equivalent to random guessing. This failure needs explanation or discussion.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a Graph Structure Learning (GSL) framework named GILA, designed to address two challenging scenarios: small-scale datasets and class imbalance. The method learns latent inter-sample relationships and introduces auxiliary (helper) nodes to mitigate class imbalance, thereby improving classification performance on imbalanced tabular data. The authors validate their approach on 44 real-world imbalanced tabular datasets and provide visualizations to illustrate the role of auxiliary nodes.

### Strengths
- The paper addresses class imbalance in small-scale tabular data—a problem that has received limited attention in the graph structure learning literature—making it practically relevant.
- Explicitly modeling inter-sample relationships via graph structure learning to enhance few-shot learning capabilities is a reasonable and well-motivated approach.
- Extensive experiments on numerous real-world datasets are conducted, accompanied by visualizations that help explain the behavior of auxiliary nodes in the representation space.

### Weaknesses
1. The core innovation—introducing auxiliary nodes for minority classes and jointly training them to alleviate class imbalance—is highly similar to prior strategies such as GraphSMOTE, which rely on oversampling or virtual node generation. Although Section 4.4 includes comparative experiments, the paper lacks sufficient theoretical or mechanistic novelty to justify a significant technical advance. The current contribution appears intuitive rather than groundbreaking.

2. While the authors claim that GILA can capture underlying data distributions even with scarce samples, small-scale datasets inherently suffer from limited information and high noise sensitivity. Introducing auxiliary nodes may exacerbate noise propagation. The paper does not discuss the sensitivity of auxiliary node initialization or training dynamics, which could critically affect performance in minority-class learning.

3. The graph structure regularization loss \( L_g \) is not clearly defined or explained. Key hyperparameters (e.g., the threshold \( \epsilon \)) are inadequately discussed, making reproduction difficult.

4. The paper does not sufficiently address whether the construction and optimization of auxiliary nodes might introduce noise or lead to structural overfitting. No robustness analyses or ablation studies are provided to evaluate this risk.

5. On datasets such as Glass2 and Abalone19, GILA performs significantly worse than baseline methods. The authors should explain why performance varies across datasets with different scales and imbalance ratios. Moreover, the set of compared baselines is limited, especially lacking comparisons with recent imbalanced graph learning methods (e.g., ImGCL, GraphENS, G2GNN, ImbGNN). Additionally, results are reported only as averages without standard deviations, making it impossible to assess result stability.

6. In extremely imbalanced scenarios, GILA may require a large number of auxiliary nodes to balance class distributions, substantially increasing computational and memory overhead. The authors should provide a complexity analysis or propose a lightweight variant.

7.  The paper only employs GCN as the base classifier. It remains unclear whether GILA is compatible with other GNN architectures (e.g., GAT, GraphSAGE, GIN, or graph transformers) and how such choices affect performance.

8. Ambiguities in Visualization and Class Definition:  
   - In Figure 4.5, during intermediate training stages (subfigures b–d), auxiliary nodes appear equidistant from both minority and majority classes, showing no clear separation. If auxiliary nodes are labeled as minority class, their final embeddings should align more closely with the minority distribution—this discrepancy requires clarification.  
   - For multi-class settings, the paper does not specify how “majority” and “minority” classes are defined. Is the majority class the one with the most samples? Are minority classes defined as only the least frequent class, or as all non-majority classes?  
     - If the latter, there could be many minority classes, raising questions about how many auxiliary nodes to introduce and how to train them.  
     - If only the single rarest class receives auxiliary nodes, how are other low-frequency (but not the rarest) classes effectively learned?

9. The effectiveness of GILA is supported solely by empirical results, with no theoretical analysis (e.g., how auxiliary nodes influence graph smoothness, feature distribution, or decision boundaries). Given the notable performance fluctuations across datasets, the current evidence is insufficient to establish the method’s general validity or robustness in imbalanced learning scenarios.

### Questions
Please refer to the weakness.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces GILA, a novel Graph Structure Learning (GSL) framework designed to handle class-
imbalanced tabular datasets — a scenario common in many real-world applications but largely
overlooked in current GSL research.
The method first constructs a graph from tabular data using IDGL as the structure learner and then
introduces helper nodes, pseudo-nodes that represent minority-class samples. These helper nodes are
learnable through an updater mechanism and influence minority nodes during message propagation to
enhance their separability from majority nodes. The authors conducted extensive experiments over many
datasets, demonstrating substantial gains over existing baselines (MLP, GCNkNN, SUBLIME, HES, IDGL).
Visualizations and correlation analyses further show that GILA is robust to class overlap and promotes
clear minority–majority separation in embedding space.

### Strengths
Introducing helper nodes as trainable pseudo-samples to mitigate imbalance in GSL is conceptually
elegant and new. It moves beyond naive reweighting or oversampling by coupling the mechanism with
graph structure learning. The experiments are comprehensive. The authors not only compares against
many baselines but also investigates robustness to class overlap, the effect of helper node quantity, and
comparisons with traditional imbalance techniques (class weighting, SMOTE). The t-SNE and graph
topology visualizations (Figures 5–6) convincingly show how helper nodes progressively encourage
minority–majority separation.

### Weaknesses
1. GILA largely inherits its graph learning mechanism from IDGL. The novelty, therefore, lies primarily in the
helper-node strategy rather than a new structure learner. This raises questions about generality — would
the same helper-node principle work equally well with other GSL methods (e.g., HES or SUBLIME)?
Besides, there is no exploration about the initialization method of the helper nodes. Whether the full zero
initialization is the optimal way? 
1. Intuitively, the helper nodes are close to the minor-class nodes and they share the same ground-
truth label. Their representation after training should be also close. But this is different from what's
shown in figure 5. Could you give some interpretation?
2. About the effect of increasing helper nodes for the minority class nodes, the performance increasing
trend doesn't saturate as shown in figure 7. Would it continue to increase if adding more helper
nodes to make minority class nodes more than the majority class nodes?
2. Compared to existing papers on Graph Neural Networks for Tabular Data Learning, it seems the author only introduce
the helper node to solve the imbalanced problem. The method is naive.

### Questions
see above.

### Soundness
2

### Presentation
3

### Contribution
2
