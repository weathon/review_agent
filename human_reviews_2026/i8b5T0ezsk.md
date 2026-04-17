# ProFit: Unsupervised Fine-Tuning of Tabular Models via  Proxy Tasks for Label-Scarce Anomaly Detection

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Anomaly detection in tabular data is crucial for applications such as fraud prevention and risk control, yet it remains challenging due to heterogeneous features, class imbalance, and limited labeled anomalies. Although pretrained tabular in context learning (TICL) models reduce label dependence, the inductive biases they develop on synthetic tasks are often misaligned with the actual data distributions encountered in downstream scenarios. Effective adaptation to new domains is thus difficult when labels are scarce. We propose **ProFiT**, an unsupervised fine-tuning framework that leverages only unlabeled target-domain data to adjust pretrained tabular models. ProFiT constructs a variety of proxy tasks by sampling different features as targets and using correlated features as inputs, encouraging the model to capture the underlying structure of the new data. To improve training effectiveness, we introduce a consistency regularizer that aligns the predictions from two different proxy views using Jensen–Shannon divergence. Experiments on tabular anomaly detection benchmarks show that ProFiT outperforms weakly-supervised and unsupervised methods, as well as vanilla TICL models. ProFiT offers a practical way to improve tabular anomaly detection under limited labeled data conditions and vast amounts of unlabeled data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel approach for unsupervised fine-tuning of existing Tabular In-Context Learning Models for anomaly detection. The proposed method focuses on the weakly supervised setting where one disposes of a small subset of anomalies and a large, unlabeled dataset.
In short, the proposed approach, ProFIT, relies on proxy tasks to refine the prediction ability of the MotherNet model. For a $d$-dimensional feature space, a target feature $t \in 1,...,d$ is selected and serves as the proxy-task's label. Among the $d$-1 remaining features, the authors propose a feature selection method that identifies the most relevant subset of features that will serve to predict the target feature.

A double-objective is used to fine-tune the TICL model based on those proxy tasks: (i) a standard cross-entropy loss on the selected target feature $t$ and (ii) an alignment loss that ensures that the logits on the same target feature $t$ produced by two sets of features $S_1$ and $S_2$ are close to each other.

The experimental results on a significant benchmark of tabular datasets demonstrate the relevance of the proposed approach, as it shows strong performance based on several threshold-dependent and non-threshold-dependent metrics.

### Strengths
S1. The paper is **well written** and easy to follow.

S2. **Experiments are sound and extensive**. The dataset benchmark is quite large, and the methods to which ProFIT is compared are quite numerous.

S3. **Ablation** studies are limited but **relevant**.

S4. ProFIT is **theoretically based** and well-founded.

### Weaknesses
**W1**. While minor, there are a few typos in the paper that are worth mentioning: 1) Line 103, Tabular In Context **Learing** Model.; 2) Line 201, $ \mathbf{X}_i = g_j(\mathbf{u}, \epsilon_i) $, where does this $j$ come from? 3) On Tables 11-16 MotherNet is referred to as MotehrNet.

**W2**. As mentioned in S1, the paper is well-written; however, there is **key information missing from the main text of the paper**.
- 1) Unless I am wrong, nowhere is it mentioned which model is fine-tuned in the experiments that are presented in the paper. One must refer to the appendix to find this crucial information. 
- 2)  The authors mention how they are interested in the weakly supervised anomaly detection set-up, and even define on line 132 $D_L$ (which is never reused before page 17 of the appendix), but never explain how those labeled anomalies are used.
- 3) Overall, section 3.2. lacks crucial details on the methods.

**W3**. The proposed pretext task is a special case of mask reconstruction.

**W4**. This work is not currently reproducible, although I acknowledge that the authors plan to release the code upon acceptance.

### Questions
**Q1**. As mentioned in **W1**, the paper may require some clarifications to make the setup easier to understand. Overall, the authors should include Appendix B in the main text of the paper, replacing less crucial sections, such as Section 3.5. 
Regarding the overall method, I have a few questions: 
- (i) How are the labeled data samples used in this context? Only as in-context samples for the TICL model? If so, it might be worth mentioning it somewhere in the main part of the paper. 
- (ii) Figure 1 suggests that there might be overlapping features between the two subsets used to predict the same selected target features. However, Algorithm 1 suggests otherwise. Could the authors explain which approach they used? 

**Q2** It appears to me that ProFIT should be better suited for datasets with a medium to large number of features. In particular, the performance on datasets with very few features, e.g., http (3) or Skin (3), could be attributed to the sole performance of the MotherNet model. This claim is supported by Tables 11 to 16, where for the Skin dataset, there is no difference in performance between the MotherNet model and the fine-tuned one using your approach. I recommend discussing this in section 5, investigating performance variations on datasets like http, Skin, Smtp, or Wilt. Additionally, including the non-fine-tuned MotherNet model in the main tables and comparison would further underscore the relevance of your method.

**Q3**: Could you elaborate on the difference between the proposed approach and vanilla mask reconstruction? It seems to me that it consists in masking $l$ feature among the original $d$ features and predicting only one of them. Why should it work better than predicting the entire set of $l$ features? It appears, as mentioned in **W4**, as a special case of mask reconstruction.

**Q4**: (More open question) Why do the authors restrain their approach to unsupervised fine-tuning of TICL models for anomaly detection, as it seems that their approach could be applied to a much broader self-supervised setup (for representation learning, for example)?


Overall, I lean towards acceptance, and I am open to increasing my score should my questions and concerns be addressed.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper introduced ProFiT, an unsupervised fine tuning framework that adapts pretrained tabular in context learning models to anomaly detection when labels are scarce by training on automatically constructed proxy tasks. By predicting a held out feature from mRMR selected and correlated feature subsets and enforcing cross subset consistency with a Jensen–Shannon divergence regularizer, ProFiT learns  representations that align with target domain structure without additional anomaly labels. Experiments on tabular anomaly detection benchmarks show that ProFiT outperforms weakly-supervised and unsupervised methods, as well as vanilla TICL models. ProFiT offers a practical way to improve tabular anomaly detection under limited labeled data conditions and vast amounts of unlabeled data.

### Strengths
1. Minimal Label Dependency for Real-World Applicability
ProFiT completes fine-tuning using only unlabeled target-domain data, fully addressing the core pain point of anomaly detection—“difficulty in obtaining labels (few anomalous samples and high annotation costs)”. Compared to weakly supervised methods (requiring a small number of labels) and traditional supervised methods (relying on large-scale labels), it has stronger adaptability to practical scenarios.
2. Resolution of “Distribution Misalignment” in Pretrained Models
Aiming at the key flaw of existing Tabular In-Context Learning (TICL) models—inductive biases formed on synthetic tasks are misaligned with the data distribution of real target domains—ProFiT enables the model to relearn the underlying structure of target data through “proxy tasks based on target-domain features”, significantly improving domain adaptation capability.
3. Consistency Regularization Enhances Model Robustness
ProFiT introduces a regularization term based on Jensen–Shannon divergence to force alignment of prediction results from two different proxy views. This design reduces biases from single proxy tasks, prevents the model from overfitting to local features, makes the learned features more general, and ultimately improves the stability of anomaly detection.

### Weaknesses
1. Strong Dependence on Inter-Feature Correlations in Data
The core logic of proxy tasks is “sampling some features as targets and using correlated features as inputs”. If the target-domain data has weak inter-feature correlations (e.g., multi-dimensional independent features) or a large number of redundant features, proxy tasks cannot be effectively constructed. As a result, the model fails to capture data structure, leading to ineffective fine-tuning.
2. Unclear Generalization to “Unknown Anomaly Types”
The paper only verifies performance on benchmark datasets, but in real scenarios, anomaly types are often “unknown (Out-of-Distribution)”. ProFiT detects anomalies by learning the structure of normal data; if unknown anomalies have little difference from the structure of normal data, its detection capability may decline significantly. This aspect is not verified in the abstract.
3. Higher Computational Cost Than Vanilla Pretrained Models
The proposed ProFiT needs to construct “multiple proxy tasks” and align predictions from two proxy views for consistency. Compared to directly using vanilla TICL models, the training process requires more computing resources and time—especially for high-dimensional tabular data, the cost increase may be more significant.

### Questions
I have two questions that I am particularly concerned about regarding this paper. 1. A limited discussion on the applicable boundaries of feature selection based on mRMR？The abstract mentions that ProFiT builds the agent task by predicting the reserved features of the relevant feature subset selected by mRMR, but it does not discuss the applicability boundary of the mRMR method itself. In practical scenarios, if the target table data has extreme features, such as extremely weak inter-feature correlations, high-dimensional data with a large number of sparse features, or nonlinear feature dependencies that are difficult to capture by mRMR (a method focused on linear/statistical correlations), the quality of the selected feature subset will decline. This directly affects the effectiveness of the proxy task and the model's ability to learn the target domain structure. However, this paper fails to address how to handle such edge cases or verify the stability of mRMR across different data types.
2. Lack of adaptation to dynamic or multi-source table scenarios. One key limitation of the current ProFiT framework is that it cannot handle dynamic or multi-source table data. In real-world anomaly detection (for example, real-time fraud prevention, dynamic risk control), data often exhibits concept drift (data distribution over time) or comes from multiple related tables (for example, the combination of user transaction data and user configuration data). At present, ProFiT, which is designed for static single-table data, cannot adapt to these common scenarios, greatly limiting its practical application scope in complex industrial environments.

### Soundness
3

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
This paper focuses on the task of weak-supervised tabular anomaly detection. The proposed method ProFit, adapts tabular foundation model to downstream anomaly detection tasks via proxy-based fine-tuning.

### Strengths
S1: Adapting tabular foundation model to downstream tasks is important.
S2: The paper structure is clear.

### Weaknesses
Reproducibility perspective:
W1: The source code is not provided.

Problem setting perspective:
W2: The current setting frames Tabular Anomaly Detection (TAD) as a supervised learning problem with sparse positive labels. However, it remains unclear why only the unlabeled samples predicted as normal by the detector are used during inference. What if the samples predicted as anomalous are also incorporated? Utilizing the full set of unlabeled data, including those flagged as anomalies, could potentially improve model robustness and performance.

W3: Given that the problem is formulated as binary classification, several powerful tabular models—such as XGBoost, CatBoost—could serve as strong baselines. These methods are well-established in tabular data benchmarks and should be included for a comprehensive comparison.

W4: Based on the above, what if the labeled true anomalies is removed? It seems the proposed framework can also be adapted to unsupervised setting by relying solely on pseudo-labels generated by an anomaly detector. 

W5: Minor. Although the paper focuses on weakly-supervised learning, semi-supervised methods (MCM[1], NPT-AD[2], DRL[3])—which are widely used in TAD and assume only normal samples are available during training—should also be included as baselines. Many such methods have been evaluated under similar "contamination" settings.
[1] MCM: Masked Cell Modeling for Anomaly Detection in Tabular Data. ICLR’24.
[2] Beyond Individual Input for Deep Anomaly Detection on Tabular Data. ICML’24.
[3] DRL: Decomposed Representation Learning for Tabular Anomaly Detection. ICLR’25.

Technical perspective:
W6: Towards the confused contribution of foundation model and the proposed meta learning mechanism. Both tabular foundation model and this method do the similar thing, which first pretrain on diverse tasks, and then transfer to previously unseen downstream supervised tasks at inference (this paper will transfer weakly-supervised anomaly detection to supervised learning as shown in Line 910 of manuscript). It is known that foundation model like TabPFNv2 trained on synthetic data performs well on real-world tabular data, including low-data and imbalanced regimes. Why then do these foundation models not suffice for this setting? If the problem is reformulated as binary classification (as in Line 910), could fine-tuning TabPFNv2 directly (with supervised loss, not with proxy-based fine-tuning) yield competitive results? If not, this would indicate a fundamental gap in the transferability of foundation model knowledge to this specific task. Moreover, it seems that theoretical framework of proxy-based fine-tuning can train a model such as Transformer from scratch on a single dataset. Thus what is the explicit benefit of using a pretrained foundation model over training from scratch on the target dataset? It is insufficient to only compare whether proxy-based fine-tuning is used on MotherNet as illustrated in Fig.3. Both more detailed explanation and ablation results (more types of variants and other foundation models like TabPFNv2 as noted above) should be included to validate the effectiveness of the proposed method.

W7: The uniqueness of the method in the context of anomaly detection is not clearly established.  It seems that this method can work on any imbalance tabular data setting, what is the unique relationship with tabular anomaly detection? What is the performance of this method on other tabular imbalance learning setting?

W8: Towards the fragile of assumption. The theoretical analysis relies on several assumptions (e.g., latent factor model, task compatibility). These are strong assumptions that columns are independent conditioned on latent factor. How are these assumptions validated, especially under complex real-world data distributions? 

W9: Towards the confused training objective. This method selects one feature column as target to prediction via minimizing the cross-entropy loss. However, tabular data including many numerical columns. If the selected column is continuous, cross-entropy may not be appropriate. How does the method handle continuous features?

W10: The meta learning mechanism, that selecting various columns as labels and the remainder as features is not novel, which is used by previous papers in tabular domain, e.g., GTL[4], STUNT[5].

[4] From Supervised to Generative: A Novel Paradigm for Tabular Deep Learning with Large Language Models. KDD’24.
[5] STUNT: FEW-SHOT TABULAR LEARNING WITH SELF-GENERATED TASKS FROM UNLABELED TABLES. ICLR’23.

W11: The choice of foundation model and anomaly detector is not thoroughly justified. Why was MotherNet selected over other tabular foundation models (e.g., TabPFNv2, TabPFN)? Ablation studies with alternative models would help validate the generality of the approach. Similarly, why was iForest chosen as the pseudo-labeler? Would the results hold with other detectors? 

W12: The number of labeled anomalies. For datasets with very few anomalies (e.g., WBC, WDBC, Wine), using 5 anomalous samples may already represent half of all available anomalies. Is this a fair comparison with other detection methods?

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
2

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
ProFiT is an effective unsupervised fine-tuning framework for tabular anomaly detection under limited labels, leveraging proxy tasks to capture feature relationships. The method is theoretically grounded and empirically strong, though its reliance on certain assumptions and hyperparameter sensitivity warrants further clarification.

### Strengths
This paper tackles an important challenge in anomaly detection under label scarcity. The proposed method, ProFiT, uses unlabeled data to create proxy tasks by predicting a held-out feature from correlated subsets. The feature selection strategy (based on correlation and redundancy) is well-motivated, and the addition of a JS-consistency loss across proxy subsets helps regularize training.
The theoretical justification is thoughtful. By linking proxy-task performance to downstream generalization through a latent factor model, the authors provide a clear intuition for why their method should work.
Empirical results are strong. Across 35 tabular datasets, ProFiT consistently outperforms both classical and recent weakly-supervised baselines. Gains in AUCPR and F1 are meaningful, especially with as few as 5 labeled anomalies.
Presentation-wise, the paper is cleanly written and structured. Figures and appendices help clarify the method and setup. The approach is practical and applicable in settings like fraud or risk monitoring, where labeled anomalies are scarce.

### Weaknesses
The theory relies on a strong latent factor assumption. While it helps the analysis, it's unclear how realistic this is for complex tabular data. A short discussion would help.
ProFiT assumes access to a pre-trained tabular model (e.g., MotherNet). It’s unclear how well the method works without it, or whether it’s portable to other backbones.
Hyperparameters like the subset size k or consistency loss weight λ aren’t thoroughly explored. Ablation or guidance would improve reproducibility.
The final inference step uses iForest to mine pseudo-normals. While practical, it introduces an unsupervised heuristic that could fail in noisy settings. Some robustness analysis would help.
The paper does not compare to simpler alternatives like masked prediction which could serve as baseline proxy-task learners.

### Questions
1. What happens if the latent factor assumption doesn't hold? Do you observe degraded performance in such settings?

2. Did you test different anomaly detectors besides iForest for support set selection? Does performance vary?

3. Can your method operate without any labeled anomalies (i.e., K=0)? Could proxy-task variance or consistency be used to rank anomalies?

4. What were your actual choices for k and M, and how sensitive is performance to them?

5. How does the base TICL model (e.g., MotherNet) perform without ProFiT? Including this would help isolate the fine-tuning gain.

### Soundness
3

### Presentation
4

### Contribution
3
