# PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 0

## Abstract
Dynamic node classification is critical for modeling evolving systems like financial transactions and academic collaborations.
In such systems, dynamically capturing node information changes is critical for dynamic node classification, which usually requires all labels at every timestamp. However, it is difficult to collect all dynamic labels in real-world scenarios due to high annotation costs and label uncertainty (e.g., ambiguous or delayed labels in fraud detection). In contrast, final timestamp labels are easier to obtain as they rely on complete temporal patterns and are usually maintained as a unique label for each user in many open platforms, without tracking the history data. To bridge this gap, we propose a pioneering method PTCL (Pseudo-label Temporal Curriculum Learning), combining the variational EM (Expectation Maximization) framework with a novel Temporal Curriculum Learning strategy to effectively leverage both final timestamp labels and pseudo-labels. We also contribute a new academic dataset CoOAG (Co-authorship graph derived from Open Academic Graph), capturing long-range research interest in dynamic graph. Experiments across real-world scenarios demonstrate PTCL’s consistent superiority over other methods adapted to this task. Beyond methodology, we propose a unified framework FLiD (Framework for Label-Limited Dynamic Node Classification), consisting of a complete preparation workflow, training pipeline, and evaluation standards, and supporting various models and datasets. Code details can be found in supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work, the authors focus on dynamic node classification task in temporal graph under the limited label setting. The paper proposed PTCL, an extensible method with temporally-weighted pseudo-labels and a variational EM framework. The authors also included the CoOAG dataset, a novel benchmark for this task. 
Across four datasets, the authors showed that PTCL improves various base TGNN's performance. The unified FLiD framework is also a nice contribution for future work.

### Strengths
There has long been a lack of study for node level task in temporal graph learning. This work help bridge this gap by focusing on the dynamic node classification task. specifically it has the following strength:

- **significance: problem that needed attention**. As mentioned, the dynamic node classification problem has been under-studied in the literature and this work focus on this task which will be beneficial for the community. 

- **extensive evaluation**. Generally there is a lack of node classification dataset in temporal graph due to lack of dynamic labels (as the authors mentioned). In this work, the authors evaluated on four datasets with a range of temporal graph methods across different baseline training approaches. The original dataset is also a nice contribution. 

- **clear presentation**. The paper is clearly presented and easy to follow

### Weaknesses
- **computational complexity**: as datasets in this work are on the smaller side (less than a million edges). It would be good for the authors to show the complexity of the approach and the compute time. If there is larger labeled datasets, it would be good to benchmark on it as well. 

- **Figure 3 is not clear**: figure 3 visualisation is not clear and end up being more confusing than just reading the description in Section 4.1.2. For example, what does the red box mean? Also B and D to represent backbone and decoder is confusing at a glance. Please update the figure or remove it. 

- **lack of detail for FLiD in the main paper**. Potentially adding a diagram to make FLiD more clear and how it helps facilitate the process.

### Questions
- **Dsub** why take a subset of the **Dgraph** dataset?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper identifies a key challenge of node classification in dynamic graph learning, where only the final-time labels are collected and the intermediate labels are missing. To address this challenge, the authors propose a framework that dynamically generates pseudo labels for intermediate time steps and progressively refines them during training. The proposed method is evaluated on several benchmark datasets, demonstrating its effectiveness compared to existing baselines.

### Strengths
1. The paper identifies a key challenge of node classification in dynamic graph learning, where only the final-time labels are collected and the intermediate labels are missing. Such a challenge is common in real-world applications but has been largely overlooked by existing works.
2. The paper proposes a framework that dynamically generates pseudo labels for intermediate time steps and progressively refines them during training. The proposed method is well-motivated.

### Weaknesses
1. The cited related works are not recent enough.
2. Although the paper made efforts in clarifying its theoretical foundation, it is still hard to follow.
3. Please avoid leaving so many dangling lines that have only one word in the last line of a paragraph.

### Questions
1. Why is using the true dynamic labels perform worse than the pesudo labels? The true labels should be the upper-bound performance.
2. Since ELI is nominated as a competing work, why there are no comparisons with it? 
3. As shown in Table 1, the baseline that uses true dynamic labels (DLS) does not have significant improvements over the baselines that use pseudo labels (CFT and NPL). To be honest, it is worse in many cases. Then what is the necessity of recovering all those labels? It seems that the curriculum learning strategy is the key to the performance boost, then the contribution is largely reduced.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a variational EM-based approach for label-limited dynamic node classification. Specifically, the authors treat each node’s label at the final timestamp as an observable variable, while the labels at historical interaction times are modeled as latent variables. They then employ an EM-based framework to learn the node classification model. In addition, the authors weight the pseudo labels generated from the variational posterior distribution based on the temporal gap between the final timestamp and the interaction time, aiming to mitigate the impact of unreliable pseudo labels.

### Strengths
- The problem of label-limited dynamic node classification is both practical and novel. Despite its relevance to real-world scenarios, it has received limited attention in prior work.
- Generally, the idea of adopting EM to solve this problem makes sense.

### Weaknesses
- The description of the method is ambiguous, which makes it difficult to evaluate. For example, when predicting node labels, the adopted dynamic graph model does not seem to consider the historical labels of nodes. It remains unclear how $p_{\theta}(y_u^t|G,Y_{F,B})$ is actually computed and how the model incorporates the historical labels $Y_{F,B}$.
- In the E-step, the variational loss is not actually utilized. The authors simply train the model using ground-truth labels, which breaks the connection with the EM framework.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The authors of this work address a dynamic node classification task in temporal graphs, with the additional assumption that there are limited node labels available during training. The proposed approach builds on semi-supervised learning with pseudo-labeling, where the model is trained using data on a final time stamp to predict pseudo-labels for earlier timestamps. A curriculum learning approach is proposed, which progressively trains the model using (pseudo-)labels from later to earlier timestamps and which can use different GNN backbone learning techniques to generate node embeddings. 

The proposed approach is evaluated in four different data sets, addressing binary node classification in data from two online platforms (binary target is to predict blocked users) and one financial social network (binary target is to predict users who do not pay back loans), as well as research areas in a scientific collaboration and citation data set. The proposed method shows mild performance improvements compared to baseline training techniques across all backbone GNN architectures. Further evaluation in an ablation study suggest a rather small contribution of the pseudo-labels and curriculum learning approach.

### Strengths
[S1] The paper addresses node classification in continuous-time temporal graphs (CTTG), which is an important and current topic in graph learning.

### Weaknesses
[W1] I could not follow the motivation for the very specific learning task considered in the paper, which assumes that node labels are available for a final but not for earlier timestamps. Moreover, some of the data sets used to motivate this do not seem to fit this setting, and I would argue that the problem should not be addressed from a dynamic node classification problem in the first place. See my questions Q1 below. Clarifying this is crucial, as the authors argue that a key contribution of their work is that they are the first to study this problem systematically.

[W2] Some key information is missing in the main manuscript (e.g. what are the actual labels in some of the data sets). Discussing the semantics of this in the main text is crucial to judge whether the data fits the proposed setting, see detailed comments in Q7 and Q8 below.

[W3] In found that some parts of the methdology, e.g. the application of the Variational Expectation Maximixation, are hard to follow and lack intuition. Also notation could be improved to make the method easier to understand, see my questions Q2, Q3, and Q6 and suggestions below. 

[W4] While the proposed method shows a moderate improvement of average accuracy, the increase is small and even maximally simple baselines that essentially assume that labels are static (cf. W1) perform well. For some results, standard deviations are missing, which makes it impossible to judge whether the small observed differences are actually significant. See Q4 and Q5 below, 

[W5] The contextualization of the method in terms of related work is very weak and important published works addressing the same problem even in the same data are neither mentioned nor used as a baseline, see Q9 below.

### Questions
[Q1] I could not follow why the Open Academic Graph data set motivates your setting of a dynamic graph, where "final static" node labels are available. First of all, I would argue that there are either static labels (that do not change over time) or final labels (indicating that they change but we only know their final values). More specifically, for the Open Academic Graph data set, research interest labels are rather aggregate labels, which result from a topic modelling method applied to all publications and venues of a researcher. I thus think that this does not fit the setting of the paper. Similarly, for the fraud setting, one could argue that the fraud label is rather an aggregate rather than a final label. Could the authors clarify why these settings correspond to a dynamic node classification problem (where labels are absent for earlier timestamps) instead of a simpler task where static classes are assigned to node based on a time series? 

[Q2] I could not follow the description of the Expectation-Maximization approach. First of all, it is unclear to me what enters in the conditional probability. The notation seems to suggest that Y_{F, B} is only conditional on the dynamic graph G, which does not include any node labels. Is that correct? I think that section 3.1 should be explained better and I also believe that a better notation could help to make it easier to understand the approach. 

[Q3] On page 4, the authors mention that, while there is a hyperparameter $\alpha$ that balances the influence of pseudo-labels and final labels in the objective function of the decoder, in practice a value of alpha works best, which actually suggests that pseudo-labels are unncessary (and which seems to support my criticism of the actual task relevance in Q1). Did I get this wrong? Are pseudo-labels only used in the maximization step? Could the authors clarify this? 

[Q4] While the proposed approach shows the best average performance in most of the data sets, a few aspects of the results should be noted. First, fitting my earlier comments about the relevance of the task, two baselines which simply copy the final labels to all previous timestamps or only use available labels already perform well. Second, for all of the results, improvements are within the standard deviation intervals of other methods, which challenges the authors statement that their method significantly outperforms other methods.

[Q5] Related to my question Q3, the analysis in section 4.3 actually shows that the influence of pseudo-labels is - at best - small. Also, standard deviations are missing in the results, so it is impossible to judge whether the differences between the approach with and without pseudo-labels are actually significant. The same is true for table 3, which evaluates the contribution of the curriculum learning approach. Please add standard deviations 

[Q6] Could the authors give an intuition what they want to capture with the consistency measure in equations 11 and 12? 

[Q7] In the main text, I could not find information on what (binary) classification task is actually addresses for the Wikipedia, Reddit and Dsub data in the experiment evaluation. Additional information on page 18 of the appendix suggests that the binary labels refer to fraudulent behavior (DSub) or banned users (Wikipedia, Reddit). Please clarify this in the main text and include information on the label distribution and label dynamics. 

[Q8] Regarding the fraud label for the DSub data set, I would first argue that the fact that users could not repay their loan is a default rather than a fraud, so please check your terminology. I think this semantics is crucial, as it also affects the classification task. In particular, I would argue that treating the question whether a credit customer will eventually default is not a reasonable setting for a dynamic node label prediction, as it is unlikely that a user will default multiple times. So, referring to Q1, to me this again seems to rather correspond to a simple (static) node classification problem in a dynamic graph. 

[Q9] In the (overly short) related work section on dynamic node classification, the authors state that dynamic node classification remains underexplored, which is hardly the case. Many works in temporal graph learning have addressed different variants of the problem, e.g. assuming static node labels but using the temporal graph to classify them or using temporal embeddings to predict dynamic node labels. This is true for GNN-based methods like TGN, TGAT, DBGNN, TGBASE, DyREP, etc. 

It is also not true that prior methods assume access to full dynamic labels, as some of these methods have considered semi-supervised or few-shot setting. In particular, fitting the setting considered by the authors and even building on a pseudo-label based training, the following method has been published two years ago: 

S Tian et al.: SAD: Semi-Supervised Anomaly Detection on Dynamic Graphs, IJCAI 2023

The fact that this paper is not referenced and that the approach proposed in there is not considered as a baseline is a major omission.

Further suggestions:

- As a general rule, please define all abbreviations upon first use (EM for expectation maximization, OAG for Open Academic Graph, SSL for semi-supervised learning)

### Soundness
2

### Presentation
2

### Contribution
1
