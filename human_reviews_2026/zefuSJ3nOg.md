# Dynamic Multi-sample Mixup with Gradient Exploration for Open-set Graph Anomaly Detection

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
This paper studies the problem of open-set graph anomaly detection, which aims to generalize a graph neural network (GNN) trained with a small number of both normal and abnormal nodes to detect unseen anomalies different from training anomalies during inference. This problem is highly challenging due to both the data scarcity of unseen anomalies and the label scarcity for training nodes. Towards this end, we propose a novel approach named Dynamic Multi-sample Mixup with Gradient Exploration (DEMO) for open-set graph anomaly detection. The core of our proposed DEMO is to leverage a dynamic framework to adapt the optimization procedure with high generalizability. In particular, our DEMO first adaptively fuses multiple seen nodes to simulate the unseen anomalies, which expands the decision boundary for the detection model with enhanced generalizability. Moreover, we dynamically adjust sample weights based on their energy gradients to prioritize uncertain and informative nodes, ensuring a robust optimization procedure. To further address both label scarcity and severe class imbalance, we maintain a memory bank of historical records to guide the pseudo-labeling process of unlabeled nodes. Extensive experiments on various benchmark datasets validate the superiority of the proposed DEMO in comparison to various baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose an approach named Dynamic Multi-sample Mixup with Gradient Exploration (DEMO) for open-set graph anomaly detection, leveraging a dynamic framework to adapt the optimization procedure with generalizability. Given experimental comparison show the effectiveness to some extent.

### Strengths
1. The authors propose an approach named Dynamic Multi-sample Mixup with Gradient Exploration (DEMO) for open-set graph anomaly detection. 
2. Given experimental comparison show the effectiveness to some extent.

### Weaknesses
1. It can be a question that, since there exists anomaly detection dataset benchmark such as [1], why the authors didn't conduct experiments on the real-world datasets. It will be better to report the experimental results on those datasets. 
2. The unseen anomaly simulation can be an significant part of the framework. Therefore, the authors should show the generated embedding of the simulation to show the effectiveness (on above mentioned datasets). 
3. The pseudo-labling generation can be an important component of the framework, and the corresponding loss require a high-quality pseudo-labels. The auhors should present if the generated pseudo labels match with the ground truths (on above mentioned datasets). 

[1] Jianheng Tang, Fengrui Hua, Ziqi Gao, Peilin Zhao, Jia Li. GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection. NeurIPS 2023.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Dynamic Multi-sample Mixup with Gradient Exploration (DEMO) to tackle open-set graph anomaly detection. Its method (1) synthesizes anomalies via multi-sample mixup at the embedding level, (2) weights samples using validation-gradient signals, and (3) adapts class thresholds with a memory bank. Experiments cover six datasets and twelve baselines with ablation studies and sensitivity analysis.

### Strengths
1. Open-set graph anomaly detection is an important problem, and is under-explored.
2. Multi-sample mixup at the embedding level is a neat way to approximate unseen anomalies.
3. Comprehensive experiments covering 6 datasets and 12 baselines with ablations study and sensitivity analysis.

### Weaknesses
1. The use of index i seems inconsistent. In Definition 3.1, i appears to index synthetic representations, but in Equation 1, i seems to refer to original seen anomaly nodes. Please clarify the indexing scheme and distinguish between source nodes and synthetic outputs.

2. In Section 3.3, some implementation details are missing for the Hessian matrix computation: (i) How is the Hessian computed in practice? (ii) What are the "optimal parameters"—current parameters at each step or from separate optimization? (iii) Computing and inverting the full Hessian is computationally prohibitive for large weights. Is any approximation like diagonal Hessian or Hessian-vector products used? Implementation details and computational feasibility discussion are needed.

3. The gradient-based weighting in Section 3.3 uses validation loss gradients to compute training sample weights, which appears to use validation data to guide training. This could have the risk of data leakage and overfitting to the validation set. It would be helpful to discuss whether this leads to data leakage or overfitting and, if feasible, include empirical evidence.


4. Section 3.4 would benefit from further clarification. Could you please specify what is meant by “samples selected for class c”? Are these unlabeled samples assigned the pseudo-label c? Given that Table 3 shows pseudo-labeling has a substantial impact on performance, including visualizations of how the thresholds evolve during training could help illustrate this component’s contribution.


5. The manuscript does not include a computational or memory complexity analysis. Please provide the method’s time complexity and memory overhead. Given the use of Hessian matrices and their inverses, a scalability analysis with respect to model sizes is also needed.


6. In Figure 2, similar colors make methods difficult to distinguish. Please use distinct colors for each method.

### Questions
See Weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies open-set GAD, where the goal is to detect both seen and unseen anomalies with limited labelled data. It proposes DEMO, a dynamic multi-sample mixup and gradient exploration framework that generates diverse anomaly samples and adaptively re-weights them based on energy gradients. A memory-bank pseudo-labelling module is further introduced to enhance training stability and handle label scarcity. The perposed method on avergae outperform the baseline methods on the selected datasets.

### Strengths
1. The paper explores an interesting and important setting of open-set graph anomaly detection.
2.  The paper is well-written and easy to follow, with clear structure and presentation.
3. Theoretical analysis is provided for the proposed augmentation methods for GAD.

### Weaknesses
1. The proposed method relies on data augmentation (multi-sample mixup) as a key contribution, but none of the baseline methods use comparable augmentation strategies. If augmentation is the main source of improvement, the paper should include comparisons with existing augmentation-based methods such as GraphSMOTE (Zhao et al. 2021) or other graph Mixup approaches to ensure a fair evaluation.

2. The ablation results show that removing the pseudo-labelling component leads to a notable performance drop. Does this mean much of the improvement may come from this module rather than the proposed mixup strategy?   It would be helpful for the authors to further clarify on this.
    
2. The GAD-related work section fails to mention recent methods that leverage label information, such as meta-learning GAD (Meta-GDN, Ding et al., 2021), cross-domain GAD (ACT, Wang et al., 2023), and generalist GAD (ARC, Liu et al., 2024; AnomalyGMF, Qiao et al., 2024).

3. Since there are two hyperparameters in the main loss, the sensitivity study should cover all datasets and include comparisons with the baselines. The results in Tables 6 and 7 also appear quite sensitive to these parameters.

4. Several references cited as related work are already accepted by recent conferences but are still listed in arXiv format. 

5. The number of supervised baselines and datasets used in the experiments is relatively limited. A good reference for finding additional datasets and resources is the GAD Benchmark (Tang et al., 2023).

### Questions
Please refer to my weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a framework, DEMO (Dynamic Multi-sample Mixup with Gradient Exploration), to address the problem of open-set graph anomaly detection (GAD). This task aims to detect not only anomalies seen during training but also novel, unseen types of anomalies, using only a small set of labeled data. The core idea is a three-part dynamic training framework. First, to generalize from limited seen anomalies, DEMO uses a multi-sample mixup strategy that adaptively fuses multiple seen anomaly nodes to synthesize new, diverse samples, thereby expanding the decision boundary. Second, it employs an energy gradient-driven feedback mechanism to dynamically re-weight all training samples, prioritizing uncertain and informative nodes to ensure a more robust optimization process. Third, to combat label scarcity and class imbalance, it maintains a memory bank of historical predictions to guide a pseudo-labeling process with adaptive, class-specific confidence thresholds. The authors demonstrate that DEMO achieves performance on six real-world graph datasets under various open-set evaluation settings.

### Strengths
There are a few things I like about the paper:
1. The paper addresses a practical challenge in graph anomaly detection, moving from the closed-set assumption to a more realistic open-set scenario. This is relevant in many application areas when the anomalies are always evolving.
2. The authors provide experimental validation on six benchmark datasets of varying scales and domains.
3. The authors provide theoretical analysis of the proposed method.
4. The authors performed an ablation study to see the source of the performance improvement.

### Weaknesses
1. The work is motivated by open set problem setting. Which aims to also detect new types of anomalies. However, the main ingredient of the proposed method is mixup, where it can only detect anomalies that are linear combinations of seen anomalies in the embedding space. It will be less effective in detecting totally novel anomalies outside the one that have been seen before (which is the advantage of a purely unsupervised model).
2. There are two types of anomalies that need to be considered in the evaluation, anomalies that come from existing sets/classes, and anomalies that come from new set/classes (open-set). The paper, however, does not distinguish these two in the evaluation. I would suggest the authors add more experimentation that distinguishes these two. Particularly, I am interested in the performance of the proposed method vs semi-supervised baselines on detecting anomalies from existing sets/classes, and the performance of the proposed method vs purely unsupervised baselines on detecting anomalies from new sets/classes.
3. In the large-scale experiments, DEMO's AUC-PR score on the Yelp dataset is noticeably lower than the best-performing baseline (0.2238 vs. NSReg's 0.3029). Given that AUC-PR is a critical metric for highly imbalanced datasets, this specific underperformance needs further exploration.
4. The adaptation of purely unsupervised models into semi-supervised models is not clearly explained in the paper.

### Questions
Please address the weaknesses mentioned above.

### Soundness
2

### Presentation
3

### Contribution
2
