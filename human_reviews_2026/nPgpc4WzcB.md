# EDEL: Error-Driven Ensemble Learning for Imbalanced Data Classification

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
The class imbalance problem poses a critical challenge in high-stakes applications such as fraud detection, where the minority class often represents rare but consequential cases. In such settings, misclassifying minority instances can lead to substantial financial loss, underscoring the need for learning algorithms that remain reliable under severe imbalance. While deep learning methods have achieved remarkable success across various domains, their effectiveness often depends on large-scale datasets, and their black-box nature limits their interpretability, which is a critical requirement in high-stakes scenarios. To address this gap, we propose **E**rror-**D**riven **E**nsemble **L**earning (**EDEL**), an adaptive machine learning algorithm that dynamically introduces misclassified instances during training, thereby placing greater emphasis on hard-to-classify samples. Through theoretical analysis and extensive experiments on multiple real-world datasets, EDEL demonstrates strong effectiveness, particularly under challenging imbalanced conditions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the class imbalance problem commonly encountered in high-stakes applications such as fraud detection, credit risk assessment, and medical diagnosis. The authors propose Error-Driven Ensemble Learning (EDEL), a new ensemble framework designed to emphasize hard-to-classify samples during training. EDEL reinjects misclassified instances into subsequent training rounds, refining decision boundaries and improving recognition of the minority class. Theoretical analysis using McDiarmid’s inequality and Bayes’ theorem is provided to justify the reduction in empirical error and the improvement in generalization. Extensive experiments on seven real-world datasets with varying imbalance ratios demonstrate that EDEL achieves superior performance in terms of AUC and F1-score compared to multiple baselines (SMOTE, RUS, CHRE, etc.) across four classifiers (DT, RF, XGB, and LGBM).

### Strengths
(1) The paper introduces an error-driven ensemble learning framework (EDEL) that adaptively reinjects misclassified samples into the training process. This progressive emphasis on hard-to-classify instances offers a perspective on handling class imbalance beyond traditional resampling and cost-sensitive learning approaches. 

(2) The paper provides both theoretical justification and empirical validation across seven diverse real-world datasets with imbalance ratios ranging from mild to extreme. The reported performance improvements in AUC and F1-score across multiple classifiers (Decision Tree, Random Forest, XGBoost, and LightGBM) demonstrate the robustness and general applicability of EDEL. 

(3) The methodology is described in a stepwise and algorithmic manner (Algorithm 1), supported by mathematical definitions of “easy-to-classify” and “hard-to-classify” samples. The structure of the paper—problem definition, theoretical grounding, algorithmic description, and experiments—is logically coherent and easy to follow.

### Weaknesses
(1) While the proposed “error-driven” mechanism is conceptually interesting, it largely resembles existing ensemble paradigms such as boosting, where misclassified samples are repeatedly emphasized during iterative training. The paper would benefit from a more explicit comparison and discussion of how EDEL fundamentally differs from or improves upon modern ensemble refinements. Without this clarification, the claimed novelty appears incremental.

(2) The paper repeatedly claims that EDEL enhances interpretability, yet no concrete explanation mechanism (e.g., feature attribution, model introspection, or visualization) is presented, fixing hard samples does not equate to interpretability.

(3) Key design choices such as the number of weak classifiers, subset partition strategy, and reinjection frequency are not systematically analyzed. An ablation or sensitivity analysis would clarify how these factors influence performance, stability etc.

### Questions
The following are my concerns and questions:

(1) In the introduction, the authors mention the general interpretability challenge in deep models but do not review or position existing interpretability tools (e.g., SHAP, LIME, or counterfactual explanation methods) relative to the imbalance problem. Without this, the motivation appears incomplete.

(2) The statement that interpretability arises naturally from observing classifier errors is conceptually weak. Interpretability usually requires explicit mechanisms (e.g., feature importance, contribution maps). The authors should explain whether EDEL provides quantifiable interpretive outputs.

(3) In the proposed method, the emphasis on misclassified samples resembles classical boosting techniques or ideas. The authors should clearly articulate how EDEL differs algorithmically or theoretically from these well-established methods.

(4) Using unweighted parameter averaging may not be optimal, especially when subsets have varying difficulty or imbalance ratios. Why not adopt adaptive weighting or validation-based combination? And the average mechanism requires that these classifiers(e.g., deep models) have the same model architecture, limiting the diversity of these classifiers, them ultimately become homogeneous.

(5) Furthermore, the definition of hard-to-classify samples does not distinguish between truly **ambiguous cases and mislabeled/noisy instances**. The reinjection of such noisy data might propagate errors instead of improving robustness. How the proposed methods solve this critical issue?

(6) In the training process of EDEL, it seems the reinjection process only performs one iteration? I mean, what is the condition of ending "while not done"?  If the iteration is only 1, it is possible there are still many hard-classify samples for each classifier, then what's the point of the reinjection process? If not, what is the ending condition?

(7) The algorithm description suggests reinjection of misclassified samples, yet the complexity derivation omits the number of such iterations. A multiplicative term  should be included to represent the number of update cycles (e.g., L rounds).

(8) The paper claims that EDEL is interpretable but provides no model-level explanation, actually it is merely stating that “hard samples were fixed”, not proving that EDEL is “explainable”. For me, the introduction of "interpretability" concept in this paper is really weird. The introduction of “interpretability” appears conceptually inconsistent with the technical contributions of EDEL. The use of this term feels more rhetorical than substantive.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on the task of imbalanced data classification, and proposes Error-Driven Ensemble Learning, an adaptive machine learning algorithm supported by theoretical analysis. The proposed method solves two challenges in this area, i.e., requirement of attention on hard-to-classify samples and requirement of attention to interpretability.

### Strengths
S1: The studied problem is important.
S2: The paper is easy to follow and well-written.
S3: The paper has sufficient and detailed theoretical analysis.

### Weaknesses
W1: Limited novelty for the first challenge. The first challenge, i.e., addressing hard-to-classify samples, seems to have been mentioned by a lot of existing methods. As noted in the manuscript (Line 1219), Focal Loss reduces the influence of easily classified samples, emphasizing hard-to-classify instances, while Class-Balanced Loss reweights losses based on class frequency to balance contributions.
Furthermore, prior works like MESA [1] not only mentioned that there are existing methods assume instances with higher training errors are more informative for learning, but also extend their solution to other critical issues, such as generating synthetic samples for minority classes. Given this context, two key questions arise: 1 What is the unique advantage or fundamental difference of the proposed method compared to these specific, well-known techniques in handling the first challenge? 2 Why were comprehensive baselines like MESA not included in the comparisons?

[1] MESA: Boost Ensemble Imbalanced Learning with MEta-SAmpler. NeurIPS 2020.

W2: Apart from the above mentioned methods, a lot of methods have been mentioned in related work, why not compare with them? Are used baseline methods the SOTA methods? If not, SOTA techniques should be compared. Or authors need to give reasons why they did not compare with them.

W3: Lack of clear explanation about technical details. Why divide the dataset into sub groups? And how to ensure that the algorithm could converge (Line 165)?

W4: Some typos. For example, STD belongs to algorithm-level according to line 348, but it belongs to data-level according to related works in Line 1196. 

W5: Since the second challenge is the requirement of interpretability, is there any quantitative analysis or intuitive qualitative analysis for explanation, apart from theoretical analysis?

### Questions
Please see weaknesses above.

### Soundness
2

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
This paper proposes EDEL (Error-Driven Ensemble Learning), a novel algorithm to address class imbalanced data classification. EDEL works by partitioning the data and having parallel weak classifiers dynamically identify and re-train on misclassified, "hard-to-classify" samples from other subsets. The authors provide theoretical proof that this process enriches minority class representation and demonstrate through experiments on 7 datasets that EDEL significantly outperforms baselines in F1-measure and AUC, especially under extreme imbalance.

### Strengths
1.The authors provide strong theoretical support for EDEL. The paper uses Bayes' theorem to prove the enrichment phenomenon of minority class instances within the misclassified sample set and employs McDiarmid's inequality to demonstrate the algorithm's convergence, presenting a rigorous line of reasoning.

2.The method was validated on 7 real-world datasets with varying imbalance ratios (IR), including an extreme case with an IR as high as 577.88. In terms of AUC and F1-measure, EDEL consistently outperforms baseline methods, showing particularly strong performance on highly imbalanced datasets.

### Weaknesses
1. In the table3，Some datasets (e.g., GMSC, CDH) exhibit large standard deviations (±0.2987，±0.2778 etc), raising questions about the method’s stability across folds.

2. The experimental evidence supporting the interpretability claim is relatively limited. Although Tables 4 and 5 provide some descriptive analysis, the evaluation remains fairly simple. The paper would benefit from richer interpretability experiments.for instance, t-SNE or feature embedding visualizations showing how hard-to-classify samples evolve before and after EDEL’s error-driven enhancement.

### Questions
1. What is the training time or computational cost compared to SMOTE ，S-T-D or CHRE? A runtime table would be helpful.

2. Could the authors consider adding some visualization components to better illustrate the interpretability aspect of EDEL? For example, visual analyses such as t-SNE projections or feature-space evolution plots could provide more intuitive evidence of how the model improves the representation of hard-to-classify samples.

3. EDEL reinjects all misclassified samples during training. Would this potentially lead to overfitting on noisy or outlier instances from the majority class? Have the authors considered applying a filtering mechanism to $\mathcal{D}_{i}^{h}$, for example, by reinjecting only the misclassified minority-class samples?

### Soundness
3

### Presentation
2

### Contribution
3
