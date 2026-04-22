# Feature Attribution for Label-Specific Feature Selection in Multi-Label Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Multi-label dimensionality reduction is an important but difficult problem. Feature selection approach searches a feature subset for multiple labels, which preserves label correlations well; however, not all features are necessary for specific labels. Label-specific approach constructs diverse feature spaces for different labels, paying more concentration on label characteristics but facing the complexity pressure arising from the number of labels. In this paper, a feature attribution based label-specific feature selection method is proposed, striking a balance between efficiency and accuracy. Feature attribution quantifies the contribution of every feature to the prediction, which commonly can be achieved with the gradient of deep learning output with respect to input. For multi-label learning, a simple shallow neural network is sufficient to construct a multi-input multi-output mapping, where the gradient of each label with respect to each input feature can be used for label-specific feature ranking and selection. Compared to state-of-the-art multi-label feature selection methods, the proposed method achieves comparable or superior performance while requiring less than 10\% of their runtime in most cases.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FALS, a rapid method for label-specific feature selection in multi-label learning. The core idea is to leverage a lightweight, shallow neural network for feature attribution. It assesses feature importance by computing the gradient of each label's output with respect to each input feature. This approach efficiently identifies the optimal feature subset for each label in parallel, addressing the high computational cost of traditional label-specific methods.

### Strengths
The core methodology is clear, simple, and effective. Furthermore, the paper provides a theoretical advantage for the label-specific approach over joint feature selection from an information-theoretic perspective.

### Weaknesses
Please see Questions.

### Questions
1)The method is built upon feature attribution. However, the introduction does not provide a clear rationale for selecting this technique as the foundation. Could the authors elaborate on this choice?
2)FALS evaluates feature gradients independently of the final classifier. Does the proposed method account for or capture potential feature interactions?
3)The complexity analysis states that FALS requires only T=10 epochs, whereas other algorithms need 100-1000 iterations. Could the authors provide a more detailed justification for this significant difference? Why is the shallow network considered sufficiently trained for attribution in so few epochs?
4)The experiments include three single-label selection methods (e.g., RF) for comparison. What is the significance of including these baselines? Additionally, only two directly related label-specific methods were compared. Are there other recent or classic algorithms in this category that could be included for a more comprehensive comparison?
5)Regarding the box plots in Figure 2, Please clarify the meaning of the graphical elements. Does the red line represent the mean or median? What do the upper and lower bounds signify? Also, outliers (circles) are shown only for Hamming Loss. What do these circles represent, and why do they appear exclusively for this metric?
6)Regarding the third conclusion from the analysis of Figure 4: The paper claims that H=sqrt(DC) performs best. However, the plots for Ranking Loss and Average Precision suggest that performance is optimal when the number of layers is 1 (i.e., H=0) or when H=D or H=C, not necessarily H=sqrt(DC). Could the authors clarify this discrepancy?
7)The theoretical analysis in the appendix (proof of Theorem 2) relies on the assumption of label conditional independence. This assumption is often violated in multi-label learning where labels are correlated. To what extent does this violation affect the theoretical guarantees and practical performance of FALS?

### Soundness
3

### Presentation
3

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
This paper introduces FALS (Feature Attribution based Label-specific Selection), a method for multi-label feature selection. The approach uses gradients from a simple two-layer perceptron (TLP) to score and select features for each label independently. The authors claim this method is highly efficient while achieving competitive performance.
This paper should be rejected because (1) the experimental validation lacks rigor, (2) the method's stability and sensitivity to its hyperparameters are not explored, and (3) Its contributions may lean toward incremental engineering heuristics rather than principle-based algorithmic breakthroughs.

### Strengths
1.Simplicity：The method's simplicity is a major advantage; relying on a shallow MLP makes it easy to understand and implement.
2.Computational Efficiency: The complexity analysis in Table 1 and the empirical runtime results in Table 4 show that FALS can be orders of magnitude faster, particularly on large datasets with many instances or labels.

### Weaknesses
1. Lack of Experimental Rigor and Insufficient Validation 
1.1 Weak Protocol: The paper uses a single, fixed 2:1 stratified split for training and testing. This approach is highly susceptible to sampling bias and provides a much less reliable estimate of generalization performance than the standard five/ten-fold cross-validation used in related work (e.g., [1] [2]).
1.2 Incomplete Metrics: The evaluation omits key multi-label metrics like Coverage, Macro-F1, or Macro-AUC. These are crucial for understanding performance, especially on imbalanced datasets, which are common in multi-label learning.

2.Unexplored Stability and Incremental Contribution
2.1 The entire output of this method (feature ranking) relies solely on the gradients of a global optimal model (TLP) trained for only $T=10$. The selected features may merely be the result of insufficient training or overfitting in a specific model, rendering this method unreliable.
2.2 The "weak network" argument reads as a post-hoc justification for a simple model, not a well-motivated design choice. Figure 4 shows that performance actually decreased as the number of layers increased, a phenomenon that has not been adequately explained.

References：
[1] Zhu Y, Kwok J T, Zhou Z H. Multi-label learning with global and local label correlation[J]. IEEE Transactions on Knowledge and Data Engineering, 2017, 30(6): 1081-1094.
[2]Mao J, Wang W, Zhang M L. LIMIC: Label Specific Multi-Semantics Metric Learning for Multi-Label Classification: Global Consideration Helps[C]//IJCAI. 2023: 4055-4063.

### Questions
1.The entire method depends on meaningful gradients from the TLP. Can you provide a rigorous sensitivity analysis showing that the final feature rankings are stable across different random initializations, training epochs, and hyperparameters? 
2.Given the model's simplicity, how would FALS perform on synthetic or real-world datasets specifically designed with highly non-linear decision boundaries or complex, higher-order feature interactions? The current experiments seem to avoid such challenging scenarios where the "weak learner" assumption would likely break down.
3.It is recommended to use k-fold cross-validation to update the experiment.
4.Key concepts are sometimes introduced without sufficient motivation. For example, the choice of hidden layer size $H=\sqrt{DC}$   is presented as an "empirical approach" without justification or reference. (line116)

Things to improve the paper that did not impact the score:
1.Tables 3 and 4 use numerical labels for dataset names. These datasets are familiar to researchers in the multi-label domain, and using numerical labels actually makes reading them less convenient. The significance test visualization (Figure 6) is very helpful. However, please consider adding text labels with the algorithm names to the nodes. This would make the plots much easier to interpret without needing to refer back to a separate list.
2.Label Independence Assumption in Theory: The theoretical justification connecting attribution to Fisher information relies on the assumption that labels are conditionally independent (a core aspect of the proof for Theorem 2). This contrasts sharply with principled multi-label methods, such as LIMIC [2], which explicitly model label co-occurrence to improve the learning process.

### Soundness
2

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
This paper proposes a fast feature selection method for specific labels based on feature attributes. It employs feature attribution techniques from the interpretable domain to quantify each feature's contribution to prediction, achieved by utilizing the gradient of the output relative to the input. The experiments are comprehensive, and the model demonstrates favorable performance.

### Strengths
This paper features a well-structured and clearly organized framework, presenting a straightforward proof of the relationship between fisher information and multi-label feature attribution. The explanation is accessible and easy to understand. It also conducts a substantial number of comparative experiments, demonstrating thoroughness, and provides an intuitive analysis of time complexity.

### Weaknesses
In my view, this paper simply applies feature attribution methods from the interpretable domain directly. Moreover, it employs the most basic direct gradient attribution. The theoretical section merely calculates partial derivatives of the output with respect to each input feature while linking them to fisher information. Furthermore, simple direct gradient attribution has significant drawbacks. Nonlinear functions inevitably exhibit phases of gradient saturation, during which gradients tend toward zero, thereby violating sensitivity criteria. Direct gradient attribution focuses on the input-output relationship at a single point in the current input. When a feature's value happens to be in a gradient saturation phase, the attribution weight assigned to that feature is often negligible, even though the feature itself is not unimportant. Therefore, directly using gradient attribution presents significant problems.

### Questions
1. This paper employs an empirical average of individual instance gradients on the training set to obtain the final feature score matrix. Why is this approach taken? I believe weighting should be applied. Instances with more label 1s should receive lower weights, as these instances are more ambiguous. Conversely, instances with fewer label 1s should receive higher weights, as they better highlight certain labels.
2. Regarding feature attribution, this paper should compare attribution methods based on backpropagation, such as DeepLift and layer-wise elevance propagation (LRP), with integral gradient (IG) algorithms. I believe using IG is more reasonable than direct gradient attribution.

### Soundness
3

### Presentation
3

### Contribution
1
