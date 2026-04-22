# Dummy Consistent Path Method for Robust Attribution with Irrelevant Features

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Attribution methods are widely used to interpret deep neural networks by identifying the input features that contribute to model predictions. Among these, path-based approaches, which accumulate gradients along a path from a baseline to an input, are often preferred for their theoretical foundations such as the Aumann-Shapley value. However, in networks with rectified activations, these methods frequently assign importance to irrelevant features—violating the Dummy Player axiom—due to the shattered gradients problem. Therefore, we propose the *Dummy Consistent Attribution Path Method* (DCAPM), a novel path-based approach designed to strictly enforce the exclusion of irrelevant feature interactions. Unlike standard approaches, our method dynamically constructs a path that nullifies spurious attributions arising from feature interactions with dummies to ensure zero attribution at every step, thereby ensuring theoretical consistency. To validate the practical impact of this theoretical adherence, we introduce a new metric that quantifies the alignment between dummy consistency and explanation faithfulness. We demonstrate that while existing methods systematically fail to satisfy this consistency, our approach yields significantly more robust and faithful explanations by adhering to the axiom.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Dummy Consistent Attribution Path Method (DCAPM), a path-based attribution method that rigorously satisfies the Dummy Player and Dummy Consistency axioms. Specifically, DCAPM dynamically constructs an integration path, nullifying spurious attributions from irrelevant feature interactions with dummies to ensure zero attribution at every step, grounded in a principled approximation of the Shapley interaction value.

### Strengths
The topic this paper focused on is very important in explainable AI.

### Weaknesses
1. The explanation for the " integration path" in introduction section is unclear. Can authors provide any vivid example for explanation? The physical meaning for the mapping $\gamma$ in Definition 1 of path function is missing, making it difficult to understand.

2. Eq. (2) only considers the interaction between two features i and j. How about interactions among features more than 2? In implementation, how to select features i and j? How about its computation complexity? Can authors provide a pseudocode for the proposed method？

3. The physical meaning and the motivation of the loss function in Eq. (3) is missing. Authors should clearly explain this function, as it is the main contribution of this paper.

4. From figure 3, I cannot agree with author's opinion that the proposed method has minimal background noise. However, DGA has less background noise than the proposed method. Thus, this result seem not convincing enough to verify the efficiency of the proposed method.

5. Could the proposed method be applied to transformer-based models and ConvNext model that have achineved sota performance nowadays?

6. Can the proposed method be applied to the dataset with spurious correlations, such as waterbird datasets?

### Questions
Please refer to the weaknesses. 
If all of my problems are addressed, I will raise my rating. 
I like this paper, however, the presentation is poor, especially lack of physical meaning for each equation. Besides, the designed experiments is too simplistic and results seems not convincing enough.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to improve gradient-based feature attribution methods, such as integrated gradients, that use a path integral to quantify feature attribution values. The authors target so-called "dummy features", which are described as features that do not contribute to the prediction of the explained instance. It is observed that due to evaluating the gradients at different regions of the feature space, such dummy features are attributed non-zero attribution values, since their gradients to not necessarily vanish along the whole path. The authors then propose the "Dummy Consistent Path Method" as a multi-step process: First, a path is constructed along masked inputs, where masks are iteratively constructed by optimizing for sparse masks that significantly drop the payout $f$. Second, the "relevant" features are identified by using a threshold for the partial derivatives obtained at each iteration. Third, the derivative of the path is then used to construct the final attribution method.

### Strengths
- This paper identifies the important problem that averaging gradients from different regions (e.g. along the path from $x$ to the baseline $x'$ in IG) may assign contributions to features that were not used by the model at the original data point $x$. As a solution, the authors essentially propose to mask only features, where the masking reveals a significant drop in the prediction, which is novel and interesting.
- Dealing with such dummy features by constraining the computation only to features that were not identified as dummy in the prediction of $x$ (and later on) is interesting
- The paper provides a comprehensive methodology, description and experimental details. It is well structured and clearly presented.

### Weaknesses
- One major problem is the motivation: The authors present dummy consistency (Definition 4) as receiving zero attribution when all gradients along the path are zero. Clearly IG satisfies this axiom. In contrast, Figure 1 claims that this axiom is violated. However, as the authors describe themselves (see "spurious interactions"), the non-zero attribution is due to the traversal across other regions of the feature space, where the model indeed uses this feature to predict the output, resulting in a non-zero gradient. Moreover, I would have expected to see DCAPM method here in this figure, as I do not see, how the novel method resolves these issues. In fact, the whole point of evaluating not only the gradients at $x$ (also for simple methods like SmoothGrad) is that we investigate regions "close" to $x$, since gradients are a very local measure, especially in high-dimensional DNNs. Therefore, any path-based method crossing regions, will be confronted with the problem that the gradients (i.e. attribution values) are not necessarily the same at another instance, and features that were not relevant before, could now be relevant.
- As a result of the first point: I think this paper's motivation is rather constructing a path that masks more effectively selected features that are "relevant" to the prediction, since their masking substantially lowers the prediction / payout $f$. As a result, the method will only investigate changes in features that were relevant at the explained instance $x$. But this could have been done without even using an iterative optimization procedure.
- Another major problem is the methodology: As the authors note, that problem of "dummy (in)consistency" arises from the fact that the gradients are evaluated along the path and do not necessarily reflect the gradients at the original input $x$. The authors restrict the update of the (continuous) masks to "important" features identified by large enough gradients, i.e. these features significantly change the prediction, if masked. However, if we assume that the feature space is partitioned by the DNN into regions, where the prediction is linear, then one of the following two scenarios might happen:
  - A: The new masked input is in the same region, i.e. the gradients remain unchanged or
  - B: The new masked input is in a different region, i.e. the gradients can arbitrarily change, meaning that previously feature identified as unimportant (zero gradient) could now become relevant (non-zero gradient). This is due to the fact that relevant features were only taken into account when constructing the mask and not when evaluating the gradients of the model. 
 
  Right now, I do not see, why this attribution method would suffer less from assigning non-zero contributions to features that were not relevant at point $x$. Moreover, I would expect the objective the authors try to optimize would be already resolved by using the most basic method Input $\times$ gradients.

### Questions
- Could you explain, what you mean by dummy consistency? In the draft, it seems that this definition is used interchangeably for features that obtain zero attribution, if they are "not relevant" (close to zero gradients) at the explained instance $x$, or whose gradients are zero along the whole path (Definition 4).
- Could you explain, why your method will assign close-to-zero-attributions for features that have close-to-zero-gradients at the explained instance $x$. As far as I understand, you only restrict the mask to "relevant" features, but the model could still predict very differently for the novel masked input.
- Follow-up to the previous question: Is it overall desirable to enforce zero-attributions for features that have zero-attributions at $x$? What if the model is very non-robust and actually uses features for instances that look very much like $x$?
- Could you add your method to the example in Figure 1? Moreover, could you add a simple baseline, like Grad$\times$Input?

### Soundness
2

### Presentation
3

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
The authors present the Dummy Consistent Attribution Path Method (DCAPM), which optimizes attribution paths with a mask-based approach while explicitly "blocking" the interaction between dummy and signal features at each step. The method is designed to satisfy the Dummy Player and Dummy Consistency axioms, thus reducing errors caused by spurious interactions, commonly referred to as "fragmented gradients." The manuscript provides method derivations, evaluation metrics, and comparison experiments.

### Strengths
1.	Clear Focus on an Important Issue: The paper addresses Dummy Consistency, an aspect of attribution stability that has been relatively under-discussed. The authors elevate this principle to the level of method design, offering both theoretical and practical significance (see Sec. 2.2 and the overall motivation).
2.	Intuitive and Interpretable Method: The use of low-resolution masks and optimization to produce dynamic baselines is clearly articulated. It provides a strategy for "finding clean baselines per sample and avoiding noise gradient accumulation in attribution" (see Sec. 3.1–3.3).
3.	Introduction of a New Metric: The authors propose the Dummy-Consistency metric (Eq. 11) to quantify the satisfaction of the axioms, which concretizes the concept of "robustness."

### Weaknesses
1.	Questionable Table 1 Results (DiffID):
For example （not only this one）, Table 1 shows a result of DiffID = 0.978 ± 6.41 for InceptionV3, where the standard deviation is much larger than the mean. DiffID measures the difference in attribution values between the most significant inserted pixels and the least significant deleted pixels. While a higher DiffID indicates the method's ability to identify important features, the large error margin significantly affects the interpretability and trustworthiness of the analysis. This issue is also present in other experiments in the appendix. Please address this inconsistency and clarify the statistical reporting. 

2.	Lack of Empirical Statistical Analysis for Dummy Consistency:
The paper emphasizes that a primary contribution is enforcing the Dummy Consistency axiom during path optimization (as discussed in Sec. 3), where spurious interactions between features are supposed to be removed. Eq. (6) is used to quantify this, and the goal is to make it zero. However, it does not show a empirical statistical analysis to support these claims. The appendix includes a proof, but would benefit from including relevant statistical evidence for the empirical results, regarding the effectiveness of enforcing the Dummy Consistency axiom in the experiments .

3.	Limited Analysis of Generalization Across Tasks:
The experiments on text tasks only compare entropy on BERT-Large (IMDb) and claim sparse attribution improvements. However, similar analyses are not conducted for image tasks. Moreover, the comparison of attribution methods through other metrics is absent on text tasks. A broader analysis is needed to demonstrate the generalizability and effectiveness of the proposed method across various tasks, including image and text domains .

4.	Formatting Issues:
The symbol τ is used in different contexts throughout the manuscript: once as a gradient threshold and once as a dummy threshold. To avoid confusion, I recommend differentiating the symbols, for example, by using τ_g for the gradient threshold and τ_dum for the dummy threshold.

### Questions
See weaknesses

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
The paper introduces a method of post-hoc, gradient based attribution in the path-method vein. The primary motivation of the paper is to craft an attribution method that does not attribute non-zero value to "dummy features", or features that are independent of label truth. To this end, the paper purports to illustrate how existing methods do report non-zero attribuiton to dummy features, motivates why attribution methods should not have this issue, and then outlines a method that avoids the issue: dummy-consistent attribution path method (DCAPM). Experimental results comparing attribuiton methods are provided, showing desirable performance for DCAPM.

### Strengths
- Descent (rigorous and clear) methematical presentation
- Positive experimental results
- descently clear writing most of the time.

### Weaknesses
- Critical issues in the treatment of the dummy consistency axiom
- The method seeks to address dummy consistency, but does not engage in an axioms-oriented development of the method:
  - the priority of the paper is not to find methods that satisfy a set of desirable axioms
  - the paper provides no characterization results (only my method satisfies this set of axioms), which is the strongest characteristic of so-called axiomatic attributions methods such as SHAP and IG.
- Some language is loose
  - focus on Shapley Value in beginning is misleading, as it is non-central to the paer compared to IG.
  - terms common in the attribution-discussion, like "shattered gradeints", are used to explain phenomena, but not treated rigorously.

### Questions
- Critial issues with the treatment of dummy:
  1) The paper confuses the following two concepts: 1) a feature is irrelevant to the true label of an image, and 2) a feature has no effect on the model's categorization of an input.
  2) the paper claims that an attribution method violates dummy consistency if there is a category-2 feature that the attibution method assigns non-zero attribution to.
  3) due to the above confusion, the paper claims that an attribution method does not satisfy dummy consistency becase a category-1 feature does not have zero attribution.

We see this in lines 201-203, where ithe paper claims that some attribution methods eroniously agggregate/report noisy gradients of irrelevant features. However, those features are presumably category-1 features; they certainly are not category-2 features.

The purpose of an attribution method is to explain the model, not to report what we believe the model should be doing. We do not entirely know what the model is doing, it could be using what we believe are spurious features. Thus, if the attribution method reports that a category-1 feature is relevant, it is difficult for us to say we know the model is not significantly using it. Even if we design a feature to be irrelevant (feature is noise), when we train a model it will try to predict using the irrelevant feature; intentionally irrelevant features usually do cause an effect in models. Thus, when an attributon reports that these features are contributers, how do we know the attribution method is reporting erroniously?

- The author at times tries to get around the above issue buy using a threshold assumption - if the gradients of a feature is significantly and consistently small, then we can assume that the feature's effect on the model is adding small-magnitude irrelevant noise to the output. I would agree that if a feature is small-magnitude irrelevant noise then the gradients are likely to be small and irregular (still needs some justification). However, I do not agree that if gradients are small and irredular, then the feature is irrelevant. What if a consistently small-gradient feature is important to a model's prediction as a small and perhaps noisy contributer? In this case, DCAPM would attribute zero to contributing features, mistaking them for irrelevant features.

- To be clear, the method seems to have some merit in reducing spurious correlations and shattered gradients (it is theoretically suggestive and via experiemntal results), but the interaction of the paper with the axiomatic attributions feamework and dummy consistency needs more careful consideration and proper hedging, perhaps in cooperation with a different framing and motivation that emphasizes the axioms less.

- I found figure 1 more confusing than illuminating. What is the color coding of the first/second columns? What are are the regions detailing? Why exactly are they being warped? Is the only difference between row 1 and 2 IG vs FG? If so, why are they different when no attributions are plotted, and what is the color difference between them? What are the black, white, and green dots? When x2 is intoduces, was the model retrained?

- Equation 6 uses an i subscript, but does not depend on i. Perhaps I am misunderstanding something?

### Soundness
1

### Presentation
3

### Contribution
1
