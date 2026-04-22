# TreeGrad-Ranker: Feature Ranking via $O(L)$-Time Gradients for Decision Trees

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
We revisit the use of probabilistic values, which include the well-known Shapley and Banzhaf values, to rank features for explaining the local predicted values of decision trees. The quality of feature rankings is typically assessed with the  insertion and deletion metrics. Empirically, we observe that co-optimizing these two metrics is closely related to a joint optimization that selects a subset of features to maximize the local predicted value while minimizing it for the complement. However, we theoretically show that probabilistic values are generally unreliable for solving this joint optimization. Therefore, we explore deriving feature rankings by directly optimizing the joint objective. As the backbone, we propose TreeGrad, which computes the gradients of the multilinear extension of the joint objective in $O(L)$ time for decision trees with $L$ leaves; these gradients include weighted Banzhaf values. Building upon TreeGrad, we introduce TreeGrad-Ranker, which aggregates the gradients while optimizing the joint objective to produce feature rankings, and TreeGrad-Shap, a numerically stable algorithm for computing Beta Shapley values with integral parameters. In particular, the feature scores computed by TreeGrad-Ranker satisfy all the axioms uniquely characterizing probabilistic values, except for linearity, which itself leads to the established unreliability. Empirically, we demonstrate that the numerical error of Linear TreeShap can be up to $10^{15}$ times larger than that of TreeGrad-Shap when computing the Shapley value. As a by-product, we also develop TreeProb, which generalizes Linear TreeShap to support all probabilistic values. In our experiments, TreeGrad-Ranker performs significantly better on both insertion and deletion metrics. Our code is available at https://github.com/watml/TreeGrad.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors proposed TreeGrad, a way of computing feature importance via solving the multilinear extension of the joint optimization problem. They showed its theoretical guarantees and empirical effectiveness, comparing against other insertion / deletion based methods.
The authors also presented an improved algorithm for computing probabilistic values in decision trees (called TreeProb), and a more numerically stable variant of TreeShap (they call it TreeGrad-Shap).

### Strengths
1. The theoretical analysis is thorough, especially the arguments in section 3, optimizing the joint makes a lot of sense
2. Clear explanation of the algorithms, algorithms mentioned are listed clearly in the paper
3. Strong experimental results, empirical evidence supports the authors' claims of their algorithms' effectiveness

### Weaknesses
1. Lack of a empirical run-time analysis: the authors claim a O(L) gradient-based method, when comparing with previous approaches with O(LD) complexity. However it is unclear how many gradient steps are enough (10? 100? from figure 3). I think it will be more beneficial to include an empirical run-time analysis.
2. Does randomness affect the method? Since it's gradient based, does different random seed affect the model explanation? I will suggest some analysis around that as well.
3. Also evaluation dataset seems to be of small-scale (9-datasets)

### Questions
Minor suggestions:
1. Line 82: ... decision trees serves as the backbone -> ... decision trees serve as the backbone
2. The algorithm names are confusing, TreeGrad, TreeProb, TreeGrad-Shap, TreeGrad-Ranker 
3. Main figure 4 could improve its look, the contrast is not clear when it comes the the paper's method / baselines, especially in the bottom parts.

### Soundness
3

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
3

### Summary
This paper aims to address the reliability issue of probabilistic values of Shapley values and Banzhaf values. The authors proposed TreeGrad-Ranker, which leverages averaged gradients of the multilinear extension as a remedy for decision tree models. The method extends Linear TreeSHAP and achieves fast O(L) time calculation. It also outperforms existing approaches in terms of insertion and deletion metrics on nine empirical dataset.

### Strengths
1. The paper is rigorous in its definitions and provides solid theoretical justifications for propose work.
2. The paper identifies the pitfalls of using probabilistic values and proposed an alternative optimization scheme.
3. TreeGrad-Ranker empirically outperform Shapley values, Banzhaf values, Beta-Shapley in terms of insertion and deletion metrics on nine datasets.

### Weaknesses
1. The abstract introduces several newly proposed names (TreeGrad, TreeGrad-Ranker, TreeProb and TreeGrad-Shap), which makes it difficult to identify the central contribution. It would be clearer and more engaging if the abstract can focus more on the main problem the paper aims to solve.
2. It’s very difficult to distinguish the performance of different methods in Figure 3 based on solely color. I recommend the authors use additional visual cues, such as distinct shapes, line types for different groups of variables, or just numerical values to demonstrate the proposed methods.
3. Due to the lack of linearity for TreeGrad-Ranker, generalization to popular ensemble trees methods like random forests and XGBoost is non-trivial, which limits the practical usage of the method.
4. Section 5 of the paper discusses the numerical instability of Vandermonde matrices in Linear TreeSHAP and uses the complex root of unity but does not cite prior work. In particular, the ill-conditioning problem of Vandermonde matrices and a similar solution for Linear TreeSHAP was previously mentioned in [1].

[1] Jiang, Z., Zhang, M., & Zhang, D. (2025). Fast Calculation of Feature Contributions in Boosting Trees. Proceedings of the 41st Conference on Uncertainty in Artificial Intelligence (UAI).

### Questions
How to set T and $\epsilon$ in practice?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work focuses on constructing a feature importance ranking for tree models. The authors concentrate on building rankings based on deletion and insertion scores. While such rankings can be constructed naively in theory, this approach is NP-hard. Traditionally, probabilistic values have been used as scores for constructing rankings, including the popular Shapley values and the less popular Banzhaf values. The authors provide a theoretical explanation of why probabilistic values are not appropriate for this task. They then propose their own approach to constructing scores for ranking by formulating a new objective that should yield better ranking scores. They demonstrate how calculating gradients can be utilized in this framework. The authors show that their gradient methods can compute any probabilistic value, including popular Shapley values and improve upon existing methods. They implemented Shapley values calculation as a benchmark and compare their approach to existing methods, demonstrating improvements and explaining issues with previous approaches. The paper concludes with an empirical evaluation of ranging generated with TreeGrad-Ranker.

### Strengths
* The authors clearly motivated the problem through Proposition 1, where they stated a type of impossibility theorem for probabilistic values.
In Proposition 2, they showed that the use of probabilistic values is equivalent to substituting linear surrogates for f(x), which provides a novel perspective on building rankings with probabilistic values.

* They clearly stated that optimizing the proposed score can be formulated as gradient optimization and showed the exact form of this gradient.
* The authors demonstrated an interesting connection between gradient calculation and calculating probabilistic values, introducing a new way of computing them (including Shapley values, which is an active and important research topic).

* They identified problems in previous approaches and designed and conducted experiments to showcase these issues.

* They evaluated the new ranking on multiple datasets with a varying number of features.

### Weaknesses
* *Proposition 1* - In this proposition, you assume that U is any function 2^N → R, which is of course true in terms of game theory. However, here we are in a use case where we have a model that takes all features and we calculate expected values (based on the conditional distribution given by the proportion in the tree). It seems that the tree plus the distribution that is also defined by the tree may be less expressive. In Bilodeau et al. (2024), the only requirement for the model class was to have 3 piecewise linear regions and non-zero probability outside of a neighborhood (though this was for marginal SHAP). Did you think about stating this theorem in the context of tree models?

* Do I understand correctly that the proposed algorithm applies only to single decision trees? If so, this limits the scope of the paper, as gradient boosted trees are widely adopted and used very often. Single trees are more interpretable, but that is precisely what makes a feature ranking method for forests needed. Can it be extended to models containing multiple trees?

* In the empirical section (Section 6), the hyperparameters of the model are not described. Only the implementation used is described. However, there is no information about the hyperparameter selection process - I think it would be important to ensure that these trees are reasonably well-tuned for the datasets. Also, the size of the trained models was not reported. (See also questions).

* The authors evaluated their method using the insertion and deletion scores. There are also other methods for benchmarking the quality of rankings, e.g., Agarwal, Chirag, et al. "OpenXAI: Towards a transparent evaluation of model explanations." The theoretical justification aligns with the fact that it works better for these scores; however, what about other metrics?

### Questions
* Regarding the use of gradient optimization and the specific step in Line 23 of Algorithm 2: Do you know if the optimization results typically converge to a global minimum or a local minimum? The landscape of this problem may differ from typical ML problems (like neural networks). Did the authors consider or experiment with alternative optimizers (e.g., SGD or ADAM)? For smaller datasets, is it possible to find the optimal solution to this problem and use it as a comparison baseline?

* Theoretical Constraints (Impossibility Theorem) Is the work subject to the constraints of the relevant Impossibility Theorem from Bilodeau et al. (2024) ? Specifically, does this theorem hold for the values calculated in Algorithm 2, or do the methods presented in the paper mitigate these theoretical problems?

* Scores as Feature Attribution - Did the authors consider using the scores generated by Algorithm 2 for feature attribution? These scores appear to fulfill the most important axioms for feature attribution. Have the authors explored whether they offer improved performance or desirable behavior in this context?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper deals with feature ranking of decision trees via probabilistic values, such as the well-known Shapley values. The quality of feature ranking is assessed with the insertion and deletion metrics. Empirically, optimizing these two metrics is closely related to optimizing a particular objective. The authors find that solving this optimization leads to generally unreliable probabilistic values. 

To address it, this paper optimizes the multilinear extension of the objective via gradient ascent called TreeGrad. Then, the averaged gradient is used to estimate the probabilistic values of features. The proposed method is better on insertion and deletion metrics.

### Strengths
1. The effectiveness of proposed method is supported by experimental results. That is both insertion and deletion metrics are significantly better than baselines.
2. I appreciate the authors’ dedicated efforts on the problem of feature ranking of decision trees.

### Weaknesses
1. The representation should be improved.

a) Too many abbreviation in abstract section (TreeGrad/TreeGrad-Ranker/TreeProb/TreeGrad-Shap/Linear TreeShap). Some of them are presented without further elaboration. 

b) The logic flow of introduction section is unclear. It arises a question: how reliable are probabilistic values in ranking features? However, it is unclear that whether the probabilistic values itself are unreliable or the optimization method is unreliable. If it were the former, the paper would fall apart.

c) The contribution summary part in introduction section lacks a clear focus. The core contribution is TreeGrad. While TreeProb/TreeGrad-Shap/TreeGrad-Ranker are byproduct contributions. Thus, the author should highlight the development of TreeGrad.

d) The overall representation lacks the description of key ideas and motivations. The current version is hard to follow and it should be restructured.

2. The key contribution of this paper is using the multi-linear extension of objective and develop an algorithm to compute its gradient. However, it is unclear why the multi-linear extension is better. Is it possible to directly compute the gradient of original objective?

### Questions
-

### Soundness
2

### Presentation
2

### Contribution
3
