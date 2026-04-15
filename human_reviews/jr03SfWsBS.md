# Unprocessing Seven Years of Algorithmic Fairness

- Decision: Accept (oral)
- Scores: 8, 6, 8, 6

## Abstract
Seven years ago, researchers proposed a postprocessing method to equalize the error rates of a model across different demographic groups. The work launched hundreds of papers purporting to improve over the postprocessing baseline. We empirically evaluate these claims through thousands of model evaluations on several tabular datasets. We find that the fairness-accuracy Pareto frontier achieved by postprocessing contains all other methods we were feasibly able to evaluate. In doing so, we address two common methodological errors that have confounded previous observations. One relates to the comparison of methods with different unconstrained base models. The other concerns methods achieving different levels of constraint relaxation. At the heart of our study is a simple idea we call unprocessing that roughly corresponds to the inverse of postprocessing. Unprocessing allows for a direct comparison of methods using different underlying models and levels of relaxation.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The problem the paper considers is building accurate models subject to a fairness constraint. There are many ways of building models but it is difficult to compare between different methods because a) the model performance depends on the underlying classifier and b) the models satisfy the fairness constraint up to different relaxations.

This paper seeks to solve both problems and run a large experiment on many different methods and models. They start with an approach they call "unprocessing" which takes the underlying classifier and removes the fairness constraint. In this way, different models can then reasonably be compared to each other. They then postprocess the classifiers to achieve the fairness constraint. There is an optimal way to achieve the postprocessing so this step also lets different models be compared to each other.

### Strengths
1. A simple way of comparing models with different fairness constraints. I hope this becomes widely adopted and used before people introduce their XYZ fairness algorithm.

2. A comprehensive evaluation of lots of models on four data sets. I especially liked two observations from their results:

* Models subject to a fairness constraint can actually achieve higher accuracy than models not subject to a fairness constraint when compared fairly (pun intended). The explanation they give is that fair training can take longer and use more resources because of the complexity in the algorithms.

* In their words: 

"Crucially, postprocessing the single most accurate model resulted in the fair optima for all values of fairness constraint violation on all datasets, either dominating or matching other contender models (within 95% confidence intervals). That is, all optimal trade-offs between fairness and accuracy can be retrieved by applying different group-specific thresholds to the same underlying risk scores."

I think this is intuitively obvious and it's nice to see experimental confirmation.

3. A technical description of how to achieve relaxed parity.

### Weaknesses
1. I found the technical description of how to achieve relaxed parity jarring from the rest of the paper. I would have liked this section to be longer and for more explanations there. I did find the figures quite helpful in understanding it.

2. A big selling point of the paper is the extent of their experiments. I think the reason they were able to do this is because they had access to a ton of compute. All the data sets and models (I believe) are easily accessible. If this is the case, I'm not sure that "having lots of compute" is really something we should reward as a contribution.

3. I found their approach intuitively obvious: Of course given a classifier, you can vary how much it violates a reward constraint in an optimal way. So I think the contribution here would be because (it seems like) no one has done this before rather than because it is so interesting.

### Questions
Is there anything in my assessment you disagree with?

Have you considered putting your approach into a popular package so that researchers can quickly and easily compare their models?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work performs an extensive benchmark for 1000 models to compare the error rate disparity and accuracy trade-offs. To make a fair comparison, the constrained models, either trained with pre-processing techniques or in-processing learning constraints, are unprocessed to yield the corresponding optimal unconstrained model. Through these assessments, the authors convey a straightforward yet crucial finding: achieving fairness is best attained by training the most effective unconstrained model available and subsequently employing post-processing techniques to fine-tune the thresholds.

### Strengths
- I like the way the authors pose the narrative of this work. The structure is well-defined, presenting experimental details clearly. 
-  I think the concept of "unprocessing" is a novel and effective method to discover the optimal unconstrained model corresponding to the constrained models.
- In general, the evaluation is solid and can provide enough insights to the practitioners.
- In my personal opinion, this paper satisfies my standard of acceptance but does not reach the rating of 8. So I would rather recommend a rating of 6.

### Weaknesses
- I would like to see a comparison between the real unconstrained model and the unprocessed version of the constrained model. This comparison is necessary and could enhance the claim that unprocessing can be applied to find the optimal unconstrained model.
- Section 4 is just a standard LP problem in solving Equal Odds with post-processing. It is not novel and there is no need to write down it in the main paper.
- The author has admitted that their evaluation is only applied to tabular data, with a focus on 5 different partitions of the FolkTables dataset. It would be interesting to see how the conclusions can still be generalized to tasks with rich representations.

### Questions
- How efficient is it to solve the LP problem? Can I just exhaustively search all the combinations of the thresholds and plot the Pareto frontiers of the fairness-accuracy trade-offs?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
There have been many proposals in the recent literature to train fair ML models. This paper evaluates thousands of such models, and finds that a simple postprocessing technique achieves the fairness-accuracy Pareto frontier.

### Strengths
This type of comprehensive benchmarking of thousands of models adds a ton of value to the algorithmic fairness literature. I think the result that a simple postprocessing step achieves the Pareo frontier is very significant. I applaud the authors for taking on this task.

### Weaknesses
None

### Questions
None

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper considers the relation fairness-accuracy tradeoff. In particular, the paper considers the relation between fairness (in terms of Equalized Odds) violation and accuracy of the predictor, before and after "unprocessing", and claims based on empirical observations that any Pareto-optimal tradeoff between accuracy and empirical EOdds violation can be achieved by postprocessing.

---

**Post-rebuttal**

The authors claim that Theorem 5.6 of Hardt et al. (2016) strengthens the result of empirical studies considered in the work. It would be helpful if such discussion can be incorporated in the manuscript to help readers understand this connection. After engaging with authors and going through comments by other reviewers, I have increased my evaluation from 5 to 6.

### Strengths
The strength of the paper comes from the extensive empirical experiments and the efforts to present the observation (that Pareto-optimal tradeoff can potentially be achieved by postprocessing. The experiments are conducted on a relatively new data set (compared to standard baseline data sets in the literature), and the setup includes exact and relaxed EOdds (Hardt et al., 2016).

### Weaknesses
The weakness of the paper comes from the lack of a certain level of theoretical derivation to justify the empirical findings. The proposed term "unprocessing", as noted by authors, "roughly corresponds to the inverse of postprocessing", is more of less confusing (for reasons detailed in Section __Questions__). While one can observe from extensive empirical evaluations that Pareto-optimal tradeoffs can be achieved (setting aside numerical indeterminacy), there is a worry that the results can only provide limited insight regarding the not-clearly-motivated unprocessing procedure.

### Questions
__Question 1__: what is the exact relation between unprocessing and postprocessing?

Based on Hardt et al. (2016), the postprocessing strategy for EOdds is trading off True Positive Rates (TPRs) and False Positive Rates (FPRs) across different demographic groups. Such procedure is _oblivious_, in the sense that only the joint distribution $(A, Y, \hat{Y})$ are utilized in the postprocessing procedure. If this specific way of postprocessing is of interest in the paper, I am not sure how to understand the relation between unprocessing and postprocessing. I can see why authors draw an analogy between unprocessing and the inverse of postprocessing. According to Equation 1, unprocessing starts from the postprocessed $\hat{Y}$ and aims to find the unconstrained optimized predictor. How can we do that with obliviously postprocessed $\hat{Y}$? How to make sure the unprocessed predictor has a sensible mapping from input features to target variable?



__Question 2__: regarding the claim that _any_ Pareto-optimal tradeoff can be achieved by postprocessing

Follow up to Question 1, if the postprocessing is defined as in Hardt et al. (2016), it would be very helpful if authors can provide a clear characterization of the relation between unprocessing and such definition of postprocessing, so that readers can understand why unprocessing is a helpful analyzing tool to understand the importance of postprocessing. Empirical evaluations can be strengthened by some certain level of theoretical analysis to make the results and message more convincing.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
