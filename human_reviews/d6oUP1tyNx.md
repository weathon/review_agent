# The KNN Score for Evaluating Probabilistic Multivariate Time Series Forecasting

- Decision: Reject
- Scores: 3, 1, 5

## Abstract
Time series forecasting is a critical task in various domains. With the aim of comprehending interconnections and dependencies among variables, as well as gaining insights into a range of potential future outcomes, probabilistic multivariate time series forecasting has emerged as a prominent approach. The evaluation of models employed in this task is crucial yet challenging. Comparing a set of predictions against a single observed future presents difficulties, and accurately measuring whether a model correctly predicts dependencies between different time steps and individual series further compounds the complexity. We observe that metrics which are currently employed fall short in providing a comprehensive assessment of model performance. To address this limitation, we propose a novel metric based on density estimation as an alternative. We showcase the advantages of our metric both qualitatively and quantitatively, underscoring its effectiveness in assessing forecast quality.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes the KNN score for evaluating multivariate time series forecasts. The paper first gives a broad overview of multivariate distributional forecasting metrics and identifies their failure modes. It then proposes the KNNS metric, motivated by the fact that out-of-sample likelihood would be a good score. The KNNS metric is then compared to baselines on synthetic and small-sized real world scenarios. 

However, the proposal appears to move from an incorrect premise. The metric itself appears rather complicated, difficult to implement in practice, and requires the correct tuning of hyperparameters--the choices of which may be data and model dependent. The experimental setup is not nearly enough to substantiate the introduction of a new forecast evaluation metric, and the results fail to demonstrate conclusive evidence.

### Strengths
The paper is well-written overall and adresses an important problem. It gives a broad overview and critique of previous evaluation methods used in other papers and identifies their failure modes. After introducing KNNS, it demonstrates an interesting link between random projections, energy scores and CRPS. The authors then use this insight to make the otherwise very compute-intensive KNNS metric somewhat manageable (computationally, albeit not statistically) in practice. 

The paper also offers good insight about how the statistical power of a forecast evaluation metric was evaluated in prior literature, and proposes a new and thoughtful experiment setup. The authors are clear in their conclusions and about the limitations of their work.

### Weaknesses
Firstly, the paper moves from the premise that a metric used for model selection in forecasting must be able to correctly capture a variety of properties about forecasts including how much the forecasts exhibit correlation and temporal regularity. However, I would argue this is only true to the extent that it makes the metric more "sample efficient" with higher statistical power. 

While simple MAD, MSE, CRPS may be oblivious to correlations or regularity of forecasts, they reflect the key desiderata of the forecasts stemming from the task: that they are close to the ground truth. As the model is better able to capture correlations, one expects that simple metrics like MSE will also vanish. In other terms, one does not require metrics are highly representative---but that they are consistent and proper. For added complexity of metrics, one should argue that the metrics result in higher statistical power under realistic finite sample constraints and forecasting scenarios (realistic true distributions). Given the inconclusivity of the paper's empirical findings and the lack of a framework for the tuning of KNNS parameter I do not believe this bar has been cleared for KNNS.

My second critique would be on the metric itself. Motivated by kNN density estimation, the authors introduce the L2 distance to the kth nearest neighbor as the model selection metric, despite high dimensionality. Besides the fact that this could result in notoriously high sampling variance depending on the true data distribution, this makes the magnitude of the metric dependent on the size of the sample (in the paper, "ensemble"). i.e., in order to compare two models one would have to compare them on the same number of sampled trajectories or the metric would be invalid otherwise. 

Finally, the KNN score's intuition is that the best model is able to place a forecast close in L2 distance to the ground truth. Note that for a univariate forecast this is equivalent to saying that the k-th best forecast in an ensemble, measured in squared error, has low squared error. Setting k=1, this is equivalent to choosing the model with the "best in hindsight" MSE. In other words, KNNS does not appear to measure the quality of distributional forecasts, a desired property set out in the paper, but only point forecast error. 

Some other points

- The introduction of the random projections could be better motivated. Multiple random projections sampled and with sufficient latent dimension would be justified for only very high dimensionality tasks such as spatiotemporal forecasting in earth sciences, etc. This doesn't appear to be the case with the experiment setup.
- The introduction of KNNS is somewhat counterintuitive. The paper first reports that out of sample likelihood (i.e., perplexity) is a desirable metric; but due to its practicality immediately moves to KNN distance. I believe this link should be substantiated. 
- $d$ is redefined in the paper as 'difference' although interchangeably also used to denote dimensions. The notations $\mathbb{y}$ and $\mathbb{Y}$ (resp. X) also appear to have differing definitions through the paper.

### Questions
Why didn't you consider using the average of the first k distances to decrease variance?

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper explores the challenge of scoring forecasts in the context of multivariate probabilistic forecasting. In response to limitations found in current scoring methods for multivariate distributions, the authors introduce the K nearest neighbor score, which relies on density estimation. Through comparisons with various existing scores on simulated and real-world datasets, the paper demonstrates the advantages of the new score, both qualitatively and quantitatively.

### Strengths
- Studying scoring rules for multivariate distributions holds significant importance in numerous applications. The development of improved scores with enhanced properties is an important topic in machine learning.

- The paper aligns with a recent empirical investigation (Caroll, 2022) that evaluates various scoring rules for multivariate distributions.

- The conducted experiments include both synthetic and real-world datasets.

### Weaknesses
- The paper's contributions are not clearly articulated and appear to be quite brief.
	- The proposed method is introduced in Section 3, but it's unclear whether it qualifies as a proper scoring rule. If it does, it's essential to provide a formal proof.
	- The statement, "We draw inspiration from the Energy Score," raises questions, as the log score and energy score are very different scores. Could you provide a theoretical justification for using the log score to evaluate multiple projections? Note that the log score may not be interpretable in this context, and your projections might not yield meaningful results.
	- Your score's definition involves both a density model and a score. Please clarify this relationship.
	- Please include a reference for your energy score proof in the Appendix.
	- The paper's contributions concerning scoring rules for multivariate distributions are unclear. Additionally, a recent and important reference, "Regions of Reliability in the Evaluation of Multivariate Probabilistic Forecasts" (ICML 2023), appears to be missing.
	- There seem to be various approximations and challenges in implementing your method, such as not enforcing orthogonality of rows and the issue of sampling enough prediction vectors due to the curse of dimensionality. It's unclear how these challenges and design choices affect your proposed score.

	`
- The paper requires significant revisions for improved clarity, mathematical rigor, and notations. 
	- For example, some specific issues include the distinction between p(X) and P in S(P, y) in Section 2, the undefined notation for P and Q in equation (1), and the unclear meaning of \mathbb{X.
	- Section 2.2 mentions "lower case x" without defining it. Please provide a clear definition.
	- The paper mentions "an ensemble of points X" and later "a set of predictions X." Please use consistent terminology to avoid confusion.
	- The notation "$i \in [1, K]$" implies continuity. Please clarify or use appropriate notation.
	- There is an issue in the denominator of expression (8) that needs correction.
	- It's important to distinguish between a score and a metric, as they are distinct concepts. Please provide a clear explanation.

- The statement, "Since all marginals are evaluated independently, certain properties of the distribution are lost," needs further clarification. If the energy score is a proper scoring rule, explain what specific properties are lost and why.

- The statement, "Only the green one mimics dependencies between time steps correctly," is disputable. Having all realizations in the predictive region does not necessarily imply a correct capture of true uncertainty.

- The assertion, "As a result, these metrics should be avoided when assessing probabilistic predictions," needs further elaboration and support. Clarify under what circumstances these metrics should be avoided and why.

### Questions
See weaknesses.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper explores an evaluation metric tailored for probabilistic multivariate time series forecasting, a notable stride within a relatively underexplored domain. It underscores the limitations of existing metrics: CRPS and CRPS-Sum cater to univariate forecasting, Energy Score exhibits insensitivity to correlation differences, and Variogram Score lacks rotation invariance. Pioneering the k-nearest Neighbor (KNN) score grounded in density estimation, the paper eloquently delineates both the qualitative and quantitative merits of the proposed metric.

### Strengths
The endeavor to refine evaluation metrics for multivariate time series forecasting is commendable, particularly as this sphere warrants further investigation. The KNN score, premised on density estimation, is presented as a remedy to the issues inherent in existing metrics, offering a novel perspective that could potentially catalyze advancements within this field.

### Weaknesses
A critical determinant of the proposed metric's efficacy is the selection of the number of neighbors; however, the paper falls short of providing a rigorous justification for this parameter choice. This omission may hinder the metric's practical adoption within the time series community. Additionally, while employing random projection for dimension reduction, the paper lacks a thorough theoretical analysis concerning the impact of this technique, which could potentially undermine the robustness or interpretability of the findings.

### Questions
The KNN method, albeit simplistic in its approach towards density estimation, forms the crux of the proposed metric. How does this method fare when juxtaposed against the Kernel Density Estimation method, especially in terms of accuracy and computational efficiency?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
