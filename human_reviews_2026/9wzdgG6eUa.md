# Neural Multivariate Regression: Qualitative Insights from the Unconstrained Feature Model

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
The Unconstrained Feature Model (UFM) is a mathematical framework that enables closed-form approximations for minimal training loss and related performance measures in deep neural networks (DNNs). This paper leverages the UFM to provide qualitative insights into neural multivariate regression, a critical task in imitation learning, robotics, and reinforcement learning. Specifically, we address two key questions: (1) How do multi-task models compare to multiple single-task models in terms of training performance? (2) Can whitening and normalizing regression targets improve training performance? The UFM theory predicts that multi-task models achieve strictly smaller training MSE than multiple single-task models when the same or stronger regularization is applied to the latter, and our empirical results confirm these findings. Regarding whitening and normalizing regression targets, the UFM theory predicts that they reduce training MSE when the average variance across the target dimensions is less than one, and our empirical results once again confirm these findings.  These findings highlight the UFM as a powerful framework for deriving actionable insights into DNN design and data pre-processing strategies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper leverages the Unconstrained Feature Model (UFM) to derive closed-form expressions for training loss in neural multivariate regression and uses them to analyze two problems: multi-task vs. single-task regression and the effect of target whitening/normalization.
The authors prove that under the UFM approximation, multi-task models achieve lower training MSE than separate single-task models and that whitening or normalizing targets improves performance when average target variance is below one.
Extensive experiments on MuJoCo and CARLA datasets confirm these theoretical predictions, suggesting that UFM offers a tractable analytical framework linking DNN theory and empirical design.

### Strengths
(1) Provides clear, mathematically grounded closed-form derivations for training MSE under the UFM, extending its use beyond classification to multivariate regression.

(2) Establishes interpretable theoretical conditions (via eigenvalue analysis of target covariance) that directly predict when whitening or normalization benefits performance.

(3) Validates theoretical insights with thorough empirical results across multiple architectures and datasets, including reinforcement learning and driving domains.

### Weaknesses
(1) Theoretical analysis is limited to training MSE and does not address generalization behavior or test-time dynamics, weakening practical relevance.

(2) UFM assumptions (infinitely expressive features, linear last layer) are highly idealized; empirical validation does not fully justify their applicability to real networks.

(3) The connection between UFM predictions and observed empirical gaps (e.g., systematic underestimation of MSE) is noted but not quantitatively analyzed or modeled.

### Questions
(1) How sensitive are the theoretical conclusions to the specific regularization parameter λ or to non-L2 forms of regularization?

(2) Could the authors extend the UFM framework to capture generalization error or test loss, beyond fitting training MSE curves?

(3) How do deviations from the UFM assumptions (e.g., limited feature expressivity or nonlinear heads) affect the predicted inequalities between multi-task and single-task models?

(4) The experiments show consistent trends, but can the authors provide quantitative measures (e.g., correlation or regression slopes) comparing theoretical and empirical MSE to assess UFM’s predictive accuracy?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the Unconstrained Feature Model (UFM) in the context of multivariate regression and the effect of whitening of regression targets on training performance. It shows that multivariate models perform better than (multiple) univariate models, depending on regularisation, and that whitening may improve performance.

### Strengths
Paper is written clearly

### Weaknesses
- The terminology used in this paper is misleading, it may lead a reader to think that conclusions are much stronger than they actually are. The paper uses interchangeably the words “multivariate” and “multi-task”, which are very different concepts. This work is about “multivariate”, not “multi-task”, so the term “multi-task” should be eliminated.

- Theoretical contributions are very weak. It is quite obvious that a multivariate model does typically better than a univariate model, because univariate models cannot model dependencies among different components of the regression targets. The effect of regularization is also obvious: if regularization is large, then the loss will not prioritize MSE and therefore MSE will be higher.

- All results are about training loss, rather than test loss, which makes them less impactful.  

- The motivation for studying whitening is not discussed.

- Discussion of related work is also weak. It’s a long list of related papers, but there is no logic for how the results off previous work has led to questions that are supposed to be resolved by this paper.

- Experimental evaluation is limited to a very small scale classification problem.


Minor:

- I disagree with the statement: “by regularizing the features, we are implicitly constraining the internal parameters θ.” You can easily imagine situations in which regularisation of the features has no effect on the parameters (basically, when parameters do not affect the norm of the features, however they can affect everything else).

- I believe there is a factor of 2 missing in Equation (4).

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies neural multivariate regression through the Unconstrained Feature Model (UFM), a theoretical framework that **replaces the nonlinear feature extractor in a deep network with an unconstrained matrix of features**. This enables closed-form optimization over feature and output weights.
The authors use this surrogate to derive analytical formulas for the minimal training mean-squared error (MSE) and apply them to two design questions:
(1) whether multi-task regression outperforms training several independent single-task models; and
(2) how whitening or normalizing the regression targets affects training performance.
The theoretical predictions are then validated empirically on robotic imitation-learning datasets.

The overall takeaway from theoretical statements can be summarized as follows: Under the Unconstrained Feature Model (UFM), training performance in multivariate regression can be completely explained by the interaction between data covariance and regularization strength.
1.	Multi-task models are inherently more efficient and achieve equal or lower training error than training separate models for each output, as long as regularization is not weaker.
2.	Whitening or normalizing target variables helps only when their average variance is small—it improves optimization in that regime but worsens it when targets vary too widely.
3.	Both effects—multi-task benefit and whitening sensitivity—emerge directly from the spectrum (spread) of the target covariance matrix, not from details of network architecture or training dynamics.

### Strengths
The paper comes with a very clear message. The theoretical results are clean and easy to digest.
Also, please take a look at the summary.

### Weaknesses
My main problem with the paper is the incremental contribution. I am not personally familiar with the scope of this area to judge overall what is a useful contribution in multivariate regression and/or UMF.
1. Each theorem is a direct algebraic extension of existing UFM results rather than a new conceptual advance. No new analytical machinery or relaxation beyond the original UFM is introduced. This is still fine, but then, I also don't see the significance of the question asked in our overall understanding of neural multivariate regression.
2. No Generalization Analysis: The work focuses purely on training MSE. Insights about generalization, the main practical concern in multi-task learning, are completely not part of the study. 
3. No quantitative comparison with other potentially tractable models, such as NTK or random-feature regression, which might produce similar qualitative conclusions. 

See also my questions.

### Questions
See weaknesses.

(1) The main question I have to the author is:
"To what extent does the UMF theory cover all there is needed to answer the questions that you raised earlier in Section 1, namely, (a) study multi-task models vs single-task (b) effect of whitening." I am not currently sure if the answers to these questions can be given in terms of the UMF theory, and under what situations the qualitative insights derived via simplifications should hold.
(2) As I see, there are two main simplifications here made. Going from Eq (1) to (2), and Eq (2) to (3). Both are significant simplifications, and very critical to do any analysis. Can you elaborate more on the first simplification from Eq. (1) to (2), where you change the quantity one regularized from the norm of parameters to the norm of features?

### Soundness
3

### Presentation
3

### Contribution
2
