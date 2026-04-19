# Exploring the cloud of feature interaction scores in a Rashomon set

- Decision: Accept (poster)
- Scores: 6, 6, 6

## Abstract
Interactions among features are central to understanding the behavior of machine learning models. Recent research has made significant strides in detecting and quantifying feature interactions in single predictive models. However, we argue that the feature interactions extracted from a single pre-specified model may not be trustworthy since: *a well-trained predictive model may not preserve the true feature interactions and there exist multiple well-performing predictive models that differ in feature interaction strengths*. Thus, we recommend exploring feature interaction strengths in a model class of approximately equally accurate predictive models. In this work, we introduce the feature interaction score (FIS) in the context of a Rashomon set, representing a collection of models that achieve similar accuracy on a given task. We propose a general and practical algorithm to calculate the FIS in the model class. We demonstrate the properties of the FIS via synthetic data and draw connections to other areas of statistics. Additionally, we introduce a Halo plot for visualizing the feature interaction variance in high-dimensional space and a swarm plot for analyzing FIS in a Rashomon set. Experiments with recidivism prediction and image classification illustrate how feature interactions can vary dramatically in importance for similarly accurate predictive models. Our results suggest that the proposed FIS can provide valuable insights into the nature of feature interactions in machine learning models.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper argues that a well-trained predictive model may not accurately preserve the true feature interactions, and multiple well-performing predictive models can exhibit variations in feature interaction strengths. Therefore, they recommend exploring feature interaction strengths within a model class consisting of approximately equally accurate predictive models. The authors suggest exploring feature interaction strengths within a model class comprising models that are approximately equally accurate. The authors introduce the concept of Feature Interaction Score (FIS) within the framework of a Rashomon set. To facilitate the calculation of the FIS within this model class, they present a practical algorithm to calculate FIS. FIS is a straightforward and heuristic method, but it is still novel to me.

### Strengths
1. The originality of this paper is great. The problem is clearly defined.

2. The proposed method is technically sound to me.

3. The definition of FIS is a novel and reasonable tool to analyze feature interactions intuitively.

### Weaknesses
1. The overall presentations need to be improved. Many figures do not have the axes' labels. 
2. The experiments should be conducted on a broader range of datasets.

### Questions
1. Is your work the first one to propose to use the loss change to measure the strength of feature interactions?
2. From my perspective,  feature interactions typically don't have a definitive "ground truth." Therefore, how can you prove the superiority of your Feature Interaction Score (FIS) compared to other baselines without directly comparing it to a ground truth?
3. What is MCR in Table 3? 
4. What is the range of mask $m_i$? Mask values are used to be 0-1. However, in your algorithm, they seem to be $m_i\in \mathbb{R}$.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents to explain the feature interactions in a model class. Two visualization tools are developed to analyze the feature interactions.

### Strengths
1. Proposed to look for multiple feature interaction sets based on shapley values.
2. Two visualization methods are proposed for analyzing and visualizing the FIS.

### Weaknesses
1. I  am not fully convinced by the motivation of this paper. i.e., why do we need to explain feature interactions in a model class?
2. The novelty of Shapley value calculation is not well presented. The focus is totally on the Rashomon set side.

### Questions
N.A.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes feature interaction scores (FIS), where it characterizes feature interactions based on a model class (which achieve similar performance for a task "Rashomon Set") instead of a single model only. Complementary to this, the paper introduces an algorithm that can be used to compute FIS based on Rashomon sets, and a "Halo" plot to visualize the said feature interactions.

Overall, the paper is well written and makes a significant contribution. I have a list of clarifying questions/comments. I would be able to make better analysis of the results based on the discussion. I look forward to author's responses.

### Strengths
1. The paper is very well-motivated
2. The proposed method is creative
2. The paper is very clear and easy to read (a few parts need some clarifications; see "Questions"), I congratulate the authors for their clarity of presentation.
3. Makes an important contribution.

### Weaknesses
1. The greedy algorithm (one of the main contributions) is unclear and difficult to follow (but can be clarified during discussion)
2. Halo plots are new, and need more discussion/explanation.

### Questions
1. In the example where the paper considers roots of a quadractic equation, it is unclear what "the input variables a, b, and c exhibit different feature interactions in these two models" means. Are the "two models" the two roots?  Further explanation is required, it is confusing what the readers are asked to infer.

2. Consider adding a sentence to connect eq (6) to (2), currently it is a sentence fragment. 

3. \mathcal{M} is not defined in (7) (its first use).

4. How is eq (6) connected to (7)? It seems that the write-up connecting these two was skipped.

5. Is there a reason to define and call the function in (5) as g(.)? Since the write-up so far uses f(.) for the predictive fuction, can the paper use f(.) instead of g(.)? The rationale is that in (8) the authors define the Rashomon set using g, while (1) uses f. Further, this change means that \mathcal{M} can be defined in (1), and can improve the flow of the paper.

6. What does the paper mean by "then inversely calculate any order of interaction by decomposing..." at the end of 3.2.1.

7. Notation \mathbbm{1}{i=1}^{p} will be more appropriate than $(1)_{i=1}^{p}$ in Algorithm 1. But more importantly, it is unclear how m+ is different from m-. They seem like a vector of size p of ones, i.e. both are the same. If so, then what is the difference between m_i+ and m_i-?

8. Algorithm 1 is somewhat difficult to follow. Since this is the key contribution, can the authors explain the role of various terms such as "learning rate", \phi_s etc. are?
Editing the algorithm to have comments, or updating 3.2.2 to reflect the terms in Alg. 1, can help. For instance, I am still unclear how the computations are taking place over multiple models, and if this is being influenced by "learning rate" somehow. 

9. Can the authors explain what they mean by "In theory, the joint effects of features should not exceed the boundary when there is no feature interaction."? Mathematical formulation can help. 

10. Which functions is depicted in Fig. 3? We see x_0, x_1, and x_2, but it is unclear what was the original functional form.

11. What does "*" refer to in computational time section?

12. Fig. 4 would benefit from axis labels, and needs to be referred to in the write-up.

13. Halo plots -- the blue curves for a fixed \epsilon account for all \phi_{i,j} (in the 2D case), but what is the x and y axis supposed to denote? Do the negative and positive values on the y axis carry meaning? Why does the x axis not have any axis values?

14. The term "MCR" is undefined.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent
