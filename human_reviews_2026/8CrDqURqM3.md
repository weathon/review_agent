# Performance-Enhanced Aggregated Representation Learning

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Machine learning models rely on representation learning to transform complex data into a simpler and more structured form for analysis. However, the performance of these models depends on how well the data is represented, and no single method can consistently capture all the important information needed for downstream tasks. This study introduces a generalized model averaging framework that integrates multiple representation learning techniques, enabling researchers to combine advanced machine learning models to enhance downstream task performance. By assigning optimal weights to different representations, the proposed approach not only achieves numerical efficiency but also provides theoretical guarantees, ensuring its optimality in real-world scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the aggregation of representations of raw input data. They propose a method to combine the different represantations and the possible sets of representations that can be extracted from the original representations. Normally, one can start with finding the optimal weights by minimizing the loss function between the actual labels and the predicted labels, where the predicted labels are coming from the models that are trained on the entrire labeled dataset on top the specific representations. However, instead of doing this the authors suggest that when we use predicted labels where the prediction comes from K-fold cross validation (basically the model that makes the prediction on a sample should not be trained on this sample), the overall performance improves. It is also suggested that instead of the actual loss function using a surrogate loss function can enhance the smoothness and and induce convexity.

### Strengths
This work focuses on very generic setup of aggregation of different representation learnings and it shows that PEARL can be used for variety of machine learning projects that involve LLMs as well as classical machine learning methods such as linear models, decisiion trees etc. as long as the representations of the raw data can be found.

Even though very similar methods are being used in machine learning competitions, the authors are suggesting that they prove PEARL reaches optimal performance on downstream tasks under certain conditions.

Authors also compare their results with different aggregation methods on several datasets as well as a synthetic dataset, so that they also provide experimental results in addition to theoretical analysis.

### Weaknesses
Even though having analysis on specifically multiple representations can be new, very similar methods which involves k-fold cross validation based weighting in the ensembling phases are commonly used in machine learning competitions. As far as I understand, the very high level idea of this paper is this : "When it comes to the combining features or ensembling of different predictions, it is better to find weights based on K-fold based predictions -not training a model and use its own predictions of the same saples-". So the core idea is not super-new.

### Questions
1 - I wanna learn the difference between the M representations from FRL models and J candidate representation sets. So, I do understand that we have M model representations and we can extract J different candidate representation sets from these M representations. However, theoretically, do we really need to mention that we have M representations initially ? We can basically think of we have J different representations where these representations can have similarities/common-subvectors without even mentioning the initial M models right ? I wanna make sure that I do not miss if this method exploits the information that these J candidate sets are constructed from M model representations.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes the following ensembling method:
1. Train a bunch of encoders using different techniques
2. Combine the dimensions of these trained embeddings. Each combination is a candidate
3. Train a predictor for each candidate
4. Obtain linear weights for the candidates using e.g. cross validation

The authors test the method on synthetic datasets, CIFAR-10, and some tabular datasets

### Strengths
1. The writing is fine. It is not particularly hard to understand the paper
2. Learning the linear weights using cross validation seems reasonable

### Weaknesses
This manuscript does not have enough novelty and sophistication to reach bar of ICLR. While it might be sufficient for a course project, I don't think it should be accepted by a top venue like ICLR.

The proposed method is essentially linearly ensembling a bunch of models, where the linear weights are obtained via cross valiation. All of these are pretty standard. I feel the only thing that might be novel is the way the method chooses the candidates, which is by combining the dimensions from different encoders. However, there are two problems:
1. From Section 2.2, it is not clear to me how the candidates should be chosen. Obviously one cannot choose every possible combination, for that will give us an exponential number of candidates. But if we use a subset, it is not clear how this subset should be chosen. Note that for a deep embedding, multiplying it by an invertible matrix typically does not change the representation. This means that if we choose one subset, then by multiplying a permutation matrix, we end up in another subset. It is also not clear to me how the "domain knowledge" can be used in choosing this subset.
2. It is not clear why we should choose a subset of dimensions at all. Equivalently, one can use all dimensions of the embeddings, and multiply it by a masking matrix that masks certain dimensions to zero. In this case, why not learn this masking matrix, when learning the downstream predictor? What does this seemingly arbitrary way of masking certain dimensions buy us?

The theoretical analysis presented is very elementary, and the experiment section does not show strong evidence that doing what the authors suggest would help in real use cases. The experimental results are very weak. For example, in Table 2, why should any one care if the accuracy on CIFAR-10 is only 62%?

In conclusion, I recommend rejecting this submission.

### Questions
N/A

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
This paper proposes a new method called Performance-Enhanced Aggregated Representation Learning (PEARL). The main idea is to combine multiple representation learning models to improve performance on downstream tasks (like classification or regression). Instead of picking just one model, PEARL uses a weighted average of predictions from many candidate models. The weights are chosen using cross-validation and a surrogate loss function to make computation easier. The authors provide theoretical guarantees showing that PEARL performs as well as the best possible combination of models and can identify the best models in the candidate set. They test PEARL on several datasets (image, text, and multi-modal) and show it often works better than baseline methods.

### Strengths
1. The work appears technically sound. The method is described in detail, and the authors support their claims with theoretical proofs (asymptotic optimality and weight consistency) and extensive experiments on synthetic and real-world datasets.

2. The paper is generally well-written and organized. The two-stage process of PEARL is clearly explained, and the algorithm is presented step-by-step. The examples help illustrate how candidate representations are constructed.

### Weaknesses
1. The novelty seems fair. The core idea of combining multiple representation learning models is not well motivated. The explanation in Lines 37-40 does not adequately elaborate on the motivation behind this approach.

2. The experiments, while varied, could be strengthened by including a comparison with more recent or sophisticated ensemble/aggregation methods beyond simple averaging and model selection. This would better situate PEARL's performance within the current landscape.

3. The theoretical results rely on several assumptions. The practical validity or verifiability of these assumptions for real-world data and complex models is not discussed, which is crucial for understanding the real-world applicability of the guarantees.

### Questions
The introduction should more clearly elaborate on the necessity of combining multiple representation learning models.

### Soundness
3

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
2

### Summary
This paper introduces PEARL, a general framework that combines multiple representation learning methods to improve downstream task performance. The authors propose a two-stage model averaging approach—first training diverse foundational representation learning (FRL) models, then aggregating their outputs using cross-validated, surrogate-loss–based weighting for computational efficiency and theoretical guarantees of asymptotic optimality and weight consistency. Experiments verify the proposal's performance.

### Strengths
1. The paper is clearly structured and well written, with a logical flow from motivation to theoretical results and experiments.
2. The paper presents a novel and well-motivated framework, PEARL, that integrates multiple representation learning methods under a unified model-averaging paradigm.

### Weaknesses
1. Training cost: The computational cost deserves further discussion. Since the selection process relies on K-fold cross-validation, the overall training expense may exceed that of conventional single-model approaches.
2. Efficiency evidence: Although the authors claim that PEARL achieves computational efficiency through the use of surrogate losses and linear weighting, the paper provides limited empirical or complexity analysis to substantiate this claim. Including runtime comparisons or scalability analyses would strengthen the argument.
3. Novelty and related work: While the paper presents a theoretically grounded and flexible aggregation framework, the central idea of combining multiple representations via weighted model averaging is conceptually related to established ensemble and model averaging methods [1, 2]. The authors could enhance the originality claim by clarifying how PEARL differs from prior unified model-averaging frameworks—particularly whether its surrogate-loss formulation or asymptotic guarantees contribute fundamentally new theoretical insights.

[1] Least Squares Model Averaging (LSMA)
[2] Model Averaging Prediction by K-fold Cross-Validation

### Questions
I wonder whether this method could be extended to large language model (LLM) settings rather than being limited to small-scale scenarios. If the framework can guide the selection of suitable training recipes or adaptation strategies for LLM agents across different tasks, its impact would be even more substantial.

Note: I am not deeply familiar with this specific research area, so my evaluation may carry some bias. I will adjust my final score based on insights from other reviewers.

### Soundness
2

### Presentation
3

### Contribution
3
