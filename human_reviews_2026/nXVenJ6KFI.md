# Identification of Task Affinity for Multi-Task Learning based on Divergence of Task Data

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Multi-task learning (MTL) can significantly improve performance by training shared models for related tasks. However, due to the risk of negative transfer between mismatched tasks, the effectiveness of MTL hinges on identifying which tasks should be learned together. In this paper, we show that for tabular datasets, this affinity between a pair of tasks can be predicted based on static features that characterize the relationship between the datasets of these tasks. Specifically, we show that we can train a regression model for predicting pairwise task affinity based on computationally efficient features, requiring ground-truth affinity values for only a small, random sample of task pairs to generalize across all possible pairs. We demonstrate on three benchmark tabular datasets that our proposed approach can predict affinity more accurately at lower computational cost than existing methods for identifying task affinity, which treat task data as black boxes and require training-based signals. Our work provides a practical and scalable solution to task grouping for MTL, enabling its effective application to tabular datasets with large numbers of tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a feature-based approach for predicting pairwise task affinity in multi-task learning (MTL) for tabular datasets.

### Strengths
The paper includes evaluation on three different benchmark datasets with different characteristics, and the experiments cover multiple aspects including prediction accuracy, computational cost, and downstream task grouping performance.

### Weaknesses
1. The biggest limitation is that this approach only works for tabular datasets with shared input dimensions. The authors acknowledge this (line 64-: "We assume a common input dimension p across all tasks"), but this severely limits applicability. Most interesting MTL problems in computer vision or NLP don't have this property. The paper should discuss more clearly when this assumption holds and provide examples beyond the three benchmarks tested.
2. The paper is quite empirical. While the hypothesis that "tasks with more similar data distributions benefit from joint training" is intuitive, there's little theoretical justification for why these specific features should predict MTL gains. Some analysis connecting feature values to properties that affect gradient-based optimization or representation learning would strengthen the work.
3. Some parts are repetitive (e.g., the motivation for feature-based prediction is stated multiple times). Some experimental details are unclear (e.g., how exactly is the train/test split done? Are results averaged over multiple random splits?)
4. Table 4: "Std Dev(σ)", why not just σ?
5. In Table 3, why does the optimal number of groups k vary so much (2-15)? How sensitive is the final performance to choosing k?
6. Can you provide more intuition or theoretical justification for why these specific features should predict MTL gains? For instance, why should Energy Distance be particularly predictive?

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an approach to predict task affinity in multitask learning (how beneficial it is to train two tasks together), based on static, precomputed features of the task datasets, without requiring expensive joint training for all task pairs. 
- Instead of treating task relationships as a black box, the paper quantifies statistical and structural similarities between tasks using easily computed dataset-level metrics. These include measures of dataset size, input-space distances, distributional divergence (e.g., energy distance, feature mean gaps), and representation similarity (e.g., cosine similarity, PCA alignment). 
- The authors construct pairwise feature vectors from these metrics and train a quadratic regression model to predict MTL gains, defined as the relative improvement in task performance when trained jointly versus independently. Crucially, the model is trained only on a small subset of task pairs with known ground-truth MTL gains. 

The framework was tested on three standard tabular MTL benchmarks, School, Chemical, and Landmine datasets. The proposed model outperformed prior task-affinity estimation methods, such as TAG (gradient-based affinity estimation) and GRAD-TAE (gradient-projection model), in both prediction accuracy and computational efficiency. For example, the proposed method achieved correlations up to 0.58 with true MTL gains on the Landmine dataset, while requiring only a fraction of the training time needed by baselines. Moreover, when applied to task-group selection using beam search and semidefinite programming clustering, the predicted affinities led to superior task groupings with lower total loss compared to alternatives.

### Strengths
- This work contributes a scalable and interpretable method for predicting task affinities using simple dataset-derived features, reducing the computation for MTL training. 

- It provides a practical pathway to automatic task grouping, particularly for tabular data with many tasks. The authors’ findings support the hypothesis that tasks with more similar data distributions yield more positive transfer, offering both theoretical and empirical validation. 

- The approach advances MTL research by improving efficiency and accuracy in task affinity prediction, enabling large-scale applications of multi-task learning.

### Weaknesses
- The framework focuses on tabular data and extracts dataset features based on tabular features. It would be better to understand how this framework can be applied in a generic setting, for example, in the scenarios of deep neural networks & image/text datasets. 

- The method predicts pairwise task affinity. Would it be possible to extend the framework to predicting higher-order task affinities, such as the multitask learning results when training on more than two tasks? 

- Is there any theoretical analysis on the prediction accuracy of using a linear model on the extracted features?

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper predict task affinity for multi-task learning using tabular dataset statiscis and static features can characterize the reslationship between the tasks without pair-wise exhasutive joint training as previous works.

### Strengths
1. The work avoids the combinatorial cost of training on all task pairs, enabling scalable and inexpensive estimation of pairwise affinities for large task sets.
2. The method efficiently identifies high-performing task groups.

### Weaknesses
1. The method is validated only on tabular MTL, so it is unclear whether the findings transfer to vision or NLP. Evaluations on standard vision multi-task benchmarks (e.g., Taskonomy) are needed to establish external validity.

2. Task similarity can evolve during training, but the proposed feature-based metric is essentially static and costly to refresh. Prior work such as Selective Task Group Updates for Multi-Task Optimization [1] and GRAD-TAE [2] can track changing task affinities without exhaustive joint training.

[1] Selective Task Group Updates for Multi-Task Optimization (ICLR 2025)

[2] Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity (ACM SIGCOMM 2024)

3. The estimator uses a hand-picked subset of features. When indicators disagree, the decision rule and its justification are unclear. A more principled aggregation is needed to show that dependence on such statistics is appropriate.

### Questions
Please respond to these concerns: external validity beyond tabular data, including results on standard vision MTL benchmarks such as Taskonomy, and how your approach handles non-stationary task relations compared to methods that track changing affinities without exhaustive joint training. Also clarify how conflicting feature signals are resolved, provide justification for relying on simple statistics.

### Soundness
3

### Presentation
2

### Contribution
2
