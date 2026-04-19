# Travelling Salesman Problem Goes Sparse With Graph Neural Networks

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 3

## Abstract
Machine learning based approaches to solve the Travelling Salesman Problem (TSP) have achieved astonishing performance in the last years.
A large number of works proposing such approaches use a type of encoder in their underlying frameworks to learn vector representations of the given problem.
Since TSP can easily be interpreted as a graph theoretic problem, Graph Neural Networks (GNNs) have been a popular encoder architecture for this task. 
However, most papers ignore that GNNs are not designed to operate on complete graph instances like the TSP.
We therefore propose two data preprocessing methods for GNNs to make the TSP instances sparse: a nearest neighbor based heuristic and a method based on minimum spanning tree called 1-Tree.
We show that making the underlying TSP instances sparse by deleting unpromising edges in the preprocessing step improves the performance of the overall learning framework while, at the same time, the runtime decreases. 
In particular, the proposed method achieves an up to $\times 2 $ performance improvement w.r.t. the optimality gap and a decrease in runtime by 10\% during training and validation, when applied to GCNs. 
For GATs, the improvements in regards of runtime and optimality gap are even bigger when sparsifying the data first: We report up to $\times 22$ improvements for the optimality gap while reducing the runtime by 50\%.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to adopt graph sparsification techniques in the data preprocessing step for the GNN-based TSP solvers. Experiments demonstrate its effectiveness in both aspects of improving the quality of solutions and the running efficiency.

### Strengths
- The methods are straight-forward and easy to follow.

### Weaknesses
- Important baselines are missing. The experiments are more like ablation studies: The authors investigate the performance of GAT/GCN-based solvers with or without the proposed graph sparsification techniques, but does not compare the performance with other solvers mentioned in the related works. To make the contributions strong enough and the results convincing, the paper should compare with the latest methods and outperform them.

- The examples in Figure 1 do not make sense. Message passing in GNN does not propagate the features directly, but with a projection matrix (e.g. GraphSAGE). Furthermore, the problem of over-smoothing of GNN not only exist in complete graphs but also in general graphs [1]. How the proposed graph sparsification technique relieve the problem should be more clearly discussed.

- The covered problems only include the 2D tsp, which limits the the contributions of the proposed techniques.

- It lacks necessary theoretical analysis.

- The proposed 1-tree sparsification method is derivated from LKH which is a very strong TSP solver. Then the use of the technique in data preprocessing indeed brings prior knowledge to the neural solver. It is very hard to say that whether the better performance comes from the graph sparsification, or comes from the prior knowledge for TSP solving. 

[1] A SURVEY ON OVERSMOOTHING IN GRAPH NEURAL NETWORKS. https://arxiv.org/pdf/2303.10993.pdf

 Based upon the above points, I believe that the work is still somehow preliminary and the paper does not meet the bar of iclr.

### Questions
- The size of instances is not given. 
- The others are in the weakness.

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes two data preprocessing methods for solving the TSP with GNNs, i.e., k-nearest neighbors heuristic and 1-Trees, which make the corresponding TSP instances sparse by deleting unpromising edges. Experiments are carried out to determine the better sparsification method and the relationships between different data distributions/training dataset sizes and sparsification parameter k.

### Strengths
1.	Sparsification (or pruning, candidate selection, etc.) methods are important for solving the TSP as they can substantially reduce the computational complexity, and is commonly used in learning-based algorithms and heuristic algorithms for the TSP.
2.	The paper is overall well written.

### Weaknesses
1.	K-nearest neighbors heuristic is already used for sparsification in the input layer (k=20 for TSP100) of GCN by Joshi et al. (2019), Fu et al. (2021) and Xin et al. (2021b) followed this setting. And the proposed “1-Trees” method is similar to the edge candidate set construction process of the LKH algorithm using the 1-tree structure. Thus, the main contribution of this paper seems to be selecting the proper k of k-nn when the problem size is fixed at 100, and transplanting the 1-tree method of LKH as a data preprocessing procedure for learning-based methods. Therefore, the novelty of this paper is not significant enough.

2.	The problem size is fixed at 100 in the experiments so that the generalization ability of the proposed method over different problem sizes is unclear. I recommend the authors add the following question in section 4: how does the problem size n (amount of cities in one TSP instance) relate to the sparsification parameter k?

3.	Comparative experiments with state-of-the-art TSP algorithms is not provided. It is uncertain whether the “1-Trees” method or changing the hyperparameter k of k-nn in existing methods like Joshi et al. (2019); Fu et al. (2021); Xin et al. (2021b) can enhance the performance of state-of-the-art learning-based TSP algorithms.

### Questions
1. Please clarify the novelty of this paper in comparisons with the literature papers.

2. how does the problem size n (amount of cities in one TSP instance) relate to the sparsification parameter k?

3. Comparative experiments with state-of-the-art TSP algorithms is not provided.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors observe that the sparsed TSP graph with KNN and 1-tree could improve the performance of the GNN-based method and reduce training time. The topic of studying the sparsity of TSP graph is interesting. The observation also seems reasonable. However, the sparsity like KNN has been used by previous work on TSP and VRP. The used 1-tree method was borrowed from LKH. Therefore, I think the contributions are quite marginal.

### Strengths
The observations are interesting.

The experiment design is mostly reasonable.

### Weaknesses
Quite some related works about neural-based methods for TSP and VRP are missing, especially from TOP AI conferences.

The sparsity like KNN has been used by previous work on TSP and VRP. The used 1-tree method was borrowed from LKH. Therefore, I think the contribution are quite marginal.

### Questions
The results of GAT with dense graphs are quite bad, which makes the results less convincing.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper use one-tree for edge elimination for GNN. 
The proposed method achieves an up to ×2 performance improvement w.r.t. the optimality gap and a decrease in runtime by 10% during training and validation, when applied to GCNs. For GATs, the improvements in regards of runtime and optimality gap are even bigger when sparsifying the data first.

### Strengths
1. one-tree based sparsity saves time.
2. The introduction and related work sections are well-written.

### Weaknesses
This paper is very hard to follow.

1. One-tree has been proposed and existed for many year, introducing sparsity to GNN is not a new idea,
see https://arxiv.org/abs/2006.07054

2. weak evaluation, on TSP 100 only.



We employ GNN for TSP with the aspiration of learning promising edges without the need for human-designed heuristics. However, the use of one-tree heuristics already narrows down the edge set. This means the sparsity is largely dependent on human-designed heuristics rather than data-driven ones.

Also, this sparsity is only limited to TSP and is not able to generalize any other problem.

### Questions
1. The table is very confusing, why select different training size? The goal is to investigate how sparsity affect GNN, not training size.

2. How to train your GNN, supervised or reinforcement or even unsupervised? How to get the TSP length? 
My understanding is that the code is using reinforcement learning framework based on Jin et al.
But in Jin et al. The authors report a 0.16\% on TSP-100. They further study TSP random200, TSP random500 and TSPLIB from 1~1002. 
If the paper use the same model, they should evaluate on the same dataset with Jin et al.

3. In the paper ```We summarize that for the GCN, smaller k led to the overall best results, whereas for the GAT there is a tendency for bigger k (but not dense graphs!) to lead to the best results.```, this is more confusing, that means graph sparsifying can be different for different GNN models, then how we decide $k$ when we use a different GNN model? 

4. We report up to ×22 improvements for the optimality gap while reducing the runtime by 50\%.  Can you reveal more details about the training and evaluation, how to get these results?



Jin et al. Deep reinforced multi-pointer transformer for the traveling salesman problem

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor
