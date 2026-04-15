# A Topology-aware Graph Coarsening Framework for Continual Graph Learning

- Decision: Reject
- Scores: 6, 8, 5, 3

## Abstract
Continual learning on graphs tackles the problem of training a graph neural network (GNN) where graph data arrive in a streaming fashion and the model tends to forget knowledge from previous tasks when updating with new data.
Traditional continual learning strategies such as Experience Replay can be adapted to streaming graphs, however, these methods often face challenges such as inefficiency in preserving graph topology and incapability of capturing the correlation between old and new tasks.
To address these challenges, we propose TA$\mathbb{CO}$, a topology-aware graph coarsening and continual learning framework that stores information from previous tasks as a reduced graph. 
At each time period, this reduced graph expands by combining with a new graph and aligning shared nodes, and then it undergoes a ``zoom out'' process by reduction to maintain a stable size. 
We design a graph coarsening algorithm based on node representation proximities to efficiently reduce a graph and preserve topological information. We empirically demonstrate the learning process on the reduced graph can approximate that of the original graph.
Our experiments validate the effectiveness of the proposed framework on three real-world datasets using different backbone GNN models.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work investigates the problem of continual learning on time-stamped graphs. The authors propose  an edge contraction method applied to time-stamped graphs, which compresses the evolving graph to a reasonable size using node representations. At each time step, tasks are learned on the coarsened graph, and the learned representations further assist in the next time step's coarsening process. The paper also proposes Node Fidelity Preservation to retain important nodes and prevent them from being compressed. Extensive experiments have been conducted to demonstrate the effectiveness of the proposed method.

### Strengths
1.The motivation is clear and propsed method is quite-novel. Paper is well-written and easy to follow.
2.The experiments are thorough and comprehensive. Recently published methods have also been included in the comparisons.

### Weaknesses
1.In the first section, two issues related to CGL are mentioned: the problem of changing class distribution and the issue of correlation between old and new tasks. However, the subsequent chapters do not seem to clearly explain how coarsening addresses these two problems.
2.A comparison of the running time with other CGL methods is not provided.

Minors:
Final average performance is referred to as average performance.

### Questions
1.How does TACO address the problems raised in section 1?
2.How does TACO perform in terms of time and memory compared to previous rehearsal-based methods?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a framework called TACO (Topology-Aware Coarsening) for continual graph learning (CGL). The primary focus is on addressing the challenges of catastrophic forgetting and inefficiency in graph neural networks (GNNs) when dealing with streaming graph data. TACO aims to preserve topological information from previous tasks in a reduced graph, which is then used for training on new tasks. The paper also introduces a graph reduction algorithm called RePro, designed to efficiently reduce the size of the graph while preserving its topological properties. The authors validate their approach through experiments on real-world datasets using different GNN backbones.

### Strengths
- **Significance:** The paper addresses a significant gap in the literature by focusing on continual learning in the context of graph data, which is less explored compared to Euclidean data like images and text. The proposed framework achieves SOTA results on three different online datasets. The author also included detailed ablation studies on the proposed fidelity preserving and important node sampling.
- **Originality**: Using a coarse-grained representation as a buffer for continual learning is very interesting and novel.
- **Clarity**: The method explanation is very clear, and there is a nice visual representation and pseudocode to illustrate the method.

### Weaknesses
- Additional experiments comparing the computational efficiency (training time and inference time) of TACO with other methods could strengthen the paper.
- The paper primarily focuses on academic datasets for validation. It would be beneficial to see how TACO performs in more practical applications such as recommendation systems.
- I am a little worried that the hyperparameters for baseline methods are not well-tuned. For example, the optimal learning rate and batch size could be different for different baselines.

### Questions
- The author argues that using majority voting to assign a class label to a super node would result in the gradual loss of minority classes. However, in my opinion, a simple solution to this issue is to treat the super node label as a soft label, allowing for the distribution of labels within the super node. Would this approach still have the same issue in terms of the representation power? I hope the author could clarify a bit on this point.
- Will the proposed method be affected by the time interval of the dataset?
- The author utilized average forgetting as the metric, which is defined as the maximum forgetting observed thus far. It would be beneficial if the author could also compare long-term forgetting and short-term forgetting with the baseline. This comparison is relevant because various continual learning practices may have distinct requirements.

**Minor comments**:

- In Appendix A, "GraphCoarseninAlgorithm" should be "GraphCoarseningAlgorithm".

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduce a graph-coarsening based method for continual graph learning. To avoid the forgetting problem, the proposed method would reduce the learnt graph part by graph coarsening and connect it to the new task graphs, so that the learning on new tasks would also take old information into consideration. The main part of the paper is the coarsening algorithm, while the other part largely follow existing works.

### Strengths
1. continual graph learning is a more practical than static graph learning setting, and should be paied more attention.
2. The proposed method outperforms the baselines.

### Weaknesses
1. The literature review is very incomplete.
2. Due to point 1, it is hard to position the paper against the literature and find out what is the contribution.
3. The dataset splitting is problematic.

### Questions
1. It is problematic to claim that the existing work focus on task-incremental-learning and sequential tasks are independent graphs. First, many benchmark works including [1,2] study the class-incremental-learning. Second, not all continual graph learning works study the independent graphs. For example, [3] does not split the growing graphs into independent graphs, [4] may also be a related work. Similar works are abundant and the authors are enouraged to do a thorought literature review.  

2. The datasets are split according to time, then how do the classes increment? Does each time stamp necessarily contain new classes?

3. According to Table 3 in the appendix, the number of classes are very small. For DBLP and ACM, 4 classes are used to construct 10 tasks. Then many tasks are actually containing same classes, and it is not a class-incremental situation with large distribution gap. 


4. Is the proposed method related to graph pooling methods? Graph pooling methods seem to be able to maintain a low computation burden while considering the node features at the same time.

[1] Carta, Antonio, et al. "Catastrophic forgetting in deep graph networks: an introductory benchmark for graph classification." arXiv preprint arXiv:2103.11750 (2021).

[2] Zhang, Xikun, Dongjin Song, and Dacheng Tao. "Cglb: Benchmark tasks for continual graph learning." Advances in Neural Information Processing Systems 35 (2022): 13006-13021.

[3] Feng, Yutong, Jianwen Jiang, and Yue Gao. "Incremental Learning on Growing Graphs." (2020).

[4] Das, Bishwadeep, and Elvin Isufi. "Graph filtering over expanding graphs." 2022 IEEE Data Science and Learning Workshop (DSLW). IEEE, 2022.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
TACO tackles graph continual learning (node classification) under streaming (time-stamped) graph settings. As a rehearsal-based method, it preserves graph topology information and captures the correlations between two tasks. To preserve graph topology information, a graph coarsening algorithm based on Node Representation Proximity (RePro) serves as the main component in the framework. RePro leverages node embedding from the first GNN layers and calculates the cosine similarity for connected nodes and merges nodes with high similarity, which satisfy the requirements of feature similarity, neighbor similarity and geometry closeness.  It also preserves representative nodes in the buffer for keeping important path information. Using GCN as backbones, it demonstrates the effectiveness by showing better experimental results using CL methods such as regularized-based methods (EWC,TWP…) and rehearsal-based methods(ER variants) on three datasets. It also provides experimental comparisons between different graph algorithms.

### Strengths
* Combining graph coarsening into task-incremental online graph continual learning is novel.

* Have very detailed algorithms and theoretical analysis.

* The experiments are comprehensive and the ablation study is provided.

### Weaknesses
* The introduction section contains inaccurate information that it is insufficient to categorize existing common CGL methods into only two categories: regularization-based and rehearsal-based in the Introduction section. Parametric isolation-based methods should also be mentioned.(Similarly issues in the related work)  Moreover, putting a Kindle e-book co-purchasing network showcase in the Intro sections is unnecessary (it’s just an example class incremental setting) and has no logistic connection for existing problems in CGL.

* The problem setting is unclear. One natural way for class-incremental setting is the graph expanding setting(new class and new nodes come in), causing distribution shifting. However, in this problem setting, we only observe subgraphs for each task. Does it mean pre-existing nodes can disappear? This setting seems unrealistic. 

* The explanation of merging supernodes is not clear. How exactly do two nodes merge? 

* This method also stores representative nodes but the strategies it uses: Reservoir Sampling: randomly sampling; ring buffer: FIFO manner and MoF. The claim of representative nodes could not hold. Which one used in your experiment is not stated.

* Compared with Var.neigh/edges, experiment gains for RePro are marginal

### Questions
GCN only works for graphs with fixed numbers across tasks. Choosing this model as a backbone assumes we already know how many nodes are in all tasks. This is not realistic in a continual learning setting.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
