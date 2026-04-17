# An Effective Embedding Approach to Shortest Path Distance Prediction Over Large-Scale Graphs

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4, 2

## Abstract
In graph data management, computing the shortest path distance between any pair of nodes is a crucial and foundational graph operation with numerous practical applications (e.g.,  travel/route planning, community search).
Traditional algorithms for solving this problem face significant challenges in time and space complexities, especially when dealing with large-scale graphs. Worse still, existing learning-based approaches often struggle with low accuracy in predicting intricate graph structures.
To address these issues, this paper introduces a novel Graph Convolutional Networks (GCN)- and Multi-View Deep Neural Networks (MVDNN)-based Distance Embedding (GM-DE) framework, which enables fast and accurate predictions of the shortest path distances. Specifically, based on our proposed pivot and anchor set selection strategies, 
GM-DE enables the calculation of embeddings for each graph node. 
Then, by feeding such embeddings into our designed GCN and MVDNN models, GM-DE can be well trained to support the mining of accurate global and local positional information for the graph nodes with the help of our constructed predictors. 
In this way, our GM-DE framework can achieve high accuracy in various complex scenarios, relying solely on basic node attributes as input without the need for scenario-specific data.
Comprehensive experiments confirm the effectiveness and efficiency of GM-DE approach in predicting the shortest path distances on a wide range of real-world graphs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents GM-DE, a framework for shortest-path distance prediction in large-scale graphs. The method combines local and global embeddings via pivot and anchor set selection strategies and introduces three predictors to integrate these representations.

### Strengths
Clear Motivation and Context.

Experiments cover diverse datasets, demonstrating the method’s generality.

### Weaknesses
The paper is poorly organized and difficult to follow; key terms (e.g., “pivot,” “anchor set,” and “view”) are not clearly introduced or intuitively motivated.

Figure1 is not explained adequately in the text.

Although efficiency and storage are reported, the theoretical and practical scalability of the method (as $|V|$ and $|E|$ increase into the hundreds of millions/billions) is only cursorily discussed and is not validated beyond the YouTube graph (1.13M nodes). 

No experiments or extrapolations are provided for truly web-scale graphs, and memory and time complexities (as well as potential GPU/CPU bottlenecks) are not analyzed in detail. Some recent works in the missing related papers category demonstrate more explicit billion-edge scalability.

### Questions
See Weaknesses.

### Soundness
2

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
4

### Summary
This paper propose a graph neural network (GNN) based approach to tackle the problem of shortest path distance prediction in large-scale graphs. It propose two types pivot/anchor nodes that encode local and global features. The global features are then fused by other layers of network to predict the final distance. Experiments show improvement on predicted results compared to traditional and neural approaches, it also shows improvement on inference time.

### Strengths
The paper is well-motivated, with good context for the reader to understand the shortest path prediction problem. 

The writing is clear and each component of the model is well-explained.

### Weaknesses
The proposed method provides limited contribution to both graph learning research and the shortest path distance (SPD) prediction problem.

- The method essentially adds global, local and random features as the input to a GNN, which is a paradiagm extensively explored in the GNN literature. The author did not discuss the advantage of the proposal and also did not include empirical comparison to these methods. [1, 2]

- The paper only uses GCN as the backbone GNN, whereas stronger GNN models can be just as effective without the added features. And the author did not compared to them.

- The anchor and pivot selection process is heuristic, and provides minimal insight into how selectin them bring advantage to estimiate the SPD.

- The experiment is pure transductive setting. Specifically, the test graph and train graph is exactly the same, and the model is just applied on different node pairs. 1, The GNN can simply remember a lot of features, and apply that during testing stage, without learning any transferrable knowledge. 2, If the graph is dynamic, for example, YouTube graph can change every second, does the trained model apply to the new graph? Do we need to train the entire model again? These unanswered question all limite the usability of the proposed system.

- Following last point, the transductive setting break the entire point to developing a learning system for SPD problem. One can simply do contraction hierarchy computation, and query efficiently. 

- The author mentioned that the training time is within acceptable limit. This a very vague description. To show that the approach is actually useful, the author should compared their method to traditional precompute+querying approach, and show that the total time is superior rather then just inference time, especially when all experiments are conducted in a transductive fashion, and traditional approach is directly applicable.

[1] Hu, Ziniu, et al. "Pre-training graph neural networks for generic structural feature extraction." arXiv preprint arXiv:1905.13728 (2019).
[2] Abboud, Ralph, et al. "The surprising power of graph neural networks with random node initialization." arXiv preprint arXiv:2010.01179 (2020).

### Questions
Please see weaknesses.

### Soundness
1

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
4

### Summary
Traditional algorithms for calculating shortest path distances have high space and time complexity. Existing learning-based methods are inferior in accuracy due to various limitations. This paper combines graph convolutional networks (GCN) and Multi-View Deep Neural Networks (MVDNNs) based distance embeddings and proposes GM-DE framework. Experimental results demonstrate the high accuracy of the proposed GM-DE framework.

### Strengths
1.	This paper proposes an interesting framework to address a fundamental problem in graph data management.

2.	This paper is well written and easy to follow, especially with the well-designed illustration.

3.	The experimental results in terms of time, space, and accuracy are provided and demonstrate the advantages of GM-DE over baselines.

### Weaknesses
1.	The two definitions in Section 3 are trivial. The definition of shortest path estimation (Definition 2) does not involve any requirements on estimation accuracy, which seems informal.

2.	This proposed method is claimed to have advantages in efficiency over traditional methods. However, no analysis on efficiency and space complexity is provided.

3.	The proposed method is verified to be accurate and efficient in five real-world networks. However, this is not convincing for the GM-DE as a universal method for shortest distance estimation. Moreover, the tested networks are not large enough. Real-world large-scale networks contain millions or billions of nodes or edges.

4.	The traditional methods are missing in the baselines, which are important to verify the efficiency and performance advantages of the proposed method.

### Questions
1.	As stated, the distortion is the ratio of embedding distances to the original distances, which is O(log n). I doubt if the embedding distance follows a similar ratio scale. Otherwise, this ratio is large in terms of accurate estimation. If following the similar ratio scale (e.g., ratio > 1), how is this guaranteed in your proposed method?


2.	Please provide the time complexity analysis of the proposed framework GM-DE.

3.	Please discuss the performance guarantee of GM-DE on general real-world networks as a universal method in shortest path estimation.

### Soundness
2

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
3

### Summary
The paper proposes GM-DE, a learning-based framework for estimating shortest-path distances on large graphs by combining (i) local pivot-based embeddings refined with a GCN and (ii) global multi-view anchor-set embeddings encoded with a Multi-View DNN (MVDNN). The model trains predictors that fuse local/global embeddings in three variants (Embedding-Fusion, Distance-Fusion, Local-Tuning).

### Strengths
1. **Well-structured system**: The paper gives a full pipeline (pivot/anchor selection, encoders, three predictor fusion strategies) and implements them end-to-end (Alg.1–3 and training details).

2. **Empirical breadth**: Evaluation spans multiple graph types (citation, social, road) and includes both accuracy (MAE/MRE) and operational costs (inference time, storage).

3. **Ablation provided**: The “w/o local / w/o global” ablation is useful and shows the complementary value of the two components.

### Weaknesses
1. **Inconsistent/unclear use of Bourgain / anchor sampling**: The description of anchor sampling uses several symbols (k, p, q, c, log n, log² n) that are inconsistent/confusing. For example the paper says “we sample k = c log² n anchor sets (where i=1,...,log n, j=1,...,c log n…)” — it's unclear whether the number of views is log n or log² n, and how p and q relate. This makes it hard to reason about memory/time scaling. (Sec.4.2)
2. **Generalization and inductive behavior unclea**r: It's not clear whether GM-DE can generalize to nodes unseen during training or to changes in graph structure (edge additions). The method seems transductive (relies on precomputed distances to pivots/anchors). Explain how it handles new nodes or edges.
3. **Hyperparameters & training details missing**: Learning rates, batch sizes, number of epochs (θ), optimizer, weight decay, pivot/anchor sampling seeds, and how pivot number scales with graph size are not fully provided.

### Questions
1. The YouTube dataset has 1.13M nodes but only 5 pivots were used. How do pivot/anchor choices affect the results vs memory tradeoff? Also the complexity of computing distances to anchors/pivots (for large graphs) is not analyzed (offline cost can be very high if many anchors are used).

2. What are the results when you apply the method on Heterophilic Datasets?

3. GCN is used for local encoder and a DNN for each view in MVDNN. Would be helpful to compare explicitly to other positional encoders?

4 Can you compare max vs mean vs min vs quantile and learned pooling (DeepSets) for d(v,S) to show the choice’s impact?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes GM-DE a GNN and neural network based method to estimate shortest path distances between two nodes in large-scale graphs.
The framework generates two embeddings:
* A __local embedding__ that is derived by selecting a set of pivot nodes and calculating the distance of all other nodes to them. This distance matrix $Nx\tau$ is then refined by DNN to generate the local embeddings $ X^{L} $.
* A __global embedding__ that is derived by selecting a subset of nodes to act as anchors. Then for each node it calculates the maximum distance of it to any node in the anchor-set. This distance matrix is then refined to generate the global emdding $ X^{G} $.
For each pair of nodes i,j these two embeddings are then mixed using three different strategies (each one yielding a separate model) to get the  final predicted distance.
The model is trained to minimize the MSE between the true distance $d_{i,j}$ and the predicted $\hat{d_{i,j}}$

### Strengths
* The proposed framework seems to outperform existing baselines and state of the art models in terms of the MSE and MAE between true and predicted distances.
* The ablation study seems to justify the choice of a local embedding $X^{L}$ and a global one $X^{G}$.

### Weaknesses
* While the proposed framework is quite comprehensive in the steps it takes, these steps might be over-specific and thus make it difficult to reproduce, or act as inspiration for future work. For example, the final results rely in (i) the decreasing in i size of anchor sets, (ii) the choice of max operator in eq. 3 , (iii) downsampling pairs of certain distance,etc. There is in some sense a lack of understanding or ablation studies on if and by how much those steps are necessary.
* The question of fast and approximate distance estimation mainly applies to large scale graphs it is only evaluated against two medium (by today's standards) graph sizes (YT with 5M nodes and DBLP with 300K).
* There is not enough information to evaluate the model in certain axis: (i) How much time it takes to train the model and how it scales? Either theoretically or by using synthetic graphs of increasing sizes. (ii) What is the dimension D chosen and how it affects predictions? Similar for other hyperparameters, like learning rate, exact number of samples chosen through training and how this affects performance,etc. (iii) How are the other methods tested? Do they follow the original implementation? Did the authors implement from scratch, which language, etc.

### Questions
See some questions from above + 
* If on average on the latest category of anchor sets for i=logn, there is one node, how is the case of empty sets handled?
* In line 271 the rationale behind choosing a max aggregator is the computational complexity. How is that different from other aggregations (like min, sum, mean)? Is there any other rationale behind this choice? Like training stability, gradient considerations?
* Similarly, for (4) why is the average a good choice? If a node is the only in a set then we pretty much have all infromation about its distance to other nodes that is washed by averaging.
* For the DF measure, the true values of w1 and w2 that balance between local and global could be interesting and another addition to the ablation study.
* Line 454.. why is the inference cost directly related to the number of pivots? Should the pivots be employed just in the training phase? Then all needed is just the learnt embeddings. Similar for line 399 why are pivots selected in test set?
* Why are there NAN values for the ADO method? This is a linear method that can scale to large graphs. As a matter of fact it is tested against a graph of 49M nodes in the paper. this method is also similar to how local embeddings are obtained before the refinement of them with DNNs as happens in this work.

### Soundness
2

### Presentation
2

### Contribution
2
