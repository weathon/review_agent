# DFCA: Decentralized Federated Clustering Algorithm

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 4

## Abstract
Clustered Federated Learning has emerged as an effective approach for handling heterogeneous data across clients by partitioning them into clusters with similar or identical data distributions. However, most existing methods, including the Iterative Federated Clustering Algorithm (IFCA), rely on a central server to coordinate model updates which creates a bottleneck and single point of failure, limiting their applicability in more realistic decentralized learning settings. In this work, we introduce DFCA, a fully decentralized clustered Federated Learning algorithm that enables clients to collaboratively train cluster-specific models without central coordination. DFCA uses a sequential running average to aggregate models from neighbors as updates arrive, providing a communication-efficient alternative to batch aggregation while maintaining clustering performance. Our experiments on various datasets demonstrate that DFCA outperforms other decentralized algorithms and performs comparably to centralized IFCA, even under sparse connectivity, highlighting its robustness and practicality for dynamic real-world decentralized networks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present a new clustering method for decentralized FL extending the core idea of IFCA in absence of a central server. Each client stores all the cluster models and it is dynamically assigned to the cluster whose model minimizes the empirical loss. In the paper, the authors also present two possible aggregation strategies, a batch aggregation strategy and a sequential strategy that better meets the constraints of real world FL deployment. The paper is well motivated and quite easy to follow.

### Strengths
I appreciated the clarity of the paper. As mentioned above it is easy to follow and well motivated. In particular, the authors did provide a good theoretical extension of IFCA's proofs to the decentralized setting, properly applying interesting results of algebraic graph theory.

### Weaknesses
1. The DFL framework is interesting and I appreciate the effort of trying to extend IFCA to this scenario, however the similarities concerns me about the novelty of the proposed approach.

2. Storing all the cluster models on each client could be impractical, mostly in low-powered IoT frameworks where the storage capacity is often limited. I am concerned that in large scale scenarios  where the number of clusters drastically increases the memory cost of storing all the models explodes. 

2a. The algorithm does not prevent from forming degenerate clusters, i.e. a cluster with a single client. In this case the model of the clusters coincides with the individual model of the client. Hence, this would imply that a client stores the individual information of another client, which is privacy concerning, with respect to standard FL privacy assumptions.

3. Since the number of clusters is fixed a priori as input of the algorithm, similarly to IFCA, why they evaluate the algorithms with $k = 4$? The evaluation shall be extended to other values of this hyper-parameter.

4. In the introduction the authors claim that their method is robust against data heterogeneity, however it is not precised with which value of Dirichlet's $\alpha$ they construct the federation. 

5. I think that the authors should compare experimentally their method to other clustering DFL algorithms, or other DFL algorithms that are specially designed to mitigate data heterogeneity.

6. I advise the authors to update the literature review --- while most of the relevant historical works on CFL are present, the most recent literature is not properly discussed.


i. I do not get why the related work section has been placed at the end of the experimental results. It compromises the readability of the paper.

ii. In the pseudocode why the number of local SGD iterations equals the number of communication rounds?

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

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
DFCA is a decentralized federated clustering algorithm inspired by IFCA, designed to mitigate data heterogeneity. It iterates through three steps. First, each client performs cluster assignment by locally evaluating all k cluster’s models on its data and self-assigning to the cluster with the minimum loss. Second, it executes a local update. Third, it conducts decentralized aggregation, where it exchanges and averages all k models with its neighbors, notably using a "sequential running average" to efficiently handle asynchronous updates.

### Strengths
1. DFCA achieves clustering in a DFL setting, and its performance is comparable to centralized IFCA under data-heterogeneous scenarios.
2. DFCA propose a practical sequential running average method to enable asynchronous aggregation. This mechanism is well-suited for realistic decentralized deployments .

### Weaknesses
1. The memory cost, computation cost, and communication cost are all large, each of which is k times higher than that of standard decentralized FL. First, DFCA requires each clients to store all k clustering models locally. Second, DFCA performs a complete inference on each of these k models using all their local data to find the model with the lowest loss. Third, the client needs to exchange all k models with its neighbors. The experiments in the paper are limited to small-scale settings such as k=2 and k=4, which masks the serious scalability issues of the design. If the number of clusters is slightly larger (such as k=10 or k=20), this k times memory and k times inference overhead is completely impractical for resource constrained devices.

2. The description of aggregation is contradictory. The aggregation defined by Eq. (6) and (7) is that client i only interacts with the neighbors N_ij in cluster j. However, the convergence analysis in Section 4 (Eq. 10) assumes that client i will interact with all neighbors N_i aggregate all k models. These two descriptions are completely inconsistent. The author must clarify which one is the true aggregation method of DFCA.

Overall, the methods proposed in this paper lack novelty and technical inspiration.

### Questions
see the above.

### Soundness
2

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
3

### Summary
This paper presents DFCA, a Decentralized Federated Clustering Algorithm that combines federated clustering (as in IFCA) with decentralized communication. The goal is to remove reliance on a central server and allow clients connected via a communication graph to collaboratively train cluster-specific models. To handle asynchronous communication, the authors propose a running average aggregation scheme and provide convergence analysis under smoothness and separability assumptions. Experiments on several benchmark datasets (MNIST, EMNIST, CIFAR-10, FEMNIST) suggest that DFCA can achieve accuracy comparable to centralized IFCA while outperforming other decentralized baselines.

### Strengths
1. The proposed approach is conceptually simple, intuitive, and easy to implement within decentralized federated learning frameworks.
2. While the theoretical contribution is not highly novel, the paper provides a useful convergence discussion that helps connect DFCA with existing decentralized SGD analyses under standard smoothness and connectivity assumptions.

### Weaknesses
1. The theoretical section mostly adapts existing results from IFCA and decentralized SGD. It does not introduce new analytical techniques or address the harder questions that arise from decentralization, e.g., how delayed neighbor updates or misclustered clients affect convergence.
2. The experimental evaluation in this paper is not sufficiently comprehensive. While standard datasets are used, the experiments focus on relatively small-scale and synthetic heterogeneity (via data rotation). 
3. The paper does not isolate the contribution of its design choices, e.g., running average vs. synchronous aggregation, GI vs. LI initialization. Additionally, comparisons with more recent personalized or clustered FL methods (e.g., pFedMe, FedProx-based decentralized variants) are missing.
4. The algorithm requires each client to maintain $k$ full model copies and assumes stable connectivity across the network. This limits scalability to large $k$ or unstable peer-to-peer networks.

### Questions
1. How sensitive is DFCA to temporary misclusterings? Does the convergence argument still hold if clients frequently switch clusters or if cluster separability is weak?
2. How does the running average scheme compare empirically with simple synchronous averaging? Is there a measurable benefit in terms of wall-clock time or communication efficiency?
3. Have the authors evaluated DFCA on larger networks (e.g., $N>500$) or more realistic non-IID data partitions (e.g., natural label skew)? How does the method behave under dynamic connectivity or dropped messages?
4. How sensitive is performance to the number of clusters $k$ and to initialization (GI vs. LI)? Could the method degrade to standard decentralized SGD when $k=1$?
5. Could the authors clarify why comparisons with more recent personalized or clustered decentralized FL baselines (e.g., FedProx-Decentralized, Per-FedAvg) were not included?
6. Could the authors provide additional experiments to strengthen the evaluation? For example, have they considered testing DFCA on larger-scale or more realistic non-IID datasets, varying the number of clusters k, or evaluating its performance under strongly asynchronous or dynamically changing communication graphs? 
7. Additionally, could the authors investigate communication and computation trade-offs, as well as scalability on larger networks?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DFCA (Decentralized Federated Clustering Algorithm), which, as the name suggests, finds clusters of federated learning clients in a decentralized manner while these clients train models on their local data. The key idea is to use a sequential running average of neighboring clients’ updated model parameters in order to compute model updates, instead of attempting to aggregate all neighbors’ parameter updates at once in each round. Such sequential averaging is naturally amenable to asynchronous updates, and the paper shows that it reduces to decentralized stochastic gradient descent (SGD) after the clients’ clusters converge, which is proven to happen after a sufficient number of update rounds. Experiments show that DFCA can nearly match the performance of centralized IFCA (Iterative Federated Clustering Algorithm), while outperforming other decentralized federated learning algorithms.

### Strengths
+ The experimental results show that DFCA outperforms several decentralized federated learning baselines, on multiple different datasets. It also nearly matches the accuracy achieved by IFCA, which is impressive for a decentralized learning algorithm.

+ The DFCA algorithm itself appears fairly easy to implement and can be naturally extended to asynchronous settings, which is especially important if clients’ network connections can change over time, which may not be synchronized across rounds.

### Weaknesses
--The client disagreement Disp_j^t is never formally defined, making it difficult to fully appreciate Theorem 1. Some assumptions are also unclear, e.g., wouldn’t the reduction to decentralized SGD rely on the fact that clients perform only one local training iteration between each update?

--It’s not clear why Assumption A3 has different graph mixing conditions for the synchronous and asynchronous cases. The conditions could also be defined more formally, e.g., what exactly does “disagreement contracts” mean? Wouldn’t that be a property of the algorithm as well as the underlying connectivity graph, not the graph mixing itself?

--The paper does not give many concrete examples of where DFCA might be deployed. For example, the assumption that clients are separated into distinct clusters appears fairly strong, so it would be useful to discuss some example applications where this assumption would be reasonable.

### Questions
1) There is no discussion of how (or whether) the proof of Theorem 1 differs significantly from prior work in the literature. My understanding is that the proof that the clusters eventually stabilize is fairly standard, though I’d appreciate clarification on this point from the authors (in particularly whether the sequential averaging makes a material difference to the proof technique).

2) Does DFCA assume that clients know their neighboring sets N_{I,j} at any given time? If not, how would they know when to resume assigning a new cluster to their data (going back to Step 1 of the algorithm) after the aggregation in Step 3 is complete? How would clients know this neighbor set in practice, since network connectivity may change over time?

3) The experiments section claims that DFCA is robust to low connectivity. I agree that the experimental results provide evidence of this claim, but is it evident in Theorem 1’s convergence result as well?

Please also see the questions listed in the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2
