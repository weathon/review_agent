# Cluster Topology‑Driven Placement of Experts Reduces Network Traffic in MoE Inference

- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Efficient deployment of a pre-trained LLM to a cluster with multiple nodes is a critical step for providing fast responses of the service to users' queries. The recent success of Mixture-of-Experts (MoE) LLMs raises the question of how to deploy them efficiently, considering their underlying structure. During the inference in MoE LLMs, only a small part of the experts is selected to process a given token. Moreover, in practice, the experts' load is highly imbalanced. For efficient deployment, one has to distribute the model across a large number of servers using a model placement algorithm. Thus, to improve cluster utilization, the model placement algorithm has to take into account the network topology. This work focuses on the efficient topology-aware placement of the pre-trained MoE LLMs in the inference stage. We propose an integer linear program (ILP) that determines the optimal placement of experts, minimizing the expected number of transmissions. Due to the internal structure, this optimization problem can be solved with a standard ILP solver. We demonstrate that ILP-based placement strategy yields lower network traffic than competitors for small-scale (DeepSeekMoE 16B) and large-scale (DeepSeek-R1 671B) models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the important problem of efficiently deploying MoE models across distributed clusters by optimizing expert placement to minimize network communication overhead. The authors formulate the placement problem as an Integer Linear Program (ILP) that considers both network topology and expert load statistics. Experiments on DeepSeek-MoE 16B and DeepSeek-R1 671B models demonstrate improvements over baseline methods across multiple network topologies (FatTree, Dragonfly).

### Strengths
* The paper tackles a genuine bottleneck in deploying large-scale MoE models. With all-to-all communication consuming significant part of runtime, this is a critical issue for practitioners.

* The ILP formulation is clear and well-presented

* Testing on 4 different topologies (FatTree, Dragonfly, and their sparse variants) with visualizations provides good coverage of real-world scenarios.

* Unlike MOETuner which times out after 12 hours, the proposed ILP solves in tens of minutes, making it practically deployable.

### Weaknesses
* The assumption about communication could be improved. Specifically, the token replication on a same device for different experts could be reduced [1].

* One limitation is that all experiments report "network hops" as a proxy metric, but no actual wall-clock runtime measurements are provided on real clusters.

* I was wondering how sensitive is the proposed method to the profiling dataset? Does OASST1 generalize well to various domains?


[1] Luo, S., Li, P., Peng, J., Wang, H., Cheng, Y. and Chen, T., 2025. Occult: Optimizing Collaborative Communication across Experts for Accelerated Parallel MoE Training and Inference. arXiv preprint arXiv:2505.13345.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work focuses on how to efficiently deploy large Mixture-of-Expert (MoE) models for inference by jointly considering the network topology in distributed environments and the workload imbalance across experts. Two variants of integer linear programming (ILP) problems are formulated to minimize the network traffic (measured as the number of network hops) by deducing the expert placement.

### Strengths
1. The problem formulation is clear and the paper is generally easy to follow.
2. Unlike previous works, this paper jointly considers multiple network topologies and the workload imbalance across experts.
3. The evaluation is conducted at a large scale (671B model over 256 GPUs).

### Weaknesses
1. The formulation focuses on the “distance” (the length of the shortest path during network transmission). However, it is not a good proxy for network transmission cost, since different connections would have divergent bandwidths, and the network latency should also be considered (especially for small transmission messages).

2. According to Table 1, it is time-consuming for the problem solving of both ILP and ILPLoad. In practice, the request distribution may shift across time, which also affects the routing distribution. Given the long problem solving time, I’m afraid it can hardly get real-world deployment.

3. The evaluation metric is the number of network hops, rather than widely used metrics like latency, throughput, and SLO. 

4. DeepSeek R1 adopts shared experts. It seems that the current work fails to take this factor into account.

### Questions
1. Can you provide metrics like latency, throughput, and SLO? If not, please explain why, and please clarify how the reduction (e.g., 20%) in hops can be translated to a meaningful improvement in end-to-end performance. Besides, the interconnected bandwidths should be specified. 

2. How can this work be extended to dynamic routing distributions and models with shared experts?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the efficient distributed inference of large-scale Mixture-of-Experts (MoE) language models, which, despite their massive parameter counts, activate only a sparse subset of experts per token, leading to highly imbalanced expert utilization and significant network communication bottlenecks across multi-GPU clusters. The authors formalize the expert-placement problem as a topology-aware integer linear program (ILP) that minimizes the expected number of network hops required to dispatch and collect expert outputs. The model represents the cluster as an undirected weighted graph whose vertices are GPUs/servers and whose edge weights reflect inter-server distances; zero weights model intra-server NVLinks. The ILP decides, for every expert in every MoE layer, which server it should reside on, subject to per-server caps on total experts and on experts from any single layer. A load-aware extension (ILPLoad) incorporates empirical expert-activation frequencies derived from the OASST1 dataset, weighting the hop count by the probability that each expert is actually activated. Extensive experiments on two model families (DeepSeek-MoE 16 B and DeepSeek-R1 671 B) and four cluster topologies (Fat-Tree, Dragonfly, and their sparse variants) with 256 GPUs show that ILPLoad reduces average network hops by 6–30 % relative to strong baselines (round-robin and greedy placement), while remaining solvable in under 25 min for the largest configuration. The study thereby demonstrates that co-designing model mapping with network topology and workload statistics yields substantial reductions in inference-time communication overhead, improving cluster utilization and response latency for production MoE services.

### Strengths
1. The topic of this research is very interesting and practical: It targets the network-communication bottleneck that arises when serving Mixture-of-Experts models across a multi-GPU cluster at inference time. It proposes an expert-placement strategy that explicitly exploits the cluster’s physical topology—an issue that, while highly practical for today’s large-model deployments, has received little prior attention. Most existing studies concentrate on training-time optimizations or on balancing expert load; in contrast, this work zeroes in on reducing inference-stage network traffic, delivering clear engineering value and research novelty.
2. This paper proposes a rational approach to formalize the expert-placement task as an integer-linear-programming (ILP) problem with a clear objective and concise constraints, demonstrating strong theoretical rigor. By incorporating expert-load statistics (ILPLoad) as prior knowledge to refine the objective, they also show a deep understanding of real-world deployment scenarios.
3. Systematic experiments were conducted across multiple network topologies (Fat-Tree, Dragonfly, etc.) and two MoE models of different scales (16B and 671B), demonstrating the effectiveness of the ILP-based method in diverse scenarios. Compared with common heuristics such as Round-Robin and Greedy, ILPLoad shows significant advantages in reducing the number of network hops, achieving gains of up to 30% or more.

### Weaknesses
1. Scalability Concerns: While the ILP approach provides optimal solutions, the computational time (1185.9 seconds for ILP, 1397.5 seconds for ILPLoad) may not be feasible for very large-scale deployments in real-time environments. While the authors acknowledge this limitation, further investigation into methods for speeding up the optimization process or approximating solutions could be beneficial. Just a reminder: as an open-source solver, the CVXPy library is built on CBC (COIN-OR Branch-and-Cut). Its performance and efficiency can be significantly lower than other commercial solvers such as Gurobi or CPLEX.
2. Optimization Problem Complexity: While the ILP provides optimal solutions, the complexity of solving large ILP problems for extremely large models, such as Kimi K2, may pose challenges for deployment. Further discussion on how the method can be adapted or scaled for even larger models, or on how to handle more complex expert routing, would be valuable.

### Questions
1. Expert Load Estimation: The ILPLoad approach uses expert load statistics from the OASST1 dataset for optimizing expert placement. How robust is this approach when expert load statistics vary between different datasets or in production environments? Is there a way to dynamically update or estimate these statistics in real-time during inference?
2. Comparison to Other Methods: You mention that the ILP-based method outperforms MoETuner in terms of runtime and network traffic reduction. Can you elaborate more on the differences in approach between MoETuner and your method? Specifically, what are the key advantages of using ILP over MoETuner, aside from runtime and network traffic?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on the efficient deployment of MoE model during inference, and proposes a expert placement strategy using topology-aware modeling and ILP to minimize transmission while addressing expert load imbalance. 
Experiments show the strategy achieves lower network traffic than competitors on DeepSeek models.

### Strengths
1. The manuscript is well-structured and easy to follow.
2. The experimental results demonstrate significant benefits.

### Weaknesses
1. This paper is best suited for submission to conferences focused on machine learning systems and computer architecture, as its contributions are closely related to cluster network design and optimization.
2. The target scenario and underlying motivation require further elaboration. Additionally, the analysis and evaluation do not sufficiently demonstrate the impact of key factors such as data volume, network bandwidth, and latency.
3. The paper lacks end-to-end evaluations and comprehensive comparisons with existing methods.
4. Considering only scale-out networks is insufficient to address the challenges of expert placement in practical MoE deployments, as this issue has already been thoroughly examined in existing works.

### Questions
Is the evaluation performed on a simulated environment or a real cluster?

### Soundness
2

### Presentation
2

### Contribution
2
