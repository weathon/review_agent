# VRouter: Micro-batch Level Load Balance via Inter-EP Routing for MoE Training

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Load imbalance within the Expert Parallel (EP) group leads to poor GPU efficiency when pre-training large-scale Mixture-of-Experts (MoE) models. Though recent approaches have attempted to mitigate this through dynamic expert rearrangement at the global-batch level, they overlook the rapid and dynamic variations in load distribution across different micro-batches. Additionally, relocating or shadowing popular experts at micro-batch level incurs substantial communication overhead due to frequent migrations of expert parameters and gradients.

To address these issues, we introduce VRouter, a novel Inter-EP routing system that achieves better load balance at the micro-batch level, without requiring any expert migration or replication. We have three key techniques: (1) VRouter utilizes the expert shifting strategy that allows workloads to be redistributed across neighboring devices, creating additional opportunities for balancing, (2) VRouter adopts expert dropping mechanism to reduce both per-device memory footprint and gradient synchronization overhead across EP groups, by selectively dropping experts while preserving load balance, and (3) VRouter applies a lightweight load-aware token routing algorithm that redistributes load across devices uniformly.
Experimental evaluations on representative MoE models demonstrate that VRouter achieves 1.05-1.13$\times$ throughput speedup over existing routing systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper analyzes the load balancing problem in MoE models from a micro-batch level. The core design is that alleviating load imbalance through an Inter-EP routing mechanism by distributing tokens across multiple EP components instead of migrating or replicating experts. The authors devise a simple and efficient algorithm for load scheduling. Experiments demonstrate that VRouter achieves 1.13× higher training efficiency than existing MoE training systems.

### Strengths
1. The author examines load balancing in the Moe architecture from the perspective of micro-batches, presenting a valuable research topic.

2. The author thoroughly discusses the problems and motivations behind load balancing under mini-batches.

3. This paper employs a method of scheduling experts across expert groups and designs a simple yet effective strategy to address the load balancing problem.

### Weaknesses
1. Traditional data parallelism does not introduce dependencies between nodes during forward and backward propagation. However, employing inter-EP communication introduces dependencies between different EP groups. Specifically, the parallel computation efficiency of each layer is constrained by the slowest EP group. 

2. The overall organization of this paper could be improved. The author might consider adjusting the placement of some figures (fig.1, fig.2 and fig.3) to help readers better connect them with the surrounding context. 

3. The communication latency is a critical factor in EP, particularly during inter-EP scheduling. Therefore, this aspect should be considered when designing expert scheduling algorithms. 

4. This work involves both intra-EP and inter-EP communication, so it is necessary to describe the different communication rates in detail in the Testbed section.

5. The work introduces a hyperparameter D that affects overall training efficiency and memory consumption, particularly causing training slowdown as described in Experiment 6.3. However, the authors do not provide a clear algorithm for determining its value.

6. This paper addresses load balancing from the perspective of micro-batches, though this approach appears to have been discussed previously. We hope the authors will compare the differences between this paper and the following works. Demonstrating these distinctions through experiments would be preferable. 
---
- Zeng Y, Huang C, Mei Y, et al. EfficientMoE: Optimizing Mixture-of-Experts Model Training With Adaptive Load Balance[J]. IEEE Transactions on Parallel and Distributed Systems, 2025.
- Li J, Sun Z, He X, et al. Locmoe: A low-overhead moe for large language model training[J]. arXiv preprint arXiv:2401.13920, 2024.

### Questions
1. What is the complete experimental environment for this testbed?
2. Should communication be incorporated into the algorithm design to better schedule tokens across different expert groups?
3. Is this paper the first to analyze load balancing in the MOE architecture from a micro-batch perspective?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces VRouter, a novel Inter-EP routing system that improves load balance at the micro-batch level in Mixture-of-Experts (MoE) models, without expert migration or replication, by employing expert shifting and dropping techniques to optimize GPU efficiency and reduce communication overhead.

### Strengths
1. The experiments are extensive, covering a wide range of situations and scenarios.

### Weaknesses
1. One potential drawback is that the paper’s approach can be seen as a variation of MOE folding, which decouples the parallelism formulation of the MOE component. This may limit the novelty of the proposed scheduling method.
2. While the low overhead is a strength, the corresponding end-to-end training improvement is also limited to around 5%.
3. Another consideration is how this approach composes with other scheduling optimizations. The performance improvement from the proposed method may not be additive with those from orthogonal techniques like DeepEP or compute/communication overlap, potentially limiting its practical impact in optimized systems.

### Questions
see weakness pls

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
This work propose inter-EP expert placement and routing to optimizing MoE model training by addressing the expert load imbalance at the micro-batch level. The experimental results show that the implemented methods achieve throughput improvements over existing MoE training framework.

### Strengths
1. It targets to address the issues commonly observed.
2. The results show that the proposed method yields benefits.

### Weaknesses
1. The paper should better situate its work in the context of recent literature. Prior work has mostly focused on load imbalance across experts at the micro-batch level. In contrast, there is a growing, more recent trend of addressing imbalance across devices and at the global-batch level, which the authors should acknowledge.
2. The paper's assumed baseline requires further justification. Existing frameworks for hybrid parallelism already support EP both within and across DP groups. The authors should verify whether their chosen baseline is appropriate, as it appears it could lead to the creation of redundant EP groups. 
3. The paper would benefit from a more detailed analysis of the target problem scenario. Specifically, the authors should elaborate on the constraints imposed by throughput and memory requirements, while also considering the impact of heterogeneous communication bandwidths (both inter-node and intra-node) on the proposed solution. It is more valuable than the general discussion on "MOTIVATION, OPPORTUNITIES AND CHALLENGES”.
4. The authors should clarify the novelty of their proposed methods. “Expert shifting" and "expert dropping" seem to be new terms for established expert placement strategies like expert migration and replication. The paper proposes building a termed inter-EP group process, which seems to be only a different framework implementation, compared to existing methods of building an EP group to support a variable number of experts and flexible placement. 
5. Reference repeat: "Demons in the detail: On implementing load balancing loss for training specialized mixture-of-expert models".

### Questions
Please refer to the issues in the Weakness.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes VRouter that tackles micro-batch-level load imbalance in large-scale MoE training without requiring expert migration or replication.  VRouter introduces three key techniques: (1) Cyclic Expert Shifting, which pairs Expert Parallel (EP) groups and cyclically shifts expert placements in one group to create heterogeneous layouts, enabling tokens to be offloaded across groups;  (2) Expert Dropping, which reduces per-device memory and gradient synchronization overhead by dropping a configurable number of experts while preserving load balance;  and (3) a lightweight load-aware token rerouting algorithm that greedily redistributes tokens from overloaded to underloaded devices based on real-time load information.  Evaluated on Qwen-series MoE models (Qwen-S/L/XL/XXL) against baselines like Megatron-LM+, FasterMoE, and SmartMoE, VRouter achieves 1.05–1.13× higher training throughput, reduces GPU memory usage by up to ~10%, attains a near-ideal Load Balance Ratio (LBR) of 1.038 (vs. 1.32–1.40 for baselines), and avoids out-of-memory errors in large EP configurations, demonstrating superior efficiency, memory optimization, and scalability.

### Strengths
* Unlike existing methods that only focus on global-batch level optimization and fail to capture dynamic load variations across micro-batches, VRouter addresses micro-batch level load imbalance.
* VRouter avoids the high communication and memory overhead of moving or copying experts by instead enabling Inter-EP token routing
* VRouter’s load-aware token rerouting algorithm is lightweight, collecting real-time load data and using greedy search for fast routing decisions.
* VRouter works efficiently at scale and integrates seamlessly with existing MoE training pipelines without requiring expert redesign or complex scheduling.

### Weaknesses
* Although VRouter uses cyclic expert shifting to create heterogeneous EP group layouts, it still faces the inherent challenge of homogeneous load patterns across EP groups.
* While Expert Dropping reduces memory and communication, dropping too many experts leads to degraded throughput due to increased load imbalance.
* It is unclear how VRouter performs at multi-thousand-GPU scale, where Inter-EP communication across nodes may introduce significant latency or bandwidth bottlenecks.

### Questions
Although the rerouting algorithm is described as “lightweight,” the paper does not quantify its runtime overhead (e.g., CPU scheduling time, latency per micro-batch), which could become non-negligible under very small micro-batches or large expert counts.

### Soundness
3

### Presentation
3

### Contribution
3
