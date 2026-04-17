# SEAFormer: A Spatial Proximity and Edge-Aware Transformer for Real-World Vehicle Routing Problems

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Real-world Vehicle Routing Problems (RWVRPs) require solving complex, sequence-dependent challenges at scale with constraints such as delivery time window, replenishment or recharging stops, asymmetric travel cost, etc. While recent neural methods achieve strong results on large-scale classical VRP benchmarks, they struggle to address RWVRPs because their strategies overlook sequence dependencies and underutilize edge-level information, which are precisely the characteristics that define the complexity of RWVRPs. We present SEAFormer, a novel transformer that incorporates both node-level and edge-level information in decision-making through two key innovations. First, our Clustered Proximity Attention (CPA) exploits locality-aware clustering to reduce the complexity of attention from $O(n^2)$ to $O(n)$ while preserving global perspective, allowing SEAFormer to efficiently train on large instances. Second, our lightweight edge-aware module captures pairwise features through residual fusion, enabling effective incorporation of edge-based information and faster convergence. Extensive experiments across four RWVRP variants with various scales demonstrate that SEAFormer achieves superior results over state-of-the-art methods. Notably, SEAFormer is the first neural method to solve 1,000+ node RWVRPs effectively, while also achieving superior performance on classic VRPs, making it a versatile solution for both research benchmarks and real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SEAFormer, a scalable and edge-aware transformer designed for real-world vehicle routing problems (RWVRPs). Specifically, it presents an efficient clustered proximity attention mechanism that restricts attention computation to nodes within the same cluster. In addition, an edge-aware module is proposed to incorporate pairwise relational information during both encoding and decoding. Extensive experiments on several large-scale RWVRPs demonstrate the effectiveness and efficiency of the proposed approach.

### Strengths
* This paper addresses several real-world vehicle routing problems.
* The proposed method is technically sound and significantly reduces computational complexity.
* The paper is well-written and enjoyable to read.

### Weaknesses
* The paper format (e.g., margins) appears to be incorrect.
* The introduction does not mention improvement-based solvers, which are an important category in this domain.
* The definition of RWVRP is not clearly stated. Recent NCO approaches have considered several VRP variants [1]. Are these also classified as RWVRPs?
* The generality of SEAFormer remains unclear. For instance, real-world VRPs often do not rely on 2D Euclidean coordinates or assume a single depot. How does SEAFormer (Eq. (1)) handle such cases?
* What is the difference between the proposed proactive masking and the conventional masking used in Transformer-based solvers? The description in lines 938–943 does not make this distinction clear. For instance, models such as MTPOMO, MVMoE, and RouteFinder also consider condition (iii) when addressing VRPL. Moreover, could the authors clarify how proactive masking further enhances solution quality and reduces the search space?
* Additional baselines should be included in Table 1, such as LKH3 and HGS for VRPTW. Furthermore, are the reported neural baselines (e.g., POMO, MTPOMO, RouteFinder) retrained under the same training settings (e.g., varied problem sizes)?
* For the ablation study in Fig. 3, how does performance change when CPA is removed?
* Have the authors conducted any hyperparameter sensitivity analyses?
* A comprehensive comparison of training and inference overheads (e.g., memory consumption, training time) should be provided.
* Section 6 should include deeper discussions on the method’s limitations and potential directions for future work.
* Minor:
  * Line 218: $s_n^{\alpha} \to s_{\alpha}^n?$
  * Line 317: what does $c_t$ within $q_t$ represent? Is it the remaining vehicle capacity?
  * Line 1210: should $\alpha$ be 1 instead of 0?
  * In Table 2, “HGS” should be replaced with “HGS-PyVRP” due to differences in performance and implementation.
  * Fig. 6 is informative and should be moved to the main paper.
  * What does $B$ denote in Figs 4 and 5?

[1] Routefinder: Towards foundation models for vehicle routing problems.

----

Overall, I believe this paper makes a valuable contribution to the VRP domain by addressing several key challenges encountered in real-world VRPs. Therefore, I recommend acceptance.

### Questions
* Could SEAFormer handle complex constraints, as studied in [2]?
* Is it possible to train SEAFormer on the four RWVRPs in a multi-task manner?
* I did not fully understand the issue of hard cluster boundaries and how Eq. (4) addresses it. Could the authors elaborate on this point?
* What are the node and edge features used for each problem (e.g., VRPTW)? Additionally, how would the model handle inputs in the form of distance matrices rather than coordinates, as in AVRP?

[2] Learning to Handle Complex Constraints for Vehicle Routing Problems.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the critical challenge of solving large-scale Real-World Vehicle Routing Problems (RWVRPs), which existing neural methods fail to address due to sequential constraints and O(n2) complexity. The authors propose SEAFormer, a novel Transformer that introduces two key innovations: 1) Clustered Proximity Attention (CPA), a domain-specific sparse attention that reduces complexity to O(n) by clustering nodes based on polar coordinates, and 2) a lightweight Edge-Aware Module that explicitly models edge-level information via a global heatmap. Extensive experiments show SEAFormer is the first neural method to effectively solve 1000+ node RWVRPs, achieving SOTA results across four variants and classic VRP, even scaling up to 7000 nodes with massive gains over recent baselines.

### Strengths
Important and Challenging Problem: The paper tackles the critical gap between NCO research (small VRPs) and industrial applications (large-scale RWVRPs). Solving 1000+ node RWVRPs is a major milestone.

Novel Architectural:  The CPA innovation (based on geometric priors of polar coordinates) is an insightful design. Meanwhile, decoupling node-level sequential attention from edge-level global features is an effective way to handle heterogeneous constraints.

Strong Experimental Results: Extensive experimental evaluations are conducted to demonstrate the effectiveness of the proposed solution.

The paper is well written and easy to follow.

### Weaknesses
1. Incomplete Complexity Analysis: The paper claims O(n) complexity (i.e., O(nRM)) in Equation (6). This conclusion relies on R and M being fixed constants. However, the authors do not discuss the boundary condition where R *M > n, in which case O(nRM) would not hold. A rigorous theoretical analysis is encouraged.

2. Insufficient Analytical Justification: The paper lacks rigorous analytical evidence to explain the source of its strong empirical performance. It remains unclear whether the "scalability" arises purely from the O(n) complexity of CPA or from a stabilizing interaction between CPA and the EAM. Consequently, the originality of the contribution is difficult to evaluate, as the authors do not convincingly demonstrate why their domain-specific “polar clustering” is essential, or whether comparable gains could be achieved using a generic O(n) sparse attention mechanism.

3. In the Related Work section, the author discusses the limitations of existing studies, but does not clearly explain how the proposed method addresses or overcomes these limitations.

4. Potential Limitation to Multi-Depot Problems: CPA's core mechanism (polar coordinates) fundamentally limits it to Single-Depot VRPs (SDVRP), making it difficult to extend to Multi-Depot (MDVRP) settings.

5. Hyperparameter Sensitivity: The appendix shows that CPA is sensitive to its key hyperparameters (R and M), which could be a barrier to practical adoption.

### Questions
On the contribution of the SGBS search: To enable a more direct comparison of architectural strength, could the authors report the performance of SGBS when applied to the strongest existing baselines, such as UDC or LEHD? This would clarify whether the observed improvements stem primarily from the SGBS procedure or from the proposed architecture itself.


Implementation details: The Appendix states that the edge module computes representations only for the 50 nearest neighbors. Have the authors conducted sensitivity analyses to evaluate how the choice of k = 50 affects performance and stability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces SEAFormer, a transformer architecture designed to address real-world vehicle routing problems (RWVRPs) by incorporating two key innovations: Clustered Proximity Attention (CPA) and an edge-aware module. The method is shown to outperform existing approaches in several benchmarks, but there are some aspects of the method’s novelty, experimental design, and theoretical foundation that could be further clarified or improved.

### Strengths
1. SEAFormer demonstrates competitive performance across multiple RWVRP variants.
2. The introduction of Clustered Proximity Attention (CPA) effectively reduces the computational complexity of traditional attention mechanisms.
3. The edge-aware module provides a practical solution for incorporating edge-level information, enhancing model accuracy and convergence speed.

### Weaknesses
1. The definition of “real-world VRP” is broad, and it is unclear what specific challenges are being addressed. Although variants like VRPTW and EVRPCS are mentioned, the paper lacks a clear explanation of why existing methods cannot be extended to these variants. The motivations behind CPA and the edge-aware module are more engineering-oriented, with little exploration of the underlying theoretical mechanisms.
2. The CPA approach lacks clear innovation when compared to existing local attention or sparse attention mechanisms. The paper should clarify how CPA offers distinct advantages over these existing approaches.
3. The paper does not sufficiently distinguish its contributions from previous work that modifies attention mechanisms with distance (such as [1-5]). A clearer explanation of how SEAFormer differs would strengthen the novelty claim.
4. The statement “SEAFormer is the first unified approach to address this comprehensive range of real-world routing constraints at scales within a single architecture” seems exaggerated, as there are other works that have addressed real-world VRPs, and it is unclear what makes this method the “first”.
5. The deterministic multi-round partitioning and boundary smoothing in CPA seem empirical, without solid theoretical justification or quantitative analysis on the trade-off between complexity and performance.
6. The edge embedding module only works for certain edges (depot-customer), raising concerns about potential information loss across the full graph. The paper should explore whether this limitation affects overall solution quality.
7. Some RWVRP variants, like EVRPCS and VRPRS, are compared with a limited number of learning-based methods. In addition, stronger baseline traditional algorithms, such as HGS, are not considered in many variants.
8. There is no in-depth comparison with existing methods on model parameters, training time, or inference time, which are critical for real-world applications.
9. If SEAFormer claims applicability to “real-world” scenarios, the paper should include experiments on real-world datasets to substantiate this claim.

```
[1] INViT: A generalizable routing problem solver with invariant nested view transformer. ICML, 2024.
[2] Towards generalizable neural solvers for vehicle routing problems via ensemble with transferrable local policy. IJCAI, 2024.
[3] Distance-aware attention reshaping for enhancing generalization of neural solvers. IEEE TNNLS, 2025.
[4] Learning to solving vehicle routing problems via local–global feature fusion transformer. Complex & Intelligent Systems, 2025.
[5] Instance-conditioned adaptation for large-scale generalization of neural routing solver. arXiv, 2025.
```

### Questions
Please refer to the weaknesses.

### Soundness
1

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
This paper presents a Transformer-like framework, called SEAFormer, for solving real-world vehicle routing problems (RWVRP), which integrates both node-level and edge-level information for decision-making. It includes two components, 1) clustered
proximity A=attention (CPA)  for computational efficiency that uses locality-aware clustering and achieving attention with linear complexity; 2) a lightweight edge-aware module that incorporates pairwise relational information into routing decisions for edge-specific constraints. Experimental results show the effectiveness of the proposed method.

### Strengths
1. This paper proposes a novel transformer-based framework for solving four real-world vehicle routing variants with diverse problem sizes.
2. The proposed CPA computes attention scores using locality-aware clustering and achieving O(n) complexity while preserving global perspective.
3. Experimental results show the superiority of the proposed method against baselines across various problem sizes and problem constraints.

### Weaknesses
1. While CPA and edge modules are well-motivated, the ideas of developing attention with linear complexity and exploiting edge-level information are not novel [1][2][3], and they also build on existing sparse attention and residual fusion ideas (e.g., Reformer, FlashAttention, GAT-based methods). 
2. The proposed CPA incorporates some hyper-parameters, e.g., Cluster Size (M) and Partitioning Rounds (R). However, it lacks a systematic analysis of their sensitivity and effects to the model performance.
3. For experimental results, the major superiority of the proposed method seems to be from SGBS inference strategy. For standard VRPs that do not incorporates edge-specific attributes, the superiority of the proposed SEAFormer is not obvious.

[1] Luo F, Lin X, Wu Y, et al. Boosting neural combinatorial optimization for large-scale vehicle routing problems[C]//The Thirteenth International Conference on Learning Representations. 2025.

[2] Meng D, Cao Z, Wu Y, et al. EFormer: An Effective Edge-based Transformer for Vehicle Routing Problems[J]. IJCAI, 2025.

[3] Meng D, Cao, Z, Gao J, Wu Y, et al. UniteFormer: Unifying Node and Edge Modalities in Transformers for Vehicle Routing Problems, NIPS, 2025.

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
3

### Contribution
2
