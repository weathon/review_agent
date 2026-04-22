# Instance-Prototype Affinity Learning for Non-Exemplar Continual Graph Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Graph Neural Networks (GNN) endure catastrophic forgetting, undermining their capacity to preserve previously acquired knowledge amid the assimilation of novel information. Rehearsal-based techniques revisit historical examples, adopted as a principal strategy to alleviate this phenomenon. However, memory explosion and privacy infringements impose significant constraints on their utility. Non-Exemplar methods circumvent the prior issues through Prototype Replay (PR), yet feature drift presents new challenges. In this paper, our empirical findings reveal that Prototype Contrastive Learning (PCL) exhibits less pronounced drift than conventional PR. Drawing upon PCL, we propose Instance-Prototype Affinity Learning (IPAL), a novel paradigm for Non-Exemplar Continual Graph Learning (NECGL). Exploiting graph structural information, we formulate Topology-Integrated Gaussian Prototypes (TIGP), guiding feature distributions towards high-impact nodes to augment the model's capacity for assimilating new knowledge. Instance-Prototype Affinity Distillation (IPAD) safeguards task memory by regularizing discontinuities in class relationships. Moreover, we embed a Decision Boundary Perception (DBP) mechanism within PCL, fostering greater inter-class discriminability. Evaluations on four node classification benchmark datasets demonstrate that our method outperforms existing state-of-the-art methods, achieving a better trade-off between plasticity and stability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies continual graph learning without storing past samples and focuses on reducing catastrophic forgetting caused by prototype drift. It identifies three pitfalls: simple mean prototypes that ignore node importance, feature distillation that overconstrains the space, and single prototypes that blur class boundaries. The method, IPAL, combines Topology Integrated Gaussian Prototypes that weight nodes by PageRank, Instance Prototype Affinity Distillation that aligns instance to prototype relations, and a Decision Boundary Perception mechanism that stresses high uncertainty cases inside a prototype contrastive learning objective. On four node classification benchmarks, the approach improves average accuracy and lowers forgetting compared with regularization, rehearsal, and prior non exemplar methods, approaching or surpassing replay while maintaining a better balance between stability and plasticity.

### Strengths
S1: Well-grounded motivation: Replaces feature-level alignment with instance–prototype relations and employs PCL to suppress drift; the rationale is clear and verifiable.

S2: Cohesive design: TIGP, IPAD, and DBP respectively address topology-aware weighting, the stability–plasticity trade-off, and boundary shaping; experiments show consistent gains over diverse baselines.

### Weaknesses
W1. Cost and scalability: PageRank seems manageable, but on very large graphs with frequent tasks the online prototype updates and DBP mining may bottleneck throughput. Please add end-to-end runtime, peak memory, scaling curves on fixed hardware, and sensitivity/throughput analyses for $S_t$, $K$, and $\beta$.

W2. Theoretical support: The KL analysis is intuitive but lacks assumptions that yield tighter PCL drift bounds and provable adaptivity of $\gamma$ and $\beta$.

W3. Boundary hard-example selection: DBP uses entropy as a boundary proxy, yet class-wise miscalibration can bias a single threshold. Provide class-level calibration/uncertainty statistics and compare mining by entropy, margin, energy, and temperature-calibrated scores to establish effectiveness.

### Questions
see the weaknesses above

### Soundness
3

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
3

### Summary
The paper presents Instance-Prototype Affinity Learning (IPAL), a framework for Non-Exemplar Continual Graph Learning (NECGL) that mitigates catastrophic forgetting without storing prior samples. Built upon Prototype Contrastive Learning (PCL), IPAL introduces three key components: Topology-Integrated Gaussian Prototypes (TIGP) to incorporate graph topology via PageRank-based node importance, Instance-Prototype Affinity Distillation (IPAD) for flexible relation-based knowledge retention, and Decision Boundary Perception (DBP) to enhance class separability using boundary-aware samples. Experiments on four benchmark datasets demonstrate that IPAL consistently improves performance and achieves a better trade-off between stability and plasticity compared with recent state-of-the-art methods.

### Strengths
1. The paper addresses a meaningful challenge in non-exemplar continual graph learning where privacy and memory constraints prevent rehearsal.
2. The three modules (TIGP, IPAD, DBP) are well-motivated, mutually complementary, and grounded in clear intuition.
3. Extensive experiments across four benchmarks with ablation and sensitivity analyses support the effectiveness of each component.

### Weaknesses
1. The framework is an evolutionary extension of PCL and prototype replay ideas rather than a fundamentally new paradigm.
2. The paper lacks concrete analysis of computational cost, parameter overhead, or runtime efficiency.

### Questions
What is the relationship with the related work [1]?
[1] Chaoxi Niu, Guansong Pang, Ling Chen, and Bing Liu. Replay-and-forget-free graph classincremental learning: A task profiling and prompting approach. Advances in Neural Information
Processing Systems, 37:87978–88002, 2024b.

### Soundness
3

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
3

### Summary
The paper studies the problem of Non-Exemplar Continual Graph Learning (NECGL), where models must learn new graph tasks without storing past data. The authors build upon Prototype Contrastive Learning (PCL) and propose a new framework, Instance-Prototype Affinity Learning (IPAL), to alleviate feature drift and catastrophic forgetting. IPAL introduces three modules: Topology-Integrated Gaussian Prototypes (TIGP) for topology-aware prototypes, Instance-Prototype Affinity Distillation (IPAD) for flexible knowledge retention, and Decision Boundary Perception (DBP) for better class separation. Experiments on four benchmark datasets demonstrate consistent improvements over existing Non-Exemplar methods, suggesting that IPAL achieves a better balance between stability and plasticity.

### Strengths
1. The use of PageRank-based topology integration (TIGP) is a meaningful attempt to incorporate graph structural importance into prototype construction, improving representation of high-impact nodes.

2. The proposed Instance-Prototype Affinity Distillation (IPAD) provides a more flexible alternative to traditional feature distillation, maintaining inter- and intra-class relations without over-constraining the feature space.

3.The Decision Boundary Perception (DBP) mechanism is a thoughtful addition that leverages high-entropy (hard) samples near class boundaries to enhance inter-class separation.

### Weaknesses
1. It is not entirely clear why PageRank-weighted nodes would yield better class prototypes, as the paper provides limited explanation or analysis for this design choice. High-centrality nodes may not adequately represent peripheral or low-degree nodes, potentially biasing the prototypes toward graph centers.

2. The framework introduces several additional components but does not report training time or memory usage. It remains unclear whether these modules introduce notable computational or memory overhead.

3. While the paper includes some parameter analysis, it remains unclear how much re-tuning would be required when applying IPAL to new datasets. The current experiments focus on a fixed set of benchmarks, and it would be helpful to clarify whether the same hyperparameter settings generalize well or if dataset-specific tuning is necessary.

### Questions
1. Could the authors provide more explanation or empirical evidence on why PageRank weighting leads to better prototype representations? Have they compared it with other node-importance measures (e.g., degree centrality or attention-based scores)?

2. How significant is the computational overhead introduced by PageRank computation, Mixup synthesis, and drift compensation? Any quantitative report on training time or GPU memory usage would be helpful.

3. Regarding hyperparameters (γ, β, K, |Sₜ|), could the authors clarify whether the same configuration works well across all datasets, or if substantial re-tuning is needed for different graph domains?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles non-exemplar continual graph learning (NECGL)—updating a GNN over a sequence of class-incremental tasks when no past examples may be stored. It observes that the usual prototype replay (PR) with cross-entropy suffers from feature drift as the encoder changes over tasks, and shows that prototype contrastive learning (PCL) drifts less, backed by a KL-divergence analysis and visual evidence. Building on that, the authors propose IPAL (Instance-Prototype Affinity Learning): a PCL-based framework that injects graph topology into prototypes, distils relations between instances and prototypes, and sharpens decision boundaries using hard examples. Across four node-classification benchmarks, IPAL improves average performance while maintaining a good stability–plasticity trade-off.

### Strengths
1. Sound theoretical footing for using PCL in NECGL:
The authors formalize “feature drift” and prove (via a KL-divergence analysis under Gaussian assumptions) that Prototype Contrastive Learning (PCL) incurs strictly less drift than conventional Prototype Replay (PR). This gives a clear, principled reason to prefer PCL in non-exemplar continual graph learning, not just an empirical hunch.

2. Topology-aware prototypes that actually use the graph:
The Topology-Integrated Gaussian Prototypes (TIGP) weight class statistics by PageRank, so influential nodes shape the prototype more than peripheral ones, an intuitively correct use of graph structure that plain mean prototypes ignore. The paper also notes PageRank is computed once per task, keeping overhead low. 

3. Relation-based distillation that preserves plasticity:
The Instance–Prototype Affinity Distillation (IPAD) regularizes instance and prototype relations instead of full feature vectors. The paper argues and illustrates that classic feature distillation can over-constrain the encoder and impede learning new classes, whereas IPAD aligns naturally with PCL and avoids that rigidity. This is a concrete, easy-to-adopt design change with clear motivation.

4. Explicit boundary sharpening to tackle single-prototype ambiguity:
The Decision Boundary Perception (DBP) module identifies high-entropy  nodes near boundaries and folds them into the PCL objective, directly encouraging inter-class separation, a targeted fix for ambiguity when one prototype can’t capture class diversity.

### Weaknesses
1. Theory relies on restrictive assumptions and local approximations:
The “PCL drifts less than PR” claim is proved under multivariate Gaussian feature distributions with positive-definite covariances, and the derivation uses a first-order Taylor approximation of the encoder’s update (i.e., infinitesimal step). That makes the result sensitive to non-Gaussian embeddings and larger optimization steps typical in practice. Clarifying the conditions (e.g., step size, τ, optimizer) under which the strict inequality provably holds—or adding stress tests for non-Gaussian / heavy-tailed features—would strengthen the claim. 
2. Computational overhead is asserted, not quantified:
The method claims “PageRank is computed once per task … imposing no extra burden”, but there’s no runtime or asymptotic analysis versus baselines, it’s important for large graphs where PageRank and per-class covariance can still be costly. Please report training time, memory, and scaling curves.
3. Single-prototype limitation is acknowledged but not fully addressed:
The paper itself notes that a single class prototype can be inadequate and causes inter-class ambiguity; DBP then adds hard instance mining yet still keeps one prototype per class. A more direct fix (e.g., multi-prototype / mixture per class or topology-aware subclusters) is not explored. Including a comparison to a multi-prototype variant would test whether DBP fully compensates for intra-class diversity.

### Questions
Q1. Theory scope & assumptions:
What concrete conditions (on feature distributions, optimizer/step size, temperature τ) are required for Theorem 1 to hold in practice, and how sensitive is the result beyond Gaussian features and first-order (infinitesimal-step) approximations?

Q2. Runtime & scalability claims:
You state PageRank is computed once per task with “no extra burden on training.” Could you provide wall-clock time, memory, and asymptotic complexity on the largest graph (Reddit-CL), and compare to PR baselines?

### Soundness
3

### Presentation
3

### Contribution
2
