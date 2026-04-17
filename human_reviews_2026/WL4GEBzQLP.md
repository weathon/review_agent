# Beyond Pairwise Modeling: Towards Efficient and Robust Trajectory Similarity Computation via Representation Learning

- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
Accurate trajectory similarity computation is crucial in ride-sharing applications, where trajectories of varying lengths need to be aligned into a uniform representation. Existing methods suffer from reliance on multi-metric supervision and the role-specific encoding required for triplet loss computation, resulting in inefficient computation. To overcome these issues, we move beyond pairwise modeling and propose a novel representation learning framework to achieve efficient and robust trajectory similarity computation, named Hyper2Edge. Hyper2Edge consists of three main components: (i) Hypergraph-based modeling to represent trajectories as hyperedges, instead of single nodes, preserving sequential and structural details; (ii) Hierarchical trajectory representation learning to capture intra- and inter-trajectory patterns; and (iii) A weighted top-$k$ InfoNCE loss to focus on nearest-neighbor relations, addressing the inefficiencies of triplet loss. Evaluated on two public benchmarks, Hyper2Edge achieves an average absolute gain of 7.42% across all evaluation metrics and an average improvement of 45.9% in accuracy compared to state-of-the-art methods, while maintaining competitive training time per epoch on par with the best-performing methods. The code is available at: https://anonymous.4open.science/r/Hyper2Edge-3D2B.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the trajectory similarity computation via representation learning and proposes a hypergraph framework named Hyper2Edge. First, it directly adopts Euclidean-based supervision to learn trajectory representations without relying on specific multi-metric supervisions. Then, this paper devises a hierarchical trajectory representation learning architecture that captures both intra- and inter-trajectory patterns. Furthermore, it introduces a weighted top-$k$ InfoNCE loss to mitigate repetitive encoding of samples. Experimental results demonstrate the effectiveness and efficiency of Hyper2Edge compared with state-of-the-art baselines on the task of trajectory similarity computation.

### Strengths
S1: This paper studies the trajectory similarity computation in Euclidean space, which is crucial for some real-world applications, such as ride-sharing services.

S2: Unlike previous works, this paper eliminates the reliance on multi-metric supervision by learning directly from Euclidean-based similarity labels, and uses InfoNCE loss to overcome the limitations of triplet loss.

S3: This paper also conducts cross-metric experiments, where the model is trained with Euclidean distance supervision and evaluated on various benchmark metrics(e.g., DTW), thereby enhancing the robustness of the model.

### Weaknesses
W1: The novelty of this work seems to be somewhat limited. The paper employs a weighted top-$k$ InfoNCE loss instead of the triplet loss to mitigate repetitive encoding, it has similarities to the approach presented in KGTS. Besides, they do not provide a theoretical analysis of the computational complexity and do not give any efficiency experiments comparing the InfoNCE loss and the triplet loss.

W2: In Section 3.3, the authors claim that the proposed weighted top-$k$ InfoNCE loss enables each trajectory to comprehensively model its similarity relationships with the entire dataset. However, because the top-$k$ weighting reduces non-neighbor contributions to near zero, the loss does not truly incorporate global information and remains fundamentally a locally focused loss.

W3: In Section OVERALL PERFORMANCE (RQ1), previous studies have evaluated model performance using various distance metrics. In contrast, this paper reports results only under Euclidean distance. Consequently, the current experimental results cannot comprehensively assess the model's performance. Additional experiments with alternative distance metrics, such as DTW, Hausdorff, ERP, and EDR, would provide a more complete and convincing evaluation.

W4: Previous studies commonly report performance across multiple distance metrics to achieve a more comprehensive evaluation. However, they can also report results under a single metric (e.g., Euclidean distance). Therefore, the statements in Section Efficiency Evaluation (RQ3)—"other methods must repeat the entire training process for multiple metrics" and "For baselines: (Time per Epoch) × 100 Epochs × 3 Metrics"— are inaccurate. To ensure a fair comparison of efficiency, the baseline models should also be evaluated under a single distance metric rather than three.

W5: In Section Ablation Study (RQ4), this paper replaces the InfoNCE loss with the MSE loss to demonstrate the effectiveness of the weighted top-$k$ InfoNCE loss. However, since the paper frequently compares the triplet loss with the InfoNCE loss, it may be more reasonable to use the triplet loss as a variant without weighted top-k InfoNCE loss instead of MSE loss.

### Questions
Please see W1-W5.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles trajectory similarity computation and the inefficiencies of pairwise/triplet-based supervision. The authors propose hyper2Edge, which models trajectories as hyperedges in a hypergraph and performs hierarchical representation learning with bidirectional message passing between nodes and hyperedges. A weighted Top-k InfoNCE objective aligns representations directly in Euclidean space, aiming to emphasize nearest neighbors while reducing redundant encoding. Experiments on standard trajectory benchmarks report strong accuracy and notable training-time improvements.

### Strengths
- Clear motivation: moving from local pairwise/triplet supervision to distribution-level alignment that matches Euclidean retrieval at inference.
- Method design is coherent: hypergraph construction + hierarchical message passing capture both intra- and inter-trajectory structure; Top-k InfoNCE focuses learning on the most relevant neighbors.
- Sufficient Experiment: multi-dataset evaluation with ablations and efficiency reporting; results indicate both effectiveness and speedups.

### Weaknesses
- Using Euclidean distance as the sole supervision target may be brittle under sampling-rate shifts, noise, or scale changes; broader robustness analysis would help.
- Efficiency analysis could further disentangle the contributions of fewer encodings, negative sampling, and graph construction costs.

### Questions
1. Under strong scale or non-rigid temporal distortions, does Euclidean-based supervision induce representation collapse or bias? Any hybrid with learnable metrics considered?  
2. Can you provide a finer breakdown of the reported training-time reduction (e.g., encoding passes, sampling overhead, graph ops)?  
3. How sensitive is performance to hypergraph sparsity and tokenization choices (e.g., clustering granularity)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes Hyper2Edge, a novel framework for trajectory similarity computation. The framework aims to address two major limitations of existing methods: reliance on multi-metric supervision and redundant encoding in triplet loss computation, both of which lead to inefficiency. The key idea of Hyper2Edge is to model trajectories as hyperedges in a hypergraph, thereby preserving both their sequential and structural characteristics. The framework consists of three main components: (i) hypergraph-based trajectory modeling; (ii) a hierarchical representation learning architecture that captures both intra- and inter-trajectory patterns; and (iii) a weighted top-k InfoNCE loss function designed to optimize global similarity while emphasizing nearest-neighbor relationships, effectively replacing the less efficient triplet loss. Experimental results on two public benchmark datasets, GeoLife and Porto, demonstrate that Hyper2Edge outperforms state-of-the-art methods in both accuracy and training efficiency.

### Strengths
S1: This paper proposes a novel framework for learning the representation of trajectories by modeling them as hyperedges, which is significantly different from traditional methods that treat trajectories as single nodes or rely on pairwise computation.

S2: The framework achieves strong performance while maintaining low computational cost.

S3: The paper is exceptionally well-written and easy to follow.

### Weaknesses
W1: The dataset scale is limited, making it difficult to convincingly validate the method’s effectiveness.

W2: The interpretability of Hyper2Edge is a potential concern, as it is unclear how well the learned trajectory embeddings reflect human-perceivable semantic similarity.

### Questions
Q1: Why the max length is 50, while min length is 200 in Table 3?

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
This paper proposes a new trajectory similarity computation approach by representing trajectories as hyperedges and designing a hierarchical trajectory embedding mechanism to learn trajectory representations. A weighted InfoNCE loss is then used for optimization while preserving the top-k similarity ranking. Experiments on two real-world datasets demonstrate the effectiveness and efficiency of the proposed method.

### Strengths
S1. The paper is well-written, and the motivation is clearly stated.

S2. The design choice of avoiding multiple similarity metrics for supervision, leading to improved efficiency, is interesting.

### Weaknesses
W1. The proposed method lacks novelty. Techniques such as hypergraph representation and the InfoNCE loss function are well-established and have been widely used in graph data learning.

W2. The performance improvement is not significant. In particular, it improves the baselines by less than 5% in most cases, and the efficiency gains are similarly modest.

W3. It is unclear why the temporal information of trajectories is not incorporated into the trajectory representation learning. Temporal aspects are crucial, as trajectories with identical spatial paths but different timestamps may represent completely different behavioral patterns.

### Questions
Q1. The method combines existing ideas (e.g., hypergraph modeling and InfoNCE optimization) without introducing new components. What's the most important part that readers can learn from this paper?

Q2. Why the performance improvement is relatively minor? 

Q3. Why is the temporal information of trajectories not considered?

### Soundness
3

### Presentation
3

### Contribution
2
