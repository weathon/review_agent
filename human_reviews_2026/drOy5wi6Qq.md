# Less Is More: Clustered Cross-Covariance Control for Offline RL

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
A fundamental challenge in offline reinforcement learning is distributional shift. Scarce data or datasets dominated by out-of-distribution (OOD) areas exacerbate this issue. Our theoretical analysis and experiments show that the standard squared error objective induces a harmful TD cross covariance. This effect amplifies in OOD areas, biasing optimization and degrading policy learning. To counteract this mechanism, we develop two complementary strategies: partitioned buffer sampling that restricts updates to localized replay partitions, attenuates irregular covariance effects, and aligns update directions, yielding a scheme that is easy to integrate with existing implementations, namely Clustered Cross-Covariance Control for TD ($C^4$). We also introduce an explicit gradient-based corrective penalty that cancels the covariance induced bias within each update. We prove that buffer partitioning preserves the lower bound property of the maximization objective, and that these constraints mitigate excessive conservatism in extreme OOD areas without altering the core behavior of policy constrained offline reinforcement learning. Empirically, our method showcases higher stability and up to 30% improvement in returns over prior methods, especially with small datasets and splits that emphasize OOD areas.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper analyzes the TD (temporal-difference) regression objective by decomposing the TD second moment and shows that directly minimizing the squared TD loss induces a harmful cross-time covariance between current and next state–action gradients. Building on this diagnosis, the authors propose two complementary fixes: (i) clustered, single-cluster minibatch sampling over stacked gradient pairs, which removes between-cluster covariance; and (ii) an explicit gradient-based corrective penalty that offsets within-cluster covariance during critic updates. They argue these modifications are “plug-and-play” for common offline RL pipelines and preserve policy-constraint objectives. Experiments on reduced-data D4RL settings report consistent gains (often sizeable) over several baselines, particularly under scarce data and OOD-emphasized splits. The paper also provides proofs and a training recipe.

### Strengths
Clear theoretical diagnosis of a TD-specific failure mode. The decomposition explicitly isolates the cross-time covariance term unique to TD learning, clarifying why standard TD minimization can be destabilizing under weak coverage. 


The clustered sampling + Frobenius-norm penalty is well-motivated, easy to integrate, and provably controls the harmful covariance while keeping the optimization intent intact.

On reduced-size D4RL locomotion (10k samples), the method substantially outperforms listed baselines; Table 1 and the accompanying discussion highlight strong average gains and improved stability.

### Weaknesses
While the theory shows TD minimization tends to raise the cross-term, the paper could better quantify how much covariance growth translates into return degradation and in what regimes (e.g., through controlled interventions that vary covariance at fixed value error).


Despite some Maze2D/AntMaze numbers in the appendix, there is no focused evaluation on challenging sparse-reward benchmarks like AntMaze with standard sparse settings and strong recent baselines, which would directly test the claimed robustness under weak signal

Baseline set skews older in places. Several comparisons rely on classical offline RL baselines; inclusion of more recent small-data/OOD-aware methods would make the empirical case more convincing.

### Questions
See weaknesses.

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
4

### Summary
This paper studies instability in offline RL under distributional shift, particularly in small-data regimes.
The authors identify a previously under-analyzed phenomenon: the TD loss induces a harmful cross-time gradient covariance that biases optimization in OOD regions.
To mitigate this, the paper introduces Clustered Cross-Covariance Control (C4), composed of two mechanisms:
(1) Clustered buffer sampling — partitions the replay buffer by gradient features and samples mini-batches from single clusters to remove between-cluster covariance;
(2) Gradient-based corrective penalty — penalizes the Frobenius norm of within-cluster cross-covariance to reduce harmful coupling.

Theoretical analysis shows that C4 preserves the lower-bound property of conservative objectives (e.g., CQL) and bounds TD instability, while experiments on reduced-size D4RL tasks (10k samples) show improved performance and stability compared to CQL, IQL, and DOGE.

### Strengths
Originality: The work offers a novel theoretical perspective linking TD error variance to cross-time gradient covariance. 

Technical quality: The derivations are mathematically sound and clear. Theorem 1 decomposes the TD variance into beneficial (variance) and harmful (covariance) components, while Theorem 2 proves that single-cluster sampling removes between-cluster interference.
The proposed loss (Eq. 12–14) is principled and compatible with existing algorithms.

Clarity: The paper is clearly structured, easy to follow, and supported by illustrative figures (e.g., Fig. 2). Algorithm 1 is concise and implementable.

Significance: The idea of controlling TD-induced covariance is relevant for improving stability and generalization in offline RL with small datasets. The method is plug-and-play and conceptually simple, which could make it attractive for integration into broader offline RL pipelines.

### Weaknesses
Outdated Baselines: The empirical evaluation does not include recent state-of-the-art methods, particularly A2PR (Adaptive Advantage-Guided Policy Regularization, arXiv:2405.19909), which achieves significantly higher returns on the same D4RL tasks.
Without comparing to such strong baselines, the claimed 30% improvement is not convincing. **Please answer the concern in the response.**

Experimental Scope and Depth: Experiments are limited to low-dimensional MuJoCo tasks with 10k samples; no results on real-world robot control or discrete domains (e.g., Atari, Adroit).

The main text lacks ablation studies (e.g., number of clusters K, λ penalty weight, clustering frequency).

The claimed “computational efficiency” of clustering is not quantified.

Theoretical Assumptions: The first-order Taylor expansion (Eq. 2) assumes smooth linearity in feature space; this may not hold in deep nonlinear critics, yet this limitation is not discussed.

### Questions
Why were recent offline RL algorithms such as A2PR (arXiv:2405.19909) and On-Policy Regularized IQL excluded?

How would C4 perform against them under the same 10k-sample setting?

How often are gradient clusters updated during training? Does dynamic re-clustering significantly affect performance or computational cost?

Can the proposed gradient-space clustering scale to high-dimensional visual inputs or large replay buffers (e.g., 1M samples)? What is the per-iteration complexity?

How should λ and K be chosen in practice? Is there a heuristic based on dataset size or OOD ratio?

### Soundness
2

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
3

### Summary
The paper analyzes the standard temporal difference (TD) loss used for training in offline reinforcement learning (RL), and identifies a cross-covariance term that acts as a harmful implicit regularizer. Based on this observation, the paper proposes an algorithm $C^4$ (Clustered Cross-Covariance Control) to mitigate the cross-covariance term through buffer partitioning and an explicit cross-covariance penalty term. Theoretical results demonstrate that these modifications are compatible with standard offline RL policy update methods, and experimental results demonstrate that these modifications lead to significant improvement across a range of offline RL benchmarks when small datasets are considered.

### Strengths
**[S1]** The paper presents novel and interesting analysis that identifies a cross-covariance term that acts as a harmful implicit regularizer in the standard TD loss function, and proposes algorithmic modifications to address this issue in offline RL.

**[S2]** Theoretical results are provided that justify the algorithmic design choices and demonstrate their impact on other components of standard offline RL algorithms such as the policy improvement update.

**[S3]** Extensive experimental results across several offline RL benchmarks demonstrate the strong performance of $C^4$ on small datasets compared to several baseline methods (unfortunately, most of this analysis is deferred to the appendix).

### Weaknesses
**[W1]** Experiments primarily focus on small datasets and do not include results on the benchmarks for the standard dataset size (e.g., Figure 6 considers a max dataset size that is 10% of the full dataset in most cases), so it is difficult to understand if there are performance tradeoffs on large datasets in order to achieve robust performance on small datasets.

**[W2]** The organization and presentation of the work could be improved. In particular, the organization of Sections 5 and 6 required the reader to extract why results are important and how they relate to Algorithm 1. Some of the notation makes it difficult to follow the results (e.g., $C$ and $c$ are used in multiple ways throughout Section 5). The length of Sections 5 and 6 also led to a very brief experiments section in the main paper, even though the authors have done rather extensive experimental analysis that would be valuable to include instead of deferring it to the appendix.

### Questions
**[Q1]** What is the intuition behind why the cross-covariance term present in the standard TD loss is harmful for performance?

**[Q2]** Is there a performance tradeoff on large datasets in order to achieve strong performance on small datasets, or does the method automatically adjust to the large dataset scenario? Please include experimental results on the standard dataset sizes to help understand this. Experiments on the default size of benchmark datasets would also make it easier to compare the results in this paper to other results available in the literature.

**[Q3]** I would recommend reorganizing Sections 5 and 6 to clearly highlight the main contributions upfront, and only including results and discussions that are most critical for providing theoretical support and intuition of the approach. This would provide additional space to include a more meaningful experiments section in the main paper, rather than deferring most of the results to the appendix.

**[Q4]** Additional clarifications on implementation and results:
- Does the fact that the TD loss is only optimized with respect to the current Q function (gradients are stopped for the TD target) impact the analysis and conclusions?
- Why does the proposed regularization term in (13) contain two components?
- What is the justification for approximating the gradient w.r.t. the input $x$ as the network’s penultimate layer (line 1091)? My understanding is this was done in DR3 (Kumar, 2022) to approximate the gradient w.r.t. the Q function network parameters, not the gradient w.r.t. the Q function input.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper investigates a fundamental source of instability in temporal-difference (TD) learning for offline reinforcement learning (RL). The authors argue that the primary cause is the cross-time covariance between feature gradients of consecutive transitions, which acts as a harmful implicit regularizer. This covariance term biases TD updates and leads to critic collapse, particularly under weak data coverage or out-of-distribution (OOD) regions. To address this, the paper proposes C4 (Clustered Cross-Covariance Control), a simple yet theoretically grounded method that clusters gradient pairs *$(g',g)$* and applies covariance regularization within each cluster. This localized control effectively suppresses cross-covariance while preserving the TD objective’s lower bound. Empirical results on D4RL MuJoCo benchmarks demonstrate substantial performance gains—especially in low-data regimes—showing improved stability and data efficiency.

Although this issue may appear specific to small datasets, the underlying mechanism suggests that it will become even more significant as environments grow larger or more complex. When the environment’s state or action space expands while the dataset size remains fixed, coverage sparsity worsens, amplifying the harmful covariance effects. Hence, the insight and methodology presented here are expected to remain increasingly relevant as future offline RL benchmarks evolve toward more realistic and high-dimensional settings.

### Strengths
The paper impressively connects a theoretically grounded understanding of TD learning instability with a practical, well-performing algorithm. The C4 framework stands out for translating a nontrivial analytical insight into empirically robust improvements, showing both conceptual clarity and engineering maturity.

**Strong theoretical foundation**: The paper begins from a clear theoretical diagnosis of a subtle but critical problem in temporal-difference (TD) learning — the emergence of harmful cross-time covariance under distributional shift and limited data. The authors derive this effect formally and distinguish it from other, beneficial forms of implicit regularization. This analytical depth provides a precise and original explanation for instability in offline RL, rather than relying on empirical speculation.

**Elegant link from theory to practice**: One of the most compelling aspects is how the theoretical insight leads directly to a concrete and empirically effective solution. The transition from analyzing TD covariance to designing the Clustered Cross-Covariance Control (C4) algorithm shows an excellent bridge between mathematical reasoning and algorithmic implementation.

**Consistent and plentiful empirical validation and ablation studies**: The experiments on D4RL MuJoCo benchmarks demonstrate that the theory translates into tangible performance benefits: smoother learning curves and more than 30% improvement in return under small data and OOD-emphasized splits. Even though the authors do not deeply explore the optimal number of clusters, the empirical results with *$K=5$* show that the method works reliably across tasks without extensive tuning—indicating good robustness in practice. 

Apart from that, the paper includes extensive ablation studies on different factors such as dataset size, which reinforce the soundness and robustness of the proposed approach.

### Weaknesses
I have only on weakness to point out.

**Sensitivity to the number of clusters K**:

The proposed method critically depends on the number of clusters used to partition the gradient feature space. If *$K$* is too small, dissimilar gradient modes are mixed, leaving between-cluster covariance unremoved; if *$K$* is too large, covariance estimation becomes noisy due to small sample counts per cluster. However, the paper fixes *$K$* (typically 5) for all experiments without analyzing sensitivity or scalability. It remains unclear how robust C4 is to this choice or how *$K$* should scale with data size or environment complexity.

### Questions
**Cluster sensitivity**:

How sensitive is performance to the number of clusters *$K$*? Did you observe degradation or instability for smaller/larger *$K$* values? Could an adaptive mechanism (e.g., based on gradient variance or cluster entropy) help automate cluster selection?

### Soundness
3

### Presentation
3

### Contribution
3
