# Federated Feature Transformation with Sample-Aware Calibration and Local–Global Sequence Fusion

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Tabular data plays a crucial role in numerous real-world decision-making applica-
tions, but extracting valuable insights often requires sophisticated feature transfor-
mations. These transformations mathematically transform raw data, significantly
improving predictive performance. In practice, tabular datasets are frequently
fragmented across multiple clients due to widespread data distribution, privacy
constraints, and data silos, making it challenging to derive unified and generalized
insights. To address these issues, we propose a novel Federated Feature Transfor-
mation (FEDFT) framework that enables collaborative learning while preserving
data privacy. In this framework, each local client independently computes feature
transformation sequences and evaluates the corresponding model performances.
Instead of exchanging sensitive original data, clients transmit these transforma-
tion sequences and performance metrics to a central global server. The server
then compresses and encodes the aggregated knowledge into a unified embedding
space, facilitating the identification of optimal feature transformation sequences.
To ensure accurate and unbiased aggregation, we employ a sample-aware weight-
ing strategy, assigning higher weights to clients with larger, more diverse, and
numerically stable datasets, as their performance metrics are statistically reliable
and representative. We also incorporate a server-side calibration mechanism to
adaptively refine the unified embedding space, mitigating bias from outlier data
distributions. Furthermore, to ensure optimal transformation sequences at both
global and local scales, the globally optimal sequences are disseminated back to
local clients. We subsequently develop a sequence fusion strategy that blends
these globally optimal features with essential non-overlapping local transforma-
tions critical for local predictions. Extensive experiments are conducted to demon-
strate the efficiency, effectiveness, and robustness of our framework. Code and
data are publicly available

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a framework named FEDFT for federated feature transformation (FFT) on tabular data. Instead of sharing raw data or model gradients, clients generate feature transformation sequences and associated performance scores, which are uploaded to a central server. The server aggregates these records via a sample-aware weighted scheme, trains an encoder–decoder–evaluator network to embed sequences into a shared latent space, and performs gradient-based optimization to explore better global transformation sequences. The globally optimized transformation is then fused with local distinctive features to improve local performance. Experiments on 13 tabular datasets demonstrate improvements over local transformation baselines and some FL methods.

### Strengths
1. This paper introduces a joint embedding–reconstruction–performance estimation design, plus local–global fusion, showing architectural completeness.

2. This paper moves beyond traditional parameter aggregation and explores federated feature transformation, which is underexplored but practically relevant for tabular settings.

### Weaknesses
1. Although raw data is not shared, transmitting performance values can leak distributional statistics. No comparison with established privacy-preserving FL techniques such as differential privacy or secure aggregation. The paper itself acknowledges lack of formal guarantees. 

2. The transformation sequences are symbolic and non-differentiable. Updating continuous embeddings and decoding them does not guarantee validity or monotonic improvement. Valid decode rate is low (Figure 7), and the optimization might be unstable or ill-defined.

3. Works like FLFE and Fed-IIFE are cited but not compared experimentally, limiting the strength of the claimed contribution.

4. Broadcasting sequences to all clients for re-evaluation can be expensive, contradicting efficiency motivations.

5. Some methodological descriptions lack clarity. For example, the KL term is referenced but not precisely defined, hyperparameters (e.g., p, q in weighting) appear heuristic without sensitivity studies, potential leakage in using aggregated validation performance for server optimization

6. The introduction highlights three major challenges that this work aims to address. However, the experiments do not fully demonstrate that the proposed method effectively resolves all of these challenges. As a result, the underlying working mechanism remains unclear, and the claimed contributions are significantly weakened.

### Questions
Does the method truly provide privacy-preserving global feature transformation?

Does the proposed sample-aware weighted aggregation truly reduce bias under non-IID data?

What is the causal mechanism by which global-to-local fusion brings improvements?

### Soundness
2

### Presentation
3

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
This paper proposes a novel federated feature transformation framework designed to construct an optimized and generalizable feature space without sharing raw data.
Each client locally generates feature transformation sequences  and their corresponding predictive performances . These records are uploaded to a central server, where an encoder–decoder–evaluator network jointly learns to embed transformation knowledge into a continuous space .
A gradient-based search in this embedding space identifies the globally optimal transformation sequence that maximizes the aggregated weighted performance across all clients.
The framework aims to improve generalization under heterogeneous data distributions by transferring transformation knowledge rather than raw features or gradients.

### Strengths
Well-designed framework — The encoder–decoder–evaluator architecture is conceptually sound and allows transformation knowledge to be encoded and optimized collaboratively without exposing local data.
Potential for cross-domain generalization — By learning transformation patterns rather than model weights, the framework could generalize better across heterogeneous or non-IID clients.
Privacy-preserving design — The method respects the federated setting and does not transmit raw data, which aligns with privacy-by-design principles.

### Weaknesses
(1) High computational complexity of the RL search process. The optimization of feature transformation sequences through RL or gradient-based exploration in embedding space is likely to be time-consuming. The paper lacks an explicit analysis of the computational and communication costs, which raises concerns about scalability to high-dimensional tabular datasets or large numbers of clients.
(2) Lack of theoretical analysis. Lack of convergence or optimality guarantees for the global optimal sequence search.
(3) Limited comparison with recent federated learning algorithms. The experiments mostly benchmark against older FL baselines (e.g., FedAvg, FedProx). Moreover, since the method is related to feature engineering, the comparisons do not include papers related to feature engineering.
(4) Reinforcement learning design is underexplained.The paper mentions “postfix encoding” and search but does not provide sufficient detail about the RL reward structure, policy updates, or exploration-exploitation balance.

### Questions
Could the authors provide complexity analysis for both local generation of records and server-side embedding training?
Please add comparisons regarding the latest feature engineering related work in federated learning.
Please provide a detailed explanation of the training process in the reinforcement learning part.
Add proofs for the convergence or optimality guarantees of the global optimal sequence search.

### Soundness
2

### Presentation
2

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
This paper proposes federated feature transformation for tabular data. The authors point out that both federated learning and feature transformation have been investigated but their intersection has relatively been unexplored. To alleviate the data heterogeneity issue among various clients, the proposed FedFT employs sample-aware weighting strategy and global-to-local feature integration. Experiments show that FedFT improves performance over the selected baselines.

### Strengths
1. Sample-aware weighting that considers both data size and quality seems effective.
2. Exploring unique challenges for tabular data can be interesting. 
3. Federated feature transformation is relatively unexplored area.

### Weaknesses
- FL has largely been explored for image and text benchmarks but this paper emphasizes tabular data. What is the unique challenge of tabular data compared to other data, such as images and texts? Why not reusing existing techniques for image and text data? Why is the proposal not applied to other data formats? 

- Contributing to local clients via federated learning has been also investigated in personalized federated learning (PFL). How is the proposal differentiated from PFL and what is the performance gap?

- FL typically trains a deep learning model itself in a federated manner. In my understanding feature transformation is already included in the model since it extracts deep representations for each input data. What is the unique value of federated feature transformation compared to federated model training? Conceptual illustration comparing feature transformation and other directions can be helpful.

- References are old considering exploding publications on ML these days, which means the research topic may not be important or outdated. Please add more recent papers and if relevant papers do not exist, the authors should justify why this field has been overlooked.

- Similarly, the FL baselines are old; FedNTD published in 2022 is the latest method in the paper. Baselines should include more recent and advanced methods published in 2024 and 2025.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
