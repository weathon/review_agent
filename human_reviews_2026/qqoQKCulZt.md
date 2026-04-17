# The Gaussian-Head OFL Family: One-Shot Federated Learning from Client Global Statistics

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Classical Federated Learning relies on a multi-round iterative process of model exchange and aggregation between server and clients, with high communication costs and privacy risks from repeated model transmissions. In contrast, one-shot federated learning (OFL) alleviates these limitations by reducing communication to a single round, thereby lowering overhead and enhancing practical deployability. Nevertheless, most existing one-shot approaches remain either impractical or constrained, for example, they often depend on the availability of a public dataset, assume homogeneous client models, or require uploading additional data or model information. To overcome these issues, we introduce the Gaussian-Head OFL (GH-OFL) family, a suite of one-shot federated methods that assume class-conditional Gaussianity of pretrained embeddings. Clients transmit only sufficient statistics (per-class counts and first/second-order moments) and the server builds heads via three components: (i) Closed-form Gaussian heads (NB/LDA/QDA) computed directly from the received statistics; (ii) FisherMix, a linear head trained on synthetic samples drawn in an estimated Fisher subspace; and (iii) Proto-Hyper, a lightweight low-rank residual head that refines Gaussian logits via knowledge distillation on those synthetic samples. In our experiments, GH-OFL methods deliver state-of-the-art robustness and accuracy under strong non-IID skew while remaining strictly data-free.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Gaussian-Head One-shot Federated Learning (GH-OFL), a data-free framework that transmits only statistical summaries to achieve state-of-the-art accuracy and robustness under non-IID settings with a single communication round. The core framework of data-free is about synthetic labeled pairs for one-shot FL.

### Strengths
- Comprehensive alternatives and corresponding clear discriptions are provided.
- Ablation study on modules shows the importance and effects of each components.

### Weaknesses
- Comparison of the computational overheads between data synthesis and training with public datasets is anlayses with details and empirical evidence.
- Ablation study would better demonstrate the trade-off between communication and performance applying one-shot FL.  
- Baselines of general FL should be provided to general audience to support the advantage of one-shot FL. In Dirichlet of 0.5 cases, FedAvg reach a accuracy at least 86.02 on ResNet-18, which is not mentioned in the main table.

### Questions
- How about a experiments that the normal FL gradually transits into a one-time by reducing the number of communication rounds?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes GH-OFL, a one-shot FL framework where clients send only per-class sufficient statistics (counts and first/second moments of frozen-encoder features, optionally after a public random projection). The server constructs (i) closed-form Gaussian heads (NB-diag/LDA/QDA), (ii) FisherMix, a cosine-margin linear head trained on synthetic Fisher-space samples, and (iii) Proto-Hyper, a low-rank residual head distilled from a Gaussian teacher.

### Strengths
+ The moment set is additively aggregable and sufficient to instantiate NB-diag/LDA/QDA; random-projection sketches reduce bandwidth and preserve additivity. The derivations for means/covariances and pooled covariance are explicit.

+ The paper motivates a Fisher subspace generalized eigen problem and uses data-free Gaussian sampling there to train FisherMix and Proto-Hyper, offering practical gains when closed-form heads are biased.

+ Results compare GH-OFL heads to OFL baselines across datasets and backbones show consistent advantages where expected.

### Weaknesses
- The datasets used are all computer-vision datasets, which limits the scalability of the proposed approach.

- Privacy discussion is not deep and the privacy analysis is qualitative.

- The experiments in this paper are not extensive, i.e., the ablation study of FL (e.g., number of clients) is not well discussed.

### Questions
- Can you provide more experiments on non CV tasks?
- Can you discuss the potential privacy-preserving methods (i.e., DP) integrating with proposed approaches?
- For high-dim encoders, how do diagonal-plus-low-rank sketches compare to full QDA in accuracy vs. memory/compute?

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
This paper presents GH-OFL, a novel family of one-shot federated learning methods that achieves state-of-the-art performance while being strictly data-free and communication-efficient. The work is technically sound, methodologically innovative, and addresses practical constraints effectively.

### Strengths
1. The proposed method is strictly data-free, requiring no public datasets or client-side inference, which is a major advantage for privacy.
2. The use of sufficient statistics and random projection sketches makes the approach highly communication-efficient.
3. The paper is generally well-written, with a clear narrative and a logical flow from problem to solution.

### Weaknesses
1. It relies on the core assumption that the feature embeddings for each class follow a Gaussian distribution, which may not hold perfectly in practice.
2. The Fisher subspace projection, while beneficial for dimensionality reduction, may discard some discriminative information present in the discarded dimensions.
3. The paper does not sufficiently explore the limitations of the Gaussianity assumption for certain types of data.

### Questions
Please refer to weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
2
