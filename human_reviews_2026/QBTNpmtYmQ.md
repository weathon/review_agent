# Global Pivots, Local Unknowns: Stable Federated Open-Set Semi-Supervised Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
We introduce Federated Open-Set Semi-Supervised Learning (FOSSL), a new and practically important federated learning setting where the server holds a small labeled set of in-distribution (ID) classes while clients provide only unlabeled, non-IID data that may include unknown classes. This setting is under-explored and presents two key challenges: pseudo-label brittleness under distributed OOD contamination and amplified heterogeneity arising from diverse OOD categories across clients. These challenges cause conventional federated SSL or centralized OSSL pipelines to become unstable when applied directly.
We propose OpenFL, a server-guided framework designed to remain robust under these FOSSL-specific difficulties. OpenFL stabilizes global training via a round-wise EMA model, maintains class-level pivots as global anchors for representation learning, and aggregates clients using reliability-aware weights. Clients perform gated pivot alignment, strengthening ID-consistent updates while suppressing the influence of uncertain or OOD-prone samples.
Across CIFAR-10, CIFAR-100, and FashionMNIST with diverse inlier/outlier splits and unseen OOD tests, OpenFL improves both ID accuracy and OOD detection while maintaining stable training. This work establishes FOSSL as a benchmark problem and provides a principled framework for learning under unlabeled, open-set, and highly heterogeneous federated environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
1. The paper formalizes Federated Open-Set Semi-Supervised Learning (FOSSL) in a "labels-at-server" regime, where clients hold only unlabeled, non-IID, open-set data, posing challenges of pseudo-label brittleness and intensified heterogeneity from diverse unknown classes.
2. It proposes OpenFL, a server-guided framework that stabilizes training using three components: Round-wise EMA (R-EMA) for a stable server-side model, Pivot-guided Open-set Alignment to guide clients with stable class references, and Reliability-Aware Aggregation (RAA) to weight clients by update quality rather than data size.
3. OpenFL consistently improves both in-distribution (ID) accuracy and out-of-distribution (OOD) detection (AUROC) across CIFAR-10, CIFAR-100, and FashionMNIST, remaining stable where federated baselines fail.

While effective, the work is primarily compositional in nature, drawing from established techniques, and thus fails to provide sufficient novelty.

### Strengths
1. The open-set formulation is novel.
2. All the proposed components of the training method are effective and improve performance empirically.

### Weaknesses
1. Using a moving average at the server has been done before, both in a centralized iteration-based fashion and in a federated round-based setting[1,2].
2. Guiding aggregation weights by the loss has been done before [3,4], using specific losses is not sufficient grounds for novelty.
3. No theoretical guarantees are provided.
4. All experiments are conducted on small-scale computer vision datasets (FashionMNIST, CIFAR-10, CIFAR-100) using a relatively shallow backbone. These benchmarks do not adequately represent the challenges of modern deep learning. It is unclear if the proposed methods would remain effective when fine-tuning or pre-training large-scale foundation models.

[1] Zhang, et.al; "How Does Critical Batch Size Scale in Pre-training?"

[2] Zhou, et.al; "Understanding and Improving Model Averaging in Federated Learning on Heterogeneous Data"

[3] Li, et.al; "Fair Resource Allocation in Federated Learning"

[4] Li, et.al; "Tilted Empirical Risk Minimization"

### Questions
1. How does the computational complexity of your method scale as the model size increases, particularly with the embedding dimension and the number of classes?

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
This paper attempts to address the problem of Federated Open-Set Semi-Supervised Learning (FOSSL) in labels-at-server setting. In this setting, a central server holds a small amount of labeled ID data, while clients possess only unlabeled data that contains both ID and OOD samples. The authors claim that existing methods fail due to pseudo-label brittleness and data heterogeneity. They propose OpenFL,which combines three main components: (1) R-EMA model on the server, (2) a pivot-guided alignment where clients align high-confidence samples to server-computed class prototypes, and (3) RAA scheme that weights clients based on the inverse of their alignment loss. The experiments show that their method achieves great ID accuracy and OOD detection.

### Strengths
1. The empirical study in Section 3, which demonstrates the failure modes of naively applying existing FSSL and OSSL methods to this setting, providing a clear motivation for the problem.
2. The paper is well-written and most parts are clearly explained.

### Weaknesses
1. OpenFL appears to be little more than a combination of existing, well-known ideas stitched together, such as exponential moving average (common strategy in SSL methods) and prototype learning. So in my opinion, the contribution is negligible.
2. The use of globally fixed thresholds for the dual-gate selection is sub-optimal. A good confidence score on a client with clean data might be a bad one on a client swamped with OOD samples.
3. The federated adaptations of centralized OSSL methods, particularly FedSCOMatch and FedProSub, perform exceptionally poorly, often leading to model collapse. Can the authors provide evidence that these are not strawman implementations? Please detail the specific adaptation strategies and hyperparameters used, and justify why you believe this represents a fair comparison.

### Questions
See in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work tackles the challenging Federated Open-Set Semi-Supervised Learning setting where labeled training data is uniquely located on the server. The authors proposed OpenFL, a novel framework that enables the server to control the federated training and guide clients to meaningfully contribute to the global model, even though they don’t possess any labelled training samples. OpenFL comprises a series of techniques such as round-wise exponential moving average, global pivots, and reliability-aware client weights aggregation.

### Strengths
- S1. The authors tackle a challenging and critical federated learning setting.
- S2. The authors propose a full end-to-end method to robustly train federated models where clients' training data is completely unlabelled.
- S3. The proposed method’s ML performance is evaluated across standard benchmarks in a meaningful setting.

### Weaknesses
- W1. Limited novelty – Many components of OpenFL (e.g., using a global pivot model or EMA updates) have been proposed previously (e.g., in recent federated semi-supervised methods like FedAnchor [1]). While OpenFL’s combination of these techniques in the FOSSL setting is useful, the approach feels incremental rather than introducing a fundamentally new concept.
- W2. Complex tuning – OpenFL introduces numerous new hyperparameters (e.g., for the EMA decay, pivot selection, client weighting). This added complexity could make the method hard to tune in practice, potentially limiting its real-world applicability. This concern is heightened by the fact that the experiments were on well-established benchmarks with presumably careful tuning; deploying OpenFL in the wild might be challenging without guidance on choosing these hyperparameters.
- W3. Limited evaluation scope – The experimental settings are not fully representative of challenging real-world federated scenarios. For instance, the paper evaluates on at most 20 clients with reasonably large local datasets, but does not test cases with a huge number of clients or with extremely scarce data per client. This omission leaves it unclear how OpenFL performs in more extreme or realistic federated conditions (e.g., hundreds of clients or clients with only a handful of samples).

[1] Xinchi Qiu, Yan Gao, Lorenzo Sani, Heng Pan, Wanru Zhao, Pedro PB Gusmao, Mina Alibeigi, Alex Iacob, and Nicholas D Lane. Fedanchor: Enhancing federated semi-supervised learning with label contrastive loss for unlabeled clients. arXiv preprint arXiv:2402.10191, 2024.

### Questions
- Q1. The abstract currently spends a lot of space on method details. Could the authors revise it to highlight the key challenges of the FOSSL setting more explicitly, rather than the implementation specifics of OpenFL?
- Q2. Can the authors clarify how abundant they assume the server training dataset is compared to the local client datasets? It is critical for setting the context of the applicability of the OpenFL method.
- Q3. Can the authors discuss more explicitly in the introduction what key challenges the components of OpenFL are meant to address?
- Q4. How would OpenFL perform in scenarios of extreme data scarcity? Consider two cases: (a) the server’s labeled dataset is very scarce relative to clients (e.g., only 1–10% the size of the total client data), and (b) each client’s local dataset is so small that a full batch can’t be formed without reusing data. Can the authors discuss how OpenFL would handle these situations?
- Q5. Despite being unpublished work, how do the authors think OpenFL compares to Fedanchor [1]? I would like to read their opinion comparing the two methods on: (a) general setting; (b) motivating examples; (c) basic working principle; (d) performance (if they have sufficient time to try to reproduce, but this point is not crucial).
- Q6. Can the authors add a fitting line in Figure 3 to help readability and quantify the correlation they claim?
- Q7. How many global pivots does OpenFL require to perform sufficiently well?
- Q8. Can the authors add to each table and figure reporting results from the ablation/sensitivity studies (tables 2 and 3, and figure 4) a horizontal line showing the performance of the best baseline method as well?
- Q9. Given that the authors used open-source software to implement and test OpenFL, will they make the code publicly available?

[1] Xinchi Qiu, Yan Gao, Lorenzo Sani, Heng Pan, Wanru Zhao, Pedro PB Gusmao, Mina Alibeigi, Alex Iacob, and Nicholas D Lane. Fedanchor: Enhancing federated semi-supervised learning with label contrastive loss for unlabeled clients. arXiv preprint arXiv:2402.10191, 2024.

### Soundness
2

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
3

### Summary
This paper addresses the Federated Open-Set Semi-Supervised Learning (FOSSL) problem, where the server has access to a small labeled dataset of in-distribution (ID) classes, while clients hold only unlabeled, non-IID data that may include out-of-distribution (OOD) samples.This paper proposes OpenFL with three components: Round-wise EMA (R-EMA): A round-wise exponential moving average model; Pivot-Guided Open-Set Alignment: Global pivots guide client-side alignment, attracting high-confidence ID samples while mildly repelling uncertain/OOD samples; Reliability-Aware Aggregation (RAA): Client contributions are weighted based on alignment loss.

### Strengths
1. Foucs on a practical and underexplored problem.1

2. Comprehensive evaluation. The study includes relevant baselines, such as SemiFL, FedFixMatch, and federated adaptations of centralized OSSL methods. The experiments cover diverse datasets, client partitioning schemes (IID and non-IID), and multiple challenging splits, providing a broad evaluation of the proposed method.

3. Good presentation and writing.

### Weaknesses
1. Technical Novelty

The proposed method combines widely adopted techniques, including EMA, prototype-based alignment, and loss reweighting mechanisms. Each of these components is well-established in related works. For example: EMA is a standard stabilization technique in many learning systems. Pivot-based alignment is a direct extension of prototype methods used in centralized contrastive learning and semi-supervised learning. Reliability-aware aggregation using alignment loss is conceptually similar to weighting schemes, e.g., importance sampling or quality-based aggregation.

The combination of these components is incremental and does not introduce a new \textbf{insight} or novel \textbf{technique}.

2. High Sensitivity to Hyperparameters. The method relies heavily on existing loss functions and their combinations (e.g., FixMatch consistency loss, OOD detection losses from OpenMatch, SSB, etc.). It introduce multiple hyperparameters, making the method parameter-sensitive and hard to be generalized. The sensitivity analysis in the experiments demonstrates that performance can vary significantly depending on these choices.

3. Limited Applicability to Real-World Federated Settings

Server-side pivots are computed from a small labeled dataset, limiting the scalability of real-world application. If possible, please use large-scale dataset.

### Questions
Beyond the above concerns on Weaknesses, please answer:

1. Novelty Clarification

2. Dataset Scalability

3. Sensitivity Analysis of Hyperparameters Across Different Settings and Datasets

The hyperparameters in this method are highly sensitive across different datasets and settings.

### Soundness
3

### Presentation
3

### Contribution
2
