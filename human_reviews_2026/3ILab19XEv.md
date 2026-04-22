# Cross-Network Structure Enhancement via Adaptive Coupling

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Network structural enhancement seeks to improve the accuracy and reliability of real-world network representations by systematically detecting and inferring missing or potential links.Existing research primarily focuses on single networks, overlooking the interdependence of real-world systems. In practice, entities often span multiple networks—for example, users migrate and interact across social platforms, forming multiplex networks. Approaches considering multiplex networks typically use static weights or simple aggregation, failing to adaptively control the influence of each network at the sample level. This can introduce irrelevant information and cause negative transfer.
To address this, we introduce Adaptive Coupling for cross-Network structure Enhancement (ACNE), the first framework that leverages adaptive, sample-wise cross-network coupling for structure enhancement in multiplex networks. We first employ GNNs to obtain network-specific representations.   Building upon this foundation, we introduce a generative–discriminative adversarial learning framework, and impose an adversarial weight perturbation in parameter space to approximate worst-case noise and stabilize the learned cross-network embeddings.  To adaptively balance the contributions between target-specific and cross-network embeddings, we design a low-rank bilinear gated fusion module.   In addition, a decorrelation regularizer is incorporated to minimize redundancy arising from overlapping communities. Extensive experiments on real-world multiplex networks show that our approach consistently surpasses existing baselines in link prediction, highlighting the effectiveness and practical value of adaptive cross-network structure enhancement.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper targets link prediction on a target layer within multiplex networks and proposes Adaptive Coupling for
cross-Network structure Enhancement (ACNE), which is trained end-to-end using a linear combination of the prediction loss, discriminator loss, and a decorrelation term. Theoretically, the authors show that gated fusion under a decorrelation constraint can tighten a generalization bound that uses the logit variance as a proxy; empirically, across five real-world multiplex graphs, ACNE outperforms strong baselines on multiple tasks (with higher AUC/ACC on London and Enron) and includes ablation and hyperparameter sensitivity analyses to support the design choices.

### Strengths
The motivation is clear and the design is modular: from within-layer encoding and cross-network adversarial coupling to sample-wise low-rank gated fusion with a decorrelation regularizer. The objectives and training procedure are precisely specified. 

The gating and decorrelation components are theoretically motivated: by minimizing the covariance spectral radius, the method reduces the variance of the fused logit, thereby ensuring that, under complementary views, the fusion risk is no worse—and potentially better—than the best single-view predictor.

### Weaknesses
The manuscript has several limitations closely tied to the following issues: 

1) Scalability and complexity are not quantified—the paper does not provide time/memory costs or upper bounds as functions of the number of layers (L), the number of candidates (B), and (|V|);

2) It did not present a trade-off analysis for the additional overhead introduced by AWP and the discriminator. 

3) The theoretical conditions and failure modes are discussed at a relatively high level. 

4) The key assumptions under which gating + decorrelation improve the generalization bound (e.g., covariance structure, rank and temperature requirements) are not empirically testable in their current form.

5) A comprehensive related work investigation should be conducted, as several important and recent works are missing, e.g., GNN-based Methods only introduced the studies proposed before 2018. The authors need to compare the features of each work in the literature with the features and main contributions of their work to make their contributions clearer. 

6)  This paper adopted eleven baseline methods to assess the effectiveness of ACNE. However, it has not been compared it with the works proposed in recent years. There is a need to include more recent state-of-the-art approaches for comparison. 

7) No open-source implementation: Sharing the code would improve reproducibility and allow for further research. 

8) Further analysis of the model's applicability to different tasks or datasets would strengthen the claims.

### Questions
1. Please provide the time and memory complexity for training and inference, explicitly detailing the dependence on (L), (|V|), and the number of candidate pairs (B). Also, quantify the additional overhead introduced by AWP and the discriminator, and discuss the trade-offs with performance. Could you add profiling on larger graphs or give formal upper bounds?

2. What empirically verifiable conditions are required for the risk improvement due to gating + decorrelation (e.g., assumptions on covariance structure, the range of rank (r) and temperature (\tau), and assumptions on the logistic loss)? Can you offer a more precise or data-dependent statement, and clarify under what circumstances the guarantees may fail?

3. Could you include an ablation study without AWP (w/o AWP) to quantify the standalone benefit of parameter-space perturbations relative to adversarial alignment alone?

4. Clarify the training/inference workflow: please provide a clear description of a single training iteration in order (and specify whether the discriminator and AWP are skipped at inference).

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
This paper proposes ACNE, a novel framework for cross-network structure enhancement that leverages adaptive, sample-wise coupling to improve link prediction in multiplex networks. The method integrates GNN-based network-specific encoders, adversarial cross-network representation learning with weight perturbation, and a low-rank bilinear gated fusion mechanism with decorrelation regularization. Extensive experiments on real-world multiplex networks demonstrate that ACNE consistently outperforms existing baselines in link prediction tasks.

### Strengths
- Innovative fusion strategy that dynamically balances target and source contributions at the sample level.

- Theoretical grounding that links adversarial training and decorrelation to generalization performance.

### Weaknesses
- The experiments are conducted on relatively small networks (e.g., Aarhus with 61 nodes, Kapferer with 39 nodes), raising concerns about scalability and real-world applicability.

- While the technical composition is solid, the core idea of cross-network modeling has been explored in related areas such as cross-domain recommendation, which may limit the perceived novelty.

- All experiments are on academic benchmarks; no industrial-scale or dynamic network scenarios are tested.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces ACNE, a framework for cross-network link prediction in multiplex networks. It uses GNNs to extract network-specific embeddings and then aligns them via an adversarial generator–discriminator game augmented by adversarial weight perturbation for robustness. A low-rank bilinear gating mechanis fuses embeddings for every node pair, while a decorrelation regularizer suppresses redundancy. Extensive experiments on five real-world datasets show that ACNE consistently outperforms eleven baselines in accuracy and AUC.

### Strengths
1. beyond static aggregation, Adaptive and sample-wise cross-network coupling with AWP and low-rank gated fusion is a well-motivated and non-trivial combination.

2. the author provides clear model design and theory analysis.

3. The proposed method achieved performance gain on 5 benchmark datasets across different settings.

### Weaknesses
In general, I am satisfied with the paper, though i'm not the expert in this domain. However, I have small concerns about the paper:

1. The individual modules are already presented in previous papers. like the low-rank bilinear gate is somehow like bilinear layers in earlier CNN networks. The adversarial discriminator follows the same spirit as DANN/CDAN. Thus the novelty is somehow limited. 

2. What's the increased complexity of the proposed methods? I was wondering whether the proposed method could be scaled to large-scale datasets.

### Questions
Please mainly see the weaknesses section for my questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents ACNE (Adaptive Coupling for cross-Network structure Enhancement), a framework for adaptive, sample-wise structural enhancement in multiplex networks for link prediction. ACNE combines GNN-based encoders for each network with adversarial coupling (via parameter-space perturbations for robustness) and a low-rank bilinear gated fusion mechanism to integrate target- and cross-network embeddings. A decorrelation regularizer reduces redundancy, and the framework is evaluated on five real-world multiplex datasets against both single- and cross-network baselines.

### Strengths
S1. The framework introduces an elegant adversarial–adaptive coupling mechanism for cross-network structural transfer, with per-sample gating and robustness regularization.

S2. Theoretical analysis (Proposition 1, Theorem 1) connects adversarial alignment and decorrelation regularization to generalization risk, providing mathematical clarity beyond empirical justification.

S3. Experiments are broad and systematic, with thorough ablations—especially on the Kapferer dataset—showing consistent gains over strong baselines.

### Weaknesses
W1. The method assumes comparable or complete node features across networks; heterogeneity and missing-feature robustness are not explored, limiting applicability in real multiplex scenarios.

W2. Key training details (negative sampling, batch composition, and cross-network pooling) are under-specified, reducing reproducibility.

W3. Theoretical guarantees hinge on assumptions (affine prediction heads, convex losses) that may not hold under practical multi-head GAT and adversarial training; the gap between theory and implementation should be better contextualized.

W4. The “first-of-its-kind” claim overstates novelty, as prior works have explored adaptive or sample-level weighting in multiplex networks under different formulations.

### Questions
Q1. How does ACNE handle irrelevant or noisy source layers? Can the gating mechanism downweight misleading information in adversarial settings?

Q2. How does the method perform under missing or highly heterogeneous features across networks? Does performance degrade gracefully?

Q3. Could the authors clarify details of negative sampling, batch construction, and aggregation in Algorithm 1 to enhance reproducibility?

### Soundness
2

### Presentation
2

### Contribution
2
