# Fed-ARPL: Adaptive and Reciprocal Prototype Learning for Semi-supervised Federated Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Federated Semi-supervised Learning (FSSL) enables collaborative training by leveraging a small labeled dataset on a central server and vast unlabeled data across clients. However, existing frameworks are hampered by two challenges: an initial Cold-start phase, where strict pseudo-label filtering criteria impede the use of unlabeled data, and a subsequent Knowledge Bottleneck, where the model's performance is capped by the server's limited and potentially biased labeled data. To address these challenges, we propose Fed-ARPL, a novel Adaptive and Reciprocal Prototype Learning framework that implements a meticulously designed three-phase learning strategy.First, a Warm-up Phase employs an adaptive thresholding mechanism to resolve the Cold-start dilemma, dynamically adjusting the pseudo-label confidence to accelerate initial convergence and establish a stable feature space. Next, a Teacher-Guided Phase leverages the server's reliable prototypes to provide unified, one-way guidance, steering all clients toward a consistent and well-structured representation. Finally, to break the Knowledge Bottleneck, the framework culminates in a Student-Feedback Phase, establishing a reciprocal paradigm where high-performing clients contribute their refined local prototypes to enrich the global consensus.Comprehensive experiments validate the effectiveness of our Fed-ARPL framework, showcasing its state-of-the-art (SOTA) performance on several widely-recognized benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Fed-ARPL, a comprehensive framework for Federated Semi-Supervised Learning (FSSL) that aims to overcome two major challenges — the Cold-start problem and the Knowledge Bottleneck.
Fed-ARPL adopts a three-phase adaptive learning strategy: an adaptive warm-up phase that dynamically selects pseudo-labeled data, a teacher-guided phase where the server provides global prototype supervision, and a student-feedback phase where clients contribute local prototypes back to the server.
Extensive experiments on CIFAR-10, CIFAR-100, and SVHN under various non-IID and label-scarce conditions show that Fed-ARPL achieves substantial gains over existing baselines, validating the effectiveness of its design. Ablation studies further validate the contribution of each proposed component.

### Strengths
1. The paper introduces a complete and coherent framework, Fed-ARPL, that systematically tackles both the Cold-start and Knowledge Bottleneck challenges in FSSL.

2. Experimental results across multiple datasets and varying heterogeneity levels clearly demonstrate that Fed-ARPL significantly outperforms existing baselines. The improvements are especially notable under severe label scarcity and strong non-IID conditions, underscoring the method’s robustness and generalizability.

3. The paper is clearly written, well-structured, and easy to follow.

### Weaknesses
1. While the integration of adaptive thresholding and prototype feedback into a three-phase pipeline is well-engineered, each individual component (e.g., prototypical aggregation) builds upon existing concepts. The novelty lies more in the combination and systematization than in a fundamentally new learning principle.

2.  The framework introduces many additional hyperparameters (e.g., τ_fss, σ_fss, λ_plc, W_gcplca, τ′, α_t, β) that must be manually tuned. This reduces practicality, as many of these thresholds require non-trivial calibration and could vary widely across datasets.

3. The baselines are relatively dated; the newest comparison (FL2, 2024) is over a year old. The paper would be more convincing if compared with very recent FSSL frameworks such as FedLGMatch or other adaptive pseudo-labeling methods from 2025.

### Questions
1. Could the authors explore adaptive or self-tuning mechanisms for the many hyperparameters, especially τ_fss, σ_fss, and W_gcplca? These parameters could potentially be inferred based on model confidence dynamics rather than fixed thresholds.

2.  What would happen if the server-side labeled data were even more limited—say, fewer than 10 samples per class? Would the adaptive thresholding still produce stable pseudo-labels, or would the teacher-guided phase fail to provide useful supervision?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the "labels-at-server" Federated Semi-Supervised Learning (FSSL) setting, where a central server holds limited labeled data and clients have only unlabeled samples. It identifies two key challenges: the cold-start problem caused by fixed pseudo-label thresholds, and a knowledge bottleneck from limited server supervision. The proposed Fed-ARPL framework with phase transitions governed in a formulated manner.

### Strengths
The paper provides a clear decomposition of the training dynamics. Its three-phase design distinctly separates early data utilization, representation manifold shaping, and capacity expansion. The phase-transition criteria are formulated, making the pipeline more principled than heuristic. Experimental results are competitive with the evolution of FSS/GCPLCA metrics and threshold acceptance rates that enhances interpretability.

### Weaknesses
While the paper is well structured, its novelty over prior FSSL and FL-prototype lines is incremental. Adaptive or confidence-based thresholds and prototype sharing are established ideas. The main contribution appears to lie in their orchestration via the proposed gating metrics. The evaluation scope is limited to small-scale, low-resolution datasets, and lacks robustness tests under server label noise, client drift, or open-set unlabeled pools. Privacy and communication trade-offs are underexplored (prototype uploads may leak distributional information, and their cost relative to model-only is not quantified). The phase criteria (FSS, GCPLCA) are empirically tuned without theoretical grounding while sensitivity analyses are dataset-specific. Finally, several presentation issues remain (e.g., typos (e.g., “Fed-APRL”), unresolved equation references (e.g., "??"), inconsistent notation).

### Questions
N/A

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
3

### Summary
This paper proposes Fed-ARPL, an Adaptive and Reciprocal Prototype Learning framework for federated semi-supervised learning. It focuses on two key and relatively understudied challenges: the cold-start problem and the knowledge bottleneck. To mitigate these problems, a three-phase training strategy is proposed: an adaptive thresholding mechanism to enhance early pseudo-label utilization, a teacher-guided prototypical learning phase to ensure global feature consistency, and a student-feedback phase enabling bidirectional knowledge exchange between clients and server. Experiments on CIFAR-10, CIFAR-100, and SVHN demonstrate that Fed-ARPL achieves state-of-the-art performance and robust convergence under both IID and Non-IID data distributions.

### Strengths
1.	The motivation of this work is convincing. It targets two fundamental and practical challenges in FSSL—the cold-start and knowledge bottleneck problems.
2.	Experimental results on CIFAR-10, CIFAR-100, and SVHN under multiple Dirichlet skews and IID settings show consistent and significant performance gains.

### Weaknesses
1.	Fed-ARPL requires each client to upload both model weights and class prototypes to the server, whereas many related works transmit only one of these components. This design choice may raise two questions: (a) Communication and computation: Does the inclusion of both significantly increase communication or computational cost compared to existing methods? A brief quantitative analysis would help clarify this. (b) Data privacy: When the server has access to both local model parameters and class prototypes, could this combination potentially increase the risk of information leakage or client data inference? It would be valuable if the authors could provide a deeper discussion or empirical assessment regarding privacy implications in this context.
2.	This work evaluates the proposed method on three widely used but relatively small image datasets (CIFAR-10, CIFAR-100, and SVHN), all with 32×32 image resolution. Would the proposed framework maintain its advantages on larger and more complex datasets (e.g., ImageNet)? Including at least one large-scale experiment could strengthen the generalizability of the proposed method. In addition, several related methods, like FedProto, are discussed in the related work section but not included in the experimental comparison. Could the authors provide some justification for the chosen baselines?
3.	Lack of theoretical justification for the new metrics (FSS & GCPLCA). The paper provides useful empirical and sensitivity analyses, but the motivation for selecting these specific metrics over standard alternatives (e.g., Silhouette or Davies-Bouldin for feature separability) could be discussed in more detail to improve clarity.
4.	There are a few missing references, like Eq.equation??.

### Questions
The questions have been stated in the weakness section. I would raise my score if these questions could be properly addressed in the rebuttal.

### Soundness
2

### Presentation
3

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
Targets FSSL’s Cold-start (too-strict early thresholds block unlabeled use) and Knowledge Bottleneck (server’s small labeled set caps model). Proposes Fed-ARPL, a three-phase scheme: Warm-up with adaptive thresholding; Teacher-Guided with server prototypes; Student-Feedback where high-performing clients contribute refined prototypes back to the server. Shows SOTA on several benchmarks with ablations.

### Strengths
- Three-phase adaptive + prototypical strategy with reciprocal feedback.
- Addresses persistent pains (cold-start, bottleneck).
- Benchmarks and ablations are indicated.

### Weaknesses
- Missing specifics on when/how to switch phases; risk of premature or delayed transitions.
- Prototype quality under client drift/non-IID not rigorously assessed; could amplify bias.
- Absent systems accounting (communication, latency) for extra prototype exchanges.

### Questions
- What automated metrics trigger phase transitions and how robust are they?
- How do you filter noisy client prototypes to prevent degrading the global model?
- Please include compute/communication overhead vs. SemiFL/FL2 with identical hardware and rounds.

### Soundness
2

### Presentation
2

### Contribution
3
