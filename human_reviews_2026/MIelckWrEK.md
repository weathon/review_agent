# A Unified Federated Framework for Trajectory Data Preparation via LLMs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Trajectory data records the spatio-temporal movements of people and vehicles. However, raw trajectories are often noisy, incomplete, or inconsistent due to sensor errors and transmission failures. To ensure reliable downstream analytics, Trajectory Data Preparation (TDP) has emerged as a critical preprocessing stage, encompassing various tasks such as imputation, map matching, anomaly detection, trajectory recovery, compression, etc. However, existing TDP methods face two major limitations: (i) they assume centralized access to data, which is unrealistic under strict privacy regulations and data silo situations, and (ii) they train task-specific models that lack generalization across diverse or unseen TDP tasks. To this end, we propose FedTDP for Federated Trajectory Data Preparation (F-TDP), where trajectories are vertically partitioned across regions and cannot be directly shared. FedTDP introduces three innovations: (i) lightweight Trajectory Privacy AutoEncoder (TPA) with secret-sharing aggregation, providing formal privacy guarantees; (ii) Trajectory Knowledge Enhancer (TKE) that adapts LLMs to spatio-temporal patterns via trajectory-aware prompts, offsite-tuning, sparse-tuning, and bidirectional knowledge distillation; and (iii) Federated Parallel Optimization (FPO), which reduces communication overhead and accelerates federated training. We conduct experiments on 6 real-world datasets and 10 representative TDP tasks, showing that FedTDP surpasses 13 state-of-the-art baselines in accuracy, efficiency, and scalability, while also generalizing effectively across diverse TDP tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes FedTDP, a unified federated framework for Trajectory Data Preparation (TDP) leveraging Large Language Models (LLMs) to address limitations of existing TDP methods. Raw trajectory data often suffers from noise, incompleteness, and inconsistency, while traditional TDP approaches rely on centralized data access (conflicting with privacy regulations like PIPL) and task-specific models (lacking generalization). FedTDP, designed for vertically partitioned trajectory data across regional silos, integrates three core modules, i.e., TPA, TKE, and FPO. Experiments on 6 real-world datasets (e.g., GeoLife, Porto) and 10 TDP tasks (e.g., anomaly detection, map matching) show FedTDP outperforms 13 state-of-the-art baselines in accuracy, efficiency, and scalability, with strong cross-task generalization.

### Strengths
- New Problem Formulation: First to formally define Federated Trajectory Data Preparation (F-TDP) for vertically partitioned trajectory data, filling the gap of federated learning research focused on horizontal partitioning.

- Privacy-Preserving Design: TPA with secret-sharing aggregation provides formal privacy guarantees without sacrificing data utility (unlike differential privacy, which distorts spatio-temporal correlations via noise injection).

- Strong Generalization & Efficiency: TKE enables LLMs/SLMs to generalize across 10 diverse TDP tasks, while FPO reduces communication overhead and training time, addressing the inefficiency of LLM deployment in federated settings.

### Weaknesses
- **The conclusion section of this paper is missing from the main text:** ICLR 2026 submission policy requires authors to organize the main text reasonably, which means that reviewers are not obliged to read the appendix provided by the authors.

- Limited Theoretical Grounding of Privacy Guarantees: While the paper claims formal privacy protection through secret sharing, the analysis lacks rigorous proof of robustness against advanced gradient or embedding inversion attacks.

- Overstated Generalization Claims: The generalization across unseen TDP tasks is not convincingly demonstrated, as performance differences might stem from dataset similarity rather than true cross-task transfer.

- High System Complexity: The integration of TPA, TKE, and FPO introduces substantial architectural and computational complexity, making real-world deployment questionable.

### Questions
Comments:

1. Inadequate Formal Privacy Analysis: The claimed privacy guarantee of TPA relies on secret-sharing aggregation but lacks quantitative leakage bounds or empirical resistance tests against reconstruction attacks.

2. Ambiguous Vertical Partition Setting: The paper assumes vertical partitioning across regions but does not clarify how overlapping or boundary trajectories are handled during training and aggregation.

3. Unclear Contribution of Submodules: Although ablation studies exist, they fail to isolate interactions between TKE and FPO; improvements may overlap rather than reflect independent effects.

4. Questionable Scalability: The model introduces heavy communication (≈5 GB per 10 rounds), which may be infeasible in real federated environments with limited bandwidth.

5. Limited Explainability of LLM Adaptation: The proposed trajectory-aware prompt design and knowledge distillation process are heuristic and lack ablation or visualization of learned representations.

6. Inconsistent Evaluation Metrics: The results mix F1, accuracy, and distance-based metrics without discussing normalization or the statistical significance of performance gains.

7. Underexplored Generalization Factors: The zero-shot and few-shot evaluations lack control experiments on domain shifts, leaving it unclear whether improvements arise from pre-training, task similarity, or federated effects.

8. The recommended LoRA layer ratio (m ≤ 25%) is arbitrary; no systematic method is proposed to balance accuracy and efficiency for different TDP tasks.

9. FedTDP’s performance on extremely sparse trajectory data (e.g., low-sampling-rate GPS) is unevaluated, limiting its utility for real-world scenarios with poor sensor quality.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes FedTDP, a unified federated framework for trajectory data preparation (TDP) that addresses two key limitations of existing methods: the assumption of centralized data access and the lack of generalization across diverse TDP tasks. FedTDP enables privacy-preserving and multi-task TDP without raw data sharing with its core innovations including a lightweight Trajectory Privacy AutoEncoder (TPA) with secret-sharing aggregation, a Trajectory Knowledge Enhancer (TKE) that adapts LLMs to TDP tasks, and a Federated Parallel Optimization (FPO) strategy to reduce communication overhead and accelerate training. Evaluated on six real-world datasets across ten representative TDP tasks, FedTDP consistently outperforms 13 state-of-the-art baselines in accuracy, efficiency, and generalization.

### Strengths
S1. The paper introduces a novel and well-motivated problem formulation: Federated Trajectory Data Preparation (F-TDP), a new and practical setting that addresses the realistic challenges of vertically partitioned trajectory data under strict privacy regulations.

S2. FedTDP is the first LLM-based framework capable of handling 10 diverse TDP tasks within a single architecture, demonstrating impressive generalization across seen and unseen tasks and datasets.

S3. FedTDP introduces novel technical contributions with three purpose-built modules: (i) TPA that enable secure aggregation while preserving spatio-temporal structure, (ii) TKE that adapts generic LLMs to trajectory semantics and TDP tasks, and (iii) FPO that improves training efficiency.

S4. The paper provides rigorous theoretical privacy analysis to quantify the upper bound of information leakage, showing that the probability of successful trajectory reconstruction is low.

### Weaknesses
W1. It would be beneficial to include key experiments such as model efficiency and generalization study in the main paper rather than appendix.

W2. For clarity, the caption of Figure 10 in Appendix D.4 should be revised to “Generalization Study.”

W3. While Appendix E outlines the limitations of FedTDP, a more in-depth discussion of each limitation would strengthen the paper.

### Questions
See Weakness.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes "FedTDP," a unified federated learning framework for Trajectory Data Preparation (TDP). It incorporates three key components: a Trajectory Privacy AutoEncoder (TPA) with secret-sharing for privacy, a Trajectory Knowledge Enhancer (TKE) to adapt LLMs to spatio-temporal data, and a Federated Parallel Optimization (FPO) method for efficiency. The authors claim superior performance over 13 baselines across 10 TDP tasks on 6 real-world datasets. Overall, the paper is well-written.

### Strengths
1.	The paper does a great job of identifying and clearly describing the challenges of vertical partitioning and multi-task generalization in trajectory data preparation.

2.	This paper conducts comprehensive experiments with enough baseline models and datasets, covering both seen and unseen tasks.

3.	The overall quality of this paper is good, with a clear structure and well-defined modules.

### Weaknesses
1.	The authors advocate for three key challenges, but the solutions feel like a patchwork of existing LLM-related techniques (secret-sharing, LoRA, prompting) rather than a fundamental innovation. The core methodology remains a conventional federated learning setup augmented with off-the-shelf LLM adaptation methods, although it may be suitable for ICLR.

2.	The motivation for the strong privacy threat model is kind of weak. The paper lacks a compelling reason for why trajectory data preparation requires such stringent protection against a semi-honest server. It should be more illustrated.

3.	The claimed "unified" and "generalized" framework relies on manually configured, task-specific prompts and external information. This suggests the generalization is not autonomous but is significantly enabled by human-in-the-loop engineering for each task, limiting the claimed novelty.

### Questions
1.	The TPA structure is a lightweight MLP. How does the reconstruction error of the autoencoder impact the accuracy of downstream tasks that require high spatial precision, such as Map Matching? 

2.	The privacy guarantee assumes the TPA model is private. In the federated setting, how is the secrecy of the aggregated TPA model's architecture and parameters maintained against the semi-honest server?

3.	How much of the model's generalization to unseen tasks is due to the inherent few-shot ability of the base LLM versus the specific design of the TKE? Could a centrally fine-tuned LLM with good prompts achieve similar results without the federated complexity? These questions could be more emphasized.

### Soundness
3

### Presentation
3

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
This paper proposes FedTDP, a unified federated framework to address Trajectory Data Preparation (TDP). The authors identify two major limitations in existing methods: (L1) they require centralized data, which is unrealistic due to privacy laws, leading to "vertically partitioned" data silos (e.g., by region); and (L2) they are task-specific (e.g., one model for imputation, another for map matching) and lack generalization. To solve this, FedTDP proposes a complex server-client architecture with a central LLM and on-device SLMs. Its core innovations are three-fold, designed to solve three new challenges: The framework is evaluated on 10 TDP tasks and 6 real-world datasets, where it reportedly outperforms 13 state-of-the-art baselines.

### Strengths
1.	Novel Problem Formulation: The paper clearly defines F-TDP (Federated Trajectory Data Preparation) as a novel problem. It accurately targets two real-world challenges: (L1) vertical data partitioning due to privacy silos and (L2) the need for a general-purpose framework to replace 10+ single-task models.
2.	Strong Experimental Results: This is the paper's greatest strength. As shown in Table 2, FedTDP consistently and significantly outperforms 13 baselines across 10 TDP tasks and 6 datasets.
3.	Excellent System Design and Ablation: The ablation study in Figure 5 is highly effective. It clearly demonstrates the performance.

### Weaknesses
1.	Misleading Privacy Guarantees: The paper's central privacy claim is unsound. It rejects Differential Privacy (DP) for degrading utility and claims its TPA + Secret Sharing mechanism provides "formal privacy guarantees". The paper's entire privacy for data-in-transit relies on "computational infeasibility", which is a (weaker) "encryption-like" guarantee, not the "formal" (mathematical, statistical) guarantee that DP provides. This core contradiction undermines the C1 contribution.   
2.	Overly Complex System: The FedTDP framework is a massive, complex system integrating numerous distinct technologies (TPA, Secret Sharing, SLMs, LLMs, Offsite-tuning, Sparse-tuning, KD, Split Learning, etc.). This "kitchen sink" approach makes the system extremely difficult to reproduce and analyze.
3.	Questionable Theoretical Analysis: The mathematical derivation for Theorem 2 (LoRA sparse-tuning probability) in Appendix C.4 is overly complex and its correctness is not immediately obvious.

### Questions
Q1: Critical Privacy Question: Your privacy analysis relies on the server not being able to reverse the embedding because the encoder is private. The paper's threat model assumes that the server is “semi-honest”, but that the server receives the client's data embedding when performing cross-client tasks. Why is this approach (sending data embeddings) inherently more privacy-secure than “sending gradients” (vulnerable to gradient reversal attacks), which the paper criticizes?
Q2: TKE Complexity: The TKE module is key to the model's high performance but is extremely complex, with four sub-components (prompts, offsite-tuning, sparse-tuning, KD) . Have you conducted an ablation within TKE to identify which of these components are essential? Is it possible the performance gain comes from just one or two (e.g., only offsite-tuning and prompting)?

### Soundness
3

### Presentation
3

### Contribution
3
