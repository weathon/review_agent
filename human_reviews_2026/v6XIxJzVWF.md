# ParaShield: Parameter-Level Directional Defense for Federated Backdoor Robustness

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Heterogeneous federated learning improves the stealthiness of backdoor attacks, presenting substantial challenges for existing defense methods to simultaneously ensure effectiveness and robustness. However, divergent optimization objectives lead to pronounced parameter-level differences between the benign heterogeneous clients and those infected with backdoor attacks. To address this issue, we introduce Parameter-level Directional Defense, termed ParaShield, which leverages Neural Influence Factors (NIF) to dynamically and rapidly capture the critical parameters. ParaShield enables the identification of parameters that are essential for maintaining model performance within the benign client updates. On this basis, we further calculate the Cosine Similarity of Critical Parameters (CPCS) and
the Sign Consistency of Critical Parameters (CPSC) to quantify directional alignment across client updates. Specifically, we initially filter out malicious model updates by analyzing the directional information of the critical parameters. Subsequently, we leverage the Mahalanobis distance in the 2D feature space formed by CPCS and CPSC to identify malicious updates deviating from the normal distribution, achieving robust aggregation. To comprehensively evaluate the robustness of ParaShield, we also construct the Projected Directional Backdoor Attack (PDBA), a stealthy backdoor attack that effectively examines defense mechanisms under realistic conditions. Extensive experiments conducted on various challenging Non-IID scenarios demonstrate the effectiveness of ParaShield.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ParaShield, a parameter-level directional defense method for heterogeneous federated learning. It identifies critical parameters through a Neural Influence Factor (NIF), filters malicious updates using cosine and sign consistency measures (CPCS/CPSC), and employs Adaptive Weighted Aggregation (AWA) with whitening and Mahalanobis distance for robust model aggregation. The authors also propose a stealthy Projected Directional Backdoor Attack (PDBA) to evaluate defense strength. Experiments on CIFAR-10/100 show ParaShield achieves the lowest ASR and highest robustness among tested defenses.

### Strengths
- **Well-motivated framework**. The paper addresses heterogeneity-induced difficulties in distinguishing benign and backdoored updates.
- **Comprehensive evaluation**. The paper includes multiple backdoor types, Non-IID settings, and ablation studies.
- **New attack**. The paper adds value for testing robustness under realistic threat models.

### Weaknesses
- No comparison with clustering- or influence-based defenses such as DeepSight [1] or FLAME [2], which are relevant for aggregation-level defenses.
- Limited methodological novelty: The main difference from AlignIns is the adaptive weighting (AWA). The large performance gain lacks clear interpretability or theoretical justification.
- Lack of interpretability: The mechanism behind the improvement from CPCS/CPSC + whitening is not analyzed or visualized.
- No runtime/memory analysis: The computational cost of each module (CPE, CPF, AWA) is not reported.
- Scalability not demonstrated: Only small CNN backbones and simple FL setups are tested. No evidence on larger models (e.g., ViT) or real-world-scale scenarios.
- Source code not released, limiting reproducibility and community validation.

[1] Rieger, Phillip, et al. "DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection."\
[2] Nguyen, Thien Duc, et al. "{FLAME}: Taming backdoors in federated learning." 31st USENIX Security Symposium (USENIX Security 22). 2022.

### Questions
- Could you provide comparisons with clustering- or influence-based defenses such as DeepSight or FLAME, to better position ParaShield among aggregation-level defense methods?
- Could you explain why the adaptive weighting (AWA) module leads to such a large performance gain over AlignIns, given the architectural similarity between the two frameworks?
- Could you provide runtime and memory analyses for each module (CPE, CPF, AWA) to better understand the computational overhead of ParaShield?
- Do you plan to release the source code, or could you share implementation details to ensure reproducibility and independent validation of your results?

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
3

### Summary
This paper proposes ParaShield, a parameter-level directional defense framework for robust federated learning against backdoor attacks. It dynamically identifies critical parameters via Neural Influence Factor (NIF), measures directional alignment among clients using Cosine Similarity (CPCS) and Sign Consistency (CPSC), and performs Adaptive Weighted Aggregation (AWA) based on Mahalanobis distance to filter and downweight malicious updates. Additionally, a stealthy Projected Directional Backdoor Attack (PDBA) is designed to evaluate defense robustness. Experiments show that ParaShield detects and neutralizes backdoors under heterogeneous settings.

### Strengths
1. Targeted method: The combination of CPCS, CPSC, and Mahalanobis distance allows dynamic and stable defense across heterogeneous settings.
2. Comprehensive evaluation: The introduction of PDBA provides a realistic and stealthy benchmark to test defense effectiveness rigorously.

### Weaknesses
1. From a parameter-wise perspective, enhancing the robustness of federated learning has already been explored — for example, by existing methods such as FDCR. Therefore, the authors need to further clarify the innovation of their approach compared to these similar methods.

2. Computational overhead: Calculating NIF, CPCS, CPSC, and Mahalanobis distance for each client increases server-side computation and communication costs. Complexity analysis is necessary.

3. Experiments on larger client scales should be included to validate the scalability.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a defense mechanism, ParaShield, against backdoor attacks in a heterogeneous federated learning setting. Due to the heterogeneity of data distribution among clients, the parameter updates from clients are different, resulting in the difficulty to identify parameter anomalies of malicious updates. ParaShield dynamically identifies critical parameters using NIF and evaluates updates based on CPCS and CPSC of these parameters. ParaShield then applies an AWA strategy that uses whitening transformation and Mahalanobis distance in a 2D feature space to detect and downweight malicious updates. To rigorously test its robustness, the authors also propose PDBA, a stealthy backdoor attack. Extensive experiments on CIFAR-10 and CIFAR-100 under Non-IID settings show that ParaShield outperforms existing SOTA defenses by effectively mitigating backdoor threats while maintaining high accuracy on benign tasks.

### Strengths
Experimentally demonstrates the critical parameters exhibit significantly different values in benign heterogeneous updates compared to backdoor updates, and demonstrate that NIF cosine similarity among malicious clients shows higher values than benign counterparts.

The work specifically focuses on federated learning under non-IID data conditions, which is a major practical challenge, and where many existing defenses fail.

Instead of treating the entire model update as a monolithic block, ParaShield innovatively operates at the parameter level.

The paper includes a clear ablation study that validates the contribution of each core module.

### Weaknesses
The paper's focus on "heterogeneous federated learning" requires clarification. The term is used interchangeably with non-IID data distributions, yet "non-IID" is the more precise and established term in the field. Adopting this specific terminology would improve conceptual clarity and better align the work with standard literature.

The paper insufficiently elaborates on essential concepts. For instance, the authors claim their PDBA attack achieves "dual stealth" (Lines 94-95) but fail to explicitly specify the two dimensions in which stealth is achieved, leaving a critical aspect of their contribution unclear.

ParaShield identifies "critical parameters" via NIF, yet the paper insufficiently argues for NIF's universality as a metric to identify critical parameters. This definition's sensitivity to model architecture, datasets, and data distribution presents a potential vulnerability: an attacker who understands and circumvents the NIF mechanism (e.g., by implanting backdoors into parameters NIF deems non-critical) could render the defense framework ineffective.

The paper's experimental validation is limited to two datasets (CIFAR-10/100) and a single model family (ResNet-9/18). Furthermore, most baseline attacks and defenses are outdated, with only two recent exceptions published during the past 3 years. Broader evaluation on larger-scale datasets (e.g., ImageNet, CelebA), diverse model architectures (e.g., VGG, ViT and other ResNet models), and sota comparative methods would substantiate the claims more convincingly.

The paper's evaluation of Non-IID scenarios may not adequately address extreme heterogeneity. In such cases, the inherent divergence among benign client updates could potentially mask the subtle directional anomalies introduced by backdoor attacks. The defense's effectiveness under these most challenging real-world conditions remains unverified.

The paper ignores the additional computational and communication costs introduced by ParaShield. For large-scale models and a large number of clients, the fine-grained, parameter-level analysis could introduce significant overhead, impacting the efficiency of federated learning, but this is not quantified or discussed in the paper.

### Questions
How is "heterogeneous setting" precisely defined in this work? Is the heterogeneity specifically in data distribution (i.e., non-IID data), data modality, or other factors? Is this terminology and its specific definition formally established in prior literature?

Given ParaShield's core reliance on identifying critical parameters, what evidence or theoretical insight demonstrates its generalizability and effectiveness across diverse model architectures, datasets, and varying data distributions?

Beyond the presented experiments, is ParaShield effective against a wider range of modern model architectures (e.g., Vision Transformers), larger-scale datasets (e.g., ImageNet), and does it outperform a broader suite of state-of-the-art attacks and defenses?

The paper tests down to a Dirichlet concentration parameter of \alpha=0.1. Would ParaShield remain effective under more extreme non-IID conditions (e.g., \alpha << 0.1), where high benign client divergence could mask backdoor signals?

What is the computational and communication overhead introduced by ParaShield's components (NIF calculation, CPF, AWA), and how does this scale with model size and the number of clients?

This work involves several key hyperparameters (e.g., CPE ratio \rho, AWA weight \beta, etc.). How sensitive is the performance to these settings across different experimental conditions? Are there guidelines for efficient tuning in practice?

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
5

### Summary
This work proposes a new server-side backdoor defense method for federated learning, called ParaShield. ParaShield first identifies critical parameters by jointly considering their absolute update magnitudes and min–max normalized scores. Based on these identified parameters, it computes two directional metrics, cosine similarity and majority sign alignment, to detect anomalous updates, which are then filtered out before aggregation. To further stabilize model aggregation under heterogeneous data, ParaShield employs an adaptive weighted aggregation that adjusts the contribution of each client update. The authors have conducted reasonable experiments to empirically evaluate the performance of ParaShield.

### Strengths
1. This work addresses the problem of backdoor defense in federated learning, which is a timely and important research topic.

2. The paper is well-written and easy to follow.

3. The proposed method, ParaShield, is technically sound, and its effectiveness is demonstrated through reasonable empirical evaluations. 

4. The proposed neural influence factor is interesting, and results (e.g., in Figure 1) do show that it successfully captures the difference between malicious updates and benign updates.

5. Compared with existing approaches, it consistently achieves higher main task accuracy and robust accuracy, while maintaining a lower attack success rate.

### Weaknesses
1. The proposed defense method is evaluated solely through empirical experiments; a theoretical analysis of ParaShield’s robustness would be valuable, if feasible.

2. The experimental evaluation is limited to a small-scale and single-modality dataset. Extending the experiments to larger-scale datasets and other modalities (e.g., text) could substantially strengthen the evaluation section.

3. The ablation study is conducted only on a single dataset and under a single attack scenario. Incorporating additional datasets and attack settings would provide a more comprehensive understanding of the contribution of each proposed component.

4. Lack of detailed defense and attack models.

### Questions
1. What is the specified value of the important parameter $\tau$? It is not clearly stated in the experimental settings section. An ablation study on the choice of $\tau$ is also missing.

2. The paper shows that the critical parameter extraction step can successfully identify malicious parameters, which consistently exhibit smaller values. In that case, why not adopt a pruning-based approach to directly remove these malicious parameters?

3. What is the time complexity of the proposed method compared to state-of-the-art approaches?

4. Is the critical parameter–based filtering process entirely borrowed from AlignIns? To the best of my knowledge, it appears quite similar to the approach used in AlignIns. If so, a clear acknowledgment and citation should be provided, for example, near line 183.

5. Since the attack model is missed, I suppose the potential attacker can poison an arbitrary number of clients and turn them into malicious clients. What is the performance of  ParaShield under the cases where more than 50% of clients are poisoned?

### Soundness
3

### Presentation
3

### Contribution
3
