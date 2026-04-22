# FU-DWS: Effective Federated Domain Unlearning via Domain-aware Weight Surgery

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Federated Learning (FL) enables distributed clients to collaboratively train machine learning models without sharing raw data, enhancing user privacy. However, stringent data protection regulations, such as the GDPR, mandate the erasure of certain domain-specific knowledge from trained models, raising the critical challenge of federated domain unlearning. Unlike traditional federated unlearning approaches that focus on removing data at the client, class, or sample level within homogeneous domains, federated domain unlearning aims to selectively remove learned knowledge associated with entire data domains, which frequently differ across clients in real-world settings. To address this challenge, we \underline{F}ederated Domain \underline{U}nlearning via \underline{D}omain-aware \underline{W}eight \underline{S}urgery (\texttt{FU-DWS}), a novel framework that leverages channel activation patterns to identify domain-specific weights and applies differential update strategies based on their importance. \texttt{FU-DWS} performs ``surgical'' weight modifications by precisely measuring channel-level domain sensitivity, then selectively pruning and fine-tuning only the components strongly associated with the forgetting domain while preserving knowledge critical to retained domains. Comprehensive evaluation against six baselines across three domain-heterogeneous datasets demonstrates that \texttt{FU-DWS} significantly outperforms existing methods in both unlearning effectiveness and computational efficiency, while maintaining stronger performance on retained domains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper systematically addresses the novel and critical challenge of Federated Domain Unlearning, which involves removing the influence of an entire data domain from a well-generalized federated learning model. The authors propose FU-DWS, a novel method grounded in the key observation that channel-level activations in a neural network are highly correlated with domain-specific knowledge. FU-DWS first identifies and prunes channels with high sensitivity to the domain targeted for unlearning, and then selectively freezes stable channels crucial for the retained domains during local recovery. Extensive experiments on multiple heterogeneous datasets demonstrate that FU-DWS significantly outperforms six existing baselines by achieving a superior balance among recovery performance on retained domains, effective forgetting of the target domain, and generalization capability on unseen data.

### Strengths
1. This work is the first to systematically define and tackle the problem of federated domain unlearning.

2. The proposed method, FU-DWS, enables precise unlearning by accurately identifying and pruning domain-salient weights. Its dual mechanism of pruning and selective freezing effectively removes the target domain's knowledge while ensuring rapid and stable recovery of performance on retained domains.

3. The method demonstrates state-of-the-art effectiveness across multiple heterogeneous datasets and model architectures. It excels in achieving an optimal trade-off between recovery, forgetting, and generalization.

### Weaknesses
1. The method's performance depends on the selection of pruning and freezing thresholds. Although the approach shows some robustness, identifying the optimal values requires tuning. The paper's mention of a "huge search space" suggests that this could impose a non-trivial tuning burden in practical deployments.

2. The evaluation of unlearning effectiveness relies on indirect metrics like accuracy on the forgotten domain and backdoor attack success rate. These methods lack a direct, statistical guarantee that the domain's information has been completely and verifiably erased from the model's parameters.

3. The paper does not sufficiently analyze scenarios where multiple domains have significant feature space overlap. In such cases, the proposed pruning strategy might lead to substantial collateral damage and performance loss on retained domains.

4. The method assumes that all clients within a single domain share an identical underlying data distribution. This simplification may overestimate the method's efficacy in real-world cross-silo settings where data distribution can vary even among clients from the same organization.

5. The requirement for clients to compute and compare activation statistics, while not involving raw data, could potentially leak information about domain characteristics. Furthermore, the reliability of these activation-based metrics is questionable for domains with very few local samples.

### Questions
1. How can the pruning and freezing thresholds be adaptively and automatically determined based on model architecture and data characteristics to enhance practicality?

2. What statistical tests or verification frameworks can be developed to provide provable guarantees of complete domain-level forgetting?

3. What is the true computational and memory overhead of calculating activation statistics on resource-constrained edge devices, and how can this process be optimized?

4. Eq.3 introduces the pruning score without further explanation or motivation. This formula is critical for understanding the pruning step, could you clarify how it is derived and why it’s defined that way?

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
5

### Summary
This paper introduces FU-DWS, an algorithm for federated domain unlearning that targets domain-level knowledge removal in cross-silo federated learning. Through a two-stage "weight surgery" (pruning and selective freezing of activation-sensitive channels), the method aims to efficiently remove knowledge tied to one domain while preserving generalization and utility on the remaining domains. Extensive experiments on three multi-domain datasets and six baselines demonstrate that FU-DWS achieves competitive forgetting, strong recovery on retained domains.

### Strengths
1. The paper is well organized and clearly written overall, with quantitative analyses in figures and tables that effectively support the main arguments.
2. The motivation targets a realistic and practically significant problem in federated learning, particularly the challenge of domain-level unlearning under privacy and regulatory constraints.    
3. The proposed approach is conceptually straightforward and practically feasible, making it easy to implement and integrate into existing federated learning systems.

### Weaknesses
1.  The approach heavily relies on activation magnitudes to estimate domain sensitivity and uses it as the sole signal. This reliance may be unstable under different network architectures, normalization layers, or training dynamics. Furthermore, if activation differences alone are insufficient to fully characterize this distinction (e.g., some domain-specific knowledge may not manifest as significant activation differences, or some shared knowledge might occasionally show large activation differences), the effectiveness of this method would be significantly compromised.
2.  In scenarios where domain knowledge overlaps, these domains may not be distinguishable solely by the activation magnitudes of different channels. The authors need to further discuss whether the channel pruning strategy might inadvertently damage knowledge belonging to retained domains when the forget domain and retained domains have similar knowledge.
3.  Some hyperparameters in this paper (e.g., $\tau_p$, $\tau_f$) rely on manual grid search, which limits their applicability in highly complex scenarios. Additionally, the related sensitivity experiments were only validated on a small scale using the Webcam dataset (Figure 8, Appendix A5).
4.  Most of the methods compared in this paper are not specifically focused on domain unlearning and were primarily proposed before 2023. Thus, they may not represent the state-of-the-art level, making the comparison potentially unfair.
5.  The experiments are primarily conducted on relatively small and image-based datasets (Domain-Digits, Office-Caltech-10, PACS). It remains unclear whether the method scales to large-scale, high-dimensional, or non-vision tasks, such as text or tabular federated learning scenarios.

### Questions
1. Can the authors clarify the mathematical justification and potential pitfalls of using mean-absolute activations ($S_{l,c}$) as a proxy for domain saliency? Given the observed overlap or instability of activation distribution across domains, how robust is this metric to adversarial domain mixing/noise?
2. Have you tested FU-DWS on settings with partial domain overlaps, or where the forget domain is semantically strongly related to the remaining ones? If not, can you comment on expected performance?

### Soundness
2

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
3

### Summary
This paper proposes a new method for **domain unlearning** in a federated learning setting. The main idea behind this problem is that, upon a request from clients holding data from a given domain, the federated model being learned must behave as it had never seen data from that domain. The main motivation for authors method is detecting parameters that are related to the specific domain. These parameters are detected by comparing the weights of the global model, and local model, which seems reasonable.

### Strengths
Overall the method is well motivated, simple and reasonable. I think this is an interesting direction for domain unlearning. The federated setting is also quite appealing and practical.

### Weaknesses
While I myself could not find any technical problems with this paper, I think it has many writing problems.

__Problem 1.__ Overall I think Figure 2 is quite difficult to interpret. For instance, Fig 2. (a) shows a heatmap of model activations, but the actual difference and the mechanism for the selection of parameters to be pruned could me made more explicit. The same holds for Fig 2 (b). Furthermore, there is no colorbar indicating what the colors in the heatmap mean.

__Problem 2.__ The function $S\_{l,c}(\theta^{k}, \mathcal{D}\_{k})$ is never explicitly defined in mathematical terms.

__Problem 3.__ Table 1 in the experiments is never mentioned in the text. I suspect the authors made a typo in the last paragraph of Page 7, where they mention Table 3. I think it should be Table 1.

### Questions
N/A

### Soundness
3

### Presentation
2

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
This submission proposes FU-DWS (Federated Domain Unlearning via Domain-aware Weight Surgery), a method designed to selectively remove domain-specific knowledge from federated models while preserving performance on retained domains. The problem addressed is federated domain unlearning, where an entire domain (e.g., hospital or dataset) must be forgotten due to legal or ethical requirements under heterogeneous, cross-silo FL settings. FU-DWS identifies activation-aware domain-salient weights that are highly sensitive to the target domain and performs a two-stage "surgical" operation: pruning domain-specific channels and freezing domain-invariant ones for efficient recovery. Experiments on Domain-Digits, Office-Caltech-10, and PACS show that FU-DWS achieves superior performance over six baselines (Retrain, Rapid Retrain, FedEraser, Increase Loss, Class-Pruning, and FedSalUn), improving unlearning efficiency up to 25× while maintaining higher accuracy on retained domains.

### Strengths
++ The activation-based channel saliency approach is novel and intuitively appealing, providing a clear mechanistic interpretation of selective forgetting.

++ The experiments are comprehensive, including multiple datasets, six baselines, and ablation studies verifying each module (pruning, freezing, resetting).

++ FU-DWS achieves state-of-the-art trade-offs among forgetting effectiveness, retained-domain recovery, and efficiency, with consistent performance across CNN and ViT backbones.

### Weaknesses
-- The motivation for focusing on federated domain unlearning (FDU) is not well-justified, as the discussion on the limitations of conventional federated unlearning (FU) is not convincing. See C1.

-- The discussion on the unique challenges of FDU compared to conventional FU is not well-justified. See C2.

-- The discussion on related work is not comprehensive and lacks details. See C3.

-- The method design has clear limitations that need to be addressed, but without solutions or discussions. See C4.

Writing Issues (Only a few typos are listed here. Please carefully revise the submission):





Line 17: “Data domains” typically refer to data samples. This sentence needs revision.



Line 19: “Catastrophic forgetting” is a specific term used in continual learning, which refers to the backward transfer (e.g., performance degradation on downstream tasks) across tasks. Please carefully ensure the correctness when using such professional terminology.



Line 62: “existing methods” → lack of references.



Line 63: What is “collateral forgetting?” There is no such term in the cited survey.



Line 83: “Recovery & Forget & Generalization” → “recovery, forgetting, and generalization”.



Line 93: “Retrain” → “retrain”.



Line 129: “1.Retrain Unlearning” → “1. Retrain unlearning”.



Line 132: “. 2. Reverse” → “; 2. Reverse”.

### Questions
C1:

- Does “homogenous setting” refer to “single-domain?” Does “heterogeneous setting” refer to “multi-domain?”
- Assume the above statement is true. Then, “single-domain” means all clients have the same data distribution, while “multi-domains” means all clients have different data distributions. The submission claims that the motivation to focus on FDU is that conventional FU is within a homogeneous (single-domain) setting, making focusing on the heterogeneous (multi-domain) setting more practical. However, the prior work \[1\] does not explicitly make such an assumption in its method. Besides, the prior work \[2\] even explicitly demonstrates that it uses non-IID data. Thus, the lack of a well-justified justification of previous studies’ assumption on the homogenous domain makes the focus on FDU less convincing and practical.

\[1\] Yi Liu, Lei Xu, Xingliang Yuan, Cong Wang, and Bo Li. The right to be forgotten in federated learning: An efficient realization with rapid retraining. In IEEE INFOCOM 2022-IEEE Conference on Computer Communications, pp. 1749–1758. IEEE, 2022.

\[2\] Junxiao Wang, Song Guo, Xin Xie, and Heng Qi. Federated unlearning via class-discriminative pruning. In Proceedings of the ACM Web Conference 2022, pp. 622–632, 2022.

C2: Assume previous studies only focus on homogeneous settings as the submission claims.

- First, when discussing the unique challenges brought by the heterogeneous setting compared to the homogeneous setting, the comparison should be within FDU (heterogeneous) and conventional FU (homogeneous). However, the submission claims the comparison is on “domain generalization,” a topic highly orthogonal to FDU.
- Second, the submission claims that current FU methods “demand substantially more resources and iterations to recover” in the heterogeneous setting. However, no solid evidence or justification supports such a claim.

C3:

- Please provide more solid evidence to justify why previous studies have only focused on homogeneous settings and demand more resources in heterogeneous settings.
- Why do previous methods “have weaker theoretical foundations and may introduce greater privacy vulnerabilities?” Any evidence or reference?

C4:

- The design relies on heuristic thresholds $\\tau_p$, $\\tau_f$ and assumes channel-wise independence. The pruning and freezing thresholds strongly influence performance, yet are selected empirically without a principled criterion. In addition, the method treats each channel independently, ignoring potential inter-channel correlations, which could cause partial forgetting or unintentional degradation of domain-invariant features.

### Soundness
3

### Presentation
3

### Contribution
3
