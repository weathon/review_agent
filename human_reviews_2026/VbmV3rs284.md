# Fed-DIP: Federated Domain Generalization by Synergizing Implicit Disentanglement and Context-Aware Prompting

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Federated Domain Generalization (FedDG) seeks to train a model on decentralized data from multiple source domains that can generalize effectively to unseen target domains. A fundamental challenge lies in achieving robust feature disentanglement—separating domain-invariant from domain-specific features—which is critical for generalization but severely hindered by the data-isolated nature of Federated Learning. Existing methods often struggle with this, leading to incomplete decoupling and limited model performance. To address this, we propose Fed-DIP, a novel framework that introduces an Implicit Decoupling Distillation mechanism. This mechanism achieves fine-grained feature separation by comparing logit outputs for local image regions, all without direct data access. This allows for the robust aggregation of domain-invariant knowledge while critically preserving rich, domain-specific information at the client side. Furthermore, to unlock the potential of this preserved local knowledge, we introduce the Context-Aware Prompt Encoder (CAPE). Unlike prior works that rely on selective prompting from a fixed set, CAPE is a fully generative solution. It dynamically synthesizes adaptive, end-to-end optimizable text prompts directly from local visual features. These generated prompts provide nuanced, contextual guidance, enabling the model to effectively leverage domain-specific insights for more robust and accurate decision-making. Extensive experiments on benchmarks including PACS, VLCS, OfficeHome, and DomainNet demonstrate that our method achieves state-of-the-art (SOTA) performance, validating the effectiveness and superiority of our framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces FedDIP, a federated learning framework designed for domain generalization. It integrates two key modules: Multi-scale Implicit Decoupling Distillation (MIDD), which aligns intermediate features across clients to promote domain-invariant representation learning, and Context-Aware Prompt Encoder (CAPE), which leverages prompt-based conditioning to adapt local models to unseen domains. The method is evaluated on standard benchmarks (e.g., PACS, Office-Home) and demonstrates improved performance over existing approaches.

### Strengths
Combining multi-scale feature distillation (MIDD) with prompt-based adaptation (CAPE) is an interesting strategy for enhancing domain generalization in federated settings

### Weaknesses
1. The interaction between MIDD and CAPE is not deeply analyzed, making it hard to assess whether their effects are complementary or redundant.
2. Theoretical justification for why multi-scale feature alignment leads to better domain invariance is limited and could be strengthened.

### Questions
1. The proposed framework assumes that aligning multi-scale intermediate features (via MIDD) leads to domain-invariant representations, but this assumption lacks formal justification. Given that such features may encode both domain-specific and task-relevant information, how does the method avoid suppressing useful signals? Furthermore, is there empirical evidence that combining MIDD with CAPE results in a synergistic effect, rather than redundant or overlapping contributions?
2. How sensitive is the performance to the scale levels used in MIDD? Would using fewer or different layers (e.g., early vs. late) change the effectiveness of the distillation?
3. Since CAPE introduces prompt tokens into the encoder, how does its adaptation generalize to completely unseen domains with different semantic distributions?
4. Has the method been tested under heterogeneous data distributions (non-IID) across clients? If so, how robust is the framework to extreme distributional shifts?
5. Can the proposed method scale to larger federated settings (e.g., dozens of clients, varying compute) without compromising communication or convergence efficiency?

### Soundness
3

### Presentation
2

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
This paper proposes Fed-DIP, a framework for Federated Domain Generalization (FedDG) that aims to train models across multiple source domains without sharing raw data, while maintaining strong generalization to unseen target domains. The method introduces two key components: (1) Multi-scale Implicit Decoupling Distillation (MIDD), which performs logit-level, multi-scale knowledge distillation between the global and local models to implicitly separate domain-invariant and domain-specific features under data isolation; and (2) Context-Aware Prompt Encoder (CAPE), which generates adaptive text prompts from local visual features to enhance domain-specific representation. Built on a ViT-CLIP backbone with lightweight adapters, Fed-DIP achieves new state-of-the-art results on standard benchmarks including PACS, OfficeHome, VLCS, and DomainNet.

### Strengths
1.The paper is well written and easy to follow. The motivation, problem setting, and overall framework are clearly introduced, and the methodology is explained with sufficient clarity and structure. Figures and equations effectively illustrate the core ideas, making the technical flow easy to understand;
2.The proposed method achieves consistently strong results across multiple standard FedDG benchmarks, including PACS, OfficeHome, VLCS, and DomainNet. It outperforms several recent state-of-the-art approaches such as FedMaPLe, FedPR, and PLAN, demonstrating both the effectiveness and robustness of the proposed framework.

### Weaknesses
1.Although the paper claims to achieve disentanglement between domain-invariant and domain-specific representations through the MIDD module, there is no explicit experiment or visualization to validate this claim. The authors mainly report classification accuracy, but they do not provide a quantitative analysis of whether the extracted features actually exhibit domain invariance or reduced domain discrepancy. It would significantly strengthen the paper if the authors could include an additional experiment similar to the “Domain discrepancy of extracted features” analysis in ALOFT [1], where inter-domain feature distances are measured to quantitatively evaluate feature invariance; 
[1] Guo, Jintao, Na Wang, Lei Qi, and Yinghuan Shi. "Aloft: A lightweight mlp-like architecture with dynamic low-frequency transform for domain generalization." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24132-24141. 2023.
2.In Table 3, the full model and its ablated variants show inconsistent behavior across datasets. Specifically, the fourth row, which includes more components, performs worse than the third-row configuration on the OfficeHome dataset (84.97% vs. 82.43%), despite having additional modules that are expected to improve performance. The authors should provide further analysis or explanation to clarify why this degradation occurs and how the interaction between these modules affects model generalization;
3.Since the proposed framework introduces additional adapter modules into each Transformer block, it would be helpful to report the corresponding computational overhead or communication cost.

### Questions
The paper would benefit from further clarification on several points raised in the weaknesses. In particular, it remains unclear how the proposed framework truly achieves feature disentanglement, since no direct quantitative or visual evidence is provided. The authors are encouraged to include experiments similar to the “Domain discrepancy of extracted features” analysis in ALOFT [1] to validate the claimed invariance. Additionally, the inconsistent results observed in the ablation study and the lack of computational cost analysis deserve further explanation. A more detailed discussion of these aspects would make the paper’s contributions and claims more convincing.

### Soundness
2

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
5

### Summary
This paper introduces a novel framework for federated domain generalization, called Fed-DIP, integrates implicit feature disentanglement and context-aware prompt generation. Its dual-adapter design enables the model to separate and utilize both domain-invariant and domain-specific knowledge without raw data sharing. Extensive experiments on standard benchmarks demonstrate SOTA generalization performance across unseen domains.

### Strengths
(1) The paper introduces combination of implicit feature disentanglement and context-aware generative prompting for federated domain generalization task.

(2) The proposed Fed-DIP demonstrates consistent SOTA performance across multiple DG benchmarks.

### Weaknesses
(1) Related works section is very poor. No discussion of current prompt learning based Federated VLM methods. Lots of works need to be cited, [1]-[6]

(2) $\beta$ is missing in eq. 15.

(3) Questions are asked below.


[1] Global and Local Prompts Cooperation via Optimal Transport for Federated Learning, CVPR 2024

[2] FedMVP: Federated Multi-modal Visual Prompt Tuning for Vision-Language Models, ICCV 2025

[3] Federated Text-driven Prompt Generation for Vision-Language Models, ICLR 2024

[4] Harmonizing Generalization and Personalization in Federated Prompt Learning, ICML 2024

[5] FedPHA: Federated Prompt Learning for Heterogeneous Client Adaptation, ICML 2025

[6] Mixture of Experts Made Personalized: Federated Prompt Learning for Vision-Language Models, ICLR 2025

### Questions
(1) How are the generated prompts through visual guided text adapter, integrated with class tokens, as the total input tokens in CLIP are limited?

(2) Why does MIDD consider multiple level of scales? Is the multi-scale distillation truly improves disentanglement compared to single-scale or does it overfit?

(3) What is the effect of $\lambda$ in eq. 7 ? Why is it not considered in the ablation study of section 5.2?

(4) Current works like FedOTP [1], FedMVP [2], FedPHA [3] should be considered as baselines. 


[1] Global and Local Prompts Cooperation via Optimal Transport for Federated Learning, CVPR 2024

[2] FedMVP: Federated Multi-modal Visual Prompt Tuning for Vision-Language Models, ICCV 2025

[3] FedPHA: Federated Prompt Learning for Heterogeneous Client Adaptation, ICML 2025

### Soundness
2

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
4

### Summary
This paper introduces Fed-DIP (Federated Domain Generalization via Implicit Disentangled Distillation and Generative Prompting), a framework that enhances the generalization ability of vision-language models (VLMs) under federated non-IID settings. The approach has two main components: (1) Multi-scale Implicit Disentangled Distillation (MIDD), which separates global and local knowledge by aligning teacher-student logits at multiple spatial scales without exchanging raw data, and (2) Context-Aware Prompt Encoder (CAPE), which dynamically generates text-side prompts conditioned on instance-level visual context. Through selective gradient routing, the domain-invariant adapter $A_{di}$ is optimized toward shared global consensus, while the domain-specific adapter $A_{ds}$ captures local domain characteristics that are not aggregated. The proposed method is evaluated on PACS, VLCS, OfficeHome, and DomainNet, showing consistent improvements over existing federated domain generalization and prompt-based methods, including FedCLIP, PromptFL, and FedAPT. Ablation and sensitivity analyses further highlight the complementary roles of MIDD and CAPE and demonstrate favorable trade-offs between performance and communication efficiency.

### Strengths
- Conceptual novelty: The method unifies two important directions, federated domain generalization and prompt-based VLM adaptation, in a coherent framework.

- Algorithmic clarity in high level design: The MIDD component provides a clean way to separate global and local knowledge without data sharing, while CAPE dynamically conditions textual prompts on visual context.

- Strong empirical results: The model consistently outperforms both CNN-based and ViT-based federated DG methods on four benchmarks.

- Efficiency: Only lightweight adapters are communicated, achieving competitive results with significantly reduced communication cost.

### Weaknesses
- Missing formal definitions: The similarity weight $\gamma(s,i)$ is never formally defined. Its computation method must be clearly specified.
- Inconsistency in loss scheduling: The description of the loss weight $\lambda$ scheduling and the omission of the coefficient $\beta$ in the total loss equation create confusion about the actual implementation.
- Teacher mechanism ambiguity: The paper mixes two implementations (local teacher vs. server-broadcasted logits), leading to ambiguity about where and how teacher logits are computed.
- Communication cost mismatch: Reported values (2.019 MB vs 3.399 MB per round) conflict across text and figures, undermining reproducibility of the claimed efficiency.
- CAPE under-specified: The number of prompt tokens, their layer-wise injection strategy, and whether prompts are shared across classes are not detailed, which limits reproducibility.
- Notation and readability: Several key symbols are undefined at first use; a concise notation table would improve clarity. 
- Related work coverage: The paper omits foundational prompt-learning works such as *CoOp* [1], *CoCoOp* [2], and *ProDA* [3], which are essential for positioning CAPE’s generative contribution.
- Privacy discussion missing: The logit exchange step introduces potential information leakage that is not acknowledged or mitigated. Discussion of possible privacy-preserving mechanisms would be valuable.

[1] Zhou, Kaiyang, et al. "Learning to prompt for vision-language models." International Journal of Computer Vision 130.9 (2022): 2337-2348.

[2] Zhou, Kaiyang, et al. "Conditional prompt learning for vision-language models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[3] Lu, Yuning, et al. "Prompt distribution learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

### Questions
1. How exactly is $\gamma(s,i)$ computed and normalized? Which distance metric is used, and is temperature scaling applied?
    
2. Does the complementary KL term risk divergence when teacher and student outputs are nearly orthogonal? What stabilizers (e.g., entropy regularization or clipping) are employed?
    
3. Is the teacher model computed locally from the previous global parameters, or are averaged logits broadcast by the server? Please provide a single authoritative description.

4. How are the text prompts injected into CLIP’s text transformer? Are they concatenated as prefixes or inserted between class tokens?
    
5. Can the authors clarify the discrepancy in communication cost numbers and provide the exact accounting of each component?
    
6. Was any privacy-preserving mechanism (e.g., differential privacy, quantization, or noise) applied to the transmitted logits?

Please refer to the Weaknesses section for the remaining questions.

### Soundness
3

### Presentation
2

### Contribution
3
