# Fed-Duet: Dual Expert-Orchestrated Framework for Continual Federated Vision-Language Learning

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Pretrained vision-language models (VLMs), such as CLIP, have shown promise in federated learning (FL) by bringing strong multimodal representations to edge devices. However, continual adaptation remains a core challenge in practical federated settings, where task distributions evolve over time and data remain non-IID across clients. In this emerging area, recent works adopt parameter-efficient fine-tuning (PEFT) as a lightweight way to reduce communication overhead, yet they fail to preserve satisfactory performance under continual learning conditions. Meanwhile, traditional federated continual learning (FCL) methods lack the capacity to maintain cross-modal alignment crucial to VLM performance. We introduce Fed-Duet, a novel Dual Expert-orchestrated framework for efficient federated continual learning in vision-language models. Fed-Duet features a dual-expert adaptation mechanism, combining server-coordinated semantic prompts with client-personalized modular adapters. These pathways are dynamically fused via a cross-attention mechanism, enabling effective knowledge transfer while preserving multimodal alignment and mitigating forgetting. We evaluate Fed-Duet across multiple challenging continual learning tasks in federated vision-language settings and demonstrate that it achieves superior performance and stability compared to existing approaches. Our work highlights the importance of coordinated expert composition in enabling scalable and robust multimodal continual learning. The code is available at https://github.com/cocogt96/Fed-Duet.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Fed-Duet, a framework to address continual federated learning for CLIP models. The key idea is a dual-channel expert system: The paper adopts (1) semantic experts: server-coordinated prompts for high-level multimodal alignment and (2) parametric experts, i.e., client-specific adapters. A cross-attention fusion mechanism integrates these two channels, while auxiliary objectives preserve cross-modal consistency and mitigate catastrophic forgetting. Extensive experiments on multiple benchmarks demonstrate the promising performance of the method in both accuracy and efficiency.

### Strengths
1. The proposed method achieves promising results on multiple benchmarks.
2. The experiments are thorough and comprehensive.

### Weaknesses
1. The proposed method is complex, I would recommend refine/extend a bit on the details of the method design and different components. Also, I personally don't think "Semantic Experts" is a good naming for textual prompt. 
2. There are too much information included in Figure 1, maybe considering separate it into multiple figures.
3. The authors mentioned about "privacy-preserving feature" of the client images will be uploaded to the server. However, there lacks details about how the strength of the noise and whether it is still possible to reproduce the images at server-side. The author should discuss this more.
4. How would the attention gating influence the model performance? There seems to be no ablation about this specific component.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a dual expert-orchestrated framework for continual federated VLM. Various experiments confirms the effectiveness of proposed method.

### Strengths
The author conducts various experiments and ablation to confirm the effectiveness of proposed method.

### Weaknesses
This paper has several weakness.


1. The author utilizes the confused keyword expression.  As mentioned in the title, "DUAL EXPERT", But in the abstract, the author utilizes the dual channel. As far as i know, channel is widely utilized in feature dimension, such as Barlow Twins or other dimension-wise operation. 

2. Turn to your two experts, prompt and adapter, it is hard to understand why prompts refer to high-level and adapters refer to low-level. Because the prompt and adapter are both suitable to inject into each transformer layer. Authors should pay more attention to conducting the observation experiments rather than drawing a complicated framework without meaningful information.

3. Suspicious experiment results: Line 335 shows identical CIFAR-100 accuracy for T=10 across different beta values, with no explanation, harming the result's credibility. （84.20 75.97）Besides, the anonymous code does not release the compared methods. I do not trust your results.

### Questions
Refer to weakness.

### Soundness
1

### Presentation
1

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
Fed-Duet proposes a novel framework for federated continual learning with vision-language models, aiming to address two key limitations in existing approaches, i.e., adaptation imbalance and cross-modal misalignment. The method introduces a dual-channel expert-orchestrated architecture, where the semantic experts channel uses learnable prompts to achieve high-level semantic alignment and the parametric experts channel employs adapters for fine-grained task adaptation. Experiments across multiple datasets demonstrate that Fed-Duet achieves strong performance in terms of accuracy, knowledge retention and continual utility.

### Strengths
1. The paper clearly identifies a critical gap by noting that traditional federated continual learning ignores the multimodal nature of vision-language models, and existing PEFT-based federated learning methods often disrupt cross-modal alignment under continual learning.

2. The semantic channel uses a cross-attention gating mechanism to fuse prompt-based experts, while the parametric channel adapts adapter-based experts. The two channels are trained in a progressive manner that enables effective synergy.

3. The experimental evaluation is comprehensive, covering class-incremental, domain-incremental, and multi-domain task-incremental scenarios under various Non-IID settings and client scales. The results consistently show superior performance compared to strong baselines.

### Weaknesses
1. The method involves a large number of hyperparameters, including $\alpha$, $\beta$, and $\gamma$ in the loss function, as well as $\lambda$ and $\tau$ in the Eq (2) and Eq. (4). The paper provides neither justification for their chosen values nor sensitivity analysis, which limits reproducibility and practical deployment.

2. The choice to freeze parametric experts after round $r = R/2$ appears arbitrary. No theoretical reasoning or empirical ablation is provided to support this scheduling decision.

3. Several presentation issues reduce clarity.

    3.1. Notation is inconsistent. For example, $S_r$ in Eq. (1) is undefined, and the symbol $\beta$ is reused for both the Dirichlet parameter in data partitioning and a loss weight.

    3.2. Figure 1 is visually confusing, as the semantic experts are incorrectly shown as inputs to the image encoder, and the overall layout does not clearly reflect the workflow of Fed-Duet.

    3.3. Baseline names in Figure 3(a)(b) use excessively small font sizes, impairing readability.

4. The global prompt pool is initialized via K-means clustering on class embeddings, but the paper does not compare this strategy against random initialization or other alternatives, making it difficult to assess its actual contribution.

### Questions
1. In Stage 2, the parametric experts are frozen and only the semantic prompts are updated using cross-entropy loss. However, the image features have already been adapted by the parametric channel, while the text prompts are tuned independently. Could this decoupled update reintroduce cross-modal misalignment?

2. What is the rationale behind the specific timing for switching from parametric expert training to semantic expert training at $r = R/2$? Was this threshold tuned, or is it fixed across all settings?

### Soundness
3

### Presentation
2

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
This paper explores the challenging problem of continual federated vision-language learning. The authors propose a framework named Fed-Duet, which introduces a dual-channel design that integrates server-coordinated prompts and client-adapted experts to address challenges in cross-modal alignment, adaptation imbalance, and catastrophic forgetting. Extensive experiments demonstrate the framework’s effectiveness and efficiency.

### Strengths
- The topic is emerging and important. It represents a key challenge for deploying next-generation AI on edge devices, combining FL, CL, and VLM.
- The idea of dual-channel orchestration, unifying prompt-based semantic alignment and adapter-based parametric specialization, is conceptually clear and inspiring.
- Fed-Duet delivers a significant performance boost in continual learning while maintaining PEFT's communication efficiency, vital for resource-constrained FL.

### Weaknesses
- The authors introduce noise for privacy preservation, but the description of this part is limited and lacks sufficient details.
- RELATED WORK does not discuss a related work, CLIP2FL.
- Statistical reporting is incomplete. Providing statistics such as the mean and standard deviation would better support the robustness claim.

### Questions
- How does it identify and activate the correct experts to mitigate forgetting under the FL constraint of not explicitly storing old task data?

- The paper proposes that Fed-Duet can maintain critical cross-modal alignment performance in Vision-Language Models (VLMs). However, the experiments are only confined to vision-text classification tasks (CIFAR-100, Tiny-ImageNet, DomainNet), with classification accuracy serving as the core evaluation metric. How can the preservation of cross-modal alignment capability be directly demonstrated?

### Soundness
3

### Presentation
3

### Contribution
3
