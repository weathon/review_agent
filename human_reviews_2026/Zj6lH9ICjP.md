# CAREFL: Context-Aware Recognition of Emotions with Federated Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Emotion recognition from images is a challenging task due to its dependence on subtle visual cues and contextual information. Recent advances in Vision-Language Models (VLMs) have demonstrated strong performance in this domain. Still, they are often limited by their large computational footprint and the privacy concerns associated with centralized training. To address these challenges, we propose CAREFL (Context-Aware Recognition of Emotions with Federated Learning), a framework for efficient emotion recognition. CAREFL combines large VLMs, specifically LLaVA 1.5, for generating rich contextual descriptions with lightweight small VLMs, SMOLVLM2, fine-tuned under a federated learning setup using Quantized Low-Rank Adaptation. This design enables accurate, privacy-preserving, and resource-efficient training on edge devices. Although this work evaluates CAREFL in the context of emotion recognition, the framework is general by design: leveraging VLMs, it can also be fine-tuned for a wide range of multimodal description and classification tasks beyond emotion analysis.
Extensive experiments demonstrate that CAREFL outperforms state-of-the-art baselines, achieving up to 96.49\% mAP and 50.36\% F1-score, surpassing heavier models such as GPT-4o, LLaVA, and EMOTIC. An ablation study further confirms the contribution of contextual enrichment, prompt design, and quantization in enhancing performance. The results show that federated fine-tuning of lightweight VLMs, when guided by contextual reasoning from large-scale models, provides a practical and scalable solution for emotion recognition in privacy-sensitive and resource-constrained environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a federated learning framework for emotion recognition from images, designed to balance contextual reasoning, privacy, and computational efficiency. The system operates in two stages: (1) a large vision–language model (LLaVA 1.5) generates contextual captions for each image, and (2) a lightweight vision–language model (SMOLVLM2) is fine-tuned with Quantized Low-Rank Adaptation (QLoRA) in a federated setting. This design enables decentralized training without sharing raw data while leveraging semantic context from the larger model. Experiments on EMOTIC and CAER-S datasets show that CAREFL achieves higher mean average precision and F1-scores compared to larger centralized models such as GPT-4o and LLaVA, while reducing memory usage and model size.

### Strengths
The paper’s contributions include: (1) proposing a novel two-phase federated framework combining large-model context generation with small-model adaptation, (2) introducing an efficient QLoRA-based fine-tuning scheme for lightweight federated training, and (3) comparative and ablation studies across datasets, client numbers, aggregation methods, and quantization settings.

### Weaknesses
Despite its technical framing, the paper appears conceptually weak and executionally shallow:

(1) The link between “context awareness” and federated learning is not clearly articulated. Context generation is performed offline using an existing large model, not integrated dynamically into the FL process. This makes the “context-aware” claim superficial.

(2) Illustrations and explanation lack clarity. Figures 1 and 2 are schematic and omit crucial architectural or algorithmic details; the paper mostly reuses known components (YOLO, LLaVA, QLoRA) with limited methodological innovation.

(3) Evaluations rely on narrow datasets (EMOTIC, CAER-S) without broader benchmarking or significant statistical analysis; performance comparisons against massive centralized models seem to be not fair and lack deeply analyzed.

(4) Many claims (e.g., “context improves emotion recognition”) are intuitive but not theoretically supported or quantitatively dissected. 

Overall, presentation feels more like a system demonstration than a rigorous ICLR-level contribution; key insights or innovations are missing.

### Questions
1. How exactly does “context awareness” influence the federated learning process? Does context affect model aggregation or only data preprocessing? Why was context generation performed offline instead of integrated dynamically during training?

2. How does the framework generalize to other tasks beyond emotion recognition?

3. How are biases or errors from LLaVA-generated captions mitigated during federated fine-tuning?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents CAREFL, a framework designed for efficient and privacy-preserving emotion recognition. First, a large vision-language model (LLaVA 1.5) generates contextual descriptions of images to enrich semantic information. Second, a lightweight model (SMOLVLM2) is fine-tuned using QLoRA within a federated learning setup. This method allows distributed training without sharing raw data. Experiments on the EMOTIC and CAER-S datasets show that CAREFL achieves high accuracy and F1-scores while significantly reducing computational and memory requirements.

### Strengths
1. The two-phase design cleverly combines large VLMs for context generation with lightweight models for federated learning is reasonable.

2. Experiments show strong performance, surpassing larger centralized models like GPT-4o and LLaVA.

3. The paper is well-written and easy to read.

### Weaknesses
1. The proposed two-phase design relies on rich contextual descriptions generated offline using LLaVA 1.5. However, in real-world or real-time emotion recognition scenarios, such offline pre-generation is impractical due to latency, computational overhead, and privacy constraints. This is inconsistent with the author's claim.

2. The experimental setup overlooks realistic aspects of federated learning, such as heterogeneous client data distributions, communication latency, and device variability.

3. Could you show examples of successful and failed predictions for a discussion?

### Questions
Please see Weaknesses.

### Soundness
2

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
The paper proposes CAREFL, a two-phase framework for multimodal emotion recognition that (1) uses a large frozen VLM (LLaVA-1.5) offline to generate rich scene/subject contextual descriptions, and (2) federatedly fine-tunes a small, efficient SVLM (SMOLVLM2) on client devices using quantized low-rank adapters (QLoRA). Experiments on EMOTIC (multi-label) and CAER-S (7 classes) show large gains in mAP and varying gains in F1/Recall.

### Strengths
1. This paper proposed a light-weight training approach, which is shown to be effective at achieving promising model performance.
2. This paper conducted comprehensive evaluation which covers different aggregation algorithms (FedDyn, FedAvg, FedProx, FedAdam), LoRA ranks, quantization settings (4-bit QLoRA vs full LoRA)
3. Large performance improve on EMOTIC benchmark

### Weaknesses
1. Claims of outperforming huge baselines need more careful parity. The paper states CAREFL outperforms GPT-4o, LLaVA and other heavy models — but many of these baselines are used in zero-shot or prompting setups while CAREFL is fine-tuned (and in federated settings).
2. Lack of evaluation benchmarks. The proposed models and baselines are mostly evaluated on EMOTIC. The results of the proposed model on CAER-S are not compared with any baselines.

### Questions
1. For results on EMOTIC, why mAP is so high while the recall and F1 are modest?

### Soundness
3

### Presentation
3

### Contribution
3
