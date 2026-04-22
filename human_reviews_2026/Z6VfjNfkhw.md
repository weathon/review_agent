# CTA-Flux: Integrating Chinese Cultural Semantics into High-Quality English Text-to-Image Communities

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 6, 2

## Abstract
We proposed the Chinese Text Adapter-Flux (CTA-Flux). An adaptation method fits the Chinese text inputs to Flux, a powerful text-to-image (TTI) generative model initially trained on the English corpus. Despite the notable image generation ability conditioned on English text inputs, Flux performs poorly when processing non-English prompts, particularly due to linguistic and cultural biases inherent in predominantly English-centric training datasets. Existing approaches, such as translating non-English prompts into English or finetuning models for bilingual mappings, inadequately address culturally specific semantics, compromising image authenticity and quality. To address this issue, we introduce a novel method to bridge Chinese semantic understanding with compatibility in English-centric TTI model communities. Existing approaches relying on ControlNet-like architectures typically require a massive parameter scale and lack direct control over Chinese semantics. In comparison, CTA-flux leverages MultiModal Diffusion Transformer (MMDiT) to control the Flux backbone directly, significantly reducing the number of parameters while enhancing the model's understanding of Chinese semantics. This integration significantly improves the generation quality and cultural authenticity without extensive retraining of the entire model, thus maintaining compatibility with existing text-to-image plugins such as LoRA, IP-Adapter, and ControlNet. Empirical evaluations demonstrate that CTA-flux supports Chinese and English prompts and achieves superior image generation quality, visual realism, and faithful depiction of Chinese semantics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the challenge of extending English-centric text-to-image (T2I) models to support Chinese language prompts while preserving cultural semantics. The authors propose CTA-Flux, a lightweight bilingual extension of the Flux model, which introduces a Chinese Linguistic Attention Branch (CLAB) and a representation alignment loss to bridge linguistic and visual distribution gaps between English and Chinese prompts. Quantitative and qualitative experiments suggest that CTA-Flux achieves comparable or better performance than Flux on English benchmarks, while significantly improving cultural faithfulness for Chinese prompts.

### Strengths
1. Cross-lingual and cross-cultural alignment in generative models is underexplored yet increasingly important.
2. The approach extends an existing large English T2I model without retraining it from scratch, which is pragmatic for deployment.
3. Maintaining plugin compatibility (LoRA, ControlNet) is an appealing engineering consideration for community adoption.
4. The authors provide both quantitative (FID, CLIP score, GenEval) and qualitative (human cultural authenticity ratings) results to support claims.

### Weaknesses
1. The method is presented as generalizable to other non-English languages, but experiments focus exclusively on Chinese. It remains unclear whether the proposed architecture and training scheme would generalize effectively to languages with very different morphology or script systems.
2. While the paper aims to improve cultural authenticity, it does not discuss or measure potential cultural stereotyping or bias amplification, which are crucial ethical aspects of “cultural-aware” generation.
3. The cultural-specific prompt set appears hand-crafted and may not represent the full diversity of Chinese culture or real user prompts. A more systematic benchmark or open-sourced dataset would strengthen reproducibility.
4. The architectural idea (auxiliary language branch + feature alignment) resembles existing multimodal adaptation techniques; the contribution lies more in application and empirical validation than in theoretical innovation.

### Questions
No

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a concise and logically coherent idea, bridging the linguistic gap between Chinese and English, which drives T2I models comprehend cultures across different regions. The overall framework leverages language models for different scripts and seamlessly integrates them with MMDiT. The experiments include several image samples generated from Chinese prompts, providing a certain level of illustration for the model’s effectiveness.

### Strengths
1. The problem is highly relevant to real-world applications, as the paper focuses on cross-lingual and cross-cultural adaptation, offering significant practical value.

2. The method is designed with simplicity and efficiency in mind. It introduces no changes to the original backbone, preserving full compatibility with community plug-ins such as LoRA.

3. The training strategy is well-designed, employing a two-stage approach. The first stage aligns Chinese and English features, while the second stage concentrates on capturing Chinese cultural semantics, effectively balancing generality with cultural specificity.

4. The approach can, to a certain extent, be generalized to other languages, demonstrating broad applicability.

### Weaknesses
1. The definitions and metrics employed for the linguistic and cultural gaps remain somewhat vague, relying heavily on manual evaluation or CLIP-based similarity scores.

2. Although the study proposes metrics to assess the quality of images generated from Chinese prompts, it lacks an evaluation of the depth of Chinese language understanding—such as how well the model handles complex linguistic phenomena like polysemy, idioms, or cultural metaphors.

### Questions
1. What is your rationale for choosing MMDiT? Why not opt another backbone to accomplish this task?

2. The Chinese prompts used in the experiments mostly refer to culturally salient items such as festivals, clothing, and food. Could you offer some illustration of prompts that are culturally neutral but linguistically complex?

3. How can the model ensure that it truly understands the Chinese language, rather than merely piecing together stereotypically "Chinese-looking" visual elements?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes CTA-Flux, which adapts an English-only text-to-image model to support both Chinese and English.

### Strengths
The paper successfully trains a text-to-image model that understands the semantic meaning of Chinese text, enabling Flux to process Chinese prompts effectively.

### Weaknesses
This work lacks novelty: understanding Chinese semantics is achieved simply by replacing the text encoder and retraining the model.

### Questions
CTA-Flux is a foundational model. Could the authors provide evaluation results on public benchmarks, such as GenEval and T2I-CompBench?

### Soundness
3

### Presentation
3

### Contribution
1
