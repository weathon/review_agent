# Seeing but Not Believing: Probing the Disconnect Between Visual Attention and Answer Correctness in VLMs

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Vision-Language Models (VLMs) achieve strong results on multimodal tasks such as visual question answering, yet they can still fail even when the correct visual evidence is present. In this work, we systematically investigate whether these failures arise from not perceiving the evidence or from not leveraging it effectively. By examining layer-wise attention dynamics, we find that shallow layers focus primarily on text, while deeper layers sparsely but reliably attend to localized evidence regions. Surprisingly, VLMs often perceive the visual evidence when outputting incorrect answers, a phenomenon we term "seeing but not believing" that widely exists in major VLM families. Building on this, we introduce an inference-time intervention that highlights deep-layer evidence regions through selective attention-based masking. It requires no training and consistently improves accuracy across multiple families, including LLaVA, Qwen, Gemma, and InternVL. These results show that VLMs encode reliable evidence internally but under-utilize it, and that making such signals explicit can bridge the gap between perception and reasoning, advancing the diagnostic understanding and reliability of VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates the cause of hallucinations in Vision-Language Models, hypothesizing that these arise because models perceive visual information but fail to leverage it effectively. The authors analyze attention distributions across layers, observing that early layers predominantly attend to textual features, while deeper layers focus more on visual evidence, and this focus is not always used for reasoning. To address this, they propose a steering mechanism that explicitly increases attention to visual evidence regions, enhancing the model’s grounding on visual cues.

### Strengths
- The paper offers an insightful analysis of attention localization across layers, showing that deeper layers indeed attend more to semantically relevant image regions when forming answers.

- The VEA step is methodologically well-designed, particularly with its denoising and Gaussian smoothing components to maintain spatial coherence in the attention masks. The method provides a simple yet interpretable tool for reducing hallucinations.

### Weaknesses
- The comparison between text and image RATP values (Fig. 1) is not clearly justified, given that the two modalities operate on different attention scales. It remains unclear how a small increase in image RATP can be interpreted as a “modality shift,” especially when text attention is still orders of magnitude higher (e.g., 0.2 → 0.6 for image vs. ~20 for text). This weakens the claim that “vision plays a stronger role in later inference stages.”

- Prior studies (e.g., [1], [2]) have reported different attention trends, showing that image tokens already dominate attention in early layers. The authors should better situate their findings within this literature, clarifying methodological or model-related differences that explain these discrepancies.

- The interpretation of Fig. 4 could be refined: while the authors claim that evidence tokens receive higher attention even in incorrect answers, the plot also shows a drop in attention between correct and incorrect predictions, which complicates that conclusion.

- The assumption that deep layers consistently capture “ground-truth” evidence regions is empirically plausible but not guaranteed. The paper would benefit from a clearer discussion on the reliability of these “evidence layers” across architectures and datasets.

- It is also unclear whether the identification of such evidence layers must be repeated per model, and how sensitive the steering mechanism is to inaccuracies in this identification.

[1] Amara, Kenza, et al. "Why context matters in VQA and Reasoning: Semantic interventions for VLM input modalities." arXiv preprint arXiv:2410.01690 (2024).

[2] Lu, Haolang, et al. "Mitigating Hallucination in Multimodal Reasoning via Functional Attention Control." arXiv preprint arXiv:2510.10285 (2025).

### Questions
The analysis appears to be based on single-token inference. Are the reported results averaged across all generated tokens for each answer, or do they correspond to specific tokens only?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a phenomenon called “seeing but not believing” by studying the internal layers of vision-language models (VLMs). Through analyses of layer-wise attention dynamics and layer-wise profiling, the authors introduce a visual evidence augmentation approach that enhances regions likely to contain stronger visual evidence. The proposed method is training-free and provides both improved interpretability and better performance for VLMs.

### Strengths
1. The paper is well written and easy to follow. The phenomenon of “seeing but not believing” is intriguing.

2. The experiments and ablation studies are comprehensive.

3. The proposed VEA approach provides both interpretability and performance improvement.

### Weaknesses
1. Some experimental details are missing. For example, in Figures 2 and 3, it is unclear which models were used for attention map visualization.

2. Including more visual examples could strengthen and better support the overall narrative.

### Questions
Did the authors observe the attention sink phenomenon [1,2] during their experiments? If so, how did they handle these situations?

Reference:
[1] See What You Are Told: Visual Attention Sink in Large Multimodal Models, ICLR 25
[2] Vision Transformers Need Registers, ICLR 24

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work investigates the phenomenon of “seeing and not believing”, where vision-language models (VLMs) attend to the correct visual regions but still produce incorrect answers. To address this issue, the authors propose an inference-time method that encourages VLMs to better leverage visual information by overlaying attention-derived masks on the input image. Experiments across four VQA benchmarks and four VLM families demonstrate the effectiveness of the proposed approach.

### Strengths
1. The authors conduct a thorough investigation of how different layers in VLMs process inputs and distribute attention, providing valuable insights for the community.
2. The proposed solution is simple yet effective, showing consistent  improvements across eight different VLMs.

### Weaknesses
1. It is important to evaluate cases where the model does not attend to the correct regions.
How often does the model still answer correctly in such cases?
Does performance degrade when highlighting regions based on incorrect attention?
2. The motivation is somewhat similar to [1], which also identifies this phenomenon and proposes attention-based approaches.
This overlap reduces the novelty of the contribution, though the improvements of the method are still appreciated.


[1] "Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly", Liu et al.

### Questions
1. In cases where the model attends to incorrect regions but still answers correctly, what happens when the attention-derived mask is applied?
Does it lead to performance degradation in such examples?
2. I would be interested to see qualitative examples that explore attention behavior in more general, real-world scenes, beyond the text-centric scenarios that are (to my understanding) the main focus of the examined datasets.

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
4

### Summary
This paper focuses on the "seeing but not believing" phenomenon in Vision-Language Models (VLMs): VLMs often perceive correct visual evidence in Visual Question Answering (VQA) but fail to leverage it for accurate answers. Through layer-wise attention analysis, it reveals three key findings: shallow VLMs layers focus on text, deeper layers sparsely attend to localized evidence regions, and deeper layers still lock onto evidence even when outputs are wrong. To solve this, the paper proposes VEA (Visual Evidence Augmentation), a training-free inference-time method. It first identifies "visual-grounding layers". During inference, VEA extracts attention from these layers, applies denoising and Gaussian smoothing to create an evidence mask, and fuses the mask with the original image to highlight evidence (weakening non-evidence regions).

### Strengths
1. The paper presents an insightful observation on VLMs’ behaviors toward images. Its visualizations and analyses further reveal how text and image interactions are modeled across different layers, showing that the encoding of semantic features first emerges in deeper layers. The inconsistency between attention maps (i.e., "seeing but not believing") highlights limitations of current VLM architectures, which is valuable for guiding future research.

2. To address this issue, the authors propose a simple yet effective algorithm. The design is straightforward and practical: it can be applied to various VLMs with zero training cost and demonstrates effectiveness across multiple benchmarks.

### Weaknesses
1. The method improves VLM performance by overlaying a salient mask on the input image, but it does not "fix the VLM’s attention behavior" (as the behavior of the VLM or attention is not changed). Additionally, the design of the algorithm will introduce extra cost, and also raises the convern about multi-turn/multi-image scenarios.

2. The proposed algorithm augments the brightness of different regions of the image. However, this augmentation will change the original image, causing information loss and changes (e.g., this will influence questions about brightness or color). For questions that rely on global context or beyond retrieval, the proposed mask method might cause troubles.

3. Compared with visual reasoning methods that interactively retrieve key parts of the image to gather information, the proposed approach is plug-and-play, however, it also introduce limitations. This point could be further discussed in the paper.

### Questions
1. The paper proposes to improve the network attention by casting masks on the input image. Why not apply this method to intermediate features or attention?
2. How does this method work on multiturn conversations, multi-image QA or video?3. 
3. How does this method influence the benchmarks that rely on global context or multiple elements, such as benchmarks for general knowledge understanding or math (such as MMMU, MMStar, AI2D, MathVista)?
4. As the author suggested, the answer from a single inference might not be accurate. The same applies to the attention map — can this method apply to the same question and image multiple times in a cascade way?

### Soundness
4

### Presentation
3

### Contribution
3
