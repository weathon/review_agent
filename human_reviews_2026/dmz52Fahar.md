# Beyond Semantics: Rediscovering Spatial Awareness in Vision-Language Models

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Vision–Language Models (VLMs) excel at identifying and describing objects but often fail at spatial reasoning. We study why VLMs, such as LLaVA, under-utilize spatial cues despite having positional encodings and spatially rich vision-encoder features. 
Our analysis reveals a key imbalance: vision token embeddings have much larger norms than text tokens, suppressing LLM's position embedding. To expose this mechanism, we developed three interpretability tools: (1) the Position Sensitivity Index, which quantifies reliance on token order, (2) the Cross-Modality Balance, which reveals attention head allocation patterns, and (3) a RoPE Sensitivity probe, which measures dependence on rotary positional embeddings. These tools uncover that vision tokens and system prompts dominate attention.
We validated our mechanistic understanding through targeted interventions that predictably restore positional sensitivity. 
These findings reveal previously unknown failure modes in multimodal attention and demonstrate how interpretability analysis can guide principled improvements. We will release code upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Vision–Language Models (VLMs) excel at identifying and describing objects but often fail at spatial reasoning. We study why VLMs, such as LLaVA, under-utilize spatial cues despite having positional encodings and spatially rich vision-encoder features. Our analysis reveals a key imbalance: vision token embeddings have much larger norms than text tokens, suppressing LLM's position embedding. To expose this mechanism, we developed three interpretability tools: (1) the Position Sensitivity Index, which quantifies reliance on token order, (2) the Cross-Modality Balance, which reveals attention head allocation patterns, and (3) a RoPE Sensitivity probe, which measures dependence on rotary positional embeddings. These tools uncover that vision tokens and system prompts dominate attention. We validated our mechanistic understanding through targeted interventions that predictably restore positional sensitivity. These findings reveal previously unknown failure modes in multimodal attention and demonstrate how interpretability analysis can guide principled improvements. We will release code upon publication.

### Strengths
1) The paper has good presentations.
2) Enough experiments show the effectiveness of the proposed method.
3) The topic is interesting.

### Weaknesses
1) Underutilization of Spatial Cues: VLMs like LLaVA fail to effectively leverage spatial information despite having access to positional encodings, leading to deficiencies in spatial reasoning tasks.
2) Imbalance in Token Norms: The significant difference in norms between vision token embeddings and text tokens suppresses the model's ability to utilize position embeddings, hindering overall performance.
3) Dominance of Vision Tokens: The attention mechanism is dominated by vision tokens and system prompts, which can overshadow spatial cues and limit the model's reasoning capabilities.

### Questions
Underutilization of Spatial Cues: What specific strategies can be implemented to enhance the utilization of spatial information in VLMs, ensuring they effectively leverage positional encodings for improved spatial reasoning?

Imbalance in Token Norms: How can we adjust the norms of vision token embeddings relative to text tokens to create a more balanced representation that allows for better utilization of position embeddings?

Dominance of Vision Tokens: What modifications can be made to the attention mechanism to ensure a more equitable distribution of focus between vision tokens, system prompts, and spatial cues, thereby improving overall reasoning capabilities?

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
This paper studies why LLaVA-1.5 underutilizes spatial cues despite having positional encodings and spatially rich vision-encoder features. It revealed that vision token embeddings have much larger norms than text tokens, suppressing LLM's position embedding. The authors proposed three interpretability tools that uncover that vision tokens and system prompts dominate attention, which reveal previously unknown failure modes in multimodal attention.

### Strengths
S1: This paper is well-written with a nice flow.

S2: The investigated problem of spatial awareness is very important in VLMs.

S3: The findings are very interesting (if they are general in VLMs), especially vision token embeddings have much larger norms than text tokens, suppressing LLM's position embedding.

### Weaknesses
W1: This study is only based on LLaVa-1.5, which is considered relatively weak and old, given the rapid growth in VLM research. The observed problems on this particular model might not generalize to other VLMs. For example, Qwen3-VL might have alleviated the spatial awareness issue, and LLaVa-1.5 may simply be less well aligned in visual and textual spaces. Therefore, my main concern is that the findings in this work could not guide further development of VLMs

W2: A very related concern is that this work only focuses on the RoPE positional encoding, which has many extensions that are widely used in more recent LLMs/VLMs. So it’s hard to justify the significance of these findings. Is it possible to cover other VLMs with different positional encoding strategies?

W3: Since this paper includes a discussion of cross-modality balance (Section 5.2), it would be beneficial to include related works about modality imbalance issues in VLMs, including but not limited to IsoBench (COLM ‘24) and SEAM (COLM ‘25).

W4: The examples and constructed benchmark mainly focus on recognition rather than reasoning. I am concerned that in those more complex visual reasoning tasks, for example, Chess boards as image inputs, the permutation will result in very different outputs. However, the limitation on recognition makes the task too easy, so that the reliance on the image token ordering becomes less important. Is it possible to see if the conclusions still hold for visual reasoning tasks?

W5: VLMs are trained with the unnormalized distribution of token embeddings of the last layer. I don’t understand how it is possible that the model can still behave normally out-of-the-box when normalization and multi-layer features are incorporated without further alignment. It will be great if the authors could explain in more detail.

W6: The authors promised to release code upon publication, but the reproducibility is limited at the current stage. Any specific reason for this plan?

### Questions
Please refer to W2 and W4-6

### Soundness
2

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
3

### Summary
The paper studies why VLMs underuse spatial and positional cues. They introduce metrics like PSI, CMB, and RoPE-sensitivity to measure position use, and apply two light interventions, Normalize and Normalize+Multilayer, and reports consistent gains, both on interpretability metrics and spatial reasoning benchmarks.

### Strengths
1. The paper makes the "underuse of spatial cues" concrete. The finding that perturbing the RoPE presentation barely hurts performance is a strong and clear signal of the problem.
2. The interventions are minimal and simple to implement.
3. RoPE-sensitivity is a good probe than pure attention scores. It gives a more faithful read of whether the model actually uses positional information.

### Weaknesses
1. The experiment (compress to extreme)  shows in Figure 2 is confounded. It changes token count, spatial frequency bandwidth, and aggregation at the same time. This mixes the effect of "LLM’s RoPE only" with "extreme compression", not strong enough support the motivation presented in this section.
2. The link between Normalize and RoPE is indirect. The claimed mechanism is "Normalize balances vision tokens and RoPE embedding", which is not so direct to make the model sensitive to RoPE embedding. It also highlights text tokens. Besides, "+Multilayer" likely improves by adding richer features (an augmentation effect), which is not aligned with the stated goal of emphasizing position or highlighting RoPE.
3. Normalize is treated as the key factor, but there is no continuous control study (for example, gradually scaling visual/text norms, or selectively scaling subsets of tokens) to establish a causal dose-response curve.
4. VQAv2/GQA are general or compositional VQA, not designed for left/right/front/back/orientation/reference frame skills. POPE measures hallucination, which is orthogonal to spatial reasoning. The authors could consider adding What’sUp or GSR-Bench to directly test spatial relations.
5. The symbols in the figures are too small, making them hard to read (e.g., in Figure 7).

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

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
This paper investigates why Vision-Language Models (VLMs) like LLaVA can underutilize spatial information despite having positional encodings. The authors identify embedding norm imbalances between vision and text tokens as a key factor suppressing RoPE effectiveness. They introduce three interpretability tools and a synthetic dataset to measure spatial awareness, then propose two interventions: normalizing vision embeddings and incorporating intermediate vision encoder layers.

### Strengths
1. The paper is well-written and presents the idea clearly.

2. The paper attempts to address an important limitation in current VLMs regarding spatial reasoning capabilities.

3. The embedding norm analysis provides clear empirical evidence of the magnitude disparity between vision and text tokens.

4. The controlled 2DS dataset isolates spatial reasoning from semantic shortcuts, though with limitations.

### Weaknesses
1. Insufficient reproducibility details: Despite promising code release "upon publication," the paper lacks crucial implementation details. For example, the RMS normalization target values appear without justification; the H2 intervention also appears with very little detail and justifications for the choice of layers, etc.

2. Questionable dataset choices for analysis: The paper uses COCO validation set for most analyses (Figures 3, 4, 6, 7), but COCO is primarily object-centric rather than spatially-focused. This choice undermines claims about spatial mechanisms. Why not use spatially-rich datasets like various Spatial-VQA datasets [1, 2] or even their own 2DS throughout?

3. Limited evaluation of 2DS dataset: While 2DS aims to isolate spatial reasoning, it's overly simplistic (colored shapes in grids) and may not reflect real-world spatial complexity. Also, the paper doesn't validate whether improvements on 2DS transfer to natural images or whether the dataset truly eliminates semantic shortcuts as claimed. Again, why not using well-established Spatial-VQA benchmarks?

4. Attention head weights do not mean much in causality. As an interpretability work, I would assume it is trying to run intervention studies on salient attention heads or providing more interesting findings in the information flow, rather than merely saliency maps.

5. Statistical rigor lacking: Some results lack error bars or significance tests. Table 2's PSI differences could be noise. 

[1] Chen B, Xu Z, Kirmani S, et al. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 14455-14465.

[2] Zhang W, Zhou Z, Zheng Z, et al. Open3dvqa: A benchmark for comprehensive spatial reasoning with multimodal large language model in open space[J]. arXiv preprint arXiv:2503.11094, 2025.

### Questions
See Weaknesses. Also,

1. Why use COCO for spatial mechanism analysis when it's not designed for spatial reasoning? Have you validated your findings on truly spatial datasets?

2. What is the reason for directly correlating modality imbalance with attention heads? Honestly, I don’t find this transition very intuitive in the writing: “Functional behavior in Transformers is often head-specific, for example there are induction, copy, memory, summary, truthfulness attention heads in LLM. We therefore work at head granularity to give a closer look at the vision and text modality balance.”

### Soundness
2

### Presentation
2

### Contribution
2
