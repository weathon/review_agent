# Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Vision token pruning has proven to be an effective acceleration technique for the Efficient Vision Language Model (VLM). However, existing pruning methods demonstrate excellent performance preservation in visual question answering (VQA) and suffer substantial degradation on visual grounding (VG) tasks. Our analysis of the VLM’s processing pipeline reveals that strategies utilizing global semantic similarity and attention scores lose the global spatial reference frame, which is derived from the interactions of tokens' positional information. Motivated by these findings, we propose N\"uwa, a two-stage token pruning framework that enables efficient feature aggregation while maintaining spatial integrity. In the first stage, after the vision encoder, we apply three operations, namely separation, alignment, and aggregation, which are inspired by swarm intelligence algorithms to retain information-rich global spatial anchors. In the second stage, within the LLM, we perform text-guided pruning to retain task-relevant visual tokens. Extensive experiments demonstrate that N\"uwa achieves state-of-the-art performance on multiple VQA benchmarks (from 94\% to 95\%) and yields substantial improvements on visual grounding tasks (from 7\% to 47\%). Code is released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies why vision token pruning in vision-language models hurts visual grounding much more than VQA, and proposes a new pruning framework called Nüwa to fix it. The authors analyze the VLM visual processing pipeline and argue that grounding depends on a global spatial reference frame built from positional embeddings and mid-layer vision tokens, while most pruning methods destroy this frame. They present a two-stage pruning method: (1) before the LLM, partition vision tokens into regions, pick salient tokens, and aggregate nearby tokens into them while keeping global spatial anchors intact; (2) inside the LLM mid-layers, prune again using text–vision similarity to keep only task-relevant visual tokens. They claim large speedups while keeping VQA performance and sharply improving grounding retention.

### Strengths
1. The analysis of attention/gradient flow, visual attention entropy, and object-centric cohesion across vision and LLM layers is thoughtful. It supports the claim that grounding relies on mid-stage spatial structure, not just final semantic alignment. This is valuable diagnostic science for the community.
2. Insight about position embeddings. The taxonomy of PE handling and the position reconstruction experiment is convincing.
3. Method makes sense relative to the diagnosis. Stage 1 is explicitly designed to preserve a dense global spatial scaffold. This is more principled than naive semantic clustering. Stage 2 then does task-aware pruning in LLM space, which matches the observed task divergence in mid layers. The story is coherent end to end.
4. They compare to random and pooling. They acknowledge that pooling is surprisingly strong and then explain why (implicit topology preservation).

### Weaknesses
1. The idea of a two-stage token pruning pipeline is not new,  similar observations and motivations have appeared in prior and concurrent works [1-9]. The authors should more carefully compare against these approaches, and make explicit what is actually novel here.
2. The RefCOCO results reported for the baselines differ notably from those in other works for example [2]. I may be misunderstanding the experimental settings, so the authors should clarify their setup and explain the discrepancy.
3. No failure analysis, for example, whether the text-conditioned Stage 2 introduces text bias (the model may discard visual evidence that contradicts misleading text).

[1] PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction.
[2] VScan: Rethinking Visual Token Reduction for Efficient Large Vision-Language Models.
[3] GLDS: Global–Local Diversity Selection for Scalable Token Pruning in Vision–Language Models.
[4] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?
[5] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration.
[6] Boosting Multimodal Large Language Models with Visual Tokens Withdrawal for Rapid Inference.
[7] ST3: Accelerating Multimodal Large Language Model by Spatial-Temporal Visual Token Trimming.
[8] Cross-modal Information Flow in Multimodal Large Language Models.
[9] LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference.

### Questions
1. How does this paper differ from the prior two-stage pruning approaches, and what are the novel contributions?
2. The RefCOCO results reported for the baselines differ notably from those in other works. Could the authors clarify their experimental settings and explain the reason for this discrepancy?
3. Has any failure analysis been conducted?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel method for VLM token pruning. The key insight motivating the approach is the recognition that preserving spatial information remains challenging yet is crucial for maintaining performance on grounding tasks after token reduction. The proposed method successfully leverages this insight to achieve superior performance and efficiency across a series of diverse VLM tasks.

### Strengths
* This paper offers a comprehensive examination of VLM token pruning, which yields valuable insights regarding the weaknesses of existing methods. 
* A novel and intuitive token pruning method is proposed and shown to be effective across a series of tasks.

### Weaknesses
* While the authors provide experimental results across diverse tasks, the selection of VLMs is limited. The generalizability and robustness of the proposed method should be confirmed by replicating the key results on a broader range of VLM architectures currently employed in the field.
* Furthermore, several tables and figures are currently too small and visually dense, making them difficult to read and interpret. The authors are encouraged to enlarge these elements and ensure that all text and data points are clearly legible.

### Questions
See above.

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
4

### Summary
This paper presents a two-stage token pruning framework, Nuwa, where the first stage focus on reducing the number of visual tokens by aggregating them via a proximity matrix, and the second state further prune the tokens with text guidance. The proposed Nuwa method is able to reach SOTA performance on multiple VQA benchmarks as well as visual grounding tasks compared to prior methods.

### Strengths
- The paper provides a thorough and insightful analysis of the challenges, pitfalls, and properties of token pruning in VLMs.
- The multiple metrics and visualizations in the analysis provides a solid support for the findings.
- The proposed Nuwa is compared with multiple current arts and is able to provide a higher pruning quality that retain the VQA and grounding accuracy with aggressive pruning.

### Weaknesses
- It is not clear how those findings have the contributed to the design of the Nuwa algorithm.
- In table 6, the results of FEATHER is only reported for the case of 192 tokens.
- The visual pruning part remain largely depend on the [CLS] token, so it is hard to apply the proposed method on vision backbones without [CLS].
- The text-guided token pruning is pretty strightforward and is not utilizing the prior insights.
- It is a two stage design that requires explicit text-guidance.
- The comparison against Visionzip is not very fair as Visionzip is a text-agnositc method which does not leverage text-guided pruning. It would be better to include the ablaiton study where the first-stage only Nuwa is compared with the Visionzip on several benchmarks.

### Questions
See weaknesses.

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
This paper tackles spatial degradation in vision token pruning for VLMs and proposes RPME to preserve spatial integrity during token reduction. The idea is clear and grounded in practical bottlenecks for high-resolution inference. Experiments across grounding and VQA benchmarks show meaningful gains, especially on spatial tasks, and the ablations strengthen the causal claims.

### Strengths
1. Clear motivation grounded in real multimodal deployment constraints.

2. Strong empirical gains on grounding datasets; convincing ablations.

### Weaknesses
1. The paper primarily evaluates on LLaVA-1.5-7B (and NeXT-7B), but does not include stronger or more modern VLM backbones such as Qwen-VL, InternVL. Since the field is rapidly standardizing around Qwen-VL and InternVL as competitive baselines, the absence of these comparisons leaves uncertainty about generalizability across architectures.

2. Lack of comparison with stronger baselines, such as PyramidDrop and Vscan.

[1] PyramidDrop: https://arxiv.org/abs/2410.17247
[2] Vscan: https://arxiv.org/abs/2505.22654

3. Limited discussion of failure modes or robustness to cluttered scenes and occlusions.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
