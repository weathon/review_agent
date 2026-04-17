# PPLLaVA: Varied Video Sequence Understanding With Prompt Guidance

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
The past year has witnessed the significant advancement of video-based large language models. However, the challenge of developing a unified model for both short and long video understanding remains unresolved. Most existing video LLMs cannot handle hour-long videos, while methods custom for long videos tend to be ineffective for shorter videos and images. In this paper, we identify the key issue as the redundant content in videos. To address this, we propose a novel pooling strategy that simultaneously achieves token compression and instruction-aware visual feature aggregation. Our model is termed Prompt-guided Pooling LLaVA, or PPLLaVA for short. Specifically, PPLLaVA consists of three core components: the CLIP-based visual-prompt alignment that extracts visual information relevant to the user's instructions, the prompt-guided pooling that compresses the visual sequence to arbitrary scales using convolution-style pooling, and the clip context extension designed for lengthy prompt common in visual dialogue. Extensive experiments have validated the performance of our model. With superior throughput, PPLLaVA achieves better results on image benchmarks as a video LLM, while achieving state-of-the-art performance across various video benchmarks, excelling in tasks ranging from caption generation to multiple-choice questions, and handling video lengths from seconds to hours.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents PPLLaVA, a prompt-guided pooling framework designed for video-based large language models (Video LLMs). PPLLaVA focuses on compressing redundant visual tokens in video sequences while maintaining semantics that are critical to user instructions. The method achieves this through fine-grained vision–prompt alignment and a novel convolution-style pooling mechanism guided by textual prompts. In addition, it extends CLIP’s textual context to better support complex, multi-turn visual dialogues. Extensive experiments demonstrate that PPLLaVA attains state-of-the-art accuracy and efficiency across a broad range of image and video understanding benchmarks, delivering notable gains in both computation speed and task performance—particularly for long-duration video analysis.

### Strengths
**1. Validity and Soundness**: 
PPLLaVA introduces a highly effective token compression strategy, achieving a reduction ratio of up to 18x. This aggressive compression directly confronts a significant bottleneck in Video LLMs: the prohibitive computational cost associated with long context lengths. Notably, this reduction does not compromise, and in some cases even enhances, performance compared to state-of-the-art models across diverse video understanding benchmarks (Table 2).

**2. Strong Generalization**: 
The model's robust generalization is further highlighted by the experiments in Table 2. These results show that PPLLaVA can be seamlessly integrated with various mainstream MLLM backbones (such as LLaVA-Next, LLaVA-Video, and InternVL3), yielding consistent performance improvements across the board. This adaptability underscores its powerful generalization and transfer capabilities.

**3. Clarity and Concrete:** 
The paper is characterized by its clarity. The mathematical exposition is particularly lucid, especially the detailed formulations governing the alignment and pooling mechanisms (see equations in Section 3.2). Furthermore, the end-to-end pipeline is effectively illustrated in Figure 2, providing a clear visual representation of the entire architecture.

### Weaknesses
**1. CLIP Context Extension Ablation:**
 The paper posits the specific contribution of the CLIP context extension (asymmetric positional embedding). However, isolated ablation results for this module are not provided in Table 3. It would be valuable to evaluate its effect in isolation from the prompt-guided pooling module and to further compare it with other methods designed to extend CLIP's text length, such as Long-CLIP[1] and VideoCLIP-XL[2].

**2. Missing Efficency Analysis:**
Efficiency claims (18× compression) are strong but lack FLOPs or latency analysis (just “throughput (s/video)” — not standardized).

**3. Potential Over-reliance on CLIP-Based Semantics:**
The method's core relies on CLIP’s ability to associate text and visual fragments. While the authors acknowledge this (pages 2–3), there is no systematic analysis of whether this dependency limits extensibility to domains beyond CLIP’s pretraining. Furthermore, the CLIP context extension is only heuristically motivated and lacks ablation results isolating this component’s pure effect.

**4. Minor Issue:** 
line 021, "clip" should be "CLIP"

[1] Zhang B, Zhang P, Dong X, et al. Long-clip: Unlocking the long-text capability of clip[C]//European conference on computer vision. Cham: Springer Nature Switzerland, 2024: 310-325.

[2] Wang J, Wang C, Huang K, et al. Videoclip-xl: Advancing long description understanding for video clip models[J]. arXiv preprint arXiv:2410.00741, 2024.

### Questions
1.  Are the CLIP-based relevance scores $S=\{s(t,w,h)\}$ computed once (frozen) or updated during training?  If CLIP is frozen, how does the model adapt to new prompt distributions?  
2. How was the interpolation rate (r=1 before 20, r=0.25 after 20) in CLIP Context Extension chosen, empirically or theoretically?
3. The paper reports “throughput (s/video)” but not FLOPs, GPU memory, or latency at batch=1. Please report actual compute savings (e.g., FLOPs, peak memory, tokens/sec) to make the 18× compression claim quantifiable.
4. How sensitive is PPLLaVA to the CLIP backbone (ViT-L/14 vs. ViT-H/14 vs. SigLIP)? Have you tested whether replacing CLIP with a weaker vision encoder (e.g., EVA-CLIP) preserves benefits?

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
3

### Summary
This paper addresses the high computational cost and inefficiency of Video LLMs caused by processing redundant visual tokens. It proposes PPLLAVA, a model featuring a prompt-guided pooling strategy. The method uses CLIP-based visual-prompt alignment to generate a Attention map, which then functions as a dynamic 3D convolution-style kernel to aggressively compress visual tokens while retaining task-relevant information. A CLIP text context extension is also introduced to handle long prompts. Experiments on seven benchmarks demonstrate SOTA performance and significantly improved inference throughput.

### Strengths
1. The paper is well-motivated and addresses a critical and practical problem in Video LLM deployment: the trade-off between performance and computational efficiency. The motivation analysis in Section 3.1 (Table 1), which demonstrates performance degradation on high-redundancy videos, provides a solid empirical basis for the proposed solution.
2. The experimental validation is extensive and a significant strength of this work. The authors validate PPLLAVA's effectiveness across diverse video benchmarks.

### Weaknesses
1. The claim of model generality could be further substantiated. While the SOTA results on InternVL3 are strong , all in-depth ablation studies (e.g., Table 3, Table 5) are conducted exclusively on the LLaVA-Next version. Replicating key component ablations on the InternVL3 would be necessary to confirm the generalization. Furthermore, testing the method on other prominent open-source VLMs, such as the Qwen2/2.5-VL series, would significantly strengthen the paper's claims of being a general-purpose module.
2. The comparison to alternative token compression strategies is insufficient. The paper's main ablation (Table 5) only compares against internal variations (e.g., max pooling) and a "TOME" method. This TOME approach, originally proposed in 2022, is a relatively dated technique. This limited ablation does not adequately benchmark the proposed method against the current state-of-the-art. The paper would be more convincing if it included direct baseline comparisons against other full, established, and more recent token reduction methods to clearly isolate the benefits of the prompt-guided approach.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PPLLaVA, a video LLM that introduces a prompt-guided pooling strategy for adaptive token compression, aiming to achieve unified and state-of-the-art performance on both short and long video understanding benchmarks.

### Strengths
- The identification of redundant content as the core issue and the proposed prompt-guided pooling as a solution are well-motivated contributions to the field.

- The paper is well-written and easy to follow.

### Weaknesses
1.  The main contribution of this work is the design of a Q-Former-like module, integrated into a pre-trained MLLM, to reduce the number of visual tokens. The authors use the CLS token from the text encoder and compute its similarity with all visual tokens to parameterize a 3D convolution operation. However, since textual queries may involve spatio-temporal information, computing similarity with individual visual tokens, which lack temporal context, could introduce non-negligible errors.

2.  In the experiments, the authors continually fine-tune their method on three different MLLMs and compare it against their base versions. This comparison appears unfair, as their model has been exposed to additional training data. A more equitable evaluation would involve fine-tuning the baseline models on the same dataset for a valid comparison.

3.  The authors omit discussion and comparison with existing state-of-the-art training-free token pruning methods. When considering such approaches, the proposed method does not appear to offer clear advantages.

### Questions
See Weaknesses

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
4

### Summary
This paper introduces PPLLaVA, a novel video understanding framework that improves both efficiency and performance by compressing visual tokens based on user prompts. The core idea is to reduce video redundancy using a prompt-guided 3D pooling mechanism, which dynamically compresses visual tokens while preserving instruction-relevant information. The model includes three key components: a CLIP-based vision-prompt alignment module to identify relevant visual regions; a prompt-guided pooling mechanism that performs 3D convolution-style pooling using prompt-based attention weights; a CLIP context extension module to support longer text inputs for multi-turn dialogues.  
PPLLaVA achieves up to 18× token compression, and consistently outperforms strong baselines (e.g., LLaVA-OneVision, InternVL3) across 7 video understanding benchmarks, including long-form video tasks. It is also plug-and-play, transferable across different base models (LLaVA-Next, InternVL3, etc.).

### Strengths
1. Innovative and Practical Solution: The prompt-guided pooling mechanism effectively tackles visual token redundancy in long video understanding, also enhancing performance in image-only tasks, indicating broad applicability.
2. Strong Empirical Performance: PPLLaVA achieves state-of-the-art results across multiple benchmarks (e.g., Video-MME, NextQA, EgoSchema), often using fewer tokens and enabling faster inference.
3. High Generalizability and Efficiency: The method demonstrates strong transferability across various base models (image-only, video-only, and unified VLMs), significantly improving throughput—up to 3× faster—with minimal to no accuracy loss.

### Weaknesses
1. The prompt-guided pooling weights, derived directly from CLIP similarity scores without learnable parameters or adaptive gating, may restrict expressiveness and adaptability.
2. The paper lacks direct comparisons with other prompt-aware compression methods (e.g., VideoAgent, VideoTree) in terms of accuracy and efficiency, despite numerous SOTA model comparisons.

### Questions
1. Have you considered making the prompt-guided pooling weights learnable (e.g., via lightweight attention or gating mechanisms) instead of directly using CLIP similarity?
2. How does the model perform when user prompts are vague or generic (e.g., “What is happening?”)? Is there a quantitative analysis of performance degradation in such cases?

### Soundness
3

### Presentation
3

### Contribution
2
