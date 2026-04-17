# EAGLE: Enhanced Visual Grounding Minimizes Hallucinations in Instructional Multimodal Models

- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Large language models and vision transformers have shown impressive zero-shot capabilities, enabling significant transferability in downstream tasks. The fusion of these models has resulted in multi-modal architectures with enhanced instructional capabilities. Despite incorporating vast image and language pre-training, these multi-modal architectures often generate responses that deviate from the ground truth in the image data. These failure cases (false positives) are known as hallucinations. Current methods for mitigating hallucinations generally focus on regularizing the language component, improving the fusion module, or ensembling multiple visual encoders to improve visual representation. In this paper, we address the hallucination issue by directly enhancing the capabilities of the visual component. Our approach, named EAGLE, is fully agnostic to the LLM or fusion module and works as a post-pretraining stage that improves the grounding and language alignment of the visual encoder. We show that a straightforward reformulation of the original contrastive pre-training task results in an improved visual encoder that can be incorporated into the instructional multi-modal architecture without any additional instructional training. Extensive empirical validation shows that EAGLE significantly reduces hallucinations across six different instructional multi-modal models and four challenging benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces EAGLE (Enhanced Visual Grounding Minimizes Hallucinations in Instructional Multi-modal Models, a technique aimed at reducing hallucinations in multi-modal models (Vision-Language Models, VLMs) by enhancing the visual encoder's grounding. EAGLE modifies the vision transformer (ViT) in a post-pretraining stage to improve visual grounding without the need for additional fine-tuning or instruction training. The authors claim that this approach significantly reduces hallucinations across six different IT-VLMs and four benchmarks. The paper provide experiments to demonstrate that EAGLE enhances visual grounding, enabling more accurate and grounded language generation from images.

### Strengths
1. EAGLE introduces a unique solution by focusing on improving the visual encoder, rather than relying on complex changes to the language model or fusion module. This post-pretraining enhancement makes it simple to integrate into existing systems.
2. The method shows strong performance across multiple IT-VLMs and benchmarks, including some challenging scenarios such as MMVP and MERLIM, demonstrating its broad applicability and robustness.
3. EAGLE's ability to reduce hallucinations without requiring additional instructional training or fine-tuning of the entire model is a significant advantage in terms of both efficiency and scalability.

### Weaknesses
1. While the paper provides some experimental validation, it lacks in-depth theoretical explanation and a detailed analysis of why the proposed modifications lead to such improvements.
2. The hallucination problem may potentially be over-simplified. The solution focuses primarily on the visual component, which may not address the full range of hallucination causes. Further investigation into the interplay between the visual and language components might yield more comprehensive solutions.
3. EAGLE relies on pre-labeled datasets like OpenImages V7 for training, which limits its flexibility and raises questions about its generalizability to other types of training data or modalities.
4. EAGLE might introduce computational overhead due to the training of enhanced visual encoders. It is suggested that the authors fully explore the computational cost, especially when integrating into larger systems with more extensive multimodal components.

### Questions
Please refer to weaknesses.

### Soundness
2

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
This paper proposes a method to mitigate hallucinations in VLMs through improved visual encoder capabilities. Experimental results on MMVP and POPE benchmarks show performance gains with the enhanced visual encoder.

### Strengths
1.	Hallucination remains an important challenge for VLMs, and enhancing the visual encoder provides a promising direction for mitigation.

2.	EAGLE demonstrates performance gains on both MMVP and POPE.

3.	The EAGLE method is clearly presented, and Figure 2 offers a concise overview of the approach.

### Weaknesses
1.	It is incorrect to claim that EAGLE does not require instructional training. As stated in Line 446, applying EAGLE to LLaVA actually requires instructional training. Considering that LLaVA represents one of the most representative approaches for building VLMs, this method may not be directly applicable to modern VLMs.

2.	The experiments only demonstrate that EAGLE can reduce hallucination. Since the method introduces some misalignment with the original model, it is important to assess how other capabilities are affected. I recommend that the authors include evaluations on MMBench and OCRBench.

3.	The overall performance on MMVP is not strong, and it is inaccurate to claim that EAGLE achieves state-of-the-art results. Previous work, such as ROSS [A], achieved 49.3% by enhancing the visual encoder during the instructional tuning phase, which is substantially higher than the 34% reported in this manuscript.


[A] Wang, H., Zheng, A., Zhao, Y., Wang, T., Ge, Z., Zhang, X., & Zhang, Z. Reconstructive Visual Instruction Tuning. ICLR 2025

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Contrary to the traditional methods that rely on additional instructional data, fine-tuning the LLM component,
enhancing the adapter module, or ensembling multiple visual encoders to improve visual representation, this work
employs a tuning strategy to enhance fine-grained visual grounding directly within the visual encoder. Then, the
tuned visual encoder will be integrated seamlessly into an IT-VLM without requiring additional instructional training
or alignment.

To this end, this work utilized the extra dataset OpenImagesv7, which contains instance masks, to pop out instance
features from the visual encoder. Then, they will be aligned with the encoded text embeddings through contrastive
learning. Once the fine-tuned visual encoder is obtained, it will be integrated into an IT-VLM. The whole model will be
fine-tuned with parameter-efficient tuning.

This work improves hallucination metrics across four standard benchmarks and six diverse architectures, which is
also consistent with the significant improvements in the challenging scenarios for VLM proposed in the MMVP-VLM
benchmark.

### Strengths
1. This work addresses the hallucination issue in IT-VLM from the perspective of enhancing the fine-grained visual
grounding capability of the visual encoder directly for the first time, which sound interesting.
2. This work is proven with sufficient experiments to consolidate that is a straightforward and effective approach
that mitigates hallucinations in IT-VLMs without requiring any additional alignment or tuning.
3. The proposed method to fine-tune the visual encoder is simple and easy to follow.

### Weaknesses
1. All experimental settings are focused solely on three visual encoders: EVA01 ViT-g-14, OpenAI ViT-L-14-336, and
OpenAI ViT-L-14-336. Therefore, it remains to be seen whether this method can be applied to other stronger
visual encoders, such as SigClip-SO400M, to illustrate the generalizability of the approach.


2. Can the solution EAGLE be applied to the latest frameworks for addressing the hallucination issue in more
modern IT-VLMs, e.g. SigClip + Qwen (Since I can't find the corresponding experiment in both the main paper or supplementary materials)? Meanwhile, It would be better to quantitatively illustrate the extent to
which it improves the hallucination problem?


3. Please kindly evaluate the EAGLE solution on the general benchmarks, such as MMbench, MME, and ORCbench,
to estimate its performance improvement on the general leaderboard. That comparison with the latest
technique -- AnyRes proposed in Llava-Next[1] is also recommended.

4. Ross [R1] shares a similar motivation by introducing a reconstructive objective aimed at enhancing fine-grained
comprehension capabilities, which aligns with the focus on improving local visual representation in the current study.
Therefore, incorporating a discussion of Ross [R1] is encouraged.

** Minor Weakness**

1. **Missing training details**: The paper lacks details on the training setup for the visual encoder. It would be helpful to provide more details about the layers that are optimized during the training stage. Please describe what comprises of
Full-finetune and GaLore as well as more details about the training setup for each of them. Is this consistent over all
the visual encoder architectures shown in the paper? What consists of a batch during the training process? For a
given image, are the instances in the image considered as separate samples or are they considered as a single
sample? What is the effect of using multiple prompts (from CLIP based open vocabulary segmentation) on the
training process? If you are fine tuning the visual encoder, aren't you losing the zero-shot capabilities of the CLIP
model? Wouldn't it just overfil on the 350 classes of OpenImages V7? It would be great to provide evaluation on
images that have object classes not present in OpenImages V7.

[R1] H. Wang et al. Reconstructive Visual Instruction Tuning. arXiv 2410.09575

### Questions
1. The core methodology is essentially an extension of contrastive learning. While it effectively aligns visual patches with language via masked average pooling, the underlying concept is not particularly novel, representing more of an incremental advancement and a refinement of existing techniques rather than a novel one.

2. A critical question regarding its necessity is whether a simpler, training-free approach could achieve a similar effect. For instance, one could pre-process the image using an off-the-shelf segmentation model, tag the detected objects with their class labels, and then feed this enriched representation into the VLM. It is plausible that such a heuristic, training-free method would also significantly reduce object hallucinations.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a post-training strategy termed EAGLE for CLIP-style models. Specifically, EGALE employs a masked average pooling to a specific object and computes an averaged representation for contrastive learning. The features are expected to be more fine-grained than the original ones, and thus reduce hallucinations for downstream applications. Experiments under various evaluation protocols demonstrate the effectiveness of the proposed method.

### Strengths
1. The motivation is clear and somewhat reasonable.
2. The paper is easy to follow and the method seems to be simple but effective.

### Weaknesses
1. Limited literature review. Throughout the paper, there are no references later than 2025. Actually, works like [1, 2] also try to solve the fine-grained problem, which have been published in ICLR 2025, not to mention their follow-ups.
2. Old baselines. The comparison baselines are too old. Specifically, 
    - For CLIP-style VLMs, new baselines, such as SigLIP2 and AIMv2, may partially alleviate the fine-grained problem. I doubt the effectiveness of EAGLE on these advanced backbones.
    - For LLaVA-style MLLMs, advanced methods usually *unfreeze* the visual backbone, e.g., [1] and LLaVA-NeXT. Therefore, the improvements of EAGLE may become marginal.
3. Lack of comparison with other related methods, e.g., [2].

**References**

[1] Reconstructive Visual Instruction Tuning. ICLR 2025.

[2] Diffusion Feedback Helps CLIP See Better. ICLR 2025.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
1
