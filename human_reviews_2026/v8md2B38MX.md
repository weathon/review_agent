# Rethinking Visual Information Processing in Multimodal LLMs

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Despite the remarkable success of the LLaVA architecture for vision-language tasks, its design inherently struggles to effectively integrate visual features due to the inherent mismatch between text and vision modalities. We tackle this issue from a novel perspective in which the LLM not only serves as a language model but also a powerful vision encoder. To this end, we present LLaViT–Large Language Models as extended Vision Transformers—which enables the LLM to simultaneously function as a vision encoder through three key modifications: (1) learning separate QKV projections for vision modality, (2) enabling bidirectional attention on visual tokens, and (3) incorporating both global and local visual representations. Through extensive controlled experiments on a wide range of LLMs, we demonstrate that LLaViT significantly outperforms the baseline LLaVA method on a multitude of benchmarks, even surpassing models with double its parameter count, establishing a more effective approach to vision-language modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes to improve the visual understanding capabilities of encoder-decoder large vision-language models through applying architectural enhancements over the LLM part. In particular, considering LLaVA as the baseline, the work proposes to learn separate QKV projection layers for the visual tokens as they are propagating through the LLM, and supplement this approach by utilizing a hybrid attention mechanism while also utilizing channel-wise concatenated visual tokens from different blocks of the visual encoder. Through experimental evaluation, the work aims to highlight the usefulness of the proposed methodology.

### Strengths
Below are the primary strengths of the work:

- The exact methodology, i.e. the combination of adding new QKV projections layers for visual tokens, considering visual tokens from different visual encoder blocks and applying bidirectional attention in this way on encoder-decoder VLMs is novel and interesting.
- The experiments demonstrate the desirable performance of the proposed method over the vanilla baselines, under a number of benchmarks with a number of models.
- The writing is clear and easy-to-follow.

### Weaknesses
Below are the primary weaknesses of the work:

**W1: Missing Citations in the Motivation Section:** The paper's primary motivation stems from the different approaches to token-level processing for visual versus textual data. While the analysis presented to support this motivation is sound, it fails to cite several critical works that have conducted similar explorations. Specifically:

- The paper uses the term "modality gap" without citing the original works that introduced it [A, B]. Although those studies may focus on contrastive learning, their findings are foundational to the exploration in this paper and require attribution.

- More importantly, the paper overlooks the work of [C], which had already explored and presented the majority of the findings discussed in the motivation section.

While I understand this analysis is intended to serve as motivation, in its current form, the text fails to establish these crucial links to the existing literature. Accordingly, if the explorations of the authors provide an orthogonal direction to the existing literature, e.g. over [C], the authors should clarify this in the text explicitly.

**W2: Ambiguities in Technical Contributions:** The authors have acknowledged (Lines 1002-1004 of Appendix D) that their methodology is not entirely novel, but rather is a _unique_ combination of existing methods, namely the additional QKV layers for visual tokens, bidirectional attention for visual tokens and the utilization of features from different visual encoder blocks. This is appreciated, though there are still several missing links to the existing literature, making the exact contributions of the work ambiguous. In particular:

- The idea of utilizing additional vision-specific parameters _within the LLM_ is explored quite well in the literature of monolithic VLMs [D, E, F, G, H], though the work claims to propose a "new paradigm" on this end. Notably,  EVEv1.5 [D] and Mono-InternVL [E] realize this idea through separating the FFN blocks, EVEv2 [F] does so by separating both the FFN and LayerNorm blocks and VoRA [G] realizes this goal by utilizing LoRA projections specifically for vision.  Although I acknowledge the novelty of the QKV aspect, I still believe that there should be at least a verbal discussion comparing LLaViT to these works with experimental comparison against them being the ideal and fair case.
- Using hybrid attention masking to allow visual tokens to attend each other bidirectionally while keeping the causal mask for textual tokens is a well-known practice in the VLM literature [I, J]. Unfortunately the main text of the work misses this and the narrative seems to present this as a novel aspect of LLaViT.

**W3: Additional Parameters and Costs:** As LLaViT learns separate QKV projection matrices, this incurs an unavoidable cost over the baseline LLaVA. Notably, this addition in parameters is included _on top of the existing visual encoder_. This might limit the future usage of the work for several memory or inference time constrained settings. 

---
[A] Liang, V. W., Zhang, Y., Kwon, Y., Yeung, S., & Zou, J. Y. (2022). Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. Advances in Neural Information Processing Systems, 35, 17612-17625.

[B] Schrodi, S., Hoffmann, D. T., Argus, M., Fischer, V., & Brox, T. (2024). Two effects, one trigger: On the modality gap, object bias, and information imbalance in contrastive vision-language representation learning. arXiv preprint arXiv:2404.07983.

[C] Shukor, M., & Cord, M. (2024). Implicit multimodal alignment: On the generalization of frozen llms to multimodal inputs. Advances in Neural Information Processing Systems, 37, 130848-130886.

[D] Diao, H., Cui, Y., Li, X., Wang, Y., Lu, H., & Wang, X. (2024). Unveiling encoder-free vision-language models. Advances in Neural Information Processing Systems, 37, 52545-52567.

[E] Luo, G., Yang, X., Dou, W., Wang, Z., Liu, J., Dai, J., ... & Zhu, X. (2025). Mono-internvl: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 24960-24971).

[F] Diao, H., Li, X., Cui, Y., Wang, Y., Deng, H., Pan, T., ... & Wang, X. (2025). Evev2: Improved baselines for encoder-free vision-language models. arXiv preprint arXiv:2502.06788.

[G] Wang, H., Ye, Y., Li, B., Nie, Y., Lu, J., Tang, J., ... & Huang, C. (2025). Vision as lora. arXiv preprint arXiv:2503.20680.

[H] Bavishi, R., Elsen, E., Hawthorne, C., Nye, M., Odena, A., Somani, A., & Tasırlar, S. (2023). Introducing our multimodal models. Adept Blog.

[I] Beyer, L., Steiner, A., Pinto, A. S., Kolesnikov, A., Wang, X., Salz, D., ... & Zhai, X. (2024). Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726.

[J] Steiner, A., Pinto, A. S., Tschannen, M., Keysers, D., Wang, X., Bitton, Y., ... & Zhai, X. (2024). Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555.

### Questions
In addition to the potential responses to the weaknesses I raised above, I have the following questions:

- Can you please clarify why only QKV projections were learned specifically for visual tokens? Why not separate the FFN blocks or LayerNorms blocks but only the QKV projection layers? How would LLaViT perform under these settings?
- Why consider the features from the 5th, 15th and 23rd blocks of the visual encoder? Why these block indices in particular? How does LLaViT work for different block choices?

### Soundness
3

### Presentation
2

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
The paper presents LLaViT, a new approach that rethinks visual information processing in multimodal LLMs (MLLMs). Instead of viewing the LLM as a purely linguistic component that receives aligned visual embeddings from a frozen vision encoder, the authors argue that the LLM itself performs substantial in-model translation of visual to textual representations. Based on this insight, they propose three architectural modifications that allow the LLM to act as an extended vision encoder:

1. Separate QKV projections for visual tokens — enabling modality-specific attention parameters to reduce misalignment between text and visual embeddings.  
2. Bidirectional attention on visual tokens — removing causal masking within visual tokens to improve information flow.  
3. Local-global visual features — concatenating multi-layer features from the vision encoder to provide both fine-grained and semantic representations.

These modifications are integrated into the LLaVA framework across multiple base LLMs (Qwen2.5 and Phi-3.5). On 17 multimodal benchmarks (e.g., GQA, TextVQA, RealWorldQA, MMVP), LLaViT consistently outperforms LLaVA-1.5 by a large margin. Ablation and visualization analyses show that the proposed mechanisms enhance visual token alignment, fine-grained comprehension, and visual-language consistency without major computational overhead.

### Strengths
- Clear conceptual motivation: reinterprets the LLM as part of the vision pipeline rather than a downstream consumer of features.  
- Introduces three simple yet effective modifications, each empirically validated with clear ablations.  
- Achieves substantial gains across diverse multimodal benchmarks compared to llava 1.5. 
- Solid ablations on multiple LLM scales, both standard and high-resolution settings, and consistent methodology.  
- Qualitative results provide intuitive evidence that visual tokens become more semantically aligned.

### Weaknesses
- The absolute performance remains relatively low compared to state-of-the-art models. While this is understandable given the limited dataset and compute budget, it raises questions about scalability. Can LLaViT be extended as a fine-tuning method for existing high-end MLLMs such as the Qwen-VL or InternVL series? Demonstrating compatibility with frontier models would substantially enhance the practical usability and relevance of this approach.

- The paper provides little discussion on the textual capabilities of LLaViT. It is well-known that multimodal fine-tuning often degrades the base LLM’s performance on pure-text tasks. Does the proposed method alleviate or exacerbate this issue? This is a minor concern and I understand if the author can't have results during the rebuttal.

- The experimental comparisons are limited to the LLaVA family. Can the proposed architecture generalize to other MLLM paradigms such as CogVLM, Flamingo, or BLIP-2 style frameworks? Since the modifications are largely architectural, validating cross-framework applicability would strengthen the paper’s generality.

- The ablation analysis lacks coverage of different vision encoders. The current experiments focus primarily on CLIP-H as the encoder. Additional results with other backbones (e.g., SigLIP2, or DINOv2) or even different scales of the same encoder would help confirm that the proposed mechanisms are not specific to a single visual backbone.

### Questions
See Weakness.

I would recommend a borderline acceptance at this stage. The paper presents clear insights and solid empirical evidence, but its current scope is confined to LLaVA-style architectures. I would be inclined to raise the score if the authors can demonstrate the generalizability of LLaViT to pretrained MLLMs and to other architectures beyond the LLaVA family, such as CogVLM or Flamingo.

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
3

### Summary
This paper introduces LLaViT (Large Language Models as extended Vision Transformers), a novel framework that rethinks the role of LLMs in multimodal systems by enabling them to function as vision encoders through three key modifications:
(1) learning separate QKV projections for visual tokens, 
(2) enabling bidirectional attention on visual tokens, and 
(3) incorporating local and global visual features. 
The authors argue that conventional LLaVA-like architectures suffer from a modality mismatch, where visual tokens are not well-aligned with the LLM's input space, and demonstrate that enhancing visual processing within the LLM leads to significant gains. Experiments across 17 benchmarks show that LLaViT outperforms LLaVA baselines, with a 3B model even surpassing 7B and 14B baselines on vision-centric tasks. Contributions include a new perspective on MLLM design, empirical validation of visual token translation in LLMs, and scalable improvements across diverse settings.

### Strengths
1. Originality: Reframing LLMs as vision encoders is genuinely novel. The three targeted modifications collectively close key visual-processing gaps, and the “visual token translation” mechanism (Sec. 2.2) offers fresh insight into MLLM internals.
2. Quality: Evaluation is comprehensive—17 benchmarks, multiple LLM families (Qwen2.5, Phi-3.5), and both Standard/Any-Res settings. Ablations (Tab. 3) and qualitative analyses (Figs. 3, 5–6) rigorously validate each component.
3. Clarity: The exposition is accessible and well-structured, with clear figures and tables; the motivation in Secs. 2.2–2.3 leads naturally to the methodological choices.
4. Significance: LLaViT achieves notable efficiency—smaller models outperform larger baselines—making it compelling for resource-constrained deployments, while the unified framework lowers integration costs and encourages broader adoption.

### Weaknesses
1. Baseline diversity: Comparisons stop at LLaVA-1.5; missing head-to-head results with recent MLLMs (e.g., Qwen-VL, InternVL) undercut claims of state-of-the-art performance.
2. Computational overhead: While parameter growth (5–12%, Tab. 4) and FLOPs (Tab. 5) are reported, end-to-end latency, peak memory, and throughput under realistic batch sizes/resolutions are not quantified—crucial for deployment.
3. Theoretical grounding: The benefit of semantic alignment is asserted but not explained; “visual token translation” remains descriptive without formal modeling (e.g., information-theoretic or perceptual analyses).
4. Data efficiency: Training hinges on high-quality PixMo-Cap; robustness in low-resource or noisy settings is untested.

### Questions
1. Generalization to video/3D: How would LLaViT extend to video or 3D? What architectural changes are needed to model temporal/spatial structure (e.g., frame tokenization, temporal pos. encodings, cross-frame attention, or 3D-aware token translation)?
2. Adaptive QKV routing: Could the separate Q/K/V projections be made input-adaptive (e.g., gating or MoE) rather than statically modality-routed to handle mixed-modality inputs more flexibly?
3. Bidirectional attention rationale: Why does bidirectional attention particularly help visual tokens? Is the gain tied to absent intrinsic ordering, and would similar effects appear in other set-structured modalities (e.g., point clouds)?
4. Parameter efficiency vs. performance: What are the accuracy–efficiency trade-offs? Could weight sharing, low-rank adapters, or pruning reduce overhead while preserving gains?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a visual information re-weighting pipeline intended to emphasize key regions in image representations and improve downstream recognition tasks. The framework is evaluated across several benchmarks to demonstrate performance improvements.

### Strengths
1. The paper is clearly written, and the structure is easy to follow.

2. The overall computational pipeline is straightforward to implement.

3. The motivation of improving visual emphasis and token weighting is generally relevant.

### Weaknesses
1. The claimed contribution based on the feature re-weighting strategy reads more like an intuitive design choice rather than a meaningful research idea.

2. The approach resembles common practices in project work, adjusting feature emphasis and reporting slightly improved accuracy without introducing new perspectives on the problem.

3. The paper does not discuss several closely related token aggregation approaches (e.g., DeepStack) and decouple transformers (e.g., MoT, EVEv2, Bagel, Mono-Internvl1.5). It is difficult to assess whether the proposed method has an obvious technical difference.

4. A major concern is that most of the compared baselines are far behind current state-of-the-art VLMs.

5. Only a small set of models and datasets are evaluated. Without comparison to modern, competitive baselines, it is unclear whether the method scales or holds advantage in realistic scenarios.

### Questions
1. How does this method differ in principle from existing token aggregation approaches and decouple transformers? A clear structural distinction is needed.

2. Could the authors provide deeper analysis beyond accuracy gains, for example, visual reasoning, interpretability, or sensitivity analysis, to demonstrate real insight?

3. The paper should include experiments based on recent SOTA VLMs, not just out-of-date baselines.

4. Please expand the conclusion to explicitly discuss limitations, the actual scope of applicability, and which scenarios this method should be favored for.

### Soundness
2

### Presentation
2

### Contribution
1
