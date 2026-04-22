# FLARE: Fully Integration of Vision-Language Representations for Deep Cross-Modal Understanding

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
We introduce FLARE, a family of vision language models (VLMs) with a fully vision-language alignment and integration paradigm. Unlike existing approaches that rely on single MLP projectors for modality alignment and defer cross-modal interaction to LLM decoding, FLARE achieves deep, dynamic integration throughout the pipeline. Our key contributions include: (1) Text-Guided Vision Encoding that incorporates textual information during vision encoding to achieve pixel-level alignment; (2) Context-Aware Alignment Decoding that aggregates visual features conditioned on textual context during decoding for query-level integration; (3) Dual-Semantic Mapping Loss to supervise feature mapping from both modalities and enable modality-level bridging; and (4) Text-Driven VQA Synthesis that leverages high-quality text to generate VQA pairs and synthesize corresponding images, enabling data-level optimization. We train FLARE at 3B and 8B scales under both fixed and dynamic resolution settings, demonstrating that our full-modality alignment significantly outperforms existing methods while maintaining strong generalizability. FLARE 3B surpasses Cambrian-1 8B and Florence-VL 8B using only 630 vision tokens. Ablation studies reveal that FLARE achieves superior performance over existing methods with minimal computational cost. Even without dynamic resolution, FLARE outperforms LLaVA-NeXT, validating the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FLARE, a new family of vision-language models (VLMs) that aim for a deeper integration of vision and language representations. Unlike previous methods that often treat vision features independently and only fuse them with text during decoding, FLARE proposes a fully vision-language alignment and integration paradigm. With extensive experiments verified its effectiveness.

### Strengths
1. The paper is clearly structured and easy to follow, making the proposed methodology and experimental results comprehensible.
2. The authors have conducted extensive experiments across various benchmarks and scales, demonstrating the effectiveness and generalizability of FLARE.

### Weaknesses
1. The effectiveness of a simple MLP projecting text tokens into vision space for a vision encoder with significantly fewer parameters than the LLM is unclear. I'm curious about how the vision encoder can effectively process these projected text tokens.
2. Equation 3 appears to have typos. Should it be $(V_i,V_q)$ instead of $V_i(V_q)$?
3. While Qwen2.5VL 7B is presented as a strong baseline, it would strengthen the paper to include experiments based on Qwen2.5 model series to further demonstrate the superiority of the proposed method.

### Questions
follow weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces FLARE, vision-language models with several designs to enable cross-modal alignment and interaction across in visual encoders and LMM backbones. Besides architectural innovation, a data synthesis strategy is proposed to construct text-centric data. Experimental results validate the effectiveness of key components in FLARE.

### Strengths
* This paper is strongly motivated, compared with prior work’s preliminary explorations, this paper conducts a comprehensive exploration of how to enhance text-guided visual encoding.
* The proposed Context-Aware Alignment Decoding module is interesting, providing a novel solution to refine in-context visual features layer by layer.
* Experimental results support the effectiveness of FLARE, the authors also provide analysis on additional computational cost, which is important to analyze the introduced modules.

### Weaknesses
* Presentation of this paper can be improved:
  1. Horizontal lines and citations are missing in Table 1
  2. In Equation 3, would it be $(V_i^s, V_q^s)$ rather than $V_i^s(V_q^s)$?
  3. Lack of explanation for $r$ and $c$ in Equation (6, 7) 

* Missing important details:
  * How to collect the curated caption pool in Figure 4
  * The number of latent tokens is randomly sampled during training, how about evaluation?
  * Except for Context-Aware Alignment Decoding, the novelty of Text-Guided Vision Encoding and Text-Driven data synthesis is limited:
    * Text-Guided Vision Encoding is quite similar to Q-Former in InstructBLIP.
    * Utilizing synthesized images for VLM training is not a novel solution, for example, SynthVLM [1].
* There are several important points to address for the method:
  * From my perspective, data filtering is important for the synthesized data, which is not included in this paper
  * Most VLMs adopt different strategies to control trainable parameters in different stages, which is not considered in FLARE
  * How could the designed module be applied to more complex scenarios like videos, multi-images, and excessively long text contexts?
  * Fore FLARE-X, why not apply the dynamic-resolution strategy that is utilized in NaViT and Qwen2.5-VL, which seems more suitable for your method compared to the subfigure-based solution.
* Conducted experiments are limited, in terms of:
  * Lack of analysis on the quality of synthesized images
  * Experiments in Table 3 do not sound fair to me, the superior performance of FLARE may come from the SigLIP2, which is better than vision encoders used by other models.
  * Current analysis in the main paper focuses on quantitative performance on benchmarks, detailed analysis is expected to reveal how the proposed modules help vision-language interaction. 


[1] SynthVLM: Towards High-Quality and Efficient Synthesis of Image-Caption Datasets for Vision-Language Models (https://arxiv.org/abs/2407.20756)

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes FLARE, a system-level approach for fully integrating heterogeneous components (hardware accelerators, runtime scheduler, and compiler toolchain) to improve end-to-end performance and developer productivity for ML workloads. The authors present the system design, an implementation supporting several accelerators, and an evaluation on representative models and benchmarks showing improved throughput and reduced end-to-end latency compared to a baseline composed of independently developed components. The paper claims contributions in modular integration APIs, cross-layer optimizations, and an evaluation demonstrating practical gains on real workloads.

### Strengths
1. The paper identifies the practical pain points of mismatched interfaces across compiler, runtime, and accelerator stacks and motivates why full-stack integration yields benefits.

2. The modular architecture and defined integration API are well described and appear reusable for multiple accelerators.

3. A working prototype that integrates compiler passes, a runtime scheduler, and at least two accelerator backends strengthens the paper’s practicality.

4. Results include latency and throughput measurements on relevant models, showing consistent improvements over the baseline and an analysis of where gains arise (e.g., operator fusion, memory reuse).

5. The paper is mostly well organized with diagrams that clarify the cross-layer interactions and a reproducible experimental setup description.

### Weaknesses
1. The evaluation compares against a narrow set of baselines; it is unclear how FLARE performs versus other recent integrated stacks or optimized vendor toolchains.

2.  While gains are reported, the paper lacks deeper ablation isolating the contribution of each integration component (compiler vs runtime vs scheduler).

3.  The current implementation targets a limited set of accelerators and ML model classes; scalability to diverse models (e.g., large sparse models, non-neural workloads) is not demonstrated.

4.The paper does not fully quantify engineering or runtime overheads introduced by the integration (e.g., compile time, memory footprint, scheduler overhead under dynamic workloads).

5.Some implementation details and tuning knobs used in experiments are missing, which could hinder reproduction by other researchers.

### Questions
1.	Which specific baseline toolchains and vendor stacks were used for comparison, and why were these chosen over others?

2.	Can you provide an ablation study that separately measures the impact of compiler optimizations, runtime scheduling, and accelerator-specific backends?

3.	How does FLARE handle dynamic workloads with runtime shape changes or mixed-precision operations; are there cases where integration harms performance?

4.	What are the compile-time and memory overheads introduced by the integrated toolchain compared to the baseline? Please provide quantitative measurements.

5.	How difficult is it to add support for a new accelerator backend; what is the required engineering effort and the expected common failure modes?

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
4

### Summary
Vision-Language Models (VLMs) often rely on shallow fusion, aligning visual and textual features only during decoding, which limits accurate cross-modal understanding. This paper presents FLARE, a framework that integrates several techniques to achieve deeper and more dynamic vision-language interaction. FLARE introduces Text-Guided Vision Encoding to incorporate textual cues into visual feature extraction, a Dual-Semantic Mapping Loss for bidirectional modality-level alignment, and Context-Aware Alignment Decoding that aggregates visual information conditioned on text for fine-grained query-level fusion. Additionally, a Text-Driven VQA Synthesis method is employed to augment training data with synthetic question-answer pairs. Experiments across multiple benchmarks show that FLARE performs competitively with existing models while often using fewer visual tokens. The paper also presents ablation studies, fair comparisons under controlled settings, and a cost analysis showing the benefits of the proposed FLARE.

### Strengths
1. The key concept of achieving deeper cross-modal alignment through mutual reconstruction, mapping text to image and image to text, is intuitive and well-motivated. It aligns with the goal of reducing modality gaps and provides a more integrated framework than typical unidirectional projection methods.
2. The proposed data synthesis offers a practical way to expand multimodal training data using LLM-generated question-answer pairs. As shown in Table 6, this augmentation leads to measurable performance gains. The release of the code, data, and models could benefit broader vision-language research. 
3. The main experiments are complete and cover a wide range of multimodal benchmarks and baselines. The paper also includes ablation studies, fair comparisons, and cost analyses, which together strengthen the empirical evaluation. These analyses clarify the contribution of each module and demonstrate the efficiency of the model.
4. The paper is clearly written and well-structured. Figures, such as those showing cross-modal alignment and architecture flow (e.g., Figures 2 and 3), effectively communicate complex ideas.

### Weaknesses
1. As shown in Table 1, FLARE performs competitively but does not substantially outperform recent multimodal baselines of similar capacity. Claims such as "achieves leading performance across all benchmarks" are overstated given the mixed improvements and the complexity of the proposed pipeline. The cost of full training may not justify the small performance gains.
2. Some architectural decisions are only heuristically motivated. For example, it is proposed to mask visual-to-textual attention in the lower encoder layers. The paper asserts that early visual features lack semantics, but provides no ablation or analysis verifying that this masking improves stability or alignment.
3. The idea of reconstructing the representation of one modality from the other is conceptually reasonable and aligns with the goal of deeper cross-modal integration. However, in FLARE, the reconstruction is performed mainly using questions rather than captions. Reconstructing a question from an image may not yield semantically grounded alignment. In contrast, using captions or paired statements would provide a more consistent and general signal for vision-language correspondence, especially considering the proposed similarity loss.
4. The proposed data synthesis is basically an LLM-based data augmentation method, and its key implementation details are missing. The paper claims that only prompt engineering is required, which oversimplifies the complexity of generating consistent and high-quality image-text pairs. In practice, diffusion-based image generation often struggles with fine-grained accuracy (e.g., hands, text, and object details), yet these limitations are not discussed. In short, while the additional data may be valuable, the approach itself is not that novel and could introduce a potentially unfair advantage in comparison with the other existing models.

### Questions
1. Section 5.2 presents controlled comparisons using identical backbones and datasets, but it remains unclear whether the proposed Synthesis augmentation was applied in these experiments.
2. The paper highlights the efficiency in using fewer visual tokens. However, this design constraint also raises an important question: if the visual token limit were relaxed, could FLARE outperform existing systems?
3. Figure 1 is intended to visualize cross-modal alignment but is difficult to interpret without clear ground-truth references.
4. The training pipeline is described as stages 1, 1.5, and 2, rather than 1, 2, and 3. It would be helpful to explain the reason.

### Soundness
1

### Presentation
3

### Contribution
2
