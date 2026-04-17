# MINT: Causally Tracing Information Fusion in Multimodal Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2, 2

## Abstract
Multimodal Large Language Models (MLLMs) have demonstrated impressive performance on tasks that involve understanding and integrating information across different modalities, particularly vision and language. Despite their effectiveness, the internal representations of these Vision Language Models (VLMs) remain poorly understood, making it difficult to interpret their predictions or identify the causes of common errors. A crucial step toward improved interpretability is understanding how visual and textual signals fuse within the language decoder of these models. This integration process is particularly important since failures to properly combine modalities frequently lead to errors such as object hallucinations and incorrect spatial descriptions. In this paper, we systematically investigate the internal mechanisms of multimodal fusion in three representative VLMs: LLaVA-1.5-7B, DeepSeek-VL2-Tiny, and Qwen2-VL-7B. We propose MINT (Multimodal INtervention Tracing), a method that builds on the principle of hidden state patching to create a causal map of multimodal processing by systematically intervening at each layer of the language decoder. From these maps, we identify a critical region we term the `fusion band'—the decisive window of layers where visual and linguistic signals are actively fused to guide the model's output. Our analysis reveals that the location and width of this band are not uniform across models; they highlight fundamental differences in their fusion mechanisms that directly correlate with a model's ability to resolve contradictions, ground language, and perform complex spatial reasoning. This causal mapping offers a diagnostic framework to explain common VLM failures. Finally, we validate the practical utility of MINT by demonstrating a surgical LoRA fine-tuning strategy that targets the identified fusion band, correcting hallucination failures with near-perfect efficiency using less than half the parameters of baseline approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates how visual and textual information are fused inside Multimodal Large Language Models (MLLMs) to better understand their internal reasoning and interpret their errors. The authors introduce MINT, a causal analysis method based on hidden-state patching that systematically intervenes in each decoder layer to trace where and how multimodal integration occurs. Applying MINT to LLaVA-1.5-7B, DeepSeek-VL2-Tiny, and Qwen2-VL-7B, the study identifies a “fusion band”—a specific range of layers where visual and textual signals interact most strongly. The paper finds that the position and width of this fusion band vary across models, reflecting distinct fusion mechanisms that correlate with capabilities in grounding, contradiction resolution, and spatial reasoning. Overall, the work provides a causal interpretability framework for diagnosing and comparing VLMs, offering insights for future model design and multimodal understanding.

### Strengths
1. The introduced probing method MINT, is systematic and causal method to trace multimodal fusion within VLMs, advancing beyond correlational analyses and offering a concrete tool for understanding internal model mechanisms.

2. By analyzing widely-used MLLMs, the study reveals model-specific fusion patterns, providing generalizable insights into how architectural differences affect grounding and reasoning capabilities.

3. The discovery offers a clear, interpretable indicator of where visual–textual integration occurs, enabling targeted diagnosis of common VLM failures (e.g., hallucination, spatial errors) and guiding future multimodal model design.

### Weaknesses
1. Lines 326–328 state that “answering yes can only stem from the visual semantic.” It would be more convincing to empirically verify that the those prompts containing <category name> do not contain textual token biases toward “yes.” In other words, the models should be tested to ensure they do not produce “yes” responses when the visual input is patched with the placeholders.

2. The description of “patching in a clean visual representation” in Section 5.4 is somewhat ambiguous. It would help to clarify whether this refers to the output of the multimodal projector or another specific representation. Clearer definition and justification are needed to support the validity of the intervention results.

3. The conclusion of a “fundamental representational failure” for DeepSeek-VL2 (Line 397) seems overstated. As discussed in Sections 5.2 and 5.3, DeepSeek-VL2 performs visual grounding in the later decoder layers, a behavior distinct from the other two models. Thus, manually patching visual tokens might introduce an excessively strong signal rather than revealing an inherent failure.

4. The analysis would be more convincing if extended to larger models. Observations drawn from smaller models may not generalize to deeper models, potentially limiting the robustness of the conclusions.

### Questions
1. Line 318-321 mentioned a blank image placeholder. What placeholder is it? A special token like <vision_pad> or other tokens.


Typos:

1. The  citation formatting is problematic through the whole paper. There should be a pair of brackets between your citations. Please check the latex grammar.

2. In Figure 8 (a), the quotation mark of ``late activation'' is mistaken.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on the interpretability of multimodal information fusion in visual language models (VLMs), proposes MINT method, constructs causal maps through hidden state patching technology, successfully identifies the key fusion region of "fusion band", and verifies the model specificity of the fusion mechanism based on three representative VLMs (LLaVA - 1.5-7 B, DeepSeek-VL2-Tiny, Qwen2-VL-7B). At the same time, it realizes the diagnosis of common faults such as spatial reasoning errors and negative understanding bias.

### Strengths
1. MINT first adapts the hidden state patching technology system in single-modal scenes to the language decoder of VLM, breaking through the limitation that traditional probe analysis (such as lightweight classifier probes) can only capture correlations, and directly locates the fusion level of visual-text information through causal intervention, filling the research gap of "multimodal fusion causal map"; the "fusion band" provides a unified index for quantifying the fusion mode of different VLMs, which has strong theoretical significance.
2. The model selection takes into account different architecture types (CLIP + Vicuna, SigLIP + MoE decoder, custom vision adapter + dedicated decoder), and the dataset covers spatial inference, NegBench, and MS COCO, which can comprehensively verify the performance of the fusion mechanism in different tasks. The evaluation indicators are clearly defined, and the appendix supplements the bootstrap statistical significance analysis to ensure the reliability of the results.

### Weaknesses
1. The paper finds significant differences in the position and width of the "fusion band" of different models (e.g. early wide fusion of Qwen2-VL, late decentralized fusion of LLaVA-1.5), but does not analyze in depth the direct relationship between architectural design and fusion mode. For example, how do architectural differences such as "visual adapter" of Qwen2-VL, "CLIP + Vicuna splicing architecture" of LLaVA-1.5, and "MoE decoder" of DeepSeek-VL2 specifically affect the timing and scope of fusion? Existing discussions only mention "architectural differences", and lack quantitative or qualitative relevance arguments.
2.  MINT is based on a single level of hidden state patching, but does not explain how to handle the impact of cross-level fusion interactions - if a layer of fusion relies on the modal representation of the preceding layer, will a single layer of patching underestimate or misjudge the fusion key layer?
3. Text grounding experiments show that text embeddings are weak and limited to early layers of visual information, and the model relies mainly on direct visual attention. But the paper does not further explore "whether enhancing text grounding can improve the fusion effect" - for example, by fine-tuning the text encoder to carry visual information in the first order, will it reduce the "fusion band" range or improve the fusion accuracy?

### Questions
This paper has made valuable contributions to the field of VLM multimodal fusion interpretability. The MINT method and the discovery of "fusion band" provide new tools and perspectives for understanding the fusion mechanism. If experiments and analysis can be supplemented to address the above shortcomings, and the generalization, depth and practical value of the conclusions can be further improved, this work will be more in line with the publication standards.

### Soundness
3

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
4

### Summary
Vision-Language Models (VLMs) integrate visual and textual information through complex internal mechanisms that remain poorly understood. This paper presents MINT (Multimodal INtervention Tracing), a causal framework for analyzing how and where information fusion occurs within multimodal decoders. Instead of relying on correlation-based probing, MINT performs representation patching, systematically swapping hidden states between different runs, to trace the causal influence of visual and textual inputs layer by layer. Through experiments on multiple VLMs, the authors identify characteristic fusion bands and model-specific fusion patterns, showing how visual information overrides language priors. The additional application shows that the framework also allows the diagnosis of failure cases.

### Strengths
1. The introduction of MINT moves beyond correlation-based probing toward causal tracing through hidden-state patching. The idea of swapping intermediate representations is simple and conceptually transparent. It makes sense that changes in output can be causally attributed to specific layers or modalities.
2. The study evaluates multiple major VLMs across benchmarks, helping with meaningful cross-model comparison. The identification of the proposed "fusion bands", and the detailed mapping of which layers contribute to visual grounding or fail, provide generalizable findings and valuable insights for targeted model improvement.

### Weaknesses
1. While the framework is clearly presented, its core technique, hidden-state patching, is not novel. The contribution primarily adapts this existing approach to the multimodal setting rather than introducing a fundamentally new causal mechanism. The study could be made more compelling by extending the intervention beyond image features to also include text patching, enabling a fuller analysis of bidirectional information flow between modalities.
2. The paper claims to present the first empirical map of the "fusion band", yet the term itself is newly introduced and lacks grounding in prior literature. Moreover, the evaluation framework relies entirely on in-house binary tasks and custom metrics (override accuracy, flip rate, and failure depth), which are defined within this paper. This makes it difficult to compare results or validate the claimed novelty against established evaluation standards.
3. All main experiments adopt a classification-style prompt, focusing on binary outputs rather than analyzing finer-grained probability shifts that could reveal more nuanced causal effects. This, in fact, again places this work among output-level analyses rather than deeper investigations into the internal latent representations of VLMs. Consequently, the experimental setup feels shallow and does not fully leverage existing multimodal benchmarks. The experiments are also incomplete and inconsistent. For instance, LLaVA is not included in all analyses. Also, there is no direct comparison with existing interpretability or causal probing methods, despite their discussion in the related work section.

### Questions
What are the new benefits MINT offers compared with earlier methods? The main concern is its lack of connection and comparison with past studies. The novelty, depth, and coverage are also limited. While minor, use abbreviations correctly. For example, the "Vision-Language Model (VLM)" repeats multiple times.

### Soundness
1

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
4

### Summary
This paper introduces MINT, a novel framework designed to construct a causal map of multimodal processing in vision-language models (VLMs) by leveraging hidden-state patching techniques. The authors conduct experiments to investigate the internal mechanisms of LLaVA-1.5-7B, DeepSeek-VL2-Tiny, and Qwen2-VL-7B. And the results reveal some insights on the 'fusion band' of VLMs.

### Strengths
- Clear and Well-Structured: The paper is well-organized, with detailed explanations of the preliminary, intuition, and methodology.

- Extensive Evaluations: Three representative VLMs are investigated with the proposed framework, accompanied by comprehensive analysis and discussion.

- Interesting Findings: The experiment results offer some findings on the multimodal processing of VLMs.

### Weaknesses
- My major concern is that the core techniques incorporated in MINT are based on the patching method introduced in the previous work, **Patchscopes**. While the derivation is clear and well-presented, it does not introduce fundamentally new concepts but rather applies existing methods in a different context.

- The models used in the experiments are somewhat outdated — all evaluated models were released over 12 months; a more comprehensive evaluation using such recent and stronger VLMs would strengthen the manuscript.

- While they find variation across models, it’s unclear why some models adopt early vs late fusion. Are these choices by design (architecture) or emergent? Are they correlated with performance tradeoffs?

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
1
