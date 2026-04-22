# Query-Kontext: An Unified Multimodal Model for Image Generation and Editing

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Unified Multimodal Models (UMMs) have demonstrated remarkable performance in text-to-image generation (T2I) and editing (TI2I), whether instantiated as assembled unified frameworks which couple powerful vision-language model (VLM) with diffusion-based generator, or as naive Unified Multimodal Models with an early fusion of understanding and generation modalities. We contend that in current unified frameworks, the crucial capability of multimodal generative reasoning which encompasses instruction understanding, grounding, and image referring for identity preservation and faithful reconstruction, is intrinsically entangled with high-fidelity synthesis. In this work, we introduce Query-Kontext, a novel approach that bridges the VLM and diffusion model via a multimodal “kontext” composed of semantic cuse and coarse-grained image conditions encoded from multimodal inputs. This design delegates the complex ability of multimodal generative reasoning to powerful VLM while reserving diffusion model’s role for high-quality visual synthesis. To achieve this, we propose a three-stage progressive training strategy. First, we connect the VLM to a lightweight diffusion head via multimodal kontext tokens to unleash the VLM’s generative reasoning ability. Second, we scale this head to a large, pre-trained diffusion model to enhance visual detail
and realism. Finally, We introduce a low-level image encoder to improve image fidelity and perform instruction tuning on downstream tasks. Furthermore, We build a comprehensive data pipeline integrating real, synthetic, and curated open-source datasets, covering diverse multimodal reference-to-image scenarios, including image generation, instruction-driven editing, customized generation, and multi-subject composition. Experiments show that our approach matches strong unified baselines and even outperforms task-specific state-of-the-art methods in several cases.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Query-Kontext is a unified multimodal model (UMM) for text-to-image generation (T2I) and editing (TI2I) that decouples multimodal generative reasoning in vision-language models (VLMs) from high-fidelity synthesis in diffusion models. It bridges them via "kontext" tokens encoding semantic cues and coarse image conditions from multimodal inputs. A three-stage progressive training strategy is introduced: (1) connecting VLM to a lightweight diffusion head using LoRA; (2) scaling to a large pre-trained diffusion model for enhanced realism; (3) adding a low-level image encoder for fidelity and instruction tuning. A comprehensive dataset integrates real, synthetic, and open-source data across diverse tasks. Experiments demonstrate it matches strong UMM baselines and outperforms task-specific SOTA methods in several cases.

### Strengths
* The paper introduces Query-Kontext, which innovatively bridges VLMs and diffusion models using "kontext" tokens that encode semantic cues and coarse image conditions. This decouples complex generative reasoning from high-fidelity synthesis, addressing limitations in existing unified frameworks where these capabilities are entangled.
* The proposed shifted 2D Rotary Position Embedding distinguishes source and reference images by offsetting coordinates, preventing confusion in multi-image tasks like composition and editing.
* Experiments demonstrate Query-Kontext matches strong unified baselines and outperforms task-specific SOTAs in several cases across T2I, TI2I, customization, and composition.

### Weaknesses
* The definition and role of kontext tokens are a little bit vague and hard to follow, like the fixed-length kontext tokens (Q) are central to bridging VLM and diffusion, yet the length K and how to encode "coarse image conditions" are not specified or ablated.
* The three-stage training and shifted RoPE are central to decoupling reasoning and synthesis, but no ablations validate their contributions—e.g., no experiments showing performance drop without stage 3's low-level encoder or without shifted RoPE in multi-subject tasks. This obscures why the design achieves stated goals.
* The data curation heavily skews toward in-house Chinese image-text pairs, compared to only 30M open-source English pairs. This increases my concern about the fairness in comparative evaluation and the soundness of empirical validity.

### Questions
1. Stage 3 introduces a low-level encoder for fine-grained cues. Is this a standard SD-VAE or a custom implementation? How does it interact with kontext tokens—does it override or complement them?
2. The shifted Rotary Position Embedding (sRoPE) is an interesting addition for handling multiple reference images and source images by shifting coordinates into positive/negative quadrants. How does this interact with varying image resolutions or aspect ratios?

### Soundness
3

### Presentation
2

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
This paper proposes Query-Kontext, a unified multimodal framework designed to decouple multimodal generative reasoning (handled by a Vision-Language Model, VLM) from high-fidelity visual synthesis (handled by a diffusion model).
The key innovation is the introduction of a multimodal “kontext” — a set of semantic and coarse-grained image tokens bridging the VLM and diffusion backbone.
A three-stage progressive training strategy is developed:
	1.	Stage 1: Fine-tune a VLM with LoRA adapters to generate kontext tokens and align with a lightweight diffusion head.
	2.	Stage 2: Scale to a large pre-trained diffusion transformer (MMDiT) and align kontext/text embeddings through text-to-image and reconstruction tasks.
	3.	Stage 3: Introduce a low-level image encoder for fine-grained fidelity and identity preservation, while keeping the VLM frozen.
A comprehensive multimodal dataset (≈200 M samples) mixing real, synthetic, and curated open-source sources is built, spanning text-to-image generation, instruction-based editing, customized subject generation, and multi-subject composition.
Experiments show that Query-Kontext achieves state-of-the-art or competitive performance on GenEval, GEdit-Bench, DreamBooth, and DreamBench, outperforming several unified or task-specific baselines such as BAGEL, Qwen-Image, and GPT-Image in multiple metrics.

### Strengths
1. Elegant decoupling of reasoning (VLM) and rendering (diffusion) via “kontext” tokens.
2. Extensive dataset curation unifying text-to-image, editing, and composition.
3. Strong empirical performance across multiple public benchmarks (GenEval 0.88; GEdit-Bench 7.66/7.65; DreamBooth DINO 0.786).
4. Comprehensive analysis including ablation, convergence, etc.

### Weaknesses
1. Although Query-Kontext presents a solid and industrial-grade unified multimodal system,  the architectural novelty of Query-Kontext is limited — aside from the sRoPE design, the framework closely mirrors existing unified models such as Qwen-Image in structure and modality bridging.
2. The experiments cover standard benchmarks but lack targeted evaluation on compositional reasoning or unseen-concept editing, which are the key scenarios where the claimed reasoning–rendering decoupling should show advantages.
3. The paper failed to present the improvements of the proposed framework compared with the inhouse MMDiT. 
4. The caption of table 2 should be on top of the table.

### Questions
See weeknesses

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Query-Kontext try to answer an critical question: can we decouple multimodal generative reasoning from high-fidelity visual synthesis, while still maintaining (or even improving) overall performance? To achieve this goal, the authors design an architecture that physically and functionally separates the reasoning and rendering processes. And, three-stage progressive training gradually enforces this separation.  After well trained on large-scale multi-modal datasets, it achieves competitive results compared to existing methods.

### Strengths
1. By applying clear modular decoupling, explicitly separates reasoning (semantic understanding, grounding) from rendering (fidelity and realism);
2. Achieves comparable results on both unified and task-specific models across T2I, editing, and personalization;

### Weaknesses
1. Query-Kontext shows that reasoning and synthesis can be effectively decoupled. But they don’t fully explain why this decoupling works, so the answer is practically convincing but theoretically shallow.
2. It is unclear the effects of shifted RoPE compared to the standard RoPE.
3. The core in-house dataset (e.g., 170M+ Chinese pairs, synthetic triplets) is not public, limiting reproducibility.
4. The connector module struggles with aligning large (10B) frozen diffusion models, forcing full fine-tuning — potentially undermining modularity claims.
5. Mostly quantitative; limited qualitative ablation or human preference studies.
6. There are format issues in the cited references.

### Questions
See the weaknesses.

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
1. Well-presented and easy to read; the overall format and organization are clean and consistent.
2. Tackles the unification of image understanding and image generation via a decoupled MLLM-driven conditioning interface plus a diffusion-based renderer.
3. Provides a clear dataset construction pipeline and reports concrete experimental conclusions.

### Strengths
1. Dataset construction, task curation, and staged training/alignment procedure are described clearly enough for reproduction.
2. Polished writing and typography; the method and training recipe are easy to follow.
3. Broad experimental coverage across T2I, editing, personalization, and multi-subject settings with sensible metrics and ablations.

### Weaknesses
1. Insufficient comparison with more recent unified models; subject-driven generation is benchmarked mainly against OmniGen, leaving uncertain where it stands versus newer UMM baselines.
2. The distinction from prior works that use an MLLM (or Q-Former–style bridges) as a multimodal instruction encoder is not articulated clearly.
3. Limited qualitative visualizations compared with other baselines.

### Questions
1. I can’t clearly tell how Query-Kontext differs from Q-Former. What advantages does Query-Kontext offer?
2. Overall, the design essentially replaces the LLM with an MLLM and appends a projector, with little architectural novelty. Moreover, although the paper claims a unified model, the evaluation on subject-driven generation compares only against OmniGen, lacking comparisons with more recent unified models.
3. Why is the text embedding concatenated again before being fed into the connector?
4. How exactly are the low-level features and the noise concatenated—what is the ordering?

### Soundness
3

### Presentation
3

### Contribution
2
