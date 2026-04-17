# R-Genie: Reasoning-Guided Generative Image Editing

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
While recent advances in image editing have enabled impressive synthesis capabilities, current methods remain constrained by explicit textual instructions and simple editing operations, lacking deep comprehension of implicit user intentions and contextual reasoning. In this work, we introduce a novel reasoning-guided generative image editing paradigm, which generates images based on complex, multi-faceted instructions accepting world knowledge and intention inference. To facilitate this paradigm, we first construct a comprehensive dataset featuring over 1,000 image-instruction-edit triples that incorporate rich reasoning contexts and world knowledge. We then propose R-Genie: a reasoning-guided generative image editor, which synergizes the generation power of diffusion models with advanced reasoning capabilities of multi-modal large language models. R-Genie leverages a reasoning-attention mechanism to bridge linguistic understanding with visual synthesis, enabling it to handle intricate editing requests involving abstract user intentions and contextual reasoning relations. Extensive experimental results validate that R-Genie can equip diffusion models with advanced reasoning-based editing capabilities, unlocking new potentials for intelligent image synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents R-Genie, a novel framework for image editing guided by reasoning over complex and implicit user instructions. To support evaluation, the authors introduce REditBench, a dataset of 1,070 image-instruction-edit triples spanning both atomic and composite tasks. The proposed architecture integrates multimodal large language models for instruction reasoning and diffusion models for precise image generation. The central innovation is a reasoning-attention mechanism, which links linguistic comprehension to visual synthesis through hierarchical and token-based modules.

### Strengths
1. The paper clearly identifies a meaningful gap in current image editing systems: their inability to interpret implicit, knowledge-dependent instructions. This reframing of image editing as a reasoning-aware task is timely and aligns well with trends toward more intelligent multimodal systems.
2. The introduction of REditBench fills a critical void. The dataset is thoughtfully constructed using RefCOCO for spatial grounding, augmented with human-annotated reasoning instructions and synthetic edits via SDXL. The inclusion of both atomic and composite edits enables a nuanced evaluation of reasoning capability.
3. The experiments demonstrate consistent superiority in both semantic alignment and visual fidelity. The failure mode analysis (e.g., target misidentification in baselines) convincingly illustrates the value of explicit reasoning.

### Weaknesses
1. While REditBench is well-constructed, 1,070 samples (850 train, 220 val) is extremely small for training or fine-tuning modern MLLMs, even with strong pretraining.  The claim that “the limited dataset suffices” (Sec 5.1) is plausible but not rigorously justified.  It raises concerns about generalization and overfitting, especially given the complexity of reasoning tasks.
2. Some compared methods (e.g., InstructPix2Pix, MagicBrush) were not designed for reasoning tasks, making the comparison somewhat unbalanced.  While this highlights R-Genie’s broader capability, it would be stronger to include reasoning-aware baselines like ReasonPix2Pix or ReasonBrain.
3. The masking scheme in Eq. (5) is unclear: are masks applied spatially or randomly?  How is the mask ratio chosen?

### Questions
Please refer to the weaknesses. If the issues I raised are addressed, I will increase my rating.

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
This paper introduces R-Genie, a novel paradigm for generative image editing that leverages multimodal large language models (MLLMs) for deep reasoning and intention inference, combined with diffusion models for high-fidelity image editing. The method is designed to address the limitations of current text-driven image editing approaches, which struggle with implicit, context-rich, or multi-step instructions requiring world knowledge and reasoning. The authors present (1) a comprehensive new dataset, REditBench, containing 1,070 image-instruction-edit triples emphasizing reasoning contexts, and (2) the R-Genie architecture, featuring a reasoning-attention mechanism and a hierarchical module that bridges semantic understanding and pixel-level synthesis. Quantitative and qualitative experiments show R-Genie outperforms state-of-the-art approaches in instruction-following image editing, particularly for complex or composite edits.

### Strengths
1. The core contribution is combining reasong capablities of MLLMs with diffusion models allowing complex editing requiring reasoning and world knowledge.
2. Dataset contribution, this paper proposed REditBench, offering a new dataset focused on reasoning intensive edits.
3. This paper introduces a hierarchical reasoning module and a reasoning-attention bridge to enable precise and rational instruction-based image editing.

### Weaknesses
1. REditBench is relatively small compared to typical image editing dataset.  It would be better to include more open-world images and instructions.
2. Limited instruction diversity, most examples shown in the paper focus on attribute reasong and compositional edits, e.g. replace, move.
3. Lack of Cross-dataset evaluation. All main results are built on REditBench, performance on other widely adopted editing dataset (e.g. GEdit and ImgEdit) would imporve external validity.

### Questions
1. After reading the paper, I think most of the examples shown in the paper can be addressed with the combination of VLM rewriting and instruction-based image editing, that is first rewriting the instruction with VLM then feed into instruction-based image editing. Compared to these methods, what is your advantage?
2. Has R-Genie been evaluated on established benchmark such as GEdit and ImgEdit.
3. How does it perform in complex editing scenarios, such as: Who is the oldest person in the picture? make him/her younger and give him/her long, black hair.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes R-Genie, which fuses editing instructions with visual features to better guide the image editing process. A new dataset, REditBench, derived from RefCOCO, is used for training and evaluating the proposed model.

### Strengths
The task that the paper addresses is meaningful as it enables reasoning-based, complex image editing.

### Weaknesses
1. Limited scale of the dataset. The paper proposes a new benchmark, REditBench, with 1070 images (850 / 220 train/val split). The training set is not sufficient for training, especially when using the contrastive objective in Eqn 6.

2. Limited evaluations. The authors only showed comparisons on REditBench, while leaving out more commonly used benchmarks such as MagicBrush, Emu, and SmartEdit. The proposed method likely overfits the proposed training dataset, resulting in better evaluation results.

3. Lacking newer baselines such as BAGEL [1], ICEdit [2].

4. Unclear details in the method description, which make the paper technically unreliable and hard to follow.

- Eqns 2 and 3 are essentially text-image and image-text cross-attention, according to Fig. 1.
- Inconsistent notations. Symbols such as $\mathbf{I}_\mathrm{global}$ are inconsistent with Fig. 1.
- The base model used is Show-o, which uses a discrete diffusion process to generate the image. The noising process is realized with a masking schedule for the tokens. What is the random noise in Fig.1 used for?
- How are visual tokens $\mathbf{I}$ input to the Phi-1.5 LLM? Do the authors follow the default settings of Show-o (VQ quantized tokens)? If so, it does not make sense to directly use them for the subsequent cross-attention modules.
- It is unknown how $h_\mathrm{reason}$ is derived. Is that the output for visual tokens after the LLM (as suggested by Fig. 1)?
- $h_\mathrm{answer}$ seems to be the token sequence for the output image. It makes no sense to use next token prediction in Eqn 5. For image tokens, masked token modeling loss is used for Show-o.
- Lots of self-contradictory info. For example, L332 $\alpha_t$ is time-dependent, but L343 - $\alpha_t$ is set to 0.5. 



[1] Deng, Chaorui, et al. "Emerging properties in unified multimodal pretraining." arXiv preprint arXiv:2505.14683 (2025).

[2] Zhang, Zechuan, et al. "In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer." arXiv preprint arXiv:2504.20690 (2025).

### Questions
Please refer to weakness point 4 for my questions.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces R-Genie, a reasoning-guided generative image editing framework that aims to handle implicit user intentions beyond explicit text prompts. The method integrates the reasoning ability of multimodal large language models (MLLMs) with diffusion-based image generation, featuring a hierarchical reasoning module and a reasoning-attention bridge for cross-modal alignment. The authors also construct a dataset, REditBench, containing 1,070 image–instruction–edit triples to benchmark reasoning-based editing. Experimental and user study results suggest that R-Genie achieves superior semantic consistency and editing precision compared to prior methods.

### Strengths
1. The paper presents a technically complete framework combining diffusion models and MLLMs, reflecting a solid understanding of both reasoning and generative modeling.

2. The introduction of a new dataset (REditBench) focusing on reasoning-based image edits is valuable for benchmarking future research in this area.

3. The experimental evaluation is comprehensive, including quantitative comparisons, ablation studies, and user studies, which enhance the credibility of the results.

### Weaknesses
1. Motivation is unconvincing. The paper argues that implicit user intentions should be inferred by the model, but it is unclear why this is necessary for image editing. In practice, users could simply provide explicit, straightforward editing instructions; forcing the model to infer “hidden” intentions may not be a meaningful or realistic goal. In addition, Figure 1 is not intuitive and lacks side-by-side comparisons with existing instruction formats, which would make the contribution clearer.

2. In Lines 71–72, the authors claim that “MLLMs struggle to accurately reason about implicit user intentions,” yet the proposed method itself still relies  on MLLMs for  this task. This raises a conceptual inconsistency that should be addressed.

3. Comparative methods are outdated. Most baselines are from 2024 or earlier; the paper should include comparisons with more recent 2025 models to convincingly demonstrate the claimed state-of-the-art performance.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
