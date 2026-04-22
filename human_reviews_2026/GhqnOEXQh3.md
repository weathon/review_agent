# PosterCraft: Rethinking High-Quality Aesthetic Poster Generation in a Unified Framework

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Generating aesthetic posters is more challenging than simple design images: it requires not only precise text rendering but also the seamless integration of abstract artistic content, striking layouts, and overall stylistic harmony. To address this, we propose PosterCraft, a unified framework that abandons prior modular pipelines and rigid, predefined layouts, allowing the model to freely explore coherent, visually compelling compositions. PosterCraft employs a carefully designed, cascaded workflow to optimize the generation of high-aesthetic posters: (i) large-scale text-rendering optimization on our newly introduced Text-Render-2M dataset; (ii) region-aware supervised finetuning on HQ-Poster-100K; (iii) aesthetic-text reinforcement learning via best-of-n preference optimization; and (iv) joint vision–language feedback refinement. Each stage is supported by a fully automated data-construction pipeline tailored to its specific needs, enabling robust training without complex architectural modifications. Evaluated on multiple experiments, PosterCraft significantly outperforms open-source baselines in rendering accuracy, layout coherence, and overall visual appeal—approaching the quality of SOTA commercial systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposes an unified diffusion-based framework for high-quality aesthetic poster generation. Authors design an end-to-end optimization workflow to jointly learn text rendering, artistic layout, and stylistic coherence.

### Strengths
This work achieve visually coherent results by integrating text, layout, and artistic style in a single generative process.
Authors proposes a high-quality Text-Render-2M dataset.
PosterCraft outperform all open-source baselines (Flux1.dev, SD3.5, BAGEL, etc.) and nearly match the closed-source Gemini-2.0-Flash-Gen.

### Weaknesses
The motivation for adopting a unified framework appears questionable. Visual styles and textual elements often have fundamentally different requirements, visual styles typically emphasize global consistency, whereas text demands attention to fine-grained details and stroke-level accuracy.How does the proposed method reconcile these competing objectives?

The paper lacks sufficient comparison with recent academic works, such as Postermaker[1]. It is recommended that the authors provide a more detailed discussion of how their approach differs from or improves upon these recent methods to better highlight the novelty and contribution of this work.
[1] Postermaker: Towards high-quality product poster generation with accurate text rendering

While the dataset scale is impressive, the lack of public availability and insufficient statistical analysis limit its value to the research community.

### Questions
Generating highly realistic text in an end-to-end manner remains a significant challenge, as it often leads to structural distortions in character shapes. The results presented by the authors are surprisingly impressive in this regard. The authors’ results are impressive—how was this achieved in an end-to-end setup?

### Soundness
2

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
3

### Summary
This work proposes a full pipeline and corresponding datasets for achieving high-performing poster generation.

### Strengths
In general, the paper looks good to me.

1. It shows a good example of the full building pipeline of a high-quality domain-specific image generation system. Core procedures like data collection/curation, preference alignment, and reflection optimization are not only covered but accomplished at high quality.

2. The proposed datasets are very helpful to this field.

### Weaknesses
None

### Questions
I notice that in Tab.3 and Tab.4, PosterCraft, while demonstrating very competitive text rendering capability, still lags behind the topmost models (TextCrafter and X-Omni). I am fine with this gap, as text rendering is not the most important component of this work. But I am curious about how the authors would attribute this gap to. For example, is the gap due to base model limitation, data quantity/quality, or the absence of specially-designed algorithms? And as an extension, in the authors' opinion, what are the most important and promising directions to further enhance the model's text rendering capability?

### Soundness
4

### Presentation
4

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
The paper PosterCraft: Rethinking High-Quality Aesthetic Poster Generation in a Unified Framework presents an end-to-end system for generating visually compelling and textually accurate posters through a unified cascaded workflow. Instead of relying on modular pipelines or predefined layouts, PosterCraft integrates four stages—scalable text rendering optimization, high-quality poster fine-tuning, aesthetic–text reinforcement learning, and joint vision–language feedback refinement—each supported by automatically constructed datasets. This framework enables the model to synthesize posters with coherent layouts, stylistic harmony, and precise typography directly from textual prompts. Experiments demonstrate that PosterCraft substantially surpasses existing open-source baselines in both rendering fidelity and overall aesthetics, achieving performance close to leading commercial systems such as Gemini2.0-Flash-Gen.

### Strengths
1. The paper introduces four large-scale, automated datasets (Text-Render-2M, HQ-Poster-100K, etc.) tailored for specific training stages. This pipeline provides high-quality, specialized data for text rendering, style fine-tuning, and preference learning, addressing a major bottleneck in the field.

2. The paper proposes a unified framework that abandons rigid, modular pipelines where layout and text are generated separately. This approach allows the model to holistically explore coherent combinations of text, art, and layout in one process, resulting in more aesthetically harmonious compositions.

### Weaknesses
1. The claimed “unified generative framework” mainly integrates existing methods rather than introducing a fundamentally new generative modeling concept. Each stage—text rendering optimization, preference learning with DPO, and vision-language feedback—relies heavily on prior work. As a result, the contribution is more engineering-oriented than algorithmically innovative, making the paper better suited to an application or dataset construction area rather than the core generative models track.

2. Each stage requires a specialized dataset and specific training configurations (e.g., full-parameter finetuning, LoRA, DPO). This makes the end-to-end training pipeline exceedingly cumbersome and expensive. Consequently, the high complexity and cost create significant barriers to reproduction and practical deployment.

3. The authors claim the method is universal, but all experiments were conducted on a single base model, Flux-dev. Without experiments on other foundational models, it is impossible to know if this is a truly generalizable framework or just a deep over-fitting customized specifically for Flux-dev.

### Questions
1. Have the authors validated this workflow's generalizability by applying it to other foundational models (e.g., Stable Diffusion 3.5 or others)? Without such experiments, it is difficult to determine if the proposed pipeline is a truly generalizable framework or a series of optimizations highly specific to the Flux architecture.

### Soundness
2

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
PosterCraft is a framework for high-quality, aesthetic poster generation that moves beyond traditional modular pipelines and rigid layouts. The approach integrates four cascaded stages: large-scale text rendering optimization using the new Text-Render-2M dataset, region-aware supervised fine-tuning (HQ-Poster-100K), aesthetic-text reinforcement learning via best-of-n preference optimization, and joint vision–language feedback refinement. Each stage is supported by automated, stage-specific dataset construction, enabling robust training without complex architectural changes. PosterCraft outperforms some recent methods in text rendering, layout coherence, and visual appeal, as demonstrated by both quantitative metrics and user studies.

### Strengths
+ The proposed abandons modular and layout-constrained designs, enabling holistic integration of text, layout, and artistic content for visually coherent posters.

+ The work also introduces and leverages large, high-quality, and stage-specific datasets, supporting robust and scalable training.

+ The proposed method outperforms some recent methods in text accuracy, aesthetics, and prompt alignment.

### Weaknesses
- When talking about text rendering performance, it is usually important to measure text redner quality and accuracy under different length of words - from simple to complex, e.g. <20 words, 20-60 words, > 60 words. The current work lacks such kind of measurements making it hard to justfiy its strength especially for compelx cases. 

- Text rendering is an important area includes multiple areas including poster, infographic and scene text. It is not clear why the work only limited to the poster domain while not show generalization on others text rendering domains. 

- Authors claim state-of-the-art performance for the proposed method. However, many important methods are missing for the evaluation for both open-sourced and commercial ones, which makes it hard to support authors' claim or strength of the proposed method. 
Open-sourced:
Glyph-ByT5 [a] and BizGen [b], PosterMaker [c], as well as 

[a] Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering, ECCV 2024
[b] BizGen: Advancing Article-level Visual Text Rendering for Infographics Generation, CVPR 2025
[c] PosterMaker: Towards High-Quality Product Poster Generation with Accurate Text Rendering, CVPR 2025

 Commercial ones:
[A] recraft, [B] GPT-4o

### Questions
Please refer to the detailed questions raised in Weakness section above.

### Soundness
3

### Presentation
3

### Contribution
2
