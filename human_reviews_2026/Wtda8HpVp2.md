# CreatiDesign: A Unified Multi-Conditional Diffusion Transformer for Creative Graphic Design

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Graphic design plays a vital role in visual communication across advertising, marketing, and multimedia entertainment. Prior work has explored automated graphic design generation using diffusion models, aiming to streamline creative workflows and democratize design capabilities. However, complex graphic design scenarios require accurately adhering to design intent specified by multiple heterogeneous user-provided elements (\eg images, layouts, and texts), which pose multi-condition control challenges for existing methods. Specifically, previous single-condition control models demonstrate effectiveness only within their specialized domains but fail to generalize to other conditions, while existing multi-condition methods often lack fine-grained control over each sub-condition and compromise overall compositional harmony. To address these limitations, we introduce CreatiDesign, a systematic solution for automated graphic design covering both model architecture and dataset construction. First, we design a unified multi-condition driven architecture that enables flexible and precise integration of heterogeneous design elements with minimal architectural modifications to the base diffusion model. Furthermore, to ensure that each condition precisely controls its designated image region and to avoid interference between conditions, we propose a multimodal attention mask mechanism. Additionally, we develop a fully automated pipeline for constructing graphic design datasets, and introduce a new dataset with 400K samples featuring multi-condition annotations, along with a comprehensive benchmark. Experimental results show that CreatiDesign outperforms existing models by a clear margin in faithfully adhering to user intent.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the graphic design generation task. A unified multi-condition driven architecture is proposed by integrating different user-provided elements with a multimodal attention mask mechanism. A new graphic design dataset with 400k samples featuring multi-condition annotations, along with a comprehensive benchmark. The experimental results demonstrate the effectiveness of the proposed method in adhering to user intent.

### Strengths
The proposed method that integrates multiple heterogeneous conditions in a unified manner for graphic design generation is reasonable and effective. The proposed large-scale, multi-element graphic design dataset could also be useful for future research in the community if released.

### Weaknesses
1. The current evaluations mainly compare the proposed method with existing MLLMs for controllable image generation. However, there is a domain gap between natural images and graphic designs. To make fair comparisons, existing multi-condition or single-condition graphic design generation works, e.g., [1-4], should also be discussed and compared.
2. The user study may not sufficiently demonstrate the practical effectiveness of the proposed method. The motivation of this paper is to ensure the strict adherence between the generated design and the user intent. Thus, the user study should investigate from the above criteria and ask participants to create inputs by themselves, rather than predefining inputs and following similar evaluation metrics used in quantitative comparisons.
3. Since the proposed dataset is also the contribution of this paper, the ablated versions of the automated pipeline for graphic design dataset construction should also be studied. For example, using different methods or changing the module order for the data construction process.

[1] Hsu, HsiaoYuan, and Yuxin Peng. "PosterO: Structuring Layout Trees to Enable Language Models in Generalized Content-Aware Layout Generation." CVPR 2025.

[2] Horita, Daichi, et al. "Retrieval-augmented layout transformer for content-aware layout generation." CVPR 2024.

[3] Seol, Jaejung, Seojun Kim, and Jaejun Yoo. "Posterllama: Bridging design ability of language model to content-aware layout generation." ECCV 2024.

[4] Yang, Tao, et al. "Posterllava: Constructing a unified multi-modal layout generator with llm." arXiv preprint arXiv:2406.02884.

### Questions
1. For entity annotation, take Fig.2 as an example, how to separate different entities as primary or secondary visual elements? Will the generated graphic design be affected by considering different entities as primary or secondary visual elements?
2. The proposed dataset is constructed via a fully automatic pipeline. Several existing models are used for design theme generation, text layer rendering, and foreground-based image generation. How to ensure the quality of the constructed dataset without involving human annotations or double-checking?

### Soundness
2

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
4

### Summary
This paper presents a multi-conditioned creative graphic design method, where the conditions include semantic descriptions, positions and b-boxes of objects and texts to be placed on the image. The method encodes each condition into tokens and then employ a  multimodal attention mask to prevent the attention procedure from information confusion or leaking. Experimental results demonstrates the designed posters are in high quality.

### Strengths
1.  The proposed multimodal attention mask is effective in control the behavior of the diffusion model as verified in the ablation study. Both layout attention mask and subject attention mask are important to the quality of the graphical design. 
2. The diffusion model trained with multi-condition dataset can automatically support editing by using the image generated in the previous steps as the condition. While the STOA T-2-I models can support editing to some extent, the controllability of the proposed method should be better by associated bounding box in the graphic design.

### Weaknesses
The overall pipeline is built upon masked attention mechanism. However, masked attention mechanism is widely used in model design. For example, in NLP, casual attention is often used to avoid a token taking information from future tokens. From the perspective of novelty,  I hesitate to give this paper a high score.  In addition, the advantage of the proposed method are limited to the graphical design due to the specific definition of input conditions.

### Questions
It is tedious for a designer to determine the position of bbox of each text in a poster,  why not directly predict the position of these boxes as did in recent CVPR papers, for instance, Unsupervised domain adaption with pixel-level discriminator for image-aware layout generation in CVPR. Overall, this paper leans towards a graphic-design rendering paper more than a design paper, since the layout should be input by a user.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CreatiDesign, a multi-condition diffusion model designed for automated graphic design generation. ​ It integrates flexible multimodal inputs, including global prompts, rendered text, foreground visuals, background elements, and their layout positions. ​ A masked attention mechanism ensures precise control, aligning layout tokens with corresponding visual elements while preventing interference. 
The authors developed a new dataset with 400K synthetically generated samples and a benchmark of 1,000 annotated samples with ground truth labels. CreatiDesign is shown to outperform existing models in various design metrics.

### Strengths
+ a multimodal controllable generation model developed based on Flex T2I model
+ block-wise multi-modality attention achieves both controllability and prevents information leakage
+ a new benchmark dataset with curated high quality 1k set

### Weaknesses
- DiT framework is similar to previous work: Art: Anonymous region transformer for variable multi-layer transparent image generation. In CVPR, 2025.
- controllability is limited, only with given images and precise layout condition, text layer is not vector font. It would be useful to allow flexible layout position in output.
- the design images in the benchmark is synthetically constructed, the quality is not guaranteed.

### Questions
Can this method be extended to support placing given visual elements in appropriate locations in the design? This is similar to traditional layout generation/variation problem. The visual knowledge can give an edge over geometry based methods.

### Soundness
3

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
This paper presents CreatiDesign, a unified multi-conditional diffusion transformer for graphic design generation. CreatiDesign can receive inputs from different modalities: keyword, layout, prompt, image, etc. To support the task setting, the authors develop an automated pipeline for constructing a 400K graphic design dataset with multi-condition annotations. Using the dataset, they fine-tune the FLUX.1-dev model, where they modify the original attention strategy to enable multi-condition interaction. Experimental results demonstrate CreatiDesign's superior performance over existing methods in design generation.

### Strengths
- The paper tackles a real-world problem with clear applications in graphic design creation. 
- It contributes a large-scale dataset with multi-condition annotations. 
- The curated benchmark can well evaluate the design quality by including multiple metrics covering different aspects.
- The experimental results show clear improvements of CreatiDesign over baselines.

### Weaknesses
1. The technical contribution of CreatiDesign is limited. The used attention masking, LoRA fine-tuning are straightforward applications of existing techniques.
2. The model setting requires users to specify exact positions for the elements, making final quality heavily dependent on user design expertise. However, most users may lack such knowledge, thereby limiting its practical applicability. 
3. In the experiments, it seems that the paper only evaluates scenarios where all conditions are simultaneously provided. It would be great if the authors could conduct more experiments using the same model: (1) text-to-design generation (intention-to-design without visual inputs), (2) content-aware text layout generation (placing text on given background images), etc. Evaluating diverse conditioning inputs would demonstrate practical utility across different design workflows and strengthen the effectiveness of CreatiDesign.
4. The dataset details are not included in the paper, such as the element number, prompt length, diversity, quality, etc.
5. How does CreatiDesign's inference time compare to other methods?

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
