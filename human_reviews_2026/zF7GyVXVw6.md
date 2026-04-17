# OmniText: A Training-Free Generalist for Controllable Text-Image Manipulation

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Recent advancements in diffusion-based text synthesis have demonstrated significant performance in inserting and editing text within images via inpainting. However, despite the potential of text inpainting methods, three key limitations hinder their applicability to broader Text Image Manipulation (TIM) tasks: (i) the inability to remove text, (ii) the lack of control over the style of rendered text, and (iii) a tendency to generate duplicated letters. To address these challenges, we propose OmniText, a training-free generalist capable of performing a wide range of TIM tasks. Specifically, we investigate two key properties of cross- and self-attention mechanisms to enable text removal and to provide control over both text styles and content. Our findings reveal that text removal can be achieved by applying self-attention inversion, which mitigates the model's tendency to focus on surrounding text, thus reducing text hallucinations. Additionally, we redistribute cross-attention, as increasing the probability of certain text tokens reduces text hallucination. For controllable inpainting, we introduce novel loss functions in a latent optimization framework: a cross-attention content loss to improve text rendering accuracy and a self-attention style loss to facilitate style customization. Furthermore, we present OmniText-Bench, a benchmark dataset for evaluating diverse TIM tasks. It includes input images, target text with masks, and style references, covering diverse applications such as text removal, rescaling, repositioning, and insertion and editing with various styles. Our OmniText framework is the first generalist method capable of performing diverse TIM tasks. It achieves state-of-the-art performance across multiple tasks and metrics compared to other text inpainting methods and is comparable with specialist methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes OmniText, a training-free generalist framework for Text Image Manipulation (TIM) tasks. The authors identify three key limitations in existing diffusion-based text inpainting methods: inability to remove text, lack of style control, and letter duplication issues. To address these challenges, OmniText leverages self-attention inversion to enable text removal by reducing text hallucinations, and redistributes cross-attention to control text content. For controllable text inpainting, the method introduces a cross-attention content loss and a self-attention style loss within a latent optimization framework to improve rendering accuracy and style customization. Additionally, the authors contribute OmniText-Bench, a benchmark dataset for evaluating diverse TIM tasks including text removal, rescaling, repositioning, and styled insertion/editing. The proposed method achieves state-of-the-art performance compared to other text inpainting approaches and demonstrates comparable results to specialist methods while maintaining generalist capabilities across multiple TIM tasks.

### Strengths
1. The paper addresses critical practical limitations in text image manipulation that have hindered the deployment of existing methods. The ability to remove text, control styles, and avoid letter duplication are essential capabilities for real-world applications, making this work highly relevant and impactful for the community.

2. Unlike previous approaches that rely on data-driven multi-task training, OmniText demonstrates strong innovation by exploiting the internal attention mechanisms of diffusion models. The self-attention inversion for text removal and cross-attention redistribution for content control represent elegant, principled solutions that leverage the model's intrinsic properties rather than brute-force data scaling.

3. OmniText-Bench provides a comprehensive evaluation framework covering diverse TIM tasks with input images, target text, masks, and style references. This benchmark will significantly benefit the community by enabling standardized evaluation and facilitating future research in this domain.

### Weaknesses
1.  As a plug-and-play method, the experiments are conducted solely on TextDiff-2. This raises concerns about the generalizability of the approach—does the mechanism heavily depend on the specific capabilities of the base model? Validation on other diffusion-based text generation models (e.g., different architectures or model scales) is necessary to demonstrate the method's robustness and broader applicability.

2. The paper lacks critical analysis of computational efficiency. Key questions remain unanswered: How much does the proposed framework increase inference time? What is the overhead of the latent optimization process and attention manipulations? For a practical method, understanding the efficiency-performance trade-off is essential.

3. The ablation experiments are fragmented—SAI and CAR modules are only ablated for text removal tasks, while the loss functions are only evaluated for editing tasks. The potential interactions between these components (e.g., how SAI/CAR affect controllable inpainting, or how the loss functions impact text removal) are not explored. A comprehensive ablation study examining all components across all tasks would provide better insights into each module's contribution and their synergistic effects.

### Questions
1. Trade-off in Table 5: The ablation results show different loss configurations exhibit trade-offs between style fidelity and rendering accuracy. Can the authors explain this phenomenon? Is this trade-off inherent to the method or can it be mitigated?

2. Multilingual Generalization: The results primarily focus on English text. Can the authors explore whether OmniText works effectively for Chinese or other non-Latin scripts? Are any modifications needed to handle different writing systems?

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
This paper introduces OmniText, a training-free framework for various Text Image Manipulation (TIM) tasks, positioning it as the first generalist method for text removal, editing, insertion, and style-based manipulation. The method builds on a pre-trained model TextDiff-2 and introduces novel attention manipulation strategies and loss functions to control text rendering without fine-tuning. The authors also present OmniText-Bench, a new dataset for evaluating diverse TIM tasks.

### Strengths
- The paper addresses the important goal of creating a unified, generalist model for diverse TIM tasks. Its training-free approach, which avoids costly retraining by manipulating a pre-trained model's internal states, is a significant practical advantage.

- The introduction of the OmniText-Bench dataset is a valuable contribution that addresses a clear gap in evaluation resources for complex, style-oriented TIM tasks.

### Weaknesses
- While the specific techniques are new, the high-level approach of using attention control and latent optimization for editing is established in the broader image editing field.

- The paper completely omits any discussion of computational cost. Methods involving per-image optimization are often significantly slower at inference. A runtime comparison against baselines is critical for understanding the practical trade-offs of the proposed framework.

- The method's generalizability is not fully validated, as it is only demonstrated on the TextDiff-2 backbone. Furthermore, the new benchmark, while useful, is relatively small, which may limit the statistical significance of the evaluations performed on it.

### Questions
- Can you provide an analysis of OmniText's inference time compared to baseline methods, especially given the latent optimization step?

- How sensitive is the output quality to the choice of loss weights and the number of optimization steps during controllable inpainting?

-  Have you attempted to apply the OmniText modules to other text-synthesis backbones to demonstrate the generalizability of your approach?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses limitations of diffusion-based text inpainting in Text Image Manipulation (TIM): inability to remove text, lack of text style control, and duplicated letter generation. It proposes OmniText, a training-free generalist for diverse TIM tasks, leveraging cross- and self-attention mechanisms. Self-attention inversion enables text removal by reducing text hallucinations, while cross-attention redistribution lowers hallucinations via adjusted text token probability. For controllable inpainting, latent optimization uses cross-attention content loss (improves text accuracy) and self-attention style loss (enables style customization). The study also introduces OmniText-Bench, a benchmark with input images, target text/masks, and style references for tasks like text removal and stylized editing. Experiments show OmniText, the first TIM generalist, achieves SOTA across tasks vs. text inpainting methods and matches specialists.

### Strengths
1) This is the first training-free generalist method enabling text insertion and editing, removal, repositioning, and rescaling, and enabling explicit control over style fidelity, text content, and text removal, which is more applicable in real world.
2) The OmniText-Bench is proposed for a mockup-based evaluation. It consists of 150 sets of input images, targets texts with masks, reference images, and ground-truth, and covers five distinct applications.
3) The qualitative and quantitative experimental results show the comparable performance with specialist models.

### Weaknesses
1) How about comparing with recent large models such as GPT-4o, Gemini 2.5 Flash Image, Qwen Image since they are also generalist?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes OmniText, a training-free generalist framework for diverse Text-Image Manipulation (TIM) tasks. It addresses the limitations of existing methods—task specialization, lack of style control, and character duplication—by manipulating self- and cross-attention mechanisms. Specifically, it introduces Self-Attention Inversion (SAI) and Cross-Attention Reassignment (CAR) for text removal, and a latent optimization framework with cross-attention content loss and self-attention style loss for controllable inpainting. Additionally, the paper presents OmniText-Bench, a benchmark dataset covering multiple TIM tasks to fill the evaluation gap. OmniText aims to unify various TIM tasks while maintaining competitive performance compared to specialist and generalist baselines.

### Strengths
1. **Novel Generalist Design for TIM Tasks**: OmniText is the first training-free framework to jointly support diverse TIM tasks (removal, editing, insertion, rescaling, repositioning, style-based manipulation), addressing the long-standing task specialization issue in existing works. Its modular design (text removal + controllable inpainting) enables flexible adaptation to different tasks without retraining, filling a critical gap in the TIM field.
2. **Well-Grounded Attention Manipulation**: The method’s core innovations—SAI and CAR for text removal, and attention-based loss functions for style/content control—are rooted in in-depth analysis of attention properties (self-attention for style, cross-attention for content). This mechanistic understanding ensures the design is not ad-hoc, and the clear mathematical formalization of attention manipulation and loss functions enhances the framework’s credibility.
3. **Comprehensive Ablation Studies Validate Component Efficacy**: Ablation experiments systematically verify the necessity of key components—SAI+CAR for text removal, and the combination of cross-attention content loss and self-attention style loss for controllable inpainting. These studies clarify how each module contributes to balancing task performance (e.g., reducing hallucinations in removal, balancing accuracy and style fidelity in editing), strengthening the persuasiveness of the framework.
4. **Valuable Benchmark for TIM Evaluation**: OmniText-Bench addresses the lack of unified evaluation benchmarks for multi-task TIM. It provides diverse samples (input images, masks, style references, ground-truth) covering practical scenarios, enabling fair comparison of generalist TIM methods and promoting further research in the field.
5. **Strong Compatibility and Practicality**: As a training-free framework, OmniText builds on the TextDiff-2 backbone and can be easily integrated into existing diffusion-based text synthesis pipelines without modifying model architectures or requiring additional training data. This low deployment cost makes it highly applicable to real-world scenarios like poster design and product packaging editing.

### Weaknesses
1. **Insufficient Qualitative Results for Edge Cases**: The paper provides qualitative results for typical TIM tasks but lacks side-by-side comparisons for edge cases, such as long-text manipulation (over 10 characters), text on complex textures (e.g., patterned fabrics), or low-resolution input images. This makes it difficult to evaluate the framework’s robustness in challenging practical scenarios.
2. **Vague Explanation of Hyperparameter Tuning**: OmniText introduces hyperparameters (e.g., weight coefficients for content/style loss) but does not explain how to tune them for different tasks or datasets. The lack of guidance on hyperparameter adaptation may hinder its usability, especially for users without expertise in TIM.

### Questions
Please refer to the detailed points I raised in the "Weakness" section and respond to each numbered item in your rebuttal with clarifications.

### Soundness
3

### Presentation
2

### Contribution
3
