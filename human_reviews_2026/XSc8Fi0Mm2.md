# RetouchLLM: Training-free Code-based Image Retouching with Vision Language Models

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Image retouching not only enhances visual quality but also serves as a means of expressing personal preferences and emotions. However, existing learning-based approaches require large-scale paired data and operate as black boxes, making the retouching process opaque and limiting their adaptability to handle diverse, user- or image-specific adjustments. In this work, we propose \textit{RetouchLLM}, a training-free white-box image retouching system, which requires no training data and performs interpretable, code-based retouching directly on high-resolution images. Our framework progressively enhances the image in a manner similar to how humans perform multi-step retouching, allowing exploration of diverse adjustment paths. It comprises of two main modules: a visual critic that identifies differences between the input and reference images, and a code generator that produces executable codes. Experiments demonstrate that our approach generalizes well across diverse retouching styles, while natural language-based user interaction enables interpretable and controllable adjustments tailored to user intent.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a training-free workflow, RetouchLLM, for image retouching. RetouchLLM uses VLMs and LLMs to understand and suggest possible enhancement operations for the input images. Unlike training a deep neural network, It is training-free and white-box, no need for large-scale paired data. RetouchLLM achieves notable improvement over previous methods.

### Strengths
### Idea/Motivation
 - The core idea of RetouchLLM is to use powerful priors from LLMs/VLMs and their understanding ability to make the inference transparent. 
- Instead of using an expensive training pipeline and black-box neural networks, a white-box and code-based workflow is proposed.

### Method
- Two core components, a visual critc and a code generator are proposed. 
- The model uses an iterative manner with a CLIP scorer to make sure a stable convergence. 
- With a code-based editing, the resolution of the input images are not lost.


### Experimental results
- Although training-free, RetouchLLM outperforms or achieves competitive results against baselines on MIT-Adobe 5k and PPR10K, against many trained models. 
- The framework shows a good generalization ability with diverse styles and different LLMs/VLMs.

### Writing/Delivery
- The paper is easy-to-follow and overall well-structured.

### Weaknesses
### Idea/Motivation
- Although it is new to Image Retouching, using VLMs or LLMs for image editing is not very new. For example, Cropper [1] uses a similar idea for image cropping, with VLMs and introduces a training-free approach. 

### Method
- Iterative refinement seems not a novel approach, as it is also shown in the Cropper paper. 
- There is a compute overhead. RetouchLLM generates N candidates per iteration (up to T=10 in practice). Compared to other training-based methods, it might be more fair to report additional costs.

### Experimental results
- User study scale. The user study only investigates 25 participants. (L489)
- Few “negative examples.” The paper would be more comprehensive if it includes some failure analysis. 


### References
[1] Lee S H, Jiang J, Xu Y, et al. Cropper: Vision-Language Model for Image Cropping through In-Context Learning[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 30010-30019.

### Questions
Please kindly refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes RetouchLLM, a training-free, white-box, code-driven image retouching framework that operates directly on high-resolution images. The system runs an iterative loop: a VLM describes stylistic gaps between a source image and a small set of style references; an LLM converts that description into an executable Python retouching program that composes a small pool of interpretable filters (exposure, contrast, saturation, temperature, highlight, shadow, texture). The same framework also supports a user-interactive mode.

### Strengths
- **Clear and simple approach**
The pipeline (critic→code generator→execute→select) is simple, elegant, and easy to extend (new filters or rules without retraining). The executable code improves the interpretability & reproducibility of the retouching process.

- **Training-free approach**
The approach doesn't need paired data or finetuning, directly handling high-res inputs. This dramatically lowers adoption barriers for photographers and apps.

- The paper provides a novel practice that replaces latent black boxes with explicit programs. This practice could inspire follow-up work beyond retouching.

### Weaknesses
- **Limited Evaluation**
The comparison includes Z-STAR and two supervised white-box/fine-tuning baselines, RSFNet, PG-IA-NILUT. But classic white-box pipelines (e.g., Exposure, Harmonizer-style operators, Neural Color Operators, LUT-based methods, or rule-based/optimization approaches) are encouraged to be included, as well as additional comparison with other LLM-based methods, such as MonetGPT.

- **Limited Novelty**
The conceptual novelty is somewhat incremental relative to the growing body of LLM/VLM tool-use and program-synthesis frameworks. The paper’s novelty lies in demonstrating that this approach is competitive for high-res retouching and yields reusable presets, which is not a fundamentally new pipeline.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces a non-training-based image editing method. It evaluates the image based on LLM and iteratively updates the image by generating code. Eventually, it ensures that the image and the reference image have consistency across multiple dimensions. The main advantages of this solution are: (1) No training required, as it utilizes the evaluation ability of VLM for image similarity. (2) Interpretable, because the image adjustment is done through generated code, making it easy to observe. (3) Compatible with high-resolution. I am not familiar with this field, but I think the shortcomings seem to be straightforward as well. Compared with the end-to-end pair-based learning methods, this method may have poorer performance in fine-grained adjustments. The process of adjusting specific areas (for instance, perhaps some people would like certain areas such as the face or hands to have higher brightness and a smoother texture) through VLM is more indirect compared to the end-to-end approach.

### Strengths
- The organization of the paper is very good and it is easy to follow.
- The design of the selection score is novel.
- Compared to the end-to-end approach, this method can fully leverage the capabilities of VLM and has a strong zero-shot learning ability.
- The experimental results are conclusive and the outcome seems very promising.
- The visualization of the paper is done well, I enjoy reading it.

### Weaknesses
- I'm not familiar with this field, but I doubt the applicability of this method for some specific features of local image editing. Does VLM only have good adaptability in certain aspects such as overall contrast and brightness adjustments?
- The baselines seem to be too few. As a method that doesn't require training, I think having more baselines is necessary.
- The choice of filters seems to be limited. Is this limitation due to interpretive requirements? If we abandon interpretability, for instance by using the prompt learning approach, would there be better results?

### Questions
- How applicable is the model? Is it only applicable to certain types of image editing on a global scale? However, I am not familiar with this field and I am unsure if this task actually only considers modifications on a global scale.
- How was the baseline selected? I think this is too little. Papers related to LLM should ensure that the baseline is adequately defined.
- How sensitive is the prompt? Does the prompt for visual critic seem to have been designed manually?

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
The paper proposes RetouchLLM, a training-free framework for image retouching. It operates iteratively: a VLM (Visual Critic) identifies photometric differences between source and reference images, and an LLM (Code Generator) translates these differences into executable Python code using 7 predefined filters. A CLIP-based selection score guides the process by choosing the best candidate at each iteration. The approach aims to provide an interpretable and adaptable alternative to data-driven methods.

### Strengths
1.  **White-Box Interpretability:** The code-based approach provides full transparency. The generated Python programs are explicit, reproducible, and reusable as preset filters (Fig 4).
2.  **Training-Free Adaptability:** The system operates without training data and can adapt to new styles using very few reference images (M=5), addressing the data dependency of supervised approaches.
3.  **Robust Engineering:** The iterative design with multi-candidate generation (Sec 3.2) is a sound strategy to mitigate the inherent uncertainty in VLM assessments.

### Weaknesses
1.  **Severely Restricted Action Space:** The framework is limited to 7 global photometric operations (Sec 3.3). It fundamentally lacks the capability for spatially localized adjustments (e.g., masking, selective color grading, complex curves), which restricts the method's scope to basic global correction rather than comprehensive retouching.
2.  **Impractical Latency:** The iterative process (T=10) requires sequential inferences from large foundation models, resulting in high latency (≈2 minutes per image, App B.1). This is prohibitively slow for practical use compared to feed-forward models.
3.  **Constrained Heuristic Search, Not Planning:** The claim of "Photo adjustment planning" is overstated. The LLM is explicitly forced by the system prompt (App B.2) to follow a rigid, predefined adjustment order (Global -> Local -> Color/Texture). This artificial constraint reduces the process to a heuristic search rather than genuine planning and prevents the exploration of optimal sequences.
4.  **Fragile Style Generalization:** Performance is bottlenecked by the VLM's ability to perceive style across different content. The significant performance drop between the paired (PSNR 29.21) and the more realistic unpaired (PSNR 22.19) setups (Table 4) highlights this fragility.
5.  **Insufficient Evaluation Protocol:** Comparisons with supervised models involve fine-tuning them on only 5 examples. This only demonstrates superiority in an extreme few-shot scenario and is insufficient to evaluate competitiveness against properly trained models.

### Questions
1.  **Localized Adjustments:** A significant limitation is the framework's reliance on global filters, which precludes spatially localized edits. This inability to perform region-specific adjustments (e.g., via spatial masks or curves) is a severe drawback. The authors should discuss potential extensions to incorporate spatially-varying operators to address this limitation.
2.  **Optimization Constraints:** Why was the rigid adjustment sequence (Global -> Local -> Color) enforced via prompting (W3)? How does performance change if the LLM is allowed to freely sequence any filter at any iteration?
3.  **Latency:** Are there concrete strategies to reduce the inference time (≈2 minutes) to practical levels?

### Soundness
2

### Presentation
3

### Contribution
2
