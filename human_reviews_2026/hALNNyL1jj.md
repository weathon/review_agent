# Next Visual Granularity Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
We propose a novel approach to image generation by decomposing an image into a structured sequence, where each element in the sequence shares the same spatial resolution but differs in the number of unique tokens used, capturing different level of visual granularity.
Image generation is carried out through our newly introduced Next Visual Granularity (NVG) generation framework, which generates a visual granularity sequence beginning from an empty image and progressively refines it, from global layout to fine details, in a structured manner.
This iterative process encodes a hierarchical, layered representation that offers fine-grained control over the generation process across multiple granularity levels. We train a series of NVG models for class-conditional image generation on the ImageNet dataset and observe clear scaling behavior. Compared to the VAR series, NVG consistently outperforms it in terms of FID scores (3.30 $\rightarrow$ 3.03, 2.57 $\rightarrow$ 2.44, 2.09 $\rightarrow$ 2.06). We also conduct extensive analysis to showcase the capability and potential of the NVG framework. Our code and models are released at https://yikai-wang.github.io/nvg.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Next Visual Granularity Generation (NVG), a hierarchical autoregressive image generation paradigm that progressively increases the token number (the “visual granularity”) with spatial resolution fixed in multiple stages. Specifically, during the inference process of each stage, a structure map (granularity structure information) is predicted with a diffusion model and then AR model is leveraged to transfer the stage-specific image to a fake final image with this structure, producing new stage residual content. On class-conditional ImageNet, NVG shows consistent but modest FID/IS improvements over VAR-style next-resolution autoregression and presents appealing spatial controllability via structure map guidance.

### Strengths
1. Novel Concept. Using the diversity token and explicit structure map to decouple and capture the different-level visual granularity, achieving good explainational exploration and controllable generation for AR. 

2. Good Methodological Techniques. The integration of diffusion model and AR with structure-aware positional embedding for explicit structure map generation and granularity visual content producing is neat and well-motivated. 

3. Empirical Experiments. On ImageNet class-conditional generation, NVG exhibits clear scaling trends with model size. Visualization of the structure-map controllability demonstrates a practical interface for layout control.

### Weaknesses
1. Structure construction assumptions. In this paper, the structure map plays a core role during the generation process. Since the structure map is constructed with a hierarchical clustering algorithm as ground-truth labels, such construction method may misalign with semantic boundaries in some situations (e.g., crowded/occluded object scenes) and then leads to failure generation. Robustness to such cases is unclear. Additionally, how about the generation results with a noisy or bad structure map during inference? 


2. Limited improvement magnitude. Gains over strong VAR baselines are small (e.g., FID reductions on the order of ~0.2–0.3 absolute). The authors may further explore the advantages of structure-map controllability for other applications. 

3. System metrics are underreported. The two-stage pipeline (structure and content generation) and multi-stage decoding likely incur additional costs, but this paper lacks memory footprints and time cost curves.

### Questions
1. Sensitivity about stage number. How sensitive are results to the number of granularity stages? 

2. Fake Final Canvas Prediction. Why does the model not produce the residual directly (like a diffusion process that predicts denoising noise)?

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
The paper introduces Next Visual Granularity (NVG), a generative framework that models image synthesis as a coarse-to-fine process guided by hierarchical structure maps. It constructs these maps through a greedy clustering of token embeddings, progressively merging similar regions to capture visual granularity from details to global structure. NVG alternates between structure generation and content generation at each stage, offering interpretable and controllable image synthesis. Experiments on ImageNet show competitive or superior performance to VAR across FID, IS, and recall, demonstrating improved structure awareness and scalability.

### Strengths
1.	The proposed structure prediction follows an intuitive and interpretable coarse-to-fine process, aligning well with how humans perceive visual composition.
2.	The framework effectively separates structure and content generation, enabling controllable and interpretable image synthesis.
3.	The model demonstrates strong scalability and competitive performance across metrics such as FID and Inception Score compared to state-of-the-art baselines.

### Weaknesses
1.	Semantic Misalignment in Clustering
NVG constructs hierarchical structures via greedy clustering based on token embedding similarity.
However, embedding-space similarity does not always reflect semantic consistency; tokens with similar color or texture may be grouped together even if they belong to different objects, leading to inaccurate granularity segmentation.
2.	Increased Computational Cost
Compared with VAR, NVG requires two sequential generation steps, structure and content at each stage.
This design improves interpretability but likely increases inference time and computational complexity, offsetting VAR’s efficiency advantage over traditional AR and diffusion models. It would be helpful if the paper reported the actual inference-time difference between NVG and VAR at a similar number of generation steps.

### Questions
1.	Since NVG heavily relies on the accuracy of the structure maps at each stage, would errors in early-stage structure generation propagate and amplify through subsequent stages, leading to degraded image quality?
2.	In cases involving multiple objects within a scene, how effectively can the structure map distinguish between different foreground objects? Does the current clustering-based construction capture such object-level boundaries?
3.	Although Table 1 reports a similar number of generation steps as VAR, NVG introduces separate structure and content generation at each stage. Does the observed performance gain come at the cost of increased computational load or slower inference speed in practice?

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
4

### Summary
This work proposes a new scheme for image generation, namely next visual granularity generation (NVG). In each granularity, the image’s latent is represented by the different binarized tokens. Each digit of the binarized token represents the grouping information of the pixel under each granularity levels. During the generation, the model start with the most coarse granularity level, where all pixels are separated into two groups. After each prediction stage, the number of the groups is doubled. This process is repeated until that all pixels have a distinct group. The model is trained in two parts. One model predicts the structure and the other predicts the content. In each prediction stage, the model first predicts structure and then predicts the content of each pixel.

### Strengths
* The reviewer likes the general idea of generating image from coarse-to-fine structure and content.
* The model shows its scalability, indicating it can be a potential model for foundational generative models.
* By leveraging the structure from reference image, the proposed method can conduct guided generation from reference image.

### Weaknesses
My major concern are related to some detail in the method and figures, the current manuscript makes the reviewer being confused.
* From the Fig 2, it seems that the models (structure and content) are predicting the residual results (that will be aggregated to final result). However, in Fig 4, the output of the structure/content model are final structure/canvas (and ln 269 also mentioned that content generator directly generated final canvas). And the “Residual” is the difference between the final canvas and the input. This is quite confusing, why two models outputs final structure and canvas, and why the residual is computed after the final canvas is obtained. In addition, there are different colored lines used in the Fig 4, the reviewer would like author to provide clear instructions on what does these colors mean.
* From the ln 239, it seems the model is conducting a flow-matching-like prediction. the reviewer has two questions here:
    * why does the author adopt this type of input?
    * is this strategy applied on each stage? If it is, does it mean it takes a flow-matching-like prediction for each step?
* The reviewer is quite not understand the ln 263-265, “To train effectively, …, the model cannot be trained effectively”. What is the thing that hampers the model training?
* The reviewer does not find the candidate pools of the content tokens. Since ln 289 mentions that “forming a distribution over all possible token candidates”, the reviewer feels like there is a candidate pool like text generation. However, the image generation is usually a continuous task and the tokens are usually continuous. I wonder that are the tokens here. Also, in the ln 269-270, the author mentioned the x-f(x) is the quantized target. I wonder what quantization there is and why use the difference as target.

### Questions
Please see my weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces content and structure pairs to assist the generation task. In particular, the structure guidance is from the features, without another pretrained network. The structure guidance can improve the generation performance.

### Strengths
This work explores the structural information for autoregressive-based image generation. It generates a structural mask by a cluster-based algorithm, without any pretrained method. The experimental results verify its effectiveness.

### Weaknesses
- The structured conditional guidance appears to have limited novelty, as similar ideas have been widely explored in text-to-image generation, such as in ControlNet [A].
- There is no ablation study evaluating the effectiveness of the structure and content guidance. In addition, qualitative comparisons with state-of-the-art methods are missing, as are qualitative results for the ablation studies.

[A] Zhang, Lvmin, Anyi Rao, and Maneesh Agrawala. "Adding conditional control to text-to-image diffusion models." ICCV 2023.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
2
