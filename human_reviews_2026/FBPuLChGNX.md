# Learning to Generate Stylized Handwritten Text via a Unified Representation of Style, Content, and Noise

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Handwritten Text Generation (HTG) seeks to synthesize realistic and personalized handwriting by modeling stylistic and structural traits. While recent diffusion-based approaches have advanced generation fidelity, they typically rely on auxiliary style or content encoders with handcrafted objectives, leading to complex training pipelines and limited interaction across factors. In this work, we present InkSpire, a diffusion transformer based model that unifies style, content, and noise within a shared latent space. By eliminating explicit encoders, InkSpire streamlines optimization while enabling richer feature interaction and stronger in-context generation. To further enhance flexibility, we introduce a multi-line masked infilling strategy that allows training directly on raw text-line images, together with a revised positional encoding that supports arbitrary-length multi-line synthesis and fine-grained character editing. Moreover, InkSpire is trained on a bilingual Chinese–English corpus, enabling a single model to handle both Chinese and English handwriting generation with high fidelity and stylistic diversity, thereby overcoming the need for language-specific systems. Extensive experiments on IAM and ICDAR2013 demonstrate that InkSpire achieves superior structural accuracy and stylistic diversity compared to prior state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper designs a unified diffusion model for processing style, content, and noise for the task of handwritten text generation, eliminating the need for additional style or content encoders and achieving performance improvements.

### Strengths
1. The proposed InkSpire is trained on a Chinese-English mixed dataset and achieves bilingual handwriting generation under a single model.
2. Figure 2 clearly show the representation process of layout information.

### Weaknesses
1. Content and style references represent different types of information; I question the rationale behind encoding different information into the same latent space.

2. The description of the layout generation method is overly simplistic. I would like more information, such as model structure, loss functions, and training details.

3. In lines 242-244, the paper claims that the preprocessing operation of resizing text-line images is suboptimal, raising my concerns:

&emsp;&emsp; The authors argue that resizing text-line images overly shrinks characters in highly slanted lines. How does the proposed multi-line masked infilling strategy avoid this issue? When the proposed strategy faces multi-line images of inconsistent sizes, is resizing also necessary? Could this also lead to the same issues?

&emsp;&emsp; The authors believe that resizing text-line images introduces inconsistent distortions. Would this issue still arise if the resizing is proportional?

&emsp;&emsp; The authors mention that text-line images lack inter-line style. How is the inter-line style defined? In Figures 7, 8, the style references are single-line, and they may also lack inter-line style, yet the proposed InkSpire still achieves nice results. Please provide an explanation.

### Questions
1. What is the rationality of encoding different types of information such as content and style into the same latent space?
2. It is recommended to provide more details about the layout generation method in inkSpire.
3. What are the advantages of the proposed multi-line masked infilling strategy compared with text-line resize preprocessing?

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
5

### Summary
The paper introduces inkSpire for the handwritten text generation task, which encodes style, content, and noise into a unified latent space. It aims to enable the efficiency of information interaction within the common latent representation and improve generative performance. Experiments evaluate the proposed method.

### Strengths
1. The proposed Aligned Positional Encoding(APE) effectively distinguishes the style and content information in the visual input.
2. The proposed inSpire can simultaneously achieve paragraph generation and editing.

### Weaknesses
1. In lines 075-076, this paper claims that a diffusion model that uniformly handles style, content, and noise can achieve more effective information interactions in a common latent representation. This claim lacks some references or proofs , please provide a more detailed analysis and experimental verification.
2. Please provide more implementation details, such as which part of the FlUX parameters was fine-tuned by lora, the image size and batchsize used?
3. The ablation study of the proposed multi-line masked infilling strategy is lacking.
4. In the English comparison in Figure 6, "One-DM" and "diffusionPen" appear to be concatenations of words rather than direct generation of text-lines. I want to know how they are implemented. Is the comparison fair? Moreover, why is there no visualization of VATr++ shown?
5. There is a lack of failure case analysis and user study.

### Questions
1. What are the benefits of encoding style, content and noise into the same latent space? Please provide more explanations and proofs. Why does a unified representation achieve better performance than complex handcrafted losses?
2. How is the implementation of the comparison methods carried out? Is their comparison fair?
3. Please provide more necessary experimental analyses. For details, please refer to the weaknesses 3 and 5.

### Soundness
2

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
This paper presents InkSpire, a unified diffusion–transformer framework for generating personalized, multilingual handwriting. The model integrates layout prediction, glyph rendering, and image diffusion into one pipeline. Unlike previous methods that use separate encoders for style and content, InkSpire represents both in a shared latent space and trains on full-page, multi-line handwriting data. A conditional flow-matching (CFM) approach is introduced for layout generation, and a modified absolute positional encoding (APE / R-APE) is proposed to stabilize feature alignment. Experiments on IAM (English) and ICDAR2013/CASIA (Chinese) show improvements over prior handwriting synthesis systems in FID, KID, and content recognition metrics.

### Strengths
* Integrates layout prediction, style conditioning, and handwriting generation in a single diffusion transformer
* The editing and generation by same model is a Novel contribution in field of handwriting generation, thou such masking and generative techniques have been explored before in printed text documents like UDOP (Tang et al., 2023) and DiffUTE (Chen et al., 2023) and should be properly cited.
* Demonstrates bilingual (English–Chinese) generation using one model
• Qualitative results show realistic handwriting style imitation and fine-grained local editing capabilities.
• R-APE and LoRA fine-tuning on a frozen FLUX.1-Fill backbone allow good performance with modest compute

### Weaknesses
* The paper claims “strong style imitation” but provides no quantitative metric for writer identity preservation e.g., writer style identification accuracy as presented in WordStylist (Nikolaidou et al., 2023) and StylusAI (Riaz et al. 2024).
* All “style fidelity” evidence is qualitative and there is no test of whether generated handwriting is still identifiable as belonging to the same writer. This is important since during inference time specifically when using the model as a handwriting generator and not editor, the style sample present is just a single line style handwriting image and model is inferring style based on that.
* Smoke testing for out of distribution handwriting styles e.g. handwriting style of someone not in the IAM db database would also provide usefull insights.

* The lack of explicit writer-style metrics, undefined masking and noise procedures, and dataset-specific ambiguities weaken its reliability. The conceptual ideas (CFM for layout, unified latent fusion) are valuable, yet the paper would need a more transparent experimental protocol and stronger evaluation of stylistic consistency.

### Questions
* Table 1 reports layout-prediction L1 errors, but the paper never specifies how masking was performed during test-time prediction (e.g., partial vs. full masking, sampling procedure).
* It is unclear whether the reported errors correspond to full-layout generation (M = all) or masked reconstruction.This omission makes the results non-reproducible and the comparison between auto regressive, masked, and CFM models ambiguous.
* The layout generation could be the single source of failure in case of handwriting generation and transparency in its complete methodology is of significant importance for paper’s reproducibility.
* No details are provided on the masking ratio or noise schedule used during training for masked modeling or CFM. Without this, one cannot replicate the training regime or interpret robustness to partial conditioning
* By embedding style, content, and noise into a single latent, InkSpire sacrifices controllability: the ability to vary style independently of text content.

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
5

### Summary
This paper introduces InkSpire, a diffusion transformer model for stylized handwritten text generation, which unifies style, content, and noise within a shared latent space, thereby eliminating the need for auxiliary encoders. It utilizes a multi-line masked infilling strategy to train directly on raw text-line images and a revised positional encoding to support arbitrary-length multi-line synthesis and fine-grained character editing. Experiments conducted on bilingual Chinese-English datasets demonstrate that InkSpire achieves superior structural accuracy and stylistic diversity compared to prior state-of-the-art methods.

### Strengths
1. This paper introduces InkSpire, a new framework that reframes stylized handwritten text generation as a multi-line masked infilling task. This approach enables the model to unify style, content, and noise within a single shared latent space.

2. To support this unified framework, the paper proposes a crucial revised positional encoding scheme, Rotated Aligned Position Encoding (R-APE). This technique effectively solves the challenge of aligning tokens between the spatially concatenated style image and the standard-font content image.

3. The paper conducts extensive experiments on both English and Chinese datasets, demonstrating the effectiveness of the proposed method.

### Weaknesses
1. The paper rightly points out that prior methods for resizing single text lines to a fixed height introduce distortion. However, it is unclear how the proposed Multi-line Masked Infilling Strategy avoids a similar limitation when processing entire multi-line images of varying sizes.

2. This paper lacks a direct ablation study for the core architectural contribution—the unified, encoder-less framework itself.

3. The LoRA training details are incomplete, which hurts reproducibility. While the paper specifies the rank (32), it omits other critical hyperparameters, such as the LoRA alpha value, and fails to report the total number of trainable LoRA parameters.

4. The evaluation relies entirely on automated metrics. We suggest adding a user study to better validate the perceptual quality and stylistic fidelity of the proposed method. 

5. We suggest that authors provide visual examples of ablation study results.

### Questions
Please refer to Weaknesses 1-3.

### Soundness
3

### Presentation
2

### Contribution
2
