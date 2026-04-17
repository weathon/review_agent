# E-CommerceVideo: A Benchmark and approach for E-Commerce Video Generation from product Images

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
We introduce the task of E-Commerce Video Generation, which aims to automatically produce dynamic product showcase videos from static product images and a text prompt describing the video's motion and background. Unlike general video generation, this task requires strict adherence to the product's identity and visual features. Even small distortions in color, texture, or logos are not acceptable for commercial use. Existing methods are not directly applicable, as single-image conditioning often leads to hallucinations of product details, while strict frame-based conditioning severely limits motion diversity.
To address this gap, we present E-CommerceVideo, a large-scale, multi-modal dataset curated from the Taobao product repository. This benchmark dataset comprises a massive collection of triplets: product images, corresponding textual descriptions, and high-quality videos. 
We further establish a simple yet efficient baseline method by adapting a pre-trained video generation model with a VAE-based spatial injection mechanism, which preserves product appearance while enabling motion synthesis. Comprehensive evaluation on VBench demonstrates the baseline’s potential as an entry point for subsequent advancements.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to address the task of creating dynamic e-commerce product showcase videos from static product images. To this end, the authors collected a E-CommerceVideo dataset, which consists of over 15K triplets of product images, text description and target showcase videos. The paper also present a simple baseline method that adapts a pre-trained video generation model for generating the showcase videos.

### Strengths
* The proposed E-CommerceVideo dataset, which consists of 15K triplets (product images, textual descriptions, and high-quality video demonstrations), provides a solid foundation for future research in the area of e-commerce video generation.

* The authors also present a simple baseline method that adapts a pre-trained video generation model with a VAE-based spatial injection mechanism to preserve the product’s appearance while generating the video motion. This approach may set a starting point for future improvements in this domain.

### Weaknesses
* Visualization of dataset samples is missing. The paper only present one data sample in Figure 1, which is insufficient for readers to get a sense of the data distribution. It would be greatly enhanced by including example images or videos from the proposed dataset. Visualizing these samples would help readers better understand the diversity and quality of the data, as well as the challenges posed by the dataset in terms of product representation and motion generation.

* While the E-CommerceVideo dataset is large, it is focused primarily on apparel products. This narrow scope may affect the generalizability of the proposed method to other product categories. Expanding the dataset to include more diverse product types (e.g., electronics, furniture, etc.) would make the benchmark more comprehensive and the method more applicable to a wider range of e-commerce scenarios.

* While the dataset might be valuable, the baseline method presented in the paper does not introduce novel techniques beyond what is already available in image-to-video (i2v) generation literature. The baseline’s use of a pre-trained video generation model with a spatial injection mechanism might be seen as a straight-forward approach, lacking more innovative advancements.

### Questions
See [weaknesses].

### Soundness
2

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
4

### Summary
This paper introduces E-CommerceVideo, a benchmark dataset and baseline method for generating product showcase videos from static product images and text prompts in the e-commerce domain. The authors collect 15,096 video-reference image-caption triplets from Taobao's product repository, with each sample containing 5 multi-view reference images, detailed textual descriptions, and high-quality demonstration videos. The proposed baseline adapts the pre-trained Wan2.2 14B text-to-video model by incorporating VAE-based spatial injection for multi-reference images and frequency-based rotational positional embeddings (RoPE) to distinguish reference tokens from video tokens. The method is evaluated on VBench, showing improvements over existing off-the-shelf video generation models in appearance quality, subject consistency, and motion smoothness. The work addresses a significant gap in domain-specific video generation by providing both the dedicated e-commerce video generation benchmark and a functional baseline approach.

### Strengths
1. The paper is generally well-written.
2. The paper identifies and formalizes an important real-world task that has clear commercial value. Unlike general video generation, the emphasis on strict product fidelity (colors, textures, logos) directly addresses e-commerce requirements.
3. The E-Commerce Video dataset fills a critical gap in the field. The systematic extraction pipeline using DWpose for keyframe selection and the multi-view reference-image design (5 perspectives per video) demonstrates thoughtful curation. 
4. The tail-aligned sampling strategy (Algorithm 1) and the adaptive reference image extraction mechanism with dynamic handling of low-quality frames show engineering rigor and practical considerations.

### Weaknesses
1. The dataset focuses exclusively on apparel with human models, which represents only a fraction of e-commerce. The reference image extraction pipeline (Section 3.2) fundamentally depends on human pose detection (DWpose), making it completely inapplicable to non-apparel products like electronics, furniture, or cosmetics that don't involve human demonstration. E-commerce encompasses diverse product categories with varied presentation styles, yet the proposed method only handles model-worn clothing.
2. Tables 2 and 3 lack essential information about metric units, scales, and value ranges. Without knowing whether scores are percentages, normalized values, or arbitrary scales, readers cannot properly interpret the results or assess whether reported improvements are meaningful. The paper should explicitly define the range and interpretation for each metric rather than solely relying on the VBench citation.
3. Experimental results don't support the paper's core claims about product fidelity and motion flexibility. No metrics directly measure logo accuracy, texture preservation, or color fidelity despite emphasizing these as critical requirements. Improvements are marginal (1-3 points), and Dynamic Degree barely increases (88.1→90.5), contradicting claims of enhanced motion diversity. The evaluation lacks targeted metrics to validate the stated contributions.
4. The paper would benefit from more comprehensive visual comparisons (especially in videos) to better illustrate the claimed improvements in product fidelity and motion flexibility.

### Questions
**Major Comments**:
1. The manuscript claims that E-CommerceVideo is the first benchmark for this task. However, methods such as AnimateAnyone (Hu, 2024) and I2VGen-XL (Zhang et al., 2023) are cited for reference-based image-to-video generation. It would be important to clarify whether these works are accompanied by datasets that contain reference images, videos, and captions. Furthermore, are there existing fashion try-on or human motion datasets with comparable triplet structures? If such datasets exist, a comparative analysis would strengthen the contribution's positioning. If no such datasets are available, this should be explicitly stated to support the novelty claim.
2. Given the potential bias of existing metrics, it is advisable to conduct a user study to more comprehensively validate the effectiveness of the proposed method.

**Minor Comments**:
1. It is recommended to include more videos in the supplementary material, both demonstrating the proposed method and comparing it with other state-of-the-art approaches. As this is a video generation task, a few screenshots in the PDF are insufficient to evaluate performance aspects such as spatiotemporal consistency.
2. In all tables, indicate with upward or downward arrows whether higher or lower metric values represent better performance.

If the authors can address the aforementioned concerns, I would be willing to increase my final score.

### Soundness
2

### Presentation
3

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
The authors propose E-CommoerceVideo, a multi-modal video dataset for e-commere video generation. Then, a baseline method is proposed to use multiple-references and textual prompt to generate a product-specific video. The authors propose a reference image selection pipeline to provide sufficient reference information. Experiments show, the proposed method can achieve promising results in e-commerce video generation.

### Strengths
1. The proposed E-commerceVideo dataset can be helpful in the application of video generation in that area.
2. The proposed dataset contains a large range of product types.

### Weaknesses
1. Attached video or a link to anonymous demo page is crucial for the evaluation of the performances. Readers can hardly tell whether the method in this paper is doing better.
2. Frechet video distance, and CLIP score are quantitative metrics that can be helpful to measure video quality. The authors can add a comparison on them.
3. The ablation experiment considering the choice of reference frames is missing. The readers can hardly tell whether the strategy is effective. 
4. The proposed method somewhat lacks novelty. The multiple reference image injection method is similar to IP-Adapter.

### Questions
Please see the weaknessses.

### Soundness
3

### Presentation
3

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
The paper targets the problem of the details alignment between the reference image and the generated video. Under the scenario of E-commerce, it is required that the generated videos fully express the details, such as the text and the logo of the reference images. The paper collects a dataset with 15096 videos from Taobao website and also proposes a baseline method. The proposed method is verified on the V-benchmark.

### Strengths
The targeting problem is very important: to make the video generation model be used in commercial scenarios, the preservation of IP and product-related details is very important. For most of the video generative models, this is hard to achieve. It is also true that the trade-off between the hallucination and the IP preservation is a challenging problem in the field.

### Weaknesses
1. The proposed metrics are not clearly proven to be able to measure the IP preservation issue. Actually, in the qualitative results as shown in Figure 5 in the paper, many of the details have been changed. For example, in the first row of the images, the texture of the cloth and the style of the button have been changed clearly.

2. The collected dataset size is relatively small.

### Questions
It could be more intuitive to see the impact of the method and the collected dataset if some of the video cases were presented.

### Soundness
3

### Presentation
3

### Contribution
2
