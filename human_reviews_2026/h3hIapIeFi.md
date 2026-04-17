# K-Gen: Unlocking High-Resolution Data-Free Knowledge Distillation via Key Region Generation

- Decision: Reject
- Scores: 6, 4, 4, 4, 4

## Abstract
Data-Free Knowledge Distillation (DFKD) is an advanced technique that enables knowledge transfer from a teacher model to a student model without relying on original training data. While DFKD methods have achieved success on smaller datasets like CIFAR10 and CIFAR100, they encounter challenges on larger, high-resolution datasets such as ImageNet. A primary issue with previous approaches is their generation of synthetic images at high resolutions (e.g., $224 \times 224$) without leveraging information from real images, often resulting in noisy images that lack essential class-specific features in large datasets. Additionally, the computational cost of generating the extensive data needed for effective knowledge transfer can be prohibitive. In this paper, we introduce \underline{K}ey Region Data-free \underline{Gen}eration (K-Gen) to address these limitations. K-Gen generates only key region of images at lower resolutions while using class-activation score to ensure that the generated images retain critical, class-specific features. To further enhance model diversity, we propose multi-resolution generation and embedding diversity techniques that strengthen latent space representations, leading to significant performance improvements. Experimental results demonstrate that K-Gen achieves state-of-the-art performance across both small-, high- and mega-resolution datasets, with notable performance gains of up to two digits in nearly all ImageNet and subset experiments. Code is available at \url{https://anonymous.4open.science/r/K-Gen-DFKD}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors look into the problem of scaling data-free knowledge distillation methods to higher resolution. This is an important very important problem and has some obvious applications in medical fields. 

They propose to generate images at a much lower resolution, but will retain the class activations using CAM. However, training on low resolution images can lead to a poor performing model, so they propose a multi-resolution generation strategy and embedding diversity losses. These extensions are quite natural and are shown to perform well.

### Strengths
The authors perform an extensive evaluation of their method and show not only a significant improvement in model performance, but at a fraction of the training costs. The authors also provide code, which appears to be self-contained and organized.

The methodology is written well, the limitations of prior methods are discussed and their approach seems very natural and effective. The authors also show how to extend their method to vision transformers, which is good to see.

The ablation of hyper-parameters is generally quite thorough.

### Weaknesses
There are many losses (see equation 10) which naturally leads to many hyperparameters for tuning. Although the authors approach enables a significant drop in training time, the need for tuning hyper-parameters may offset this. I understand that αce, αbn, αadv just followed Tran et al, but this point is still a relevant weaknesses for applying K-Gen to new datasets. The sensitivity of αkr and αkr is quite small, which is good, so perhaps this is overall only a minor weakness. However, it would be good to see that these hyperparameters do generalise to data with a much bigger domain shift from imagenet and even better the domains/applications that motivate this task (e.g. medical images).

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

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
This paper presents a novel framework for data-free knowledge distillation (DFKD) that effectively scales to high-resolution datasets such as ImageNet. It generates lower-resolution synthetic images guided by Class Activation Maps (CAMs) to focus on class-discriminative regions, improving computational efficiency while preserving essential semantic information. The framework is further extended to Vision Transformer (ViT) architectures, showcasing its generality beyond convolutional networks. In addition, it introduces Multi-Resolution Data Generation to capture both coarse- and fine-grained features, along with an Embedding Diversity Loss that maintains distinct latent representations at lower resolutions, thereby enhancing feature diversity and overall performance.

### Strengths
The paper presents a well-engineered framework for data-free knowledge distillation (DFKD), addressing a critical bottleneck in scaling existing DFKD approaches to high-resolution datasets such as ImageNet. The paper includes a comprehensive set of experiments covering multiple datasets (CIFAR, ImageNet, and higher-resolution benchmarks) and architectures (CNNs and Vision Transformers). The ablation studies are also presented to illustrate the effects of the proposed components.

### Weaknesses
1. CAM (Class Activation Map) is already a well-established technique for visual localization, and the ideas of low-resolution generation, multi-resolution synthesis, and embedding diversity have been explored in prior generative and distillation works. The main contribution of K-Gen lies in integrating these existing components into a practical data-free distillation framework, resulting in limited conceptual novelty.
2. Although the experiments cover multiple datasets and architectures, the work would benefit from additional sensitivity studies, such as examining sensitivity to resolution settings, CAM quality, or embedding-diversity parameters, to better understand the robustness of the method. In particular, since CAMs are often noisy or inaccurate in localizing class-discriminative regions, an analysis of how this inaccuracy affects generation quality and distillation performance would strengthen the paper.
3. The paper shows limited visualization or discussion of what the generated “key-region” images actually look like, how realistic they are, or whether they truly capture semantically meaningful structures. This makes it difficult to interpret how the key-region generation contributes to knowledge transfer.
4. There appears to be an inconsistency in Table 2 where a student model achieve higher accuracy than their corresponding teacher network(V11 and R18). This seems unusual for a data-free distillation setup and should be clarified or corrected for consistency.

### Questions
The method relies on Class Activation Maps (CAMs) to identify class-discriminative regions. However, CAMs are known to be fragile and sometimes inaccurate, especially for images containing multiple objects or complex backgrounds. It would be helpful to include a sensitivity analysis showing how variations or inaccuracies in CAM quality affect the generated regions and the overall distillation performance.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of scaling DFKD to high-resolution images, where existing methods suffer from noisy synthetic samples and excessive computational cost. To this end, the authors propose K-Gen, a novel framework that synthesizes key regions at lower resolutions, guided by CAM to preserve essential class-specific features. Furthermore, the authors introduce multi-resolution data generation and an embedding diversity loss to maintain rich feature representations and enhance model robustness. Extensive experiments demonstrate that K-Gen achieves state-of-the-art performance with significant improvements in both accuracy and efficiency.

### Strengths
* This paper targets high-resolution image scenarios by proposing attribution-guided local generation and backpropagation, significantly reducing the computational cost while improving accuracy.
* The authors discuss the related literature in detail.
* The paper is easy to follow.
* The authors provide code for reproducibility checks.

### Weaknesses
1. The authors should include an detailed analysis of how different attribution methods affect the final performance. Given that CAM is applied only at a local layer to maintain training efficiency, does this design cause variations in the captured regions? The paper only provides a few qualitative visualizations, so what would happen if random regions were used instead of CAM-guided ones?
2. The proposed method contains many handcrafted hyperparameters, such as image scales and loss weights. How are these parameters selected, and how sensitive is the final performance to their variations?
3. This paper is poorly written, with incorrect citation formatting, figures and tables containing text that is too small to read, and excessive use of LLM-style dashes.
4. This paper does not discuss the limitations of the proposed method.

### Questions
My questions are in Weaknesses Section.

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
3

### Summary
- The draft introduces K-Gen to address the limitations of data-free knowledge distillation.
- Existing methods (according to the draft) primarily relied on low-resolution datasets such as CIFAR-10 and CIFAR-100, but extending these approaches to high-resolution datasets often results in noisy synthetic images.
- To overcome this, the draft proposes multiple components—particularly generating at low resolution using Class Activation Maps from the teacher model and ED losses—which help ensure that the generated images retain critical, class-specific features by focusing on the most informative pixels.
- Experimental results demonstrate that K-Gen achieves state-of-the-art performance across various datasets, with performance improvements reaching up to two digits in most ImageNet and subset experiments.

### Strengths
- Instead of generating images at the desired resolution, this draft introduces the KR loss, which trains the generator model to produce low-noise images by leveraging the KR loss (but, not sure how important the CAM maps from the teacher model are, though). The draft demonstrates that this, along with the diversity loss, results in substantial gains (efficiency and generalization).
 
- In addition, the authors propose the $\mathcal{L}_{aed}$ loss, which encourages the generator to produce images that the student model has not yet learned. By combining the KE loss with the ED loss at each training step, the generation model is guided to create more diverse images that remain unseen by the student model—without compromising image quality or pixel-level detail.

- The use of $\mathcal{L_{ed}}$  and $\mathcal{L_{aed}}$ together allows the student model to continuously learn from new examples, while $\mathcal{L_{aed}}$ explicitly drives the generator to produce more diverse, novel samples. 

- The proposed method generates synthetic images at multiple resolutions, which helps them to capture both coarse and fine-grained features. 

- Instead of comparing the number of generated images with other methods, they compare based on data memory size and computation size. This strategy is considered fair because generating a lower-resolution image (e.g., 112×112) requires approximately the same resources as a fraction of a higher-resolution image (e.g., one 224×224 image is equivalent to four 112×112 images). This approach enables training on larger numbers of samples within the same memory and computational budget, allowing the model to encounter a more diverse range of examples.

- The authors provide extensive ablation studies, analyzing how different parameter choices affect performance, how each component contributes to the final results, and how image resolution impacts the quality of synthetic data.

### Weaknesses
- It is not clear what the actual contribution of the Gaussian-like, central CAM map as the target is. How about a central but circular or a non-central but Gaussian-like target CAM map? It may be the resolution that is reducing the noisy/irrelevant content during the synthesis. Please refer to the "Questions" section of this review.

- [Minor] The presentation of the content needs to be improved. For instance, while synthesizing, Key regions may refer to essential parts of a larger (high-resolution) image, as opposed to synthesizing low-resolution images with less noise. Figure 1(b) adds to this confusion.

### Questions
- During the Student training, does the method work with images of different resolutions after rescaling them to a fixed resolution, or is the student trained over the images at their original resolution at which they are generated? Discuss/clarify this in the case of both CNNs (with GAP layer) and ViT variants.

- What is the motivation to push the student feature on the synthetic data $\hat{f}_\mathcal{S}(\hat{x})$ close to the text embedding of the label $f_y$? In other words, why should the student feature space align with the semantic space learned by the text encoder (e.g., LLM) used to represent the labels?

- The proposed KR loss appears to encourage the synthesis to happen in the central region of the synthetic image. Does the method impose the KR loss in both low- and high-resolution synthesis cases? What is the implication of this loss in the case of the high-resolution synthesis? 

- It is not clear how the proposed KR loss reduces the number of noisy pixels (claimed by the draft as a drawback of existing DFKD methods). It may be because the synthesis occurs at a low resolution, the less important (or discriminative) information present in the low-resolution synthetic images is smaller compared to synthesizing at high resolution. In other words, it may be the resolution that is responsible, rather than making the pattern central to the synthetic data by imposing the CAM target. How about a non-central, but constrained (e.g., Gaussian-like) target map? Will it also be equally effective? Have the authors conducted any experiments in this direction?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a bottleneck in Data-Free Knowledge Distillation (DFKD): its poor performance and high computational cost when applied to large, high-resolution datasets like ImageNet. The authors argue that prior methods fail because generating full-resolution (224X224) synthetic images without real data results in noisy, feature-poor samples.

The proposed method, K-Gen, tackles this by generating synthetic images at a *lower resolution*. The core innovation is a "Key Region Loss" guided by Class Activation Maps (CAM). This loss forces the generator to synthesize images that, despite their lower resolution, retain critical class-specific features by maximizing activation in salient areas.

The authors conduct extensive experiments showing that K-Gen achieves new state-of-the-art results on ImageNet, outperforming previous methods by a large, double-digit margin while drastically reducing training time. The method is also shown to be effective for ViT models and on mega-resolution datasets.

### Strengths
- The proposed method generated a focused, low-resolution "key region" is more effective and efficient than generating a full, noisy high-resolution image. The use of CAM to guide this low-resolution generation is a sound technical approach.
- The performance gains shown on ImageNet (Table 1) are highly significant, with "two-digit" improvements over prior SOTA (e.g., NAYER, DeepInv).
- Alongside accuracy, the method demonstrates massive computational savings (e.g., Figure 1d, 24.25% in 9 hours vs. DeepInv's 3.15% in 61.2 hours).
- The authors have rigorously validated their method with the ablation studies (Table 4).

### Weaknesses
- The Key Region Loss relies on a predefined target mask, M_target, which is defined as "a Gaussian centered on the image" (line 228). This seems like a strong prior that assumes the most salient object features are always central. While this simplification clearly works well for ImageNet (as shown in Appendix F), it may not generalize to datasets where objects are commonly off-center (e.g., in detection or complex scene datasets).
- The adaptation for ViT models is not as clearly integrated as the CNN approach. The paper states CAM "cannot be extracted" (line 240) and later suggests attention maps are a substitute (line 673). However, the primary method described in Section 3.3 and Appendix A.1 focuses on patch reduction (7X7) and center-biasing the position embedding. It is unclear if a "key region loss" (analogous to L_kr) is actually used for ViTs, or if the performance gain simply comes from the efficient patch reduction.

### Questions
- Could authors clarify how "key region" generation is enforced for ViT models? Is there a loss function analogous to L_kr that uses the [CLS] token's attention map (as hinted in Appendix A.1)? Or, does the ViT student train without a key region loss, relying only on the 7X7 patch reduction?
- Have you investigated the impact of the centered-Gaussian M_target? What is the performance on datasets where the object of interest is not typically centered? Does this prior harm generalization, or does it perhaps act as a beneficial regularizer for the generator by simplifying its task?
- The results (e.g., Table 10, 16) suggest an optimal lower resolution (e.g., 96X96$ or 128X128) exists, outperforming both smaller (64X64) and larger (224X224) synthetic images. Do you have a hypothesis on how to determine this optimal resolution? Do you expect it to vary based on the teacher architecture (e.g., its receptive field) or the dataset's characteristics?

### Soundness
3

### Presentation
3

### Contribution
2
