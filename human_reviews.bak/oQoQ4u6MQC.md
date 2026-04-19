# DreamDistribution: Learning Prompt Distribution for Diverse In-distribution Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 5

## Abstract
The popularization of Text-to-Image (T2I) diffusion models enables the generation of high-quality images from text descriptions. However, generating diverse customized images with reference visual attributes remains challenging. This work focuses on personalizing T2I diffusion models at a more abstract concept or category level, adapting commonalities from a set of reference images while creating new instances with sufficient variations. We introduce a solution that allows a pretrained T2I diffusion model to learn a set of soft prompts, enabling the generation of novel images by sampling prompts from the learned distribution. These prompts offer text-guided editing capabilities and additional flexibility in controlling variation and mixing between multiple distributions. We also show the adaptability of the learned prompt distribution to other tasks, such as text-to-3D. Finally we demonstrate effectiveness of our approach through quantitative analysis including automatic evaluation and human assessment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a method for personalized image generation. Unlike prior approaches that rely on a fixed text embedding, this paper introduces a probabilistic formulation that samples from a prompt distribution, parameterized by a set of learned soft prompts. They construct a benchmark containing 12 visual concepts (e.g., instances of real objects and illustrations). They also provide a few qualitative examples on text-guided editing, composing prompt distributions, and text-to-3D. Finally, they use their method to generate a synthetic variant of ImageNet, where they demonstrate that a classifier trained on their synthetic images outperforms those produced by prompting or other personalization baselines.

### Strengths
- The method is novel and clearly explained; the method introduces a probabilistic formulation that helps with sample diversity. The paper is also careful about the scope of its claims; it studies “different instances in a same category set” not “same-instance personalization” (L357-358).
- The synthetic data generation experiment in Sec 4.5 is a nice metric. Existing personalization methods perform much worse than text-only prompting for producing synthetic data, but the proposed method is able to perform better.

### Weaknesses
- The paper could benefit from providing more details about Sec 4.5.
    - For each class, how many images was each personalization method trained on?
    - What is the distribution of noise seeds used to generate the images in Table 1 (e.g., is each synthetic image derived from an independent and completely random sample)?
    - It would be nice if the paper quantitatively verified the claim in L528 (other methods perform worse because they “show limited diversity”) with a diversity metric.
    - It would be nice if the paper provided a qualitative figure similar to Figure 3 for the ImageNet classes.
    - In Table 1, I would be curious to see a synthetic “upper bound” where one uniquely captions each real ImageNet image and generates images from this diverse pool of captions.
    - In Table 1, I would be curious to see a personalization baseline that introduced some randomness, e.g., textual inversion with Gaussian noise added to the learned text embedding.
- The paper could benefit from more clearly stating the types of visual concepts that are in scope for this work.
    - It would be nice if the investigation personalizing to ImageNet classes were extended to class hierarchies. For example, the paper could compare results when considering the class granularity to be “animal” vs. “dog” vs. “Rough Collie” vs. “Lassie”. Then it would be easier to see at what granularity the probabilistic formulation is vs. is not necessary.

### Questions
- Below are a few minor comments.
    - L375: missing space; should be “…text-editing **generation. We**…”
    - L413: should be “Text-**guided** Editing”
    -  L50-53 is very tangentially related to the rest of the paper; I think the first paragraph of the introduction could benefit more from the motivation and technical context of the paper.

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
This paper proposes distribution learning in the prompt space to increase the diversity for personalized generation problems.
The effectiveness of the proposed method has been proved on text-to-image, text-to-3D and synthetic dataset generation.

### Strengths
1. The writing of this paper is good and easy to follow.
2. The proposed soultion for increasing the generation diversity is straightforward and effective.
3. With application to different scenarios (text to image, text to 3D, and synthetic dataset generation), the authors provide different perspectives to demonstrate the effectiveness of the method.

### Weaknesses
1. The methods compared in this manuscript are no longer the state-of-the-art for personalized generation.  It would be better if the authors can compare with more recent and widely used methods, eg, IP-Adapter. 
2. The main contribution of this paper happens in the prompt space. However, I didn't see too much discussion and experiments that focus on the prompt space. For example, how does the orthogonal loss affect the generation quality and diversity? How different the K learnable prompts will be if the prompt distribution learning is not used?
3. I didn't see either qualitative or quantitative comparisons with our personalized text-to-3D methods, although the authors claim the diversity is better.

### Questions
Please refer to the weakness part.

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
This paper presents DreamDistribution, a solution for personalized text-to-image diffusion-based image generation. The key goal is to model the textual distribution of a training set, and guide the diffusion models to sample images that are adhere to the semantics of the reference images. Some learnable soft textual prompts and the reparameterization trick are introduced, enabling the generation of novel images by sampling prompts from the learned distribution. The authors also show the adaptability of the proposed DreamDistribution by applying it to various downstream tasks. Quantitative and qualitative analysis demonstrates the superiority of DreamDistribution against competing tuning based methods.

### Strengths
1. The writing of the paper is clear and easy to follow.
2. Extensive visual results are shown for intuitive comprehension of the proposed method.
3. The intuition presented in the paper is simple yet quite straightforward, supported by the comprehensive comparative experiments.
4. The proposed method only requires maintaining and updating a few learnable text prompts, which greatly preserves the image prior of the diffusion model.

### Weaknesses
1. The model relies heavily on the text-image alignment of diffusion, and since the diffusion model itself has not been tuned, I am curious about whether DreamDistribution can generate the new concepts if the diffusion model has not seen the reference images before.
2. I did not observe a significant visual improvement compared to textual inversion. And maybe the performance gap will become even smaller when stronger base models are utilized.
3. The novelty of proposed method is quite limited. Learnable tokens are very common in NLP and visual-language tasks, and the orthogonal loss is borrowed from some dictionary learning based methods, the reparameterization trick is borrowed from VAE training. These narrow the technical contribution of the proposed DreamDistrition.
4. DreamDistribution requires large amount of reference domain images to converge. I wonder how it performs when training samples are limited to 1~10, and how can it beats textual inversion in such scenarios.
5. The method is based on the assumption that the text distribution corresponding to a concept follows a Gaussian distribution. Is this a reasonable assumption? Also, does the number of learnable prompts required for different concepts need to be the same?

### Questions
Please refer to the Weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a method to learn personalized prompts that not only capture the common concept embedded in the given reference images but also reflect variations between them. Specifically, instead of learning just one soft prompt, multiple soft prompts are learned, and their mean and deviation are used to generate diverse input prompts. To ensure that each prompt learns significantly different values, an additional objective function that encourages orthogonality between the prompts was introduced during training. The authors demonstrate that this method successfully learns prompts that reflect diverse characteristics of given reference compared to previous works and suggest that it can be extended to 3D generation models.

### Strengths
- The explanation of the target problen, the method proposed to solve it, and the experimental setup are clear.
- The effectiveness of the proposed method is convincingly validated through various experimental results.

### Weaknesses
- The baseline methods do not share the target problem of this work. Models like DreamBooth or Custom Diffusion focus on learning only the common concept by eliminating variation, so they may not be fair comparison targets.
- In Figure 5-(a), when only the mean value is used, it seems that this model also fails to generate variable images. This suggests that simply learning one soft prompt and randomly perturbing it might suffice. However, without ablation studies, it is hard to be sure whether the proposed method itself is truly meaningful.
- In task like this, human evaluation metrics are particularly important. However, the result in Figure 4-(b) is based on evaluations from only 10 participants, which undermines reliability.

### Questions
- In models like Custom Diffusion, if various text prompts containing the learned special token are used, isn't it possible to generate diverse images?
- What are the results when only one learned soft prompt is used? Can we confirm what characteristics each soft prompt has learned?
- How the results change according the value of M?

### Soundness
2

### Presentation
3

### Contribution
2
