# FUSION IS ALL YOU NEED : FACE FUSION FOR CUSTOMIZED IDENTITY-PRESERVING IMAGE SYNTHESIS

- Decision: Reject
- Scores: 3, 6, 3, 3

## Abstract
Text-to-image (T2I) models have significantly advanced the development of artificial intelligence, enabling the generation of high-quality images in diverse contexts based on specific text prompts. However, existing T2I-based methods often struggle to accurately reproduce the appearance of individuals from a reference image and to create novel representations of those individuals in various settings. To address this, we leverage the pre-trained UNet from Stable Diffusion to incorporate the target face image directly into the generation process. Our approach diverges from prior methods that depend on fixed encoders or static face embeddings, which often fail to bridge encoding gaps. Instead, we capitalize on UNet’s sophisticated encoding capabilities to process reference images across multiple scales. By innovatively altering the cross-attention layers of the UNet, we effectively fuse individual identities into the generative process. This strategic integration of facial features across various scales not only enhances the robustness and consistency of the generated images but also facilitates efficient multi-reference and multi-identity generation. Our method sets a new benchmark in identity-preserving image generation, delivering state-of-the-art results in similarity metrics while maintaining prompt alignment.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper is dedicated to personalized text-to-image generation. Unlike previous works, this approach not uses frozen face encoder and static facial embedding to guide the generation. Instead, they utilize the hidden states within the pretrained UNet from reference facial image to replace static facial embedding, furthermore, they choose to concatenate K, V from text and facial image.

### Strengths
- Personalized generation is an important task in the field of images and has not been well solved.
- It reveals a good balance between face fidelity and prompt following ability.
- It is interesting to use UNet to handle reference image instead of an extra image encoder.
- This method is simple but achieves a new SOTA in identity-preserving image generation.

### Weaknesses
- The novelty is kind of limited.
- Although it may be a problem of base model (SD1.5), it is hard to claim that the facial fidelity is satisfied. This is my major concern.
- The experimental results are not convincing enough, the diversity may be a big problem.

### Questions
- What is the attention mask in formula (4)? It is not explained here how this mask is constructed and what its role is.
- In Figure (3), the improvement brought by more reference images is not obvious.
- In this experiment, the model was trained using only 80K images, and its diversity is questionable. It should be fair to use more non-celebrity (different gender, age, ethnicity) in testing.
- In Figure (5), the generated image has a copy-paste problem. Why does this happen?
- The credibility of the quantitative results is questionable. It is hard to believe that InstantID's similarity is worse, although it does suffer from text controllability.
- Why intermediate hidden state is a better representation of face information? More discussion will be useful.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a novel text-to-image (T2I) synthesis method leveraging a modified UNet from Stable Diffusion to improve identity preservation in generated images. By altering UNet's cross-attention layers, it achieves robust multi-identity and multi-reference generation, setting new benchmarks in identity similarity metrics.

### Strengths
The paper presents a novel text-to-image synthesis method that excels in several key areas. By directly incorporating the target face image into the diffusion process, it significantly enhances the identity preservation capabilities of T2I models. This method consistently generates high-quality, realistic images that accurately maintain the fidelity of facial features and expressions aligned with text prompts. It also offers remarkable flexibility and scalability, supporting complex image generation scenarios involving multiple identities and reference images. Additionally, the method is designed for computational efficiency, requiring less training time and fewer resources, which makes it highly practical for real-world applications.

### Weaknesses
1. The method can struggle with fine facial features, especially when the face occupies a small portion of the image due to the limitations of the underlying Stable Diffusion model.

2. Potential Overfitting: The method's reliance on direct face image integration might lead to overfitting specific facial features or identities, especially in a dataset with limited diversity.

3.Could we possibly tackle the issue of facial feature degradation by improving the core architecture of Stable Diffusion or ControlNet itself, instead of just relying on the fine-tuning or adjusting the condition strength mentioned in the article?

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes an identity-preserving image synthesis approach that directly passes the reference identity image to the diffusion model and fuses the attention maps obtained by a new cross-attention module with the original text-guided ones. The authors provide qualitative and quantitative results to demonstrate the effectiveness of the method.

### Strengths
The authors propose to pass the reference image to the diffusion model directly to avoid using an extra image encoder and potential encoding gap issues. The method can generalize to multiple reference images of the same identity and multiple identities in the same output image.

### Weaknesses
- **Incomplete experiments**: 
  - The paper ensembles a very similar approach to IP-Adapter (referred to as IPA-FaceID-Plus in paper) with two major differences: (1) no extra image encoder, and (2) cross-attention fusion at the attention mask stage. Both are claimed by the authors to improve the performance over previous approaches. However, the authors fail to provide an ablation study for the audience to know the exact effect of these two modifications and how much they contribute to the final performance.
  - Lack of quantitative comparison with important baseline InstantID. The authors claim they use SDXL which is a superior base model over the one they use, therefore eliminated for fair comparison. But for an important baseline, the authors should either upgrade the proposed method to SDXL or downgrade InstantID to the same base model to make the comparison.

- **Unsatisfying Quality**:
Qualitative results do not show clear advances compared to previous methods. Fig.3 in particular gives very bad-quality results. Besides, Fig.5 shows signs of overfitting the reference pose and facial expression in the generated results, which raises the concern that the improved ID preservation results (in Tab.1) and improved PSNR and SSIM (in Tab.2) may be caused by this overfitting. Fig.6 does not reveal clear improvements over IPA-FaceID-Plus and InstantID also.

### Questions
The main concerns about the paper are listed in the Weakness section. Here are some minor questions for the authors regarding unclear writing etc.:

- Equation 4: What does Q' represent? Q value should not be modified in the proposed method based on my understanding. Also, what does M here represent? Please provide the complete formula and how it differs from the standard attention mechanism for clear understanding.
- Line 304: If you mention this technique can be used for face morphing, it is better to provide several examples to illustrate.
- Figure 4: This figure appears to be unclear. Which identities are used for each output? Why do some outputs contain three identities and some two (are those with two missing one reference or do they only get two references as input)? What depth map is used for each output?
- Section 4.1: How long does it take for training? How large is the larger dataset used to train the second version?
- Line 376: How are the face regions obtained?
- Line 427: Fig.6 has nothing to do with emotion. I guess you mean Fig.1?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper improves text-to-image generation by preserving individual identities in generated images. Unlike previous methods, it directly incorporates face images into the UNet of Stable Diffusion. By adjusting the cross-attention layers, the model better captures facial features across scales. This approach achieves high accuracy in both identity similarity and prompt alignment.

### Strengths
- This method surpasses previous approaches in identity fidelity, as evidenced by superior performance in relevant metrics.
- This method supports customizable multi-identity generation.

### Weaknesses
- The writing of the paper is quite rough, with at least 10 instances of incorrect punctuation usage and a generally disorganized flow of ideas.
- The paper lacks sufficient contribution, as the core method is still based on IP Adapter without original innovations. Additionally, it is challenging to observe any performance advantages.
- The results are highly limited; in Figure 4, the customized faces are barely recognizable, with significant loss of facial details. The authors need to further explain the reasons behind this. Additionally, the paper claims to address the issue of decoupling expressions from the reference image; however, in Figure 5, each result appears to be a copy-paste of the reference image’s expressions.

### Questions
See weekness.

### Soundness
2

### Presentation
2

### Contribution
2
