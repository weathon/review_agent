# Scaling Concept With Text-Guided Diffusion Models

- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
Text-guided diffusion models have revolutionized generative tasks by producing high-fidelity content based on text descriptions. Additionally, they have enabled an editing paradigm where concepts can be replaced through text conditioning. In this work, we explore a novel paradigm: instead of replacing a concept, can we scale it? We conduct an empirical study to investigate concept decomposition trends in text-guided diffusion models. Leveraging these insights, we propose a simple yet effective method, ScalingConcept, designed to enhance or suppress existing concepts in real input without introducing new ones. To systematically evaluate our method, we introduce the WeakConcept-10 dataset. More importantly, ScalingConcept enables a range of novel zero-shot applications across both image and audio domains, including but not limited to canonical pose generation and generative sound highlighting/removal.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work explores scaling concepts up or down when performing text-guided image editing using diffusion models. The authors conduct preliminary studies and observe that diffusion models can remove concepts with a null prompt, and this ability can extend to the audio modality. Consequently, they devise a method that combines the reconstruction noise prediction process with the original text prompt and the removal noise prediction process with a null-text prompt. Additionally, they introduce several combination mechanisms, such as constant, linear, and non-linear strategies. Experiments were conducted on a newly constructed dataset, WeakConcept-10, to validate the effectiveness of the proposed method.

### Strengths
1. The authors successfully demonstrate that diffusion-based methods can scale concepts across both image and audio modalities.
2. They introduce the WeakConcept-10 dataset.
3. Experiments cover multiple applications in image and audio domains.

### Weaknesses
1. The motivation is unclear. The authors' claim that existing editing methods focus primarily on replacing one concept with another seems overly arbitrary and somewhat misleading. Many current editing paradigms modify concepts without direct replacement, such as [1-2].

[1] Direct Inversion: Boosting Diffusion-Based Editing with 3 Lines of Code    
[2] Inversion-Free Image Editing with Natural Language    

2. This work's main focus is on the inversion process and how to combine the information-preserving and information-modifying branches. However, it appears to overlook relevant prior work in this area.

3. Regarding the hypothesis: The authors hypothesize that the removal effect is due to the interaction between cross- and self-attention mechanisms in diffusion models, but no experiments are provided to support this claim.

4. In the preliminary study, there is a divergence between the observations and the final proposed method. Concept decomposition intuitively suggests the ability to extract different concepts from the input image separately. Yet, the authors only test the ability to remove a concept using a null prompt, without directly demonstrating the decomposition capability. Furthermore, the link between the concept removal ability and the later concept enhancement method is unclear.

5. Is the method limited to enhancing or suppressing existing concepts? If it cannot add or modify new concepts, this work faces significant limitations. Moreover, many current text-editing methods are already capable of object enhancement, such as creative enhancement, weather manipulation, etc.

6. Most experiments focus on concept enhancement without sufficiently validating the model's performance in concept suppression.

7. Formatting issue: In Section A.4, there is an incomplete reference ("in ??").

### Questions
1. In the preliminary study, is the prompt “[class].” randomly selected from any class present in the image, or is it only partial? Since the conclusion states that diffusion models can decompose a concept, it's important that this concept can be an arbitrary one.

2. What are the differences and connections between this work and existing research on image enhancement?

[1] Diffusion Models for Image Restoration and Enhancement – A Comprehensive Survey   
[2] Generative Diffusion Prior for Unified Image Restoration and Enhancement

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
While traditional methods focus on replacing concepts with new ones, the paper introduces an approach to scale existing concepts. By analyzing how these models decompose concepts, the authors propose a method called ScalingConcept, aiming to either enhance or suppress concepts present in the input without introducing new elements.

### Strengths
1.	The paper explores a novel aspect of scaling rather than replacing concepts, which adds a fresh perspective to the application of diffusion models.

2.	ScalingConcept is simple and effective, which indicates practical applicability in enhancing or suppressing concepts.

3.	ScalingConcept's ability to support novel zero-shot applications is across both image and audio domains.

### Weaknesses
1.	There is no mention of how ScalingConcept compares to existing methods or techniques, which might raise questions on its relative performance improvements.

2.	The paper does not discuss the complexity of applying ScalingConcept in diffusion models.

3.	It is necessary to demonstrate the method's scalability and effectiveness across diverse datasets beyond WeakConcept-10.

4.	In the application zoo, only a few cases are provided. It is unclear whether the method is effective universally.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes a new paradigm for text-to-image generation through the scaling concept. Precisely, the authors control the intensity of the original concept by adjusting the noise difference across two distinct branches (reconstruction and removal): $\hat{\varepsilon}_t = \varepsilon_t^{\varnothing} + \omega_t \cdot \left( \varepsilon_t^r - \varepsilon_t^{\varnothing} \right)$. The authors conducted an empirical study investigating concept decomposition phenomena in current text-guided diffusion models and baseline comparison experiments on the proposed WeakConcept-10 dataset. Additionally, they explore the practical application of the proposed method in various scenarios, like Canonical Pose Generation, Object Stitching, and others.

### Strengths
1. The paper is logically structured, and all figures and tables are well-presented.
2. The paper presents a novel motivation, introducing a simple and understandable paradigm for text-to-image generation based on the scaling concept. The authors also conduct empirical investigations on existing image and audio generation tasks to support their claims.
3. The proposed ScalingConcept method seems promising in various downstream tasks, including Object Stitching, Pose Generation, and Creative Enhancement.

### Weaknesses
1. The scientific question in this paper is not straightforward. The author seems inspired by their observations without clearly pointing out the flaws of existing methods.
2. The empirical study is inconsistent with the proposed method's idea. For example, Figure 2 shows the removal concepts using different prompts during the reconstruction. However, the proposed method removes image concepts by using a blank prompt.
3. The analysis of Figure 3 lacks illustrative examples. As noted in point 2, replacing a prompt and using a blank prompt are distinct. The former relies more on model capability and a suitable prompt.
4. The main issue with this paper is the unreliable experimental comparison. The authors only compared two baseline methods on the closed dataset WeakConcept-10. Moreover, they did not even analyze the quality of the proposed WeakConcept-10. Therefore, I doubt whether such a comparison is meaningful and reliable.
5. I think amplifying a certain concept in an image inevitably leads to weakening other concepts. For instance, in Figure 7, when the concept of "clock" is emphasized, the "finger" appears distorted. However, the authors do not analyze these shortcomings.
6. Some formatting issues. The references in Appendix A.2 are not displayed correctly.

### Questions
1. Why does LPIPS performance degrade when using the Noise Regularization early exit strategy? The author should conduct more analysis on it。
2. The authors should conduct more theoretical analysis rather than an empirical study on the rationality of the proposed method.

### Soundness
2

### Presentation
2

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
The paper presents ScalingConcept which aims to enhance or suppress existing concepts in real input data using text-guided diffusion models. The proposed method leverages the concept removal and reconstruction capabilities of diffusion models to achieve concept scaling across both image and audio domains. The paper introduces the WeakConcept-10 dataset to evaluate the performance of the method and demonstrates its application in various zero-shot tasks such as canonical pose generation, object stitching, weather manipulation, and sound highlighting/removal.

### Strengths
- The paper conducts a thorough empirical analysis of the concept removal phenomenon in text-guided diffusion models, establishing a solid foundation for the proposed method.

- The ScalingConcept method is versatile, showcasing applications across multiple domains (image and audio) without additional fine-tuning.

- The introduction of the WeakConcept-10 dataset is a valuable contribution that provides a benchmark for evaluating concept scaling methods.

- The experiments are well-designed, with detailed comparisons against baseline methods and extensive ablation studies that illustrate the effectiveness of the proposed approach.

### Weaknesses
- While the method performs well on the WeakConcept-10 dataset, its generalizability to other datasets or more complex concepts remains uncertain. For example, in what scenario we will need this technique? It is applicable to combine removal and addition by existing methods to achieve the same performance. Further validation on diverse and challenging datasets would strengthen the paper.
In addition, the paper does not thoroughly address the scalability of the method to larger datasets or higher-resolution inputs, which could be a limitation for real-world applications.

- The complexity might limit its practical applicability. The reliance on precise hyperparameter tuning and inversion techniques may pose challenges for real-world applications.

- The method's effectiveness is highly dependent on the quality of text-to-X associations. In cases where the text prompt is not well-aligned with the model's understanding, the results might be suboptimal.

### Questions
- How sensitive is the method to the choice of hyperparameters (e.g., scaling factor and schedule)?

- The WeakConcept-10 dataset is a great start, but how does the method perform on more diverse and complex datasets? Have you tested it on standard benchmarks like COCO?

- Can you provide more insights into potential real-world applications of ScalingConcept? Especially the unique ones?

### Soundness
3

### Presentation
3

### Contribution
2
