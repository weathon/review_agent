# Unbiased Object Detection Beyond Frequency with Visually Prompted Image Synthesis

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
This paper presents a generation-based debiasing framework for object detection. Prior debiasing methods are often limited by the representation diversity of samples, while naive generative augmentation often preserves the biases it aims to solve. Moreover, our analysis reveals that simply generating more data for rare classes is suboptimal due to two core issues: i) instance frequency is an incomplete proxy for the true data needs of a model, and ii) current layout-to-image synthesis lacks the fidelity and control to generate high-quality, complex scenes. To overcome this, we introduce the representation score (RS) to diagnose representational gaps beyond mere frequency, guiding the creation of new, unbiased layouts. To ensure high-quality synthesis, we replace ambiguous text prompts with a precise visual blueprint and employ a generative alignment strategy, which fosters communication between the detector and generator. Our method significantly
narrows the performance gap for underrepresented object groups, e.g., improving large/rare instances by 4.4/3.6 mAP over the baseline, and surpassing prior L2I synthesis models by 15.9 mAP for layout accuracy in generated images.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a generation-based debiasing framework for object detection that goes beyond frequency-based methods. The authors introduce a Representation Score (RS) to quantify representation quality by combining instance frequency, visual diversity, and contextual diversity, enabling targeted layout recalibration to synthesize unbiased data. To improve synthesis fidelity, they replace textual layout prompts with visual blueprints that provide unambiguous spatial guidance and propose a duality-aware generative alignment mechanism to align the detector and generator. Experiments on MS COCO and NuImages demonstrate substantial gains in both detection performance and generation fidelity, establishing a new state of the art in debiasing for object detection.

### Strengths
1. The motivating study in Section 2 is carefully designed and convincingly demonstrates the “frequency trap” and “fidelity gap.” This provides a solid empirical foundation for the proposed approach.
2. Extensive experiments on MS COCO and NuImages show large and consistent improvements. The ablation studies are thorough, demonstrating the contribution of each component.
3. The proposed framework integrates multiple innovations:
(i) an RS-driven layout recalibration module,
(ii) a visual blueprint-prompted synthesis mechanism, and
(iii) a duality-aware generative alignment strategy.
Together, these components form a well-engineered pipeline with clear rationale and internal consistency.

### Weaknesses
1. The experiments focus primarily on Faster R-CNN + ResNet-50. It is unclear whether the proposed method generalizes to other detection paradigms (e.g., DETR, YOLOv11). Low performance caused by data bias may not be significant in these modern models.

2. When the number of categories is large (e.g., 1000 classes), the colors representing different categories in the Visual Blueprint become highly similar, leading to confusion in the generated features. Therefore, compared with layout-based image generation methods, this approach is only suitable for scenarios involving a limited number of categories.

3. While RS integrates frequency, visual, and contextual diversity, the weighting and combination (Eq. 2) seem heuristic. No formal analysis or ablation on hyperparameter sensitivity (e.g., β) is provided.

4. The paper is dense, with long mathematical sections and multiple notations introduced in quick succession. Some figures (e.g., Fig. 2 and Fig. 3) could be clearer in explaining how each component interacts dynamically during training.

### Questions
1. Using the Visual Blueprint as the control condition is exactly the same as in ControlNet. So why is the performance of ControlNet + Resampling so poor?

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
4

### Summary
This paper rethinks the effectiveness of generated data for training object detectors, and proposes the representation score (RS) as a better metric than frequency to select more beneficial layouts, followed by layout recalibration. To enhance the generation quality, the authors propose to utilize the visual blueprint as the task prompt and then design a new loss term to enhance detector-generator dual awareness. Extensive experimental results demonstrate the effectiveness of the proposed method.

### Strengths
- The problem setting is fairly clear, including enhancements for both layout and image generation.
- The preliminary exploration of layout frequency is extensive.
- The authors conduct experiments on both fidelity and trainability on both COCO and NuImages.

### Weaknesses
- About Representation Score:
  - In line 161, considering that the box size s and horizontal position u are both continuous, do you conduct any quantization to construct the RS group?
  - In Sec. 2, the authors conduct extensive experiments to demonstrate that frequency is not the best metric for layout selection, which, however, cannot directly connect with the complicated definition of RS in Equ. 1.
  - Considering `Freq-Aware Gen` is still a solid baseline, I would expect to see a comparison between it and the complicated RS, with or without the more complicated RS calibration.
- About Visual Blueprints:
  - Besides comparing with the textual prompting of GeoDiffusion, a more direct comparison would be using ControlNet, but the visual prompt is a simple class binary map (i.e., a map of size HxWxC, and the area belonging to class C will be 1 in the C-dimension). It seems that this simple baseline is quite similar to the visual blueprint.
  - Duality-Aware Generative Alignment is proposed to enhance the detector instead of the generator. Do I understand correctly?
- Overall, this paper studies an interesting problem. The first half is good, but the solution is mostly based on intuition. Considering the complexity of the proposed method, a more detailed step-by-step ablation is necessary. I would like to review the author's rebuttal to inform my final decision.

### Questions
In Figure 4 (right), 1st row and 2nd column, the `dining table` class name is missing.

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
This paper proposes a generation-based debiasing framework for object detection. It claims that frequency is an incomplete proxy and previous methods lacks fidelity. Then it designs a representation score to replace frequency, this score can quantify how well a concept is represented across both sample density and representation diversity. This paper also presents a visual blueprint to replace text prompts so that offer a clear instruction to improve the fidelity.

### Strengths
1. The performance is good. The improvement is substantial compared to previous methods, such as GeoDiffusion.
2 . This method is simple yet effective. The two main contributions, the representation score and visual blueprint, are easy to understand and significantly improve the performance.
3. The motivation is clear. The observations in Sec. 2 explain why frequency is not enough and fidelity is important.

### Weaknesses
1. Lack a figure to describe the overall pipeline in detail. Figures 2 and 3 are used to explain the layout recalibration and blueprint construction; however, there is no figure to explain the complete pipeline, which can be confusing.
2. The effect of Generative Alignment is negligible. Can you explain its necessity?
3. There is no connection between the two contributions in this paper, which makes the paper an incremental industry-focused work rather than a cohesive academic study.

### Questions
See weakness above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a novel debiasing framework for object detection, named "Beyond Frequency: Scoring-Driven Debiasing for Object Detection via Blueprint-Prompted Image Synthesis." The key contributions are:
Representation Score (RS): A new metric that goes beyond mere frequency counts to diagnose representational gaps in the training data by considering both sample density and representation diversity.
Visual Blueprint-Prompted Image Synthesis: A method that replaces ambiguous text prompts with precise visual blueprints to guide the generation of high-quality, unbiased images.
Generative Alignment Strategy: A mechanism that fosters communication between the detector and generator, ensuring high-fidelity synthesis of debiased samples.
Dynamic Debiasing Engine: A system that continuously refines the RS based on detector errors, ensuring adaptive and targeted debiasing throughout training.
The method significantly improves the performance for underrepresented object groups and achieves state-of-the-art results in terms of both debiasing effectiveness and generation fidelity.

### Strengths
1. The paper introduces a novel approach to diagnosing and addressing dataset biases by integrating representation scores with visual blueprints and generative alignment. This method is innovative as it overcomes limitations found in existing techniques.
2. The research is characterized by rigorous experiments and comprehensive analysis. The proposed methods are well-implemented, validated, and demonstrate excellent performance improvements.
3. The contributions are significant as they tackle a major challenge in object detection by providing a practical solution that enhances model performance and fairness.

### Weaknesses
I appreciate the great efforts for this paper with clear motivation, thoughtful analyses, detailed strategies and comprehensive evaluations. Even so, after carefully considering the contributions of this work, I have some main concerns on the insights the paper conveys, besides some concerns on details in the paper.
1. *The insights*
  - a) The study for the motivation in Section is too empirical and heuristic. Those observation and analyses mainly focus on the performance comparison. The performance gains/gap motivates this work. This idea is somewhat reasonable. But, it is strongly suggested to provide deeper analyses with qualitative evaluation. E.g., do object detectors need those generated images that are not always well generated by models? What fresh impacts or findings those generated images would bring to detectors, in comparison with conventional augmentation methods? What generated images are more valuable to object detectors?

  - b) Because of a), although this paper is well written, it would be an unconscious misguidance to the main idea proposed in this paper: the layout generation and L2I generation. The core work of this paper is to focus on the L2I generation and there is very few discussions on the object detectors, e.g., only “duality-aware generative alignment” returns to the main task-object detection. If so, why not clearly introduce the work of L2I? Overall, I) I appreciate the whole work, but I sincerely suggest that we can make a different effort for more insights and more essential findings on the research topic, not just performance gap or gains that motivate us. II) For the concerns aforementioned, it is not easy to response or well address in this paper. But, if the authors have more insights or analyses (e.g., mentioned in a)), we can discuss in the rebuttal stage.

 *Concerns on technical details:*

2. The paper could benefit from a more detailed analysis of the sensitivity of the representation score's parameters (e.g., βin RS(G) = Dfreq(G) · (Dvis(G) + β · Dctx(G)) .). Understanding how these parameters affect the results would provide more insight into the robustness of the method.

3. The paper does not discuss the computational cost of the proposed methods. Given the complexity of the generative models and the dynamic debiasing engine, it would be useful to include an analysis of the computational requirements and potential trade-offs.

3.	For the layout generation, it is hard to ensure the accurate and plausible generation results. So layout recalibration is important. But, the strategy in Line180-185 and even the part of “Layout recalibration” worth more considerations.

4.	The layout generation and L2I generation is not always perfect. It is evitable that some negative impacts they make. Thus, it is strongly suggested that those impacts and analyses can be truly provided and analyzed. Those analyses would be a valuable suggestion or lesson for other researchers.

5.	Some details are suggested to provide. E.g., 1) Line263-267: it is kind to explain the more details of the implementation for the reproduction.

### Questions
1. What is the computational cost of the proposed methods compared to existing techniques? Are there any trade-offs in terms of training time or resource requirements?
2. Could the authors provide visualizations of the class distributions in both the original and generated datasets? This would help in understanding how the proposed method alters the distribution to achieve debiasing.
3. Could the authors provide more details on how the parameters of the representation score (e.g., β, τ) were chosen? Were any experiments conducted to analyze their sensitivity?

### Soundness
3

### Presentation
3

### Contribution
3
