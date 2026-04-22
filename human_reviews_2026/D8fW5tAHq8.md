# COME: Advancing Representation Learning and Generative Modeling for High-Quality Text-to-Motion Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
Text-to-Motion generation aims to synthesize realistic 3D human motion from natural language descriptions. Although continuous diffusion models naturally align with the temporal and spatial continuity of motion, they have underperformed discrete token-based approaches in generation quality. However, as T2M tasks evolve to include motion editing, personalization, and multimodal control, they increasingly demand fine-grained semantics, compositionality, and diverse sampling—capabilities better supported by continuous frameworks. Motivated by these real-world demands and the inherent continuity of motion, we revisit continuous diffusion modeling and identify two core limitations: (1) motion representations are often crowded and poorly separable, which increases the difficulty of generation and denoising; (2) suboptimal generative modeling that further degrades generation quality. To address these challenges, we propose COME, a continuous diffusion framework that enhances both motion representation and generative modeling. COME comprises two main components: the Motion Contrastive Masked Autoencoder (MoCMAE) and the Cross-Condition Diffusion Transformer (ccDIT).
MoCMAE employs an asymmetric hybrid architecture that integrates Masked Motion Modeling to extract key spatio-temporal features and Contrastive Learning to further enhance feature discriminability, thereby providing an expressive latent space. Meanwhile, ccDIT incorporates ccDIT block for global and fine-grained semantic comprehension and then utilizes Stable-Min-SNR-$\gamma$ to address training-inference inconsistencies and the conflicts across different timesteps, thus boosting generation quality. Extensive experiments show that COME achieves SOTA performance while improving inference and training efficiency, highlighting the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes COME, a continuous diffusion framework for text-to-motion generation. COME includes a motion contrastive masked autoencoder (MoCMAE) serves as a continuous motion tokenizer with better inter-sample separability, and a cross-condition diffusion transformer(ccDIT) which specifically align sentence-level semantics and word-level semantics through different modules. Besides, COME utilizes Stable-Min-SNR-gamma strategy to address training-inference inconsistencies.

### Strengths
1. Extensive quantitative results. The authors provide thorough quantitative experiments across multiple datasets and tasks, including HumanML3D, KIT-ML, BABEL, MotionX++, etc. The proposed method COME achieves state-of-the-art quantitative results across standard T2M benchmarks, comparing with both continuous and discrete paradigms.

### Weaknesses
1. Insufficient technical novelty. The overall methodology design appears to be a naive ensemble of architectures and training strategies that have existed for quite a while, lacking clear task-driven motivation or theoretical insight. 

      - MoCMAE directly employs the preliminary masking strategy from MAE [1] and combine with contrastive learning loss. This combination has been explored in earlier works (e.g., [2]) and is not novel by itself. The paper doesn't include necessary discussions on those similar contrastive masked autoencoder frameworks in previous works.

      - ccDIT have different level of text embeddings injected using AdaLN-Zero and cross attention layer, which is rather a straight-forward extension to existing transformer architecture in text-to-motion methods, without any further analysis nor insights. Comparable DiT-based or hybrid conditioning designs have already been adopted in prior motion generation methods [3–5]. The paper fails to provide sufficient analysis or justification of the claimed advantages or unique contributions of this architecture.

      - Stable-Min-SNR-gamma strategy appears to be a training technique borrowed and slightly revised from an existing diffusion-based image generation paper [6]. Yet, the authors fail to provide any further analysis nor discussion about the technical insights of this technique in the text-to-motion generation tasks specifically. 

1. What makes it so important to highlight the asymmetric encoder-decoder architecture? What would be the changes in encoding performance and inference cost respectively, if adding the layers of decoder to make it symmetric comparing to the current MoCMAE design? 

2. The details of the text features processing are not clear. How are the word-level language features encoded and processed? How does the word-level features separate from the sentence-level features?


2. The qualitative results in Figure 4 are difficult to interpret as the prompts that the authors provide contain a lots of adjectives that are ambiguous by itself (e.g., fast, high, slowly) without any reference or clear assessments, and the visualizations use frame shots that often overlap with each other.  It would be more convincing if the authors can provide 2-3 sets of video demos for generated samples from the existing methods and from the proposed method for comparison.

    - E.g.  the second prompt in Figure 4, all methods seem to generate generally correct motions as described, but it’s quite hard to tell whose details (speed, height) are better with the overlapped frames without any definition or reference given.

    - Some of the generated examples in the provided demo videos are not well performed. E.g. in demo003, text prompt is “the person runs fast, kicks high in the air”, but the generated motion performs more like  a “jump” instead of a “high kick”.

    - The axe or reference for temporal direction should be mentioned alongside.


3. Ablation studies in Table 4 suggests that the contrastive learning module contributes relatively marginal improvements in generation performance compared to the impact of masking strategy and TransBlock, contradicting the authors’ claim that it is critical for learning a discriminative latent space. A more detailed analysis—such as latent-space visualizations or cluster separability metrics—would help validate this claim. 


4. No details about test set settings, control signal settings and the controllability evaluation metrics for the comparison results in Table 6 for motion control evaluation (Similarly,  Table 13 for motion editing), which makes the presentation hard to follow nor evaluate.

5. The demo videos of motion control are confusing. It's not clear how the control is realized in each demo videos. What's the representation for the control signal in each case?  Some of the generated samples are confusing and even look like failure cases, e.g. control_6_joint.mp4. 




[1] He, et al. “Masked Autoencoders Are Scalable Vision Learners.” In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022.

[2] Mishra, et al. “A simple, efficient and scalable contrastive masked autoencoder for learning visual representations.” arXiv preprint arXiv:2210.16870 (2022). 

[3] Chen, et al. Executing Your Commands via Motion Diffusion in Latent Space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[4] Tseng, et al. “EDGE: Editable Dance Generation From Music.” In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 448-458. 

[5] Shan, et al. Towards Open Domain Text-Driven Synthesis of Multi-Person Motions. n Proceedings of the European Conference on Computer Vision (ECCV) 2024.

[6] Shanchuan, et al. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 5404–5411, 2024.

### Questions
Please see the weaknesses points.

### Soundness
2

### Presentation
1

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
The paper proposes COME, including a Contrastive Masked Autoencoder (MoCMAE) and a cross-condition diffusion transformer (ccDiT) to improve the performance of text2motion generation. It first identifies the core limitations of previous works are 1) poor motion representation and 2) suboptimal generative modeling methods. and then address them by proposing a better design motion autoencoder and a diffusion transformer with modern designs. Extensive experiments demonstrated the improved performance using the proposed design across a suite of different motion generation tasks.

### Strengths
1. I agree that a better motion representation is important, and the proposed MoCMVAE makes sense along this line. 
2. The proposed latent diffusion model adopts several modern designs that should clearly improve performance compared to other old versions of latent diffusion models. 
3. I appreciate that the authors have conducted extensive experiments on a wide range of tasks and datasets. These experiments demonstrate that the proposed design could improve motion generation performance for broader tasks and areas.

### Weaknesses
1. Although it shows better performance comprehension evaluation across different tasks, the technical novelty is relatively limited. 
2. Constrative Masked Autoencoder is interesting, but more closely related work about motion autoencoder and latent representation should be discussed and compared, to better highlight the difference and motivation of the proposed design. 
3. ccDiT is rather a common design. Min-SNR-γ is commonly used in other areas, and the design of "Stable-Min-SNR-γ" is relatively minor. According to Table 5, the improvement is subtle. 
4. Given that most designs are relatively general or random, that is not very well motivated or introduce motion-specific insight, I'm looking for significantly stronger performance. However, the text2motion generation performance is not good enough. Some sota methods are missing, for example, StableMofusion [1].

Overall, I appreciate the effort to conduct extensive experiments across tasks and datasets. However, I believe ICLR is looking for technical novelty and insights that can have a broader impact to the community. So, I’m on the fence and leaning toward negative.

[1] Huang, Yiheng, et al. "Stablemofusion: Towards robust and efficient diffusion-based motion generation framework." Proceedings of the 32nd ACM International Conference on Multimedia. 2024.

### Questions
1. I wonder what the core contributions are in ccDiT, and how it is different from previous related works.

### Soundness
3

### Presentation
4

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
This paper proposes COME, a continuous text to motion model with a masked motion autoencoder (MoCMAE) for robust continuous latents and a conditioning-aware diffusion transformer (ccDiT) with AdaLN-Zero + token-level cross-attention. This paper reports strong FID/R-precision on HumanML3D and KIT-ML with 10-step DPM-Solver++ sampling.

### Strengths
1. Strong quantitative numbers on HumanML3D and competitive results on KIT-ML under standard metrics (FID, R-Precision).

2. One framework supports text-to-motion, motion editing, and text-to-motion control.

### Weaknesses
1. The motivation of closing the large performance gap between discrete and continuous is fine, but it’s quite similar to MARDM [1]. I understand the authors address it in a different way, but they do not compare with MARDM, neither methodologically nor in experimental results.

2. The authors claim “features encoded by existing continuous models (e.g., VAEs) are often overly concentrated” and rely on a t-SNE plot to support this. t-SNE does not preserve global distances or density, so that figure is not meaningful. The authors should use better diagnostics to show limitations of current continuous feature extractors and quantify “separability” in experiments comparing existing continuous motion encoders with the proposed masked motion encoder.

3. The model architecture and training are incremental. The authors just combine standard components and training recipes, and there is no strong technical improvement here.

4. Missing strong baseline results. Compare against recent, stronger works (MARDM [1], MotionStreamer [2], MotionLCM v2 [3]) for the text-to-motion generation task, and MaskControl [4] for the control task, with both qualitative and quantitative results. Current visualizations are not apples-to-apples.

5. Weak visual results in the supplement. For T2M generation, many cases show severe foot floating and foot–ground penetration. Compared with MoMask [5] (from the MoMask official website) on the same prompts (e.g., demo009, demo010), the motions look clearly less natural. For motion editing, I don’t see any real difference for edit_3. For T2M control, the human body shape (bone lengths) appears to change over time. In summary, most visualizations are not reasonable to me.

Reference

[1] Rethinking Diffusion for Text-Driven Human Motion Generation

[2] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space

[3] MotionLCM-V2: Improved Compression Rate for Multi-Latent-Token Diffusion

[4] MaskControl: Spatio-Temporal Control for Masked Motion Synthesis

[5] MoMask: Generative Masked Modeling of 3D Human Motions

### Questions
Refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a new text-to-motion generation framework, COME. It learns expressive latents via propsoed MoCMAE and generates motion sequence via ccDIT transformer and Stable-Min-SNR-r schedule to align training and sampling.

### Strengths
1. The paper demonstrates impressive quantitative and qualitative results, establishing a new state-of-the-art with a substantial margin.

2. The proposed framework is novel. The associated conclusions and experiments make significant contributions to the research community.

3. The paper is well-written, ensuring that its content is easily understandable for readers.

### Weaknesses
I have no major concern about this paper. There are some minor suggestions:

1. It will be better if the authors can provide user study on more benchmarks.

2. The GPU hours of training for each stage should be reported.

3. In Table 12, it should be "Continuous" instead of "Conitnue"

### Questions
Please kindly refer to the weaknesses mentioned above.

### Soundness
3

### Presentation
3

### Contribution
3
