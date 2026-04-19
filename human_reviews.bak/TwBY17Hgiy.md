# Multi-task Learning with 3D-Aware Regularization

- Decision: Accept (poster)
- Scores: 6, 6, 6, 6

## Abstract
Deep neural networks have become the standard solution for designing models that can perform multiple dense computer vision tasks such as depth estimation and semantic segmentation thanks to their ability to capture complex correlations in high dimensional feature space across tasks. However, the cross-task correlations that are learned in the unstructured feature space can be extremely noisy and susceptible to overfitting, consequently hurting performance. We propose to address this problem by introducing a structured 3D-aware regularizer which interfaces multiple tasks through the projection of features extracted from an image encoder to a shared 3D feature space and decodes them into their task output space through differentiable rendering. We show that the proposed method is architecture agnostic and can be plugged into various prior multi-task backbones to improve their performance; as we evidence using standard benchmarks NYUv2 and PASCAL-Context.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes a structured 3D-aware regularizer for multi-task learning (MTL) in computer vision. The regularizer interfaces multiple tasks by projecting features from an image encoder to a shared 3D feature space and decoding them into task output space through differentiable rendering.

### Strengths
* The paper provides clear explanations of the proposed method, the problem it addresses, and the evaluation process. 

* The paper's contributions have significant implications for multi-task learning in computer vision. The structured 3D-aware regularizer can be integrated into existing models, improving their performance and reducing noise in cross-task correlations.

### Weaknesses
* The paper could benefit from a more extensive analysis of the proposed method's computational efficiency. While it is mentioned that the regularizer does not introduce additional computational cost for inference, a more detailed analysis or comparison with other methods in terms of computational efficiency would strengthen the paper.

* It would be beneficial to discuss the limitations or potential failure cases of the proposed 3D-aware regularizer. It would be valuable to address any potential drawbacks or scenarios where the regularizer may not be as effective.

* The paper lacks a thorough comparison with existing state-of-the-art methods in multi-task learning. It would be beneficial to include comparisons with other regularization techniques or approaches that address the problem of noisy cross-task correlations.

### Questions
See weakness section for more details.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper tackles the problem of multi-task learning for dense tasks like depth prediction, semantic segmentation, normal estimation. Cross-task correlations are noisy and there is no 3D regularization. The paper proposes introducing a latent that is structured and 3D-aware (by outputting a K-planes representation) and uses projection with differentiable rendering and a task specific decoder. The 3D awareness combined with view dependent rendering can be used to both regularize the 3D structure and multi-view consistency on the same 3D latent. Results are shown in NYUv2 and PASCAL-Context datasets.

### Strengths
The paper is quite well-written and easy to follow. The main idea of the paper is clear - to output a 3D feature representation from an image (using K-planes), which can then be used as part of a NeRF-like neural rendering pipeline to project the feature into a 2D feature image which can then be used by a task-specific decoder.

### Weaknesses
Predicting 3D representations (to use in neural rendering) from unlabeled 2D images and/or text is a relatively common idea [1, 2, 3, 4, 5]. The paper applies this idea in a multi-task setting, and adds a multi-view consistency for multi-view dataset. The results are not highly compelling compared to the baselines in the paper, which may internally learn some 3D structure too. So the explicit formulation of the 3D latent structure is not well motivated, unless the task requires some view editing or novel view synthesis, or an explicit 3D object representation, etc.

____
[1] Chan, Eric R., et al. "pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

[2] Cai, Shengqu, et al. "Pix2nerf: Unsupervised conditional p-gan for single image to neural radiance fields translation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[3] Lin, Chen-Hsuan, et al. "Magic3d: High-resolution text-to-3d content creation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[4] Zhang, Jingbo, et al. "Text2NeRF: Text-Driven 3D Scene Generation with Neural Radiance Fields." arXiv preprint arXiv:2305.11588 (2023). 

[5] Gu, Jiatao, et al. "Stylenerf: A style-based 3d-aware generator for high-resolution image synthesis." arXiv preprint arXiv:2110.08985 (2021).

### Questions
1. Its unclear if the results are statistically significant compared to the baseline. Since the method can be applied on any baseline MTL architecture, it is unclear if the improvements (especially in Table 2) are just due to randomness of the SGD training dynamics or is an actual improvement. Results over 3-5 trials should be reported if the improvements are marginal/inconsistent.

2. Table 3 shows that multiview consistency may not help in a MTL scenario considered in the paper. What is the significance of this section?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a structured 3D-aware regularizer that interfaces multiple tasks through the projection of features extracted from an image encoder to a shared 3D feature space and decodes them into their task output space through differentiable rendering. The experiment results are relatively good and promising.

### Strengths
Overall, I like this idea pretty much. For me, this work is like creating a new research line that makes most recognition methods 3D aware, which is dual to another research line that makes 2D image generation tasks 3D-aware. Although the introduced 3D-awared encoder does not explicitly give the 3D representation (I mean this method could not output decent 3D mesh), it gives way to investigating the 3D properties in most recognition tasks.

### Weaknesses
I paid a lot of expectations on the experiments after reading the Abstract and Introduction sections, but the experiments were not strong enough to match my expectations.

### Questions
For example, 
1. how is the 3D quality of the learned nerf? 
2. Any geometric insights about the performance improvement? otherwise, it will just be like a regularization paper (1 point improvement, that's all).
3. I think MTL might not be the best task to show the benefit of your method, do you have any ideas on that?

Overall, I think this is a decent paper to accept, and I know the questions I made above might not be easy to answer. I will increase my score if the authors can offer some convincing and motivated answers.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes to improve multi-task dense prediction by introducing a 3D representation branch during training. This 3D-aware branch(a triplane representation with volumetric render) shares the same backbone as a conventional multi-task learning setup. This NeRF branch encourages the feature extracted from the backbone to contain rich 3D information and to follow the physical imaging process. 
 This 3D-aware head also enables supervising with multi-view consistency. Experiments show that this auxiliary 3D aware branch helps improve the performance of the conventional branch during inference when the 3D aware branch is dropped.

### Strengths
(1) Interesting idea, the auxilary 3D aware branch  forces the backbone to be truly 3D-aware. This method also improves the alignment between different prediction heads and the depth/density head as they will be rendered following the physical imaging process.

(2) Using 3D aware representation as an auxiliary branch only for regularization also enables fast inference, which is a new and interesting idea to me.

### Weaknesses
(1) The 3D aware branch uses Triplane representation and volumetric render, which needs a specific camera model (intrinsic matrix). So it is suspicious that the model can somehow overfit to the specific camera parameters.  As a comparison, pixel-aligned scene representation (i.e., PiFU) can resolve this problem and it uses NDC representation.
For scenes with different camera parameters, the performance is unclear.

(2) Although the tri-plane representation has significantly reduced the memory cost and rendering of NeRF. Rendering the whole image and calculating the gradient is still a heavy burden. Common practice includes random sampling or pixel binning.  In the supplement material, there is an incomplete explanation. It is important to clarify the NeRF sampling implementation and extra training cost(memory footprint and FLOPS)

### Questions
My main concerns are (1) the potential camera overfitting problem, and (2) The extra training cost introduced by the NeRF head.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
