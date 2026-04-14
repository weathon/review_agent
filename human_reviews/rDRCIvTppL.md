# Boosting Camera Motion Control for Video Diffusion Transformers

- Decision: Reject
- Scores: 6, 5, 5, 6

## Abstract
Recent advancements in diffusion models have significantly enhanced the quality of video generation. However, fine-grained control over camera pose remains a challenge. While U-Net-based models have shown promising results for camera control, transformer-based diffusion models (DiT)—the preferred architecture for large-scale video generation—suffer from severe degradation in camera motion accuracy. In this paper, we investigate the underlying causes of this issue and propose solutions tailored to DiT architectures. Our study reveals that camera control performance depends heavily on the choice of conditioning methods rather than camera pose representations that is commonly believed. To address the persistent motion degradation in DiT, we introduce **Camera Motion Guidance (CMG)**, based on classifier-free guidance, which **boosts camera control by over 400%**. Additionally, we present a sparse camera control pipeline, significantly simplifying the process of specifying camera poses for long videos. Code and models will be released upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper focuses on the challenge of camera pose control in video generation on transformer-based models (DiT). The study finds that conditioning methods, rather than pose representation, are key to performance. The authors introduce Camera Motion Guidance (CMG) to improve accuracy and a sparse camera control pipeline for easier pose specification. Their solutions work for both U-Net and DiT architectures, improving camera control across models.

### Strengths
1. The proposed CMG for camera motion control is clear and reasonable.

2. The experiments are comprehensive, featuring a variety of qualitative and quantitative results.

3. The paper is well-written and easy to understand.

### Weaknesses
1. The main insight of this paper is the significant role played by the high dimensionality of the conditioning. However, this conclusion is not clearly supported by Table 1. Without CMG, both MotionCtrl [1] (low dimension) and CameraCtrl [2] (high dimension) show poor performance, suggesting that it is specifically the combination of CMG and high-dimensional conditioning that leads to substantial improvements. The reason why DiT-CameraCtrl alone fails to work effectively remains insufficiently explained.

2. Another key insight is the design of CMG. While reasonable, it is not necessarily designed for DiT (it also has the potential for U-Net), making the connection to the main topic less coherent.

3. The motivation for sparse camera control is not well established. Unlike other sparse conditions, such as sparse image control (e.g., SparseCtrl [3]), sparse camera motion can be easily interpolated from intermediate camera poses, reducing the need for such a method.

### Questions
Please see the weakness.

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
The paper presents a systematic analysis of design choices for incorporating camera motion control into video diffusion models. Key findings include:  1) camera control performance depends more on the choice of conditioning methods rather than camera pose representations. 2) classifier-free guidance can significantly boost camera control. 3) Camera control can be applied sparsely.

### Strengths
The analysis of the paper is quite thorough, and the conclusion seems reasonable. It is interesting to see a head-to-head comparison of different camera pose representation and how the condition is inserted. It is also interesting to see the effect of applying CFG helps improve camera control.

### Weaknesses
The claim that being the “the first camera control model for space-time video diffusion transformers” is over-claimed given Banhmani et al’s paper was arxived over three months before this submission.

The analysis is not complete, first in ablations of table 1, it misses experiments of DiT-CameraCtrl with RT representation. 
Second, the analysis only evaluate the effect in terms of camera control accuracy of mostly static scenes. It does not show how different settings affect overall video quality, including how it affects object motions. It would make the analysis more meaningful to provide samples presenting dynamic objects, and includes evaluation of the video quality.

While the analysis is interesting and reassuring, the paper feels more like a technical report without new method being proposed, and the conclusions are not surprising findings. 

Ln 193 “Comparing with DiT, U-Net has a worse condition-to channel ratio leading to weakened control.” is there a typo? Should DiT and U-Net be exchanged order in this sentence?

### Questions
Could authors show additional analysis on how different settings affect the video quality? including how they may degrade dynamic object motions.

### Soundness
3

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
3

### Summary
This paper investigates the role of camera motion control in transformer-based video diffusion models, with a particular focus on Dit-based video models. The authors identify that the dimensionality of camera control embeddings significantly impacts model performance. To enhance generative control over video outputs, they introduce a Camera Motion Guidance mechanism, which improves the model's adherence to specified motion conditions. Additionally, they propose a data augmentation technique that selectively drops some camera motion signals during training, equipping the model with robust sparse camera control capabilities. Experiments on the RealEstate10K test set demonstrate the effectiveness of these proposed methods, showing improved performance in generating realistic videos with controlled camera movement.

### Strengths
- The paper is well-written, with clear and structured explanations that make the content accessible and easy to follow.
- The proposed method is theoretically sound, and the authors provide a well-reasoned justification for each component.
- The authors conduct extensive experiments that rigorously explore the effects of motion control on Dit-based video generation models, offering a comprehensive evaluation of their method's effectiveness.

### Weaknesses
- The paper has a noticeable similarity to [1], which also examines camera motion control in transformer-based video generation models. A more detailed discussion and comparison with this related work should be provided to highlight the novel aspects of the proposed approach and clarify its distinct contributions. Additionally, experimental comparisons with VD3D should be included to further validate the effectiveness of the proposed method.
- From my perspective, the Camera Motion Guidance mechanism does not represent a significant novelty, as classifier-free guidance is already a standard setting in existing generative modeling repositories, where it can be straightforwardly adapted by substituting text conditions with camera motion conditions.

References:

[1]. VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control, arXiv 2024.

### Questions
- What is the practical advantage of using sparse motion conditions? In Table 4, it appears that increasing the sparsity of camera motion negatively impacts inference performance. Additionally, during inference time, why not directly interpolate camera poses from the sparse keyframes and provide both the interpolated poses and key poses to the model?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper provides in-depth analysis on the design choices of adding camera control to U-Net-based and DiT-based video generation models. The investigated choices include the representation of camera parameters, camera conditioning methods, as well as guidance methods of diffusion model. This paper reveals that it is the conditioning methods more crucial than the representation, which is opposite to what is commonly believed. In terms of guidance methods, authors propose a camera motion guidance (CMG) that introduces camera motion guidance in the classifier-free guidance. The integration of these findings results in a camera control method with significant improvement in terms of motion metric.

### Strengths
Overall I think this is a good paper. Although technical novelty is limited when viewed in isolation, technical improvements come from an inspiring analysis of the details, and significant improvements are observed in experiments. 
- Paper is easy to follow
- The in-depth analysis is informative and inspiring. I especially like the concept of condition-to-ratio. And the subsequently emphasizing the condition method rather than camera embedding looks reasonable.
- The design of camera motion guidance looks reasonable as well. 
- Experiment results in table 1 are encouraging, where DiT-based method matches the performance of U-Net-based method.

### Weaknesses
- The idea of condition-to-ratio should be further investigated or emphasized, such as adjusting the number of dimensions to vary condition-to-ratio so that its effect can be observed more clearly.
- While authors state the proposed solution is broadly applicable across DiT models and U-Net models, I only see results of DiT models in experiments. U-Net models with CMG should be included to verify the claim.
- Only one backbone DiT model is evaluated
- In terms of sparse camera control, I wonder how to obtain the sparse camera control signal with good naturalness and smoothness. A way I can think of is subsampling from **a complete one**, but in such cases we can just input the complete signal instead without the need of using a sparse camera control. Any thoughts?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
