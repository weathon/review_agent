# A Recipe for Watermarking Diffusion Models

- Avg Score: 5.33
- Decision: Reject
- Scores: 5, 6, 5

## Abstract
Diffusion models (DMs) have demonstrated advantageous potential on generative tasks. Widespread interest exists in incorporating DMs into downstream applications, such as producing or editing photorealistic images. However, practical deployment and unprecedented power of DMs raise legal issues, including copyright protection and monitoring of generated content. In this regard, watermarking has been a proven solution for copyright protection and content monitoring, but it is underexplored in the DMs literature. Specifically, DMs generate samples from longer tracks and may have newly designed multimodal structures, necessitating the modification of conventional watermarking pipelines. To this end, we conduct comprehensive analyses and derive a recipe for efficiently watermarking state-of-the-art DMs (e.g., Stable Diffusion), via training from scratch or finetuning. Our recipe is straightforward but involves empirically ablated implementation details, providing a foundation for future research on watermarking DMs.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper provides an empirical study on watermarking for deep diffusion models (DMs). The authors propose a simple yet effective pipeline to embed watermark information into generated contents and DMs.

### Strengths
1.	The paper is well-written and the methodology is explained clearly. The authors have also provided a comprehensive explanation of their research with supportive visualizations.
2.	The proposed watermarking pipelines are efficient and robust against some common distortions, which could have practical implications.

### Weaknesses
1.	The paper does not discuss how the proposed watermarking pipelines handle adversarial attacks or deliberate attempts to remove or modify the watermark. For example, finetune latent diffusion models with trigger prompt “[V]” again to remove the watermark.
2.	In unconditional or class-conditional generation, the watermark string is fixed. Injecting a new watermark string requires training a new model from scratch, which is time-consuming.
3.	The average PSNR (Peak Signal-to-Noise Ratio) presented in Table 1 is below 30 dB. In contrast, the majority of watermarking schemes typically achieve satisfactory visual quality when the PSNR is above 40 dB.

### Questions
1.	The training strategies for unconditional or class-conditional generation could potentially be optimized to minimize its cost.
2.	The robustness could be further demonstrated by considering additional post-processing operations, such as JPEG compression under varying quality factors.

Please refer to the following paper: Fernandez P, Couairon G, Jégou H, et al. The stable signature: Rooting watermarks in latent diffusion models[J]. ICCV2023.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
While the use of watermarking for copyright protection and content monitoring is a well-established approach, its application in the context of DMs is relatively unexplored. The work proposes a comprehensive analysis and a practical recipe (including two frameworks) for effectively watermarking cutting-edge DMs, such as Stable Diffusion. The suggested approach involves adapting conventional watermarking techniques to accommodate the unique characteristics of DM-generated content, providing a foundational guide for future research in this domain.

### Strengths
The strengths lie in the following aspects:
1) Originality: this work attempts to introduce watermark techniques into the generative neural network domain (diffusion model), which watermarks the neural model.
2) Clarity: the work was well-written and easy to follow. The organization of this work is satisfactory.
3) Results: the authors conducted extensive experiments to validate the effectiveness of the proposed methods.

### Weaknesses
The weaknesses can be identified in the following aspects:

1) Methodology: While the proposed framework is indeed well-explored in discriminative learning tasks, its technical contribution appears somewhat limited. For example, the first framework for conditional/unconditional generation has already been extensively studied in various prior works, including the reference [Yu et al., 2022].

2) Experiments: Despite the comprehensive nature of the conducted experiments, some crucial experiments were not included. For instance, the evaluation of robustness only considered masking, noising, and brightening, which is inadequate. Please refer to the subsequent questions for further details.

3) The quality of the watermarked images is not entirely satisfactory, as the average PSNRs fall below 30dB, indicating a significant impact of the watermark embedding on the original generative models.

[Yu et al. 2022] Ning Yu, Vladislav Skripniuk, Sahar Abdelnabi, and Mario Fritz. Artificial fingerprinting for generative models: Rooting deepfake attribution in training data. In IEEE International Conference on Computer Vision (ICCV), 2021.

### Questions
Some concerns need to be addressed:
1) Concerning the first framework, the authors proposed the incorporation of a watermark bit string into the training dataset. The experimental results validated the effectiveness of this approach. I concur with this strategy. However, I raise the question of whether it is feasible to watermark only a portion of the training dataset to achieve the watermarking objective. For example, is it feasible to watermark only 30% or 50% of the samples?

2) Regarding the second framework, in the design of the trigger prompt, the authors recommended using the uncommon identifier '[V]' as input. Should other rare identifiers, such as '!M~', be considered as well? Will the conclusions drawn from these considerations remain unaffected?

3) Regarding the experiments, when assessing the resilience of the watermarked images, only three types of distortions, namely masking, noising, and brightening, were taken into account. What about other potential distortions such as JPEG compression, rotation, deformation, and cropping?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes watermarking techniques for diffusion models to address legal challenges in copyright protection and generated content detection. It details two watermarking pipelines for different DM types and provides practical guidelines for implementation, balancing image quality with watermark robustness.

### Strengths
1. The paper addresses a highly relevant contemporary issue.
2. The experiments are thoroughly and rigorously executed.
3. The manuscript is well-crafted, presenting its arguments in a clear sequence.


Suggestions:
1. Consider relocating some of the visual elements to the appendix.
2. Shortening the captions of figures may enhance their readability.
3. It may be beneficial for the authors to concentrate on a single methodology to provide a more focused exploration of the subject matter.

### Weaknesses
1. Copyright scenario is not clear. Is the copyright protection for model owner or for user who downloaded? 
2. Detecting generated contents is also not clear. Are the authors proposing method for detecting generated content? If so, where is the related experiments?
3. Watermarking Stable Diffusion using Dreambooth has less novelty. The Dreamfusion itself is designed for training personalized concept to use it for Stable Diffusion's rich representation. In this sense, the authors change the personalized concept to watermark images. 

1. The manuscript could benefit from a clearer delineation of the copyright scenario. It would be helpful to specify whether the copyright protection mechanisms are designed to safeguard the interests of the model owner or the end-users who utilize the model.

2. The section on detecting generated content could use further clarification. If that is the case, could you please direct me to the experiments that validate this approach?

3. The approach to watermarking Stable Diffusion via Dreambooth may appear to have limited novelty since Dreamfusion is inherently capable of training personalized concepts for Stable Diffusion. It seems that this method lies in the adaptation of personalized concepts into watermark images.

### Questions
1. Regarding the watermarking process in Stable Diffusion, could you elucidate on the protocol if a caption such as "A photo of QR code" were provided? Is there a safeguard in place to prevent inadvertent leakage of the watermark under such circumstances?

2. Could you specify the lower bound of the bit-wise accuracy for the watermarking technique? Such a metric would be instrumental in assessing the robustness of the approach.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
