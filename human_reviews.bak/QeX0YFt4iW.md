# Multi-modality Adversarial Attacks on Latent Diffusion Models

- Decision: Reject
- Scores: 6, 5, 5, 3

## Abstract
Latent diffusion models (LDMs) achieve unprecedented success in image editing, which can accurately edit the target image with text guidance. However, the multi-modal adversarial robustness of latent diffusion models has not been studied. Previous works only focus on single modality perturbation, such as image or text, making them less effective while more noticeable to humans. Therefore, in this paper, we aim to analyze the multi-modal robustness of latent diffusion models through adversarial attacks. We propose the first Multi-Modality adversarial Attack algorithm (MMA), which modifies the image and text simultaneously in a unified framework to determine updating text or image in each step. In each iteration, MMA constructs the perturbed candidates for both text and image based on the input attribution. Then, MMA selects the perturbed candidate with the largest $L_2$ distortion on the cross attention module in one step. The unified query ranking framework properly combines the updating from both modalities. Extensive experiments on both white-box and black-box settings validate two advantages of MMA: (1) MMA can easily trigger the failure of LDMs (high effectiveness). (2) MMA requires less perturbation budget compared with single modality attacks (high invisibility).

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The proposed MMA is a multimodal adversarial attack against img2img latent diffusion models in a white-box and black-box setting. which searches for the most optimal adversarial image and adversarial text iteratively. Compared with the single-modality attacks, MMA can generate adversarial samples with smaller perturbation magnitudes, and mislead the target models successfully.

### Strengths
1. The paper is generally well-written and easy to follow. 
2. It's the first study of multimodal attack on latent diffusion models, which demonstrates the advantage of multimodal attacks over single modality attacks on img2img tasks.
3. The evaluation metrics in the paper are from two perspectives: imperceptibility and effectiveness, which are better for evaluating the performance of the baselines and MMA.

### Weaknesses
1. Related missing references:
[1] Diffusion Models for Imperceptible and Transferable Adversarial Attack
[2] Towards Adversarial Attack on Vision-Language Pre-training Models
2. In the transfer-based black-box setting, there is a lack of comparison with image transfer-based attacks such as TI-FGSM, SINIFGSM, and ... in experiments
3. There is no error analysis. When does the method fail and why? What are the limitations of this method?

### Questions
1. what's the difference between the Image and MMA w/o Image, and Text and MMA w/o Text in experiments, please describe in detail.
2. Co-attack is a multimodal adversarial attack designed for vision-language pretraining models, integrating both BERT-attack and PGD methods. Would it be feasible to deploy Co-attack against the LDMs discussed in the paper and subsequently contrast its efficacy with the proposed MMA?
Co-attack: Towards Adversarial Attack on Vision-Language Pre-training Models
3. In the transfer-based attack, can the methods, such as TI-FGSM and DI-FGSM be used to attack the target LDMs?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a multi-modality adversarial attack method called MMA for latent diffusion models, which perturbs both the image and text inputs simultaneously to generate more effective and imperceptible adversarial examples compared to single-modality attacks.

### Strengths
The methods are intuitive and easy to implement.

### Weaknesses
1. The method in the paper is not particularly innovative and is overly simplistic. The approach, especially the text search mechanism, seems rather basic and lacks novelty. No theoretical analysis is provided on the distortion induced by multi-modality attacks compared to single modality.

2. The paper's primary method appears constrained to untargeted attacks. This limitation curtails its adaptability and potentially its effectiveness in targeted adversarial scenarios.

3. The human evaluation is limited to using CLIP similarity as a proxy. Direct human studies evaluating imperceptibility would be more convincing.

4. The efficiency and computational overhead of the method are not analyzed or compared to baselines.

### Questions
See the above weakness.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents an adversarial attack framework aiming to conduct attacks on the text and image modalities in a unified framework, to reduce the attack magnitude on a single modality and make the attack unnoticeable.

### Strengths
The argued first consideration of a unified multi-modal adversarial attack framework.

Good writing and easy to follow.

### Weaknesses
lacking deep consideration about the multi-modal generation problem. 

The evaluation is not convincing.

### Questions
All my concerns come from the main challenge, the core contribution of the paper, all of which are more important and should be clarified rather than the first work on the adversarial attack on multiple modalities, in my opinion. Besides, there also exist some questions. 

1. The authors only present a solution for a multi-modal adversarial attack via attacking two modalities and selecting a prominent one. I cannot capture the core problem in the multi-modal adversarial attack and difficulties. Please make a thorough illustration. 
2. The evaluation metric designed by the authors is confusing. The paper evaluates PSNR between the original generated image and the adversarial generated image, but I am not sure whether this comparison pair has convincing effects.  I understand the authors aim to estimate the generation effect, but directly comparing two generated images seems to lack stable indications. 
3. The adversarial between the two modalities has no relationship in the paper, also verified in Table 1. Under this condition, I think the paper requires further improvement. 
4. Please clarify the differences in the utilization of SD-2-1, SD-1-4, and SD-1-5.
5. The imperceptibility has no visible verification.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces the Multi-Modality Adversarial Attack (MMA) algorithm, specifically designed for latent diffusion models (LDMs). MMA simultaneously modifies text and image components within a unified framework, effectively addressing multi-modal robustness. Extensive experiments demonstrate MMA's effectiveness in triggering LDM failures while requiring a smaller perturbation budget compared to single modality attacks, enhancing invisibility.

### Strengths
1. Exploration of an intriguing research topic focused on attacking latent diffusion models, shedding light on a relatively unexplored area.
2. Straightforward and easy-to-understand approach to address the challenges associated with the subject matter.

### Weaknesses
1. The evaluation metrics for the attack used in the paper cannot be used to quantitatively measure the success rate of the proposed attack. The proposed measurements are all performance metrics related to the image quality of the latent diffusion model and are relative indicators that only show that the attack is working roughly well.
2. The rationale behind the necessity of this attack is not thoroughly convincing. The paper lacks a clear articulation of the contributions that the proposed attack brings to the field of latent diffusion models.
3. The absence of human evaluation metrics to quantitatively assess the effectiveness of the attack in generating adversaries is a notable limitation.
4. The paper does not provide sufficient details regarding the success rate of the attack. Information on how many instances the attack succeeded and the overall success rate is missing, leaving a gap in the evaluation of its performance.
5. The use of citation methods such as \citep and \citet could be improved for better clarity and presentation in the paper, to distinguish between citations with or without parentheses.

### Questions
NA

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
