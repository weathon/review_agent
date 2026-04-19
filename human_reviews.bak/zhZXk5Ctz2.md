# Rethinking RGB Color Representation for Image Restoration Models

- Decision: Reject
- Scores: 5, 3, 8, 6

## Abstract
The per-pixel distance loss defined in the RGB color domain has been almost a compulsory choice for training image restoration models, despite its well-known tendency to guide the model to produce blurry, unrealistic textures. To enhance the visual plausibility of restored images, recent methods employ auxiliary objectives such as perceptual or adversarial losses. Nevertheless, they still do not eliminate the reliance on the per-pixel distance in the RGB domain. In this work, we try to redefine the very representation space over which the per-pixel distance is measured. Our augmented RGB ($a$RGB) space is the latent space of an autoencoder that comprises a single affine decoder and a nonlinear encoder, trained to preserve color information while capturing low-level image structures. As a direct consequence, per-pixel distance metrics, e.g., $L_{1}$, $L_{2}$, and smooth $L_{1}$ losses, can also be defined over our $a$RGB space in the same way as for the RGB space. We then replace the per-pixel losses in the RGB space with their counterparts in training various image restoration models such as deblurring, denoising, and perceptual super-resolution. By simply redirecting the loss function to act upon the proposed $a$RGB space, we demonstrate boosted performance without any modification to model architectures or other hyperparameters. Our results imply that the RGB color is not the optimal representation for image restoration tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper addresses limitations in the RGB color representation when conveying local image structures. The authors introduce an augmented RGB (aRGB) space, developed using an encoder, which captures both color and structural details. This new space offers more freedom in selecting loss functions and showcases performance improvements in various image processing tasks. Additionally, the aRGB space enhances interpretability, with the authors providing a comprehensive analysis of its properties and benefits.

### Strengths
This paper proposed an augmented RGB (aRGB) space is the latent space of an autoencoder that comprises a single affine decoder and a nonlinear encoder, trained to preserve color information while capturing low-level image structures. The results imply that the RGB color is not the optimal representation for image restoration tasks.

### Weaknesses
Based on the experiments, compared to previous methods, the improvement brought by vggloss is quite limited, with an increase of 0.1dB (PSNR) in Table 1 and 0.02dB in Table 2. Moreover, it hasn't been compared with other perceptual methods, such as lpips or ssim loss.

Although this paper claims to introduce a method that doesn't calculate loss in the RGB domain, the loss function used in training still falls within the category of pixel-based feature scale. Overall, it represents a relatively minor improvement to the loss function for low-level vision. Hence, the performance enhancement is limited.

Is the selection of the number of "experts" highly dependent on experience? Will different tasks have significant variations? It seems that an inappropriate selection of the number of experts might lead to even lower performance than not using this loss function at all.

### Questions
See weakness.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors propose aRGB loss for image restoration. The proposed loss is defined based on the latent space of an autoencoder, which consists of a single affine decoder and a nonlinear encoder. The autoencoder is trained to preserve color information while capturing low-level image structures. The authors replace per-pixel losses in the RGB space with their counterparts in training various image restoration models such as deblurring, denoising, and perceptual super-resolution.

### Strengths
+ A new latent representation space is proposed and employed for restoration loss design.
+ The aRGB loss is defined for diverse image restoration tasks.

### Weaknesses
-In the paper, the performance of the proposed loss are demonstrated on perceptual SR task. The results in table 1 are confusing. The PSNR and SSIM of RRDBNet are the highest among all the settings, but they are not bolded. The SSIM of the last setting is worse than most of settings for DIV2K-Val dataset, but it is bolded as better score.
-For perceptual SR and image deblur tasks, there are considerable baselines perform better than ESRGAN and MPRNet. For example, restormer and NAFNet could be used for deblurring evaluation. In this way, we can test whether the proposed loss could consistently boost performance and lead to a new SOTA.
-The performance gains are too small, which can hardly verify the effectiveness of the proposed loss.

### Questions
The proposed loss is similar to Fourier loss, which is also decomposed the image upon pre-defined basis. Can authors discuss the difference between them and compare their performance?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper introduces a novel approach to address the limitations of per-pixel RGB distances in image restoration. The authors propose a new representation space called augmented RGB (aRGB) space, where each pixel captures neighboring structures while preserving its original color value. By replacing the RGB representation with aRGB space in the calculation of per-pixel distances, the authors demonstrate performance improvements in perceptual super-resolution, image denoising, and deblurring tasks. In addition, the aRGB space allows for better interpretability through comprehensive analysis and visualization techniques. The contributions of this paper lie in the introduction of a versatile representation space, performance improvements in various image restoration tasks, and interpretability.

### Strengths
- The paper introduces the augmented RGB (aRGB) space for better image restoration.
- The paper provides a comprehensive and insightful analysis and visualization techniques for the aRGB space, enhancing interpretability. The analysis is solid and convincing.
- The versatility of the aRGB space allows for more freedom in choosing the loss function.

### Weaknesses
- The performance improvement of the proposed aRGB space in the denoising and debluring tasks seems insignificant. In Table 2, comparing the first two rows, and the last two rows, the PSNR gains are only 0.02 dB and 0.03 dB, respectively. In Table 3, the PSNR improvements between the last two rows are 0.07 dB on GoPro and 0.02 dB on HIDE dataset.

Additional comments
- Equation 6, L_{pair} should be L_{pixel}

### Questions
- The space aRGB is originally designed to encode structure information on a pixel basis. Why it can exhibit suppression of artifacts in SR tasks?
- The training process of aRGB auto-encoder does not involve any loss regarding local structure. Is it possible that the encoder also learns other information, e.g., texture, style?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes augmented RGB representation to alleviate the issue that per-pixel loss functions defined in the RGB color space tend to produce blurry, unrealistic textures. The proposed aRGB is designed with a nonlinear mixture-of-experts encoder and a linear decoder to meet two requirements. The experiments are conducted on various loss functions across different image restoration tasks for demonstration.

### Strengths
The paper analyzes the drawbacks of the per-pixel loss functions in the RGB space, To alleviate the issues of the tendency to producing blurry blurry, unrealistic textures, the paper proposes an aRGB representation to include the local texture for training. The analyses are sound and profound.  Based on the developed encoder and decoder, the method improves the performance of three image restoration tasks using different kinds of loss functions.

### Weaknesses
The additional architecture for aRGB representation transmission may introduce more computation consumption during the training phase. The improved performance on image motion deblurring seems to be minimal.

### Questions
1. The authors design a nonlinear mixture-of-experts encoder and a linear decoder for aRGB representation. Can this design principle be applied to guide the architecture design of image restoration networks?
2. Is the additional en/decoder equivalent to adding an additional branch for learning? I doubt whether the improved performance is yielded by the additional computation overhead.
3. The widely used dual-domain loss in models, such as MIMOUNet (Cho et al, ICCV'21) and SFNet (Cui et al, ICLR'23), can introduce global information refinement. How does aRGB compare to this loss function? This function does not lead to much computation overhead.
4. Does aRGB lead to extra computation overhead during training and inference?
5. Does the proposed the aRGB architecture rely on the dataset trained on?
6. The reviewer thinks that the performance improvement on GoPro is minimal. For example, only a 0.05 dB PSNR gain is obtained for MPRNet on GoPro. What do the authors think about this?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
