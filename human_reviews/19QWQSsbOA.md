# Multi-scale Conditional Generative Modeling for Microscopic Image Restoration

- Decision: Reject
- Scores: 6, 5, 3, 6

## Abstract
The advance of diffusion-based generative models in recent years has revolutionized state-of-the-art (SOTA) techniques in a wide variety of image analysis and synthesis tasks, whereas their adaptation on image restoration, particularly within computational microscopy remains theoretically and empirically underexplored. In this research, we introduce a multi-scale generative model that enhances conditional image restoration through a novel exploitation of the Brownian Bridge process within wavelet domain. By initiating the Brownian Bridge diffusion process specifically at the lowest-frequency subband and applying generative adversarial networks at subsequent multi-scale high-frequency subbands in the wavelet domain, our method provides significant acceleration during training and sampling while sustaining a high image generation quality and diversity on par with SOTA diffusion models. Experimental results on various computational microscopy and imaging tasks confirm our method's robust performance and its considerable reduction in its sampling steps and time. This pioneering technique offers an efficient image restoration framework that harmonizes efficiency with quality, signifying a major stride in incorporating cutting-edge generative models into computational microscopy workflows.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposed a multi-scaled generative model that uses a diffusion model (DM) for low-frequency image and a GAN for high frequency images. The wavelet transform provides multi-scale images without lossy encoding process. The lossless compression is particularly important for microscopic imaging where high-frequency component are sparse and non-Gaussian. Additionally, the authors showed the near-Gaussian property of low-frequency component and thus employed Brownian Bridge Diffusion Process (BBDP). The idea of employing different networks (DM and GAN) to different resolutions according to the characteristics of microscopic dataset is novel. The proposed MSCGM (multi-scale conditional generative model) showed improved super-resolution result with fast inference time.

### Strengths
The paper analyzed the characteristics of microscopic images and proposed adequate methodology to address the sparsity and non-Gaussianity. Since the wavelet transformation divides the image into two subbands (high- and low- frequency coefficients) losslessly, handling each subband in a different manner is original. 
The contribution of the work is clear and well demonstrated. 
In addition, the work could be further applied to different modality images where sparse or non-Gaussianity exist.

### Weaknesses
Although the idea of the paper is novel, the effectiveness of the work has not been thoroughly assessed. The use of WT and the superiority of the proposed method compared to conventional method should be further evaluated. The specific comments are described in Questions.

### Questions
The paper demonstrated that the low-frequency coefficients in higher scales show Gaussian tendency and thus applied this to BBDP. The idea is novel and well hypothesized, but it would be helpful if other DM methods, such as IR-SDE and ReFusion methods that are implemented on 4x super-resolution experiment, are also tested on microscopy image dataset. Only CMSR (GAN: non-diffusion model), is compared at the moment, not showing the effectiveness of proposed near-Gaussianity assumption.
Similarly, applying BBDM to full resolution image does not seem to be fair comparison. Since many works demonstrated the effectiveness of multi-scale diffusion models, BBDM should be implemented in a same manner as the proposed method to prove the superiority of WT instead of other compression technique. Please conduct an ablation study that replaces WT with simple down-sampling.
Is there any specific reason why the proposed work adopted BBDM which was initially designed for image translation where input and target domains are different? Super-resolution tasks seem to have similar domains for input and target. Justify the choice of BBDM for super-resolution.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors present a novel multi-scale generative model that leverages the Brownian Bridge process within the wavelet domain. This approach enhances inference speed while maintaining high image quality and diversity. The model is further integrated with computational microscopy workflows, expanding its applicability. The authors evaluate its performance on both microscopy and natural image datasets, demonstrating that it achieves slightly better results compared to existing methods such as IR_SDE, Refusion, and BBDM, with the added advantage of faster inference.

### Strengths
• Integrating the Brownian Bridge Diffusion Process (BBDP) with adversarial learning in a multi-scale wavelet diffusion framework is innovative, enhancing image quality and stability. 

• The model achieves notable speed improvements, delivering faster inference without sacrificing image quality. 

• Performance remains consistent across diverse experiments, demonstrating robustness on both microscopy and natural images.

### Weaknesses
• The paper lacks a clear motivation for applying this model to computational microscopy workflows. The rationale for this specific application is unclear and lacks context, the relevance to microscopy appears out of place. A discussion on how this functionality benefits microscopy would help justify this direction and clarify its practical utility.

• The primary advantage of this method is its reduced inference time; however, the paper lacks a direct comparison with other methods that similarly aim to improve efficiency. Including such a comparison would provide valuable context and help quantify the benefits more clearly.

• The general evaluation lacks depth and is missing ablation studies. 

• There appear to be configuration issues with the comparison methods. For instance, IR-SDE [1] is cited as requiring 100 steps, but the authors use 1000, which significantly prolongs inference time. With the correct configuration (100 steps), the inference time should drop from 32 seconds to approximately 3 seconds.

• The choice of metrics is limited and somewhat inadequate for a super-resolution task. Relying solely on PSNR and SSIM may overlook important aspects of image quality. Including pixel-based metrics would provide a more comprehensive evaluation and might show shortcomings of the proposed method.


[1] Luo, Ziwei, et al. "Image restoration with mean-reverting stochastic differential equations." arXiv preprint arXiv:2301.11699 (2023).

### Questions
Especially considering that inference time is one of the main benefits, why was it not compared to models with fewer step counts or at least an in-depth analysis of how step counts influence the SOTA model performance? E.g. [2], or other methods that can be applied to the problem domain?

[2] Phung, Hao, Quan Dao, and Anh Tran. "Wavelet diffusion models are fast and scalable image generators." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors propose a multi-scale conditional generative model (MSCGM) for image restoration, incorporating multi-scale wavelet transforms and a Brownian bridge stochastic process. The wavelet transform is included due to its reversibility, which maintains information integrity in the latent diffusion space, in contrast to traditional Latent Diffusion Models (LDM). The Brownian bridge stochastic process is leveraged to introduce conditional images in both forward and reverse processes. While the authors aim to address microscopic image restoration, the motivation and results in the paper do not consistently support this focus.

### Strengths
1. The authors recognize the loss of detail in LDM, a known issue, and apply it to the microscopic image restoration context, an interesting direction.
2. They introduce the novel idea that the Brownian bridge stochastic process could effectively integrate conditional images.

### Weaknesses
1. **Lack of Consistency:** The paper lacks organization and clarity. Although the title emphasizes "Microscopic Image Restoration," the experiments primarily focus on "Natural Image Super-resolution" and "Low-light Natural Image Enhancement." Only a small subset of results explores microscopic images. If the model is intended for general image restoration, it would be more accurate to propose it as a ‘unified image restoration’ model. I suggest the authors either refocus their experiments more heavily on microscopic image restoration to align with the title, or broaden the title to reflect the wider scope of image restoration tasks covered in the paper.
  
2. **Introduction Needs Refinement:** The introduction lacks a clear problem definition and research motivation. The first two paragraphs provide a broad overview of diffusion processes that diverges from the paper’s focus. The discussion on latent diffusion downsampling is a well-known issue and could be alleviated by higher resolutions. The authors should clearly articulate why microscopic images especially require the multi-scale wavelet transform in the introduction. Please include a discussion of how their approach compares to or builds upon these existing wavelet-based diffusion models in the Introduction, highlighting any key differences or improvements.
  
3. **Lack of Acknowledgment of Prior Work:** The paper does not credit previous studies applying wavelet transforms in diffusion models, which could mislead readers into believing the concept originated here. Papers like "Wavelet Diffusion Models are Fast and Scalable Image Generators (CVPR 2023)" and "Training Generative Image Super-Resolution Models by Wavelet-Domain Losses Enables Better Control of Artifacts (CVPR 2024)" are directly related and should be cited with comparisons to clarify this study’s contributions.
  
4. **Figure 1 Illustration Issues:** The paper title focuses on "Microscopic Image Restoration," yet Figure 1 uses natural images. Including examples of microscopic images to show the degradations introduced by LDM and Refusion compared to MSCGM would enhance clarity.
  
5. **Methodology Development Clarity:** The description of the wavelet transform on page 4 is overly general, with key details moved to the appendix. Clear explanations of any novel model designs or algorithmic adaptations should be provided in the main text.
  
6. **Quality of Mathematical Presentation:** Symbols in the equations are used without proper declarations or explanations. Inconsistent symbols, like the variable for the normal distribution \( N \), further detract from clarity.
  
7. **Algorithm 1 Lack of Context:** Algorithm 1 on page 5 is underdeveloped. Symbols are not defined before use, and the algorithm lacks defined input-output requirements.
  
8. **Figure 2 Diagram Confusion:** Figure 2 is difficult to interpret. The illustration doesn’t clearly label network modules, workflow processes, or shared parameters (only a line is shown), which fails to clarify the model structure effectively.
  
9. **Lack of Dataset Information:** The results section includes evaluations of microscopic images, but there’s no description of the dataset. Is it public or private? What is the image count? Without these details, readers cannot analyze or reproduce the results. Please provide a detailed description of the microscopic image dataset used, including its source, size, and any preprocessing steps applied.
  
10. **Insufficient Ablation Studies:** Results provide only a simple comparison with LDM, without deeper exploration of MSCGM’s components or ablation studies to justify the performance benefits of each module.
  
11. **Unconvincing Model Performance:** The model’s performance requires further validation through comparison with advanced models. Numerous diffusion-based image restoration models from 2024 exist, yet none are used for comparison. This weakens the paper’s credibility. Key diffusion-based image restoration works worth considering include:  
   - RDDM ([link](https://cvpr.thecvf.com/virtual/2024/poster/31373))  
   - HIR-Diff ([link](https://cvpr.thecvf.com/virtual/2024/poster/29665))  
   - WF-Diff ([link](https://cvpr.thecvf.com/virtual/2024/poster/30059))  
   - DeqIR ([link](https://cvpr.thecvf.com/virtual/2024/poster/31759))  
   - GDP ([link](https://cvpr.thecvf.com/virtual/2023/poster/22095))

### Questions
Please see my concerns in Weakness

### Soundness
2

### Presentation
1

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
The paper introduces a multi-scale conditional generative model (MSCGM) aimed at enhancing microscopic image restoration by combining wavelet transforms and a Brownian Bridge diffusion process. The authors leverage multi-scale wavelet transforms to efficiently model low- and high-frequency image components, significantly improving the generation quality and speed of image restoration compared to traditional diffusion models.

### Strengths
1. MSCGM’s wavelet-based decomposition and conditional modeling shows substantial improvements in sampling speed and better reconstruction quality.
2. By adapting the generative approach to frequency characteristics, MSCGM enhances detail in restored images, especially in high-frequency components crucial for microscopy images.
3. The authors presented a new loss function.

### Weaknesses
1. Equation 18 combines multiple objectives—L2 loss, Structural Similarity Index Measure (SSIM), and Wasserstein distance—but the rationale behind each component’s inclusion is not fully explained. Additionally, the roles and relative importance of the scaling parameters λ, ν, and α are unclear. 
2. The training procedure for MSCGM is not explicitly described. Unlike the clear training steps outlined for BBDP, MSCGM lacks a step-by-step description of its training pipeline. 
3. While Table 1 compares MSCGM with other models in terms of PSNR, SSIM, and sampling time, it does not include training time or the number of trainable parameters for each method. Without these metrics, it is challenging to gauge MSCGM’s overall computational cost relative to other approaches. Including such details would provide a more comprehensive view of the model’s efficiency.
4. In Section 4.2, the authors state that FID is considered as an evaluation metric. However, this metric is not included in Table 1. As FID is widely used in assessing generative models for image quality, its inclusion would offer further insights into MSCGM’s performance in distributional similarity to real images.
5. Equations from 4 to 15 are borrowed from BBDP paper. It is better to include them under the Preliminaries section.

### Questions
1. Could the authors provide more detailed explanations regarding the choice and role of each loss term in Equation 18 and explain how they determined the relative weighting (λ, ν, α values) between the terms.
2. Could the authors provide a comparison of training time and the number of training parameters for MSCGM versus other models?
3. Could the authors to provide a detailed algorithm or pseudocode for MSCGM training, similar to what they provided for BBDP Algorithm.

### Soundness
3

### Presentation
3

### Contribution
2
