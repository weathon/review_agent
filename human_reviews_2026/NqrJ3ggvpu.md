# MagicStain: High-Fidelity Pathology Image Virtual Staining via Guided Single-Step Diffusion

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Virtual staining, which leverages computational methods to generate different stain styles from a pathology image of an already stained tissue section, offers a cost-effective alternative to chemical multiple staining. Despite extensive research based on generative adversarial networks (GANs) and diffusion models, achieving high-fidelity, high-quality, and computationally efficient results remains a significant challenge for virtual staining methods. While diffusion-based approaches typically produce more photorealistic images than GAN-based counterparts through multi-step sampling, this comes at the cost of high computational overhead and inference latency. In this paper, we propose MagicStain, a novel single-step diffusion model tailored for generating high-resolution virtual stains. Specifically, we adapt a pretrained single-step diffusion model to enable efficient virtual staining. By introducing pathology priors from a pretrained pathology-specific vision language model and integrating pathology- and structure-consistency losses on both the original images and the H-channel, MagicStain achieves high-fidelity and high-quality generations. To address the limitations of single-step diffusion models in high-resolution virtual staining, we further propose a two-stage progressive training strategy that enables high-resolution adaptation with low training cost. Extensive experiments on three virtual staining datasets, each involving translation between different staining dyes/biomarkers, demonstrate the superiority of MagicStain in terms of fidelity, visual quality, and computational efficiency compared to existing methods. Our code and trained models will be released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose a new methodology for generating virtual stains conditioned on either H&E or frozen FFPE images. The approach distills multi-step diffusion models into a single-step variant to increase inference speed, incorporates pathology-encoded information as a prior using PathClip embeddings, and introduces a hematoxylin channel loss to enforce structural constraints during generation. The model is evaluated on three different datasets with relevant ablation studies.

### Strengths
In my opinion, the paper has several strengths:

1. The use of diffusion distillation methods to improve convergence and generation efficiency is both technically sound and impactful.

2. The evaluation across three different datasets, including an out-of-distribution dataset without any fine-tuning, is particularly interesting and relevant.

### Weaknesses
There are several weaknesses of the papers:

1. First is novelty. I have some concerns about novelty of the proposed methods and Choice of losses.

(a) Using histopathology pre-trained models for generation has been explored before [1,2], perhaps with different encoders, and has been shown to perform well. Therefore, claiming the use of PathClip embeddings as a novelty seems questionable. Additionally, the paper does not experimentally or textually justify why PathClip would outperform vision-only encoders used in previous works such as UNI or Virchow. What is the specific reason that PathClip embeddings should improve generation? Is there evidence that these embeddings provide a tangible advantage for virtual staining tasks?

(b) Although the H-channel loss has been used before[3], it seems primarily relevant for FFPE-to-H&E staining, where the goal is to accurately reconstruct the hematoxylin channel. For other tasks, such as H&E-to-IHC translation, the focus should be on the DAB channel rather than the H-channel. If the model emphasizes the H-channel too strongly, it could produce inaccurate results for the DAB channel, which is undesirable. Therefore, the rationale for adding this loss in the broader context appears flawed. 

(c) Progressive training may be useful for models with slow inference times, but the proposed model is single-step, so its relevance is unclear. Reducing WSI generation time from five minutes to two or three minutes is unlikely to be significant for pathology applications. Progressive training would be more justified for multi-step diffusion models, where it could reduce inference times from hours to minutes. Therefore, I find the rationale for applying progressive training here neither logically sound nor practically relevant.


(2). No reasoning is provided for using the Wilcoxon signed-rank test. Why not a simple t-test? Which assumption of the t-test is violated here that necessitates a different statistical test?

(3) No relevant downstream application is provided. The proposed downstream task is unfamiliar to me— is it a standard in the field, or are the authors proposing a new task? If it is new, they should provide references to support its validity. If not, it is unclear how we can trust the results of the downstream evaluation.

(4) Manual evaluation is performed only on H&E WSI images; no manual evaluation on IHC images is reported, even in the appendix.


[1] Ho, Man M., Shikha Dubey, Yosep Chong, Beatrice Knudsen, and Tolga Tasdizen. "F2fldm: Latent diffusion models with histopathology pre-trained embeddings for unpaired frozen section to ffpe translation." In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 4382-4391. IEEE, 2025.

[2] Yang, Hao, JianYu Wu, Run Fang, Xuelian Zhao, Yuan Ji, Zhiyu Chen, Guibin He, Junceng Guo, Yang Liu, and Xinhua Zeng. "Cross-channel Perception Learning for H&E-to-IHC Virtual Staining." arXiv preprint arXiv:2506.07559 (2025).

[3] Peng, Qiong, Weiping Lin, Yihuang Hu, Ailisi Bao, Chenyu Lian, Weiwei Wei, Meng Yue, Jingxin Liu, Lequan Yu, and Liansheng Wang. "Advancing H&E-to-IHC virtual staining with task-specific domain knowledge for HER2 scoring." In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 3-13. Cham: Springer Nature Switzerland, 2024.

### Questions
I have several question some of which i stated in the Weaknesses above, other questions and suggestions that i have are:

1. You have reported that models are trained for 15-20 epochs. But while reading other papers, pix2pix and CycleGAN models are generally trained for 200 epochs, why this discrepancy ?

2. Comparisons to [1] and [2] are missing, even though both were trained under the same settings as the proposed method. Other papers in the literature also achieve reasonable performance, making such comparisons important.

3. Do you plan to release the datasets used in this study publicly? If not, how can others access the data or verify that the results are reproducible? As no public datasets are used in the paper, reproducibility will be hard if dataset access is limited. Or you need to use at-least one public dataset.


[1] Li, Fangda, Zhiqiang Hu, Wen Chen, and Avinash Kak. "Adaptive supervised patchnce loss for learning h&e-to-ihc stain translation with inconsistent groundtruth image pairs." In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 632-641. Cham: Springer Nature Switzerland, 2023.

[2] Liu, Shengjie, Chuang Zhu, Feng Xu, Xinyu Jia, Zhongyue Shi, and Mulan Jin. "Bci: Breast cancer immunohistochemical image generation through pyramid pix2pix." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1815-1824. 2022.

### Soundness
3

### Presentation
3

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
This paper addresses the challenge of virtual staining in histopathology. To this end, the authors propose a one-step diffusion-based framework. To mitigate the domain gap in image synthesis, a retrained VLM model is incorporated for two main purposes: (1) to provide histopathological priors extracted from query images, and (2) to enforce morphological consistency between synthesized and ground truth images under full supervision. Furthermore, a two-stage progressive denoising strategy is introduced to enable high-resolution image synthesis.

### Strengths
1. The manuscript is clearly written and easy to follow. The motivations behind the proposed mechanisms for the virtual staining task are well justified and logically presented.
2. Extensive experiments on three datasets are conducted, and the reported results demonstrate strong overall performance.

### Weaknesses
1. The proposed method adopts one-step diffusion models for virtual staining, a specific case of pix2pix-style translation. To overcome inherent limitations, several learning mechanisms, such as introducing image priors via a pre-trained VLM, adversarial training, and 2-stage progressive training, are integrated into the baseline. While the results are promising, the overall contribution appears to be primarily an effective engineering integration of established techniques for a well-defined problem. Hence, the theoretical novelty seems limited, making this work potentially better suited for application-oriented venues like MICCAI.
2. In Figure 2, the output of the VAE is depicted as images, which is misleading since VAE latents should be numerical representations rather than image outputs. Updating Figure 2 to improve accuracy is recommended.
3. It appears that the VAE remains frozen during the entire training process. It is unclear whether the pre-trained VAE is sufficient to capture histopathological features without fine-tuning. As prior studies have shown that VAE quality strongly affects synthesis outcomes, it would be valuable to compare results with and without VAE fine-tuning.
4. Although the paper introduces a two-stage training strategy to support high-resolution synthesis for WSI, the generated images (1024 resolution) are still much smaller than real WSIs. If applied to full-scale WSIs, how many stages would be required? While the approach is promising, scalability remains an open concern.
5. The images in Figure 3 are too small to effectively demonstrate visual results, which limits their interpretability.

### Questions
I would like to hear feedback for the following comments:
1. Prior studies have shown that VAE quality strongly affects synthesis outcomes, why did authors still decide to freeze the VAE and only finetune the DM component? Is there any evidence to support this selection?
2. Although the paper introduces a two-stage training strategy to support high-resolution synthesis for WSI, the generated images (1024 resolution) are still much smaller than real WSIs. If applied to full-scale WSIs, how many stages would be required? While the approach is promising, scalability remains an open concern.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces MagicStain, a single-step diffusion framework for virtual staining in digital pathology. Virtual staining aims to computationally translate an image of a tissue section from one stain modality (e.g., H&E) to another (e.g., IHC, Masson’s trichrome), providing a cost-effective and non-destructive alternative to chemical staining.
Traditional GAN-based models are efficient but often lack fidelity, while diffusion models produce more photorealistic results at the cost of heavy sampling (dozens of steps) and long inference times. To bridge this trade-off, MagicStain builds on a pretrained single-step diffusion backbone (e.g., SD-Turbo or similar latent diffusion variant) and adapts it for the pathology domain via three key contributions:
1. Pathology-aware conditioning using a pretrained vision–language model (PathCLIP): Introduces semantic priors reflecting cellular and histological structures, improving the alignment between input and generated stain styles.
2. Pathology- and structure-consistency losses: Applies supervision in both RGB and H-channel (hematoxylin channel) space to preserve nuclei morphology and tissue micro-architecture during stain translation.
3. Progressive high-resolution adaptation: Proposes a two-stage training pipeline (512 → 1024 px) using LoRA fine-tuning to efficiently scale up to gigapixel-level pathology slides while maintaining learned priors.

### Strengths
1. Practical and clinically relevant problem
- The motivation is clear and well grounded with virtual staining, which has direct diagnostic impact.
2. Strong engineering execution 
- The single-step diffusion adaptation is efficient by adapting other well-pretrained models, and yields large speed-ups. 
3. Integration of domain knowledge
- The use of PathCLIP priors and H-channel supervision meaningfully connects pathology semantics with generative modeling.

### Weaknesses
1. Incremental conceptual novelty
- The adaptation of single-step diffusion and domain priors is technically sound but conceptually incremental, as it more tends to applicational work, instead of theorectically insights.

2. Lack of theoretical justification:
- No analysis of why PathCLIP guidance and H-channel losses interact synergistically with diffusion dynamics. The approach appears largely empirical. It will be great to have an ablation that evaluate the integration of PathCLIP embeddings is helping or not.

3. Dependence on pretrained backbones:
-Performance may rely heavily on the pretrained single-step model’s capabilities (e.g., SD-Turbo). The contribution is then more about fine-tuning heuristics than architectural advancement.

### Questions
1. How does MagicStain differ from prior single-step or distilled diffusion models like LDM-Turbo or EfficientDM, beyond domain-specific losses?
2. What is the quantitative speed-accuracy trade-off versus few-step diffusion (e.g., 2–4 steps)? Does single-step truly match their quality, or is there a perceptual compromise?
3. How critical are the PathCLIP priors? Is there any experiments that can show the importance of it? Can the model function without them, or do they primarily anchor the translation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
MagicStain Propose single step diffusion framework adapted from prior art for H&E to IHC image-to-image translation (virtual staining). It adapts SD-Turbo and pix2pix-Turbo to injects PathCLIP pathology priors, adds H-channel (LPIPS and MSE)losses, and uses a two-stage 512 to 1024 progressive training with LoRA for high-res outputs. Experiments on three datasets report top FID/KID (and competitive PSNR/SSIM), external-set generalization, and a downstream win in classification.

### Strengths
1. Speed: As compared to previous methods especially the ones with multiple steps it is much faster and therefore practical to use for clinicians.
2. Well motivated and supported claims: PathCLIP prior and losses on H-channel and expert embeddings target pathology semantics are well motivated. Ablations show each piece helps giving justification for the choices made.
3. Two stage recipe for high resolution: Clear two-stage LoRA adaptation to 1024 with supporting ablations is very helpful. Working on WSI images is more clinically relevant since it follows how pathologists work with WSI (high magnification to low magnification).
4. Strong overall, external validation and downstream task: Overall the evaluation shows better results for the proposed work with downstream classification task where Magic stain wins even as compared to the real IHC images.

### Weaknesses
1. Metric choice not fully pathology-standard. The paper leans on FID/KID (with natural Inception?, please clarify this) and PSNR/SSIM; these do not directly quantify clinical correctness or region-level fidelity in histopathology. Authors acknowledge structural misalignment for SSIM/PSNR but still rely on them. Consider pathology encoders / clinical labelers (e.g., PLIP similarity, cell/lesion segmentation or classification).
2. “Beats Real IHC” on downstream classifier needs careful interpretation and more explanation. Surpassing the classifier trained on real IHC likely reflects cross-cohort domain shift (train on HER2, test on Ext) rather than virtual images exceeding real-image information. Mentioning single sentence about it is not enough. Please analyze shifts and add CI/significance. Also, this should be paired with performance on indomain instead of out of cohort if domain shift is not properly analyzed. 
3. The paper’s novelty largely lies in applying established components to pathology. While the system is well-validated, the methodological contribution appears incremental.
4. Some claims need more evidence: For example on line 321, "Notably, lower FID and KID values suggest better pathology matches, e.g., cancerous status" need more evidence and justification. Lacking domain expert evaluation for clinical accuracy and confidence in the results presented.

### Questions
Q.1 Please adapt the current evaluation for Pathology and domain expert evaluation with strengthen the results provided. 
Q.2. I suggest downweighing/restating some of the claims pointed out in the weakness or providing more evidence.

### Soundness
2

### Presentation
3

### Contribution
2
