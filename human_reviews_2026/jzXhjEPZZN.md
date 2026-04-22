# High-Fidelity Synthetic Transmission Electron Microscopy Image Generation Using Diffusion Probabilistic Models for Data-Limited Semiconductor Metrology

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Advanced semiconductor nodes have significantly increased the demand for Transmission Electron Microscopy (TEM) characterization, posing unprecedented challenges for metrology extraction and defect inspection due to device complexity and shrinking critical-dimension (CD). However, the destructive nature of TEM sample preparation, combined with time-intensive imaging, high acquisition costs, and reproducibility issues, severely limits the availability of diverse datasets required for conventional experimental analysis and, consequently, for training machine learning (ML) models. As a result, artificial intelligence-based synthetic data generation and augmentation have become essential in semiconductor TEM research. Existing generative approaches often fail to capture the complex noise characteristics, surface features, and stochastic variations present in real TEM images, which are critical for accurate semiconductor metrology evaluation. In this research, we present a novel generative framework utilizing Denoising Diffusion Probabilistic Models (DDPMs) specifically designed for synthetic TEM image generation under extreme data scarcity. Our approach employs a progressive patch-based training strategy that scales from low-resolution patches to full-resolution images, enabling from-scratch model training with datasets containing as few as 15 images. We integrate a custom adaptation of the TrivialAugment (TA) algorithm, incorporate domain transfer for cross-process compatibility, and apply super-resolution-enabling inpainting techniques alongside classifier guidance. This framework culminates in full-image training, generating coherent high-resolution TEM images that preserve global structural and spatial relationships essential for analyzing large-scale device structures, in compliance with real FAB semiconductor metrology requirements. Our synthetic images are visually indistinguishable from real TEM data, with a high structural similarity index (MS-SSIM > 0.94) confirming their high-fidelity reproduction, as validated by domain experts. The generated synthetic datasets aim to facilitate robust training of downstream ML models for defect detection, grain and phase boundary segmentation, and metrology applications, addressing critical data availability bottlenecks in advanced semiconductor manufacturing while preserving the statistical and physical properties of scarce real TEM imaging data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a diffusion-based generative framework for high-fidelity synthetic Transmission Electron Microscopy (TEM) image generation under extreme data scarcity in advanced semiconductor manufacturing. The method employs a patch-based progressive training strategy for denoising diffusion probabilistic models (DDPMs), starting from low-resolution patches and scaling up to full-resolution images using as few as like 15 real samples. Experimental results demonstrate visually realistic synthetic images with high structural similarity (MS-SSIM > 0.94) to real data, aiming to support downstream machine learning tasks such as defect detection, metrology, and segmentation in semiconductor process analysis.

### Strengths
(1) The paper tackles a novel and practically important problem—synthetic TEM image generation for semiconductor metrology under extreme data scarcity. The adaptation of diffusion probabilistic models (DDPMs) to this specialized domain, along with a patch-based progressive training strategy, represents a creative methodological contribution.

(2) The technical design is well-motivated and leverages multiple complementary components (progressive patch-based training, customized data augment, classifier guidance, domain transfer, and inpainting). Experimental results demonstrate strong structural similarity (MS-SSIM > 0.94) between synthetic and real TEM data.

(3) The manuscript is generally well-organized and readable, with clear descriptions of model architecture, training process, and evaluation methodology

### Weaknesses
(1) Although the paper claims high structural similarity, the evaluation primarily relies on perceptual and pixel-wise metrics. There is no quantitative validation showing how the synthetic data actually improves downstream ML tasks (e.g., defect detection or segmentation performance). Demonstrating this would substantively strengthen the paper’s impact.

(2) The experimental evaluation is weak. The study does not include a baseline comparison against other state-of-the-art image synthesis methods. Without this, it is difficult to assess whether the proposed framework truly offers a performance advantage in fidelity, which is the biggest problem for this paper.

(3) Some modules mentioned in the paper, such as domain transfer, classifier guidance, and inpainting, are escribed in detail but not rigorously validated. For instance, domain transfer results are shown qualitatively without quantitative or expert-based verification.

(4) The manuscript states that synthetic images are “indistinguishable” from real ones and “comply with FAB metrology requirements,” but this claim relies on subjective expert evaluation and lacks quantitative metrology metrics.

(5) The training and evaluation are performed on as few as 15 TEM images, all from a single domain. While this supports the “data-scarce” claim, it also raises concerns about over-fitting and generalizability.

### Questions
The following are my questions and concerns:

(1) It remains unclear how overfitting is mitigated when training diffusion models “from scratch” on as few as 15 TEM images. The dataset contains only 15 images, which is extremely small for training a DDPM. How do the authors prevent overfitting or mode collapse? Is the generated diversity statistically meaningful given such limited input data?

(2) motivates the framework as improving defect detection and metrology workflows, yet offers no preview of quantitative downstream improvements. It would strengthen the contribution to show even preliminary evidence that synthetic data enhances downstream ML model accuracy or robustness.

(3) Many listed contributions (e.g., TrivialAugment usage, domain transfer, inpainting) appear to be adaptations of existing diffusion model capabilities rather than new algorithmic innovations, and some of them are not validated well. 

(4) The rejection criteria for extreme augmentation values are not well defined. How are thresholds determined, and are they based on physical realism or empirical inspection?

(5) The paper reports only absolute similarity metrics without comparing against any baseline generative models (e.g., GANs, VAEs, or diffusion without progressive training). Without such baselines, it is unclear whether the proposed method offers real improvement.

(6) In section 5.3, the perceptual test reports only a binary outcome (experts could not distinguish) rather than quantitative accuracy, confidence scores, or inter-rater agreement. And only 15 synthetic samples were used. Were these randomly selected or cherry-picked from the “best” MS-SSIM subset? Selection bias could inflate perceived realism.

(7) The section 5.4 claims that such characteristics may be beneficial for downstream applications such as grain or layer boundary segmentation and defect classification, but this is speculative. Was this empirically tested on a downstream task?

(8) While the proposed domain-transfer mechanism is conceptually interesting and potentially valuable, the section lacks any meaningful empirical or numerical validation, also the classifier guidance and  inpainting.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper explores the training of a Denoising Diffusion Probabilistic Model (DDPM) using an extremely limited dataset of only 15 TEM images in the context of semiconductor metrology. The key contribution lies in a novel, iterative training strategy that progressively increases the resolution of image patches over time, enabling the corresponding U-Net layers to be trained in a stepwise manner.

### Strengths
- The paper presents a clear and well-motivated problem statement, effectively highlighting the challenges of data scarcity in semiconductor metrology.
- Training DDPMs from scratch with as few as 15 images is an extremely challenging task; proposing a feasible approach to this problem represents a potentially significant contribution.
- The integration of multiple established diffusion techniques, such as inpainting, domain transfer and classifier guidance, demonstrates solid understanding and thoughtful use of existing methodologies.
- The iterative patch-based training strategy, which progressively increases patch resolution in alignment with U-Net depth, is an interesting and conceptually sound idea that could inspire further research in resolution-aware model training.

### Weaknesses
- The paper lacks ablation studies for several key methodological choices:
    - The decision to operate at the patch level rather than the full image level (as mentioned in l.212) is not experimentally validated or quantitatively analyzed.
    - The proposed progressive training strategy is not ablated to isolate its individual contribution.
    - There is no proper comparison to relevant baselines, including the aforementioned design alternatives.
- The core contribution (the patch-based iterative training approach) is never rigorously ablated or compared against simpler setups. While the authors implement several established diffusion techniques, their validity and contribution to performance (e.g., classifier guidance or inpainting) are not quantitatively assessed.
- There is a high risk of overfitting given the extremely small training set. This concern is supported by the reported metrics in Table 2, where minimal visual differences between generated and real images could reflect memorization or reconstruction artifacts rather than genuine generalization. How do the authors make sure that this does not happen?
- Table 1, which is intended to present hyperparameters, is incomplete—basic parameters such as learning rate and batch size are missing, making reproducibility difficult.
- In l.431, the authors state: “We adapt the RePaint methodology (Lugmayr et al., 2022)”—however, it is unclear whether this constitutes a novel adaptation or simply a reimplementation of the original method.
- The scope of the paper may not align well with ICLR’s focus on representation learning, as the work primarily presents an application-oriented diffusion setup rather than new insights into representation or learning mechanisms.
- Although several diffusion techniques (inpainting, classifier guidance, domain transfer) are incorporated, no quantitative evaluation of their impact or effectiveness is provided.
- The authors claim that the generated images could benefit downstream applications such as grain or layer boundary segmentation and defect classification (l.362), yet this assertion remains untested.
- The description of the DEVICE-TEM dataset (particularly regarding the selection of “100 high-quality samples” based on MS-SSIM across multiple seeds, l. 322) implies a selective filtering process of generated outputs, which raises questions about the objectivity and consistency of evaluation.
- The statement “maximize data utilization while preserving spatial coherence” (l.212) is unclear and needs elaboration, particularly regarding what “spatial coherence” refers to in the patch-based training context.

### Questions
- Table 2: Could the authors clarify what is meant by “DEVICE-TEM PSNR-sorted subset”? What subset is being referred to, and how was it selected?
- The authors mention that “JPEG compression applied to our training data introduces artifacts that influence pixel-wise comparisons” (l.308). Why was JPEG compression used in the first place, given its potential to degrade high-frequency details critical in TEM images?
- Regarding l.312 (“Although the best synthetic samples from the subset closely resemble the original images, they maintain subtle variations rather than being pixel-wise replicas, see Fig. 3.”): How do the authors ensure sufficient diversity in the generated samples and prevent overfitting or mode collapse? Would training a downstream model on the generated data help validate this claim?
- In Section 5.2 (“Structural Similarity and Inference Time”), what is the specific research objective? The results presented appear to confirm well-known trade-offs rather than provide novel insights.
- How does training dataset size affect the method’s performance? Have the authors tested the approach with fewer or more images to assess scalability or robustness?
- What happens if the model is fine-tuned from pretrained weights rather than trained entirely from scratch? Would this improve stability or reduce overfitting?
- In Section 4.2 (Data Augmentation), why are the augmentations not applied on-the-fly during training? Was this a technical limitation or a deliberate design choice?
- The description of the augmentation process — “grayscale-specific modifications and rejection criteria for extreme augmentation values and hash-based duplicate filtering…” — is vague. Could the authors provide more technical detail on how these steps are implemented and how they affect the data distribution (e.g., thresholds, parameters, or examples of rejected augmentations)?
- How are the training patches selected? Are they generated using a simple sliding window, or is a more sophisticated sampling strategy employed?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study proposes a novel generative framework that utilizes a denoising diffusion probability model (DDPM) specifically designed for generating synthetic TEM images in situations of extreme data scarcity. The authors' approach employs a progressive, patch-based training strategy that scales from low-resolution patches to full-resolution images, enabling the model to be trained from scratch using a dataset containing only 15 images.

### Strengths
The subject matter is relatively novel, and improvements to existing methods have been considered.

### Weaknesses
The experiments are insufficient; larger-scale experiments and more in-depth performance analysis could be considered.

### Questions
Are you considering conducting experiments with more cases?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This research presents a progressive patch-based training framework using Denoising Diffusion Probabilistic Models (DDPMs) that successfully generates high-fidelity 1024×1024 resolution synthetic semiconductor TEM images from scratch with as few as 15 training images under extreme data scarcity. Through customized data augmentation, domain transfer, and inpainting techniques, the generated synthetic images achieve MS-SSIM >0.94 structural similarity scores and are visually indistinguishable from real TEM data as validated by domain experts. This approach effectively addresses the critical data scarcity bottleneck in advanced semiconductor manufacturing caused by the destructive and costly nature of TEM acquisition, providing reliable training data augmentation for downstream machine learning tasks including defect detection, semantic segmentation, and metrology applications in 2nm and beyond technology nodes.

### Strengths
1. It achieves high-fidelity TEM synthesis from extremely small datasets via progressive patch-to-full-image diffusion training.
2. The realism is rigorously validated—strong MS-SSIM/PSNR metrics and expert blind tests show synthetic images are hard to distinguish from real ones.
3. Its toolkit (domain transfer, classifier-guided control, and RePaint-style inpainting) makes the method versatile for downstream metrology, segmentation, and defect detection.

### Weaknesses
1. This paper lacks an overall structural diagram to explain the main methods.  The incremental "part-to-whole" training approach makes it difficult to determine whether a continuously expanding model or multiple independent models for different stages are being used, and there is no unified architectural diagram to illustrate this.

2. Your encoder-freezing strategy is presented as a key trick but lacks ablation or sensitivity analyses to justify its benefit and chosen freeze ratio.

3. Domain-transfer and classifier-guided sampling are underspecified (e.g., how to pick T', classifier design, and robustness), limiting reproducibility and practical adoption.

4. Your writing organization also has some weaknesses, particularly in areas like domain transfer/classifier guidance/inpainting, which are presented as both "method extensions" and interspersed demonstrations in the experimental section. They are not clearly separated and are hard to follow.

### Questions
1. Could you provide a detailed cost calculation and implementation report?
2. Why freeze the encoder for “half of the high-res stage”? Can you show ablations for 0/25/50% freeze ratios and their effect on convergence and stability.
3. Where are attention blocks inserted at each scale, and how do channel widths/depth change per stage?
4. Beyond MS-SSIM/PSNR/LPIPS and expert Turing tests, do synthetic images improve downstream tasks (segmentation, metrology, defect detection) on real data?

### Soundness
2

### Presentation
2

### Contribution
2
