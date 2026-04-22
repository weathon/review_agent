# Frequency-Balanced Retinal Representation Learning with Mutual Information Regularization

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 4

## Abstract
We propose a frequency-oriented perspective on retinal representation learning by analyzing masked autoencoders (MAE) through the lens of spatial frequency. Our analysis shows that MAE favors low-frequency content while under-encoding diagnostically critical high-frequency structures in retinal images. Because retinal pathology often manifests in high-frequency detail, this bias limits diagnostic performance and motivates frequency-balanced representations. Within a mutual-information (MI) formulation of MAE, we introduce the Frequency-Balanced Retinal Masked Autoencoder (RetMAE), which augments the reconstruction objective with a MI regularizer that suppresses low-frequency redundancy and accentuates clinically salient high-frequency information. Without altering the architecture, RetMAE learns frequency-balanced features that surpass those of MAE-based retinal encoders in both quantitative and qualitative evaluations. These results suggest that a frequency-oriented view provides a principled foundation for future advances in ophthalmic modeling, offering new insight into how MAE’s reconstruction objective amplifies ViT’s low-pass tendencies in spatially heterogeneous retinal images and enabling a simple MI-based correction that improves retinal encoders.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a frequency-balanced masked autoencoder framework (RetMAE) that enhances retinal representation learning by introducing a high-frequency mutual information regularizer to emphasize clinically critical high-frequency structures while suppressing redundant low-frequency content.

### Strengths
This paper does present a clear motivation that high-frequency structures are clinically important in retinal imaging, and it attempts to incorporate this insight through a mutual-information-based regularizer.  And the presentation is good.

### Weaknesses
(1) According to Figure 2, the loss L_hmi proposed in this paper and the loss L_rec in the original MAE appear to optimize the latent feature in two fundamentally different directions. Specifically, L_rec aims to optimize 𝑧 such that it can reconstruct the full Image from Image_mask1. In contrast, L_hmi optimizes 𝑧 to enable the reconstruction (or transformation) of Image_mask2 from Image_mask1, where mask2 is generated through the high-frequency masking strategy proposed by the authors. This is clearly contradictory and constitutes the most significant issue.
(2) Were all evaluations in Table 1 conducted on the standard MAE model? The results shown in this table seem to indicate that the standard MAE already has a strong ability to represent high-frequency information, which contradicts the authors’ claim of “under-encoding high-frequency diagnostic structures.” For example, in the T_low row of Table 1, even after masking 25% of high-frequency information, the CKA value remains as high as 0.990, indicating that the model still retains stable reconstruction capability for high-frequency components. Conversely, in the T_high row, when high-frequency information is used as input, the model shows low CKA because it cannot reconstruct the full image, which is expected. However, the AUROC increases, demonstrating that high-frequency information is highly discriminative; when low-frequency, lesion-irrelevant content is removed, the model’s prediction accuracy improves. Therefore, the evidence presented in Table 1 may actually support the importance of high-frequency features rather than demonstrating the insufficiency of MAE in encoding them. I believe the authors have not conducted a sufficiently thorough investigation in this aspect.
(3) As highlighted in Comment (1), there is a potential conflict between the two loss terms. The authors should explicitly report the training weights assigned to each loss or provide a sensitivity analysis (e.g., a performance graph under different loss weight configurations) to demonstrate the impact of the loss balance on model performance.
(4) The innovation of the proposed method appears to be limited, as it essentially adds a frequency-based loss on top of MAE, while the use of high-frequency representations to capture lesion-related features in retinal images has already been explored in numerous prior studies.

If the authors can provide a clear theoretical justification or empirical evidence resolving the contradictions I raised—particularly regarding the compatibility of the two loss objectives and the interpretation of Table 1—I am willing to reconsider my rating and increase my score accordingly.

### Questions
Were all evaluations in Table 1 conducted on the standard MAE model?

### Soundness
1

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
This paper focuses on the high-frequency information in fundus images that is clinically relevant. The authors introduced an information-theoretic auxiliary supervision into the MAE pretraining paradigm to guide the encoder toward clinically important regions, without requiring architectural modifications. The overall logic—from problem identification to the proposed solution and experimental validation—is clear and well-structured.

### Strengths
1. The paper presents a clear and coherent logical flow from motivation and problem formulation to analysis, proposed solution, and experimental validation, making it easy to follow and understand.

2. It offers new and interesting insights into model training for ophthalmic applications.

3. The theoretical derivation of the proposed method is sound, and the approach itself is easy to reproduce.

### Weaknesses
1. Limited novelty. The inspiration of this work appears to be directly derived from Huang, Tao, et al. “Learning Mask Invariant Mutual Information for Masked Image Modeling.” arXiv preprint arXiv:2502.19718 (2025). Although this paper is cited, I would still like the authors to explicitly clarify which parts of the current method are independently proposed.

2. I have concerns regarding the generalizability of the proposed approach.
(i) The low-pass filtering property originates from the ViT architecture itself, and this phenomenon is not unique to MAE.
(ii) The application scenario in this work is limited to color fundus photography. From the perspective of developing a robust retinal foundation model, MAE is not the only viable paradigm even within the image modality. For example, VisionFM, which follows the iBOT framework, also builds a powerful image encoder. From the standpoint of understanding the mechanism of MAE, this paper does not provide new insights. The authors need to further elaborate on the substantive contribution of their work.

3. The performance evidence is limited. The chosen downstream tasks are relatively few and of low difficulty (e.g., binary classification of DR, glaucoma, and AMD). Considering that the motivation of this work focuses on clinically relevant high-frequency lesions, the authors are encouraged to validate their method on more challenging tasks to substantiate its claimed clinical value.

### Questions
1. This paper introduces an additional high-frequency contextual feature constraint into the latent space of MAE. Some previous studies have imposed constraints directly on the masking strategy (e.g., image-entropy-based masking). I encourage the authors to discuss this line of work to further highlight the value of their proposed method.

2. According to Figure 3, the performance of RetMAE appears to saturate after pretraining on approximately 12.8k images. Increasing the data size beyond this point seems to yield no significant improvement. The authors attribute this to saturation of model capacity and computation, which is a reasonable explanation. However, given that 12.8k is far smaller than the typical data scale used for foundation model pretraining—and that the employed encoder architecture has been shown in other domains to effectively utilize much larger datasets—this phenomenon remains concerning. The authors should provide a more convincing explanation for this observation.

3. Not all retinal lesions exhibit high-frequency characteristics—for example, large hemorrhages or retinal detachments. I would like to see visualizations of such cases to better understand how the proposed method behaves under these conditions.

### Soundness
3

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
3

### Summary
The presented work postulates that there is a discrepancy between the most salient visual features in medical images and those learned by current representation learning techniques. In particular, the authors claim that the prevalent masked autoencoder disregards high-frequency features related to retinal pathology in color fundus photographs. To demonstrate this, they rank image patches by the amount of high-frequency content. Subsequently, they show that patches with reduced high-frequency content shape latent space structure while providing substantially less information for downstream tasks. In response to this observation, the authors propose RetMAE, an extension of the loss function of the established masked autoencoder. RetMAE regularizes the latent space by maximizing its mutual information with embeddings of patches with increased high-frequency content. In experiments using four ophthalmological datasets, the authors show that RetMAE outperforms various baselines that rely on a basic masked autoencoder. This effect persists when auxiliary signals, such as text or a pre-trained vision encoder, are included as learning signal, albeit in a diminished capacity.

### Strengths
- The work’s main motivation that current pre-training paradigms for vision transformers result in suboptimal feature extractors when applied to medical images is very interesting. The authors convincingly support this hypothesis in a set of initial experiments (Figure 1, Section 4, Supplementary Material A1), showing that salient image features differ in natural and medical images, and that masked image modeling has an inductive bias towards low-frequency features.

- The provided solution to this problem is theoretically well founded and experimentally shown to improve performance. As such, it has the potential to benefit the large scientific community in the field of medical image analysis.

-  The clinical application of ophthalmological image analysis is very well selected. Many retinal diseases manifest as small pathologies, resulting in high-frequency image features. Furthermore, there are several prominent works that have used large-scale pre-training of masked autoencoders to derive ophthalmological foundation models.

- The paper is clearly structured, nicely illustrated and generally well written. Additionally, the authors include extensive supplementary material that provides in-depth technical detail about the method and experimental setup.

### Weaknesses
- The motivating hypothesis that medical images contain more salient high-frequency image features is only explored for color fundus photographs. The importance and reach of the work would substantially increase if similar findings were shown for other types of data. Similarly, the efficiency of the proposed solution is only demonstrated for color fundus photographs so that it is unclear whether the proposed method seamlessly translates to other settings or requires extensive hyperparameter tuning for both the extraction of high-frequency information and the loss weighting.

- The proposed solution is highly complex. In particular, it relies on estimation of mutual information via an Donsker-Varadhan estimator, which is known to numerically unstable (Belghazi, Mohamed Ishmael, et al. "Mutual information neural estimation." International conference on machine learning. PMLR, 2018). I could envision that conceptionally much simpler solutions exist that emphasize high-frequency features. For example, one could adjust the masking scheme to prioritize patches with increased high-frequency content (similar to Xie, Jiahao, et al. "Masked Frequency Modeling for Self-Supervised Visual Pre-Training." The Eleventh International Conference on Learning Representations). Alternatively, one could provide the high-pass-filtered image as additional input (Wang, Wenxuan, et al. "Fremim: Fourier transform meets masked image modeling for medical image segmentation." Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2024). The authors should discuss the aforementioned works in more detail and include them as baselines.

### Questions
- At the moment, the performance of the proposed method is only quantified via linear probing using the latent representations. I believe that full fine-tuning should also be conducted considering that the ultimate downstream performance matters most in applied domains such as medical image analysis.

- Considering that most ophthalmological foundation models make their training code and weights public, I believe that the authors should strongly consider doing the same.

- Several core concepts of the paper are only very briefly introduced or require consultation of the supplementary material. I suggest that the section on the interpretation of the masked image modeling objective through the lens of a Lagragian is slightly extended so that it can be understood without consulting previous works by Huang et al. and Tishby and Zaslasky. Similarly, the frequency score calculation should be briefly introduced in the main manuscript considering its vital importance, instead of only being introduced in the supplementary material.

- Additionally, I struggled with the notation on several occasions. Already in the very first mathematical paragraph, the variable $N$ is overloaded, $D$ not introduced, and mutual information $I$ is not defined. Later, the use of $X$ varies to signal whether it denotes input or decoded image tokens. The authors should carefully parse the manuscript once again, aiming to improve the clarity of its mathematical passages.

- On a minor note, the acronym CKA is not introduced at its first appearance in the introduction section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a frequency-aware masked autoencoder (MAE) for unsupervised pretraining on retinal images, called RetMAE. This is accomplished by including a high-frequency regularization term in the objective function that reduces low-frequency redundancy. The authors main claim is that the diagnostic information in retina images are encoded in high frequencies, and thus better representing these areas yields better downstream accuracy. There is a section on evaluating the frequency bias in MAE representations applied to fundus images in which the paper presents centered kernel alignment (CKA) and linear probing as evidence. The experimental section utilizes five publicly available datasets and compares against two other MAE-based approaches as well as a vision-language baseline. 

Although the paper doesn't contribute a significant new algorithmic framework, its approach in providing frequency-based context latents to guide the representation learning paradigm for applications in which frequency bias is an impediment could be a valuable contribution. There are, however, a few areas of both theory and practice that need clarification. 

On theory:

1- What is the purpose of using a variational autoencoder with a fixed variance Gaussian mapping? From theorem 1, it reduces the reconstruction error to minimizing the conditional in eq 2. However, it is not clear if this enforced constraint is warranted. Is this constraint enforcing any aspect of the regularization framework?
2- The choice of using $\mathcal L_{MINE}$ as the context-alignment training objective is not quite clear. In other words, why does estimating MINE maximize the conditional?

On Application:
1- Does this framework extend beyond retinal fundus images? Could other factors than frequency bias be incorporated in the regularization objective?
2- How much computational complexity is added to the problem by incorporating the proposed high-frequency regularization?
3- How does the pretrained encoder fare in a formal classification tasks rather than the employed linear probing?

Additional suggestions:

1) use the figure in appendix A instead of Figure 1 in the main paper. The figure from the appendix better justifies the frequency bias of retinal fundus images as compared to natural images, e.g. ImageNet.
2) CKA and its relevance to the claimed frequency bias should be explained more clearly.

### Strengths
The approach in utilizing a bias term as regularization to improve representation learning in certain application is interesting.
This approach could be potentially significant for applications that are not based on natural images.

### Weaknesses
Better discussion is needed to connect the theoretical aspect of the work (MI) with the practical tools utilized (MINE estimation).

### Questions
Provided in the summary.

### Soundness
3

### Presentation
3

### Contribution
2
