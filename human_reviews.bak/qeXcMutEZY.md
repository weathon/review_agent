# Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models Trained on Corrupted Data

- Decision: Accept (Poster)
- Scores: 8, 8, 8, 5, 6

## Abstract
We provide a framework for solving inverse problems with diffusion models learned from linearly corrupted data. Firstly, we extend the Ambient Diffusion framework to enable training directly from measurements corrupted in the Fourier domain. Subsequently, we train diffusion models for MRI with access only to Fourier subsampled multi-coil measurements at acceleration factors R$=2, 4, 6, 8$. Secondly, we propose $\textit{Ambient Diffusion Posterior Sampling}$ (A-DPS), a reconstruction algorithm that leverages generative models pre-trained on one type of corruption (e.g. image inpainting) to perform posterior sampling on measurements from a different forward process (e.g. image blurring). For MRI reconstruction in high acceleration regimes, we observe that A-DPS models trained on subsampled data are better suited to solving inverse problems than models trained on fully sampled data. We also test the efficacy of A-DPS on natural image datasets (CelebA, FFHQ, and AFHQ) and show that A-DPS can sometimes outperform models trained on clean data for several image restoration tasks in both speed and performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this work the authors provide a framework for solving inverse problems with diffusion models that are trained on corrupted data. The authors extend the existing ambient diffusion framework to enable Fourier domain-corrupted measurements and propose ambient diffusion posterior sampling, an algorithm that leverages generative models pre-trained on one type of corruption to solve other inverse problems. The authors show the performance of their method on accelerated MRI reconstruction and on various problems for natural images.

### Strengths
S1) The paper is very well written and provides an appropriate amount of detail. Similarly, all analysis is easy to follow.

S2) The authors include a nice theoretical result in Appendix A.

S3) The quantitative performance of A-DPS is impressive, and the selected baselines offer an easy way to contextualize it (particularly for MRI).

S4) In my opinion, and based on the results, this is the first diffusion-MRI method to truly push the state-of-the-art forward since [1].

[1] Ajil Jalal, Marius Arvinte, Giannis Daras, Eric Price, Alexandros G Dimakis, and Jon Tamir, "Robust
compressed sensing mri with deep generative priors," 2021

### Weaknesses
I find this paper to be very impactful (hence my score). That said, it is very important to me that the following two weaknesses are addressed:

W1) While the experimental performance of A-DPS is impressive, I think that the MRI results suffer from insufficient evaluation with all competitors. NRMSE being the only metric reported in Table 2 is a concern. It would be worthwhile to add some feature-based metrics like LPIPS/DISTS for all methods. You use VGG to compute FID, you can also use VGG for those metrics (since VGG agrees with radiologists' perceptions [1]). Alternatively, you could also use AlexNet as your feature extractor [1]. Either way, I think that it is important to see these results for your method and competitors. The lack of additional metrics holds this paper back. I understand the space concern given the size of Table 2, but even putting these results in an appendix would be useful to provide readers full context and make a convincing statement about A-DPS' superiority.

W2) I am confused by the bolding in Table 2. For methods trained with clean data, no values are bolded. For the R=2 and R=4 sections, A-DPS has equal performance to A-ALD in some metrics. The A-ALD metrics that are equally performant should be bolded for fairness. Please revise this table so that it is consistent.

[1] P. M. Adamson, A. D. Desai, J. Dominic, C. Bluethgen, J. P. Wood, A. B. Syed, R. D. Boutin, K. J. Stevens,
S. Vasanawala, J. M. Pauly, A. S. Chaudhari, and B. Gunel, “Using deep feature distances for evaluating
MR image reconstruction quality,” 2023

### Questions
See weaknesses. Although, I do have one actual question:

Q1) Did you consider training your MRI diffusion model (ambient or other) with the multi-coil data? To be clear: I am not asking you to do this, as it is a fundamental departure from what you have here. That said, I wonder if you tried this. I know that in, e.g., fastMRI, the number of coils is inconsistent, but you can get around this issue with virtual coils [1]. I suspect that the overall performance could be further improved with a multi-coil diffusion process, rather than a coil-combined one.

[1] T. Zhang, J. M. Pauly, S. S. Vasanawala, and M. Lustig, “Coil compression for accelerated imaging with
Cartesian sampling,” 2013

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a posterior sampling method for inverse problems called Ambient Diffusion Posterior Sampling (A-DPS), which utilizes a diffusion model trained only on corrupted images. They modify the existing Ambient Diffusion framework to work for the accelerated MRI problem, allowing the training of an unconditional diffusion model with only subsampled MRI measurements. Then, they integrate the Ambient Diffusion models into the DPS framework to form a posterior sampling approach. The experiments on accelerated MRI and natural image datasets consistently show better recovery performance than the state-of-the-art DPS when the conditioning is highly corrupted but not when the corruption level is low.

### Strengths
- The paper addresses a significant problem in solving inverse problems when ground truth data is unavailable. By only requiring corrupted measurements for training, the method presents an avenue to solve a wider set of inverse problems.
- The framework is novel as it incorporates the benefits of unconditional Ambient Diffusion with the conditional flexibility of DPS. 
- The experiments are adequately extensive and effectively demonstrate the benefits (and shortcomings) of their approach compared to several existing methods.

### Weaknesses
While the general methodology is explained well, the description of particular details could be modified to improve clarity and soundness
- Eq: 2.9: Are these equations correct? It seems to me that it should be $y_{t,\text{train}} = A_{\text{train}}(y_{\text{train}} + \sigma_{t}\eta)$ if $A_{\text{train}}$ is an orthogonal projection matrix or $y_{t,\text{train}} = A_{\text{train}}(x_{\text{train}} + \sigma_{t}\eta)$ if not
- What is the input to the diffusion model $h_\theta$? In Eq 2.10, it takes the additional subsampling matrix $\tilde{P}$. In Eq 2.11, it takes the full forward operator $A_\text{train}$. 
- An intuitive explanation of $h_{\theta}(\tilde{y}\_{t, t_{\text{train}}}, \tilde{P})$ would be helpful for understanding the overall algorithm. Specifically, what does the training accomplish and why can this be directly plugged into Eq. 2.11?
- What is the reason or intuition for why an A-DPS trained on one corruption can generate posterior samples for a different corruption? A good hypothesis could better sell this property of the method.

Minor Points:
- L93-94: This sentence is confusing because it sounds like the Ambient Diffusion framework is "the first framework to solve inverse problems with diffusion models learned from linearly corrupted data". I would reword this to avoid confusion.
- Eq 2.4: It was not clear what $\tilde{A}_\text{train}$ represents. I would explain this matrix first before the equation, so the equation is more interpretable
- L255: Typo with "In that regards the multiplicity..."
- L364: Typo with "Similar to the results in The authors of ..."
- The construction of Fig 3 is confusing with the labeling of "Recon-FS". Since it's vertically displaced, it is easy to misread. I would suggest changing this to "Error" or rotate it vertically so "Recon-FS" is on a single line

### Questions
- Would it work better to train a single network on multiple acceleration rates rather than a separate network for each acceleration rate?
- Were the evaluations done on a single posterior sample for each method? Or do you generate many posterior samples and compute a posterior mean?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a novel method for learning a diffusion model from corrupted data, supported by a theoretical analysis of training the MMSE restoration estimator under such conditions. Empirical results indicate that the learned MMSE restoration estimator can operate similarly to an MMSE denoiser, enabling effective score-based generation and restoration tasks. Overall, this is a highly interesting paper with the potential to open many new research directions. However, certain aspects of the analysis and discussion could benefit from further improvement.

### Strengths
(1) This paper presents a novel framework for learning a ‘diffusion model’ solely from corrupted data, which is especially valuable in the field of medical imaging reconstruction, where only undersampled data is often available. 
(2) A solid theoretical proof for learning MMSE restoration estimator from corrupted data under suitable assumption.
(3) Based on this diffusion model, the authors demonstrate its effectiveness in both generative tasks and solving inverse problems, highlighting the efficiency and potential impact of this approach.

### Weaknesses
(1) This work introduces an intriguing approach to learning a diffusion model from corrupted data. However, the authors’ analysis assumes that the learned MMSE restoration estimator can approximate a learned MMSE denoising estimator to derive a score function for sampling—an assumption that could benefit from further clarification. In approaches like DPS, posterior sampling depends on the score derived from Tweedie’s formula, which is specifically suited for denoisers. In this study, however, the corresponding score function is not discussed in sufficient detail. A discussion of recent approaches such as DRP[1] and SHARP[2], which utilize the MMSE restoration estimator with a modified score function, could enhance the analysis and provide a more fitting framework for this model.

(2) The theoretical approach in this work is compelling and well-suited for training with randomly inpainted images. However, in the MRI setting used here, different training samples may share the auto-calibration signal (ACS) region, potentially leading to an over-representation of corresponding k-space regions in the training data. A more solid approach might involve introducing a diagonal weighting matrix in the loss function to account for these oversampled regions. A similar work discussing this point for MRI reconstruction is [3].

(3) Further explanation to how the FID value is calculated for MRI data will be helpful.

(4) Some minor suggestions regarding the paper's presentation: (1) In Table 3, it would be beneficial to standardize the significant digits for the FID values to improve readability. (2) For Figures 7 and 8, adjusting the color bar could enhance visual clarity and make the figures easier to interpret.

Reference: 
[1] Y. Hu, M. Delbracio, P. Milanfar, and U. S. Kamilov, “A Restoration Network as an Implicit Prior,” Proc. Int. Conf. Learn. Represent. (ICLR 2024) (Vienna, Austria, May 7-11).
[2] Y. Hu, A. Peng, W. Gan, P. Milanfar, M. Delbracio, and U. S. Kamilov, “Stochastic Deep Restoration Priors for Imaging Inverse Problems.”  [https://arxiv.org/abs/2410.02057]
[3] C. Millard and M. Chiew, "A Theoretical Framework for Self-Supervised MR Image Reconstruction Using Sub-Sampling via Variable Density Noisier2Noise," in IEEE Transactions on Computational Imaging, vol. 9, pp. 707-720,

### Questions
The questions have been covered in the weakness part.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper applies the recent advancement in self-supervised learning for diffusion models, named ambient diffusion, to inverse problems. The effectiveness of this approach is shown through its application to MRI reconstruction and image inpainting.

### Strengths
1) The paper represents an early investigation into the application of self-supervised learning for diffusion models in the context of inverse problems.
2) The paper is well-organized, and the presentation is easy to follow.

### Weaknesses
1) Although this paper introduces the first self-supervised diffusion method for inverse problems, the overall novelty is somewhat limited. It appears to be a straightforward integration of existing ambient diffusion with well-known diffusion posterior sampling, with application to MRI reconstruction. While the investigation holds value from a research perspective, the new insight/concept might be limited.

2) In the training phase (as described in Equation 2.10), the network takes \tilde{A} as input, considering one type of imaging artifact. However, during testing (Equation 2.11), the network uses A as input, which features a different imaging artifact. This inconsistency raises concerns about potential suboptimal performance. Could the authors clarify this aspect?

3) There is insufficient discussion regarding why the proposed A-DPS method outperforms FS-DPS, especially at high undersampling rates. Given that FS-DPS uses a diffusion model trained on fully sampled data, the results seem unexpectedly favorable.

4) The manuscript lacks discussions with recent theoretical developments in self-supervised learning for MRI [1,2], which share similar proof strategy with Theorem 1 in this paper.

5) The numerical validation could be improved, with certain experimental setups are not fully explored (see specific questions below). Additionally, the baseline methods for MRI reconstruction cited are well-known but somewhat dated (2019, 2020). Given the extensive literature on MRI reconstruction, these baselines may not represent the current state of the field. Comparing recent alternatives, such as [1-3] and self-supervised or supervised methods in survey [3] and [4], respectively, could improve this study. There is also no comparison between A-OS and other end-to-end methods. Despite that, I fully understand the time constraints of the rebuttal period and do not intend to suggest that the authors include comparisons with new baselines.

I am willing to increase my score if the authors could address my concerns.

[1] Millard, Charles, and Mark Chiew. "A theoretical framework for self-supervised MR image reconstruction using sub-sampling via variable density Noisier2Noise." IEEE transactions on computational imaging (2023).

[2] Gan, Weijie, et al. "Self-supervised deep equilibrium models with theoretical guarantees and applications to MRI reconstruction." IEEE Transactions on Computational Imaging (2023).

[3] Sriram, Anuroop, et al. "End-to-end variational networks for accelerated MRI reconstruction." Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part II 23. Springer International Publishing, 2020.

[4] Akçakaya, Mehmet, et al. "Unsupervised deep learning methods for biological image reconstruction and enhancement: An overview from a signal processing perspective." IEEE Signal Processing Magazine 39.2 (2022): 28-44.

[5] Heckel, Reinhard, et al. "Deep learning for accelerated and robust MRI reconstruction: a review." arXiv preprint arXiv:2404.15692 (2024).

### Questions
1) The further corruption ratio is simply R+1. How does this further corruption ratio impact the performance, says R+2, R+3, etc?
2) Does the supervised DPS share the same network architecture with the proposed method?
3) How does the coil sensitivity map in this training dataset and the L1-wavelet baseline be estimated?
4) What is the time complexity of the methods?
5) What is the actual under sampled ratio considering the sampling mask for different R has the same 20 lines of the auto calibration signal?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper uses ambient diffusion models as a prior for solving ill-posed inverse problems. The ambient diffusion is trained on corrupted data acquired from a linear ill-posed operator. The authors also extend the ambient diffusions to use MRI k-space subsampled measurements as training data. The authors show the proposed framework can outperform DPS trained on clean images when dealing with severely ill-posed inverse problems.

### Strengths
1. The paper is motivated by the challenge of solving inverse problems without access to clean images, a common problem in real-world imaging applications.
2. Experimental results show the proposed framework is effective for severely ill-posed problems, sometimes outperforming priors trained on clean images.

### Weaknesses
1. Despite promising experiments, the novelty of the paper seems limited; the authors primarily extend ambient diffusion to MRI and adapt it within the DPS framework for solving inverse problems.

2. Below Eq. 2.11, the authors state, "We remark that similar to DPS, the proposed algorithm is an approximation to sampling from the true posterior distribution". However, the accuracy of this approximation should be rigorously compared to that of the original DPS, as it relies on estimating the score function from corrupted data.

3. The authors should consider including additional, recent self-supervised learning approaches as baselines, such as the following paper:
Gan, Weijie, et al. "Self-supervised deep equilibrium models with theoretical guarantees and applications to MRI reconstruction." IEEE Transactions on Computational Imaging (2023).

4. The paper needs to be further polished, for example, Eq 2.9 seems incorrect.

### Questions
Please take a look at the comments under weaknesses.

### Soundness
2

### Presentation
4

### Contribution
3
