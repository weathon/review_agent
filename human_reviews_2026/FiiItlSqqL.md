# SuperF: Neural Implicit Fields for Multi-Image Super-Resolution

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
High-resolution imagery is often hindered by limitations in sensor technology, atmospheric conditions, and costs. Such challenges occur in satellite remote sensing, but also with handheld cameras, such as our smartphones. Hence, super-resolution aims to enhance the image resolution algorithmically. Since single-image super-resolution requires to solve an inverse problem, such methods must exploit strong priors, e.g. learned from high-resolution training data, or be constrained by auxiliary data, e.g. by a high-resolution guide from another modality. While qualitatively pleasing, such approaches often lead to "hallucinated" structures that do not match reality. In contrast, multi-image super-resolution (MISR) aims to improve the (optical) resolution by constraining the super-resolution process with multiple views taken with sub-pixel shifts. Here, we propose SuperF, a test-time optimization approach for MISR that leverages coordinate-based neural networks, also called neural fields. Their ability to represent continuous signals with an implicit neural representation (INR) makes them an ideal fit for the MISR task. The key characteristic of our approach is to share an INR for multiple shifted low-resolution frames and to jointly optimize the frame alignment with the INR. Our approach advances related INR baselines, adopted from burst fusion for layer separation, by directly parameterizing the sub-pixel alignment as optimizable affine transformation parameters and by optimizing via a super-sampled coordinate grid that corresponds to the output resolution. Our experiments yield compelling results on simulated bursts of satellite imagery and ground-level images from handheld cameras, with upsampling factors of up to 8. A key advantage of SuperF is that this approach does not rely on any high-resolution training data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed a SuperF, a test-time optimization method for multi-image super-resolution based on implicit neural representations. SuperF designs an improved sub-pixel alignment and continuous representation of the high-resolution signal based on INR. The experimental results demonstrated that proposed method achieve star-of-the-art performance.

### Strengths
A lot of experiments demonstrate the effectiveness of the proposed method and its components. An INR-based method is proposed for multi-image super-resolution.

### Weaknesses
Some detailed descriptions and reasons for the proposed method should be provided. 

More experimental results should be provided for comprehensive comparison. 

The writing of the paper needs improvement.

### Questions
Which dataset’s results are shown in Figure 2. The detailed description should be provided.  

Why choose SatSynthBurst and SyntheticBurst datasets for evaluation? Why build SatSynthBurst from the WorldStrat dataset? Can the WorldStrat dataset be used as an evaluation dataset? Did the previous method also use the same configuration?

In Table 1 and Table 2, what does the value in parentheses mean? The detailed description should be presented in the caption of the Table. The complexity of the proposed method should be provided. 

Why the proposed (MSE) and the proposed (GNLL) have different performance. GNLL-based model has the best results in most cases for the SatSynthBurst dataset. However, MSE-based model has the best results for the SatSynthBurst dataset and large margins. In Figure 5, this phenomenon also occurs. The reason should be provided. What do these results represent?

Why use two loss functions, i.e., mean squared error(MSE) and the Gaussian negative log-likelihood (GNLL)? Do previous methods also use two loss functions? 

The author claims that SuperF is a test-time optimization method. How to demonstrate the test-time optimization for the proposed method? Relevant experiments and analysis should be provided

Why use 4 MLP layers in the INR decoder and 3 MLP layers in the Uncertainty decoder? Corresponding ablation studies should be provided for specific layers.

The compared methods, i.e., Nametal.(2022), Lafenetreetal.(2023), they were two years ago, the latest methods should be discussed and compared

Some typos should be revised, such as the left quotation mark in line 018. Please review the entire manuscript.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents SuperF, a test-time optimization method for multi-image super-resolution (MISR) using implicit neural representations (INRs). It improves sub-pixel alignment of low-resolution frames without requiring high-resolution training data, avoiding "hallucinated" structures. SuperF outperforms existing methods on both satellite and ground-level images, offering a data-efficient solution for super-resolution.

### Strengths
The use of implicit neural representations (INRs) for multi-image super-resolution (MISR) is innovative, offering a novel approach that leverages sub-pixel alignment optimization. 
By avoiding the need for high-resolution training data, SuperF is a practical solution for real-world applications, including satellite imagery and handheld cameras.
The approach generalizes well across different datasets without the need for retraining, demonstrating versatility in various contexts like remote sensing and environmental monitoring.

### Weaknesses
The iterative optimization process, although memory-efficient, can be time-consuming, which might limit its applicability in real-time or mobile scenarios. Moreover, the method's performance is somewhat dependent on the Fourier feature scale hyperparameter, which requires careful tuning for different domains.

### Questions
1.	The current approach assumes that all frames depict the same scene, but how does SuperF perform when there are significant changes between frames, such as moving objects or occlusions?
2.	While the method shows domain sensitivity, could there be an automatic or adaptive mechanism for adjusting the scale based on the input domain, thus reducing the need for domain-specific tuning?
3.	Given that cloud cover can obscure parts of satellite images, how well does SuperF perform when parts of the scene are occluded or affected by noise, especially with respect to the uncertainty estimation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces SuperF, a test-time optimization framework for multi-image super-resolution (MISR) using implicit neural representations (INRs). It jointly optimizes a shared coordinate-based MLP and frame-specific affine alignments to reconstruct a high-resolution image from multiple low-resolution inputs. The approach requires no training data and is validated on both synthetic satellite and handheld burst datasets, demonstrating strong PSNR and SSIM improvements over prior TTO baselines.

### Strengths
SuperF is well-motivated, bridging the gap between INR-based image representation and multi-frame super-resolution. The joint optimization of alignment and representation is elegant and mathematically consistent. The experiments are thorough, covering domain generalization, ablations, and sensitivity analyses. The work also provides a new synthetic satellite burst dataset, adding clear community value.

### Weaknesses
1. The paper lacks direct visual or quantitative comparison with modern learning-based MISR networks (e.g., DeepBurstSR or Transformer variants), limiting practical benchmarking.
2. The uncertainty decoder’s effect is minor and under-analyzed; more discussion or visualization would clarify its contribution.
3. Runtime (several seconds per image) and computational cost are only briefly mentioned; profiling across resolutions would make claims about efficiency more credible.
4. The sensitivity to Fourier feature scale is described, but a cross-domain auto-tuning or normalization strategy would strengthen robustness.
5. The method assumes static scenes; discussion on handling dynamic or partially misaligned inputs (e.g., moving clouds, handheld motion) is needed for real-world applicability.

### Questions
See weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents SuperF, a new method for improving multiple-input super resolution (MISR).
SuperF uses coordinate-based neural networks that represents images as continuous signals rather than fixed pixels.
It performs test-time optimization and  does not require pre-trained high-resolution data.
SuperF shares one neural representation across  T low-resolution (LR) frames and jointly optimizes both alignments and reconstruction.

SuperF sbegins with T LR frames, each frame  defined on a finite set of points. 
The method estimates the underlying signal at a denser set of points relative to  LR input frames.
It assumes that each LR  frame is a convolution of a boxcar filter withan affine transformation of the high-resolution (HR)  target.  
To merge multiple frames, the method aligns the LR frames at the sub-pixel level.

SuperF achieve its goal by aligning the the LR frames using affine transformation while modeling the HR image as a continuous signal through the neural network.
The HR prediction is blurred with a boxcar filter and downsampled to match the LR resolution. 
The optimization for SuperF focuses on adjusting both the neural network parameters and the affine transformations so that all the LR frames agree on the same HR signal.

This paper provides experimental result to support it claims.

### Strengths
The paper is well organized and clearly presented. It provide sufficient information to understand the motivation of the work. It also provides adequate information about prior and related work. This help in establishing the originality of the work. The approach of this work as an test-time optimization also underscores the significance of the contribution. The method provides a solution to the need for resources-efficient methods deployment of machine learning methods, especially on resource-constrained hardware. Finally, the experiments are well designed, clearly presents, and supports the central claims of the paper. This reflects the overall quality of the paper.

### Weaknesses
1).  Algorithm Statement or Architecture Diagram: While this paper did a good job at explaining the components of SuperF, it does not provide enough information to understand how these components fit together, especially for the purpose of reproducibility. For example, Table 5 in the Appendix shows that there are two decoder networks (Fourier feature-based decoder and  uncertainty decoder). It is not clear how they come together as a singe model under the SuperF model architecture. Having an algorithm statement or a complete model architecture will improve the reproducibility and the potential impact of SuperF.

2). MSE vs GNLL:  Lines 245-246 state that  you have used the GNLL instead of  mean squared error. What was the assumption on the parameters of the  underlying Gaussian distribution for the GNLL?

### Questions
Please check my comments under the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
