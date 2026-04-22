# Exploring State-Space Models for Data-Specific Neural Representations

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
This paper studies the problem of data-specific neural representations, aiming for compact, flexible, and modality-agnostic storage of individual visual data using neural networks. Our approach considers a visual datum as a set of discrete observations of an underlying continuous signal, thus requiring models capable of capturing the inherent structure of the signal. For this purpose, we investigate state-space models (SSMs), which are well-suited for modeling latent signal dynamics. We first explore the appealing properties of SSMs for data-specific neural representation and then present a novel framework that integrates SSMs into the representation pipeline. The proposed framework achieved compact representations and strong reconstruction performance across a range of visual data formats, suggesting the potential of SSMs for data-specific neural representations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes to use state-space models (SSMs) to build neural representations. The main motivation for the use of SSMs is that many signals are discrete observations of an underlying continuous signal, permitting the use of classical signal processing techniques. In this paper, the authors extend an older approach of compressing signals by storing them as their coefficients in a fixed basis, which can be done using SSMs. This paper proposes a new SSM-based architecture which they call S3K. Based on their analysis of different architectures, this paper combines S3K with multi-scale features and a Laplacian pyramid to create a new SSM-based architecture for neural representations, which they call LPNet-S3K. The proposed method is evaluated on 2D image reconstruction, video reconstruction, and 3D shape reconstruction.

### Strengths
**(S1) Novelty**: this paper introduces the new idea of using SSMs to create neural representations and proposes a novel SSM-based architecture. 

**(S2) Architectural analysis**: This paper analyzes different architectures to create the best architecture for SSM-based neural representations.

**(S3) Theoretical justification**: This paper also provides some theory to back up their claims.

### Weaknesses
**(W1) Motivation/Limited applicability**: the S3K/LPNet-S3K method is mainly evaluated on toy tasks of the following form: given a signal, reconstruct that signal. The proposed method is not evaluated for more complicated tasks involving neural representations. Potential more complex tasks that are signal-to-signal include video interpolation, video superresolution, and 3D reconstruction from 2D partial views. Additionally, it is not possible to build S3K-based neural representations that map coordinates to function values, which represent a significant class of neural representations used today. 

**(W2) Computational trade-off of convolutional methods vs MLP methods**: The authors mention that, compared to convolutional methods like the one proposed in the paper, MLP methods offer higher fidelity but slower decoding speed. It seems to me that this may not matter in practice when there exist fast and efficient MLP neural representations such as InstantNGP [3]. 

**(W3) Experimental setup**: For the 2D image and 3D object reconstruction experiments, the only baselines are ablations of the proposed model and ConvNeXt, which was not designed as neural representation. I think the performance would be better contextualized by adding previous neural representation methods, such as FINER, which was mentioned in the Appendix A.8.4. Additionally, recent baselines for video reconstruction are either not mentioned in the main paper or not included in the paper. These include HiNeRV (only mentioned in the Appendix), *-Boost [1], NVRC [2]. 

**(W4) Empirical results**: The empirical justification of the method is mixed. For 2D and 3D reconstruction, the proposed method is worse than MLP methods but better than the use of a vanilla convolutional network (ConvNeXt). For video reconstruction, the method is generally better than all but HiNeRV, with some recent methods possibly missing. The increased performance comes at the cost of much more expensive training due to the increased expense of the LPNet-S3K encoder (20x more memory, 4x more FLOPS). 

[1] Zhang, Xinjie, et al. "Boosting neural representations for videos with a conditional decoder." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[2] Kwan, Ho Man, et al. "NVRC: Neural video representation compression." Advances in Neural Information Processing Systems 37 (2024): 132440-132462.

[3] Müller, Thomas, et al. "Instant neural graphics primitives with a multiresolution hash encoding." ACM transactions on graphics (TOG) 41.4 (2022): 1-15.

### Questions
**(Q1)**: How does the proposed method compare to recent methods such as Boost [1] and NVRC [2]?

**(Q2)**: How does the decoding speed of convolutional methods compare with that of efficient MLP methods such as InstantNGP [3]?

**(Q3)**: Can SSM-based methods be extended to create coordinate neural networks?

[1] Zhang, Xinjie, et al. "Boosting neural representations for videos with a conditional decoder." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[2] Kwan, Ho Man, et al. "NVRC: Neural video representation compression." Advances in Neural Information Processing Systems 37 (2024): 132440-132462.

[3] Müller, Thomas, et al. "Instant neural graphics primitives with a multiresolution hash encoding." ACM transactions on graphics (TOG) 41.4 (2022): 1-15.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors explored the application of state space model (SSM) in visual representation. They analyzed the feasiblity to use an SSM to extract the frequency domain information and designed a pyramidal network structure (S3K). They combined this design with different convolution encoders and verified the performance on multiple input modalities, incouding DVS, 2d image and 3d points. The result shows that their method have some advantages in visual representation quality and extendability to different modalities.

### Strengths
- Explored the novel application of SSM in visual representation encoding
- Clearly organized architecture and consistent logic
- Flexibile input forms, covering multi modalities

### Weaknesses
- Insufficient analysis on the motivation of method design

  The authors didn't explain the meaning of taking outer product of the 1d convolution kernel into multi dimansional

- Some methods not clearly described

  The article didn't define how to convert an image into the input of the proposed S3K Conv module

- No comparison with populat visual encoding methods like VAE and diffusion

  VAE method is commonly used in visual encoding in various visual and multimodal models. New methods should compare with important baselines and clarify the improvements and advantages

- Format issue in figure caption and referneces

  Figure numbering skipped 2, and the later figure references are mis-marked

### Questions
1. In Section 3.1, the authors claimed that SSM can encode a signal into frequency domain (coefficients of a sine series). For this purpose, DFT is the common approach, with mature and efficient algorithmic and hardward implementations. For digital image, there are also DCT and many other methods to yield the frequency domain representation of an image. What are the advantages to use SSM compared with these transformations?
2. In Section 4.2, the authors extend the dimansion of the kernel by making outer product. What is the meaning of this operation in the aspect of State Space Models?
3. What is the relationship between an input (2d image for instance) and the input function $\phi(t)$?
4. What are the advantages of the proposed S3K method compared with the broadly used VAEs?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper explores the integration of state-space models (SSMs) into data-specific neural representations (e.g., implicit neural representations for images, videos, and 3D objects). The authors first analyze the theoretical foundation of SSMs, showing that their latent states effectively capture the coefficients of continuous basis functions and thus suit compact signal modeling. Based on this insight, they propose a new Structured State-Space Kernel (S3K), which translates SSM dynamics into convolutional kernels, enabling multi-dimensional processing and inherent downsampling.

The final architecture, LPNet-S3K, integrates S3K layers into a Laplacian pyramid structure and demonstrates strong performance across diverse modalities—images (Kodak, CLIC2020), videos (Bunny, UVG, DAVIS), and 3D objects (Objaverse). The results show consistent gains over ConvNeXt- and Mamba-based baselines and over established NeRV-family models, validating both theoretical and empirical advantages.

### Strengths
1. Instead of simply integrating the off-the-shelf state-space model, this paper provides the first systematic theoretical and empirical exploration of applying SSMs to data-specific neural representations. The analysis linking SSM parameters to signal reconstruction coefficients offers a new, signal-processing–grounded perspective on neural compression. In particular, it found a non-trivial conclusion that SSMs consistently outperform the transformer in data representation, which is also explained from the perspective of a classical sinusoid signal processing task.

2. The work revisits how SSMs should be incorporated and identifies some design findings: stacking SSMs can harm reconstruction; applying them to downsampled multi-scale features improves quality; and Laplacian pyramid decomposition provides the best balance between redundancy and expressivity.

3. The experiments are broad and convincing—covering 2D, video, and 3D data. LPNet-S3K consistently improves reconstruction quality with comparable model size and decoding speed. Note that performance gains are achieved via encoder modification, keeping decoders identical to baselines.

### Weaknesses
While empirical findings (e.g., stacking degradation, Laplacian advantage) are useful, their underlying reasons are only intuitively discussed. A deeper analytical or visualization-based explanation (e.g., spectral bias, basis overlap) would strengthen the claims.

While this paper shows the advantages of state space model in data representations, I encourage the author to build a SOTA implicit neural representation-based neural codec based on it.

### Questions
See in weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This well-written paper proposes using modern state-space models (SSMs) for data-specific implicit neural representations (intentionally overfitting SSMs to a single datum) for image, object, and video data. They introduce a structured state-space kernel (S3K) that turns an SSM into a convolutional kernel, allowing for natural n-D preprcessing and learnable downsampling. The final encoder uses a Laplacian-pyramid design with S3K layers and plugs into existing decoders, and they compare against a rich set of baselines.

### Strengths
Clear link between SSMs and continuous INRs through basis expansions. 

S3K has a concise definition and extends clearly to nD using 1D outer products, allowing it to potentially drop into many encoder models.

Good ablations studies and architectural insights: they evaluated (a) stacked, (b) image pyramid, and (c) Laplacian pyramid variants (showing the last to be preferable across models). This shows that stacking in SSMs may harm reconstruction (they speculate through repeated projection amplifying artifacts). 

They provide a number of ablations, including on input adaptivity, real vs. complex parameters, inverted bottlenecks, and SiLU/RMSNorm. 

On video benchmarks, replacing only the encoder improves PSNR. 

Theory shows relationship between diagonalizable A and spectrum and supports compression framing.

### Weaknesses
Fidelity metrics (PSNR/SSIM) alone don’t substantiate “compact representation.'

The memory/compute overhead of explicit kernels is substantial (the authors acknowledge this).

### Questions
Are fidelity metrics (PSNR/SSIM) enough to support compression claims? It would be nice to see, e.g., bitrate comparisons / rate-fidelity curves?

Although acknowledged, the compute/memory overhead is significant. Could the authors implement one of the suggested efficient formulations, at least in one case/example?

Could you compare against SSM-convs that don’t build explicit kernels?

How sensitive are results to state size? Does this work for very high res images/videos on realistic training budgets?

### Soundness
3

### Presentation
4

### Contribution
3
