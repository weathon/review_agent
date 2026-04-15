# Degradation-aware Unfolding Knowledge-assist Transformer for Spectral Compressive Imaging

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 5

## Abstract
Snapshot compressive spectral imaging offers the capability to effectively capture three-dimensional spatial-spectral images through a single-shot two-dimensional measurement, showcasing its significant potential for spectral data acquisition. However, the challenge of accurately reconstructing 3D spectral signals from 2D measurements persists, particularly when it comes to preserving fine-grained details like textures, which is caused by the lack of high-fidelity clean image information in the input compressed measurements. In this paper, we introduce a two-phase training strategy embedding high-quality knowledge prior in a deep unfolding framework, aiming at reconstructing high-fidelity spectral signals. Our experimental results on synthetic benchmarks and real-world datasets demonstrate the notably enhanced accuracy of our proposed method, both in spatial and spectral dimensions. Code and pre-trained models will be released.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a degradation-aware unfolding knowledge-assist transformer for spectral compressive imaging. It aims to address the challenge of reconstructing high-fidelity 3D spectral signals from 2D measurements. The authors propose a two-phase training strategy that embeds high-quality knowledge prior in a deep unfolding framework. They employ a lightweight counselor model trained via a vector-quantized variational autoencoder (VQ-VAE) to guide the reconstruction process. Experiments are conducted to evaluate the performance of  the method against existing ones.

### Strengths
1. The presentation of the paper is satisfactory, and the main idea is easy to access
2. The experiments showed modest improvement over existing ones.

### Weaknesses
1. This  paper is closely related to [2022c] Cai et al. "Degradation-aware unfolding half-shuffle transformer for spectral compressive imaging, NeurIPS2022. Its main idea essentially is similar to that of [2022c]. The paper did not provide a detailed comparison on the main differences and why the proposed one works better than [2022c].

2, The work appears to be a re-engineered version of existing research, particularly the aforementioned work by Cai et al. (2022). The paper lacks compelling arguments to substantiate the purported benefits of the modifications introduced in their method.

3. Given the application-oriented nature of this research, the limited dataset employed for testing undermines the credibility of the claimed performance gains over existing methods.

4. While the ablation studies do indicate minor improvements attributable to the proposed method, these gains are not substantial enough to fully support the paper's claims. It raises the question of whether the observed performance improvements might be more a result of fine-tuning the neural network rather than the novel aspects of the proposed method.

### Questions
See the discussion above

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a degradation-aware knowledge-assist deep unfolding framework for preserving fine-grained details. The denoiser is learned with a VQ-VAE-based counselor network and a sophisticated mask cross encoder. The proposed method achieves comparable experimental results on synthetic benchmarks and real-world datasets

### Strengths
1.	The proposed method achieves the SOTA results.
2.	The overall paper is easy to follow.

### Weaknesses
1.	The contribution of this paper is incremental. The main contribution is combining a VQ-VAE in the first stage of the Deep Unfolding framework. Although the idea of utilizing a codebook makes sense, the current approach is very trivial.
2.	Visual performance is poor. In Figure 4, the result of RDLUF_MixS2-9 is clearer in details than result of the proposed method. Moreover, Moreover, CST-L+ seems to obtain better visual effects than the proposed method in real experiments. This is inconsistent with the author's claim that more texture information can be restored
3.	Lack of sufficient experimental verification. The number and location of the VQ-VAE module should have an impact on the results, and the author should provide relevant experiments
4.	Lack of comparison of computational cost and parameter numbers to prove the proposed method is more efficient. The complexity of Deep Unfolding methods is not only reflected in the number of stages, but also in the computational complexity of each stage

### Questions
1.	In the two-stage training of the network, there is a significant difference in the input of VQ-VAE. The first stage is a clean image, and the second stage is a very degraded image. Why not put VQ-VAE at the end of the network to make its input more similar to the input of the first stage? Would it be better to replace all denoisers in the network with VQ-VAE?
2.	What is the meaning of ‘video’ in the sentence ‘with a counselor model from the ground truth video as external knowledge prior and aim to guide the reconstruction’ in the 2.2 section?

### Soundness
3 good

### Presentation
3 good

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
In this paper, a spectral SCI reconstruction network was introduced, which incorporated high-fidelity knowledge from clean HSIs and explicit cascade degradation information into a deep unfolding framework for spectral compressive image reconstruction. A two-phase training strategy was employed, starting with a lightweight counselor model based on a vector quantized variational autoencoder (VQ-VAE) to learn high-fidelity knowledge from ground-truth images. Subsequently, a U-net bridge denoiser equipped with the mask-aware knowledge-assist attention (MKA) mechanism was introduced to incorporate multi-layer degradation information and high-fidelity prior knowledge effectively. Importantly, the unfolding framework could integrate external prior knowledge from the counselor model and cascade degradation information, enhancing the reconstruction performance. The key contributions included the proposal of a degradation-aware knowledge-assist deep unfolding framework, a U-net bridge denoiser that combined high-fidelity HSI knowledge and cascade degradation information, and extensive experimental validation demonstrating the method's superior accuracy in both spatial and spectral dimensions, using synthetic benchmarks and real datasets.

### Strengths
The paper contributes to the field of image processing by addressing challenges related to degradation mismatch and offering a framework that can enhance the quality of spectral compressive image reconstructions.

### Weaknesses
The ablation experiments are conducted exclusively on synthetic data. To enhance the comprehensiveness of the research, it would be beneficial to include experiments on real datasets as well, offering a more comprehensive evaluation of the proposed method's effectiveness.

The terminology "bridge U-net" appears to refer to a U-net framework with transformer embeddings, and it seems that the bridges between the encoder and decoder functions are just skip connections. Providing clarity on these terms and their roles within the model would aid in understanding the architectural components.

### Questions
A significant challenge addressed in the paper is the impact of degradation mismatch on reconstructed images. It might be worthwhile to explore the possibility of applying sophisticated denoising techniques as an initial step before proceeding with other processing stages. This approach could potentially lead to improved results and mitigate the issues arising from degradation mismatch.

The authors mention that the first stage denoiser is a pre-trained VQ-VAE, while subsequent stages use U-shape networks. It is essential to justify the necessity of designing a special module for the first stage and explain the criteria used for selecting the type and size of these modules in the following stages. Providing these justifications would enhance the transparency and rationale behind the model's design.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a degradation-aware deep unfolding neural network for spectral compressive imaging. The unfolding neural network is assisted by the vector quantized hyperspectral image knowledge, integreated with the degradation information via a U-net bridge denoiser. A  two-phase scheme is introduced for model traiing. Extensive experiments on the synthetic benchmark wereconducted demonstrate the effectiveness of theproposed method.

### Strengths
1. Superior reconstruction accuracy has been achieved over existing works, in both spatial and spectral dimension.

2. The proposed network is degradation-aware and knowledge-assisted.

3. Utilization of convolutions, transformer blocks and maks-aware self-attention in an unfolding network leads to performance gain.

### Weaknesses
1. The novelty of this approach appears limited. Unfolding networks with transformers or self-attention mechanisms have been previously introduced in previous works, such as [A1] for spectral compressive imaging. Additionally, the U-net bridge denoiser can be described as skip connections augmented with multiple convolution layers (as mentioned on Page 7, "we implement a bridge module including several convolution layers"), which is a common architectural element found in existing literature.

[A1] Degradation-Aware Unfolding Half-Shuffle Transformer for Spectral Compressive Imaging NIPS 2022.


2. The MASK-AWARE KNOWLEDGE-ASSIST DENOISING moduel integrates mask priors into the Key and Value components of Self-Attention. This approach shares similarities with the one proposed in [A2]. Furthermore, the experimental results provide support for the advantages of incorporating mask priors into the Values of Self-Attention.

[A2] Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction, CVPR 2022. 

3. The experimental part misses the comparison in terms of computational compleixty, specifically the number of FLOPS and running time. Indeed, the proposed model appears to be computationally intensive, due to its utilization of deformable and self-attention operations.


4. The performance improvement achieved through the two-stage scheme is marginal, with only a 0.1dB PSNR gain.

5. The writing needs improvement. For instance, the Abstrtact mentions little information of the proposed method.

6. The zoom-in regions seem to be mistakenly into the last rows, in both Fig. 4 and Fig. 5.

### Questions
1. Please clarify the difference between this work and [A1] as well as [A2].

2. Please compare in terms of the number of FLOPS and running time.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
