# Autoregressive Video Autoencoder with Decoupled Temporal and Spatial Context

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Video autoencoders compress videos into compact latent representations for efficient reconstruction, playing a vital role in enhancing the quality and efficiency of video generation. However, existing video autoencoders often entangle spatial and temporal information, limiting their ability to capture temporal consistency and leading to suboptimal performance. To address this, we propose Autoregressive Video Autoencoder (ARVAE), which compresses and reconstructs each frame conditioned on its predecessor in an autoregressive manner, allowing flexible processing of videos with arbitrary lengths. ARVAE introduces a temporal-spatial decoupled representation that combines downsampled flow field for temporal coherence with spatial relative compensation for newly emerged content, achieving high compression efficiency without information loss. Specifically, the encoder compresses the current and previous frames into the temporal motion and spatial supplement, while the decoder reconstructs the original frame from the latent representations given the preceding frame. A multi-stage training strategy is employed to progressively optimize the model. Extensive experiments demonstrate that ARVAE achieves superior reconstruction quality with extremely lightweight models and small-scale training data. Moreover, evaluations on video generation tasks highlight its strong potential for downstream applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposed a lightwiehgt autoregressive video autoencoder, referred as ARVAE. Instead of splitting to small video clips or blocks, ARVAE compresses and reconstructs videos in a frame-by-frame way. Given the previous frame, ARVAE calculates the motion field between the consecutive frames with a motion estimator, then the temporal encoder transforms the motion field to temporal motion and multi-scale features, afterwards, the spatial encoder produces the spatial supplement to encode the difference of the current frame. The motion extraction blocks downsample the motion fields in N+1 stages. N feature extraction blocks obtain multi-scale pyramid image after the warping from the previous frame. The spatial encoder applies downsampling and convolution to encode the hierarchical features.  ARVAE has been trained in a multi-stage way. Using very few training videos on Vimeo-90K and Inter4K, ARVAE performs well on MCL-JCV and WebVid-Val in terms of PSNR, SSIm and LPIPS.

### Strengths
The experiments show ARVAE works well on two datasets compared with 6 recent video autoencoders.

### Weaknesses
The goal and motivation of this work are not that clear. The purpose of ARVAE is to help video generation or achieve better video compression? What are the compression ratio and efficiency?

The idea of performing motion estimation first and then encode the spatial difference frame-by-frame is not novel, which is the common practice for decades in video compression standards like MPEG or H.264/5 etc.

The technical presentations are not clear. What are the exact architectures of the temporal encoder and spatial encoder? What are the exact definition of the propagated features P_e? etc.

The paper claims ARVAE “without loss of information” (ll.098). Since there are motion estimator and many downsampling and upsampling operations, and the performance is measured by PSNR, please justify in what sense it is “without loss of information”.

### Questions
What are the exact architectures of motion extraction blocks and feature extraction blocks?  the same number of N blocks for both kinds of blocks? What is the exact warping operation? The warping is applied to the entire frame?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel Autoregressive Video Autoencoder (ARVAE) that encodes and reconstructs video frames in an autoregressive manner. The key innovation lies in the decoupled temporal and spatial latent representation, where motion fields are used to model temporal consistency, and residual spatial information is encoded as a supplement for new content in each frame. The framework includes a temporal encoder/decoder, a spatial encoder/decoder, and a multi-stage training strategy that progressively increases sequence length to enhance temporal modeling. Extensive experiments demonstrate that ARVAE achieves state-of-the-art reconstruction quality with significantly fewer parameters and training data than existing baselines, and the method is applicable to both video reconstruction and generation tasks.

### Strengths
- Clear Motivation and Novelty: 

The paper addresses a long-standing challenge in video autoencoding: entangled spatial-temporal modeling, and propose temporal-spatial decoupling is a conceptually clean and potentially impactful approach.

- Autoregressive Design:

Autoregressive frame prediction aligns well with the temporal nature of video data and allows for variable-length sequence modeling, which is an advantage over fixed-length VAEs.

### Weaknesses
- Lack of theoretical insight into key design choices:
The paper proposes a decoupled latent representation that separates temporal motion from spatial content. While this design is empirically effective, the underlying rationale is not well articulated. It remains unclear why this separation leads to better generalization or compression efficiency compared to joint modeling. A deeper conceptual or theoretical discussion would strengthen the paper’s contribution.


- Efficiency claims are insufficiently supported:
Although the method is described as lightweight and efficient, the paper lacks quantitative runtime analysis. Key metrics such as inference time, memory usage, and decoding FLOPs are not reported. Without such data, it is hard to objectively validate the claimed efficiency advantages over existing models.

- Experimental design lacks clarity in some parts:
While the authors conduct both qualitative and quantitative comparisons, the presentation of results could be more focused. In particular, the visual comparisons do not clearly highlight where ARVAE outperforms other methods (e.g., handling of motion, texture fidelity, or temporal consistency). Similarly, the numerical results, though comprehensive, are not always clearly interpreted. More targeted analysis would help clarify the distinctive benefits of the proposed method.

- Ablation studies could be more thorough:
The ablation experiments successfully demonstrate the importance of individual components (e.g., multi-scale features, state features, and multi-stage training). However, the analysis does not explore potential failure cases or limitations. For example, it would be helpful to understand how the model behaves under challenging conditions such as fast motion, occlusion, or long-range propagation. This would provide a more complete picture of the method’s strengths and weaknesses.

- Presentation and clarity:
Some figures—particularly the architecture diagrams—lack sufficient detail or clear labeling, which may hinder understanding for readers unfamiliar with the topic. Additionally, certain sections in the methodology could benefit from more formal definitions and clearer notation to improve technical clarity.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The manuscript presents a new video autoencoder ARVAE, aiming to address difficulty in capturing the temporal consistency of existing video autoencoders. The video frames are compressed and reconstructed in a frame by frame manner in ARVAE, such that a flexible number of frames can be processed. For each frame, ARVAE converts it into decoupled spatial and motion representations using a two-branched model, and the decoder reconstruct the image based on the decoupled representations. Emprical results show that ARVAE is highly parameter efficient, using 0.1M parameter to achieve similar performances to video autoencoders with more than 10M parameters.

### Strengths
1. ARVAE is motivated to address the temporal consistency in video generation, through a decoupled video representation composed of spatial and motion components. Intuitively speaking, this strategy is capable of preserving spatial details of previous frames. 

2. ARVAE shows strong parameter efficiency, achieving strong reconstruction performance compared to video autoencoders with 100x larger parameter scale.

3. Ablation experiments show the effectiveness of using multi-scale propagated features and reconstruction with state features.

### Weaknesses
1. Though reconstruction is an important evalaution aspect for autoencoders, it is even more important to evaluate whether the latent is good for further generation process. Yet the manuscript provides limited evaluation on the generation side. First, since the representations are decoupled, it is unclear how the spatial and motion representations are used during the generation process. Second, the comparison is limited to the comparison between system-level generation results, with no controlled experiments on whether the learned representation is easy or difficult to generate. This usually requires using a standard generation model, VSD for example, and compare different latents under the same training data and number of iterations. 

2. It seems to me that the ARVAE could not achieve temporal compression. That is to say the temporal compression ratio is 1, which is probably the underlying reason why the model is able to achieve strong reconstruction performance with such small parameter scale. The temporal compression ratio also needs to be explicitly specified in the comparison. It would be better if FLOPs comparison could be included as well. 

3. Though the manuscript is motivated to improve temporal consistency, the experiments fail to show that ARVAE is capable of improving the temporal consistency of reconstructed or generated videos. Though qualitative results are provided for video reconstruction, the difference between different autoencoders is negligible.

### Questions
- Is spatial feature and temporal feature concatenated for video generation?

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
5

### Summary
This paper proposes an Autoregressive Video Autoencoder (ARVAE) for video reconstruction. The core idea is to encode and decode each frame autoregressively, conditioned on its predecessor. The proposed latent representation is decoupled into a temporal motion component, derived from an estimated optical flow, and a spatial supplement component, representing the residual information. The authors claim that this approach achieves superior reconstruction quality with a significantly smaller model and less training data compared to state-of-the-art methods that perform joint spatiotemporal compression. Experiments are conducted on standard benchmarks, and an application to video generation is briefly explored.

### Strengths
1. The primary strength of this paper lies in its reported quantitative results. As shown in Table 1, the proposed ARVAE achieves state-of-the-art or highly competitive performance on PSNR, SSIM, and LPIPS metrics across multiple benchmarks, while claiming to use a fraction of the model parameters and training data of strong baselines like Step-Video and LTX-Video. If these results are validated to be under fair comparison, they would be quite significant.
2. The autoregressive approach, which leverages the previous frame to reconstruct the current one, is conceptually simple and intuitive. It builds upon a long-standing principle in classical video coding (i.e., predictive frames), making the method easy to understand.

### Weaknesses
[Poorly Justified Motivation] The paper's motivation is weak and questionable. The authors claim that existing methods "model the video with spatial and temporal information intermingled, failing to effectively capture the temporal consistency" (Lines 41-42), and they single out "3D attention mechanisms" as the target of criticism. This is problematic for two main reasons: First, 3D attention is not a dominant component in most state-of-the-art video VAEs; 3D convolutions are far more prevalent. Attacking a niche design choice makes the motivation feel misplaced. Second, the claim that 3D convolution leads to harmful spatiotemporal entanglement is not a community consensus. The authors provide no theoretical analysis, empirical evidence, or citations to support this assertion, making the premise of the work feel unsubstantiated.

[Crucial Methodological Details are Missing] The paper omits critical details necessary for a fair assessment of the work's contributions. The "off-the-shelf image-based autoencoder" used for the first frame is never specified. The quality of this "Img-frame" is fundamental to the entire autoregressive chain, and its architecture and performance are unknown. The "motion estimator" is treated as a black box. The paper does not specify the network architecture (e.g., PWC-Net, RAFT).

[Questionable Experimental Choices and Result Quality]
The model is not trained with a GAN loss, which typically leads to blurry reconstruction results for VAEs. The authors seem to have avoided this issue by presenting very few qualitative examples (only 4 images in the paper), three of which are cartoons, and one is inherently blurry. This selective visualization feels like cherry-picking and fails to convincingly demonstrate the model's ability to reconstruct sharp details in realistic, complex scenes.

[Misleading or Unfair Comparisons in Experiments] The experimental results, while seemingly impressive, are presented in a misleading way that conceals major disadvantages.
1.	Unfair Compression Ratio: The proposed method does not perform temporal compression (the temporal compression ratio is 1x), whereas the SOTA methods it compares against (e.g., Step-Video, LTX-Video) employ significant temporal compression (e.g., 4x or 8x). This is a massive, unfair advantage that is not highlighted in the tables, rendering the numerical comparison of reconstruction quality largely meaningless.
2.	Impractical for Downstream Tasks: The lack of temporal compression means the number of latent tokens for a video clip would be extremely large (4x or 8x more than competitors). This would make training a downstream diffusion model computationally prohibitive, questioning the practical utility of the proposed VAE.
3.	Inaccurate Parameter Count: The claim of a tiny 6M parameter model is highly suspect. The model consists of an off-the-shelf image autoencoder, a motion estimator, and the ARVAE encoder/decoder. This reported number appears to be incorrect or at least incomplete, potentially omitting the parameters of the motion estimator and the first-frame encoder.
4.	Poor Temporal Generalization: Table 5 shows that performance degrades as the sequence length increases, even for a short 16-frame clip. This is a drawback of autoregressive models, and the paper fails to evaluate the model on longer sequences to assess the critical issue of error accumulation.
5.	Outdated Generation Baselines: The video generation experiment (Table 6) compares against GAN-based methods from 2022. This comparison is meaningless in the current landscape dominated by diffusion models.
6.	Missing Inference Time Analysis: The paper completely omits any discussion of inference speed. I suspect the author's motion estimator is a flow model, but flow models like RAFT require an extremely high number of iterations to produce accurate optical flow and are exceptionally slow. Yet the author avoids any relevant description of this.

In summary, the paper appears to selectively present results and comparisons in a way that is favorable to itself while omitting critical details and fair evaluations. The motivation is weak, the method description is incomplete, and the experiments are fundamentally flawed in their setup and reporting. Therefore, I cannot recommend acceptance.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
