# Trajectory-aware Shifted State Space Models for Online Video Super-Resolution

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Online video super-resolution (VSR) is an important technique for many real-world video processing applications, which aims to restore the current high-resolution video frame based on temporally previous frames. Most of the existing online VSR methods solely employ one neighboring previous frame to achieve temporal alignment, which limits long-range temporal modeling of videos. Recently, state space models (SSMs) have been proposed with linear computational complexity and a global receptive field, which significantly improve computational efficiency and performance. In this context, this paper presents a novel online VSR method based on Trajectory-aware Shifted SSMs (TS-Mamba), leveraging both long-term trajectory modeling and low-complexity Mamba to achieve efficient spatio-temporal information aggregation. Specifically, TS-Mamba first constructs the trajectories within a video to select the most similar tokens from the previous frames. Then, a Trajectory-aware Shifted Mamba Aggregation (TSMA) module consisting of proposed shifted SSMs blocks is employed to aggregate the selected tokens. The shifted SSMs blocks are designed based on Hilbert scannings and corresponding shift operations to compensate for scanning losses and strengthen the spatial continuity of Mamba. Additionally, we propose a trajectory-aware loss function to supervise the trajectory generation, ensuring the accuracy of token selection when training our model. Extensive experiments on three widely used VSR test datasets demonstrate that compared with six online VSR benchmark models, our TS-Mamba achieves state-of-the-art performance in most cases and over 22.7% complexity reduction (in MACs). The source code for TS-Mamba is available at https://github.com/QZ1-boy/TS-Mamba.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a trajectory-aware shifted state space model for online video super-resolution, named as TS-Mamba. TS-Mamba leverages trajectory to select relevant tokens from previous frames, shifted Hilbert scanning to compensate the spatial discontinuities, and a trajectory-aware loss to supervise trajectory generation. Experimental results show the competitive or superior of TS-Mamba compared with SOTA, with reduced complexity.

### Strengths
The integration of temporal trajectory into Mamba for online VSR is original. 
The trajectory mechanism for video is a well-motivated and reasonable way to capture the longer temporal dependencies while maintaining low complexity. That is suitable for online video-related tasks. 
Novel use of state space models (SSMs) with elaborately designed shifted operations to compensate for Hilbert scanning discontinuities，and sufficient illustrations enhance understanding and readability. 
This paper is a well-organized and well-presented, with ablations and detailed training settings. The illustration of the SSMs scan is impressive and clear.

### Weaknesses
Author proposes a trajectory-aware shifted Mamba aggregation module to achieve the long-term spatio-temporal information aggregation. However, the existing trajectory/temporal alignment methods (e.g., deformable alignment, flow-guided deformable alignment) also can achieve this goal, a more detailed comparison with these methods would strengthen this part. 

The experiments only use standard window sizes, without specific analysis on long-term dependency (e.g., varying temporal window T). A comparison between using the fixed temporal window (i.e., T=15) and other temporal window sizes would help determine the optimal size of temporal window of trajectory modeling for the proposed method.

The author did not thoroughly analyze the shortcomings of existing methods, and the description is also quite vague. It is suggested that the author conduct a more in-depth analysis of the shortcomings of existing methods to further highlight the advantages of the method proposed in this paper.

### Questions
(1) It is suggested to add a new set of experiments ablate the temporal window size. This is a low-effort, high-impact experiment that directly validates a core contribution of the paper. 
(2) The authors should revise the introduction and related work sections to include a concise yet specific "Limitations of Prior Work" paragraph that directly motivates the design choices of their own model.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on online video super-resolution (VSR) and proposes a new model called TS-Mamba, which is built upon State Space Models (SSMs). The method introduces long-term trajectory modeling combined with the lightweight Mamba framework to improve spatio-temporal information aggregation efficiency in video sequences.

### Strengths
The integration of Mamba-based spatial modeling with trajectory-aware design is conceptually clear and implemented in a well-structured manner.

The proposed Trajectory-aware Shifted Mamba Aggregation (TSMA) module is technically sound and engineering-wise elegant.

The model achieves reasonable complexity-efficiency trade-offs — it incorporates long-term temporal modeling while maintaining relatively low MACs and parameter counts compared to prior online VSR baselines.

### Weaknesses
1. Biased motivation.

The paper claims that “most existing VSR methods solely employ one neighboring previous frame”, yet this statement overlooks numerous recent works that already explore long-range temporal modeling. For instance:

[1] Wang, Xijun, et al. LiftVSR: Lifting Image Diffusion to Video Super-Resolution via Hybrid Temporal Modeling with Only 4×RTX 4090s. arXiv:2506.08529 (2025).

[2] Liu, Yong, et al. UltraVSR: Achieving Ultra-Realistic Video Super-Resolution with Efficient One-Step Diffusion Space. arXiv:2505.19958 (2025).

[3] Zhuang, Junhao, et al. FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution. arXiv:2510.12747 (2025).
These studies demonstrate substantial progress in long-term temporal modeling, undermining the paper’s claim that such research is scarce. This weakens the motivation and positioning of the work.

2. Lack of analysis on real-time VSR methods.

In lines 109–117, the authors argue that optical-flow or deformable-convolution-based methods are computationally expensive. However, multiple studies have achieved real-time performance with efficient architectures, such as:

[1] Cao, Yanpeng, et al. Real-Time Super-Resolution System of 4K-Video Based on Deep Learning. IEEE ASAP 2021.

[2] Jiang, Y., Nawała, J., Feng, C., et al. RTSR: A Real-Time Super-Resolution Model for AV1 Compressed Content. IEEE ISCAS 2025.
The paper fails to include any quantitative or empirical comparison with these real-time approaches and instead relies on generic qualitative claims. This lack of evidence weakens its argument regarding computational superiority.

3. Outdated comparison on standard benchmarks.

The experiments are conducted on REDS4, Vid4, and Vimeo-90K-T datasets, but do not include recent state-of-the-art models that have achieved higher PSNR and SSIM. For example:

Lin, Jing, et al. Unsupervised Flow-Aligned Sequence-to-Sequence Learning for Video Restoration. ICML 2022 — achieving 31.96 dB and 37.63 PSNR on REDS4 and Vimeo-90K-T, outperforming the proposed method.

Dong, Shuting, et al. DFVSR: Directional Frequency Video Super-Resolution via Asymmetric and Enhancement Alignment Network. IJCAI 2023.
Without comparisons to these recent methods, the experimental evaluation is incomplete and fails to convincingly establish the method’s competitiveness.

Limited novelty in applying Mamba to VSR.
The use of Mamba-based State Space Models for video super-resolution is not new.

Event-based Video Super-Resolution via State Space Models (CVPR 2025) already introduced a similar Mamba framework for VSR.
The authors do not clarify how TS-Mamba fundamentally differs from or advances beyond these prior Mamba-based designs. Thus, the methodological contribution appears incremental rather than novel.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper focuses on online video super-resolution which aims to reconstruct a high-resolution video from its low-resolution counterpart using only past frames, i.e., without access to information from future frames. This research has high industrial values, especially for real-time and streaming applications. In this paper, authors propose a variant of Mamba models, called TS-Mamba, to model the long-term temporal information bewtween frames. Specifically, to effectively leverage similar features from previous frames, the proposed methods aims to select similar features along the trajectory. This token selection strategy keeps spatial consistency and avoids artifacts in reconstruction. The original Mamba models efficiently process 1D sequential data, but the Raster scan mechanism cannot handle image data, especially in boundary areas. To this end, TS-Mamba introduces a variant of Hilbert scan in the spatial domain, which is restricted within and between local windows. By incoorporating the shifting operator, the proposed model can effectively process image content from adjacent windows for restoration. To better supervise model during training, it additionally propsoes trajectory loss. The experiment results have demonstrated promising results in REDS4 and Vimeo-90K datasets with BI degradation and BD degradation. Overall, this is a good paper. Unlike previous methods, it introduce Mamba structure for online video super-resolution and has demonstrated a better trade-off between restoration quality and speed. However, I still have several suggestions, questions and confusions about this paper listed as below.

### Strengths
1. The paper comprehensively discusses the related works in online video super-resolution and points out the existing limitations of the existing methods.
2. TS-Mamba solely utilizes past frames for online video super-resolution and aggregates long-range information through trajectory-aware token selection, which motion paths across multiple previous frames.
3. The proposed method is efficient without sacrificing quality, leading to a better trade-off between reconstruction quality and processing time.

### Weaknesses
1. what is $$v_{\tau_{i}^{h_{j}}}? Are they math typos in Eq.(7) and Eq.(8)?
2. The method proposes a trajectory-aware method and define a temporal trajectory among video frames in Eq.(3). However, I am confused on this definition. It is better to elaborate more on what positions among video frames belong to the same trajectory.
3. The method introduces a dual-path block, i.e., intra-window compensation branch and inter-window compensation branch. What is the key difference between two blocks? Why one block can handle intra-window content and another one can hanld inter-window content?
3. The method proposes variant scanning strategies. What scanning strategy it adopts in experiment? And why using this scanning strategy instead of other strategies?
4. It is better elaborate on the descriptions of shifted SSMs block, especially for math notations. It seems that elimintation value is a hyper-parameter. Does it matter for the restoration performance?
5. In the paper, it proposes trajectory-aware loss. How to compute this loss? Does it compute L2 loss between the trajectories in LR and the corresponding counter in HR images?
6. It is better to demonstrate visual results associated with Table 2. However, due to page limited, it is still acceptable.

### Questions
I include the questions and concerns on this paper in Weakness section. Please authors prepare their rebuttal reference to the questions listed in Section 6.

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
4

### Summary
This paper proposes TS-Mamba, a state-of-the-art method for online video super-resolution (VSR), leveraging trajectory-aware shifted state space models (SSMs) for efficient spatio-temporal information aggregation. The model addresses challenges in long-term temporal modeling for real-time applications while keeping computational complexity low. The approach combines token-level spatio-temporal aggregation with a novel trajectory-aware shifted Mamba aggregation (TSMA) module. TS-Mamba constructs trajectories within video frames to select similar tokens from previous frames and aggregates them using shifted SSM blocks. This enables improved video frame restoration with reduced computational overhead. The proposed model is evaluated on several benchmark datasets (REDS, Vimeo-90K-T, Vid4) and outperforms five state-of-the-art online VSR methods in terms of PSNR/SSIM while reducing complexity by over 22.7%.

### Strengths
1. Performance: The model achieves superior performance in terms of PSNR/SSIM and visual quality across multiple benchmark datasets (REDS, Vid4, Vimeo-90K-T) and degradation types (BI and BD), demonstrating its robustness in real-world video restoration scenarios.

2. Computational Efficiency: TS-Mamba successfully reduces complexity by 22.7% in terms of MACs compared to existing methods, making it a strong candidate for real-time online VSR applications. The model is also one of the fastest among the tested methods.

3. Comprehensive Ablation Study: The authors provide an extensive ablation study validating the importance of trajectory-aware components and the shifted SSM blocks, as well as the impact of different design choices (e.g., token number, shift operations). This strengthens the validity of their claims and helps illustrate the contributions of each part of the model.

### Weaknesses
1. Lack of Comparison. The proposed scheme fails to discuss or compare with several recent restoration schemes that leverage state-space models (SSMs, e.g. Mamba) for superior performance. For instance, MambaIR and MambaIRv2 introduced a residual Mamba-based backbone (with convolution and channel attention) to capture global dependencies in image super-resolution and denoising, outperforming a SwinIR Transformer baseline. The more recent TAMambaIR improves efficiency by modulating the state-space transition for complex textures and using multi-directional scanning, achieving state-of-the-art results across image restoration tasks (e.g. super-resolution, deraining, low-light enhancement). In the video domain, VSRM proposes dual Spatial-to-Temporal and Temporal-to-Spatial Mamba blocks for long-range spatio-temporal feature extraction and a deformable cross-Mamba alignment module for flexible frame alignment, yielding new state-of-the-art performance on VSR benchmarks. Likewise, The authors should incorporate and evaluate these methods to ensure a comprehensive comparison with the current state-of-the-art in video super-resolution. （I believe that the VSR method, when transformed into the online VSR setting, can be compared.）

2. Insufficient Theoretical Justification: While the paper presents an innovative approach, the theoretical justification for the trajectory-aware shifted state space model is somewhat lacking. Specifically, a formal analysis of the long-term spatio-temporal aggregation process and a comparison with existing models using similar principles (e.g., in flow-guided deformable attention models) would provide more clarity. Additionally, the authors mention the use of shifted operations to enhance spatial continuity but could further explore the theoretical implications of this operation on model behavior.

3. Insufficient Validation. A notable weakness is the absence of experiments on real-world VSR datasets. The method’s results are only reported on standard synthetic benchmarks (REDS4, Vid4, Vimeo-90K) with bicubic or simulated blur degradation, but no evaluations on real degraded videos were provided . As real applications involve unknown and complex degradations (compression artifacts, sensor noise, motion blur, etc.), the paper should have validated the approach on established real-world video SR benchmarks. For example, testing on datasets like RealVSR (ICCV 2021), which is built with a dual-camera system on the iPhone 11 Pro Max, would demonstrate the model’s robustness to in-the-wild conditions

4. Potential on Certain Datasets: The model performs exceptionally well on the benchmarks but lacks a more nuanced discussion of failure cases or scenarios where the method might struggle. A brief mention of potential limitations (e.g., when frames are highly dynamic or occlusions are prevalent) would enhance the robustness of the paper's claims.

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
