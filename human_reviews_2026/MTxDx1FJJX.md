# UniE2F: A Unified Framework for Event-to-Frame Reconstruction with Diffusion Model

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Event cameras excel at high-speed, low-power, and high-dynamic-range scene perception. However, as they fundamentally record only relative intensity changes rather than absolute intensity, the resulting data streams suffer from a significant loss of spatial information and static texture details. In this paper, we address this limitation by leveraging the generative prior of a pre-trained video diffusion model to reconstruct high-fidelity video frames from sparse event data. Specifically, we first establish a baseline model by directly applying event data as a condition to synthesize videos. Then, based on the physical correlation between the event stream and video frames, we further introduce the event-based inter-frame residual guidance to enhance the accuracy of video frame reconstruction. Furthermore, we extend our method to video frame interpolation and prediction in a zero-shot manner by modulating the reverse diffusion sampling process, thereby creating a unified event-to-frame reconstruction framework. Experimental results on real-world and synthetic datasets demonstrate that our method significantly outperforms previous approaches both quantitatively and qualitatively. The code will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose UniE2F, a framework for event-to-frame video reconstruction using a fine-tuned Stable Video Diffusion model. While the residual guidance and zero-shot adaptation are positioned as core contributions, they are conceptually weak, insufficiently motivated for real-world scenarios, and lack rigorous empirical validation. The residual guidance builds on known ideas without demonstrating clear necessity or advantage, and the zero-shot claims are not convincingly benchmarked. Overall, the work offers limited novelty and fails to establish meaningful improvements over prior art.

### Strengths
The paper introduces a technically sound residual guidance mechanism and a unified zero-shot adaptation strategy, both grounded in prior concepts. While synthetic results are strong and ablations are thorough, the novelty is limited and real-world relevance remains unclear. The appendix and video were informative.

### Weaknesses
The paper's introduction and problem setup focus exclusively on reconstruction. It never motivates why the main contributions, i.e., event-based VFI or VFP are important problems to solve or what their real-world applications are.

A key benefit of event cameras is their high temporal resolution, allowing reconstruction at any arbitrary time. The proposed interpolation method (confirmed in App. D) generates a fixed sequence of frames, not a single frame at an arbitrary timestamp t. This misses the primary advantage of the sensor.

One significant weakness for applicability is the prohibitive computational cost. Appendix F (Table 8) reveals the method requires 46,753 MB (~47 GB) of VRAM and 245.364 TMACs, which is orders of magnitude more than competitors (e.g., HyperE2VID at 1052 MB and 0.060 TMACs). The 48-second latency for 12 frames (Sec 5.1) makes this method completely unusable for any practical or real-time application, which is the entire point of event cameras.

The paper claims SOTA on real-world reconstruction, but this is only true for MSE/SSIM. The LPIPS score (Table 1) is significantly worse than prior work (0.674 vs 0.562). This major discrepancy, which suggests poor perceptual quality, is never discussed or explained.

The key "zero-shot" contribution (VFI/VFP) does not generalize well. Table 2 shows it is clearly outperformed by standard baselines (e.g., CBMNet) when they are simply retrained on the target data. This suggests the "unification" is more of a curiosity on synthetic data than a robust, general-purpose tool.

### Questions
Major: 

- The LPIPS scores in Table 1 (Reconstruction) and Table 2 (VFI/VFP) are consistently and significantly worse than baselines on real-world data. This contradicts the excellent qualitative results in Figure 3. Can you explain this discrepancy? Is the LPIPS metric failing, or are the qualitative examples cherry-picked? This is a crucial point of confusion.

- The computational costs (Table 8, ~47GB VRAM) and latency (48s) are prohibitive and render the method unusable for any practical event-camera application. The limitation section (App. G) mentions this, but I'd like to ask: Do you believe this is a fundamental limitation of using large diffusion models for this task, or do you have concrete evidence that distillation/pruning can bridge the orders-of-magnitude gap to competitors?

- Your interpolation method (Sec 4.3, App. D) generates a fixed sequence of frames (e.g., 10 frames between $V_0$ and $V_{11}$). Why did you not pursue a more "event-native" approach, such as reconstructing a single frame at an arbitrary timestamp $t \in (0, 11)$?

- Could you expand on the motivation and real-world applications for zero-shot event-based interpolation and prediction? The paper currently justifies reconstruction but not these other tasks.

Minor:
- The introduction, preliminary, and related work sections are quite lengthy, spanning nearly four pages, yet they include redundant background and omit discussion of several key prior works. Could the authors clarify their criteria for selecting related work, particularly regarding early image reconstruction methods and event stacking approaches that are not cited?

Suggestions: 
Figure 1 illustrates general diffusion model concepts but does not appear to convey any paper-specific insights.
There are more important tables from the appendix that can be moved to the main paper if such sections become shorter.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a unified framework UniE2F for event-to-frame reconstruction. By fusing the generative prior of pre-trained video diffusion models with the physical characteristics of event data, it enhances the quality of reconstructing high-fidelity video frames from sparse events, and its effectiveness has been verified through experiments. The main contributions and ideas of the paper can be summarized as follows:

1. Unified Task Framework
(1) Event-driven Frame Reconstruction: Based on the pre-trained Stable Video Diffusion (SVD) model, event data is encoded into 3-channel tensors as conditional inputs. The model learns the mapping relationship between events and video frames through fine-tuning, establishing fundamental reconstruction capabilities.
(2) Inter-frame Residual Guidance Mechanism: Leveraging the physical correlation between events and inter-frame brightness changes, a ResNet-based inter-frame residual prediction module is introduced. The latent variables of the diffusion model are optimized via gradient descent to improve the temporal consistency and accuracy of reconstructed frames.
(3) Zero-shot Interpolation and Prediction: By modulating the score function of the reverse diffusion process, the prior knowledge from the reconstruction task is transferred to video frame interpolation and prediction tasks without additional training, enabling unified handling of "reconstruction-interpolation-prediction".

2. Core Technical Innovations An event-based inter-frame residual guidance strategy is proposed. Theoretical proof demonstrates that its gradient aligns with the tangent space of the data manifold learned by the diffusion model, ensuring the optimization process does not compromise generation quality. A score function modulation method is designed, which uses the latent variable deviation of reference frames to correct the sampling process, enhancing the temporal consistency and visual fidelity of interpolation/prediction tasks.

3. Data Construction and Experimental Validation Training data is synthesized from real-world videos, and tests are conducted on both real-world and synthetic datasets. Extensive experiments and ablation studies verify that UniE2F outperforms most existing state-of-the-art (SOTA) models.

### Strengths
1. This paper proposes a systematic three-stage event-to-frame reconstruction framework: Event-Based Video Frame Reconstruction → Inter-Frame Residual Prediction Training → Inter-Frame Temporal Residual Guidance. Ablation studies validate the effectiveness of each component. Innovatively integrating the generative prior of the pre-trained Stable Video Diffusion (SVD) model with the physical characteristics of event data, the method achieves unified handling of "video frame reconstruction-interpolation-prediction" through three core modules: event-conditioned fine-tuning, inter-frame residual guidance, and score function modulation. It breaks through the limitation of traditional methods confined to single tasks and can adapt to interpolation and prediction tasks in a zero-shot manner without additional training.

2. An event-based inter-frame residual guidance mechanism is designed. It predicts inter-frame residuals via ResNet and optimizes the latent variables of the diffusion model combined with gradient descent. Meanwhile, it is theoretically proven that this gradient aligns with the tangent space of the data manifold, ensuring the optimization does not compromise generation quality and the reconstruction error is bounded. This effectively enhances the temporal consistency and detail accuracy of the reconstructed frames.

3. Relevant experiments verify the effectiveness of the proposed modules. Compared with other methods, UniE2F achieves state-of-the-art (SOTA) performance.

4. The paper is logically structured and easy to understand.

### Weaknesses
1. Testing is only conducted on sequences extracted from TrackingNet and HS-ERGB, without validation on other datasets, making it impossible to demonstrate the true effectiveness and generalization of the method.

2. Specific details of the used Stable Video Diffusion (SVD) are not provided, such as the pre-trained model employed, parameter count, and other relevant specifications.

3. The originality is insufficient. From a methodological perspective, introducing residual guidance optimization is one of the core innovations of the paper, but the ablation experiments show that its improvement on performance is not significant.

4. Figure 2 (the method framework) could be more detailed. For example, relevant components of the training phase should be added.

5. Although the method achieves state-of-the-art (SOTA) performance on the synthetic test set, it exhibits suboptimal performance in most cases on real-world datasets.

6. Compared with suboptimal methods, while this method achieves certain performance improvements, its computational cost and memory footprint are several times or even dozens of times higher than those of traditional methods.

### Questions
1. The paper only verifies performance on the TrackingNet and HS-ERGB datasets, without involving datasets for extreme scenarios such as low light and fast motion. How robust is the method in such complex scenarios? Will performance degrade due to a sudden increase in event sparsity or noise interference?

2. Regarding the used SVD model, the network structure details and parameter count are not specified. Could you provide supplementary explanations? Have you tried SVD models with different parameter counts?

3. Could the method be trained on real-world datasets and verified for effectiveness on real-world datasets?

4. Is the way of encoding event data into 3-channel tensors (sum of all events, sum of positive events only, sum of negative events only) the optimal choice? Have you tried temporal dimension encoding (e.g., event occurrence frequency)?

### Soundness
2

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
4

### Summary
This paper proposes an event-to-frame reconstruction method based on a stable video diffusion model. The approach leverages the physical correlation between event streams and video frames to guide and enhance the quality of event-based reconstruction. Furthermore, the authors extend this method to video interpolation and prediction tasks, with experimental results demonstrating its effectiveness.

### Strengths
1. The proposed method achieves strong performance in event-based reconstruction, interpolation, and frame prediction tasks, outperforming previous approaches.

2. The paper presents comprehensive results, including quantitative comparisons, images, videos, and animations.

### Weaknesses
1. Unfair comparison. The proposed UniE2F is fine-tuned from the Stable Video Diffusion (SVD) model, which itself has been pre-trained on large-scale datasets covering tasks such as video generation and reconstruction. In contrast, the comparison methods were trained only on synthetic datasets, making the comparison potentially unfair. Since UniE2F naturally benefits from SVD’s strong pretrained prior, the authors should provide results without pretrained weights to enable a fairer evaluation.

2. Excessive computational cost. As shown in Table 8, UniE2F requires orders of magnitude (up to 1000×) more computation than other methods, while offering limited performance improvement. This greatly undermines the advantages of event cameras, such as low power consumption and high temporal resolution. The authors should consider introducing more efficient strategies—for example, knowledge distillation or transfer learning—to significantly reduce computational cost while maintaining reconstruction quality.

3. Lack of real event camera data. All experiments were conducted on synthetic datasets, which exhibit a clear domain gap from real-world event data. The authors should evaluate their method on real datasets such as HQF, IJRR, and MVSEC to demonstrate robustness and generalization.

4. Lack of diversity in event representations. In event-based reconstruction, researchers commonly use voxel grids as event representations. The authors adopt a different representation but do not explain the rationale. Moreover, how do various representations—such as EST, ECM, and Voxel Grid—affect the reconstruction results? The authors should discuss this.

5. The authors use a ResNet to predict inter-frame residuals, but given that event cameras capture data with high temporal resolution, they can theoretically provide event information for any time interval. Why predict intermediate residuals instead of directly aggregating events between frames? The authors should clarify this design choice.

6. In line 180, the authors mention using a three-channel event representation. What are the advantages and underlying rationale for this choice? The justification should be provided.

7. Since the pretrained SVD model can already perform video frame interpolation (VFI) and video frame prediction (VFP) tasks using image inputs alone, it appears that the event data serves only as an auxiliary input. The authors should report results showing how UniE2F performs with only image input or only event input in VFI and VFP tasks, to better illustrate the true contribution of event information.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

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
This paper proposes UniE2F, a unified diffusion-based framework for reconstructing, interpolating, and predicting video frames from event camera data. UniE2F leverages a pre-trained video diffusion model and introducing an event-based residual guidance mechanism. The framework treats event-to-frame reconstruction as a conditional generation process and achieves state-of-the-art performance on both synthetic and real datasets. The approach demonstrates strong generalization across multiple event-driven vision tasks while maintaining a coherent generative formulation.

### Strengths
1. Novel residual-guided conditioning: The proposed event-based inter-frame residual guidance is an elegant and physically interpretable mechanism that connects asynchronous event streams with frame-level brightness changes. By explicitly modeling the residuals between consecutive latent frames, the diffusion process receives direct gradient cues from event dynamics, leading to sharper textures and temporally consistent reconstructions.

2. Unified generative formulation: The same residual-guided diffusion framework can handle event-to-frame reconstruction, interpolation, and prediction without retraining or architectural modification. This unification is conceptually clean and practically useful.

3. Zero-shot frame interpolation and prediction: The framework is extended to perform zero-shot video frame interpolation and future frame prediction. By modulating the reverse diffusion sampling process, the same architecture can handle not only reconstruction but also interpolation and prediction tasks, demonstrating flexibility and strong generalization without additional training.

### Weaknesses
1. Limited methodological clarity: The paper proposes a residual-guided diffusion mechanism but provides insufficient details on its implementation. In particular, the normalization of the residual signal, its integration within denoising steps, and its weighting against the diffusion prior remain unclear, limiting reproducibility and interpretability.

2. Computational inefficiency: The framework relies on a pre-trained video diffusion backbone, which is typically expensive in both computation and memory. The paper lacks a detailed analysis of inference speed, GPU memory usage, and scalability to long sequences or real-time deployment.

3. Limited comparison with recent baselines: The paper does not include a comparison with RE-VDM or other recent state-of-the-art event-to-video diffusion methods. Including these baselines would provide a clearer picture of the method's relative performance and strengthen the empirical evaluation.

### Questions
1. Loss weighting: If the residual contributes to the loss, what is the weighting factor s and how sensitive is the performance to different s values? Table 3 only shows comparisons for s = 0 and s = 0.1. Have the authors conducted a more thorough ablation study to evaluate the effect of this weighting?

2. Choice of linear guidance schedule in Table 3:   In Table 3, the paper compares different guidance strategies, including linear increasing and decreasing residual guidance. Could the authors clarify the motivation for adopting a linear schedule?  Was this choice empirically found to be optimal, or is it mainly for simplicity?  Have other non-linear schedules (e.g., constant, exponential, cosine) been tried, and how do they affect reconstruction quality or temporal consistency?

3. Generalization and robustness: How does the method perform under extreme lighting, very sparse event streams, or noisy events? Are there failure cases, and what types of motions or event patterns cause degradation?

### Soundness
2

### Presentation
3

### Contribution
3
