# Hi-Light: A Path to high-fidelity, high-resolution video relighting with a Novel Evaluation Paradigm

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Video relighting offers immense creative potential and commercial value but is hindered by challenges, including the absence of an adequate evaluation metric, severe light flickering, and the degradation of fine-grained details during editing. To overcome these challenges, we introduce Hi-Light, a novel, training-free framework for high-fidelity, high-resolution, robust video relighting. Our approach introduces three technical innovations: lightness prior anchored guided relighting diffusion that stabilises intermediate relit video, a Hybrid Motion-Adaptive Lighting Smoothing Filter that leverages optical flow to ensure temporal stability without introducing motion blur, and a LAB-based Detail Fusion module that preserves high-frequency detail information from the original video. Furthermore, to address the critical gap in evaluation, we propose the Light Stability Score, the first quantitative metric designed to specifically measure lighting consistency. Extensive experiments demonstrate that Hi-Light significantly outperforms state-of-the-art methods in both qualitative and quantitative comparisons, producing stable, highly detailed relit videos.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents Hi-Light, a training-free framework for high-fidelity and high-resolution video relighting.  
It introduces three modules: lightness-prior anchored guided diffusion for luminance stability, Hybrid Motion-Adaptive Lighting Smoothing Filter for flicker reduction, and LAB-based Detail Fusion for detail recovery, along with a new metric Light Stability Score (SLS) for evaluating temporal lighting consistency.  
Experiments show that Hi-Light achieves significantly better stability and detail preservation than prior methods such as Light-A-Video and TC-Light on the proposed metric.

### Strengths
1) Clear motivation: addresses flickering and detail loss in video relighting, both key unsolved problems.  
2) Extensive experiments and ablations demonstrate strong performance gains over prior methods in the proposed benchmark.
3) Easy to follow, well-structured, and reproducible with provided code and detailed analysis.

### Weaknesses
**Major Weaknesses**  

1) The evaluation metric design is biased. As stated in L414–415 (“It is notably close to the original input video, which represents the ideal target for these metrics”), the chosen SSIM and consistency metrics mainly favor temporal coherence and similarity to the input video rather than the quality of relighting. This is a fundamental flaw for a video relighting paper.  

2) The LAB fusion may compromise relighting quality. As shown in Fig. 8 (top row), the “LAB-DF Fused video” appears worse than the smoothed and intermediate relit videos. Directly fusing the L channel mainly preserves global tone but loses illumination contrast and shadow variation, thereby reducing the realism of relighting.  

3) The proposed metric lacks novelty and is not fully justified. SSIM is widely used, and the consistency score, obtained by thresholding grayscale frames and measuring average brightness and bright pixel counts, does not effectively capture temporal consistency. A comparison with existing metrics (e.g., consistency and smoothness in VBench [1]) is necessary to clarify its advantage.  

4) The HMA-LSF module relies on optical flow warping, which can be noisy. Although filtering alleviates some artifacts, large-motion cases where optical flow fails should be analyzed and visualized to assess robustness.  

5) Only one visual comparison (Fig. 5) is shown, and no video comparison results are provided. In Fig. 1 (second row), the relighting quality of the woman’s right hand in the last frame is also unsatisfactory. For a video relighting or editing task, more qualitative and video-based results are essential to convincingly demonstrate performance. More qualitative and dynamic visual examples are needed to support the claims. In addition, all current examples appear to use static-camera video inputs; results under larger camera motion should also be presented to evaluate robustness.

**Minor Weaknesses**  

6) The method contains many hyperparameters, and it is unclear whether they are fixed globally or tuned per scene. Robustness across diverse videos should be further analyzed.  

7) Runtime comparisons with other methods are missing.  

8) The paper lacks discussion on limitations and potential future work.   

9) Figure presentation issues: Fig. 3 includes unexplained icons, and its caption could be more detailed. In Fig. 5, method names could be properly aligned.  

**Reference**   

[1] VBench: Comprehensive Benchmark Suite for Video Generative Models, CVPR 2024

### Questions
1) Could the authors provide relighting quality metrics, such as those used in Light-A-Video [1] and other existing consistency metrics like VBench [2], to better evaluate lighting realism and temporal stability?  
2) Can more visual comparisons with baseline methods be added, including qualitative and video-based examples, to clearly demonstrate improvements in video relighting?  
3) Could the authors include runtime comparisons with baseline methods?

**Reference**   

[1] Light-A-Video: Training-free Video Relighting via Progressive Light Fusion, ICCV 2025

[2] VBench: Comprehensive Benchmark Suite for Video Generative Models, CVPR 2024

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a training-free, high-resolution video relighting framework designed to generate relit videos without degrading high-frequency details from the input and while maintaining better temporal stability. Specifically, the paper introduces a lighting prior as auxiliary information to reduce luminance fluctuations at low resolution, a hybrid motion-adaptive light smoothing filter to mitigate flicker, and a LAB-preservation fusion strategy to recover high-frequency details. Additionally, the paper proposes an evaluation metric to measure temporal lighting stability.

### Strengths
- The presented relighting method appears to be a plug-and-play framework that can integrate seamlessly with various video diffusion models, making it highly practical for real-world applications. 
- The proposed approach supports high-resolution video relighting, which significantly enhances its utility for real-world use cases.

### Weaknesses
- The paper suggests that prior methods degrade high-frequency details such as hair or foliage; however, it does not provide sufficient qualitative evidence to demonstrate that the proposed approach retains these details more effectively in relit videos.
- The paper introduces a new lighting stability score along with an adapted SSIM metric. However, it does not include evaluations using existing metrics from prior work, which would enable a fairer comparison with baselines.
- The paper claims improved computational efficiency by performing relighting at low resolution. This claim would be stronger if runtime comparisons with baseline methods were provided to demonstrate computational effectiveness.
- The paper proposes leveraging optical flow to reduce temporal lighting flicker but does not provide details on how the optical flow is generated or evaluated for robustness.
- The paper explains the combination of low-resolution lightness with high-frequency details but does not clarify how LAB components are upscaled or how artifacts, if any, such as color bleeding, desaturation, and ghosting are mitigated.
- The paper does not address how hard shadows or highlights are handled when relighting under different lighting directions or sources. The presented qualitative results do not capture this scenario.
- Are there cases where the proposed pipeline fails to generate a satisfactory relit video? Including failure cases would provide a more comprehensive evaluation of the method’s limitations.
- Figure 3 is difficult to interpret, and the caption lacks sufficient detail to explain the framework. Adding relevant labels and descriptive terms to the figure would make the pipeline easier to understand.

### Questions
- The proposed metric primarily measures temporal lighting stability. How do you ensure color consistency over time? Why not evaluate with existing metrics such as VBench [a] for temporal flicker, motion smoothness, aesthetic quality, and background consistency?
- The paper states that \alpha in Eq. 7 adapts to motion magnitude but provides no qualitative analysis of its impact.

Additional references:

[a] Vbench: Comprehensive benchmark suite for video generative models. CVPR 2024.

### Soundness
2

### Presentation
1

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
The paper presents Hi-Light, a training-free, backbone-agnostic workflow for video relighting, motivated by observed limitations in existing methods: temporal lighting flicker, detail degradation in high-resolution content, and the absence of a metric dedicated to lighting stability. With the designed pipeline, the method uses multiple off-the-shelf video diffusion backbones to improve temporal stability and visual details. This method formalizes an evaluation setup and introduces the Light Stability Score (SLS) to quantify temporal smoothness alongside fidelity metrics such as SSIM, aiming to provide a more complete assessment of relighting quality. Empirical results indicate state-of-the-art performance, with reported SSIM of 0.943 and SLS of 0.509, and comparative analyses showing improvements over prior baselines; human studies are reported to correlate with these metrics, supporting the validity of the evaluation.

### Strengths
The method leverages properties of the color space to improve video stability. It requires no additional training and can be implemented with existing video diffusion backbones. The idea is elegant and, according to experimental results, effective.

The proposed new metric makes comparisons between methods systematic.

### Weaknesses
In the demo videos presented by the authors, there are no scenes with significant motion, and the experiments do not include a separate analysis or comparison of scene motion. Since the algorithm relies on optical flow, it is likely to be sensitive to motion. Therefore, it remains unclear whether the method can be applied to videos with larger or more complex motion.

Although the authors claim this is a relighting task, the control over the light sources is neither very fine-grained nor accurate. The authors mention that it  support controll of the direction of light. However, the corresponding figures and references cannot be found in the paper, and the existing figures also fail to effectively demonstrate the effect of controlling the light source direction. The current results appear more like a form of video stylization that emphasizes lighting differences.

### Questions
How much motion can the proposed method handle in the video?
During relighting, how can the direction of the light source be controlled? If it can be controlled, how fine is the level of control?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes a new video relighting framework called "Hi-Light" composed of three key components: i) a lightness-prior-based fusion scheme that mitigates luminance oscillations via diffusion; ii) two Plug-and-Play (PnP) filters called Hybrid Motion Adaptive Light Smoothness Prior (HMA-LSF) and LAB Detail-Preserving Fusion (LAB-DF) that remove flickers and restore texture respectively, and iii) a new quantitative metric call Light Stability Score for video relighting.

### Strengths
The paper is overall well written and the contributions are easy to understand. Video relighting is a challenging problem, where temporal flickering is a serious problem that is difficult to quantify and difficult to address. The proposed techniques are lightweight.

### Weaknesses
The contributions look like a concatenation of exceedingly simple techniques collected from different sources into a single framework for video relighting. Individually, each contribution is not particularly novel. Specifically, 

1. The light stability score (section 3.1.1) is an ad-hoc procedure based on brightness thresholding, which leads to three time-series signals that assess the video's light fluctuation. Overall it looks very ad-hoc (not grounded in theoretical analysis) and overly simplistic.  

2. To quantify detail preservation (section 3.1.2), Structural Similarity Index (SSIM) is simply adopted, a very well known image quality assessment metric. Equations (1) are (2) are standard SSIM. There is no novelty here.  

3. HMA-LSF (section 3.2.2) is a simple combination of optimal flow and bilateral filter (a primitive edge-preserving filter that dates back to 1990s). The blending operation in equation (7) is simplistic and does not vary across the frame (i.e., the same $\alpha$ is used throughout). 

4. LAB-DF (section 3.2.3) is a simple linear blending operation in equation (9).

### Questions
1. What are the key novelty for individual components, HMA-LSF and LAB-DF, above and beyond existing works, in flicker removal and texture restoration, respectively?

2. How sensitive is the HMA-LSF to the accuracy of the optical flow estimation? Bilateral filter is a combination of domain and rang filters. How are the two filters adjusted (via parameters $\sigma$'s in the Gaussian kernel) for optimal performance? Is parameter $\alpha$ in equation (7) automatically adjusted in a data-driven manner, or hand-tuned beforehand?

3. How is $\beta$ in equation (9) optimized?

### Soundness
2

### Presentation
3

### Contribution
1
