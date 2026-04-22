# FreeViS: Training-free Video Stylization with Inconsistent References

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Video stylization plays a key role in content creation, but it remains a challenging problem. Naïvely applying image stylization frame-by-frame hurts temporal consistency and reduces style richness. Alternatively, training a dedicated video stylization model typically requires paired video data and is computationally expensive. In this paper, we propose FreeViS, a training-free video stylization framework that generates stylized videos with rich style details and strong temporal coherence. Our method integrates multiple stylized references to a pretrained image-to-video (I2V) model, effectively mitigating the propagation errors observed in prior works, without introducing flickers and stutters. In addition, it leverages high-frequency compensation to constrain the content layout and motion, together with flow-based motion cues to preserve style textures in low-saliency regions. Through extensive evaluations, FreeViS delivers higher stylization fidelity and superior temporal consistency, outperforming recent baselines and achieving strong human preference. Our training-free pipeline offers a practical and economic solution for high-quality, temporally coherent video stylization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes FreeViS, a training-free video stylization framework that effectively addresses the propagation issues in reference-based stylization. The key idea is a dual-branch design that incorporates high-frequency compensation, additional inconsistent references, and optical-flow-guided masked attention. The method can be applied to both image-based V2V stylization and stylized T2V generation, demonstrating higher stylization fidelity and competitive temporal consistency compared to prior training-free approaches.

### Strengths
- The paper identifies and analyzes the root causes of temporal artifacts in existing training-free stylization, and proposes a well-motivated dual-branch design to address them.
- As a training-free framework, the method is practical.
- The method naturally extends to both video-to-video stylization and stylized text-to-video generation, which increases practical value.
- Compared with existing baselines, the stylized outputs show richer texture, stronger style faithfulness, and fewer propagation errors, supported by convincing qualitative comparisons.

### Weaknesses
- While qualitative results are strong, the quantitative metrics in Table 1 and Table 2 are not consistently better than competing methods.
- The ablation study is limited, showing only a single metric and lacking a broader evaluation across other metrics.
- The dual-branch pipeline leads to higher memory usage and longer generation time, yet computation efficiency is only briefly mentioned in the appendix. A quantitative comparison of VRAM usage and runtime is needed.
- Several design choices, such as reference sampling frequency, number of references, and optical-flow strength, are not thoroughly analyzed. A sensitivity study would make the method's robustness and design justification clearer.

### Questions
- Why is the video consistency score in Table 1 lower than baselines? Why is style consistency in Table 2 weaker than StyleCrafter? A brief explanation in the paper would improve the reader’s understanding.
- How much slower or heavier is the dual-branch pipeline compared to comparison method? Please provide generation time and VRAM usage comparisons.
- How does performance change when sampling references more or less frequently? Does the choice of image stylization method for reference frames affect results? A sensitivity analysis would strengthen the claims.
- How do other metrics change across ablations, beyond the single metric currently reported? Showing full-metric ablation tables will better support the effectiveness of each component.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents FreeViS, a training-free video stylization framework that leverages a pre-trained image-to-video model, multi-reference integration, and high-frequency compensation to generate stylized videos with rich details and strong temporal coherence.

### Strengths
The main innovations are:
1. Training-Free Multi-Reference Integration: A novel pipeline that ingeniously integrates multiple stylized references into a pre-trained I2V model, effectively suppressing error propagation and eliminating flickering without any fine-tuning.

2. High-Frequency Compensation for Layout & Motion Preservation: A dedicated mechanism that utilizes high-frequency signals to constrain content structure and motion dynamics, ensuring superior temporal consistency.

3. Flow-Based Texture Preservation: The innovative use of optical flow motion cues to actively maintain and preserve style textures, enhancing the overall style richness and visual quality.

### Weaknesses
1. The proposed method's robustness is intrinsically tied to the accuracy of the optical flow algorithm. Inaccuracies in flow estimation are likely to propagate through the system, potentially leading to residual temporal flickering, which represents a fundamental limitation of the current framework.

2. A significant concern is the ambiguous contribution of the base model versus the novel components. The observed temporal coherence in the stylized videos may be predominantly attributable to the inherent strong temporal consistency of the underlying video generation model, rather than the proposed technique itself. This makes it difficult to isolate and validate the actual improvement brought by the authors' method.

### Questions
1. Could you provide more details on how the system handles potential errors from the optical flow computation? Specifically, are there any mechanisms to correct or compensate for flow inaccuracies to prevent visible flickering in the final output?

2. Regarding Equation 2, how was the cutoff frequency for the low-pass filter determined? Could you discuss the sensitivity of the final results to this parameter and whether an ablation study was conducted to guide its selection?

### Soundness
3

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
3

### Summary
This paper proposes FreeViS, a training-free video stylization framework that aims to solve temporal inconsistency (flickering) in frame-by-frame methods and propagation error in single-reference I2V-based methods. The key innovation is integrating multiple stylized reference frames into a pretrained I2V model through three main technical components: 1. Indirect High-Frequency Compensation (IHC) that preserves layout/motion while maintaining stylized colors, 2. an isolated attention mechanism with dynamic injection and optical flow-based masking to handle inconsistent references, and 3. Explicit Optical Flow Guidance (EOG) to preserve textures in low-saliency regions. Experiments demonstrate improvements over existing methods in both stylization quality and temporal consistency.

### Strengths
- The IHC mechanism is well-motivated by frequency analysis showing high frequency components encode layout/motion while low frequency components control appearance.
- Solve the multi-reference challenges, dynamic injection addresses stuttering, flow-based masking resolves inconsistencies.
- The qualitative results in Figures 1, 4, 5, 6, 16, and 17 are very convincing and show a clear improvement in style propagation and temporal consistency over baselines.
- The quantitative evaluations and ablation study  are comprehensive and support the authors' claims.
- Clear improvements over baselines in both video-to-video stylization and stylized text-to-video generation tasks

### Weaknesses
- Missing hyperparameter ablation studies and analysis, the method combines three attention modes using hyperparameters β and γ (Eq. 8), but no ablation or analysis is provided. How sensitive is the final video quality to these parameters? Additionally, γ is only non-zero in the "final stage" but this stage is not clearly defined, what threshold determines this?
- No inference runtime comparisons with baselines are provided. This makes it difficult to assess the practical computational cost relative to quality gains.
- Reference frame selection not justified or ablated. The paper uses first, middle, and last frames, calling this an "empirically effective choice" without systematic study. Why specifically these 3 frames? How does this compare to different numbers of references? What is the performance vs. efficiency trade-off with more references?

### Questions
- Can you provide ablation studies on hyperparameters β and γ? How is the "final stage" for γ activation defined?
- Can you provide inference runtime comparisons with baselines?
- Can you justify the choice of 3 reference frames and provide ablations with different numbers/positions of references?

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
The paper introduces FreeViS, a training-free video stylization framework that addresses the challenges of temporal consistency and style richness in video generation. Instead of relying on paired video data or fine-tuning, FreeViS integrates multiple stylized reference images into a pretrained image-to-video (I2V) model. Its design incorporates indirect high-frequency compensation to refine content layout and motion fidelity, multi-reference integration to reduce flickering and propagation errors, and explicit optical-flow guidance to maintain style textures in low-saliency regions. The authors claim that FreeViS outperforms recent baselines in both stylization quality and coherence.

### Strengths
1. Originality: The idea of combining inconsistent stylized references with a pretrained I2V model in a training-free manner is practical and relevant, especially for low-resource settings.
2. Quality: The method is well-motivated and shows promising results in terms of stylization fidelity and temporal coherence. The integration of high-frequency compensation and flow cues is conceptually sound.
3. Clarity: The paper is generally easy to follow, with clear motivation and methodology. Figures and examples help illustrate the core ideas.
4. Significance: FreeViS addresses a real-world need for efficient and high-quality video stylization without the burden of training, making it potentially impactful for content creators and researchers.

### Weaknesses
1. Limited Novelty: Several components (e.g., AdaIN, cross-frame attention, flow-based guidance) have been used in prior works like Style Injection in Diffusion, StyleID, and Text-to-Video Zero. The novelty lies more in the integration to video diffusion model than in individual techniques.
2. Unclear Use of Multiple References: Although the paper claims to use multiple stylized references, the experiments and visualizations consistently show only a single style image. There is no explanation of how multiple references are selected, fused, or weighted.
3. Insufficient Evaluation of Key Components: The Indirect High-Frequency Compensation is mentioned but not thoroughly analyzed. The ablation image lacks accompanying discussion, making it hard to assess its contribution.
4. Optical Flow Reliability: The use of flow in low-saliency regions is promising, but flow errors can degrade detail preservation, especially for moving objects. Figure 7 shows only one example, and it’s unclear whether it involves motion.
5. Style Diversity: The method is tested on relatively simple styles. It’s unclear how well it generalizes to complex or abstract styles with texture details.

### Questions
1. Could you clarify how multiple stylized references are used in practice? 
2. How does the method handle inaccurate optical flow, especially in fast-moving scenes or occlusions? 
3. Have you tested FreeViS on more diverse styles with texture details (e.g. oil painting with strokes) or content with texture details (e.g. hair)? If so, how does it perform in terms of fidelity and consistency?

### Soundness
3

### Presentation
3

### Contribution
2
