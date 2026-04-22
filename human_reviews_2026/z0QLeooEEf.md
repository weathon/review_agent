# ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Existing multi-view 3D object reconstruction methods heavily rely on sufficient overlap between input views, where occlusions and sparse coverage in practice frequently yield severe reconstruction incompleteness. Recent advancements in diffusion-based 3D generative techniques offer the potential to address these limitations by leveraging learned generative priors to ‘‘hallucinate" invisible parts of objects, thereby generating plausible 3D structures. However, the stochastic nature of the inference process limits the accuracy and reliability of generation results, preventing existing reconstruction frameworks from integrating such 3D generative priors. In this work, we comprehensively analyze the reasons why diffusion-based 3D generative methods fail to achieve high consistency, including (a) the insufficiency in constructing and leveraging cross-view connections when extracting multi-view image features as conditions, 
and (b) the poor controllability of iterative denoising during local detail generation, which easily leads to plausible but inconsistent fine geometric and texture details with inputs. Accordingly, we propose ReconViaGen to innovatively integrate reconstruction priors into the generative framework and devise several strategies that effectively address these issues. Extensive experiments demonstrate that our ReconViaGen can reconstruct complete and accurate 3D models consistent with input views in both global structure and local details.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
1. **Originality-wise**: The paper introduces a novel framework that uses implicit, 3D-aware features from a reconstruction network (VGGT) to condition a diffusion generator (TRELLIS) , and a rendering-aware feedback loop (RVC), correcting the denoising trajectory during inference. It represents a new and sophisticated approach to solving the trade-off between reconstruction completeness and generative accuracy.

2. **Quality-wise**: The claims are substantiated with a robust and comprehensive evaluation on challenging benchmarks like Dora-Bench and OmniObject3D.

3. **Clarity-wise**: The manuscript is clearly written, with well-structured methodology, detailed explanations, and intuitive visualizations that enhance understanding.

### Strengths
1. Clear problem formulation and strong motivation: The paper accurately identifies the central conflict in modern 3D reconstruction between the "incompleteness of reconstruction" and the "inaccuracy of generation." The analysis of the root causes—such as insufficient cross-view correlation and poor controllability of the generation process—provides a solid foundation for the proposed method.

2. Innovative and well-Designed framework: ReconViaGen is more than a simple cascade of reconstruction and generation models. It achieves a deep fusion of the two priors by integrating the feature-level output of the reconstructor directly into the conditioning mechanism of the diffusion generator. The coarse-to-fine strategy, guided by separate global and local conditions, is a logical and effective way to leverage multi-level reconstruction cues.

### Weaknesses
This paper is good writing and well visualized, but still with some minor weakness:

1. Concern about computational cost: While highly effective, the RVC mechanism requires 3D decoding, rendering, loss computation, and gradient calculation at multiple steps of the diffusion inference process. This will undoubtedly add significant computational overhead. The paper does not provide a detailed analysis of this overhead, which could be a potential bottleneck for applications requiring fast inference. A discussion on the trade-off between speed and accuracy would be beneficial.

2. The effectiveness of the RVC mechanism is critically dependent on the accuracy of the estimated camera poses. In challenging sparse-view scenarios where pose estimation is inherently less reliable, any inaccuracies will directly corrupt the rendering-based loss, providing erroneous guidance to the diffusion model. This risks actively misleading the denoising trajectory rather than correcting it. The paper lacks a crucial robustness analysis to quantify the sensitivity of the final reconstruction quality to errors in camera pose estimation.

---

I have listed my concerns, and the score will be adjusted based on the author's response.

### Questions
Please refer to Weaknesses part.

### Soundness
2

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
5

### Summary
This paper proposes to incorporate VGGT [2] features into Trellis [1] generation process via simple cross attention in order to improve the adherence of the Trellis Generators (SS and SLAT) to the provided input views.

This paper demonstrates an interesting insight towards how the 3D global information aggregated by VGGT is more meaningful to the Trellis denoisers compared to providing the raw unprocessed RGB images directly, however it falls short of shedding a scientific light onto why this is the case. Moreover, the important high-frequency details in their results actually come from a highly empirical inference-time iterative refining step (RVC) which almost doubles the generation time (at the least, details missing), and not from the featurewise cross connection of VGGT and Trellis. Thus this research raises concerns about the actual utility of the proposed mutli-step pipeline. More detailed concerns in the Weakness section. 



[1] Xiang, J., Lv, Z., Xu, S., Deng, Y., Wang, R., Zhang, B., ... & Yang, J. (2025). Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 21469-21480).

[2] Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., & Novotny, D. (2025). Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 5294-5306).

### Strengths
1. The authors combine two very strong recent works VGGT and TRELLIS with a clean Cross-Attention module into an E2E pipeline which yields decent improvements in the results. 

2. The paper is presented in a clear and lucid manner.

### Weaknesses
The paper however raises the following concerns:

**Major Concerns**:
1. The RVC step contributes the most amount of visual details to the generated results, and as such, this step lacks sufficient details and discussion in the paper. Most importantly, how does an RVC only baseline fare against the `(d)-full` version of the method. For the Rendering aware correction, something simple as the predicted point-clouds of VGGT or even sparse reconstructions from COLMAP are meaningful because in this version no training is needed. Moreover, it would be quite important for this paper to present details about how the hyperparameters of the final step are tuned, and whether per-asset generation tuning is needed of if some fixed values could be set in the pipeline and then generalised for inference. Lastly, how much generation latency does the RVC step add to the full pipeline? 

2. As alluded to earlier in the summary, imho, a discussion about why the global features from VGGT help the Trellis generation so much is imperative for this paper. Of course we can say that the learned features aggregated by VGGT through the regressive prior capture much more 3D structural information than just the 2D multi-views, but then the question that rises is how come the generative 3D prior implicitly learned by Trellis is not enough? If it's possible to entirely use Trellis (through finetuning), then the solution becomes even more cleaner than the current one which integrates two different large-scale foundation models. 

**Minor concerns**:
1. Are the two ConditionNets for the global condition as well as the local per-view condition tokens the same? This is not clarified properly in the draft as the Mathematical notation is the same for the two, and this is not clarified explicitly. 

2. What is the intuition behind the weighted fusion of the per-view features cross-attention conditioning? Just for computational efficiency? Details are missing for this particular operation. How can the MLP predict from just the current' view's features which of them are important and which aren't when fusing multiple views? Why not just add all the features together and them apply a linear layer/MLP? Ideally, the simplest operation should be applying a cross-attention with the KV cache coming from all the views, but might cause OOM. Clarification from the authors would be appreciated. An ablation on this would make the paper more well rounded.

3. How is the SLAT-FLow transformer trained? Similar to the SS-FLow? Implementation Details paragraph lines 350-359 do not cover this explicitly.

4. What is the total time E2E time required for the 3D model generation starting from Multi-view inputs to receiving the final 3D model output? How much the RCV contribute to the generation latency? A detailed latency table would be useful for demonstrating the utility of the final proposed pipeline. 

5. I think it would be very beneficial to include more qualitative examples for the (c) v/s (d) ablation study. It will be very insightful to understand where exactly the ~1.5 dB improvement is coming from.

### Questions
Responses to the Major Concerns 1 and 2 would be most informative to me in making the final decision in the discussion phase. 

**Minor Suggestions**:
1. The visual metrics (PSNR, LPIPS and SSIM) for the VGGT baseline, however low, will be helpful for completeness of the scientific study. They have been possibly left out because of the low numbers coming from the holes in the renders of VGGT, but not detailed why exactly they have been left out. Clarification appreciated.

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
4

### Summary
The paper proposes ReconViaGen, a coarse-to-fine framework that integrates pose-free multi-view reconstruction priors (VGGT) with diffusion-based 3D generative priors (TRELLIS) for accurate and complete 3D object reconstruction from sparse, uncalibrated views. The method: (1) builds multi-view-aware conditions from VGGT via a ConditionNet, forming a Global Geometry Condition (GGC) for coarse Structure (SS) Flow and Per-View Conditions (PVC) for fine SLAT Flow; (2) introduces a rendering-aware velocity compensation (RVC) that, during inference, adjusts rectified-flow velocities using SSIM/LPIPS/DreamSim losses between rendered views and inputs; (3) refines camera poses by registering inputs into the generator’s space through rendering, VGGT estimation, feature matching, and PnP-RANSAC. Experiments on Dora-Bench and OmniObject3D show consistent SOTA improvements on image consistency (PSNR/SSIM/LPIPS), geometry (CD), and completeness (F-score), with ablations isolating the benefits of GGC/PVC/RVC and robustness to number/quality of views.

### Strengths
1.	Injects reconstruction priors (VGGT) into a 3D diffusion generator, addressing a well-known failure mode of existing generative methods—global structure drift & local inconsistency.
2.	The clear division of labor and strong motivation: GGC for structural cues and PVC for fine details match information granularity.
3.	RVC corrects the diffusion velocity field with differentiable rendering feedback at inference, simple to implement yet clearly improves input consistency.
4.	Broad evaluation with both image-consistency and geometry metrics plus pose accuracy; ablations are comprehensive and convincing.

### Weaknesses
1.	While results support “GGC is more suitable for SS and PVC is more suitable for SLAT,” deeper interpretability is missing.
2.	RVC hyperparameter/stability analysis is light: sensitivity to α, timestep threshold (t < 0.5), and the outlier rejection threshold (0.8) is not quantified; inference-time overhead (decoding/rendering per step) is not reported.
3.	Pose refinement pipeline is engineering-heavy.
4.	RVC is inference-only, creating a potential train–inference gap.
5.	Some theoretical clarity on how RVC interacts with rectified flow objectives is missing; decoding to different 3D representations (RF/3DGS/Mesh) and their effect on metrics could be further analyzed.

### Questions
1.	Provide attention visualizations and cross-view token similarity to show GGC’s global aggregation in SS and PVC’s local correspondence in SLAT?
2.	Conduct “Information bottleneck” experiments in SS (vary GGC token capacity) to study structure convergence and downstream SLAT performance?
3.	Provide sensitivity studies for RVC: α, timestep scheduling (t threshold), loss weights, and the 0.8 outlier discard rule? How do these affect convergence, runtime, and stability?
4.	Whether RVC, being inference-only, creates a potential train–inference gap. Why not introduce a lightweight RVC-like self-distillation during training to reduce distribution shift?
5.	Report performance on real multi-view captures (e.g., DTU object subset or an in-house dataset) to validate real-world generalization?

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
This paper proposes the ReconViaGen framework, which aims to combine strong reconstruction priors (from VGGT) with diffusion-based generative priors (from TRELLIS) to address the limitations of pure reconstruction methods that lead to incomplete results and pure generative methods that, while complete, lack consistency with the input view. The method employs a coarse-to-fine strategy: using Global Geometric Conditions (GGC) to guide coarse structure generation and Per-View Conditions (PVC) to guide fine detail generation. Additionally, a Rendering-aware Velocity Compensation (RVC) mechanism is introduced, which is used only during inference to ensure pixel-level alignment of the generated results with the input view. Experiments on Dora-Bench and OmniObject3D datasets demonstrate that this method achieves SOTA levels in both completeness and consistency.

### Strengths
- The paper provides clear and detailed descriptions of the methodology and experimental design, making it easy to understand and replicate. 

- The proposed ReconViaGen framework effectively addresses the limitations of pure reconstruction and pure generative methods by combining their strengths. The coarse-to-fine strategy, utilizing Global Geometric Conditions (GGC) for coarse structure and Per-View Conditions (PVC) for fine details, allows for a more accurate and consistent 3D reconstruction from single images. The results on Dora-Bench and OmniObject3D datasets demonstrate significant improvements in both completeness and consistency compared to state-of-the-art methods.

### Weaknesses
- The requirements for multi-view inputs may limit the applicability of the method in scenarios where only single-view inputs are available. In practical applications, users often can only provide a single view. It is unclear how the method would perform with single-view inputs, and the paper does not discuss or analyze this aspect.
- In Section 4.3, the ablation study does not provide visual comparisons to illustrate the effects of different modules, relying solely on numerical results to demonstrate their effectiveness. This makes it difficult for readers to intuitively understand the impact of each module on the final results.

### Questions
- The paper uses multi-view inputs as conditions for generation, but the results do not seem significantly better than current single-view 3D model generation methods (like Direct3D-S2[1]). Have the authors tried using single-view inputs as conditions for generation? If so, what were the results? I think it would be interesting to see how the model performs with single-view inputs compared to multi-view inputs.
- When using VGGT as the feature extractor for multi-view inputs, if the generated multi-view images are inconsistent, how much does this affect the quality of the final 3D model?

[1] Direct3D‑S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention

### Soundness
3

### Presentation
2

### Contribution
3
