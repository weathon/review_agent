# Post-Training Quantization for Video Matting

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Video matting is crucial for applications such as film production and virtual reality, yet deploying its computationally intensive models on resource-constrained devices presents challenges. Quantization is a key technique for model compression and acceleration. As an efficient approach, Post-Training Quantization (PTQ) is still in its nascent stages for video matting, facing significant hurdles in maintaining accuracy and temporal coherence. To address these challenges, this paper proposes a novel and general PTQ framework specifically designed for video matting models, marking, to the best of our knowledge, the first systematic attempt in this domain. Our contributions include: (1) A two-stage PTQ strategy that combines block reconstruction-based optimization for fast, stable initial quantization and local dependency capture, followed by a global calibration of quantization parameters to minimize accuracy loss. (2) A Statistically-Driven Global Affine Calibration (GAC) method that enables the network to compensate for cumulative statistical distortions arising from factors such as neglected BN layer effects, even reducing the error of existing PTQ methods on video matting tasks up to 20%. (3) An Optical Flow Assistance (OFA) component that leverages temporal and semantic priors from frames to guide the PTQ process, enhancing the model’s ability to distinguish moving foregrounds in complex scenes and ultimately achieving near full-precision performance even under ultra-low-bit quantization. Comprehensive quantitative and visual results show that our PTQ4VM achieves the state-of-the-art accuracy performance across different bit-widths compared to the existing quantization methods. We highlight that the 4-bit PTQ4VM even achieves performance close to the full-precision counterpart while enjoying 8× FLOP savings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes PTQ4VM, a post-training quantization framework for video matting that aims to maintain temporal and spatial quality under low-bit quantization. The framework consists of three components: Block-wise Initial Quantization (BIQ) for stable block-level quantization; Global Affine Calibration (GAC) to compensate for post-quantization statistical shifts; and Optical Flow Assistance (OFA) to enforce temporal consistency using motion-guided priors.

### Strengths
1. The paper is well-written, easy to follow. 
2. The overall pipeline is simple, modular, and compatible with existing matting architectures, showing practical deployment potential.

### Weaknesses
1. How does the proposed BIQ method handle feature dependencies across quantized blocks? If each block is optimized independently, could quantization errors accumulate or disrupt global feature consistency?
2. The OFA module assumes accurate optical-flow alignment, but how robust is the framework when flow estimation is noisy or fails under motion blur and occlusion? Also, has the computational overhead of running optical-flow inference been measured?
3. Although the paper claims to be “training-free,” both BIQ and GAC require calibration data. What happens when the deployment domain differs from the calibration domain (e.g., lighting, background, or motion changes)?
4. The reported gains over existing PTQ baselines appear marginal. Based on ablation table, the effectiveness of OFA is quite unstable, does it caused by the unstable quality of flow estimation?

### Questions
See above weakness.

### Soundness
3

### Presentation
3

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
This paper presents PTQ4VM, a post-training quantization framework designed for video matting models.
It consists of three main components: BIQ (Block-wise Initial Quantization), which stabilizes quantization by optimizing scale and rounding at the block level; GAC (Global Affine Calibration), which corrects distribution shifts after BN folding through learnable global scaling and bias factors; and OFA (Optical Flow Assistance), which introduces an optical-flow-based regularization to improve temporal consistency between frames. Experiments on RVM and MatAnyOne demonstrate that PTQ4VM achieves near-FP32 accuracy under 4-bit quantization without retraining, while significantly reducing computational cost.

### Strengths
1. The paper addresses a new and underexplored problem—post-training quantization for video matting—where temporal stability is as crucial as spatial accuracy. While PTQ has been well studied for image-based tasks, its extension to temporally dependent applications is novel and practically meaningful, showing potential for efficient deployment in real-time video systems.

2. The overall framework design is coherent and systematic, combining block-wise, global, and temporal calibration stages (BIQ–GAC–OFA) in a way that builds logically from local to global stability. Each stage has a clear motivation and is supported by ablation experiments that justify its inclusion, giving the paper a well-engineered and reproducible structure.

3. The method shows strong empirical performance, maintaining almost FP32-level accuracy even at 4-bit precision. This demonstrates that the proposed calibration techniques effectively reduce quantization errors, supporting the claim that accurate low-bit inference can be achieved without any retraining.

4. PTQ4VM is evaluated on both RVM (CNN-RNN) and MatAnyOne (Transformer-based) architectures, suggesting the framework’s potential generality across different network structures. The consistent behavior across these models indicates that the proposed approach could be extended beyond matting to other temporally sensitive vision tasks.

### Weaknesses
1. Although OFA is proposed as the key contribution to improve temporal consistency, the empirical improvement on DTSSD is limited or inconsistent across experiments. This raises doubts about how much OFA truly contributes to stability, and whether its effect depends on the quality of the optical flow model used during calibration.

2. From a methodological standpoint, the work is largely incremental, extending ideas already explored in earlier PTQ methods such as BRECQ and bias correction. While the integration of these ideas into a single pipeline is well executed, it does not fundamentally change the underlying quantization paradigm or introduce new theoretical insights.

3. The baseline comparison is somewhat outdated, as it only includes earlier methods like BRECQ and QDrop. More recent PTQ techniques would provide a stronger context for assessing the relative contribution of PTQ4VM and its empirical advantage.

4. Although the paper claims that GAC stabilizes feature distributions, the evidence remains qualitative and lacks quantitative validation.
Additional analysis showing layer-wise mean or variance alignment before and after calibration would make the distribution-correction claim more convincing and scientifically grounded.

### Questions
1.
1-1. OFA is introduced to enhance temporal consistency, yet Table 2 shows several cases where DTSSD becomes worse after adding the module. Could the authors clarify whether OFA truly improves temporal stability or if the effect is within the variance of the metric?

1-2. Since RAFT may not capture the fine-grained pixel motions that are critical in video-matting scenarios, have the authors considered using a more accurate flow estimator such as GMFlow [1], or FlowFormer [2] for warping? It would be informative to see whether OFA’s limited impact stems from the quality of the flow model itself.

1-3. The paper employs DTSSD as the sole metric for temporal consistency, but this measure might not fully reflect perceptual flicker or long-term drift.Could the authors justify why DTSSD is the most appropriate choice, or consider including an additional metric?

2. The comparison includes BRECQ and QDrop, which are relatively old. It would strengthen the paper to include or at least discuss more recent PTQ methods such as GPTQ [3], SmoothQuant [4], or OmniQuant [5] to better position the contribution.

3. BIQ appears conceptually similar to BRECQ’s block-wise reconstruction. Could the authors clarify what concrete design difference—such as the optimization schedule, block partitioning, or objective weighting—makes BIQ more stable or effective?

4. The paper claims that GAC stabilizes intermediate feature distributions, but the evidence is mostly qualitative. Quantitative measurements of layer-wise mean and variance before and after calibration would make this claim more convincing.

[1] Xu, Haofei, et al. "Gmflow: Learning optical flow via global matching." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[2] Huang, Zhaoyang, et al. "Flowformer: A transformer architecture for optical flow." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.

[3] Frantar, Elias, et al. "Gptq: Accurate post-training quantization for generative pre-trained transformers." arXiv preprint arXiv:2210.17323 (2022).

[4] Xiao, Guangxuan, et al. "Smoothquant: Accurate and efficient post-training quantization for large language models." International conference on machine learning. PMLR, 2023.

[5] Shao, Wenqi, et al. "Omniquant: Omnidirectionally calibrated quantization for large language models." arXiv preprint arXiv:2308.13137 (2023).

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the performance degradation that occur when applying standard Post-Training Quantization (PTQ) methods to video matting models. To solve this, the authors propose PTQ4VM which integrates three key techniques: (1) Block-wise Initial Quantization (BIQ) for more stable and accurate optimization; (2) Global Affine Calibration (GAC) to correct statistical distribution shifts introduced by quantization; (3) Optical Flow Assistance (OFA), which uses temporal priors from adjacent frames to enforce smoothness during the calibration stage.

### Strengths
1. The paper is well-drafted, with a clear, logical flow that makes the motivations and contributions easy to follow.
2. The method delivers consistent accuracy gains over PTQ baselines under multiple bit-widths.

### Weaknesses
1. Flow errors (fast motion, occlusion, camera shake) may misguide calibration. The method uses RAFT (accurate but heavy) during calibration—calibration-time compute and wall-clock cost are not reported. Sensitivity to using lighter flow or imperfect flow is not analyzed.
2. The paper mentions that an "appropriate block partitioning" is used for the BIQ stage but does not go into detail about how these blocks are defined or if different partitioning strategies were explored.

### Questions
1. The calibration set used in the experiments is quite small (256 images). How sensitive is the performance of PTQ4VM to the size and content of the calibration set? Would using a larger or more diverse calibration set lead to further improvements?
2. Does OFA improve or degrade performance in scenarios with heavy occlusions or strong camera motion? Could you share failure cases and quantitative breakdowns?
3. Are there constraints or kernel support issues on common deployment backends (TensorRT, TFLite, CoreML) for W4A4?
4. Have you evaluated on additional video matting datasets or different frame rates/resolutions to test robustness to distribution shifts?

### Soundness
2

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
3

### Summary
The paper presents an approach for post-training quantization tailored for video matting models. The task is to compress a pre-trained video matting model from full 32-bit precision to 8- or 4-bit while retaining as much quality as possible. The authors propose improvements over prior approaches:
- A global affine calibration optimization to correct errors stemming from fusing batch normalization into weights before quantization. The quantization process introduces errors and shifts expected feature statistics, causing growing error in subsequent layers, and global affine calibration does global optimization to combat those shifts.
- An optical-flow-based alpha warping motion consistency loss to better guide the quantization process to improve video matting quality.

The authors opt for a block-wise quantization granularity to balance quality with significant memory requirements of video matting models.
The evaluation across two video matting datasets and three quantization schemes shows that the proposed approach consistently outperforms existing ones in the quality of the resulting model. An extra evaluation on MatAnyone shows that the proposed method generalizes across the base model architectures.

### Strengths
1. The authors clearly explain the approach and the motivation of different components.
2. The evaluations show that the proposed method convincingly outperforms the existing ones.
3. The authors provide extensive ablation studies in the appendix.
4. The quantization problem is important, especially for use cases like mobile video conferencing that uses video matting for the camera feed.

### Weaknesses
Major weaknesses:
1. The proposed optical-flow-based motion compensated alpha loss is hardly original. Video matting methods have been using it for their training for a while, so it’s a natural component to try in a quantization method tailored for video matting models.
2. The evaluation is limited to one video matting model (RVM), plus the second one (MatAnyone) in a limited evaluation in the appendix. Evaluating on more video matting methods would allow to more confidently judge the generalizability of the proposed quantization approach.
3. It would be good to see a subjective evaluation in addition to objective metric evaluation. The superiority of the quantized model to the FP32 model on some setups, as pointed out by the authors, suggests the limitations of the objective metrics in question.

Minor weaknesses:
1. Spaces are missing in many places around citations, e.g. line 54 “quantizationJacob”, line 58 “(2023)(QAT)”, line 80 “layersIoffe” etc.
2. Figures 1 and 2 are not referenced in the text.

### Questions
See weaknesses. Would be good to see:
- Evaluation on more video matting methods.
- Subjective evaluation.
- Fix formatting errors.

### Soundness
3

### Presentation
2

### Contribution
3
