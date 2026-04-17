# QuantV2X: A Fully Quantized Multi-Agent System for Cooperative Perception

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Cooperative perception through Vehicle-to-Everything (V2X) communication offers significant potential for enhancing vehicle perception by mitigating occlusions and expanding the field of view.  However, past research has predominantly focused on improving accuracy metrics without addressing the crucial system-level considerations of efficiency, latency, and real-world deployability. Noticeably, most existing systems rely on full-precision models, which incur high computational and transmission costs, making them impractical for real-time operation in resource-constrained environments. In this paper, we introduce QuantV2X, the first fully quantized multi-agent system designed specifically for efficient and scalable deployment of multi-modal, multi-agent V2X cooperative perception. QuantV2X introduces a unified end-to-end quantization strategy across both neural network models and transmitted message representations that simultaneously reduces computational load and transmission bandwidth. Remarkably, despite operating under low-bit constraints, QuantV2X achieves accuracy comparable to full-precision systems. More importantly, when evaluated under deployment-oriented metrics, QuantV2X reduces system-level latency by 3.2$\times$ and achieves a +9.5 improvement in mAP30 over full-precision baselines. Furthermore, QuantV2X scales more effectively, enabling larger and more capable models to fit within strict memory budgets. These results highlight the viability of a fully quantized multi-agent intermediate fusion system for real-world deployment. The system will be publicly released to promote research in this field. Please refer to the supplementary materials for the demo webpage and codebase.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents QuantV2X, a fully quantized end-to-end cooperative perception framework for Vehicle-to-Everything (V2X) systems. It introduces unified quantization across both neural models and transmitted feature representations, aiming to reduce computation, communication bandwidth, and latency without compromising perception accuracy.

### Strengths
- The focus on system-level deployability rather than only model accuracy is refreshing and directly relevant to real-world autonomous driving. QuantV2X targets both computation and communication bottlenecks, which are typically overlooked in prior V2X works.
- The integration of quantization into both the perception network and inter-agent communication pipeline is technically comprehensive and well-justified. The unified INT4/INT8 pipeline and compatibility with TensorRT suggest good deployment readiness.
- Introducing heterogeneity and spatial alignment losses to counter quantization noise is a reasonable solution to a domain-specific problem of feature misalignment in multi-agent fusion.

### Weaknesses
- While the integration is well-engineered, the quantization techniques themselves (post-training quantization, AdaRound, codebook compression) are based on existing literature. The main novelty lies in application-level integration. The paper would benefit from a clearer description of what is technically novel beyond combining known quantization and compression techniques.
- The claim that latency reduction can outweigh quantization-induced accuracy loss is convincing, but more quantitative trade-off analysis would strengthen the argument.

### Questions
Although component analysis is included, which modules are most sensitive to quantization?

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
This paper presents QuantV2X, the first fully quantized multi-agent system for efficient and scalable V2X cooperative perception. It applies unified end-to-end quantization: PTQ converts pre-trained FP32 models to low-bit networks with spatial alignment correction, while quantized codebook indices replace FP32 BEV features to reduce bandwidth. Experiments on DAIR-V2X, V2X-Real, and OPV2V show QuantV2X retains up to 99.8% accuracy (INT4/INT8), cuts latency by 3.2×, and improves mAP30 by 9.5% under strict GPU memory limits.

### Strengths
1. Innovative gap-filling design: It is the first fully quantized system for V2X cooperative perception, shifting from accuracy-centric to system-level efficiency optimization (latency, memory, bandwidth), addressing long-neglected deployment bottlenecks in prior work.

2. Effective technical framework: The PTQ process uses only 0.5% training data for calibration (no full retraining), and the alignment module boosts performance recovery (e.g., from 95.2% to 99.8% for $L_{P}+L_{S}$ in AP30). Codebook-based communication reduces transmission size from 8.6 MB (FP32) to 0.03 MB while preserving feature fidelity.

3. Strong engineering value: Compatible with TensorRT, it achieves 2.61× throughput and 2.87× efficiency gains (INT8 vs. FP32) on RTX 3090.

### Weaknesses
1. Limited technical scope: It only validates INT4/INT8 and INT8/INT8 quantization, with no experimental data to support the claim that 2-bit quantization is impractical for V2X. It also focuses on LiDAR-camera fusion but ignores other modalities (radar), limiting applicability to complex multi-modal scenarios.

2. Narrow experimental design: System latency tests rely on fixed ROS/TensorRT environments, excluding real-world variables (signal occlusion, diverse on-board chips). No long-term stability or extreme environment (bad weather) tests are conducted, and comparisons lack state-of-the-art LLM-derived quantization methods (e.g., improved GPTQ).

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents QuantV2X, a fully quantized multi-agent cooperative perception framework aimed at enabling efficient, scalable, and deployable V2X systems. The approach unifies quantization across both neural network models and transmitted BEV feature representations to reduce computational and communication costs.

### Strengths
This paper focuses on addressing the critical challenge of deploying collaborative perception systems in realistic environments.
By quantizing both the full-precision model and the shared BEV features, it effectively improves computational and communication efficiency while maintaining—or in some cases slightly improving—accuracy.
Moreover, the system-level experiments, which incorporate realistic processing pipelines, convincingly demonstrate the proposed method’s practicality and its ability to tackle the original motivation.

### Weaknesses
- Major Weaknesses
1. Limited methodological novelty despite a strong motivation

The proposed codebook learning stage and PTQ stage constitute the core phases where quantization is performed. 
In the codebook learning stage, the authors adopt an existing codebook-based quantization strategy and apply it to BEV feature sharing, rather than introducing a novel technique. 
In the PTQ stage, a post-training quantization process is applied for calibration in the collaborative model; however, this approach is a standard practice in the quantization field and does not present any distinctive methodological innovation.
The use of reconstruction loss between quantized and full-precision outputs to align the quantized model follows standard practice in prior quantization studies and lacks novelty.

More critically, the PTQ stage appears to be a direct application of conventional model quantization techniques to the collaborative perception domain, rather than a quantization method specifically designed for multi-agent scenarios. 
The stage does not effectively exploit information unique to collaborative perception—such as inter-agent feature dependencies or fusion-related cues that arise during intermediate fusion. 
Consequently, aside from the use of codebooks for BEV feature sharing, the proposed PTQ stage could be applied identically to a standalone vehicle model, suggesting that the approach remains a general model-quantization framework rather than a collaboration-aware one.

Furthermore, the quantization procedure itself lacks originality, as it adopts the AdaRound method without modification, and the experiments indicate that AdaRound primarily drives the observed performance gains. 

Therefore, although the paper emphasizes being the first to quantize not only BEV features but also the entire collaborative perception model, this contribution appears to be more of an application of existing quantization techniques to the collaborative perception domain rather than a genuinely novel quantization approach that exploits characteristics unique to collaborative perception itself.

2. Concerns about experimental validity

The paper discusses heterogeneity alignment and claims to employ such a setting in its experiments. 
However, prior works addressing heterogeneity typically validate alignment by performing inference with auxiliary agents whose sensor–encoder pairs were not used during training. 
In contrast, this paper does not specify which sensor–encoder pairs were used in the training and inference stages. 
Furthermore, if the same pairs were employed in both, the resulting model would likely be overfitted to those specific combinations. 
Hence, without evaluating the model on unseen sensor–encoder pairs, it is difficult to assert that genuine heterogeneity alignment has been achieved.

Additionally, in the system-level experiments (Table 3), two major issues arise.

(1) Unclear decomposition of latency components

In Table 3, the transmission feature size provides only an estimate of how much communication-level inefficiency contributes to $T_{comm}$, whereas the latency associated with model-level inefficiency—$T_{local}$ + $T_{fus}$—cannot be inferred from the reported data. 
Consequently, it is difficult to disentangle how performance degrades as a function of $T_{comm}$ and $T_{local}$ + $T_{fus}$. 
Reporting inference time and mAP@30/50 for three configurations—(a) after the codebook-learning stage ($T_{comm}$), (b) after the PTQ stage ($T_{local}$ + $T_{comm}$ + $T_{fus}$), and (c) after the PTQ stage but with FP32 BEV feature sharing ($T_{local}$ + $T_{fus}$)—would provide a clearer understanding of performance degradation attributable to each inefficiency component.

(2) Questionable generalization of quantization effects

The degree of accuracy degradation under system-level latency likely depends on the fusion model used. 
However, the experiments only compare quantized vs. non-quantized settings for Pyramid Fusion.
To support claims of generality, results should be reported for all fusion models across the same three experimental settings (a–c above), including both inference time and mAP@30/50.

- Minor Weaknesses
1. Insufficient explanation of how the combination weight $\alpha_n$ is generated (page 4).
2. The specific values of $n_L$ and $n_R$ used in QuantV2X are not stated.
3. Poor visual distinction between solid and dashed lines in Figure 4’s legend.
4. Missing explanation for the “Bits (W/A)” notation in Figure 4.
5. The distribution of pose errors is not described.
6. Lack of explanation for the model used as the upper bound.
7. Missing details on BEV feature dimensions (e.g., ego detection range, voxel size).
8. The description of V2X-ReaLO is absent from the main text.
9. The paper reports mAP as the evaluation metric on the V2X-ReaL dataset, but it does not specify which object classes were included in the detection task.

### Questions
- Major Questions
1. What methodological novelty justifies this work meeting the acceptance threshold?
2. How are the training and inference settings for heterogeneity alignment designed, and how can the current results support the claim that alignment was achieved?

- Minor Questions
1. Why were KL Divergence and L2 loss chosen for heterogeneity and spatial alignment losses, respectively?
2. For the spatial alignment loss, the paper mentions bounding box representations, but the code uses only the regression head output of the multi-class detection head—why?
3. Why is a uniform distribution used to model communication latency?
4. What are the training times for Stage 2 and Stage 3?

### Soundness
3

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
This paper introduces QuantV2X, the first fully quantized multi-agent system for cooperative perception in autonomous driving. By jointly quantizing both the neural network models and the inter-agent communication messages, QuantV2X significantly reduces computational cost, memory usage, and communication bandwidth while maintaining accuracy comparable to full-precision systems. The framework integrates post-training quantization, codebook-based message compression, and an alignment module to address feature degradation and cross-agent inconsistencies. Extensive experiments on real-world V2X datasets demonstrate that QuantV2X achieves up to 3.2× lower system latency and +9.5 mAP improvement, highlighting its strong potential for scalable and efficient real-world deployment of collaborative autonomous driving systems.

### Strengths
The work is original in its integration of model-side and communication-side quantization into a unified, deployment-oriented framework that directly tackles real-world challenges, the limitation of communication bandwidth, which is essential for enabling real-world deployment. 

The paper is clearly written with a smooth and logical flow, making it easy to follow.

### Weaknesses
1. The paper claims that QuantV2X outperforms linear quantization; however, linear quantization is a very basic baseline that is rarely used in modern deployments. To make the comparison more convincing, the authors are encouraged to include results against stronger quantization baselines, such as non-linear or piece-wise quantization methods as reviewed in [1].

2. While the experiments demonstrate that QuantV2X preserves performance under ideal conditions, it remains unclear how robust the system is to real-world imperfections such as GPS inaccuracy, pose noise, or communication latency. Evaluations under these non-ideal scenarios would strengthen the paper’s claim of deployability and highlight the practical resilience of the proposed method.

3. The writing quality could be improved for readability. Several figure and table elements (e.g., Figure 2, Table 2, and Table 8) contain fonts that are too small to read clearly, and there is no vertical spacing before Table 1, making the layout appear compressed. Improving the figure scaling and spacing will enhance the paper’s overall clarity and professional appearance.

[1] Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., & Keutzer, K. (2022). A survey of quantization methods for efficient neural network inference. In Low-power computer vision (pp. 291-326). Chapman and Hall/CRC.

### Questions
Can this tokenization method be extended to broader tasks such as segmentation, tracking, or collaborative end-to-end driving?

### Soundness
3

### Presentation
3

### Contribution
3
