# MedQuanBench: Quantization-Aware Analysis for Efficient Medical Imaging Models

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Quantization is a crucial technology for facilitating the deployment of medical AI models, especially on 3D radiological data. However, existing studies often lack comprehensive evaluations across diverse architectures, modalities, and quantization techniques, which limits our understanding of the real-world trade-offs among applicability, efficiency, and performance. In this work, we introduce MedQuanBench, a large-scale and diverse benchmark designed to rigorously evaluate quantization techniques for 3D medical imaging models. Our benchmark spans a wide range of modern architectures (e.g., CNNs and Transformers). We systematically evaluate representative post-training quantization strategies across model scales and dataset sizes. Additionally, we perform detailed sensitivity analyses to identify which model components are most vulnerable to quantization, including layer-wise degradation and activation distribution shifts. Our results show that 8-bit quantization consistently preserves segmentation accuracy across diverse architectures, making it a reliable choice for deployment. Furthermore, with appropriate configuration, such as selecting proper quantization granularity based on the model structure, 4-bit precision can also achieve near-lossless performance. These results show MedQuanBench as a fundamental benchmark for optimizing quantization strategies and guiding the development of deployment-ready, low-bit medical imaging models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MedQuanBench, a large-scale benchmarking platform designed to systematically evaluate the effectiveness of quantization techniques on 3D medical imaging models. The authors systematically evaluate representative post-training quantization (PTQ) strategies across a variety of modern model architectures, including CNNs and Transformers, across various model scales and dataset sizes. Furthermore, the paper conducts a detailed sensitivity analysis to identify the model components most vulnerable to quantization, exploring issues such as layer-level performance degradation and shifts in activation distributions.

### Strengths
The benchmark is designed to be comprehensive. 

It covers:
* Diverse architectures: From classic CNNs (nnU-Net) to hybrid architectures (MedFormer, SwinUNETR, UNETR) and pure CNN variants (STU-Net).
* Diverse modalities and tasks: Including CT and MRI, as well as various organ, tumor, and brain segmentation tasks.
* Diverse scales: Models ranging from 10M to 2B parameters, and datasets ranging from hundreds to tens of thousands of samples, are tested.

**Hardware-Aware Perspective:** This paper offers a significant strength. The analysis in Section *Quantized Operation on Real Hardware* is excellent. The authors not only provide quantization formulas but also explain in detail how different quantization granularities map to hardware primitives on modern GPUs (using NVIDIA Blackwell as an example).

### Weaknesses
**1. Quantization Methodology Incompleteness**

Symmetric vs. Asymmetric Quantization: The paper exclusively uses symmetric quantization (Eq. 1), ignoring asymmetric schemes critical for non-negative activations (e.g., ReLU outputs). Asymmetric quantization ($X_q = round(X/S) + Z$) often outperforms symmetric methods in low-bit regimes (e.g., INT4) by better fitting skewed distributions. This omission weakens the benchmark’s applicability to real-world medical models.

Calibration Methods: Reliance on naive min-max scaling (Eq. 1) is suboptimal. Advanced calibration techniques, such as KL divergence and percentile-based scaling, mitigate outlier sensitivity and improve INT4 robustness. The authors must justify why these were excluded or add experiments comparing calibration strategies.

---

**2. Underdeveloped "Advanced PTQ" Evaluation**

Hyperparameter Sensitivity: Advanced methods (smoothing, SVD, rotation) in Table 4 show marginal gains, but their hyperparameters (e.g., smoothing factor α, SVD rank) lack optimization studies. For instance, α=0.5 (Appendix D) may be arbitrary; a sensitivity analysis of α ∈ [0.1, 0.9] is needed to validate "limited effectiveness."
Scope of Application: Applying these methods only to the "most sensitive layer" (Sec 4.3) overlooks their intended global/block-wise use. Testing them holistically (e.g., activation smoothing across all layers) would provide stronger evidence for their (in)effectiveness.

---

**3. Hardware Evaluation Gaps**

INT4 Acceleration Omission: While INT8 hardware results (Table 5) are thorough, INT4 lacks real-device profiling. Claims about Blackwell’s 4-bit support remain theoretical. Without latency/memory metrics for INT4, the "near-lossless" performance claim (Abstract) is unsupported for clinical deployment.

---

**4. Presentation & Technical Issues**

a) Inefficient Data Visualization.
One problem is table design. Vertical tables (Tables 1–3, 5) waste space and hinder cross-architecture comparison. For example, Table 1’s left half is empty; DSC/NSD drops require vertical scanning. It is extremely hard to compare between experiments.

b) Underutilized Content.
Some important content is buried in appendices. SegFormer3D’s layer-wise sensitivity (App F, Table 11) can be moved to the main text (Sec 4.3) to strengthen architectural insights. Other important experiments in appendix can also be moved if the author save the main text space by reorganizing the table. Another acceptable way is to replace bulky tables with small multiples of line/bar charts to compare quantization granularity across models.

### Questions
Thanks to your appendix experiment, otherwise the article would be far from enough just by relying on the experiments shown in the main text. Try to reorganize your presentation especially those tables.

Others see weaknesses.

Considering relevance, please cite this work if you did not write it. 

*Post-Training Quantization for 3D Medical Image Segmentation: A Practical Study on Real Inference Engines*
https://arxiv.org/pdf/2501.17343v1

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
The paper proposes MedQuanBench, a large-scale benchmark for post-training quantization (PTQ) of 3D medical imaging models across CNN and hybrid (Transformer-based) architectures, bit-widths (INT8/INT4), and quantization granularities (per-tensor, per-channel/token, and an adaptive per-voxel strategy for 1×1×1 convs). It evaluates four datasets (BTCV, TotalSegmentator V2, AbdomenAtlas 1.1, WholeBrain) and reports segmentation metrics (DSC/NSD) under varied scales. Key findings: INT8 is near-lossless across backbones, while INT4 degrades sharply for hybrid architectures under coarse granularity. A layer-wise sensitivity study identifies 1×3×3 convolutions as the main INT4 bottleneck; replacing a sensitive 1×3×3 convolution with 1×1×1 recovers much of the drop. Hardware profiling on NVIDIA Ada/TensorRT shows ~3.2–3.8× model size reduction and ~2.1–2.7× speedups for INT8 quantization with negligible accuracy loss.

### Strengths
- Clear empirical conclusion: INT8 preserves FP32 accuracy broadly; INT4 requires careful granularity, especially for hybrid architectures
- Hardware-aware analysis with real INT8 deployment on TensorRT, reporting latency and memory gains consistent across models
- Scale studies across model and dataset sizes, highlighting increasing INT4 sensitivity with larger, fine-grained tasks
- Activation analyses showing spatially localized outliers in medical models vs. channel-localized outliers in LLMs, motivating granularity choices

### Weaknesses
- Lack of motivation: The paper assumes that this is an issue in the medical domain that requires careful investigation, particularly in the segmentation setting but without setting the stage for why it is indeed an important problem to solve.
- Unclear contributions: The paper describes the benchmark dataset as a contribution, but it is not clear to me what different insights subsets of the total dataset provide and why one needs to evaluate quantization on the whole benchmark vs. a subset.
- Method coverage: Focuses on PTQ; limited exploration of QAT or mixed-precision baselines that may further close INT4 gaps in critical layers. 
- Limited clinical validation: Strong segmentation metrics, but few task-level clinical end-points (e.g., time-to-diagnosis, error costs) to contextualize acceptable accuracy loss. 
- Robustness of the approach: Sensitivity analyses are primarily layer-wise; fewer robustness tests for distribution shift beyond datasets listed (OOD clinical sites, scanners, protocols). 
Interpretability of failures: While sensitive layers are identified, the failure modes under INT4 (e.g., boundary errors for small structures) could use more granular error breakdowns/visuals.
- Generality of the approach: I find this is too specific to a particular context of medical imaging (i.e. segmentation) to be useful for the broader ICLR community.

### Questions
- Why does the study focus exclusively on post-training quantization (PTQ)? Could including quantization-aware training (QAT) or mixed-precision methods provide a more complete landscape of quantization robustness in medical imaging models?

- How were the datasets selected? Is the intention to represent medical imaging diversity (e.g., modality, anatomy, resolution) and in what way?

- The analysis identifies 1×3×3 convolutions as particularly quantization-sensitive. Can you provide intuition on why these layers—versus larger kernels or attention modules—dominate degradation?

- The paper demonstrates negligible accuracy loss for INT8 quantization, but what degree of degradation (e.g., 1–2% DSC drop) is clinically acceptable for deployment?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces MedQuanBench, a study of post‑training quantization (PTQ) for 3D medical image segmentation that combines four public datasets (BTCV, Total Segmentator V2, AbdomenAtlas 1.1 and Whole Brain), to evaluate multiple CNN and hybrid CNN-Transformer architectures at both 8‑bit and 4‑bit precision. The paper compares per‑tensor, per‑channel/token and per‑voxel (adaptive stratification) scaling and find that INT8 quantization preserves full‑precision performance while significantly reducing model size and latency. Per‑tensor INT4 quantization severely degrades the performance on transformer models, whereas CNNs fare better and can recover much of their performance using per‑voxel scaling (e.g., STU‑Net‑B improves from 0.647 to 0.829 Dice in per-channel to adaptive stratification). Larger model and dataset scales increase INT4 sensitivity, and layer‑wise analysis reveals that the 1 × 3 × 3 convolution is particularly critical and replacing it with a 1 × 1 × 1 convolution and using per‑voxel quantization improves INT4 robustness. Further, the paper notes that the activation smoothing, SVD and rotation offer limited or no gains, and supplements the experiments with real‑hardware profiling.

### Strengths
The following are the strengths of the paper.

1. By comparing per‑tensor, per‑channel/token and per‑voxel (adaptive stratification) scaling, the paper highlights the importance of granularity. It clearly shows that coarse per-tensor quantization is not suitable for 4‑bit precision, whereas finer granularity may recover accuracy. These insights/findings are helpful.

2. The authors perform incremental dequantization and identify the 1 × 3 × 3 convolution as the most sensitive layer. Replacing it with a 1 × 1 × 1 convolution closes the gap to full‑precision performance. Such targeted analysis helps guide architecture design for low‑bit inference.

3. Table 5 presents model size and latency measurements of real INT8 quantization using NVIDIA TensorRT, confirming the practical benefits of quantization.

### Weaknesses
The following are the weaknesses of the paper.

1. The paper claims that quantization is under‑studied in medical imaging, yet several prior works address this issue. MedQ introduced lossless ultra‑low‑bit quantization for U‑Net segmentation in 2021 **[1]**. U‑Net Fixed‑Point Quantization (2019) **[2]** also demonstrated 4‑bit weight quantization for medical segmentation and reported memory reduction with minimal accuracy loss. Recent EfficientQ (2024) provides a PTQ method tailored for medical segmentation and is publicly available **[3]**. None of these works are discussed, instead the paper mainly references general PTQ methods and LLM quantization. This weakens this study and leaves readers unaware of existing solutions.

2. The hardware profiling numbers in Table 5 match those reported in a separate study that introduced a TensorRT‑based PTQ framework for 3D medical segmentation **[4]**. The **[4]** pre‑quantized U‑Net, SwinUNETR, UNesT and others and published the exact model size and latency reductions (e.g., U‑Net from 23.11 MB/2.62 ms to 6.61 MB/1.05 ms). MedQuanBench reuses these numbers but presents them as part of its own study, without acknowledging the source. Reusing results without citation is problematic and may mislead readers into believing these measurements were performed in this study.

3. MedQuanBench is presented as a benchmark yet consists solely of separate evaluations on four datasets without any aggregate metric or consolidated score. Results are reported independently for BTCV and AbdomenAtlas 1.1, so it is unclear what benchmark performance means or how different methods would be ranked overall. The lack of unified evaluation criteria weakens the value of calling it a benchmark, which is the main message of the paper.

4. (Minor) The text around lines 351–358 asserts that larger datasets increase sensitivity but appears under Table 2 without referencing Table 3, making the narrative confusing and suggesting the paper’s structure needs refinement.

---

**[1]** MedQ: Lossless ultra-low-bit neural network quantization for medical image segmentation (Medical Image Analysis, 2021)

**[2]** U-Net Fixed-Point Quantization for Medical Image Segmentation (MICCAI 2019)

**[3]** EfficientQ: An efficient and accurate post-training neural network quantization method for medical image segmentation (Medical Image Analysis, 2024)

**[4]** Post-Training Quantization for 3D Medical Image Segmentation: A Practical Study on Real Inference Engines (arXiv: 2501.17343)

### Questions
MedQuanBench addresses a practical problem, how to deploy memory and compute intensive 3D segmentation models on limited hardware by using low-bit quantization. However, the contributions seem minimal and are not well presented. The paper provides per-dataset analyses yet calls the work MedQuanBench. It also fails to properly discuss related work and reuses reported results without citing the original sources. Clarification on these points would be appreciated.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies post-training quantization (PTQ) for 3D medical segmentation. It explores 8-bit and 4-bit settings and compares quantization granularities. Results show 8-bit is essentially lossless, while 4-bit is fragile unless granularity is fine and certain layers are handled carefully. A small architectural tweak (replacing a 1×3×3 conv with 1×1×1) improves 4-bit stability.

### Strengths
1. INT8 works effectively out of the box, delivering almost no accuracy loss while providing noticeable improvements in inference speed and model size.
2. The paper has some good insights on quantization, showing evidence that coarse per-tensor INT4 quantization fails on many models, especially hybrids that include transformer blocks.
3. The paper identifies the sensitivity of 1×3×3 convs, and the simple 1×1×1 replacement is a good takeaway that could help the community.
4. Evaluations are conducted on multiple models (CNN + hybrid) and datasets.

### Weaknesses
My main concern is that the paper positions itself as a benchmark but does not clearly define a unified evaluation metric, which makes it hard to compare performance across methods. Furthermore, the exploration of INT4 improvements is quite narrow, while granularity and the 1×1×1 replacement are studied, other methods, such as mixed precision or selective dequantization, could have been tested to provide a more detailed picture.

### Questions
It would help if the benchmark framing were made more consistent, for instance, having a single unified metric per model and dataset would make it easier for future work to compare against this benchmark.

### Soundness
2

### Presentation
2

### Contribution
2
