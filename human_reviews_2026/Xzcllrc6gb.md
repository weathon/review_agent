# Quantized Visual Geometry Grounded Transformer

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6, 8

## Abstract
Learning-based 3D reconstruction models, represented by Visual Geometry Grounded Transformers (VGGTs), have achieved remarkable progress with large-scale transformers. Their prohibitive computational and memory costs severely hinder real-world deployment. Post-Training Quantization (PTQ) has emerged as a common practice to compress and accelerate models. However, we empirically observe that PTQ faces unique obstacles when compressing billion-scale VGGTs: the data-independent special tokens induce heavy-tailed activation distributions, while the multi-view nature of 3D data makes calibration sample selection highly unstable. 
This paper proposes the first **Quant**ization framework for **VGGT**s, namely **QuantVGGT**. This mainly relies on two technical contributions: First, we introduce *Dual-Smoothed Fine-Grained Quantization*, which integrates pre-global Hadamard rotation and post-local channel smoothing to robustly mitigate heavy-tailed distributions and inter-channel variance. Second, we design *Noise-Filtered Diverse Sampling*, which filters outliers via deep-layer statistics and constructs frame-aware diverse calibration clusters to ensure stable quantization ranges.
Comprehensive experiments demonstrate that QuantVGGT achieves the state-of-the-art results across different benchmarks and bit-width, surpassing the previous state-of-the-art generic quantization method with a great margin.
We highlight that our 4-bit QuantVGGT can deliver a **3.7$\times$** memory reduction and **2.5$\times$** acceleration in real-hardware inference, while preserving over **98\%** reconstruction accuracy of the full-precision counterparts. This demonstrates the vast advantages and practicality of QuantVGGT in resource-constrained scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposed QuantVGGT, the first comprehensive post-training quantization framework designed for visual geometry grounded transformer. QuantVGGT consists of two components: Dual-Smoothed Fine-Grained Quantization architecture for smoothing the distribution for better quantization performance from both global and local perspective and Noise-Filtered Diverse Sampling strategy for constructing information-maximized calibration dataset. Extensive experiments on multiple tasks compared with different strong baseline methods shown that QuantVGGT greatly outperforms the compared methods and achieves SOTA performance across different bit-width and tasks.

### Strengths
1. The motivation of QuantVGGT is clear and quite meaningful, as a systematic quantization research on large-scale 3D foundational models in is heavily unexplored and crucial.

2. The proposed method is supported by extensive empirical analysis and theoretical theorems, and appears to effectively solve the corresponding problems.

3. The paper writing is very clear and accompanied by key illustration figures and method flowcharts, which are easy to understand and implement.

4. The paper conducted high bit (8-bit) and low bit (4-bit) experiments on multiple task datasets, significantly surpassing strong baseline methods from different fields, demonstrating the effectiveness and generalization of QuantVGGT.

5. QuantVGGT reported the true efficiency in hardware, calibration consumption, various detailed ablation experiments, and extensive experimental analysis. This fully demonstrates the effectiveness of its method and the effectiveness of acceleration in real-world scenarios.

### Weaknesses
1. The symbol writing on line 192 seems to be incorrect. And for the compared method in line 377, I believe it should be spelled QuaRot.

2. Providing acceleration effects at different sequence lengths will further enhance its practicality in different scenarios.

3. Adding different comparison methods reconstruction visualization will better demonstrate the visual effectiveness of the proposed methods.

### Questions
1. The symbol writing on line 192 seems to be incorrect. And for the compared method in line 377, I believe it should be spelled QuaRot.

2. Providing acceleration effects at different sequence lengths will further enhance its practicality in different scenarios.

3. Adding different comparison methods reconstruction visualization will better demonstrate the visual effectiveness of the proposed methods.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This submission presents QuantVGGT, the first Post-Training Quantization (PTQ) framework for Visual Geometry Grounded Transformer (VGGT), a state-of-the-art model for learning-based 3D reconstruction. The work addresses two critical challenges in VGGT quantization: (1) heavy-tailed activation distributions induced by data-independent camera and register tokens, and (2) unrepresentative calibration dataset building due to the multi-view complexity of 3D data.
For challenge 1, the authors propose Dual-Smoothed Fine-Grained Quantization (DSFQ), which consists of a pre-global Hadamard rotation for dispersing outliers and smoothing heavy tails, a post-local channel scaling for normalizing inter-channel variance. Furthermore, the authors adopt token-wise quantization for activation and outer-dimension-wise quantization for weight, forming fine-grained quantization, which leads to minor quantization error.
For challenge 2, the authors propose Noise-Filtered Diverse Sampling (NFDS), which filters noisy outliers using deep-layer activation statistics and constructs frame-aware calibration clusters concerning VGGT’s inductive bias of relationships between first-frame and subsequent-frames.
Comprehensive experiments on Co3Dv2 for camera pose estimation and DTU for point map estimation demonstrate that, QuantVGGT outperforms existing quantization methods across various bit-widths both in theoretical validation and real-hardware deployment.

### Strengths
1. Originality: Target model quantization on VGGT, a new SOTA 3D reconstruction model, analyzing concrete experimental phenomena, heavy-tailed activation distribution and unrepresentative calibration dataset, and proposing corresponding observation-driven solutions, DSFQ and NFDS.

2. Validation: Detailed mathematical theoretical derivations and justifications for technical innovations. Comprehensive experiments with additional ablation studies across tasks, quantization bit-widths, comparison methods, and proposed components.

3. Clarity: Hierarchical narrative logic, clear writing, well-organized sections, and informative visualizations make complex concepts (e.g., Hadamard rotation and frame-aware sampling) accessible to readers from both 3D vision and quantization backgrounds.

4. Significance: Bridge the gap between large-scale 3D reconstruction model performance and deployment efficiency, with results that are both scientifically influential and practically useful, which advances model quantization in 3D field and enables edge 3D reconstruction.

### Weaknesses
1. Insufficient Related Works: Not involve other works on VGGT acceleration and optimization, such as FastVGGT using token compression and FasterVGGT using sparse attention, let alone comparison with these methods in Sec 4 to highlight the efficiency of QuantVGGT.

2. Rough Analysis of Special Tokens: The paper mentions that data-independent special tokens cause heavy tails, but does not explicitly and concretely state the respective influences of camera tokens and register tokens.

3. Experiment Generalization:
(1) Lack of experiments on the widely adopted W4A8 quantization configuration to enable more comprehensive comparisons with existing methods.
(2) Lack of experiments on more tasks (e.g., Multi-view Depth Estimation on DTU, Image Matching on ScanNet-1500) and datasets (e.g., Camera Pose Estimation on RealEstate10K, Point Map Estimation on ETH3D) compared with the original VGGT paper.

4. Experiment Credibility:
(1) In Sec 3, the authors claim that VGGT has an inductive bias for modeling relationships between the first frame and subsequent frames. From the theoretical perspective, this phenomenon perhaps originates from the unique design of VGGT’s two distinct sets of special tokens. However, the paper barely supply sufficient experimental evidence to reflect it.
(2) Activation Distribution only in two adjacent blocks (frame/global block 7/8) in Appendix D fail to confirm the claimed ubiquity of the heavy-tailed phenomenon across different layers.
(3) Lack of more subjective visualization results in Appendix H in across more scenarios, including comparisons between the FP16 baseline, QuantVGGT, and other quantization methods under various quantization settings.

### Questions
1. For frame-aware clustering in NFDS, the correlation vector c^i measures similarity between the first frame and subsequent frames. How does this strategy perform on real-world 3D sequences where the first frame is an outlier (e.g., occluded or low-light)?

2. Why does the QuantVGGT in W8A8 outperform the FP16 baseline in Camera Pose Estimation on Co3Dv2? Please analyze the phenomenon and clarify the potential causes. Moreover, provide a reasonable explanation for the marginal performance difference between QuantVGGT and QuaRot under W8A8 setting.

3. Is there a possibility of deploying QAT or extremely low-bit PTQ in VGGT? How should the quantization strategies be adapted or specially designed for these scenarios respectively?

4. Have you ever studied whether other 3D reconstruction models (e.g., DUSt3R, MASt3R) also have similar phenomenon of heavy-tailed activation distribution and unstable calibration? Would DSFQ and NFDS perform well on them?

5. Presentation and Visualization Suggestions: 
(1) Supplement brief explanations of variables R_j^*, V_j^*, and V^* directly in Theorem 3.2, and add a hyperlink to Appendix A immediately for quick reference to related details.
(2) Consider enlarging the size of Fig. 4 (a) and supplementing the specific attributes (labels or features) corresponding to each cluster in Fig. 4 (b)(c)(d) in Appendix E to improve readability.
(3) Include a concise introduction to "global robust moments" before Eq. 9.
(4) Give an explicit definition of "static quantization" and "dynamic quantization" and clarify whether "tensor-wise" and "token-wise" refer to activation quantization specifically.
(5) Merge Table 6 in Appendix D into Table 3 of Sec 4.3 to avoid redundant presentation.
(6) Provide detailed hardware platform information together with absolute latency and optimize the presentation format of Fig. 6.
(7) Consider consolidating foundational knowledge of rotation-based quantization (e.g., Hadamard transform) and migration-factor-based quantization (e.g., SmoothQuant) into Sec 3.1 (Preliminary) for better logical coherence.
(8) Add optional ablation studies for hyperparameters, including the filtering threshold T (or p), the cluster number K and the calibration samples scale N.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces QuantVGGT, the first post-training quantization framework specifically designed for billion-parameter Visual Geometry Grounded Transformers (VGGTs), tackling two unique challenges in 3D reconstruction: (1) Dual-Smoothed Fine-Grained Quantization (DSFQ) employs a pre-global Hadamard rotation to disperse heavy-tailed activations from data-independent tokens, followed by post-local channel smoothing to stabilize variance without runtime overhead; and (2) Noise-Filtered Diverse Sampling (NFDS) filters outliers using deep-layer statistics and constructs calibration clusters based on VGGT’s intrinsic frame-relative geometric bias, ensuring representative sampling. The approach achieves state-of-the-art results under 4-bit quantization, preserving over 98% of full-precision accuracy while delivering 2.5× speedup and 3.7× memory compression—enabling practical deployment of large 3D vision models in resource-constrained environments.

### Strengths
S1: Novel and Insightful Problem Formulation. The paper's core originality lies not in inventing a new algorithm, but in being the first to identify and formalize the unique quantization challenges specific to 3D vision transformers—namely, the instability caused by data-independent special tokens and multi-view calibration. This re-framing of the problem is a significant contribution in itself.

S2: High-Quality and Tailored Technical Solutions. The proposed methods (DSFQ and NFDS) are technically robust and specifically tailored to the identified problems. In particular, NFDS cleverly leverages the model's own geometric inductive bias for sampling, demonstrating a deep understanding of the architecture that goes beyond generic quantization techniques.

S3: High Practical Significance and Impact. The work's significance is substantial, as it bridges the gap between powerful, large-scale 3D models and real-world deployment. Achieving 98% accuracy at 4-bit with significant memory and latency reduction makes these models viable for edge devices, unlocking new possibilities in robotics, AR/VR, and other resource-constrained applications.

S4: Excellent Clarity and Presentation. The paper is exceptionally clear and well-written. The use of informative figures to visualize abstract concepts like activation distributions, combined with intuitive and rigorous explanations, makes the complex contributions accessible and highly convincing.

### Weaknesses
W1: Lack of Real-World Hardware Benchmarks. The reported latency and memory gains are theoretical. The paper lacks empirical validation on actual hardware using standard inference engines (e.g., TensorRT, ONNX Runtime), making it difficult to confirm if the claimed speedups translate to real-world deployment.

W2: Unclear Sensitivity to Calibration Data Size. The robustness of the method to the number of calibration samples is not explored. An ablation study on performance with significantly fewer samples (e.g., 10 or 20) is missing, which is critical for understanding its practicality in data-scarce scenarios.

W3: Limited Generalization to Other Architectures. The evaluation is limited exclusively to the VGGT architecture. It remains unclear if the proposed solutions are generalizable to other 3D vision transformers (e.g., DUSt3R, MASt3R), which limits the broader impact of the work.

W4: Missing Comparison to Quantization-Aware Training (QAT). The paper does not include a comparison to QAT. A QAT baseline, even with minimal fine-tuning, would provide crucial context on the absolute performance ceiling and help better evaluate the effectiveness of this PTQ-only approach.

### Questions
Q1: How does performance degrade as the calibration set size is significantly reduced (e.g., to 20, 10, or 5 samples)?

Q2: In the extreme low-data regime (e.g., <10 samples), does your NFDS method maintain superior stability (lower variance) compared to random sampling?

Q3: How specific are the identified quantization challenges to VGGT? Could the core principles of DSFQ and NFDS be applied to other 3D vision transformers like DUSt3R or MASt3R?

Q4: Could you clarify what "random Hadamard matrix" means? How was it generated, and is the model's performance sensitive to the specific matrix chosen?

Q5: Is the remaining performance gap at W4A4 a fundamental limitation of PTQ for this model? How would it compare against a minimally fine-tuned QAT baseline (e.g., after 1 epoch)?

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
3

### Summary
This paper presents QuantVGGT, the first post-training quantization (PTQ) framework specifically designed for Visual Geometry Grounded Transformers (VGGTs) they propose:
Dual-Smoothed Fine-Grained Quantization (DSFQ) — applying a Hadamard pre-rotation and channel-wise post-smoothing to reduce outlier sensitivity and inter-channel variance.
Noise-Filtered Diverse Sampling (NFDS) — filtering outliers using deep-layer activation statistics and building frame-aware diverse calibration clusters for robust calibration.

### Strengths
This is the first quantization framework tailored for billion-scale 3D transformers, a meaningful step for deploying 3D vision foundation models efficiently.

Ablation studies clearly isolate contributions from DSFQ and NFDS.

Quantizing VGGT can significantly lower compute/memory costs for 3D perception and reconstruction in real-world systems

### Weaknesses
Only two datasets (Co3Dv2, DTU) are used. Broader testing on outdoor scenes could strengthen claims of generality.

The DSFQ pipeline introduces additional preprocessing (Hadamard transform, smoothing, and clustering). The cost and integration details for deployment (e.g., on mobile devices) are not deeply analyzed.

The paper only focuses on PTQ, so it’s unclear how much performance could be recovered with small-scale fine-tuning.

Theorem 3.2 is theoretically elegant but its real contribution to calibration robustness could be better demonstrated through comparative sampling visualizations or empirical sensitivity analysis.

### Questions
Do the authors plan to further validate the robustness of their method on more diverse 3D datasets (e.g., ScanNet, DL3DV, or Tanks and Temples)?


How does QuantVGGT perform under mixed precision or asymmetric quantization setups?

How sensitive is QuantVGGT to the choice of Hadamard matrix and smoothing coefficient α?
Would small perturbations in these hyperparameters impact stability?

Does the method generalize to other architectures such as DUSt3R or diffusion-based 3D transformers?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper tackles PTQ of billion‑parameter VGGT, a recent 3D reconstruction backbone. The authors diagnose two obstacles for VGGT PTQ: (i) heavy‑tailed activations induced by data‑independent special tokens (camera & register tokens), and (ii) unstable calibration due to the multi‑view nature of 3D inputs. They propose Dual‑Smoothed Fine‑Grained Quantization (DSFQ) and Noise‑Filtered Diverse Sampling (NFDS) to address the issues.

### Strengths
1. The paper provides convincing evidence that VGGT’s data‑independent registration tokens produce heavy‑tailed, high‑variance channels that break naïve PTQ; the visualizations in Fig. 3 and Fig. 7 make this concrete. 
2. DSFQ is well‑motivated: Hadamard rotation preserves matmul but spreads outliers, and the subsequent channel scale is computed after rotation, avoiding pre‑scale instability. The choice of token‑wise activation is sensible for transformer matmuls and validated in Table 5.
3. Strong empirical results at low bit‑widths. W4A4 QuantVGGT clearly surpassing generic baselines.

### Weaknesses
1. The baseline set is strong (GPTQ, SmoothQuant, QuaRot, DopQ‑ViT), but some competitive activation‑aware baselines (e.g., AWQ‑style or ViT‑specific PTQ variants) are not included; also unclear if all baselines received equal calibration size/selection tailored to 3D sequences.
2. External validity beyond VGGT. Claims focus on VGGT‑1B and VGGT is a great model. However, it is unclear whether the observations and designs can translate to other 3D backbones.

### Questions
Have you tried QuantVGGT on models without VGGT’s special tokens, eg. Fast3R? Does NFDS remain effective?

### Soundness
3

### Presentation
3

### Contribution
2
