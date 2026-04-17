# Rethinking the Spatiotemporal Distribution for High-Fidelity Parallel ANN-to-SNN Conversion

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Spiking Neural Networks (SNNs) have attracted increasing attention for their low power consumption and constant-time inference on neuromorphic hardware. Among existing approaches, ANN-to-SNN conversion is one of the most effective ways to obtain deep SNNs with accuracy comparable to traditional ANNs, and recent work has even extended it to \emph{parallel} conversion, where the full spike train is emitted in a single pass. Despite this promise, we find that ANN-to-SNN parallel conversion suffers from severe performance degradation at ultra-low timesteps ($T \leq 4$), limiting its practical use.
In this work, we analyze the source of this performance gap and demonstrate that it originates from assumptions in the standard quantization–clip–floor–shift (QCFS) formulation, which, under the one-shot firing rule, introduces a step-dependent bias. To overcome this, we propose a \emph{distribution-aware parallel calibration} that corrects spatiotemporal mismatches while leaving the backbone and firing rule unchanged. Our method consists of two stages: (1) \textbf{spatial recalibration}, which adapts normalization layers to spike-domain statistics, and (2) \textbf{temporal correction}, which learns a per-channel, time-collapsed aggregated membrane potential bias to offset timestep-dependent errors.  
On ImageNet-1k, our approach boosts ResNet-18 top-1 accuracy from $\mathbf{25.20\%\!\to\!62.28\%}$ at $T=4$ and ResNet-34 from $\mathbf{50.67\%\!\to\!68.23\%}$ at $T=8$. These results demonstrate that revisiting—and correcting—standard QCFS premises in the \emph{parallel setting} is essential for accurate, low-latency SNNs without retraining the backbone.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a two-stage distribution-aware calibration method for ANN-SNN Parallel Conversion, which further enhances the performance of SNN parallel inference under the condition of ultra-low time latency.

### Strengths
1. This work points out the problem of suppressing early contributions while magnifying late ones in vanilla parallel conversion, and makes exploration based on this.

2. This work provides a theoretical explanation for the LW-FOCF proposed in the second stage.

### Weaknesses
1. This work has the most advantages and application value in ultra-low latency and training-free inference scenarios, it actually solves the potential defects of vanilla parallel conversion in correcting quantization errors, as vanilla parallel conversion can already achieve complete lossless conversion for QCFS with $T=\tilde{T}$. Therefore, I tend to think that the main contribution of this work is a local and limited optimization for parallel conversion.

2. Consistent with vanilla parallel conversion, the current applicable visual task and network backbone for this work are still image classification and CNNs. The parallel conversion based on other visual tasks and network structures is worth further consideration.

### Questions
See Weaknesses Section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of performance degradation faced by the "parallel conversion" method in the field of ANN-to-SNN conversion under ultra-low time steps. The authors of the paper argue that the reason is the mismatch in spatiotemporal distribution and propose a two-stage calibration method to solve this problem. The improved experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1. The writing provides a mathematical proof of the proposed theory, which is logically self-consistent, and the presentation of pictures and tables is appropriate.
2. In terms of results, the accuracy has been improved, demonstrating the effectiveness of the proof method to a certain extent.

### Weaknesses
1. Although Appendix B.1 provides ablation results on output distributions across different network depths when Stage 1 or Stage 2 is used independently, it lacks direct ablation experiments quantifying how Stage 1-only or Stage 2-only configurations affect final accuracy. This casts doubt on the effectiveness of Stage 2.  
2. A pseudocode description of the algorithm is absent.  
3. The paper extensively cites and compares against FS-PC as a baseline. However, the relationship between the proposed method and FS-PC could be more explicitly elaborated. It is recommended that the Methods section clearly specify: (a) the exact components of the FS-PC baseline, and (b) which specific modules the "Ours" method adds to this baseline.

### Questions
1. Is the proposed method applicable to converted non-parallel neurons, such as Leaky Integrate-and-Fire (LIF) or Integrate-and-Fire (IF) neurons? This raises uncertainties regarding the method’s scope of application.  
2. The authors repeatedly characterize the method as "lightweight," yet the actual costs appear significant relative to the benefits. As indicated in the appendix, Stage 2’s additional calibration step requires approximately 3,000 epochs—even though it introduces only ~10⁴ trainable parameters. Moreover, in the main text section "Stage 2: Ladder-Weighted First-Order Correction Field (LW-FOCF)," the authors note that the training dataset is employed as the calibration set. These substantial additional costs far surpass those associated with retraining an ANN or SNN. We therefore seek clarification: what specific advantages does this method provide in comparison to conventional ANNs? 
3. Please clarify the size of the calibration set: In Section 4.1 "MOTIVATION I," the authors state, "Empirical evidence shows that this assumption that pre-BatchNorm activations retain the same distribution breaks down, particularly at ultra-low timesteps T." We request clarification on: (a) What constitutes this "empirical evidence"? (b) The authors assert, "The procedure is lightweight—requiring only a small calibration set and no gradient updates." What is the precise size of this "small calibration set"?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The author proposed a calibration pipeline for parallel conversion, which includes updating the BN parameter to match the distribution and introducing a learnable single-channel membrane potential bias to minimize error.  However, However, my concern lies in the practical soundness and the novelty of this method.

### Strengths
Improvement over the existing baseline (Hao et al., 2025) and clear figures.

### Weaknesses
1. This work focuses on the so called *parallel conversion*. This kind of conversion is not biological plausible that composes all spike train into a single time step. With this design, the SNN will loses its asynchronous advantage. It looks more like a low-bit quantization rather than ANN2SNN conversion. Regarding that, I have additional several concerns:

- The proposed Spatial Recalibration has been a common technique to use in quantization work, such as [1].
- Figure 1 illustrates the distribution mismatch. The authors and prior work claimed a lossless approach under a uniformed distribution prior. But it has been a common sense that activation/parameters distribution never follows a uniform distribution. Therefore, the behavior shown in Fig.1 is in line with everyone's expectations.
- Furthermore, the distribution mismatch problem is not unique problem in the parallel conversion. Even in standard case, the ReLU2LIF conversion introduces the mismatch. This problem has been clearly illustrated in SNN Calibration proposed 4 years ago[2], which also proposed similar calibration techniques. 

2.  I have concerns about the scope of this work. It seems that this work primarily focused on the ResNet-34 and VGG-16 on the ImageNet dataset. However, even a much tiny architecture like MobileViT-S can achieve 78.4 accuracy with 5.6M parameters. I wonder how this work can be used to prove the SNN efficiency combined with latest ANN advancements. 

3.  The goal of this work is to obtain extremely low timesteps SNNs. Can the authors analyze which one is better, the Quantized ANN with 2-bit activations or the Converted SNN with 4 timesteps？My opinion is 2-bit quantized ANN is better given by its less activation memory and easier hardware logic. 

[1] PD-Quant: Post-Training Quantization based on Prediction Difference Metric. CVPR 2023

[2] A Free Lunch From ANN: Towards Efficient, Accurate Spiking Neural Networks Calibration. ICML 2021

### Questions
See weakness. Besides, for point 1, how does the author view the similarity with [2] and exising spatial recalibration techniques, and what is the essential difference between the contribution of this paper and directly applying those existing recalibration techniques to SNNs with parallel neurons?

### Soundness
1

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
This paper investigates the severe accuracy degradation of parallel ANN-to-SNN conversion under ultra-low timesteps T, attributing it to a spatiotemporal distributional mismatch: spatially, BatchNorm statistics are misaligned with spike-domain activations; temporally, the parallel firing rule concentrates spikes in late timesteps. The authors propose a two-stage calibration framework spatial recalibration and temporal bias correction, that significantly boosts ImageNet accuracy for ResNet-18/34 especially at T=4/8.

### Strengths
1.The analysis of spatiotemporal bias in parallel conversion is clear , particularly the breakdown of QCFS assumptions at ultra-low T.

2.The reported improvement on ImageNet is competitive, such as ResNet-18 from 25.2% to 62.28% at T=4.

### Weaknesses
1.Both stages essentially combine existing ideas ,which is lack of innovation.

2.Experiments are restricted to CNNs on image classification benchmarks. The method is not evaluated on modern Transformer architectures, which dominate current vision and language tasks.

3.A core motivation for SNNs is ultra-low power consumption, yet the paper provides no energy estimates, spike activity statistics, or hardware deployment analysis

### Questions
1.Is the temporal bias dependent on a fixed T? If T changes (e.g., from 4 to 8), must the model be recalibrated?

### Soundness
2

### Presentation
3

### Contribution
3
