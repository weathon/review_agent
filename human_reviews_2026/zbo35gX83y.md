# Spiking Neural Network with Mixture of Heterogeneous Enhancement Experts for Robust Underwater Object Detection

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Underwater object detection faces unique challenges from haze, color distortion, and low contrast caused by light absorption and scattering, which significantly degrade image quality and detection performance. We propose HE-MoESNN, a spiking neural network that integrates a Mixture of Heterogeneous Enhancement Experts (HE-MoE) with a lightweight Forward Spiking Neural Network (FSNN) backbone. Unlike conventional MoE frameworks that feed identical inputs to all experts, HE-MoE assigns modality-specific inputs consisting of dehazing, color correction, and contrast enhancement to three parallel experts and fuses their outputs through a shared gating router. This design promotes expert diversity and enables the network to exploit complementary enhancement cues. FSNN improves efficiency by replacing costly ANN activations and conventional convolutions with signed spiking neurons and ternary convolutions, reducing computation while maintaining competitive accuracy. Extensive experiments on the RUOD and DUO benchmarks demonstrate that HE-MoESNN achieves state-of-the-art performance while maintaining high computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a spiking neural network framework that combines heterogeneous enhancement experts with a forward spiking neural network backbone for underwater object detection. The HE-MoE module assigns different enhancement modalities to three specialized experts. The FSNN backbone replaces standard convolutions with ternary operations using signed spiking neurons to reduce computational cost. Experiments on RUOD and DUO datasets demonstrate competitive performance with improved efficiency compared to existing methods.

### Strengths
1. The heterogeneous expert design that assigns modality-specific inputs to different experts is a meaningful departure from conventional MoE approaches.

2. The paper provides thorough ablation studies examining individual enhancement components, backbone comparisons, and systematic evaluation across multiple model scales.

3. The focus on computational efficiency through spiking neural networks addresses real deployment constraints for underwater detection systems.

### Weaknesses
1. The core components (spiking neural networks, MoE architectures, underwater enhancement techniques) are well-established, the main contribution is their combination rather than fundamental algorithmic innovation.

2. The computational cost analysis only considers arithmetic operations while ignoring memory overhead, routing costs, and the fact that all three experts must be loaded simultaneously, potentially making the efficient SNN approach less efficient than claimed in real deployment scenarios.

3. The paper claims computational efficiency as a key advantage but their largest model (HE-MoESNN-L) actually uses more parameters and FLOPs than many baselines.

### Questions
1. Why exactly three experts, and how was this number determined? The paper doesn't explain if 2 or 4 experts might work better or provide guidelines for choosing enhancement types.

2. Does the method work beyond these two specific datasets? The authors only test on RUOD and DUO but underwater conditions vary dramatically. What about different water clarity, depths, or lighting?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose HE-MoESNN, a spiking neural network that integrates a Mixture of Heterogeneous Enhancement Experts (HE-MoE) with a lightweight Forward Spiking Neural Network (FSNN) backbone. Unlike conventional MoE frameworks that feed identical inputs to all experts, HE-MoE assigns modality-specific inputs—including dehazing, color correction, and contrast enhancement—to three parallel experts and fuses their outputs through a shared gating router.

### Strengths
1. The manuscript presents a clear and well-structured method.
2. Good performance is demonstrated according to the comparisons reported in the paper.

### Weaknesses
1. While the proposed framework is technically well-described, the network design is incremental and lacks theoretical depth. Although the paper combines image enhancement techniques with a mixture-of-experts architecture to design a lightweight network, it lacks novel insight. 
2. It is unclear why these three enhancement methods were selected. How do other enhancement methods perform in comparison? Moreover, the paper lacks a thorough analysis of the computational efficiency and enhancement quality of the chosen methods, as well as their impact on the subsequent model. Overall, the work lacks comprehensive justification and analysis.
3. The rationale for selecting YOLOX as the base model is not clearly justified. It would be valuable to analyze how the proposed HE-MoE performs when combined with other representative object detection architectures.

### Questions
Please see the Weaknesses.

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
5

### Summary
This paper introduced the HE-MoESNN framework for underwater object detection, targeting the challenges of image degradation caused by light absorption and scattering. The proposed HE-MoE module integrates heterogeneous experts designed for defogging, color correction, and contrast enhancement, and employs a shared gating router to adaptively fuse their outputs. Furthermore, the paper employed a lightweight FSNN backbone to replace traditional convolutional structures, which significantly reduced computational cost while maintaining high accuracy on the RUOD and DUO data sets.

### Strengths
1. The HE-MoE module breaks the limitation in traditional MoE where all experts share the same input, enabling experts to specialize in different enhancement domains. The application of FSNN in underwater object detection also exhibits certain innovativeness.
2. The HE-MoE module possesses energy efficiency advantages on embedded or edge underwater platforms 
3. The paper included detailed component-wise analyses that verified the contribution of each module (HE-MoE, FSNN,different enhancement modalities, etc.).

### Weaknesses
1. The MoE baseline uses a single RGB input, while the proposed HE-MoE receives three enhanced images (dehazed, color-corrected, and contrast-enhanced). Therefore, it remains unclear whether the observed improvement stems from the multi-expert architecture itself or merely from the enhanced image inputs.
2. The paper frequently emphasized the complementarity among the three enhancement experts, but no experimental analysis (e.g., feature similarity or mutual information) was provided to verify whether these experts indeed learn complementary representations.
3. HE-MoESNN-L’s FSNN-FLOPs only include low-energy addition/subtraction, while DJLNet’s ANN-FLOPs involve high-energy multiplication-addition. HE-MoESNN-L has the multi-modal HE-MoE module , and DJLNet only a single-modal decolorization module. It’s unclear if their performance gap (mAP 59.0 vs 57.5) and efficiency difference came from "HE-MoE + FSNN" advantages or inconsistent computational/functional configurations.

### Questions
1. The motivation focused on mitigating underwater degradation, yet experiments only showed detection accuracy. Have the authors evaluated whether the model actually improves visual quality or robustness under different degradation levels to substantiate the claimed motivation?
2. Beyond theoretical FLOPs and number of parameters, did you measure real-world inference speed or energy consumption (e.g., FPS, latency, or power usage) to assess deployment feasibility on underwater devices?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces HE-MoESNN, a spiking neural network model for underwater object detection. The key
idea is to use a mixture of enhancement experts, where each expert focuses on a different type of underwater
image improvement (one for dehazing, one for color correction and one for contrast enhancement). A routing
module combines the outputs from these experts. The detection network is built as a Forward Spiking Neural
Network with signed spiking neurons and ternary convolutions to reduce computation. The model is evaluated
on the RUOD and DUO datasets and shows higher detection accuracy and better efficiency compared to both
standard deep neural networks and previous SNN-based methods.

### Strengths
The motivation of the paper is clear and makes sense for the underwater setting. The authors explain that underwater images often suffer from haze, color distortion and low contrast and the enhancement modules they choose directly target these issues.

The idea of using a heterogeneous mixture-of-experts is meaningful. Since each expert receives a differently enhanced version of the image, this encourages each branch to learn different features instead of all doing the same thing.

The use of a Forward Spiking Neural Network with signed spiking neurons and ternary convolutions helps reduce computation. This is useful for situations where power or hardware resources are limited, such as underwater robots or remote-operated devices

The experiments are broad and include comparisons with general object detectors, underwater-focused models and also previous spiking neural network detectors. The proposed model shows improvements in both detection accuracy and computational efficiency.

The ablation studies are helpful because they show the effect of each key component separately, including the heterogeneous experts module, the MoE vs standard fusion and the difference between the spiking and non-spiking backbone.

### Weaknesses
The paper mentions that the model is efficient due to the spiking design and ternary convolutions, but there are no actual hardware tests. There are no latency, FPS or energy measurements, so it is hard to know how the model performs in real deployment, especially
since SNN efficiency can depend on the device used.

The results seem to come from single runs and there are no standard deviation or confidence intervals reported. Because of this, it is difficult to know how stable the improvements are or whether they might vary with different random seeds.

The experiments are only done on two underwater datasets (RUOD and DUO). Both datasets are from similar underwater environments. So, it is unclear how well the model would perform in other settings, such as different water types, lighting, turbidity or general degraded images.

Some implementation details are not described fully, such as how the enhancement modules are parameterized or how the routing and spiking thresholds are tuned. This may make it harder for others to reproduce the exact performance.

### Questions
The paper mentions that the model is efficient due to the spiking design and ternary convolutions, but there are no actual hardware tests. There are no latency, FPS or energy measurements, so it is hard to know how the model performs in real deployment, especially
since SNN efficiency can depend on the device used.

The results seem to come from single runs and there are no standard deviation or confidence intervals reported. Because of this, it is difficult to know how stable the improvements are or whether they might vary with different random seeds.

The experiments are only done on two underwater datasets (RUOD and DUO). Both datasets are from similar underwater environments. So, it is unclear how well the model would perform in other settings, such as different water types, lighting, turbidity or general degraded images.

Some implementation details are not described fully, such as how the enhancement modules are parameterized or how the routing and spiking thresholds are tuned. This may make it harder for others to reproduce the exact performance.

### Soundness
2

### Presentation
2

### Contribution
2
