# Achieving Ultra-Low Latency and Lossless ANN-SNN Conversion through Optimal Elimination of Unevenness Error

- Decision: Reject
- Scores: 2, 8, 2, 4

## Abstract
Spiking Neural Networks (SNNs) are a promising approach for neuromorphic hardware deployment due to high energy efficiency and biological plausibility. However, existing ANN–SNN conversion methods suffer notable accuracy degradation under low-latency inference, primarily caused by the $\textbf{unevenness error}$. 
To mitigate this error, prior works commonly adopt trade-off strategies at the cost of higher latency and energy consumption, such as longer time-steps, more complex spiking neuron models, or two-stage inference mechanisms. In this paper, we present a principled and efficient solution to the unevenness error. Specifically, we first develop a unified framework to quantify the unevenness error and then derive a sufficient condition for eliminating it: under an approximately constant input current, matching the ANN quantization function ($\operatorname{floor}$, $\operatorname{round}$, $\operatorname{ceil}$) with the SNN’s initial membrane potential ($0$, $\frac{\theta}{2}$, $\theta$), where $\theta$ is the firing threshold, and setting the quantization level $L$ equals to the number of time-steps $T$, which ensures exact ANN–SNN correspondence.
This finding challenges the prevailing belief that more time-steps always yield better accuracy; instead, it reveals that there exists an optimal time-step that matches the ANN’s quantization characteristics, avoiding redundant inference latency from excessive time-steps. 
Extensive experiments on CIFAR-100, ImageNet-1K, CIFAR10-DVS, and DVS-Gesture validate our theory. For example, our method achieves a state-of-the-art 74.74\% top-1 accuracy on ImageNet-1K using ResNet-34 with only 8 time-steps, demonstrating the effectiveness of our approach in low-latency SNN inference.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work presents a theoretical analysis of unevenness error and proposes a unified framework for optimal elimination of the error.

### Strengths
1. This work discusses how to eliminate error in three specific cases of the quantization function (floor, ceil, round).

### Weaknesses
1. The key point of this work is not actually about eliminating unevenness error, the so-called error elimination is just a packaging story background. As shown in Algorithm 1, line 19, the authors set $\forall t, q_t^l=\frac{W^l\sum_{t=1}^Ts_t^{l-1}\theta^{l-1}}{T}$, which means that the input current at each time-step is exactly the same. This is essentially equivalent to replacing the result of an $L$-level threshold function at one time-step with the result of a single-threshold function at $L$ consecutive time-steps, and the entire process is completely equivalent. Therefore, I tend to think that the contribution of this work to the SNN community is very poor.

2. Traditional ANN-SNN Conversion cannot be directly applied to time-series datasets such as CIFAR10-DVS. Therefore, it is curious how this work deals with the specific details, which do not seem to be discussed in the main text. I tend to think that this work may have adopted a scheme similar to the multi-threshold spiking model, followed by equivalent conversion in SNN inference stage. This idea has already been proposed in previous works.

### Questions
See Weaknesses Section.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses a fundamental bottleneck in ANN2SNN conversion: performance degradation under low-latency inference due to the unevenness error. The authors propose a Quantization-Voltage Matching (QVM) framework that provides a theoretical and practical method to eliminate this error completely. QVM achieves lossless conversion by aligning the ANN quantization function (floor, round, or ceil) with the initial membrane potential of the SNN neuron and by setting the quantization level L equal to the number of time-steps T. This configuration ensures that the spike count in the SNN exactly matches the quantized activations in the ANN, thus achieving theoretically zero conversion error. Extensive experiments on CIFAR-100, ImageNet-1K, CIFAR10-DVS, and DVS-Gesture show that QVM achieves state-of-the-art accuracy with drastically reduced latency with only 8 time-steps, surpassing all prior methods.

### Strengths
1. This paper provides a mathematically rigorous derivation for the sufficient conditions eliminating unevenness error (Theorem 3) and bridges the gap between quantization theory and membrane potential dynamics.
2. This paper achieves ultra-low latency (T=8) while maintaining accuracy comparable to full-precision ANN baselines.
3. Comprehensive empirical validation includes detailed ablation on quantization functions, membrane potentials, and quantization levels. Figures and ablation convincingly validate the theoretical claims.

### Weaknesses
1. The theory is derived for Integrate-and-Fire (IF) neurons; extension to Leaky IF (LIF) or adaptive threshold models is not shown, as LIF is more frequently used in recent research.
2. While energy efficiency is implied via reduced time-steps, no measured power or latency-on-hardware benchmarks are presented.
3. The paper references algorithmic pseudocode, but training configurations and implementation details are minimal (e.g., hyperparameters for quantization).

### Questions
1. Could the QVM framework extend to LIF neurons or temporal coding schemes beyond rate coding?
2. Is there a measurable energy efficiency improvement on neuromorphic chips or FPGA?
3. Could this principle generalize to quantized transformer-based or other architectures in SNNs?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a Quantization–Voltage Matching (QVM) framework to address the unevenness error in ANN–SNN conversion. The authors derive sufficient conditions for eliminating this error and prove that aligning ANN quantization functions with corresponding SNN initial membrane potentials can achieve theoretically lossless conversion. Experiments on CIFAR-10/100, ImageNet, and DVS datasets show state-of-the-art accuracy under very low-latency inference.

### Strengths
Extensive experiments across multiple benchmarks (CIFAR, ImageNet, DVS) convincingly support the theoretical claims. Achieving near-lossless accuracy at only 8 time-steps on large-scale datasets highlights the real-world applicability of QVM.

### Weaknesses
The major weakness of this paper lies in its organization and presentation. In the *Method* section, the authors present several theorems and derive the sufficient condition for eliminating the unevenness error. However, the detailed description or implementation procedure of the proposed QVM framework, which might be one of the most important part of the paper, is missing. Furthermore, the paper introduces a large number of mathematical symbols and notations without providing a summary or notation table, which significantly hinders readability. I strongly recommend that the authors reorganize the paper to improve logical flow, move the algorithmic details of QVM into the main body, and include a comprehensive table summarizing the symbols and their meanings. Therefore, I suggest resubmission after substantial revision and improvement of structure and clarity.

Although the motivation of ANN–SNN conversion is energy efficiency, the paper does not evaluate or discuss the computational overhead, energy consumption, or neuromorphic hardware compatibility of QVM.

### Questions
Since Theorem 3 claims that unevenness error can be theoretically eliminated, why does a small accuracy gap still remain between quantized ANN and converted SNN in practice (e.g., Table 1)?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the construction of ultra-low-latency SNNs under the ANN-to-SNN conversion framework. The authors systematically identify and formalize three key conversion errors including quantization, clipping, and unevenness and propose a new strategy named QVM that sets the number of time steps T equal to the quantization level L. Experiments show competitive accuracy on event-based datasets like CIFAR10-DVS and DVS-Gesture even at extremely low latency T=4.

### Strengths
1.The paper clearly dissects and discusses the origins and impacts of three error sources, particularly providing nuanced insights into the unevenness error.

2.The evaluation include both conventional vision datasets and event-based neuromorphic datasets, demonstrating strong generalization and positioning the method favorably against existing ANN-conversion-based SNNs.

### Weaknesses
1.The core mechanism of QVM, how ANN activations are mapped to initial membrane potentials, threshold settings, or whether calibration/fine-tuning is used, is not clearly described. No pseudocode is provided.

2.The paper reports no energy consumption, energy efficiency, or hardware simulation results.

3.Modern vision and language models heavily rely on ViTs or their spiking variants. The work only validates on CNN backbones and provides no evidence of applicability to Transformer-based architectures.

### Questions
As in weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
