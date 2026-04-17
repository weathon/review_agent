# Achieve Latency-Efficient Tempora-Coding  Spiking LLMs via Discretization-Aware Conversion

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large language models (LLMs) have achieved remarkable success while introducing critical energy bottlenecks that challenge sustainable deployment. Spiking neural networks (SNNs) provide a promising approach for energy-efficient spiking LLMs via ANN-to-SNN (A2S) conversion. Among various spike coding methods, time-to-first-spike (TTFS) coding is particularly appealing as it conveys information with a single spike, further reducing energy consumption. However, existing TTFS-based A2S conversion relies on continuous-time assumptions, requiring prohibitively large latencies (e.g., 4096 time steps) to approximate ANN's continuous values. 
This dependency leads to unacceptable inference delay in deep models, particularly LLMs, posing significant challenges for developing practical temporal-coding spiking LLMs. In this paper, we propose a discretization-aware theoretical framework that establishes a precise correspondence between discrete TTFS-based SNNs and ANNs. Our key insight reveals that conversion errors are bounded by latency-dependent terms. Motivated by these, we introduce the Quantization-Consistent ANN-to-SNN (QC-A2S) conversion, which integrates low-bit quantization with discretization-compatible TTFS neurons, achieving latency-efficient temporal-coding spiking LLMs.
Comprehensive evaluation on LLaMA models demonstrates comparable performance with dramatically reduced latency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a discretization aware ANN-to-SNN conversion framework for TTFS coding. The authors' core contribution is that, instead of adhering to traditional continuous-time assumptions, they establish a direct correspondence between a discrete TTFS-SNN and a QANN. Based on this theory, the authors have successfully applied this conversion method to Large Language Models  such as LLaMA-2 and LLaMA-3.

### Strengths
1. This paper map the conversion error of TTFS-based ANN-SNNs and model quantization error and provide a clear theoretical proof.
2. This work deploys TTFS-based SNNs on LLaMA models at the billion-parameter scale. It seems the very first attempt to use TTFS coding on such large-scale models.

### Weaknesses
See the Questions section.

### Questions
I agree that TTFS coding is a promising research direction to replace traditional rate encoding, given its potential for ultra-high sparsity. However, I have several critical questions regarding the practical advantages of the proposed method that need to be addressed:

1. Figure 1 illustrates a layer-by-layer serial inference pipeline. If the number of layers $L$ is very large (as in LLMs), the total inference latency will be $L \times T$ (where $T$ is the time window), which may be unacceptably long. Is it possible to implement a parallel TTFS inference scheme (e.g., similar to the mechanism shown in Figure 2 of the paper "Efficiently Training Time-to-First-Spike Spiking Neural Networks from Scratch")? If not, could the authors discuss the primary difficulties in implementing such parallel TTFS-SNN inference?

2. One of the paper's core motivations is the energy efficiency of SNNs. However, the advantages of the proposed TTFS-SNN over its corresponding QANN (Quantized ANN) in terms of computation and storage are not clear to me: a) Computational Complexity: In Equation 7 of section 4.1, it seems that to perform the forward of TTFS-based SNN, we first need to encode the time information into an integer (from spike time to an integer ($t-t^{l-1}_j$) and then multiply it by the weight and $\eta$. What is the advantage of this compared to multiplying the weight by the integer input  in an ANN? The traditional hardware advantage of SNNs is replacing high-energy MAC operations with simple ACC operations. However, in this paper's framework (e.g., Eq. 7 and Appendix E), multiplication operations do not seem to have been eliminated. I observed that the Softmax and RMSNorm implementations in Appendix E also appear to contain complex multiplication and power operations.  b) Storage Cost: The authors use $T=64$ in their experiments. If represented in a one-hot fashion, storing this spike train might require 64 bits. In contrast, the QNN baselines (W8A6) use only 6-bit activations, which is much smaller.  If we don't consider storing the spike train by using dataflow-type neuromorphic chip, the single spike information also need auxiliary routing information like the neuron id which effect the energy efficient. c) Leakage energy: T=64 will lead to much larger static energy consumption compare with 1 in ANN. Can author discuss it also? Further, since rate-based SNN can reduce the time window size to a super small number like 1 or 4. Then is TTFS still have advantage?

If this TTFS-SNN  perform a similar number of complex multiplications as a sparse QANN, and may even be at a disadvantage in storage and memory access, what is its true advantage in SNN compared to a QNN that can directly leverage hardware sparsity ？

3. Given the issues above, I wish to see a direct comparison between the proposed TTFS-SNN and the QNN it was converted from, in terms of energy consumption and actual wall-clock inference time. If the SNN merely matches the QNN's accuracy but cannot demonstrate an advantage in energy or speed, what is the significance of this conversion? The experiment of SNNs seem conduct on A100 also, one more question is : do experiments conducted on NVIDIA A100 GPU truly reflect the real advantages of the SNN?

One more minor question: In Equation (1) and (4), what is the blue color intended to emphasize?

### Soundness
2

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
3

### Summary
This paper proposes a method called QC-A2S for ANN-to-SNN conversion, aiming to address the high latency issue in Time-to-First-Spike (TTFS) coded Spiking LLMs. By establishing a theoretical equivalence between discrete TTFS-SNNs and quantized ANNs, the paper integrate low-bit quantization techniques with TTFS neurons to reduce inference latency. the effectiveness is validated on the LLaMA family of models.

### Strengths
1.The integration of quantization theory with TTFS-SNN conversion, proposing a discrete-equivalent paradigm shift away from continuous-approximation, shows a degree of innovation.

2.This paper establishes the equivalence between TTFS-SNNs and quantized ANNs, it provides a theoretical analysis of the error bounds.

### Weaknesses
1.Although an error bound is proposed (Theorem 3), there is no systematic experimental validation of its tightness, nor is the actual distribution of errors across layers and time-steps presented

2.In the comparison with TTFSFormer, only the high-latency setting (T=8192) is used, failing to show TTFSFormer's performance at medium latencies. this makes it difficult to assert the absolute advantage of QC-A2S

3.Despite the proposed method is emphasized hardware-friendly, there are no actual deployment tests on any neuromorphic hardware or FPGAs. the claimed energy efficiency advantages remain theoretical speculation

4.In expeirments, the quantitative analysis of the energy-accuracy trade-off is missing, so it is difficult to judge whether the method is truly superior to pure quantization methods in practical applications

5. The experiments is limited to the LLaMA architecture. The applicability to other Transformer variants (eg. encoder-only model) or non-autoregressive models is not tested, it is suggested to test more architectures

### Questions
1,does the error bound in Theorem 3 have practical guiding significance? 

2.Why wasn't TTFSFormer compared at medium latencies (T=256 or T=1024)? Is it because its performance would still be competitive with or superior to QC-A2S?

3.Are there plans to validate energy efficiency on real neuromorphic hardware? If not, isn't the claim of being energy-efficient overly optimistic?

4.Is QC-A2S applicable to non-LLaMA architectures?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the issue that ANN-to-SNN conversion based on TTFS requires significant latency. The authors propose a discretization-aware framework that establishes a correspondence between discrete-TTFS-based SNNs and ANNs.

### Strengths
1.	The fundamental issue with existing TTFS methods requiring extremely high latency is clearly identified, which indeed limits their application on LLMs.
2.	A complete theoretical derivation from continuous to discrete TTFS neurons is provided (Theorems 1-3), establishing an equivalence relationship with quantization (Corollary 1).
3.	Evaluations are conducted across multiple LLaMA models and various datasets.

### Weaknesses
1.	Theorems 1 and 2 are essentially discrete versions of the continuous theory proposed by Zhao et al. (2025). The extension from continuous to discrete is relatively straightforward, with limited theoretical innovation. The error analysis in Theorem 3 lacks comparative validation between the theoretically predicted error and the actual observed error. The comparison in Figure 2 is insufficiently clear and requires more detailed explanation.
2.	The experimental setup has issues. The absence of comparisons for TTFSFormer at T=64 prevents a fair evaluation of method performance. Table 3 lacks comparisons with other models, diminishing its persuasiveness. Table 3 only investigates the impact of T, neglecting the influence of quantization bit depth. We recommend adding an analysis of the trade-off between quantization bit depth and latency.
3.	Experiments max out at 70B parameters, with no discussion of scalability for larger models.
4.	Recommendations: Add charts, visualizations, pseudocode, etc., to enhance narrative clarity.

### Questions
1.	The specific implementation details of the method remain somewhat ambiguous. Section 4.4 states, “we first apply established techniques, such as post-training quantization,” but does not explicitly specify which PTQ method is used. Is it PrefixQuant? If so, what distinguishes this approach from directly employing PrefixQuant?
2.	How is the trade-off between quantization bit depth n and time window T selected? What is the rationale for choosing T=64? Is this choice empirical or based on theoretical analysis?

### Soundness
3

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
This paper aims to convert LLM into SNN using TTFS encoding. To address the issue of high inference latency in the converted SNN, it is proven that the activation process of discrete TTFS neurons is equivalent to a quantization function. Therefore, pre-trained quantized LLM can be converted into SNN using TTFS encoding, and the conversion results are verified on the LLaMA series models.

### Strengths
The writing is smooth, the theory is sound, and the implementation results prove the validity of the theory.

### Weaknesses
1. It is still unclear what advantages this method has compared to the baseline, i.e., the quantized LLM.
2. The guiding idea of this method (i.e., converting the quantized ANN into an SNN) may not be original. Similar ideas have been proposed in previous works, and this work may only extend this idea to SNNs with TTFS encoding.[1][2]
3. The lack of a pseudocode description of the algorithm makes it insufficiently clear.

[1] Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks

[2] SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking

### Questions
1. What specific quantization methods are used when quantizing LLM in this paper? Are specific quantization methods required? Are both weights and activations quantized? For operations in the nonlinear parts, are the calculations performed using quantized activations and weights? Please elaborate in detail.
2. Please explain the advantages of converting the quantized LLM into an SNN? Does it improve inference efficiency, or does it reduce energy consumption? Can data support be provided?

### Soundness
3

### Presentation
3

### Contribution
3
