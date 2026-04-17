# One-Timestep is Enough: Achieving High-performance ANN-to-SNN Conversion via Scale-and-Fire Neurons

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Spiking Neural Networks (SNNs) are gaining attention as energy-efficient alternatives to Artificial Neural Networks (ANNs), especially in resource-constrained settings. While ANN-to-SNN conversion (ANN2SNN) achieves high accuracy without end-to-end SNN training, existing methods rely on large time steps, leading to high inference latency and computational cost. In this paper, we propose a theoretical and practical framework for single-timestep ANN2SNN. We establish the Temporal-to-Spatial Equivalence Theory, proving that multi-timestep integrate-and-fire (IF) neurons can be equivalently replaced by single-timestep multi-threshold neurons (MTN).
Based on this theory, we introduce the Scale-and-Fire Neuron (SFN), which enables effective single-timestep ($T=1$) spiking through adaptive scaling and firing. Furthermore, we develop the SFN-based Spiking Transformer (SFormer), a specialized instantiation of SFN within Transformer architectures, where spike patterns are aligned with attention distributions to mitigate the computational, energy, and hardware overhead of the multi-threshold design.
Extensive experiments on image classification, object detection, and instance segmentation demonstrate that our method achieves state-of-the-art performance under single-timestep inference. Notably, we achieve 88.8\% top-1 accuracy on ImageNet-1K at $T=1$, surpassing existing conversion methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new ANN-to-SNN conversion framework called **Scale-and-Fire Neuron (SFN)**, which enables **single-timestep (T=1)** inference while maintaining high accuracy. The authors introduce the **Temporal-to-Spatial Equivalence Theory**, which proves that multi-timestep integrate-and-fire (IF) neurons can be equivalently replaced by single-timestep **multi-threshold neurons (MTN)**.  

Building upon this theory, the SFN integrates:
1. A **membrane potential scaling mechanism**, where the **scaling factor (λ)** is optimized via **Bayesian optimization** to balance spike sparsity and accuracy.
2. An **adaptive fire function**, whose discrete thresholds $ \theta_i $ are *determined* based on the activation distribution and are proportional to the optimized scaling factor ($ \theta_i = \lambda \theta y_i $).

The authors extend this design to Transformer architectures, forming the **SFN-based Spiking Transformer (SFormer)**, which aligns spike distributions with attention patterns. Experiments show strong performance across multiple vision benchmarks:
- **ImageNet-1K:** 88.8% Top-1 accuracy at $T = 1$.  
- **COCO-2017 Detection:** 78.2% mAP@0.5.  
- **Energy efficiency:** up to 81% reduction in energy consumption relative to the ANN baseline.

Overall, the paper provides both a theoretical and practical framework for **high-performance single-timestep ANN-to-SNN conversion**, bridging temporal spiking integration and spatial multi-threshold encoding.

### Strengths
The paper introduces the **Scale-and-Fire Neuron (SFN)** and the **Temporal-to-Spatial Equivalence Theory**, enabling single-timestep ($T=1$) ANN-to-SNN conversion with high accuracy. Experiments on ImageNet and COCO demonstrate good empirical performance and energy savings. The integration of a Bayesian-optimized scaling factor ($\lambda$) and adaptive multi-threshold firing is both elegant and effective.

The work is clearly written. It shows that high-performance ANN-to-SNN conversion can be achieved without multi-timestep accumulation. The framework’s applicability to Transformer architectures (SFormer) further enhances its relevance to large-scale vision models and energy-efficient AI research.

### Weaknesses
1) Originality: the Temporal-to-Spatial Equivalence is new, but the practical recipe (multi-threshold neurons at single step $T=1$ with scaling $\lambda$ and density-aware thresholds) overlaps with prior multi-threshold or dynamic-threshold conversion (Huang et al., 2024; Li et al., 2025; MT-SNN). The paper does not show a clear win at matched $N$, nor operator-aware bounds beyond the theory’s non-negative/linear assumptions.

2)  SoftMax is explicitly handled via a max-driven cap $\theta_{\text{softmax}}$; GELU is only implicitly covered (no operator-level rule/ablation); LayerNorm handling is missing (no pre/post-LN placement, no treatment of signed activations). This is a material reproducibility and validity gap for ViT-style models.

3) Practical efficiency evidence is incomplete  
The method trades time for spatial threshold multiplicity. Per-neuron updates scale with $N$, and spike/event traffic can grow with $N$. No $N$-sweep or absolute energy (mJ/inference), and no accuracy-matched multi-timestep IF ($T>1$) baseline with end-to-end latency/energy and per-layer spike histograms.

4) Cost/justification of $\lambda$  
$\lambda$ is selected by Bayesian optimization; the search budget/range/compute cost are undisclosed, and there is no comparison to simple analytical or percentile estimators, weakening the “training-free” claim.

5) Sensitivity and robustness  
A max-driven $\theta_{\text{softmax}}$ is outlier-sensitive; there is no percentile-based alternative or drift analysis. Stability of density-aware thresholds under activation shifts is also unreported.

6) Large $N$ can be counterproductive  
Energy and event counts tend to rise with $N$; neuron-side work grows $O(N)$ and can erode the latency gain; hardware burden (threshold storage/lookup, event queues) increases; fine binning can overfit activation noise and hurt generalization; returns diminish once $N$ already covers the bulk of the activation mass (often around a moderate value like 32). The chosen $N$ should be justified via an $N$-sweep (latency/energy/event counts/bandwidth/accuracy) and a fair $T>1$ baseline.

References

Huang, X. et al., 2024. “Towards High-Performance Spiking Transformers from ANN to SNN Conversion.” https://arxiv.org/pdf/2502.21193  
Li, Y. et al., 2025. “Multi-Threshold Neuron Models for Single-Step ANN-to-SNN Conversion.” https://arxiv.org/pdf/2503.00301  
Wang, Z. and Zhang, T., 2023/2024. “MT-SNN: Enhance Spiking Neural Network with Multiple Thresholds.” https://arxiv.org/pdf/2303.11127  
Fan, Y. et al., 2025. “A multisynaptic spiking neuron for simultaneously encoding spatiotemporal dynamics.” Nature Communications. https://www.nature.com/articles/s41467-025-62251-6

### Questions
Q1. Originality and theoretical contribution
The Temporal-to-Spatial Equivalence theory is interesting, but its practical form (multi-threshold + λ-scaling + adaptive thresholds) resembles prior MT-SNN or dynamic-threshold methods (Huang 2024; Li 2025). Could the authors clarify what is fundamentally novel in Scale-and-Fire Neuron?

Q2. Theorems 1–2 assume non-negative inputs and linear operations, which may not hold with GELU, LayerNorm, or signed activations.
Can the authors quantify the impact of these violations and explain how such nonlinearities are handled or approximated in practice?

Q3. Scalability and applicability
Table 1 shows O(N) neuron cost, but experiments fix N = 32.
Can the authors include an N-sweep (e.g., 8–64) showing accuracy, latency, and energy trade-offs?

### Soundness
3

### Presentation
2

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
This paper proposes SFN, which enables converted SNNs to achieve good performance within 1 time step.

### Strengths
The proposed SFN enables converted SNNs to achieve good performance with only 1 timestep.

### Weaknesses
1. The proposed Scale-and-Fire Neuron (SFN) largely mirrors the non-uniform activation quantization with calibration used in quanted ANNs.  The so-called "Temporal-to-Spatial Equivalence Theory" is quite obvious and superficial. The resulting SFN actually transmits floating-point values and relies heavily on intricate searches for an appropriate scaling factor, which is unlikely to be a fixed constant that is completely task-independent and model-independent (as also acknowledged by the authors in Section 3.2.2), undermining the purported motivation and simplicity of spiking/neural dynamics.

2. This work proposed to replace "multi-step membrane potential integration" with "single-step multi-threshold", aiming to transform temporal integration into spatial multi-level thresholds. However, this eliminates neural dynamics (membrane potential evolution and temporal correlation of residual membrane potential) with spatial quantized intervals. The resulting SFN is more of an engineering parameterization of step interval and threshold assignment, and is unrelated to neural dynamics.

3. To achieve better performance, the authors stacked multiple heuristics: positive and negative branches, specialized upper threshold bounds for the SoftMax layer, Bayesian optimization of λ, quantile p, and varying threshold densities at each level. These are all a combination of a posteriori calibration properties, resulting in increased implementation complexity and parameter sensitivity, undermining the simple temporal processing mechanism that SNNs are supposed to possess. In contrast, a clean IF/BPTT or reliable ANN quantization scheme is more maintainable and verifiable.

4. The purpose of ANN-to-SNN conversion is to convert the pretrained ANN into an SNN to take advantage of the high performance of ANNs and the high energy efficiency of SNNs.  However, the proposed SFN cannot actually be converted into a single/simple IF/LIF model during inference, and even complicates the membrane potential accumulation process of the spiking neuron. In fact, during this converted SNN inference, the computational complexity of each operation is not just O(1). Therefore, it is unreasonable to estimate the energy consumption of 0.9pJ per synaptic operation as in previous literature.

5. SFN is not a standard neuron. Multiple threshold comparisons, non-uniform threshold mapping, positive and negative branching, and dedicated SoftMax upper bounds all introduce control flow and table lookup overhead; implementing this type of piecewise non-uniform thresholding is not "free." The energy consumption metrics used in this paper are still approximated by an operator energy model (treating multiplication as multiple additions), without actual end-to-end chip or FPGA-level simulations/measurements or timing analysis. More importantly, while single-step execution with N thresholds reduces the repetition of non-neuronal operators, it significantly increases the number of firing spikes and additional operations, e.g., comparisons per layer when N is large, potentially offsetting the energy/latency gains. Overall, SFN is not a standard neuron, while the paper does not provide the corresponding hardware deployment strategy and energy consumption/delay analysis. Therefore, the provided analysis results at the operation-level using normal IF/LIF models are obviously not rigorous.

### Questions
1. SFN requires a lot of hyperparameters. Can they have a certain degree of generalization across different tasks/models?

2. SFN actually transmits a floating-point value. What's the actual difference between SFN and quantized ANN activation? Running a quantized ANN only requires one timestep, while it does not need that many complicated hyperparameters or additional runtime computations(e.g., comparisons of positive/negative values). 
If the author cannot provide the hardware implementation strategy or analysis, what are the actual benefits of SFN compared to mature quantized ANNs that can run on dozens of well-commercialized hardware?

3. It seems that SFN aims to force the firing rate of the SNN to align with the activation value of each layer of the ANN. If T is not 1, can it run step-by-step(asynchronously) rather than layer-by-layer(synchronously)?

4. For SNNs, a single time step is almost a degenerate case. This means no stateful units, no time integration, and no dynamics. I am curious whether the proposed SFN can work well on neuromorphic datasets or why does the author consider SFN to be classified as a spiking neuron? Clearly, the values ​​transmitted by SFN are not binary spikes, cannot convert MAC to AC, are not energy-efficient, and layer-by-layer calibration would prevent the network from operating asynchronously.

### Soundness
1

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
4

### Summary
This paper presents a novel framework for high-performance, single-timestep (T=1) Artificial Neural Network (ANN) to Spiking Neural Network (SNN) conversion. The authors introduce a "Temporal-to-Spatial Equivalence Theory" to formally connect multi-timestep Integrate-and-Fire (IF) neurons with single-timestep Multi-Threshold Neurons (MTN). Based on this theory, they propose the Scale-and-Fire Neuron (SFN) and an SFN-based Spiking Transformer (SFormer). In the end, the authors show a new SOTA on ImageNet-1K with 88.8% top-1 accuracy at T=1.

### Strengths
1. The authors attempt to provide a rigorous theoretical underpinning for their single-time step approach through the "Temporal-to-Spatial Equivalence Theory". Grounding the methodology in a formal equivalence (even under ideal conditions) is a commendable effort that adds depth and clarity to the proposed conversion framework. 

2. The proposed Scale-and-Fire Neuron (SFN) is a well-motivated design. It moves beyond a naive multi-threshold implementation by incorporating a scaling factor (λ) and an adaptive firing function. This design directly addresses the practical challenges of converting large models where activation distributions can be highly skewed and varied. The use of Bayesian optimization to tune λ is a principled approach to finding a good balance between accuracy and spike sparsity.

### Weaknesses
1. In Table 2, the performance comparison between different architectures is obviously unfair, the author should compare with other ANN2SNN methods using the same model architecture 
2. Since the model uses multi thresholds and the time step is 1 only, the model is more similar to an activation quantized only model. Therefore, the comparison with some typical quantization methods like [1], is also necessary. 
3. The citations of all existing other methods on all performance comparison tables are missing.

[1] Elias Frantar, et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.

### Questions
None.

### Soundness
2

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
This paper aims to reduce the inference time step of ANN-to-SNN conversion to 1 time step, while maintaining high conversion accuracy. Specifically, it proves that under certain conditions, multi-time-step Integrate-and-Fire (IF) neurons are equivalent to one-time-step Multi-Threshold Neurons (MTN). Then, it proposes the Scaleand-Fire Neuron (SFN) model for use in the Transformer architecture to convert ANNs into SNNs with one-time-step.

### Strengths
The writing of this paper is fluent, with a well-structured organization. All the theories claimed in the paper have been properly illustrated and elaborated.

### Weaknesses
1. **Question on the correctness of Theorem 1**: Theorem 1 requires that "the input is bounded by θ", which constitutes a rather strong constraint. How to ensure this constraint can be satisfied? This is because the input is correlated with both input activations and weights.  


2. **Clarification on the definition of variables and equivalence of outputs**: What is the meaning of \( o_M \) in Equation (27)? Is it consistent with the definition of \( o(t) \) in Equation (6)? It is noted that \( o_M \) has no temporal dimension, while \( o(t) \), although containing a temporal dimension, only represents the output at a single time step. In summary, please provide a detailed explanation of the following:  
   - What exactly are the equivalent outputs of the MTN and the IF  model?  Is their equivalent output given by Equation (32)?  
   - Additionally, since the entire paper focuses on **one time step**, the temporal dimension becomes meaningless. In this case, the work degrades to a simple activation quantization task. Consequently, the MTN—i.e., the model described in Equation (6), merely serves as an activation quantizer for artificial neural networks (ANNs).  

3. **Question on the necessity of this work**: Theorem 2 is self-evident and does not require a separate proof, which raises doubts about the necessity of this study. For ANN-to-SNN conversion works, the core goal is to transform ANNs into SNNs. However, the essence of this work is an ANN-to-ANN conversion (the MTN  functions only as an activation quantization function and does not introduce temporal dimension).  

4. **Further question on the necessity of this work**: Considering that this paper discusses one timestep, are the t in h(t) and o(t) in formulas (6), (8), and (10) all 1? If so, then SFN is just quantifying the activation after observing the output of the ANN. 

5. **Question on the scope of application of the proposed theory and the relationship between MTN and SFN**: Are the MTN and SFN the same concept? Specifically:  
   - Theorems 1, 2, and 3 are all proofs for the MTN, yet no proof is provided to verify the relationship between the MTN and the SFN.  
   - Why can Theorems 1, 2, and 3 be applied to the SFN and still hold true?  

6. **Comment on experimental results**: The experimental results on COCO-2017 object detection may lack persuasiveness, as the backbone networks used (in this work and comparative studies) are not consistent.

### Questions
The problem is detailed in the above weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
