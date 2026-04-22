# Parallel Training in Spiking Neural Networks

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
Spiking neurons mimic the spatiotemporal dynamics of biological neurons and their spike-based communication, endowing Spiking Neural Networks (SNNs) with biological plausibility and low-power operation. Yet these dynamics impose strict temporal dependencies on neuronal states, preventing parallel training and creating a fundamental bottleneck to efficient, scalable optimization. This work introduces a novel functional perspective to address this challenge. Specifically, we argue that the reset mechanism, which induces state dependencies, should be removed. However, any modification must satisfy two principles: i) preserving — and even enhancing — the functions of reset as a core biological mechanism; and ii) enabling parallel training without sacrificing SNNs’ inherently serial inference, which underpins their energy efficiency. To this end, we identify functions of the reset mechanism and analyze how to reconcile parallel training with serial inference, upon which we propose a dynamic decay spiking neuron that combines a causal convolution structure with an optimized spike firing pattern. We demonstrate the efficiency and effectiveness of our approach across diverse network architectures and task benchmarks, including image classification, neuromorphic event processing, time-series forecasting, and language modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel parallel spiking-neuron model to accelerate SNN training. Specifically, the membrane potential is generated via Eq. (19), which resembles the charging equation of an LIF neuron; because this equation is fully linear, it can be rewritten as a non-recurrent sum and efficiently implemented with matrix multiplication. The authors further enhance the model by making the coefficients in the charging equation depend on the inputs of several previous time-steps, yielding a learnable gating effect. Finally, integer spikes are emitted during training while binary spikes are used at inference, speeding up training. Compared with the conventional LIF neuron and the recent PSN, the proposed method is faster than LIF but still slower than PSN at T = 32. Extensive experiments on multiple tasks show that it outperforms PSN in most cases.

### Strengths
- The method is presented in detail, allowing readers to grasp its mechanics easily.  
- The experimental section is very comprehensive, covering almost every scenario one could consider.

### Weaknesses
The proposed neuron remains slower than PSN; although the authors attribute this to the absence of a customized CUDA kernel, whether a CUDA implementation could actually surpass PSN still needs verification.

### Questions
Is the use of an internal sigmoid operation reasonable? It may be unsuitable for deployment on neuromorphic chips.

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
This paper provides an insightful analysis of the reset mechanism in spiking neurons and introduces a DSN model that supports parallel training while maintaining serial inference capability. The topic is relevant, and the motivation is clear. However, the paper lacks a theoretical analysis to justify the effectiveness of DSN’s nonlinear operations. The experimental design also raises concerns regarding fairness. Furthermore, the manuscript does not clarify whether DSN preserves the sparse firing property essential to SNN efficiency or analyze the additional inference latency introduced by the model.

### Strengths
1.The authors provide a detailed analysis of the reset process in spiking neurons, attributing its functions to introducing nonlinearity and regulating the membrane potential.

2.The authors propose a DSN model that enables parallel training while preserving the serial inference capability.

### Weaknesses
1.Lack of Theoretical Justification: Although the authors identify the reset process as introducing nonlinearity, they do not provide a theoretical explanation showing that DSN achieves a more principled or effective nonlinear behavior.

2.Fairness of Model Comparison: DSN introduces additional parameters (137M) compared to SPiKE-SSM (75M) and SpikingSSM (75M), which could partly account for the observed improvement.

3.Potential Loss of Sparsity: The binary behavior of the reset mechanism enforces the sparse firing property fundamental to SNN efficiency and biological plausibility. The manuscript does not clarify whether DSN retains this property.

4.Higher Inference Latency: The introduction of the additional N-dimensional may incur extra inference latency[1], particularly for DVS datasets and sequential datasets.

### Questions
1.The authors are encouraged to discuss the design choice of N across different datasets and include ablation studies on N to better demonstrate the effectiveness of the proposed method.

2.DSN appears to introduce additional MAC operations, which may increase deployment complexity on neuromorphic hardware. The authors should clarify whether the energy consumption analysis includes the internal neuronal dynamics.

3.The authors should further elaborate on how DSN preserves the event-driven property of spiking neurons during inference, particularly within the Transformer architecture.

[1]Scaling Spike-driven Transformer with Efficient Spike Firing Approximation Training, T-PAMI 2025.

### Soundness
2

### Presentation
2

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
The authors use spiking neural networks for image and neuromorphic dataset classification. They propose a new spiking neuron model called the Dynamic Decay Spiking Neuron (DSN).

### Strengths
* The DSN is new.
* The DSN can be computed either sequentially or in parallel

### Weaknesses
1) The advantages w.r.t. an earlier proposal, the PSN / sliding PSN  (Fang et al. 2023) are not clear. The DSN is significantly more complex, and less neuromorphic-hardware-friendly:
- it uses integer spike, not binary spikes (Eq 7). With binary spikes, the DSN is less accurate (87.45%) than the PSN (88.45%) on Seq CIFAR10 (Table 2)
- it uses a dynamic, input-dependent leak rate (Eq 8 and 9)
- the Enhanced DSN uses a non-local neuron mixing operation.

The DSN is about 3 times slower than the PSN (Table 1)

These disadvantages are not always compensated by a boost in accuracy. For example, on CIFAR10-DVS the DSN and the sliding PSN have the same accuracy (Table 4).

As the authors correctly say, the PSN has T^2 parameters, "resulting in substantial
memory and computational overhead" when T is large. But the sliding PSN has only k parameters. So the authors should add the sliding PSN in tables 1 (throughput) and 7 (energy).

2) The DSN accuracy is lower than the SOTA on ImageNet and CIFAR10-DVS (Table 4)

### Questions
Do you think the DSN is implementable on existing neuromorphic chips?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides a detailed review of previous parallel spiking neuron models, argues that the "reset" mechanism in SNNs serves two core functions: introducing nonlinearity and controlling membrane potential, and claims that traditional soft/hard reset mechanisms are inefficient. Based on this analysis, the paper proposes a Dynamic Decay Spiking Neuron (DSN). This model removes the reset mechanism, replaces the fixed membrane potential decay with an input-dependent causal convolution, and incorporates an integer-valued firing pattern. The DSN is designed to support both parallel training and serial inference. Experiments on diverse network architectures and benchmarks—including image classification, neuromorphic event processing, time-series forecasting, and language modeling—validate the efficiency and effectiveness of the proposed method.

### Strengths
1.  The paper analyzes the role of the reset mechanism from a novel "functional perspective," using this as a guiding principle to design the new neuron.
2. The DSN model is extensively validated across diverse data modalities and network architectures, demonstrating state-of-the-art or highly competitive performance.
3. The paper is clearly written and well-structured.

### Weaknesses
1. **Limited Model Innovation:** The core mechanism of the proposed DSN can be understood as applying a sigmoid function to the output of a sliding PSN [1] and using that as a gating signal for the previous state $H_{t-1}$ and the current input $X_t$. And it integrates the integer-valued firing mechanism from ILIF [2], using integers instead of binary spikes as the neuron's output.

2. **Unfair Experimental Setup:** The paper's experimental comparisons suffer from significant fairness issues. According to the characteristics of ILIF, during inference,  a single-timestep (T=1) integer firing (e.g., N=4) is equivalent to the spike accumulation of multiple time-steps (e.g., T=4, N=1). Therefore, studies of ILIF typically adopt a (T=1, N=4) configuration for a fair comparison against other spike-fire methods using (T=4, N=1). However, this paper uses a (T=4, N=4) configuration, which is effectively equivalent to T=16 (with N=1), and compares against baseline models using a (T=4, N=1) setup. This comparison is not equivalent and may lead to a significant overestimation of the proposed model's performance.

3. **Unreasonable and Incomplete Energy Analysis:** The paper's energy consumption analysis is seriously flawed. Section 5.3 states that its evaluation follows [3]. However, [3] uses LIF neurons, whose operational energy is negligible compared to the synaptic layer, allowing the authors to ignore the neuron layer's energy cost. In contrast, the DSN includes computationally intensive operations at the neuron layer (such as floating-point causal convolution and the expensive sigmoid function). The energy cost of these operations is likely comparable to, or even higher than, that of the synaptic layer. Therefore, an analysis method that only considers synaptic energy is unreasonable and incomplete.  Furthermore, the DSN outputs an integer rather than a binary spike. This distinction is critical and must be factored into the synaptic energy analysis. It is unclear whether the paper has properly considered this in its synaptic energy calculations. The authors should separately list and compare the energy consumption of both the synaptic and neuron layers.

4. **Inefficient Computation, Contradicting the Goal of PSN:** As shown in Table 8, the DSN's runtime is significantly slower than that of PSN and even slower than the CUDA-implemented LIF neuron [4]. The authors attribute this to their Triton implementation being slower than CUDA, but this explanation is unconvincing. Furthermore, the original motivation for Parallel Spiking Neurons is to accelerate SNN training on GPUs. The fact that the DSN's training is even slower than traditional serial neurons runs counter to this design goal.

> [1] Fang W, Yu Z, Zhou Z, et al. Parallel spiking neurons with high efficiency and ability to learn long-term dependencies[J]. Advances in Neural Information Processing Systems, 2023, 36: 53674-53687.
>
> [2] Luo X, Yao M, Chou Y, et al. Integer-valued training and spike-driven inference spiking neural network for high-performance and energy-efficient object detection[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 253-272.
>
> [3] Yao M, Hu J, Zhou Z, et al. Spike-driven transformer[J]. Advances in neural information processing systems, 2023, 36: 64043-64058.
>
> [4] Fang W, Chen Y, Ding J, et al. Spikingjelly: An open-source machine learning infrastructure platform for spike-based intelligence[J]. Science Advances, 2023, 9(40): eadi1480.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
