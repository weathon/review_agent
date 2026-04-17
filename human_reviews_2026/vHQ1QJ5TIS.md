# Spiking Brain Compression: Post-Training Second-order Compression for Spiking Neural Networks

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Spiking Neural Networks (SNNs) have emerged as a new generation of energy-efficient neural networks suitable for implementation on neuromorphic hardware. As neuromorphic hardware has limited memory and computing resources, weight pruning and quantization have recently been explored to improve SNNs' efficiency. State-of-the-art SNN pruning/quantization methods employ multiple compression and training iterations, increasing the cost for pre-trained or very large SNNs. In this paper, we propose a novel one-shot post-training compression framework, Spiking Brain Compression (SBC), that extends the classical Optimal Brain Surgeon (OBS) method to SNNs. SBC replaces the current-based objective found in common layer-wise compression method with a spike train-based objective whose Hessian is cheaply computable, allowing a single backward pass to compress synapses and analytically rescale the rest. Applying SBC to SNN pruning and quantization, Our experiments on models trained with neuromorphic datasets (N-MNIST, CIFAR10-DVS, DVS128-Gesture) and large static datasets (CIFAR-100, ImageNet) show state-of-the-art results for SNNs one-shot post-training compression methods, with single-digit to double-digit accuracy gains compared to ANN methods applied to SNNs. Combined with finetuning, SBC is also competitive with the accuracy of costly iterative methods, while cutting compression time by two orders of magnitude.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents Spiking Brain Compression (SBC), a post-training, second-order compression framework designed specifically for SNNs. SBC generalizes the Optimal Brain Surgeon (OBS) and Optimal Brain Compression (OBC) approaches by introducing a spike train-based loss using the Van Rossum Distance (VRD), whose Hessian can be efficiently computed, allowing practical, one-shot compression (pruning and quantization) of SNNs. The framework is empirically evaluated, and the results show its effectiveness.

### Strengths
1. This paper introduces a carefully designed loss function and Hessian computation specifically tailored for SNNs.

2. Large-scale experiments are conducted on ImageNet, demonstrating the method’s scalability.

### Weaknesses
1. While the motivation for SBC is strongly tied to hardware and neuroscience applications, there are no real hardware deployment or energy measurement results. All claims regarding neuromorphic hardware compatibility and acceleration remain speculative in this submission. Could the authors provide wall-clock power/energy measurements or evaluations based on actual chip deployment?

2. The choice of surrogate gradient (constant function $g(u)$) is justified heuristically without comparative analysis or exploration of alternatives. The impact of this surrogate choice on empirical results is not systematically ablated. 

3. The procedure for the full SBC pruning workflow is spread between main text, algorithms, and appendices—in particular, the transitions between magnitude-based pruning (LAMPS), SBC's Hessian-based scoring, and masking are not fully transparent in the mainline narrative

### Questions
Why is the unpruned case not shown in the second row of Figure 2?

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
4

### Summary
This paper proposes Spiking Brain Compression (SBC), a multi-level compression framework for SNNs inspired by biological brain efficiency mechanisms. It integrates three biologically grounded principles: synaptic pruning, neurogenesis, and plasticity modulation, into a unified, end-to-end trainable architecture. The authors introduce a three-phase training pipeline (pruning, adaptation and consolidation) that parallels biological brain learning processes. Experiments across five datasets and architectures demonstrate great structural sparsity improvement, and reduction in energy consumption, and less than 1% accuracy loss compared to uncompressed baselines.

### Strengths
1. Comprehensive biologically inspired design: The hierarchical compression strategy (structure, dynamics, and learning) provides an elegant conceptual alignment with real neural systems.
2. Strong empirical performance achieves ultra-high sparsity with minimal accuracy loss and demonstrates superior energy efficiency and spike sparsity compared to state-of-the-art baselines on diverse datasets.
3. Ablation studies isolate contributions of each compression level, which is easy for readers to understand. Visualization of synaptic connectivity evolution highlights biologically plausible network reorganization.

### Weaknesses
1. While biologically motivated, the paper lacks formal theoretical proofs regarding convergence, optimality, or stability of the multi-level compression process.
2. The multi-level, multi-phase training pipeline is computationally expensive, involving multiple forward passes and adaptation stages.
3. There are no discussions of wall-clock training time or scalability to large models (e.g., ImageNet-scale SNNs).

### Questions
1. How does SBC perform on large-scale datasets or spatiotemporal event data beyond CIFAR and DVS-Gesture?
2. What is the computational cost (training time, memory) of the three-phase process compared to baseline SNN training?
3. Could SBC be extended to transformer-based SNNs or biologically recurrent models? As Transformers are now used more commonly in recent research.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Spiking Brain Compression (SBC), a one-shot post-training compression framework for Spiking Neural Networks (SNNs). It extends the classical Optimal Brain Surgeon (OBS) methods to spiking architectures by introducing a spike-train-based loss function grounded in the Van Rossum Distance (VRD). SBC efficiently computes a layer-wise Hessian using a surrogate membrane potential (SMP), enabling both pruning and quantization without retraining.

### Strengths
Originality
1. Extends second-order compression theory OBS into the spiking domain
2. Introduces a VRD-based spike-train similarity loss and surrogate Hessian (SMP), aligning compression with temporal spike dynamics
3. Demonstrates scalable one-shot compression for large SNNs such as Spiking ResNets and Spiking Transformers.

Quality
Theoretical derivations are mathematically rigorous and well-grounded in prior compression literature.
Experimental results are thorough and convincing: multiple datasets, architectures, and comparisons against both post-training and pruning-aware training (PAT) methods.
SBC consistently outperforms existing one-shot methods and matches or surpasses PAT approaches after light fine-tuning.

Clarity
The paper is well-organized, with clear modular breakdowns (theory, algorithm, experiments). Figures and tables effectively support key findings.

### Weaknesses
1. Limited novelty. SBC extends second-order post-training compression (OBS/OBC) to the spiking domain by introducing a spike-train–based loss (VRD) and an efficient surrogate Hessian (SMP), enabling accurate one-shot pruning and quantization of large SNNs without retraining. However, it’s important to note that this novelty is incremental rather than fundamental it’s an adaptation of existing ANN second-order frameworks to SNNs, not a new theoretical paradigm. The main creative step is the integration of temporal spike-train similarity (VRD) into the compression process, which is conceptually elegant but mathematically simplified.

2. Lack of energy and latency evaluation. While SBC claims neuromorphic efficiency, no direct energy or inference-time benchmarks are provided on hardware (e.g., Loihi, TrueNorth). Empirical measurements would strengthen claims of real-world deployability.

### Questions
1. Energy Efficiency Validation. The method is motivated by hardware constraints, do the authors plan to report actual energy, latency, or memory footprint on Loihi 2 or similar neuromorphic platforms?
2. How does SBC’s performance scale with different calibration dataset sizes or distributions? Would domain-shifted calibration samples (e.g., synthetic or augmented data) degrade results?
3. Scalability Beyond Vision Tasks. Could SBC extend to event-based NLP or sequential SNNs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Spiking Brain Compression (SBC), a novel and efficient one-shot, post-training compression framework for SNNs. By extending second-order methods with a well-motivated, spike-train-based objective function, the work demonstrates SOTA results for one-shot SNN pruning and quantization.

### Strengths
1.The paper introduces a highly innovative approach by adapting a second-order compression framework to SNNs. The use of a spike-train-based objective function derived from the Van Rossum Distance, coupled with the efficient Surrogate Membrane Potential (SMP) Hessian approximation, provides a theoretically sound and effective solution for this challenging problem.

2.The method demonstrates SOTA results for one-shot, post-training SNN compression. By achieving high accuracy while drastically reducing compression time by orders of magnitude compared to iterative methods, the work offers a highly practical and efficient solution for deploying large-scale SNNs on resource-constrained hardware.

### Weaknesses
1. Mismatch between claims and experimental validation on Spiking Transformers
The paper repeatedly claims its applicability to complex SNNs, including Spiking Transformers (e.g., line 053, line 089, and a detailed derivation in Appendix B.2). However, the experimental section provides no results on any Spiking Transformer architecture; validation is limited to CNN and ResNet-based models.

2. The motivation for SNN quantization is not unclear
The introduction (lines 062) states that PTQ for SNNs is underdeveloped. This motivation could be stronger, for example, by explicitly linking PTQ to the property of SNNs.

3. Citation formatting
The in-text citation format is incorrect throughout the manuscript.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
