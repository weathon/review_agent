# Towards Efficient SNNs: Sensitivity-Guided Pruning for Deep Spiking Architectures

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Spiking Neural Networks (SNNs) offer compelling advantages in energy efficiency and biological plausibility but face performance and deployment challenges due to redundant structural units in suboptimal architectures. Existing compression techniques predominantly rely on unstructured connection-level pruning, which often necessitates specialized hardware for efficient execution. To overcome these limitations, we propose SPTE (Sensitivity-guided Pruning by Taylor Expansion), a structured pruning framework that leverages Taylor expansion to estimate each convolutional kernel's sensitivity to the loss function during training. This enables the iterative removal of less critical components. Extensive experiments across four benchmark datasets demonstrate the effectiveness of SPTE. Remarkably, SPTE achieves 78.09\% connectivity sparsity on CIFAR10 with a +1.49\% accuracy gain, outperforming previous state-of-the-art methods in both performance and model compactness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposes a Taylor-expansion-based structured pruning method for SNNs, and validates its performance on static and neuromorphic datasets.

### Strengths
1. In addition to the inference accuracy after pruning, this work also conducted statistical analysis on inference speed.

### Weaknesses
1. As shown in Tab.2, the performance comparison for this work is too simple:
- Firstly, some works that maintain superior performance on SNNs with extremely high sparsity have not been included. 
- Secondly, this work did not include Synaptic Operations (SOPs), which is an important indicator for evaluating SNN power consumption. 
- Thirdly, the results presented in this work are all under conditions of low sparsity (Connectivity > 50%), without verifying the performance of the method under conditions of extremely high sparsity. Some cases even experienced accuracy loss (Acc. Loss < 0%) under the conditions of Connectivity > 50%. 
- In addition, this work only conducted performance validation on convolutional architectures and did not conduct performance validation on large-scale datasets (e.g. ImageNet-1k). 
- The authors should consider using a network structure consistent with the comparative works for clear and detailed performance comparison. Meanwhile, the inference accuracies achieved in this work is clearly unsatisfactory in the current SNN community.

2. Pruning towards the synaptic layers is just one solution to reduce the power consumption of SNNs. Other techniques such as pruning for neuron layers and lightweight quantization for SNNs have also been proposed. Therefore, I tend to think the contribution of this work to the SNN community is relatively limited.

3. The layout of figures, tables and formulas in this paper still needs further polishing.

[1] Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework. ICLR 2024.

[2] QP-SNN: Quantized and Pruned Spiking Neural Networks. ICLR 2025.

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
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents STE, an innovative structured pruning framework for SNNs, which effectively tackles the challenge of network compression. Extensive experiments highlight its promising capability to significantly reduce model size and FLOPs across various benchmark datasets.

### Strengths
1. Innovation Method: Hardware-Friendly Structured Pruning Framework. By using first-order Taylor expansion for sensitivity evaluation, STE performs channel- or kernel-level pruning that aligns with the spatiotemporal characteristics of SNNs.
2. Superior Compression–Accuracy Trade-off: STE achieves remarkable model compression while maintaining or even improving accuracy across multiple benchmarks.

### Weaknesses
1. The citation style is inconsistent and does not conform to the official ICLR formatting guidelines.
2. Related Work:
This section lacks depth and persuasiveness due to an insufficient number of relevant citations (which I think is incomprehensible). A qualified study needs to be supported and corroborated by previous papers. Expanding the discussion to include more recent and influential SNN pruning and optimization works would strengthen the context of this research.
3. The claim of “ensuring compatibility and performance retention across diverse architectures” (lines 069–070) appears overstated, given that experiments are restricted to VGG- and ResNet-style networks.
4. The results in Table 2 raise concerns: under identical architectures, the proposed STE method occasionally performs worse than competing pruning techniques.

### Questions
Please refer to “Weaknesses”

### Soundness
2

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
The paper proposes ​​STE (Sensitivity-guided pruning by Taylor Expansion)​​, a structured pruning framework for Spiking Neural Networks (SNNs). STE uses Taylor expansion to estimate the sensitivity of each convolutional kernel to the loss function, guiding the pruning process to preserve important structures for efficient execution on general-purpose hardware.

### Strengths
The focus on ​​structured pruning​​ is a benefit, as it produces hardware-friendly models that can run efficiently on standard GPUs and CPUs without the need for specialized sparse acceleration hardware.

The method appears to be straightforward and simple, building on the established concept of Taylor expansion for importance estimation. This makes it potentially easy to understand and implement.

By estimating sensitivity during the training process, the method can dynamically adapt the pruning strategy, potentially leading to better preservation of accuracy compared to one-shot pruning methods.

### Weaknesses
A major concern is that the core framework of the proposed method may not be sufficiently novel, potentially being an incremental application of existing ideas to ANNs. The use of channel/kernel sensitivity evaluation via Taylor expansion is a known technique in the ANN literature, and the paper may not demonstrate enough adaptation or innovation to make it compelling for the SNN domain.

Table 2 compares the method against too few existing state-of-the-art benchmarks, making it difficult to assess its true competitive standing.

The citation format is problematic (e.g., "Recent studies Lietal. (2024a;b)"). Please use \citep appropriately. 
Sections 2.1 and 2.2 lack citations entirely, and 2.3 has too few, weakening the literature review.

The use of "So" at the beginning of a sentence (Line 191) is inappropriate for academic writing.

The acronym "STE" is already widely used in the SNN field for "Straight-Through Estimator," which is the standard method for training SNNs. This creates immediate and significant confusion.

### Questions
What is the specific novel contribution of this work that differentiates it from simply applying existing ANN structured pruning techniques to SNNs?

The method is demonstrated on convolutional architectures. Can the proposed framework be effectively applied to more modern, Transformer-like SNN architectures?

The paper reports inference speed on a GPU. However, a key motivation for using SNNs is their efficiency on neuromorphic hardware. What are the expected benefits or performance of the pruned models on neuromorphic processors, and why was this not evaluated?

Given the critical name conflict with "Straight-Through Estimator (STE)," would the authors consider changing the name of their method to avoid confusion and improve the identity of their work?

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
5

### Summary
This paper proposes a novel structured pruning framework for SNNs named STE (Sensitivity-guided pruning by Taylor Expansion). The method leverages first-order Taylor expansion to estimate the sensitivity of convolutional kernels to the loss function, enabling the iterative removal of less critical components in a structured manner. The pruned models show significant reductions in FLOPs and improvements in inference speed.

### Strengths
1. The paper successfully adapts a principled, Taylor-expansion-based sensitivity analysis from ANNs to the SNN domain, addressing a significant gap in structured pruning for spiking architectures. The approach is well-motivated by the limitations of existing unstructured pruning methods.

2. The paper provides valuable insights beyond mere accuracy and sparsity numbers. The analysis of sensitivity distribution across layers (Figure 3) offers an intuitive explanation for why deeper layers can be pruned more aggressively, strengthening the methodological foundation.

### Weaknesses
1. While the paper compares favorably against other SNN pruning methods, it would be strengthened by a brief discussion or comparison with state-of-the-art structured pruning techniques applied to ANNs on the same tasks. This would help contextualize STE's performance within the broader model compression field.

2. The description of the iterative pruning process could be more detailed. Specifically, the total number of pruning iterations, the schedule for fine-tuning (e.g., number of epochs per iteration), and the associated computational cost are not explicitly stated.

### Questions
1. The method calculates sensitivity over the entire temporal dimension (T). How does the performance of STE change with a different number of timesteps? Is the sensitivity metric consistently reliable across varying temporal sequence lengths?

2. Can this approach be extended to other mainstream neural network frameworks like spiking transformers? 

3. Can this approach be extended to more large-scale datasets/tasks?

4. Can this approach be extended to more non-vision tasks?

### Soundness
3

### Presentation
3

### Contribution
2
