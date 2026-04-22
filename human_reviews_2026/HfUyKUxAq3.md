# LANCE: Low Rank Activation Compression for Efficient On-Device Continual Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
On-device learning is essential for personalization, privacy, and long-term adaptation in resource-constrained environments. 
Achieving this requires efficient learning, both fine-tuning existing models and continually acquiring new tasks without catastrophic forgetting. 
Yet both settings are constrained by the high memory cost of storing activations during backpropagation. 
Existing activation compression methods reduce this cost but rely on repeated low-rank decompositions, introducing computational overhead, and have not been explored for continual learning.
We propose LANCE (Low-rank Activation Compression), a framework that performs a one-shot higher-order SVD to obtain a reusable low-rank subspace for activation projection. 
This eliminates repeated decompositions, reducing both memory and computation. 
Moreover, fixed low-rank subspaces further enable on-device continual learning by allocating tasks to orthogonal subspaces without storing large task-specific matrices.
Experiments show that LANCE reduces activation storage by up to 250$\times$ while maintaining accuracy comparable to full backpropagation on CIFAR-10/100, Oxford-IIIT Pets, Flowers102, and CUB-200. 
On continual learning benchmarks (Split CIFAR-100, Split MiniImageNet, 5-Datasets), it achieves performance competitive with orthogonal gradient projection methods at a fraction of the memory cost.
These results position LANCE as a practical and scalable solution for efficient fine-tuning and continual learning on edge devices.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This manuscript investigated the problem of on-device continual learning, which a sequence of tasks need to be learned with constrained resources. Unlike the existing activation compression methods that need to repeat the low-rank decompositions, the authors proposed Low-rank Activation Compression (LANCE), a framework that adopted one-time higher-order Singular Value Decomposition (SVD) to obtain a reusable low-rank subspace for activation projection. Then, these fixed low-rank subspaces were leveraged to update the task parameters along the directions orthogonal to the existing ones. Experiments on serval commonly used datasets were conducted to demonstrate the effectiveness and efficiency of the proposed LANCE framework.

### Strengths
1. This manuscript considers the activation compression for on-device continual learning, which is an interesting yet underexplored topic.
2. Based on the claims made by the authors, the proposed method can significantly reduce the usage of memory and FLOPS in continual learning.

### Weaknesses
1. It seems that Phase 3 is just a simple application of gradient projection memory that has been well explored in the tech-stream, like GPM, weakening the overall novelty. 
2. The use of Higher-Order Singular Value Decomposition (HOSVD) seems like an unflattened version of GPM. Theoretically, the computational overhead should be higher than GPM. However, the results in Table 2 showed an extremely light memory usage compared to GPM, which is confusing. I wonder if the Mem metric used in this manuscript is a reasonable metric.

### Questions
1. HOSVD seems like an unflattened version of SVD in GPM. Based on Section 2.1, it seems that we need to store unfolded copy $X_{i}$ along different dimensions, which may need more memory compared to the flattened version of SVD in GPM. However, in Table 2, it seems that LANCE only needs an extremely small amount of memory compared to GPM. The authors adopted "memory usage measured as peak activation storage" instead of the total memory required during training. I wonder if this is a reasonable metric.
2. Could the authors please estimate and compare the FLOPS of different methods in Table 2?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work addresses the challenge of fine-tuning and continual learning under memory constraints, particularly the high activation-storage cost of backpropagation. While existing activation-compression methods reduce memory, they require repeated low-rank decompositions, adding substantial computational overhead, and remain unexplored in continual-learning scenarios. The authors proposed LANCE which performs a one-time higher-order SVD to derive reusable low-rank subspaces for activation projection, eliminating repeated decompositions and reducing both memory and computation. These fixed subspaces also support on-device continual learning by assigning tasks to orthogonal subspaces. Experiments on standard datasets such as CIFAR-10/100, Pets, Flowers102, and CUB-200 show up to 250× reduction in activation storage with accuracy comparable to full backpropagation, and competitive continual-learning performance on Split CIFAR-100, Split MiniImageNet, and 5-Datasets. Overall, LANCE provides a scalable, memory-efficient approach to fine-tuning and continual learning on edge devices.

### Strengths
A new one-shot activation compression method based on low-rank subspaces that can enable efficient fine-tuning and continual learning. 

The proposed approach is novel with a good impact for sustainable AI, especially on the edge devices. 

The integration of theory in the form of convergence analysis makes this a good contribution. 

The writing is clear, and I did not find many issues with it.  However, the results section is sloppy and can be improved. 

The main impact in this paper emerges from reducing the number of computations, which is also reflected in the results. The accuracy of LANCE is very similar to existing SOTA methods, and memory consumption is also very similar to HOSVD in Table 1. However, the FLOPS  are reduced compared to HOSVD, similar to ASI. I see LANCE bringing combined benefits from HOSVD and ASI.  For the CL scenario, though, the accuracy of LANCE is quite inferior compared to CODE-CL. However, it is more efficient than all existing projection-based CL.

### Weaknesses
When looking at the algorithm, the procedure to obtain low-rank activation on the device is done for each batch. The batch size used in the paper seems to be 64, as the authors followed a previous paper from Apolinario. So, I conclude that the results shown in the paper are with batch size 64, and the computational cost can be amortized for big batches. For some of the edge devices, 64 is a big number, as they won't have enough memory to process big batch sizes.  It is unclear how LANCE will perform with a batch size of 2, 4, or 8, which are more realistic on tiny edge devices.  Will computational savings be high? I would like to ask the authors to specify what type of edge devices they are talking about in the paper - a more Raspberry PI ir MCU class devices. 

In Section 3.1, the authors mention that LANCE gradients remain within ∼70 degrees of the full gradients. Why is it a useful result? Why 70%? Why not 90%? This clarification is missing from the paper. 

The paper has not evaluated some of the other baselines, such as Rethinking Gradient Projection Continual Learning: Stability‑Plasticity Feature Space (CVPR 2023), Efficient Continual Learning with Local Model Space Projection (LMSP) (2025).

### Questions
What seed value was used to produce the results in Table 1? What happens if you evaluate with three or five different random seeds? 

What did you change to produce five trials for obtaining the results in Table 2? 

It is unclear what BWT means and how it is measured. Table 2 dumps some numbers without any explanation of these numbers as to what they signify and how the BWT obtained using LANCE compares against other methods.

### Soundness
3

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
4

### Summary
This paper introduces LANCE (Low-rank Activation Compression), a framework designed to address the high activation memory cost that hinders on-device fine-tuning and continual learning (CL). The core contribution is a "one-shot" approach where a single Higher-Order SVD (HOSVD) is performed offline to find a fixed, reusable low-rank subspace for activations. During on-device training, activations are simply projected onto this subspace, eliminating the computational overhead of repeated decompositions common in prior work. The framework is extended to continual learning by enforcing orthogonality between the subspaces of sequential tasks to prevent catastrophic forgetting. The authors claim their method achieves up to 250x activation memory savings with minimal accuracy loss, making it a practical solution for resource-constrained edge devices.

### Strengths
Novelty and Simplicity: The core idea of using a one-time, fixed subspace is an elegant and novel simplification of prior low-rank activation compression methods that rely on iterative updates.

Innovative Continual Learning Mechanism: The extension to continual learning via orthogonal subspaces is clever and highly memory-efficient. By handling the orthogonality constraint during the offline calibration phase, it avoids storing large projection matrices in active memory during on-device training, offering a distinct advantage over methods like GPM.  

Rigorous Validation: The claims are well-supported by a convergence analysis and comprehensive experiments across a variety of models and standard fine-tuning and continual learning benchmarks.

### Weaknesses
Lack of Limitation Discussion: While the paper provides a solid convergence analysis, it lacks a dedicated "Limitations" section. Such a section would be the appropriate place to proactively discuss the practical boundaries of the method. This includes potential failure modes of its core assumptions (e.g., subspace stability under significant domain shift) and conceptual trade-offs, such as the preclusion of positive forward transfer. 

Unverified Assumption of Subspace Stability: The framework's success relies on the critical assumption that an activation subspace calibrated on a small data sample remains effective under a potential domain shift between the calibration and target data. This assumption is not tested, representing a major potential failure point for real-world deployment.

Lack of Nuance and Clarity: The positioning against highly related work like ASI (which uses efficient updates rather than full re-computation) could be more nuanced.

### Questions
Robustness to Domain Shift: How robust is the performance of the fixed subspace if there is a significant domain shift between the calibration data and the target fine-tuning data?

Positive Forward Transfer: Does the strict orthogonality constraint in the CL setting harm performance on sequences of related tasks by preventing positive knowledge transfer? Could the framework be adapted to allow for this?

Memory Advantage over GPM: To clarify, is the primary memory advantage over GPM in the CL setting that the memory matrix $M^t$ is not required in SRAM during the on-device training loop for a given task?

Clarification on ASI Baseline: For the comparison with ASI, were the experiments re-run under your exact experimental setup? If the results were taken from the original ASI paper, how can you ensure a fair comparison given potential differences in implementation and experimental conditions? Furthermore, what are the key trade-offs between LANCE's fixed subspace and ASI's adaptive approach, especially in non-stationary environments?

Clarification of Table 1: Could you please clarify the "TFLOPS" column in Table 1? The value for standard backpropagation is listed as zero, which seems incorrect and contradicts the complexity analysis in Appendix C.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a one-shot activation compression framework, LANCE, for efficient on-device fine-tuning and continual learning. It uses a single higher-order SVD to find reusable low-rank subspaces, eliminating repeated decompositions during training.

### Strengths
1. Theoretical analysis is reasonable.
2. The paper is clearly written and well-organized, with consistent notation, intuitive figures, and pseudocode/algorithm.
3. LANCE has potential impact for efficient on-device continual learning, particularly for IoT applications where memory is limited.

### Weaknesses
1. The evaluation lacks real on-device results. Important metrics like memory savings are not demonstrated on real hardware, which weakens the practical impact claim.
2. Evaluation on large datasets like ImageNet is lacking (only Split MiniImageNet is evaluated).
3. More ablation study should be provided, like how different compression ratios affect gradient fidelity.

### Questions
1. If the calibration batches do not well represent future data distributions, how robust of the proposed method?
2. Can the authors implement LANCE on actual edge hardware to validate the claimed efficiency?

### Soundness
2

### Presentation
3

### Contribution
2
