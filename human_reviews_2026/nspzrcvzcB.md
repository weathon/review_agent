# Entropy-Monitored Kernelized Token Distillation for Audio-Visual Compression

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
We propose a method for audio-visual knowledge distillation. Existing methods typically distill a student model from the latent embeddings or outputs of a teacher. The former requires matching feature dimensions, if not the same architecture, between teacher and student models while the latter supports any teacher-student pairing, but tends to be less performant. Unlike them, we do not explicitly distill from latent embeddings or outputs, but the pairwise relationships between embeddings across samples for each modality; this is realized as a kernel, which is the crux of our method, "Kernelized Token Distillation (KTD)". Specifically, we tokenize and embed the input for a given modality, and compute the Gram matrix across tokens, from which we distill. As audio and visual modalities afford different information for a task, we adaptively modulate distillation by measuring the entropy of each modality, leading to an Entropy-Monitored Kernelized Token Distillation (EM-KTD) scheme. Our method allows for flexibility in complexity of kernel function to model relationships across tokens, which are selectively distilled to ensure high-fidelity supervision for the student. We evaluate EM-KTD on VGGSound and AVS-Bench, where we use 94% fewer parameters than the teacher while preserving 96.9% in performance for audio-visual event recognition and 96.5% on audio-visual segmentation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
For network-agnostic audio-visual knowledge distillation, this paper proposes the Entropy-Monitored Kernelized Token Distillation (EM-KTD). Specifically, the Kernelized Token Distillation (KTD) distills pairwise relationships between latent tokens, captured in a Gram matrix. Then, an Entropy-Monitored (EM) scheme is proposed to selectively distill knowledge. It dynamically weighs the distillation loss for each modality (audio, visual, fused) based on the entropy of its feature embeddings. Experiments are conducted on audio-visual event classification and segmentation tasks, demonstrating the effectiveness of the proposed method.

### Strengths
- The core idea of KTD is novel and well-motivated. Using kernelization to transfer the geometric structure of the teacher's latent space without requiring feature dimension alignment is an elegant solution to a common problem in knowledge distillation.
- The method is validated on two distinct and challenging audio-visual tasks: classification and segmentation.
- The results are impressive, showing that EM-KTD can compress a teacher model by over 90% while retaining nearly all of its performance, clearly demonstrating the practical utility of the proposed method.

### Weaknesses
- The kernelization step has a computational complexity of $O(N^2)$ with respect to the number of tokens $N$ for each instance. While the paper shows strong performance, a more explicit discussion of the training time trade-offs in the main text would be beneficial. 
- The paper states that for the Entropy Monitor, "additional task heads... are trained to minimize the cross entropy loss" on the frozen teacher model (Sec. 3.3). This is a crucial implementation detail that lacks clarity. It is unclear *when* these linear probes are trained. Are they pre-trained on the entire dataset before the student distillation process begins? Or are they trained concurrently? A more detailed explanation of this procedure is needed for reproducibility and to fully understand the method's mechanics.
- As shown in Table 2, the EM-KTD is not much superior to KTD in the S4 segmentation task.
- The Figure can be improved. For example, it is challenging for readers to understand the 'entropy-monito' mechanism in Figure 2.

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents Entropy-Monitored Kernelized Token Distillation (EM-KTD), a novel framework for compressing audio-visual models. The method features two key components: 1) Kernelized Token Distillation (KTD), which distills the pairwise relationships between latent tokens captured in a Gram matrix, making the approach architecture-agnostic and highly expressive. 2) An Entropy-Monitored (EM) scheme that adaptively weights each modality's contribution based on its predictive entropy, ensuring high-fidelity supervision. This framework establishes a new state-of-the-art on two audio-visual tasks.

### Strengths
1.The paper proposes a novel EM-KTD framework. By operating on the space of token relationships, it enables flexible and architecture-agnostic knowledge transfer.

2.The method demonstrates state-of-the-art performance on audio-visual benchmarks, achieving a 94% parameter reduction while retaining over 96% of the teacher's performance.

3.The claims are well-supported by comprehensive comparative experiments and thorough ablation studies that validate the contributions of each component.

### Weaknesses
1.The O(N²) computational complexity of the pairwise kernelization step may limit the method's scalability to tasks with more input tokens.

2.The EM scheme relies on entropy-loss-based classification tasks. This might limit its direct applicability to other tasks (e.g. regression tasks).

3.The paper's validation on heterogeneous teacher-student architectures is a key strength, but this experimental context is detailed only in the appendix. Highlighting this setup in the main text would better frame the results and underscore the method's flexibility. To provide a more comprehensive evaluation, an additional experiment in a homogeneous setting is recommended to evaluate the method's performance when architectures are matched.

Typos:

1. The formulation of the Huber loss in Equation (2) has two minor errors: it is missing an equals sign, and the condition should be ||p - q|| < 1 instead of ||p, q|| < 1.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

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
The paper proposed EM-KTD, a novel framework to compress audio-visual models with knowledge distillation. The proposed method expands the idea of MTST (i.e., distilling pairwise relationships between token embeddings) to audio-visual understanding models, and makes the following improvements:

1. Measures the similarity between latent embeddings with kernel functions rather than cosine similarity.
2. Replaces KL-divergence loss with Huber loss, eliminating the need to use masking and Softmax mapping as in [MTST.](http://MTST.In) In this way, the student model can replicate the geometry of teacher latent space more precisely.
3. Proposes Entropy Monitor to adjust the loss weight of each modality entropy, based on entropy of uni-modal classification predictions.

The authors evaluated the proposed method on two audio-visual understanding tasks: event classification and segmentation. Experiment shows a clear advantage of EM-KTD over both vanilla training and previous distillation baselines, providing an resource-efficient solution for audio-visual understanding.

### Strengths
1. The paper proposed an effective solution to the latent dimension mismatch problem in audio-visual knowledge distillation.
2. Reasons for core design choices are clearly explained evidently supported by experiments, such as why removing softmax mapping (Appendix A) and why distilling token relationships within one instance in each modality (Appendix B).
3. The proposed model is architecture-agnostic and supports various kernel functions (linear, polynomial, RBF), allowing trade-offs between computation complexity and performance.

### Weaknesses
1. Compatibility to long training samples: Calculating the Gram matrix is an $O(N^2)$ operation, which may be computationally inefficient, especially when masking is not applied as in MTST. This might affect effectiveness of the proposed method in modalities with variable input lengths, like audio.
2. While the advantage of KTD over MTST is concrete, the advantage of EM-KTD over KTD is comparably less significant.
3. Figure 2 is a bit confusing. The “fusion modality” are not depicted and mechanism of entropy-weighted loss is ambiguous.

### Questions
Lacking Multi-modal Distillation Baselines: Although EM-KTD is designed for audio-visual knowledge distillation, all baselines included are not specifically designed for multi-modal knowledge distillation. Can the proposed method be compared with multi-modal knowledge distillation methods mentioned in Related Work?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Traditional knowledge distillation methods either require alignment of the teacher-student structure or fail to effectively utilize the internal structural information of modalities, and they lack an adaptive mechanism for the varying amounts of information across different modalities. To address these issues, the paper proposes Kernelized Token Distillation (KTD): instead of distilling the features themselves, it distills the similarity structure among tokens within a single sample, achieving structure-agnostic cross-modal distillation. Additionally, the paper introduces the Entropy-Monitored mechanism: by using the classification entropy of each modality from the teacher model to dynamically adjust the distillation weights and suppress the interference of low-information modalities. The paper validates the effectiveness on the VGGSound and AVS-Bench datasets, maintaining a considerable level of teacher performance even when the student model has significantly fewer parameters.

### Strengths
The EM-KTD method uses the "structure among tokens" as the distillation target, avoiding the dimension matching problem. The motivation for the entropy supervision mechanism is reasonable, and the overall method is innovative. The clarity of the paper is acceptable, but it lacks intuitive explanations for "kernelized token" and "Gram matrix." It is suggested to optimize Figure 1 and Figure 2 to emphasize that the similarity matrix comes from multiple tokens within a single sample. The experimental quality of the paper is solid, with comprehensive comparisons to current mainstream methods and extensive ablation experiments. As a general method, if the paper could validate it on more modalities and tasks, it would further enhance its significance.

### Weaknesses
1. The paper's explanation of the entropy prediction head $g_m(⋅)$ is insufficient. Firstly, in the methods section, it does not describe which stage of the process the entropy predictor is trained in. In the experimental section, it does not mention the source of the weights for $g_m(⋅)$. In the ablation part, it seems that the impact of the structure of $g_m(⋅)$ on the distillation results is not discussed.

2. The explanation of dataset labels and information on line 192 is not clear enough. Clarifying the meanings of $n$ and $N$ in the Dataset section would make the paper more understandable.

3. Validating the method on more datasets would more fully demonstrate its generalizability.

### Questions
1. I still have some confusion about the entropy monitor $g_m(⋅)$. Firstly, how is it trained? If the task is not classification, how should the model evaluate the entropy?

2. Although the paper tries linear, polynomial, and RBF kernel functions, it does not provide a systematic selection criterion or adaptive mechanism. When used for different tasks or datasets, is it necessary to manually select the kernel function?

3. The paper mainly focuses on the visual and audio modalities. I am curious whether, as a general method, it would yield similar results on tasks involving other modalities (such as visual-text, audio-text, etc.). If this could be validated, it would well demonstrate the method's versatility.

### Soundness
3

### Presentation
2

### Contribution
2
