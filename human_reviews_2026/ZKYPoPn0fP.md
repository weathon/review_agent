# Differentiable JPEG-based Input Perturbation for Knowledge Distillation Amplification via Conditional Mutual Information Maximization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Maximizing conditional mutual information (CMI) has recently been shown to enhance the effectiveness of teacher networks in knowledge distillation (KD). Prior work achieves this by fine-tuning a pretrained teacher to maximize a proxy of its CMI. However, fine-tuning large-scale teachers is often impractical, and proxy-based optimization introduces inaccuracies.
 To overcome these limitations, we propose Differentiable JPEG-based Input Perturbation (DJIP), a plug-and-play framework that improves teacher–student knowledge transfer without modifying the teacher. DJIP employs a trainable differentiable JPEG layer inserted before the teacher to perturb teacher inputs in a way that directly increases CMI. We further introduce a novel alternating optimization algorithm to efficiently learn the coding parameters of the JPEG layer to maximize the perturbed CMI. Extensive experiments on CIFAR-100 and ImageNet, across diverse distillers and architectures, demonstrate that DJIP consistently improves student accuracy-achieving up to 4.11% gains-while remaining computationally lightweight and fully compatible with standard KD pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Differentiable JPEG-based Input Perturbation (DJIP), a plug-and-play framework that boosts knowledge distillation by maximizing a frozen teacher’s conditional mutual information (CMI) without altering its weights. DJIP inserts a trainable, differentiable JPEG module before the teacher to learn input-space perturbations, and pairs it with an alternating optimization routine that dynamically updates class centroids to maximize perturbed CMI. On CIFAR-100 and ImageNet—spanning CNNs and ViTs and multiple distillers (KD, DKD, AT)—DJIP delivers consistent student accuracy improvements (up to 4.11%), while composing cleanly with existing methods.

Contributions
1. A lightweight, broadly compatible input-perturbation mechanism via differentiable JPEG compression.
2. An alternating CMI-maximization algorithm that overcomes fixed-centroid limitations (e.g., MCMI).
3. Extensive empirical validation demonstrating orthogonality to prevailing techniques and scalability across datasets, architectures, and distillers.

### Strengths
1. Originality. DJIP reframes JPEG-based input perturbation as a continuous optimization problem, in contrast to prior discrete toggling or adversarial noise schemes. The proposed alternating routine offers a principled advance for CMI maximization/estimation, jointly adapting perturbations and class centroids to better capture teacher–input dependence.
2. Quality. The evaluation is broad and careful: two large-scale datasets, 10+ architectures (CNNs and ViTs), and 15+ KD methods, with component-wise ablations (Table 5) that isolate the contribution of the differentiable JPEG layer and the centroid updates. A thorough hyperparameter sweep (Appendix A.5) tests sensitivity and demonstrates stable gains across settings, strengthening the empirical claims.
3. Clarity. Exposition is accessible: the CMI objective is introduced with clear intuition and notation (Sec. 3.2), and the overall pipeline is modularized in Fig. 1, making it straightforward to slot DJIP before a frozen teacher. Algorithm boxes and training pseudocode further improve reproducibility.
4. Significance. Because DJIP is plug-and-play and lightweight (≈5–12% parameter overhead), it is practical for resource-constrained deployments. Notably, it narrows or even surpasses larger baselines (e.g., a 3B DJIP student outperforming a 14B non-DJIP counterpart), and its benefits persist across diverse backbones and distillers—evidence of orthogonality rather than method-specific tuning. Collectively, these attributes position DJIP as a scalable, implementation-friendly upgrade path for KD pipelines.

### Weaknesses
1. Computational Efficiency. Table 4 reports parameter overhead, but end-to-end latency, peak memory, and throughput under realistic batch sizes/hardware are not measured. These metrics are essential for edge/smartphone deployment. Suggest reporting wall-clock inference (ms/img), GPU/CPU memory footprints, and A/B latency deltas with and without DJIP across batch sizes.
2. Baseline Diversity. Comparisons emphasize CKD and MCMI, but omit recent non-JPEG perturbation approaches (e.g., GAN/ diffusion-generated counterfactuals, learned augmentors). Including such baselines would clarify whether DJIP’s gains stem from JPEG structure or from perturbative training per se.
3. Theoretical Limits. Convergence evidence (Fig. 5) is empirical; there is no formal rate or stationarity guarantee for the alternating updates. A brief analysis (e.g., monotonic ascent under bounded curvature, or conditions ensuring convergence to a critical point) would solidify the algorithmic contribution.
4. Data Efficiency. Experiments rely on high-quality data (e.g., PixMo-Cap); low-data regimes (few-shot/long-tail KD) remain untested. Evaluating DJIP under subsampling, noisy labels, or class imbalance—and reporting data-efficiency curves—would demonstrate robustness where KD is most needed.

### Questions
1. Computational Overhead & Scaling. How does DJIP’s end-to-end cost scale with input resolution and batch size relative to vanilla KD? Please report latency (ms/img), throughput (img/s), and peak memory across GPU/CPU settings. Have you explored hardware-aware optimizations of the JPEG layer (e.g., fused CUDA kernels, SIMD/Neon intrinsics, TensorRT plugins), and what is the measured speedup?
2. Format Generalization. Beyond JPEG, have you evaluated differentiable alternatives (e.g., WebP/AVIF or learned compression) to test the mechanism’s generality? A head-to-head comparison controlling bitrate/PSNR would clarify whether gains stem from JPEG’s structure or from compression-induced perturbations more broadly.
3. Algorithmic Stability. Does the alternating scheme exhibit instability when class centroids shift quickly (e.g., early training, long-tail classes)? Would momentum/EMA, trust-region or proximal updates, or line-search on the CMI objective improve stability and convergence consistency? Any failure cases or sensitivity analyses?
4. Black-Box Teachers. How does DJIP perform when the teacher is accessible only via queries (API/closed-source), precluding gradient flow? Can zeroth-order/finite-difference estimates, score-matching proxies, or pseudo-label distillation approximate the CMI objective, and with what accuracy–cost trade-offs?

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
The paper introduces Differentiable JPEG-based Input Perturbation (DJIP), a novel framework designed to enhance teacher-student knowledge transfer in knowledge distillation (KD) without modifying the teacher model. 
DJIP uses a differentiable JPEG layer that perturbs the teacher’s inputs to directly increase conditional mutual information (CMI), which improves distillation effectiveness. 
The paper also proposes a new alternating optimization algorithm to efficiently learn the JPEG layer's coding parameters to maximize the perturbed CMI. 
Extensive experiments on CIFAR-100 and ImageNet demonstrate that DJIP consistently boosts student accuracy by up to 4.11% while being computationally efficient and compatible with standard KD pipelines.

### Strengths
1. The paper proposes a novel technique, Differentiable JPEG-based Input Perturbation (DJIP), which is a plug-and-play framework for improving knowledge distillation (KD) without requiring modifications to the teacher model. 

2. The method is extensively evaluated on CIFAR-100 and ImageNet datasets, demonstrating consistent improvements in student accuracy, with gains up to 4.11%. 

3. DJIP is computationally lightweight and integrates seamlessly with standard KD pipelines, making it an efficient solution for enhancing knowledge transfer. The method optimizes just the JPEG layer without modifying the teacher model, ensuring low overhead.

4. The proposed method works well across both same-architecture and cross-architecture (CNN-to-ViT) settings, which shows that DJIP can be applied broadly in knowledge distillation tasks, including distilling between heterogeneous model types.

5. DJIP is orthogonal to other state-of-the-art methods like MCMI, as demonstrated by the paper's results. This suggests that DJIP can be integrated with existing KD techniques to further enhance performance, providing flexibility in distillation pipelines.

### Weaknesses
1. While the method is effective in practice, the paper provides limited theoretical analysis of why perturbing the input via the JPEG layer improves distillation beyond just the CMI maximization objective. 

2. The method heavily relies on a differentiable JPEG layer, which could limit its applicability to certain use cases or architectures where JPEG compression may not be optimal or desirable. 

3. The alternating optimization algorithm introduced for learning the JPEG coding parameters adds a layer of complexity. While efficient, the algorithm may require fine-tuning, and its performance could vary depending on the choice of hyperparameters, such as lambda.

4. The usage of JPEG limits the method within the image domain. The method’s effectiveness might degrade on different types of data, especially those that are not image-based or have different structures.

### Questions
1. How does DJIP compare to other input perturbation methods in terms of generalization to different types of data, such as non-image datasets or tasks beyond image classification? Are there any potential limitations or challenges when applying DJIP to these domains?

2. The proposed alternating optimization algorithm is central to maximizing the perturbed CMI. Could you provide more details on how this algorithm scales with larger models or more complex datasets, and whether there are any concerns regarding its stability or convergence in these cases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Differentiable JPEG-based Input Perturbation (DJIP), a plug-and-play framework for improving knowledge distillation (KD) without modifying the teacher model.
DJIP introduces a differentiable JPEG layer before the teacher network to perturb inputs in a way that maximizes conditional mutual information (CMI) between inputs and outputs.
The authors also design an alternating optimization algorithm that jointly optimizes cross-entropy and CMI losses, enabling efficient learning of JPEG quantization parameters.
Extensive experiments on CIFAR-100 and ImageNet show consistent student accuracy improvements (up to 4.11%) across CNNs and ViTs, and compatibility with many KD baselines.

### Strengths
1.	Solid theoretical foundation: The CMI-based alternating optimization is mathematically consistent and improves upon fixed-centroid MCMI.
2.	Comprehensive empirical validation: Tested on multiple datasets and KD frameworks, with consistent improvements.
3.	Orthogonality: Demonstrated compatibility with both MCMI and CKD, suggesting general utility according to the paper.
4.	Reproducibility: Implementation details and appendices are complete and transparent.

### Weaknesses
1.	Limited theoretical novelty: The alternating CMI formulation is an incremental improvement over prior MCMI, and the key novelty lies in engineering implementation (JPEG layer).
2.	Insufficient qualitative analysis: The paper lacks visualization or interpretation of how and why JPEG perturbations affect teacher responses.
3.	Limited performance improvements: As an additional, plug-and-play module, DJIP offers limited performance enhancements to networks, which focuses more on engineering optimizations than theoretical innovations.

### Questions
1.	Could you provide a runtime and GPU memory usage of training DJIP layer?
2.	How sensitive is the method to the hyperparameter λ? Are there any cases where maximizing CMI harms student performance?
3.	Did the authors observe any instability during training due to learning rate choice? How robust is DJIP to different optimizer configurations (e.g., step size, momentum, or learning)
4.	Have you considered using other differentiable codecs (e.g., differentiable WebP or learned compression networks) as a generalization of DJIP?

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
4

### Summary
This paper proposes DJIP, a plug-and-play framework to enhance KD without modifying teacher model weights. Experiments show that this method has certain effectiveness. Experiments show that this method has certain effectiveness.

### Strengths
1. This paper improves the effectiveness of knowledge distillation without modifying the teacher model’s weights. The work is both interesting and effective.
2. The paper is well written, and the proposed method is plug-and-play and easy to follow.
3. The paper provides a detailed and thorough theoretical analysis.

### Weaknesses
1. The application scope demonstrated in this paper is somewhat limited. Could the proposed method be extended to LLMs (e.g., LLaMA) or generative models (e.g., Stable Diffusion)? Discussing applications beyond classification tasks would help enhance the paper’s breadth and contribution.
2. The JPEG-layer perturbation operates in the pixel space. It is unclear whether such pixel-level perturbations have a significant impact on the feature space. The rationality and reliability of the supervision signal constructed in this manner are therefore questionable.
3. The paper lacks an explanation regarding the selection strategy for key parameters, such as the frequency of alternating updates.
4. The paper does not sufficiently discuss several recent SOTA works[1,2,3,4,5,6] on knowledge distillation.

[1] f-Divergence Minimization for Sequence-Level Knowledge Distillation. ACL 2023.

[2] DistiLLM: Towards Streamlined Distillation for Large Language Models. ICML 2024.

[3] MiniLLM: Knowledge Distillation of Large Language Models. ICLR 2024.

[4] Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models. COLING 2025.

[5] ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via alpha-beta-Divergence. ICML 2025.

[6] DA-KD: Difficulty-Aware Knowledge Distillation for Efficient Large Language Models. ICML 2025.

### Questions
In addition to the issues mentioned in the Weaknesses, I have a few more concerns:

1. The paper claims orthogonality between DJIP and MCMI since DJIP explicitly optimizes the input space. Could the authors provide quantitative evidence (e.g., gradient cosine similarity between DJIP and MCMI objectives) to support this orthogonality claim?
2. Does increasing the number of quantization parameters consistently improve performance, or is there a saturation point?

### Soundness
3

### Presentation
3

### Contribution
3
