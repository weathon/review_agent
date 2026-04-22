# Constructing Variable-depth ViTs from Repeatable Low-bit Learngene

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 2

## Abstract
Large-scale Vision Transformers (ViTs) have achieved remarkable success across a wide range of computer vision tasks. 
However, fine-tuning and deploying them in diverse real-world scenarios remains challenging, as resource constraints demand models of different scales. The recently proposed Learngene paradigm mitigates this issue by extracting compact, transferable modules from well-trained ancestor models to initialize variable-scaled descendant models. Yet, existing Learngene methods mainly treat learngenes as initialization modules for descendant models, without addressing how to construct these models more efficiently. In this work, we rethink the Learngene methodology from the perspectives of quantization and parameter repetition. We introduce Repeatable Low-bit Learngene (RELL), which compresses ancestor knowledge into a small set of quantized, cross-layer shared modules via quantization-aware training and knowledge distillation. These repeatable low-bit modules enable flexible construction of descendant models with varying depths through parameter replication, while requiring only lightweight adapter tuning for downstream adaptation. Extensive experiments demonstrate that RELL achieves superior parameter efficiency and competitive or better performance compared with existing Learngene methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes RELL (Repeatable Low-bit Learngene): extract a small set of quantized, cross-layer shared Transformer blocks via QAT + distillation from an ancestor model, then construct variable-depth ViTs by repeating these blocks and tuning only lightweight adapters (LoRA). Experiments across multiple downstream datasets show strong parameter-efficiency and high performance vs prior Learngene methods and standard LoRA baselines.

### Strengths
1. This paper improves upon the Learngene family of methods from the perspectives of quantization and weight sharing, effectively leveraging the regularization benefits of quantized weights and the storage and deployment efficiency brought by cross-layer sharing.

2. By first compressing the ancestor model and then reconstructing a descendant through the proposed RELL framework, the authors demonstrate that the resulting models not only outperform those trained from random initialization on downstream adaptation tasks, but also achieve notable performance gains over previous Learngene variants—all while maintaining a smaller trainable parameter set and overall model size.

3. The paper includes comprehensive ablation studies on quantization precision and the number of shared/repeated blocks, which further strengthen the empirical analysis.

4. Figures and writing are clear, structured, and easy to follow, contributing to good readability.

### Weaknesses
1. In the comparison with LoRA-based methods, there is no baseline using a pure standard LoRA (i.e., a full-precision model without quantization or layer pruning). Including this would make the performance gains more convincing.

2. The current experiments are conducted only on models roughly equivalent to ViT-Base in size. It would be beneficial to evaluate the approach on larger model scales to verify its scalability.

3. It would be valuable to analyze how the performance of the compressed-then-reconstructed models compares to that of equally sized models obtained via conventional pretraining and fine-tuning. Quantifying this potential gap would help clarify the trade-offs of the proposed approach.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes RELL (Repeatable Low‑bit Learngene), a two‑stage framework to construct variable‑depth ViTs from a small set of quantized, cross‑layer shared blocks. Specifically, first, an auxiliary model is built by layer-sharing and trained with quantization‑aware training (QAT) and distillation from an ancestor model. Then, each low‑bit block in the auxiliary model is frozen and repeated to setup the task-specific model, while LoRA adapters are added and fine‑tuned. Extensive experiments show that RELL yields better accuracy with substantially fewer trainable parameters and smaller model size than prior Learngene methods, and proves better learning curves compared to from-scratch model training as well.

### Strengths
1. **Clear presentation.** The paper motivates limits of prior Learngene approaches (full‑precision expansion and full‑parameter fine‑tuning) and justifies quantization + repetition as a natural solution. The training details are clearly presented and the claims are valid reasonable.
2. **Promising efficiency compared to existing Learngene methods.** Table 1 shows that Learngene clearly outperforms existing Learngene methods in terms of both efficiency while achieving comparable or better performance. The learning curves presented in Figure 3 are promising as well.
3. **Reasonable design analysis.** This article presents persuasive design analysis in Section 4.2.3, claiming that the RELL designs are more cost-effective than simple modifications in model and PEFT structures.

### Weaknesses
1. **Lack of baselines dealing with the same problem, but from other perspectives.** This article leverages the Learngene paradigm to solve the problem that practical applications have different requirements for the deployed model. Considering the challenge addressed, this article should also thoroughly discuss and compare its method with related methods that solve the same problem from the token pruning perspective. Notable works are [1,2].
2. **Training efficiency.** Beside the inference efficiency information, the author should also report the training information for RELL and other Learngene methods. Good performance of RELL could be a result of over-training.
3. **More evaluation datasets.** Only 5 tasks are chosen for evaluating RELL. Experiments on more datasets should be added to further validate the effectiveness of RELL, such as the VTAB-1k dataset.

[1] PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation (ECCV 2024)

[2] Dynamic Tuning Towards Parameter and Inference Efficiency for ViT Adaptation (NeurIPS 2024)

### Questions
1. How to initialize auxiliary model layers when pre-training the auxiliary model?
2. Why are different model structures selected for the ancestor model and the auxiliary/final task-specific models? The authors claim that they follow previous works for this design choice. However, in my opinion, RELL differs significantly from previous Learngene methods. To show that RELL is more applicable in actual applications compared to existing methods, the authors should conduct additional experiments applying RELL to more "general" problem settings. For example, we desire compressed backbones for real-world tasks, and the same architecture (i.e., both are standard DeiTs, but with different layers/hidden dimensions/MLP hidden sizes) is preferred for both the ancestor, the auxiliary model, and task-specific models. 

See weaknesses for other questions.

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
2

### Summary
This paper proposes Repeatable Low-bit Learngene (RELL), a Learngene framework for constructing variable-depth ViTs under resource constraints. RELL compresses knowledge from a large pre-trained ancestor model into a small set of quantized, cross-layer shared modules via quantization-aware training (QAT) and knowledge distillation. These low-bit modules are then reused through parameter repetition to build descendant models of flexible depth, with only lightweight LoRA adapters fine-tuned for downstream tasks. Experiments on multiple datasets show that RELL achieves superior parameter efficiency and competitive or better accuracy than existing Learngene and LoRA baselines, especially in low-data regimes.

### Strengths
The proposed RELL framework is simple and effective: it integrates parameter repetition, 4-bit quantization, and LoRA fine-tuning to drastically reduce storage and training cost while preserving performance; the method is practical, enabling flexible depth scaling without retraining the backbone; experiments are solid, demonstrating clear advantages over baselines on CIFAR100, CUB-200, etc., particularly when training data is scarce.

### Weaknesses
1. Limited model and task scope: All experiments use DeiT-based ViTs and image classification. No evaluation on modern architectures (e.g., Swin, ConvNeXt) or dense prediction tasks (detection, segmentation), limiting claims of general applicability.

2. Inadequate comparison with recent quantization + PEFT methods: Missing comparison with QLoRA [1] or LoftQ [2], which also freeze quantized weights and tune low-rank adapters. Without such baselines, the claimed efficiency gains may be overstated.

3. Ambiguity in RELL block design: The auxiliary model shares weights every k layers, but the rationale for extracting exactly 3 RELL blocks (k=4) is unclear. Distillation uses only logit-level KL divergence (Equation 5)—feature- or attention-level distillation, common in ViT compression, is not explored, potentially limiting knowledge transfer fidelity.

4. Lack of real-device deployment analysis: QAT quantizes weights only, keeping activations in full precision (Appendix A.1). While this preserves accuracy, it may overstate real-world hardware efficiency. Actual latency or GPU memory consumption is not reported.

[1] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs. NeurIPS, 2023.

[2] Yixiao Li, Yifan Yu, Chen Liang, Nikos Karampatziakis, Pengcheng He, Weizhu Chen, and Tuo Zhao. LoftQ: Lora-fine-tuning-aware quantization for large language models. ICLR, 2024.

### Questions
1. Why are only classification tasks evaluated? Can RELL be applied to dense prediction tasks like object detection or semantic segmentation?

2. How does RELL compare to QLoRA or LoftQ under the same backbone and bit-width?

3. What is the rationale for choosing 3 RELL blocks? Table 5 shows 6 blocks yield slightly better performance—does this suggest using more blocks?

4. If activations were also quantized (not just weights), how much would performance drop? What would this mean for edge deployment?

### Soundness
3

### Presentation
3

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
In this paper, the authors provide a more efficient solution for learning models of different scales from a pretrained model. Building upon the Learngene paradigm, the authors propose Repeatable Low-bit Learngene (RELL), which compresses pretrained knowledge and eases the learning of different model scales. Experiments on multiple benchmarks show good performance.

### Strengths
1. Experiments on different benchmarks show good performance.

### Weaknesses
1. It is hard to see the motivation for this paper. The authors claimed, “Yet, existing Learngene methods mainly treat learngenes as initialization modules for descendant models, without addressing how to construct these models more efficiently.” However, these fine-tuning methods can be considered efficient. Do the authors intend to be more efficient?

2. Fig. 2 is hard to understand. For example, in Fig. 2(a), what does the top-right part mean and what does the bottom-right part mean? Also, what do the arrows with/without an “x” mark indicate?

3. As stated in line 93, “transform the knowledge of a large ancestor model into a compact yet versatile set of modules that are both quantized and repeatable.” What is the module size, and how much of the original knowledge can it keep?

4. The key problem of this paper, learning different scales of models from a pretrained mode, is very similar to Stitchable Neural Networks. However, there is no discussion or comparison with this work and its follow-ups.

[1] Pan, Zizheng, Jianfei Cai, and Bohan Zhuang. “Stitchable Neural Networks.” CVPR, 2023.

5. What do the different colors indicate in Fig. 2? The presentation and figures are misleading.

6. For evaluation, these small datasets cannot demonstrate the effectiveness of the proposed method. The authors may run experiments on ImageNet and other tasks beyond classification, such as detection on MS COCO or segmentation on ADE20K, etc.

### Questions
1. The Learngene method, as proposed in [2], is not well known in the community. If the proposed method is based on this, I would expect more detailed background in the submission to aid understanding.

[2] Wang, Qiu-Feng, et al. "Learngene: From open-world to your learning task." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 8. 2022.

2. If the goal is to cover a wide range of computational budgets, why not provide a series of variants with different model sizes? This is common practice (e.g., DiT, ConvNeXt) or learning like Stitchable Neural Networks. What are the limitations of this strategy?

3. A common issue in some papers is the introduction of meaningless new terms. Such definitions do not add novelty and make the paper harder to follow. For example, “ancestor model” is not a common term in deep learning or computer vision; it has the same meaning as “pretrained model.” There is no need to create a new term for the same concept.

### Soundness
2

### Presentation
2

### Contribution
1
