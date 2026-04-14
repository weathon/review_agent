# Knowledge Distillation with Multi-granularity Mixture of Priors for Image Super-Resolution

- Decision: Accept (Spotlight)
- Scores: 8, 8, 6

## Abstract
Knowledge distillation (KD) is a promising yet challenging model compression approach that transmits rich learning representations from robust but resource-demanding teacher models to efficient student models. Previous methods for image super-resolution (SR) are often tailored to specific teacher-student architectures, limiting their potential for improvement and hindering broader applications. This work presents a novel KD framework for SR models, the multi-granularity Mixture of Priors Knowledge Distillation (MiPKD), which can be universally applied to a wide range of architectures at both feature and block levels. The teacher’s knowledge is effectively integrated with the student's feature via the Feature Prior Mixer, and the reconstructed feature propagates dynamically in the training phase with the Block Prior Mixer. Extensive experiments illustrate the significance of the proposed MiPKD technique.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes a new knowledge distillation method which considers multi-granularity mixturw of priors for image super resolution. Motivated by the success of masked autor encoder in reconstructing missing pixels from masked input, this work explores this machinism into feature prior mixer and block prior mixer to distill teacher's feature. Experiments are done on various datasets to show it effectiveness but the improvement is not significant,

### Strengths
1. The proposed method is well motivated and grounded.  This work tackles the challenge of aligning representations between models of different sizes, which is important for improving the performance of lightweight models.
2. Experiments are done on various datasets (Set5, Set14, BSD100, and Urban100) to show its superiority over competitors in super-resolution task.
3. The authors perform ablation studies to evaluate the contribution of each component (feature prior mixer, block prior mixer, encoder type) and discuss different loss weighting strategies.

### Weaknesses
1. This method lacks comparison with SOTA feature-based knowledge distillation methods, such as, [1][2][3][4] which all contributed to projecting features before distillation to improve student's performance albeit not specifically in the super-resolution domain.
[1]ViTKD: Feature-based Knowledge Distillation for Vision Transformers
[2]Improved feature distillation via projector ensemble
[3]Knowledge distillation via softmax regression representation learning
[4]Masked Autoencoders Enable Efficient Knowledge Distillers
2. As a closely related work [4] also proposes masked auto encoder for feature distillation.  It would be beneficial to discuss this work in both the introduction and experimental sections, highlighting the differences and demonstrating where the proposed method offers improvements.
Besides, [5] has included knowledge distillation used for super resolution.  Method such as RDEN in [5] should be included in related work.
[5]The Ninth NTIRE2024 Efficient Super-Resolution Challenge Report
3. Three key factors influence Knowledge distillation performance: knowledge, position, loss. Current ablations studies have discusses knowledge and loss but lacks discussion of where to put.  Understanding the optimal positions for applying the feature prior mixer and block prior mixer would provide valuable insights into the flexibility and effectiveness of MiPKD.

### Questions
The main concern with the proposed method is the lack of comparison with closely related feature-based knowledge distillation approaches, such as ViTKD [1], Improved Feature Distillation via Projector Ensemble [2], and Knowledge Distillation via Softmax Regression Representation Learning [3]. These methods, while not specifically targeting super-resolution, focus on feature projection before distillation, making them relevant benchmarks. Including these comparisons would provide a clearer understanding of the proposed method’s strengths and positioning relative to state-of-the-art techniques. Additionally, a deeper discussion on the differences between the proposed method and Masked Autoencoders [4] would clarify its unique contributions.   [5] serves as a benchmark for super-resolution where knowledge distillation is applied to enhance performance. Including a discussion of this benchmark and similar methods would help clarify how the proposed method contributes to this area.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors propose MiPKD, a multi-granularity mixture of prior knowledge distillation framework designed for image super-resolution tasks. MiPKD facilitates the transfer of “dark knowledge” from teacher models to student models across diverse network architectures. The framework employs feature and block prior mixers to reduce the capacity disparity between teacher and student models for effective knowledge alignment and transfer. Extensive experiments are conducted on three SR models and four datasets, demonstrating that MiPKD significantly surpasses existing KD methods.

### Strengths
1. The integration of feature fusion within a unified latent space  and stochastic network block fusion are innovative for SR model knowledge distillation.
2. The paper provides clear explanations and discussion, supported by well-organized charts and ablation studies, which effectively highlight the framework’s contributions and innovations.
3. The paper is well organized and the motivation of MiPKD is clear

### Weaknesses
1. The writing is sometimes unclear, with inconsistent notations and undefined terms:
	- The feature maps $F$ in Fig. 2, Eq. 2, and Eq. 3 are not consistently bolded.
	- The three feature maps in Eq. 3 are inconsistently formatted.
	- The loss weights $\lambda_1$ and $\lambda_2$, mentioned in lines 358–359, are not defined or referenced elsewhere in the paper.
2. While the paper provides comparisons for EDSR and RCAN, the evaluation on SwinIR is not as comprehensive.

### Questions
1. What is the rationale for randomly sampling the forward propagation path in the block prior mixer? If the output of Feature Prior Mixer is passed to both teacher and student models and compute two losses, will the knowledge distillation be more effective?
2. What are the specifications of the encoder and decoder networks (e.g., number of layers, hidden dimensions)? Would a larger auto-encoder result in a better student model?

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
This paper introduces a multi-granularity prior knowledge distillation (MiPKD) framework for image super-resolution (SR) tasks. By incorporating both a feature prior mixer and a block prior mixer, MiPKD effectively transfers knowledge from a larger teacher model to a compact student model, enhancing model compression while preserving SR performance. Unlike conventional KD methods tailored to specific teacher-student architectures, MiPKD flexibly accommodates different network depths and widths.

### Strengths
1. Originality: MiPKD’s innovation lies in its multi-granularity knowledge mixing mechanism. Through coordinated use of feature and block prior mixers, MiPKD enables adaptable knowledge transfer across different teacher-student architectures, a notable improvement in KD design.

2. Quality: Experimental design is thorough and covers a wide array of SR network architectures and compression configurations. Results indicate MiPKD’s superior performance on multiple datasets, confirming its generalizability and efficacy across tasks.

3. Clarity: The paper’s organization is clear, with helpful visuals and an accessible writing style that conveys the method’s complexity effectively.

### Weaknesses
1.Explanability. Although MiPKD seems good in experiments, I still don't know why it is effective. The method introduces random masks I and R_k in the feature prior mixture and block prior mixture. but it needs to clarify why these masks are effective in distilling SR networks. I will be appreciated if the author could convince me through feature map analysis or theoretical deductions. 

2. Experiment Details: While MiPKD demonstrates good results, further details regarding the impact of different mask generation strategies in the feature mixer could clarify the specific role and contribution of these strategies.

3. Generality: The paper primarily discusses MiPKD’s performance in SR tasks, but it seems that MiPKD is not specially designed for the SR task. Thus, the authors are encouraged to show the applicability to other CV tasks. Additional experiments on a broader range of tasks could enhance the method’s perceived generalizability.

4. Experiments on current SOTA SR networks. Experiments are carried out on EDSR, RCAN and SwinIR networks. If possible, the authors could do experiments on more recent SOTA networks, such as DRCT-L and HMANet etc.

### Questions
1. Does the block-level mixing lead to instability in training for certain models? Were there hyper-parameter adaptations needed for different architectures during training?

2. Have the authors considered extending MiPKD to other vision tasks, such as object detection or semantic segmentation, to verify the broader applicability of the proposed framework?

### Soundness
3

### Presentation
2

### Contribution
2
