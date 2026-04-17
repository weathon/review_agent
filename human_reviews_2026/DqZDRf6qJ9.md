# BrainDistill: Implantable Motor Decoding with Task-Specific Knowledge Distillation

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Transformer-based neural decoders with large parameter counts, pre-trained on large-scale datasets, have recently outperformed classical machine learning models and small neural networks on brain–computer interface (BCI) tasks. However, their large parameter counts and high computational demands hinder deployment in power-constrained implantable systems. To address this challenge, we introduce $\textbf{BrainDistill}$, a novel implantable motor decoding pipeline that integrates a neural decoder with a distillation framework. First, we propose $\textbf{TSKD}$, a task-specific knowledge distillation method that projects task-relevant teacher embeddings into compact student models. Unlike standard feature distillation methods that attempt to preserve teacher representations in full, TSKD explicitly prioritizes features critical for decoding through supervised projection. To evaluate the framework, we define the task-specific ratio ($\textbf{TSR}$), a new metric that quantifies the proportion of task-relevant information retained after projection. 
Building on this framework, we propose the Implantable Neural Decoder ($\textbf{IND}$), a lightweight transformer architecture that combines linear attention with continuous wavelet tokenization, optimized for on-chip deployment. 
Across multiple neural datasets, IND consistently outperforms prior neural decoders on motor decoding tasks, while its TSKD-distilled variant further surpasses alternative distillation methods in few-shot calibration settings. Finally, we present a quantization-aware training scheme that enables integer-only inference with activation clipping ranges learned during training. The quantized IND enables deployment under the strict power constraints of implantable BCIs with minimal performance loss.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Braindistill: implantable motor decoding with task-specific knowledge distillation presents three main contributions: a linear attention Transformer (IND), a distillation technique (TKSD), and metric to evaluate the data-specific effectiveness of a projection (TSR). This work provides compelling empirical evidence for all three components, albeit mostly on private datasets.

### Strengths
1. Advantage over the state-of-the-art on the datasets considered is clear
2. Ablations are thorough and validate the choices of the authors
3. The stability shown under quantization is a nice bonus

### Weaknesses
1. While the paper advertises 6 datasets (3 private and 3 public), the relevant performance comparisons against baseline models are only performed on the 3 private ones, and the public ones are left for ablations. The same benchmarking must be performed on the public ones to ensure unbiased results.
2. A lighter version of EEGPT (the teacher model) needs to be included as a baseline. As it stands, it is not clear whether EEGPT performs better than IND because of the size or because of the architecture.
3. Related to the point above, I assume that EEGPT is chosen due to its large size (100M), while the others are considerably smaller. This should be made clear, and the parameters of all models also need to be reported.
4. The paper is quite disjointed and tough to parse. I found it rather difficult to read and decipher how all the components go together.

### Questions
1. It’s not fully clear what “training from scratch” means for the student model. For distillation, from my understanding the teacher is trained on 159 sessions (X_{offline}), then the student is distilled using X_{recalib}, then the student is tested on X_{online}. For the “scratch” case, is the student trained on X_{offline}, X_{recalib}, or both at the same time?
2. The definition of X_{offline} is not fully clear. Is it patient-specific or not?
3. \Delta is not defined in Eq. 1
4. What’s the advantage of not using the same classifier during the two phases of distillation?
5. The division of the splits is quite confusing at first, should make clear that, e.g., 1-1 means training on the training split of session 1 and testing on the testing split of session 1.
6. The second term of L_{TKSD} is highly reminiscent of ridge regression, you might find works on the Tikhonov factor useful for the determination of \lambda.
7. For clarity, it should be specified that the classifier must be a single layer + non-linearity

### Soundness
2

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
3

### Summary
The paper introduces a new novel distillation pipeline for motor decoding via bringing student / teacher embedding space together by minimising two objectives. In addition, another contribution of the paper is the utilization of the overlap between projecting spaces as a metric for the distillation error.

### Strengths
The paper presents a solid teacher - student model. The maths behind the methodology is also well-described and has a nice flow. The fact that it goes beyond EEG to also ECoG and Spikes is also very interesting. 

Writing:
Paper is well-written and good structured.

### Weaknesses
The main objective of the paper is not clear. Is the main purpose the distillation methodology or the IND architecture which (as the authors claimed is pretty basic) ? This should be better described.

Overall:
The paper shows some merits but it would be interesting to have my questions answered.

### Questions
1. In my opinion the paper’s main contribution is the distillation methodology rather than the IND architecture. Have you tried to add this framework on other models like the ones you compare with ? The results would be interesting to be added here.
2. I would also like to see comparison with other PERF methods like LoRA. A recent work for EEG [1].
3. How about comparison with SOTA foundation models ?

[1]: Na Lee, Konstantinos Barmpas, Yannis Panagakis, Dimitrios Adamos, Nikolaos Laskaris, & Stefanos Zafeiriou (2025). Are Large Brainwave Foundation Models Capable Yet ? Insights from Fine-Tuning. In Forty-second International Conference on Machine Learning.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces BrainDistill, a framework for efficient and deployable neural decoding in implantable brain–computer interface (BCI) systems. The approach integrates a Task-Specific Knowledge Distillation (TSKD) method with a lightweight transformer-based Implantable Neural Decoder (IND). TSKD compresses teacher embeddings into a low-dimensional, task-relevant subspace, guided by a new metric called the Task-Specific Ratio (TSR) that quantifies how much task-related information is preserved after projection. IND further combines continuous wavelet tokenization with quantization-aware linear attention to enable low-power, integer-only inference suitable for on-chip deployment.
Experiments across ECoG, EEG, and spike datasets demonstrate consistent improvements in decoding accuracy and robustness over prior distillation baselines and traditional decoders, while the quantized IND achieves a reported 3× reduction in power consumption (5.66 mW) with minimal accuracy loss.

### Strengths
1.	The motivation, i.e. bridging the gap between large neural decoders and implantable hardware, is timely and clearly articulated. TSKD addresses a concrete limitation of standard distillation (feature mismatch and capacity gap) with a principled projection-based approach.
2.	The two-step projection method (supervised compression followed by fixed alignment) is well-designed. TSR provides an interpretable quantitative measure that correlates with distillation quality and offers practical diagnostic value.
3.	The integer-only quantization with learnable clipping ranges is implemented carefully and validated with realistic energy estimates. 
4.	The framework is tested on a wide range of neural modalities (ECoG, EEG, spikes) and datasets, consistently outperforming KD, SimKD, VkD, and RdimKD. Ablations on tokenization and projection methods support the claims well.

### Weaknesses
1.	It would be helpful to understand whether TSKD’s projections depend critically on the quality of the teacher classifier and how sensitive TSR is to teacher miscalibration.
2.	The paper assumes TSR correlates with downstream accuracy, but this relationship is only shown qualitatively. Quantitative correlation plots between TSR and decoding performance across projection types would strengthen the claim.
3.	The power numbers appear simulation-based rather than measured. Including hardware prototype details or synthesis-level validation would improve the credibility of the “implantable” claim.
4.	Some mathematical derivations (Eqs. 1–5) are dense. The paper would benefit from a clearer high-level description of intuition behind Eq. (4) and the projection procedure.

### Questions
1) Does TSR quantitatively predict decoding accuracy, and is it a reliable task relevance metric?
2) Are the hardware power savings realistic and fully measured (not simulation-only)?
3) Are distillation baselines implemented under strictly identical conditions?

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
The paper introduces BrainDistill, a pipeline designed for implantable brain–computer interface (BCI) decoders that operate under strict power constraints. The authors propose the Task-Specific Knowledge Distillation (TSKD) method that projects teacher embeddings into a task-relevant subspace for more efficient student learning and a lightweight transformer-based decoder using Continuous Wavelet Transform (CWT) tokenization and linear attention for quantization and implantation. The authors use Task-Specific Ratio (TSR) metric to measure how much task-relevant information is preserved during projection. They evaluate the approach on human and primate datasets (ECoG, EEG, spike data) and demonstrate reduced power consumption via their quantization scheme for integer-only inference.

### Strengths
1. Implantable BCIs impose unique hardware limits and the authors correctly identify power efficiency as a major bottleneck. So, the quantization analysis and power consumption estimates are relevant to practical deployment.

2. The mathematical exposition of projection-based distillation is useful for understanding of feature compression.

3. Covers three neural recording modalities (ECoG, EEG, spikes) and shows decoding performance improvements across these modalities.

### Weaknesses
1. No comparison is done against other task-oriented or projection-based distillation methods (e.g. [1-3]) under identical training conditions.

2. There is no ablation comparing IND architecture vs. TSKD itself.

3. Architecturally, the model is very similar to [1-3] and similar task-specific KD approaches and novelty of the method is under question. There is no support for the following: "However, existing KD methods primarily aim to preserve teacher embeddings as fully as possible (Miles et al., 2024; Zhou et al., 2025; Guo et al., 2023), which becomes problematic when the student model lacks the capacity to mimic complex teacher features, resulting in limited performance gains."

4. There is no baseline comparison between IND and other quantization methods e.g. [4, 5].

5. The core of the reported results in the main text are from a private dataset (Human-C). Furthermore, no code is provided which hinders reproducibility.

6. Figures 2 and 3 are mostly descriptive and do not provide any insights or explanations of model performance.

[1] Less is more: Task-aware layer-wise distillation for language model compression

[2] Task-oriented feature distillation

[3] Improving Knowledge Distillation using Orthogonal Projections

[4] Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference

[5] Post-training 4-bit quantization of convolution networks for rapid-deployment

### Questions
1. What distinguishes TSKD from other supervised subspace alignment approaches? Can you point out to any novelty beyond changing the dimensionality of the projection output?

2. Why is TSR a better metric than simpler reconstruction loss or mutual info measures for projection quality?

3. How much of the performance gain arises from CWT tokenization versus the linear attention module (ablations)?

4. Can the method be extended beyond motor decoding (e.g., speech or visual BCIs)?

5. How can the results be verified? All code and private datasets should become publicly available.

### Soundness
2

### Presentation
2

### Contribution
2
