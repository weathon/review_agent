# Noise-Resilient Quantum Neural Networks via Zero-Noise Knowledge Distillation

- Decision: Reject
- Scores: 6, 2, 8, 4

## Abstract
Quantum neural networks (QNNs) show promise for learning on noisy intermediate-scale quantum (NISQ) devices, but two-qubit gate noise remains a significant barrier to practical implementation. Zero-noise extrapolation (ZNE) reduces errors by running circuits with scaled noise levels and extrapolating to the zero-noise limit, although it needs many evaluations per input and is susceptible to time-varying noise. We propose zero-noise knowledge distillation (ZNKD), a training-time technique that involves a ZNE-augmented teacher QNN supervising a compact student QNN. Variational learning is used to optimize the student's ability to duplicate the teacher's extrapolated outputs, resulting in robustness without the need for inference extrapolation. We additionally present a formal analysis that demonstrates how robustness flows from the ZNE teacher to the distilled student, with proofs regarding noise scaling, extrapolation error, and student generalization. In dynamic-noise simulations (IBM-style $T_1/T_2$, depolarizing, readout), ZNE-guided distillation lowers student MSE by $0.06$-$0.12$ ($\approx$10-20\%) across Fashion-MNIST, AG News, UCI Wine, and UrbanSound8K, keeping students within $2\%$--$4\%$ accuracy of the teacher and achieving $6{:}2$-$8{:}3$ ratio of teacher to student.  ZNKD, which amortizes ZNE to training, provides an efficient way to drift-resilient QNNs on NISQ hardware without per-input folding or extrapolation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Zero-Noise Knowledge Distillation (ZNKD), a hybrid framework that combines zero-noise extrapolation (ZNE) with knowledge distillation (KD) for training quantum neural networks (QNNs) robust to hardware noise. The method trains a noise-mitigated “teacher” QNN using Richardson extrapolation and transfers robustness to a smaller “student” QNN, which can then operate without costly noise extrapolation at inference. The paper provides formal theoretical results establishing bounds on robustness transfer and empirical evaluations on several small datasets (Fashion-MNIST, AG News, Wine Quality, and UrbanSound8K) under IBM-style noise models and limited hardware experiments. Results suggest a 10–20% MSE reduction versus non-distilled baselines and up to 8:3 model compression.

### Strengths
1.	The idea of amortizing noise mitigation through distillation is novel within quantum machine learning.
2.	Theoretical sections provide a formal foundation linking extrapolation error, approximation gap, and generalization.
3.	Experiments include multiple modalities (image, text, audio, tabular), illustrating the generality of the pipeline.
4.	Writing quality is generally clear, with correct referencing and structured proofs.

### Weaknesses
1.	The related work (Section 1) insufficiently contextualizes prior studies that combine error mitigation with compression or distillation. For instance, [Cerezo et al. 2021] and [Gou et al. 2024] are mentioned but not contrasted analytically. The paper should specify how ZNKD differs in mechanism or achievable robustness beyond replacing extrapolation by distillation.
2.	The compression ratios (6:2–8:3) are arbitrary and unexplained. It is unclear how teacher-to-student dimensional reduction is chosen or whether the student topology is optimal. Without ablation, one cannot assess trade-offs between expressivity and noise resilience.
3.	The link between ZNE theory (Section 2.3.1) and the distillation mechanism (Section 2.2.1) is unclear. There is no empirical demonstration that Richardson-extrapolated teacher labels are smoother or more stable targets for the student than raw noisy outputs.
4.	Although the paper argues amortization of ZNE cost, it does not quantify teacher training overhead (number of fold levels, total circuits executed). Practical resource savings remain unclear.

### Questions
1.	Can you provide quantitative runtime comparisons (in circuit executions) between ZNKD training and classical ZNE at inference to substantiate the claimed efficiency gain?
2.	How sensitive is ZNKD performance to mismatch between the simulated noise model and real hardware noise?
3.	Could you report results on larger circuits (≥16 qubits) or different ansätze to assess scalability?
4.	How were the Richardson extrapolation orders (λ ∈ {1,3,5}) chosen, empirically or theoretically?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a zero-noise knowledge distillation on QNN, which create noise-resilient QNNs for NISQ. The approach trains a teacher QNN with zero-noise extrapolation to generate noise-mitigated outputs, then distills this knowledge to a compact student QNN. The student inherits noise robustness without requiring runtime extrapolation, achieving 10-20% lower loss than non-distilled models while maintaining 6:2-8:3 compression ratios.

### Strengths
1. ZNE is responsible for teacher denoising, while the student takes on robustness, decoupling the costs of training and deployment; the algorithm and loss definition are intuitive and clear.
2. The paper attempts to tackle a highly relevant and practical problem in the NISQ era.

### Weaknesses
1. There are several data inconsistencies throughout the paper, such as 3090+256gb, which is later changed to 3090+128gb. The main text uses 64 shots, while the distillation early stopping method uses 1000 shots, and the appendix changes to 1024 shots.
2. The choice of metric is questionable: the classification task mainly reports MSE, which has limited explanatory power for the "improvement of 0.06–0.15", and the main text only mentions accuracy "incidentally", while the main table still focuses on MSE.
3. Missing ablation studies in main text.

### Questions
1. Could you please clarify and explain the differences in the settings for shots/noise parameters/λ? Which are used in the main results and which are only used in the appendix discussion? Do the corresponding figures need corrections?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Zero-Noise Knowledge Distillation (ZNKD), a training-time technique that enhances noise robustness in QNNs for NISQ hardware. The method combines zero-noise extrapolation (ZNE) with teacher-student distillation, where a large, ZNE-augmented teacher QNN supervises a smaller student QNN. During training, the student learns to reproduce the teacher’s extrapolated (near-noiseless) outputs, thus inheriting noise robustness without requiring extrapolation or circuit folding during inference.

The authors provide a formal analysis showing how robustness properties transfer from teacher to student, including proofs for extrapolation error bounds and generalization. Experiments on several datasets demonstrate consistent improvements in MSE and accuracy over baselines, achieving 10–2\% reductions in error and maintaining close alignment between teacher and student performance.

### Strengths
Well-motivated and timely contribution: The work directly addresses one of the central challenges in QML—the mitigation of noise on NISQ devices—by amortizing the cost of ZNE into the training phase.

Theoretical rigor: The formal treatment of robustness transfer, including proofs of noise scaling and extrapolation error, strengthens the methodological foundation.

Comprehensive empirical validation: The authors benchmark ZNKD across multiple datasets and compare it against relevant baselines, demonstrating consistent performance improvements.

Practical impact: Moving ZNE to the training stage significantly reduces inference overhead, making the approach more suitable for deployment on near-term quantum devices.

Clear writing and presentation: The paper is well organized, with intuitive figures and strong technical exposition.

This is a well-executed paper that offers a promising and practical path toward noise-resilient QNNs on NISQ devices. The theoretical grounding and breadth of experiments justify its acceptance. However, a more transparent analysis of computational cost and explicit discussion of teacher–student trade-offs would strengthen its impact and reproducibility.

### Weaknesses
Resource cost not fully analyzed: Although ZNKD avoids per-inference extrapolation, it still depends on an expensive teacher model trained with ZNE. The paper does not provide a quantitative analysis of total training cost (e.g., total number of circuit executions or measurement calls) relative to baseline methods. Including this in Table 3 or as a separate resource table would clarify the true computational trade-off.

Teacher dependence: The performance advantage largely stems from the strong teacher QNN, which is already near state-of-the-art compared with existing approaches. It is therefore unclear how much of the observed gain arises from distillation versus the teacher’s own performance.

Missing citations: While the authors reference general distillation literature, they omit several relevant works in quantum knowledge distillation, including:

Knowledge Distillation in Quantum Neural Networks using Approximate Synthesis
Bridging Classical and Quantum Machine Learning: Knowledge Transfer from Classical to Quantum Neural Networks using Knowledge Distillation
Hybrid Quantum–Classical Machine Learning with Knowledge Distillation

### Questions
How does the distillation benefit scale with the teacher’s quality?

Can the authors provide resource scaling estimates (e.g., number of circuit evaluations or measurement calls) for teacher during the separate training?

Add a resource cost column to Table 3 or a figure comparing total training-time circuit evaluations among ZNKD, ZNE, and baseline models.

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
2

### Summary
The paper is concerned with Zero-noise extrapolation (ZNE) for addressing the challenge of gate noise in quantum circuits. ZNE increases quantum circuit noise and extrapolates the measurement outcomes to the zero-noise limit, which may be very costly. The paper introduces a new student-teacher knowledge distillation approach for error prevention and compression. This conceptually new method is analyse mathematically and in experiments using both simulated noise and real quantum hardware.

### Strengths
- The paper addresses a timely and relevant topic: noise mitigation is a crucial issue for enabling quantum machine learning on NISQ devices
- The experiments appear to be comprehensive, covering different noise levels and datasets. 
- The mathematical analysis appears to be sound and provides theoretical insights on the relation between the ZNE scaling factors and the sample size.

### Weaknesses
Mismatch between provided source code and details in the paper:
- The source code is using qiskit version 0.45.3 (according to its "requirements.txt" file), which is depreciated since February 2024. 
- The code is not consistent at all with the procedure described in Section 2.2.1. For example, consider the loss described in Step (iii) "Distillation Objectives" using tanh and ZNE corrected expectations. In contrast, in the code (take wine/kd.py) in sections 1.6.-1.8 a bigger QNN is trained, then used to generate predictions, and then a smaller QNN is trained on those predictions. 
- The code also does not include any of the benchmarks reported in Table 4. On the other hand, some of these benchmarks (e.g. Wang et al. 2025) do not include experiments with the datasets considered here. This heavily undermines the paper's claims. 

Style and presentation:
- Writing and presentation: The level of presentation is currently below ICLR standards. Section 2 requires the reader to jump back and forth between Supplement and Main text several times within the first lines of reading. The main paper should be self-contained, with supplement  providing additional background and further supporting material. However, the supplement should not be needed for a basic understanding of what the paper aims to do. For example, the presentation of the method in Section 2.1 starts with "After defining gate-level decoherence using the Lindblad-informed noise model in A.1 ..." Then in "Motivation" it goes on "Using the single-gate fidelity euqation 32, demonstrates...". The same happens in several places below. 
- Typos: The paper still contains many typos, such as typsetting of "U" in l102 and expressions like "These denoised outputs are used as soft labels for training students (clients)." (what does "clients" refer to here?). Section 3.2 (lines 403-405) contain several misplaced "!"-symbols. 
- References out of place: In line 90-91 ZNE is attributed to a paper from 2024. However, this does not appear to be the correct reference.  Another example is Wang et al. (2025), which (differently than stated in the text) is not concerned with knowledge distillation or quantum transfer learning. Nevertheless, it is listed as a benchmark method in Table 4. 
- The paper seems to still contain an LLM prompt: line 284-285 states: "Give an explanation of the extrapolated estimator’s mean squared error (MSE) as ...".

### Questions
Please see Weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
3
