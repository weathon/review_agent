# L-SR1: Learned Symmetric-Rank-One Preconditioning

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
End-to-end deep learning has achieved impressive results but remains limited by its reliance on large labeled datasets, poor generalization to unseen scenarios, and growing computational demands. In contrast, classical optimization methods are data-efficient and lightweight but often suffer from slow convergence. While learned optimizers offer a promising fusion of both worlds, most focus on first-order methods, leaving learned second-order approaches largely unexplored.

We propose a novel learned second-order optimizer that introduces a trainable preconditioning unit to enhance the classical Symmetric-Rank-One (SR1) algorithm. This unit generates data-driven vectors used to construct positive semi-definite rank-one matrices, aligned with the secant constraint via a learned projection. Our method is evaluated through analytic experiments and on the real-world task of Monocular Human Mesh Recovery (HMR), where it outperforms existing learned optimization-based approaches. On the HMR task, it surpasses a fully-trained baseline using only 10% of the training data, underscoring its data efficiency. Featuring a lightweight model and requiring no annotated data or fine-tuning, our method offers strong generalization and is well-suited for integration into broader optimization-based frameworks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Based on the SR1 concept, this paper proposes a Learnable Second-Order Precondition (L-SR1) that learns a positive semi-definite low-rank precondition through three lightweight modules: an input encoder, a vector generator, and a coordinate-wise learning rate generator. A secant-line consistency penalty is introduced during meta-training to enforce constraints, without adding any extra computation during inference. The proposed method is verified on analytical benchmark function families and humanoid mesh recovery (HMR, 3DPW). After replacing the update module of LGD in HMR, L-SR1 achieves higher accuracy, fewer steps, and fewer parameters, demonstrating clear advantages in data efficiency and computational and memory cost.

### Strengths
1. The paper proposes a trainable preconditioning unit that augments classical SR1 using data-driven rank-one components and a learned projection that enforces secant consistency while keeping the preconditioner positive semi-definite. This represents an uncommon and creative combination in the learned-optimizer space.
2. It reframes the question of how to satisfy the secant constraint without losing positive semi-definiteness as a learnable projection or penalty design, which removes a long-standing instability of SR1 in a way that is amenable to meta-learning.
3. On quadratic functions and classic benchmarks, the method with the learned projection shows faster decrease and better alignment with a high-quality direction. The performance-profile evaluation indicates the best overall profile among baselines, reflecting a solid and standardized methodology.
4. In HMR (3DPW), the approach improves accuracy over a learned gradient descent baseline while using fewer parameters. The reported curves and tables show the steps to the best error and the model size, following known evaluation protocols.
5. The paper clearly explains the SR1 background, introduces the learned components and projection, and then transitions to analytic and real-world experiments with a consistent and coherent narrative.
6. The work addresses a real pain point in learned optimization. Most learned optimizers are still first-order methods, while this approach advances learned second-order optimization with a lightweight and stable design, lowering the barrier to effectively using curvature information in practice.

### Weaknesses
1. The main comparisons are with LGD and an LBFGS variant, but the table omits many recent HMR systems and learned optimizers that are well recognized in the community. Please include fair, apples-to-apples comparisons with at least one or two recent strong HMR frameworks (using the same 2D detections and evaluation protocol) and add a modern learned optimizer baseline (e.g., transformer-based) under the same iteration budget.
2. The ablations cover hidden size and encoder inputs, but not the parameters that likely affect stability and generalization, such as buffer length, unroll steps, secant-penalty weight, and learning-rate scaling. Please provide a grid or heatmap over buffer length × unroll steps × penalty weight × LR scale, reporting final error, iterations to target, failure or “bad-step” rate, and variance across random seeds.
3. It is not entirely clear how the buffer, base matrix, and generated vectors interact during inference beyond the training-time penalty. Please add clear, step-by-step pseudocode for inference, along with a table reporting test-time constraint residuals, curvature–direction alignment, and any guardrails used (e.g., fallback or damping).

### Questions
1. Are any consistency checks or safeguard mechanisms (such as trimming, damping, or fallback) applied during inference? Please provide a small table reporting the constraint residuals at each test step, the proportion of failed steps, and the frequency of guardrail activations.
2. Please supplement the performance profile using exactly the same number of iterations and/or the same wall-clock budget. Additionally, include a curve showing the time or number of steps required to reach the target error, ensuring that the conclusions are not affected by asymmetric budgets or unbalanced rollout lengths.
3. Please provide a small grid or heatmap varying buffer length, unroll steps, projection penalty weight, and learning-rate scaling, and report the final error, the number of steps needed to reach the target, the proportion of bad steps, and the variance across random seeds.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes L-SR1, a learned second-order optimizer that augments classical SR1 with a trainable low-rank preconditioner and a learned projection that enforces both the secant constraint and positive semidefiniteness, then applies it to analytic functions and monocular HMR.

### Strengths
- Principled design bridging QN and learning. The method is explicitly grounded in the QN update​, the secant condition, and the need for PSD preconditioners for descent directions; the learned projection aims to satisfy both simultaneously.


- Lightweight, limited-memory, dimension-invariant formulation. L-SR1 uses rank-one outer products with a fixed-size buffer and element-wise modules to generalize across problem sizes without retraining.


- Learned projection and per-coordinate step sizes. The optimizer integrates a vector generator and per-coordinate learning-rate generator; the projection improves convergence while keeping the model compact.

### Weaknesses
- **Insufficient Comparative Analysis**: The paper should compare against a wider set of HMR methods (both optimization-based and modern learned/learnable refiners beyond LGD/SPIN) and report more metrics (e.g., MPJPE, PA-MPJPE, PVE, jitter/contact, temporal stability) on more datasets (e.g., Human3.6M, EHF, AGORA) under matched settings. As written, the HMR main table includes only a few baselines. Moreover, the paper fails to provide an analysis of the computational cost and runtime of the proposed method versus more competitors beyond LGD. This essential data is needed to properly highlight the claimed efficiency.

- **Poor Visual Quality**: A major concern is the poor visual alignment of the fitted SMPL models in Figure 5 and Figure 9. For example, the case in the third row of Fig. 9 is clearly misaligned with the input image in the leg area. The meshes frequently appear misaligned with the subjects in the original images, suggesting a lack of robustness.

### Questions
- Why was HMR selected as the experimental testbed, and which limitations of existing HMR approaches does this work seek to overcome? The connections between the proposed method and the HMR are unclear. Please elaborate with stronger justification.

- Justification of Optimization: The fundamental advantage of optimization-based methods is often their ability to achieve higher accuracy by iteratively refining the fit to the image evidence. However, given that your method's performance (PA-MPJPE: 51.58) lags significantly behind current state-of-the-art learning-based methods (e.g., PromptHMR [CVPR 2025]: 36.6; CameraHMR [3DV 2025]: 35.1), the following question arises: If the accuracy of your optimization-based method is not superior to, and in fact is substantially worse than, its learning-based counterparts, what is the practical and methodological significance of using a much slower optimization approach?

- Scope of “no annotations / no fine-tuning.” For HMR, please specify the exact inner objective (2D keypoints? camera model? masks/contours?), any external detectors used, and how this complies with the “no annotations” claim; otherwise, the “10% data surpasses full training” result is hard to interpret. 

- Supervision boundary. Clarify “no annotated data / no fine-tuning”: what signals supervise HMR (e.g., only 2D keypoints?), what detector(s) and confidence treatment are used, and how is the 10% data experiment constructed (splits, metrics, variance)?

- Discuss failure cases (e.g., heavy occlusion, noisy 2D keypoints input, temporal consistency).

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces ​​L-SR1 (Learned Symmetric-Rank-One)​​, a novel learned second-order optimizer that enhances the classical SR1 algorithm by integrating a lightweight, trainable preconditioning unit. This unit generates data-driven vectors to construct positive semi-definite rank-one matrices, with a key innovation being a learned projection mechanism that enforces the critical secant condition.

### Strengths
The theoretical foundation of this work is  presented with clarity.

### Weaknesses
The method is only tested on HMR, a specific application in image processing.
It is not know whether this method is applicable or not for other tasks.

### Questions
1. As a reviewer not deeply familiar with optimizers, I am unclear about the specific contribution of this work in the optimization field. While the authors propose an improved SR1 method, I am uncertain whether such approaches are common or novel in optimization research.
2. Ignoring novelty, the work is valuable as it applies learned optimizers to SR1. The authors appear to address key challenges in adapting SR1 into a learned framework.
3. I have some concerns regarding the sufficiency of the experimental validation. The evaluation is primarily centered on the Monocular Human Mesh Recovery (HMR) task. I am curious to know in which other tasks or domains this method could be effectively applied. HMR itself can be addressed by both optimization-based methods (e.g., SMPLify) and regression-based approaches. This leads to a question of whether the proposed method is specifically tailored to similar iterative optimization frameworks. If so, its potential application domain might be somewhat limited, though I acknowledge this could be a general challenge for the field of learned optimizers rather than a specific shortcoming of this paper.
4. Furthermore, the comparative analysis in the experiments appears relatively limited. For instance, Table 1 primarily reports results on the 3DPW dataset. It would strengthen the paper to include comparisons on other standard benchmarks in this domain. Additionally, the baseline methods used for comparison seem to be from 2020 or earlier. Including comparisons with more recent state-of-the-art approaches would provide a more compelling and up-to-date demonstration of the method's competitiveness.

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
This paper presents a learned second-order optimizer, named Learned-SR1 (L-SR1), which proposes to use a trainable preconditioning unit to enhance the classical Symmetric-Rank-One (SR1) algorithm while ensuring positive definite preconditioning matrices that satisfy the
secant constraint. The proposed method shows superiority over existing optimizers in benchmark tasks such as Human Mesh Recovery (HMR), and also demonstrates the highest performance profile and converging to the global minimum faster.

### Strengths
A lightweight, self-supervised learned optimizer that integrates a trainable preconditioning unit into the SR1 framework is introduced.

A learned projection mechanism that enforces both the secant condition and positive semi-definiteness, preserving core Quasi-Newton properties within a learned architecture is introduced. 

Experiments show that the proposed method works well on HMR. 

The paper is well written and easy to read.

### Weaknesses
The claimed generalization of the proposed Learned-SR1 is not effectively validated. Currently, there is only simple evaluation on HMR task, and the compared baselines do not represent the current state-of-the-art for HMR. 

The paper lacks comparison with more optimization algorithms, e.g., AdamW, AdaHessian, etc. Moreover, the theretical analysis of the learned projection mechanism is insufficient. 

Currently, the evaluation is conducted only on a single dataset (3DPW), which fails to demonstrate the effectiveness and robustness of the proposed method. 

This work lacks a thorough ablation study for each design in the proposed method.

### Questions
How does L-SR1 perform on tasks beyond HMR?

There is no visual comparison for HMR. It would be better if visual comparisons could be included in Figure 5.

### Soundness
3

### Presentation
2

### Contribution
2
