# Could Student Selection Be the Missing Piece for Efficient Distillation?

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Selecting the optimal student architecture remains an overlooked challenge in knowledge distillation (KD). Current approaches typically rely on model size constraints or random selection, ignoring how student architecture and inductive biases impact distillation effectiveness. We formulate this as an unsupervised model selection problem, where the goal is to select the best student for a given teacher without requiring ground-truth labels or expensive training cycles. We propose a transferability metric based on the Neural Tangent Kernel (NTK) that quantifies function space alignment between teacher and student models. Specifically, our cross-model NTK measures the directional similarity between teacher and student gradient vectors on unlabeled data, capturing how effectively the student can mimic the teacher's function through gradient-based optimization. Unlike existing transferability metrics that require ground-truth labels and focus on model-dataset relationships, our approach directly models the model-model relationship central to KD. To ensure practical applicability with modern networks, we implement an efficient approximation using Johnson-Lindenstrauss random projections that preserves gradient inner products without computing full NTK matrices. Experiments demonstrate that our metric is robust, and reliably predicts the post-distillation performance, outperforming existing transferability scores adapted for KD and baseline selection strategies, even in low-data scenarios. Our approach enables efficient identification of compatible student architectures before training, eliminating the need for resource-intensive trial-and-error in model compression pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper argue that current approaches typically rely on model size constraints or random selection, ignoring how student architecture and inductive biases impact distillation effectiveness. The paper proposes C-MoNA (Cross-Model NTK Alignment), a unsupervised, and training-free metric to predict post-distillation performance. C-MoNA measures the function space alignment between a teacher and a student candidate by computing a cross-model Neural Tangent Kernel (NTK).

### Strengths
- This paper is easy to follow and understand;

### Weaknesses
The paper overlooks the large body of work in Neural Architecture Search (NAS), particularly the numerous studies that combine NAS with KD. For instance, the 2023 work DisWOT already achieves a training-free paradigm very relevant to this paper’s scope, yet it is neither cited nor compared. Same for the Search to distill. 

[1] DisWoT https://arxiv.org/pdf/2303.15678
[2] Search to distill: Pearls are everywhere but not the eyes

The benchmark design is problematic — selected baselines are non-mainstream and miss direct comparisons with standard zero-cost NAS proxies. Furthermore, almost all evaluation relies solely on Kendall’s tau for ranking consistency, without assessing downstream task performance, limiting practical applicability.

The evaluation uses a very limited set of teacher–student pairs, making the computed Kendall’s tau unreliable in representing general trends. Moreover, ranking consistency is poor — in most cases below 50%, which significantly undermines its viability as a practical student network selection method.

### Questions
Why did the authors not conduct systematic comparisons with NAS-based methods, particularly training-free KD-NAS approaches such as DisWOT? Do they consider these works incomparable with theirs?

Why was Kendall’s tau evaluation conducted with such a small number of teacher and student architectures? Has the method’s stability been tested on a more diverse set of combinations?

Apart from Kendall’s tau, have the authors considered validating effectiveness by measuring the downstream task performance of the selected student models after actual KD fine-tuning?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the problem of pre-training, label-free selection of student models that best match a given teacher in knowledge distillation (KD). The authors define a cross-model NTK between teacher and student and compress it into a Frobenius-normalized scalar α∈[0,1]used as a prior compatibility score (computational cost is reduced via FJLT). They validate the metric by measuring the weighted Kendall’s τ between the prior α-based student ranking and the post-KD performance ranking, reporting consistently positive τand low variance across three classification datasets, an object detection task, and low-data settings (25–100 images). This supports the claim that C-MoNA, which directly measures model–model alignment, is more suitable for student selection than model–data transferability metrics.
Contributions
1.	Formulates label-free student selection in KD around model–model compatibility.
2.	Proposes cross-model NTK alignment (α) with Frobenius reduction to measure global alignment.
3.	Uses FJLT-based approximation to make computation feasible for large models.
4.	Demonstrates predictive power via prior→posterior rank agreement (τ) in classification, detection, and low-data scenarios.

### Strengths
1.	They clearly formulate the KD problem as pre-training student selection based on model–model compatibility, which they identify as the key driver of success in distillation.
2.	They propose a metric that uses the cross-model NTK between teacher and student to summarize global alignment as a 0–1 scalar α, thereby directly quantifying inter-model compatibility.
3.	By approximating high-dimensional gradient operations with FJLT (JL) projections, the method becomes scalable to large networks and teacher–student pairs with mismatched parameter dimensions.
4.	They demonstrate experimental validity by showing that, in classification/detection and low-data (25–100 images) regimes, the prior α-based ranking is consistently positively correlated (Kendall’s τ) with the post-KD performance ranking.

### Weaknesses
1.	Rather than reporting only means, include standard deviations and/or 95% confidence intervals in the main text to quantify uncertainty and make effect sizes interpretable.
2.	The paper lacks analysis of domain shift: when the target input distribution differs substantially from the teacher’s pretraining distribution, it is unclear whether and when the α–KD performance correlation holds or breaks.
3.	Under identical conditions where different student-selection algorithms yield different rankings, the paper reports only τ and does not directly demonstrate whether the chosen selection actually succeeded.

### Questions
1.	Could you provide τ sensitivity curves (mean ± standard deviation) across projection dimension, numbers of unlabeled samples, and state the minimum practical projection dimension and sample lower bound you recommend?
2.	Under domain shift (e.g., long-tailed targets or style-transferred images), does the α–KD performance correlation hold or break, and if so, why; please report concrete hold/break cases and any mitigations
3.	Many results are table-only; will you add figure summaries to improve readability and highlight effect sizes?
4.	To better support the central claim, can you move at least one end-to-end result to the main text showing that high-α students indeed learn and achieve superior test performance, and explain why these results were placed only in the appendix?

### Soundness
3

### Presentation
1

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
This paper addresses the often-overlooked problem of student model selection in knowledge distillation. Instead of relying on model size or ad hoc heuristics, the authors propose C-MoNA, a transferability metric based on Cross-Model NTK alignment. Their approach is both unsupervised and label-free, focusing on model-model instead of model-data relationship. They employ Johnson-Lindenstrauss projections for scalability and validate C-MoNA across image classification and object detection tasks, in both full- and low-data regimes.

### Strengths
1. Novelty - The proposal of a cross-model NTK for student selection in KD is novel. While NTK theory has been used in other areas, using it to estimate transferability between models—before training—is a significant contribution. The proposed C-MoNA is both theoretically motivated and empirically validated.

2. Method Soundness - The formalization of why model-data metrics fail in KD (e.g., the breakdown of distillation scaling laws and capacity-gap interactions) is clear and well-grounded in recent literature. Also, the derivation of the cross-model NTK and the choice of Frobenius norm as the alignment score are mathematically motivated.

3. Evaluation - The evaluation is good, including diverse teacher and student architectures across benchmarks, low-data evaluation, correlation analysis and ablation. The results not only show the strength of their proposed metrics, but also show when and why prior metrics fail.

### Weaknesses
1. The paper does not provide a formal theoretical guarantee or bound on how well the C-MoNA score correlates with actual distillation performance. Specifically, Eq. (9) is used to infer how aligned a student is with a teacher. However, there is no bound or derivation provided that shows how \alpha theoretically correlates with downstream KD performance (e.g., final test accuracy or distillation loss). This limits its use in safety-critical or performance-guaranteed settings. The authors rely entirely on empirical correlation using Kendall's \tau.

2. The proposed metric assumes that similarity in initial gradient directions (i.e., via cross-model NTK at initialization) is a reliable predictor of training dynamics, as indicated in Section 3.2: “NTK encapsulates the inductive biases inherent in network architectures... predicting final function learned... without requiring training.” But modern training involves nonlinear behaviors far from NTK’s lazy regime assumptions, especially for transformer architectures. No experiments in the paper test the robustness of C-MoNA in deep or highly nonlinear training regimes, such as those with hundreds of epochs or curriculum learning.

3. The paper lacks qualitative or quantitative analysis of the scenarios where C-MoNA fails or underperforms—which is critical for practitioners to understand when not to use it. For instance, in Table 5 (VOC object detection), C-MoNA underperforms compared to PACTran in the “Yolo11m-Wild” setting (0.122 vs. 0.521), but the reason is not clear.
	​

### Questions
N.A.

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
This paper proposes a method to select the optimal student model in knowledge distillation without needing ground-truth labels or expensive training cycles. The authors designed a metric based on the Neural Tangent Kernel (NTK) to quantify the function space alignment between the teacher and student models. To transfer this metric to modern networks, it uses Johnson-Lindenstrauss (JL) projections for an efficient approximation. Experiments demonstrate that this method is robust and transferable.

### Strengths
1、The paper firstly formalizes the problem of unsupervised student model selection in knowledge distillation.
2、It proposes a metric based on cross-model NTK alignment that requires no ground-truth labels. It also uses JL projections to transfer the method for modern models.
3、Extensive experiments demonstrate the method's robustness and effectiveness across different models and datasets.

### Weaknesses
1、The introduction of Eq.(1) is too simple, without the explanation for the formulas for each loss term or the function of different parameters. Moreover, LSe is mistakenly written as LeS in Eq.(1).
2、The "Mathematical Proof of Ranking Divergence" section is not rigorous. It is unclear how the ranking reversal LT/LS1e>LT/LS2e is derived from Eq.(2). The proof does not analyze the function of g(∙) and h(∙).
3、What is the meaning of “@” in line 224? 
4、Why choose JL projection for dimensionality reduction? It will be more convincing if there are comparative experiments of different projection methods. 
5、The core content seems to be a combination of existing techniques (NTK and JL projection) rather than a theoretical innovation. 
6、The baselines are all before 2023. The paper does not compare against SOTA from the last few years. Moreover, C-MoNA does not achieve the optimal results in many cases presented in Tables 3, 4, and 5. It is doubtful about the effectiveness.

### Questions
see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
