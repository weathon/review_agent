# TRUST THE UNCERTAIN TEACHER: DISTILLING DARK KNOWLEDGE VIA CALIBRATED UNCERTAINTY

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
The core of knowledge distillation lies in transferring the teacher’s rich ‘dark knowledge’—subtle probabilistic patterns that reveal how classes are related and the distribution of uncertainties. While this idea is well established, teach- ers trained with conventional cross-entropy often fail to preserve such signals. Their distributions collapse into sharp, overconfident peaks that appear decisive but are in fact brittle, offering little beyond the hard label or subtly hinder- ing representation-level transfer. To address this limitation, we revisit distillation from a distributional perspective and propose Calibrated Uncertainty Distillation (CUD), a framework designed to make dark knowledge more faithfully accessi- ble. Instead of uncritically adopting the teacher’s overconfidence, CUD encour- ages teachers to reveal uncertainty where it is informative and guides students to learn from targets that are calibrated rather than sharpened certainty. Across di- verse benchmarks, CUD yields students that are not only more accurate, but also more calibrated under shift and more reliable on ambiguous, long-tail inputs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Calibrated Uncertainty Distillation (CUD), which reformulates knowledge distillation as a two-stage process. In Stage 1, a Difficulty-aware Uncertainty Shaping (DUS) module is employed. It uses a gated entropy bonus combined with focal loss to increase predictive entropy, thereby redistributing probability mass toward semantically related non-maximal classes. In Stage 2, Wrong-mass Clipping (W-Clip) is proposed to address over-confident yet incorrect teacher predictions. It transfers the smaller value between a predefined budget and the margin between the top-1 incorrect class and the ground-truth class to the correct class, yielding a wrong-mass–budgeted soft target. Finally, the student is trained using a tempered KL divergence toward the calibrated teacher distribution, optionally combined with a cross-entropy loss.

### Strengths
1. The authors provide a clear and well-supported analysis showing that cross-entropy–trained teachers suffer from overconfidence and distribution collapse, which result in the loss of dark knowledge.
2. Both DUS and W-Clip are lightweight and architecture-agnostic modules that can be seamlessly plugged into existing systems, offering low implementation cost and high practicality.

### Weaknesses
1. The experiments are limited to text datasets, without validation on other modalities, which restricts the generality of the conclusions.
2. The compared baseline methods appear somewhat outdated.
3. Many knowledge distillation methods have been proposed to mitigate the over-confidence of teacher models. The authors should clarify what concrete advantages their approach offers, particularly considering that the comparisons and data modalities are relatively limited.
Moreover, since teacher models are usually large, fine-tuning them can incur considerable computational cost. It would be helpful to discuss how the training overhead of the proposed method compares to other approaches, and whether the increased cost is reasonable given the performance gains.

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
3

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
This paper introduces a novel knowledge distillation framework that corrects overconfidence in teacher models, which often lose meaningful uncertainty structure when trained with cross-entropy due to hard labels. CUD achieves faithful, calibrated transfer through two components: Difficulty-aware Uncertainty Shaping (DUS), which fine-tunes the teacher to increase entropy on hard or misclassified examples while keeping correct predictions sharp, and Wrong-mass Clipping (W-Clip), which reduces probability mass on the most overconfident wrong class while preserving relative structure.

### Strengths
1) The paper reformulates KD as a constraint-based projection problem, uniting calibration and uncertainty into a theoretical foundation.
2) The two components (DUS and W-Clip) are simple, interpretable, and easy to implement, yet show strong results.
3) The paper is well written.
4) Highlights a real gap in KD—teachers’ overconfidence—and offers a compelling argument for calibrated transfer.

### Weaknesses
1) Experiments are restricted to single-label text classification. Extension to other modalities (vision, multimodal) or multi-label/generative tasks would strengthen generality.

2) Hyperparameter sensitivity: The method introduces several tuning parameters, and while defaults are reported, their robustness across datasets is not deeply analyzed.

3) Although calibration is claimed to improve OOD robustness, explicit shift experiments (e.g., corrupted or domain-changed data) are missing.

### Questions
1) Can the proposed method generalise to vision transformers or multimodal LLMs (e.g., CLIP or LLaVA)?

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
This paper proposes Calibrated Uncertainty Distillation (CUD), a knowledge distillation framework that addresses teacher overconfidence by (1) training teachers to preserve difficulty-aware uncertainty and (2) calibrating the transfer process to suppress spurious confident predictions. The authors evaluate on text classification tasks and demonstrate improvements particularly on high-cardinality datasets.

### Strengths
1. The paper clearly articulates a significant limitation of conventional KD, that cross-entropy trained teachers produce overconfident, collapsed distributions that fail to transfer rich "dark knowledge."
2. The constraint-based reformulation (R1 and R2) provides a clear theoretical framework.
3. Unlike feature-based methods, CUD works with response distributions alone, making it applicable when architectural gaps are severe.
4. The evaluation covers diverse cardinalities (2-150 classes) and multiple student capacities, with honest reporting of where the method does and doesn't help.
5. The paper clearly explains why gains are largest for high-cardinality tasks and minimal for binary classification. These types of observations are often missing from KD papers.

### Weaknesses
**Major Issues**

1. The main idea of DUS largely combines focal loss with gated entropy regularization. While “difficulty-aware uncertainty shaping” is an appealing framing, the actual mechanism feels like a reinterpretation of existing methods. The link to the constraint-based formulation (Eq. 2) also feels somewhat loose. The paper doesn't clearly delineate what is conceptually new versus what is the engineering of existing ideas.

2. Experiments are limited to BERT-based text classification. Despite mentions of GPT-4 and LLaMA in the introduction, no large-model or cross-domain evaluation is provided, making generalization unclear.

3. Missing comparisons with label smoothing and temperature tuning, being the most obvious baseline for "calibrated targets". 
4. The method introduces many hyperparameters, but only γ is tuned. The rest are fixed without justification, which weakens the claim of being architecture-agnostic. .Why clip at W=3.0 for example weighting? This seems arbitrary. 
5. Results are single-run with no variance or significance reporting, so reliability is uncertain.
6. Claims around calibration and robustness are unsubstantiated. Metrics like ECE or Brier score are absent, and no OOD tests are included.
7. The ablation study is incomplete. It’s unclear whether improvements come from DUS itself, better teacher calibration, or the W-Clip combination.

**Minor Notes**
1. Figure 1 is too high-level; actual distribution plots or attention maps would be more informative.
2. The connection to Bayesian deep learning and epistemic uncertainty is missing. This literature is highly relevant.
3.  Many implementation details are missing. How is N(y;x) computed?
 While the limited seciton is included, it doesn't address the core limitation that this is only validated on BERT text classification

### Questions
1. Can you provide learning curves showing convergence for different methods?
2. What is the actual computational cost (FLOPs, wall-clock time) of DUS teacher training?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Calibrated Uncertainty Distillation (CUD), a response-based KD framework with two components: Difficulty-aware Uncertainty Shaping (DUS) that modifies the teacher using a focal-entropy objective, and Wrong-mass Clipping (W-Clip) that reallocates probability from an incorrect top-1 class to the ground-truth label under a budget. The method is motivated by a constraint-based view of calibrated targets and a projection idea, then implemented with the two heuristics.

### Strengths
1. Clear identification of overconfidence as a barrier to effective response-based KD and an attempt to encode difficulty-aware uncertainty in the teacher, not just in the student loss. The guiding conditions C1 and C2 make the design intent legible.


2. A unifying lens that frames distillation through constraints R1 and R2 and a projection program in equation (2), which could, in principle, connect KD with calibrated probabilistic targets.

### Weaknesses
1. he paper reformulates target calibration as a constrained projection problem, choosing a distance Dist and constraints R1, R2, then selecting the closest distribution to the teacher, see equation (2). However, the method never solves this optimization. The subsequent implementation bypasses the projection by applying hand-crafted rules (DUS and W-Clip). There is no existence, uniqueness, or characterization of the solution under any specific Dist, nor KKT analysis or proof that the heuristics approximate the solution. This disconnect weakens the theoretical core that the paper builds in Section 2.3 and 2.4.

2. R1 requires a semantics-preserving neighborhood (N(y;x)) and lower bounds on probability mass in that neighborhood. R2 introduces a set (W(x,y)) of incompatible classes. The paper does not instantiate either (N(y;x)) or (W(x,y)) anywhere in the core method, nor provide a constructive procedure to obtain them. Without operational definitions, R1 and R2 remain rhetorical, not algorithmic, and the guarantees they imply cannot be verified. 

3. W-Clip reallocates mass from the predicted top-1 class (k^*) to the ground-truth (y), which requires access to (y). Yet Section 3.3 claims that unlabeled examples naturally contribute only the KD term. There is no description of how W-Clip operates without labels, and Algorithm 1 explicitly branches on teacher correctness, which again requires (y). The semi-supervised narrative is not supported by the mechanics of W-Clip.  

4. The constraints R1 and R2 are linear in probabilities and define a convex feasible set on the simplex. With a strictly convex Bregman divergence like KL, the projection exists and is unique. The paper could exploit this to derive a closed-form or iterative solution with KKT multipliers. Instead, it replaces the projection by two heuristic steps, with no proof that the end result satisfies the constraints or lowers a proper calibration loss. This leaves the mathematical story incomplete. 

5. Please cite and discuss the ECCV 2024 paper “How to Train the Teacher Model for Effective Knowledge Distillation,” European Conference on Computer Vision, Cham, Springer Nature Switzerland, 2024. That work studies how to optimize teacher training so that the distilled signal is more effective, which is directly relevant to your teacher-shaping step. Please clarify similarities and differences

### Questions
See the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2
