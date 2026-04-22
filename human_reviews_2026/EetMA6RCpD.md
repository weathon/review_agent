# OPC: One-Point-Contraction Unlearning Toward Deep Feature Forgetting

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Machine unlearning seeks to remove the influence of specific data or classes from trained models to meet privacy or legal requirements. However, existing methods often achieve only shallow forgetting: while outputs change, internal representations still retain enough information to reconstruct the forgotten data or behavior. We demonstrate this vulnerability via feature and data reconstruction attacks, showing that most unlearned features remain informative enough to recover both model performance and raw inputs from the forget set. To address this issue, we propose OPC (One-Point Contraction), a simple yet effective unlearning method that contracts the output representations of forget data toward the origin. By limiting representational capacity to a single point, OPC selectively erases feature-level information associated with the forget set. Empirical evaluations on image classification benchmarks show that OPC achieves strong unlearning efficacy and superior robustness against recovery and reconstruction attacks. We further extend OPC to generative diffusion models, validating its effectiveness in the context of conditional image generation. Applied to Stable Diffusion, OPC enables fine-grained removal of concept-level information, achieving state-of-the-art performance in generative unlearning. These results demonstrate OPC’s broad applicability and its potential for precise, task-aware control of forgetting across both discriminative and generative domains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper first identifies the issue of shallow unlearning in most existing methods by conducting a recovery attack. It then proposes One-Point Contraction (OPC), which achieves deep unlearning by normalizing the representation of the forgotten data toward the origin point of the feature space.

### Strengths
1. The paper is clearly structured and well-presented. 

2. Demonstrates the connection between uncertainty and classification entropy using Theorem 3.1, which clearly motivates the use of the proposed normalization term for unlearning.

3. The extended empirical results, including classification and generative diffusion tasks, consistently show that the proposed method outperforms existing methods in terms of data privacy concerns.

### Weaknesses
1. Efficiency and Ablation on OPC Updates: While OPC achieves strong forgetting performance, its efficiency in class unlearning settings is generally lower than other baselines, notably BT, which attains similar results with fewer training iterations.  The paper attributes this to differences in training epochs, yet no ablation or scaling analysis is provided. It would strengthen the work to include systematic ablations showing how OPC’s efficiency and convergence vary with the sizes of $D_f$ and $D_r$, as well as the number of OPC update steps. Such results would clarify whether the slower convergence stems from the contraction mechanism itself or merely from hyperparameter selection.

2. The authors depart from the common unlearning evaluation protocol that compares results against a retrained model as a gold standard. This omission is understandable given OPC’s distinct behavior—collapsing $D_f$ features to the origin rather than erasing them through retraining—but it raises potential generalization and safety concerns. Since OPC enforces degraded responses on $D_f$, the model’s behavior might deviate from retraining even on semantically related test inputs, potentially violating the expected equivalence criterion in unlearning (i.e., consistent predictions with retraining on$D_r$). Discussing these implications would enhance the reader’s understanding of the trade-off between strong forgetting and functional consistency.

3. One stated motivation is to treat $D_f$ as OOD data. However, the paper does not clearly define the degree or notion of OOD induced by OPC. For example, in CIFAR-10 unlearning, should $D_f$ representations behave like pure noise, or like samples from the distribution of SVHN? Making $D_f$ too distinct from the overall data manifold might create new security risks, such as detectable distributional signatures or abnormal uncertainty patterns that could reveal which classes were unlearned. A more explicit discussion of this trade-off between effective forgetting and distributional detectability would be valuable.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper claims that existing machine unlearning (MU) methods present shallow forgetting, i.e., while output changes, internal representations still contain enough information for reconstructing the forgotten data. Therefore, this paper proposes the MU method, OPC (One-Point Contraction), to achieve deep forgetting by contracting the representations w.r.t. to the forgetting data toward the origin. To achieve this, OPC introduces an additional term that minimizes the $\ell_2$ norm of the logits, with small norms implying high uncertainty, similar to OOD samples.
Experiments on CIFAR10, SVHN with ResNet, and CIFAR10 with DDPM, as well as SD models, show the effectiveness of the proposed method.

### Strengths
- The paper is easy to follow, and the motivation is clear.
- The proposed method is simple yet effective.
- Extensive experiments are conducted.

### Weaknesses
- The main concern is about the claim that forgetting data should be treated as unseen (OOD) samples. As for OOD, low norms could be due to the samples lying outside the training distribution, while for unlearning, forgetting data belongs to the training distribution and could be highly entangled with retain data, especially for sample-wise unlearning settings.
- The theoretical part is also based on the output (logits); it cannot provide the guarantee that latent representations no longer contain information about the forgetting data.

### Questions
- It would be better to justify the claim about treating forgetting data as OOD samples.
- Some visualization for the feature distribution, such as t-SNE, might help to confirm that OPC removes information.
- Line 142 mentioned low-norm features, but Eq.(2) considers minimizing the norm of the logits, which is confusing.
- Lines 397-406, could the author provide more descriptions about the auxiliary linear classifier, in-domain classification, and cross-domain classification?
- The loss function used consists of different terms. Is there any need to add a coefficient for balance?


If the authors can address the concerns and provide clearer answers to the points, I would be inclined to raise the score.

### Soundness
2

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
2

### Summary
This paper raises an important issue in the existing machine unlearning works that they often achive only shallow forgetting (the internal representations still retain enough information to reconstruct the forgotten data or behavior). The authors propose an unlearning framework POC to tackle this issue. The key idea is to perform a local contraction around the target point in the feature or parameter space, rather than retraining the model or applying global forgetting methods. The authors claim that OPC achieves faster forgetting with minimal performance degradation on the remaining data. Experiments are conducted on several benchmarks with multiple baselines. The results show that OPC achieves competitive or superior forgetting efficacy while maintaining test accuracy and low computational cost.

### Strengths
1. The proposed method builds on a clear geometric intuition: contract model behavior locally around the target sample. This is more interpretable than gradient-matching or complex optimization-based unlearning. 
2. The method seems computationally light compared to retraining-based or iterative gradient alignment approaches, making it more practical for large-scale use.
3. The presentation of the paper is good, that it is well-written and easy to follow.

### Weaknesses
1. Lack of justification for design choices: the motivation for ``contraction'' as a forgetting mechanism is qualitatively presented, but lacks empirical argument why it is the most suitable one for unlearning. Providing such results would help understand this mechanism.
2. About ablation study/sensitivity analysis: the choice of contraction strength, feature space vs parameter space contraction, and number of steps are not sufficiently justified. 
3. About the experiments: the paper seems lack experiments on class-level unlearning, which would enhance the solidness of the paper if provided.
4. The evaluation mostly relies on test accuracy and membership inference. There is limited discussion of whether OPC truly removes the semantic influence of the target sample. Since the paper claims that existing methods often achieve only shallow forgetting, it is important to provide deep analysis on how ``deep'' the forgetting achieved by the proposed method.

### Questions
1. What property makes contraction particularly suitable for unlearning compared to other local perturbation strategies? Have the authors tried other strategies? 
2. Could the authors provide empirical evidence supporting that OPC performs deeper forgetting than other existing methods? 
3. How sensitive is the forgetting performance to the contraction coefficient and the number of contraction iterations? 
4. How sensitive is the method to the contraction strength, the number of contraction steps, or the choice of feature versus parameter space? Are there any guidelines for selecting these hyperparameters?
5. Have the authors considered testing on class-level unlearning?

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
5

### Summary
This paper introduces "deep feature forgetting" as a desirable property for machine unlearning, arguing that existing methods perform "shallow forgetting" by only modifying model outputs while leaving internal representations vulnerable. To achieve deep forgetting, the paper proposes One-Point Contraction (OPC), a method that adds a regularization term to the unlearning objective, aiming to contract the feature representations of the forget set to the origin.

### Strengths
1. The concept (deep feature forgetting and shallow forgetting) is important for MU.

### Weaknesses
1. A Fundamental and Disqualifying Methodological Flaw: There is a critical disconnect between the paper's stated goal and its actual implementation. The paper claims to achieve "deep feature forgetting" by contracting feature representations f_θ(x) to the origin. However, the proposed loss function (Eq. 1) minimizes the L2 norm of the logits m_θ(x), where m_θ = g_θ ◦ f_θ. Minimizing the logit norm does not guarantee that the feature norm is minimized. A model can easily achieve a small logit norm by learning to zero-out the weights of the final classifier layer g_θ for forget data, while leaving the "deep features" f_θ(x) largely unchanged. This is the very definition of the "shallow forgetting" the paper claims to solve. This fatal flaw invalidates the entire technical premise of the paper. The authors' own experiment in Appendix D.2 ("Training-Free Unlearning"), which shows that class unlearning can be achieved by only modifying the prediction head, ironically serves as evidence for this failure mode.
2. The concept of deep forgetting proposed in this paper is also presented in GS-LoRA, and the authors should compare it with this method.
3. The technical contribution is limited. The novelty of the proposed method, OPC, is also extremely low. The core idea is to regularize the model to produce low-confidence predictions on forgotten data by minimizing an output norm. This is a direct application of well-known principles from OOD detection. Simply applying this standard regularization technique to unlearning does not constitute a significant conceptual contribution.
4. The experiments are sufficient (Table 1). The author should compare with more SOTA and recent methods.
5. Circular Attack Model: The main evidence for OPC's superiority is its robustness to the "Feature Mapping (FM) recovery attack" (Section 4.3). This is an exercise in circular reasoning. OPC is designed to collapse forget features to the origin (a vector of zeros). The attack tries to find a linear map W to recover original features. Of course W * 0 = 0, so no information can be recovered. The method is built to be invincible to this specific, simple attack. This does not prove its general robustness; it merely confirms the method does what it was told.
6. Misinterpretation of CKA: The authors present low CKA scores as direct evidence of "deep forgetting." However, for a method that maps all forget features to a single point (the origin), the resulting feature set has zero variance, which trivially leads to a CKA score of or near zero. This is a direct, mathematical consequence of the OPC objective, not an independent verification of its forgetting quality.
7. Ambiguous and Potentially Unfair Experimental Protocol: The authors state they "do not prematurely stop unlearning" (line 196), which is a departure from standard protocols where efficiency is a key concern. The varying and often large number of epochs used for different methods (Tables C.1, C.2) suggests an inconsistent and potentially unfair tuning effort, where OPC may have been optimized more heavily than the baselines.

### Questions
NA

### Soundness
1

### Presentation
2

### Contribution
1
