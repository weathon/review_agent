# Robust Algorithmic Recourse Design Under Model Shifts

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 5

## Abstract
Algorithmic recourse offers users recommendations for actions that can help alter unfavorable outcomes in practical decision-making systems. Although many methods have been proposed to design easily implementable recourses, model updates or shifts may render previously generated recourses invalid. To assess the robustness of recourses against model shifts, we propose an uncertainty quantification method to calculate a theoretical upper-bound of the recourse invalidation rate for any counterfactual plan and any prediction model, without requiring distributional assumptions about the feature space. Furthermore, given the inherent trade-off between recourse cost and recourse robustness, users should be empowered to manage the implementation cost versus robustness trade-off. To this end, we propose a novel framework that leverages the derived invalidation rate bounds to generate model-agnostic recourses that satisfy the user's specified invalidation needs. Numerical results on multiple datasets demonstrate the effectiveness of the derived theoretical bounds and the efficacy of the proposed algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Algorithmic recourse, which suggests actions to alter unfavorable outcomes in decision-making systems, faces challenges due to model updates. A new uncertainty quantification method is introduced to assess recourse robustness against model shifts, offering a theoretical upper bound for recourse invalidation. Additionally, a novel framework helps users balance implementation cost and robustness by generating model-agnostic recourses based on the derived invalidation rate bounds, with promising results on various datasets.

### Strengths
1. The research problem make senses in realistic problems.
2. The upper bound is agnostic w.r.t. prior distribution (e.g., Gaussian), thus offers a more flexible approach.

### Weaknesses
1. My main concern is on whether the studied problem is novel enough. I admit that the problem of designing a robust post-hoc explanation problem is important for dynamic decision systems. However, such topic is not your contribution, as similar researches have emerged in recent top conferences in 2 years. 
2. Hence, as your contribution offers a more cost-efficient or flexible approach, I think the novelty or importance of your work is not well present. As you referred to before, current robust CF methods cannot handle efficient cost and robustness at the same time. However, I do not see any theoretical analysis in your method section regarding. this clarification, as your method section focuses on deriving the bound. Meanwhile, I think that it is important to theoretically clarify how your contributed method can achieve more efficient and robust CF against previous methods. 
3. Besides, I also have some minor questions. 
- Your robustness stands on the conformity score function, which depicts the variation of the predictors. However, the word robustness usually refers to the variation of the underlying distribution, or data. Although the variation of predicted distributions can be regarded as another measure to quantify the model shift, it is not clear whether the underlying data shifts or just the model shifts (e.g., new fine-tuning approaches). 
- Another direction for your paper is to consider more sophisticated description of how the data changes, i.e., considering strategic adaptation raised by population's incentives and the resulting predictors. I really think that considering the model shift or data shift in general is meaningless, especially in the era of LLMs. Well pre-trained models with a large amount of data can easilly beats over tricks towards distribution shift or model generalization. Hence, only specific topics on model shift or data shift, i.e., strategic adaptation, are still meaningful.

### Questions
As seen in Weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper aims to address the inconsistency in recourse generation: as predictive models often change over time, there is a concern about the reliability and robustness of the generated recommendations. The paper addresses this issue by proposing a method to generate model-agnostic recourses that are both robust to model shifts and have lower implementation costs. The challenges of measuring the robustness of a recourse under a shifted model and accommodating users' different tolerance levels for recourse invalidation are discussed. The paper introduces the concept of recourse invalidation rate and utilizes conformal predictive inference techniques to bound the invalidation rate. An extended alternating direction method of multipliers (ADMM) approach is proposed to efficiently find the minimal cost recourse that satisfies the invalidation rate constraint.

### Strengths
1. This paper studies an important problem in the literature, and it potential has high impact if carried out properly.

### Weaknesses
1. The authors do not clearly state the contributions of this paper in the introduction.
2. The authors do not provide a justification for why their methods work. In many cases, the model parameters are shifted because the underlying distribution of the dataset is changed. It is thus unclear why having access to the $D_{train}$ and $D_{calib}$ set can solve this distributional shift problem when these two sets have no predictive power of the future distributions.
3. The authors do not illustrate whether the bound $\hat L$ and $\hat U$ are practically sensible. I guess that $\hat L$ is trivially (close to 1) and $\hat U$ is conservative (much bigger than 1). I am not convinced that the construction of these values is informative and effective at dealing with model shifts.
4. The authors should discuss the privacy concerns of their method because their approach requires access to the dataset.
5. Section 4 is confusing to read. They contain mostly formulas with ad-hoc definitions of mathematical terms and have no discussion to provide insights/intuition.
6.   The numerical experiments do not show strong dominance against DiRRAc.

### Questions
My feeling is that the authors are over-complicating the problem. How about a simpler approach as follows:
Step 1: Get all empirical values of the score.
Step 2: Construct a kernel density estimator
Step 3: Formulate a robust version by perturbing the kernel density using some divergences or total variation.

I guess that the above three step can generate the same effect as what the authors want to do in this paper. But it is more interpretable and easier to understand.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Algorithmic recourse is the process of offering users recommendations for actions they can take to receive a positive classification from a machine learning model after they have received a negative classification. The focus of this work is on designing robust recourses, i.e. recommendations for improvement which are robust to shifts in the underlying predictive model being used to make decisions. The authors propose an uncertainty quantification method to upper-bound the invalidation rate (i.e. the probability the offered recourse is no longer valid under model shift) for a given pre-computed recourse. 

Using their proposed uncertainty quantification method (recourse invalidation rate), the authors design an algorithm to manage the trade-off between the cost of recourse (e.g. the "effort" required for a user to achieve this recourse) and the probability that the given recourse will be invalid under model shift. In particular for a user-specified invalidation rate, the algorithm aims to return a recourse with minimum cost such that the probability of the recourse being invalid is at most the user's given rate. The authors formulate this objective as a (non-convex) optimization problem, and numerically solve it using gradient-free optimization methods. 

The authors empirically validate their recourse invalidation bounds and find that the average empirical bounds are always tighter than their theoretical counterparts, sometimes by a significant margin. They also empirically evaluate the performance of their proposed algorithm and find that it generates more robust recourse solutions which are often easier to implement when compared to existing methods for recourse.

### Strengths
While the authors are not the first to study the robust algorithmic recourse problem, their results are novel within this space (to the best of my knowledge). In particular, their proposed uncertainty quantification method (recourse invalidation rate) is an intuitive measure of of uncertainty under model shifts, which does not require distributional assumptions on the feature space. 

The numerical results are also a welcome addition to the submission. In particular, the comparison between the empirical and theoretical bounds was helpful for gauging how tight the bounds of Theorem 12 and Theorem 13 are. Additionally, the numerical comparison between Algorithm 2 and several baselines in real-world data showed that the authors' proposed method often outperforms the relevant baselines.

### Weaknesses
I found the writing of this submission somewhat challenging to understand. In particular, Section 4 (Recourse Invalidation Estimation for a Given Recourse) reads as a laundry list of propositions, theorems, and lemmas with various definitions sprinkled in between. It was unclear to me which parts are the salient features, and which parts are there to aid in the understanding of more important results. I also found the inclusion of the comparisons to previous work in this section to be odd, as Section 2 (Background and Related Work) appears to be a more appropriate place for such comparisons. If the submission is accepted, I encourage the authors to rewrite Section 3 in a way which is more concise and highlights important results. 

Another weakness of the submission is the lack of any theoretical performance guarantees for Algorithm 2. (In fact, even the algorithm's runtime is not specified.) Even if theoretical performance guarantees cannot be obtained due to the non-convexity of the problem domain, it would have been nice to see a discussion on what exactly makes obtaining performance guarantees in this setting challenging. 

Finally, while not a major weakness, the authors' theoretical upper bounds on recourse invalidation are often significantly loose when compared to their empirical counterparts.

### Questions
What is the runtime of Algorithm 1?

What is the runtime of Algorithm 2? 

Does Algorithm 2 enjoy any non-trivial performance guarantees? In other words, is it possible to say something about the cost of the solution returned by Algorithm 2?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a new algorithmic recourse aiming for robustness against model shift. The method, named Probabilistic Invalidation-based Robust Recourse Generation (PiRR), is formulated with ideas from conformal prediction and solved with extended ADMM. The paper also proposes a new metric called invalidation rate, with two upper bounds. Then, three baseline recourse methods are evaluated demonstrate the power of these bounds. The effectiveness of PiRR is also compared against four other methods in term of the new metric.

### Strengths
- The paper contributes a novel approach and formulation to the field.
- Most of the paper is well-written, except for section 4.

### Weaknesses
- The paper is not following the conventional citing style.
- The paper did not present an evaluation of PiRR in the traditional cost-validity trade-off performance. This makes it hard to put the method in context of the current landscape of the field.
- Section 4 seems cramped and could be improved.

### Questions
- Assumption 3 seems too restrictive to be practical, since you need a bounded future perturbation. Can you elaborate on how is your formulation advantageous compared to previous works?
- Can you clarify the utility for each upper bounds presented in Theorem 12 and 13? Why are both presented and included in the evaluation?
- Based on section 2, the method in (Nguyen et al., 2022) seems to be highly related to this paper. Why was it left out from further discussion?
- The updated version of (Pawelczyk et al., 2022) is (Pawelczyk et al., 2023).
- Some other robust recourse baselines are listed below.

References

Tuan-Duy Nguyen, Ngoc Bui, Duy Nguyen, Man-Chung Yue, and Viet Anh Nguyen. Robust bayesian recourse. In proc. Uncertainty in Artificial Intelligence, pp. 1498–1508, Eindhoven, Netherlands, Aug. 2022.

Martin Pawelczyk, Teresa Datta, Johannes van-den Heuvel, Gjergji Kasneci, and Himabindu Lakkaraju. Algorithmic recourse in the face of noisy human responses. arXiv preprint arXiv:2203.06768, Oct. 2022.

Martin Pawelczyk, Teresa Datta, Johannes van-den-Heuvel, Gjergji Kasneci and Himabindu Lakkaraju. “Probabilistically Robust Recourse: Navigating the Trade-offs between Costs and Robustness in Algorithmic Recourse.” International Conference on Learning Representations (2023).

Victor Guyomard, Françoise Fessant, Thomas Guyet, Tassadit Bouadi, and Alexandre Termier. "Generating robust counterfactual explanations."  ECML/PKDD (2023).

Junqi Jiang, Jianglin Lan, Francesco Leofante, Antonio Rago, and Francesca Toni. "Provably Robust and Plausible Counterfactual Explanations for Neural Networks via Robust Optimisation." arXiv preprint arXiv:2309.12545 (2023).

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
