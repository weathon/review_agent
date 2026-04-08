## Human Reviewer 1

### Summary
Motivated by the information bottleneck principle, the paper proposes concepts’ information bottleneck model (CIBM), which uses an information bottleneck regularizer (IB regularizer) for training concept bottleneck models (CBMs) to mitigate the problem of concept leakage while preserving prediction accuracy. Based on theoretical observations, the authors give two types of regularizers, bounded and estimator based, for training CBMs. Through experiments, it is shown that IB regularizers maintain or improve prediction performance while mitigating concept leakage. Moreover, using CIBM, the authors propose a measure to quantify the quality of concept sets.

### Strengths
1. The motivation for using information bottleneck principle to tackle concept leakage looks valid and interesting.
2. The experiment results suggest that IB regularizers are indeed helpful to mitigate concept leakage while preserving final prediction accuracy.

### Weaknesses
1. As authors frame CIBM as theoretically principled integration of IB principle to CBM, examining the validity of the estimators is important. However, theoretical analysis for deriving the bounded CIB (section 3.1, section 3.2, appendix B) seems to have fatal errors.
- The paper claims to upper bound the L_{UB-CIB}, but in eq (A.1), -\beta H(C|X) should be +\beta H(C|X), and in equations from (A.2a) to (A.4l), the minus sign should be added to all the right hand sides (definition of entropy). These errors seem to make the CIB upper bound in eq (3) or (A.5) completely wrong. 
- Also, due to above observations, equation (6) seems to be wrong.
2. If the goal of the paper is to address concept leakage, comparing with previous works that address concept leakage would be a valuable addition. Also, these works do not seem to be addressed properly in the related work section.
3. At section 4.3, using only random intervention seems insufficient. The authors should include experiment that uses more effective intervention strategies explored in [1] such as UCP.

Reference

[1] Shin et al., A Closer Look at the Intervention Procedure of Concept Bottleneck Models, ICML 2023.

### Questions
1. Can the usage of two estimators (bounded CIB and estimator-based CIB) be theoretically justified, given that the theoretical analysis seems to be wrong?
2. How does the IB-regularizer compare against other methods that mitigate leakage?

### Soundness
1

### Presentation
3

### Contribution
1

### Rating
2

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes to enhance the interpretability of Concept Bottleneck Models (CBMs) by applying an explicit information-bottleneck (IB) regularizer to the concept layer. It argues that traditional CBMs suffer from concept leakage (i.e., unintended information flowing into concept activations) and that minimizing I(X;C) while maximizing I(C;Y) and I(Z;C) yields cleaner and more faithful concept representations. They introduce two practical methods, integrate them into multiple CBM families and evaluate on three benchmark datasets. Experimental results suggest reductions in leakage metrics, modest improvements in classification accuracy, and more reliable concept-level interventions.

### Strengths
1. The paper improve target accuracy, reduce concept leakage, and enhance intervention effectiveness.
2. The information-theoretic framing is conceptually sound: targeting I(X;C) aligns well with the desired minimal-sufficient concept representation.
3. The proposed methods are modular and broadly applicable: the regularizer is applied across many CBM variants without altering the architecture radically.
4. Empirical evaluations are relatively comprehensive: multiple methods, datasets, metrics (leakage, concept accuracy, task accuracy, intervention AUC) are reported.

### Weaknesses
1. Stability of empirical results.
The empirical performance of different regularizers across datasets is not consistently positive.
In Table 1, certain variants of the proposed CIB regularizers improve accuracy or interpretability metrics on some datasets, but degrade or show negligible effects on others (e.g., AwA2).
This inconsistency suggests that the approach may be sensitive to dataset characteristics or model initialization, raising questions about its stability and generalization.
2. Missing theoretical justification of guaranteed improvement.
The paper introduces the modified information-bottleneck objective
$I(Z;C) + I(C;Y) - \beta I(X;C)$,
yet provides no theoretical guarantee, such as an error-bound or generalization-bound proof, showing that adding this IB term will necessarily improve the performance or faithfulness of existing CBMs.
3. Overstated claims: The paper suggests that CIBMs “close the accuracy gap to black-box models without sacrificing
interpretability,” yet in many experiments the black-box baseline still significantly outperforms the IB-regularized CBMs. The practical significance of the interpretability-performance trade-off remains underexplored.

### Questions
1. Could the authors provide more detailed explanation or proof of under what conditions minimizing I(X;C) while maximizing I(C;Y) ensures minimal-sufficient concept representations (i.e., no leakage)? Are there assumptions (e.g., about concept annotation completeness, model capacity, independence among concepts) required for the theory to hold?
2. In Table 1, for certain model-dataset combinations the accuracy difference between vanilla CBM and CBM+IBB/IBE is very small (e.g., <0.5%) and within one standard deviation. 
3. In Eq. (5) and (7), the paper includes an entropy term H(C).
How is this quantity computed or approximated when C is continuous or high-dimensional?
Does this require batch-level estimation or additional variance reduction techniques?
4. What is the computational overhead of including IB regularization, in terms of runtime or memory?
Have the authors observed any optimization instability due to the extra MI term?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
In this work, the authors propose CIBMs which utilize an Information Bottleneck regularizer on the concept layer to reduce concept leakage for a base CBM. Evaluating their approach on different CBMs (CBM, CEM, ProbCBM, AR-CBM), the authors show strong accuracy, concept leakage, and intervention capabilities.

### Strengths
- Concept leakage is an important problem within CBMs, undermining faithfulness and intervention guarantees. The authors present a simple approach for mitigating leakage, utilizing an Information Bottleneck regularizer.
- Experimental details are verbose and the approach can be easily attached to prior works.

### Weaknesses
- Figure 3 does not really convey any information. The expanded results in the Appendix are useful but perhaps a table at different points along the x-axis would help for the main text?
- Accuracy and intervention results both show minimal improvement. Concept leakage results seem like the strongest result, but are not explored in much detail. What does the improved OIS/NIS mean downstream if the intervention performance is not improved? Is there an interpretation of these two metrics that better quantifies the improvement of CIBMs?

### Questions
- What dataset is Table 2 for? Why is this run on only one dataset?
- use beta=.5 for all experiments?
- "We also compare against more recent CBM variants such as ... intervention-aware CEM (Espinosa Zarlenga et al., 2022)" - is this CEM or IntCEM [1]? Why evaluate on one but not both?
- The motivation for the selected baselines in general is unclear, why is more recent work looking at intervention/leakage not included?
- Some typos throughout the main text and appendix.

[1] Espinosa Zarlenga, Mateo, et al. "Learning to receive help: Intervention-aware concept embedding models." Advances in Neural Information Processing Systems 36 (2023): 37849-37875.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper introduces Concept Information Bottleneck Models, which incorporate a loss term aimed at reducing concept leakage in Concept Bottleneck Models. The objective is to maximise task-relevant information encoded in the concepts while suppressing task-irrelevant information present in the input X. The proposed regulariser is applied on top of standard CBM-like architectures. Through a series of experiments, the authors demonstrate that this additional term mitigates leakage.

### Strengths
The authors address a key limitation of CBMs (concept leakage) through a loss-based approach rather than architectural modifications. This design choice enhances the applicability of the proposed method, allowing it to be integrated with existing CBM variants. Notably, the models most susceptible to leakage (e.g., CEM) show the greatest improvement, further validating the effectiveness of the approach. The paper presents extensive experiments on standard benchmarks, applying the proposed regularisation to several widely used CBM-like models. Moreover, the paper is well written and easy to follow.

### Weaknesses
- The authors evaluate their method on datasets where the set of concepts is extensive and highly representative of both the task and the concept space. However, it would be interesting to assess how the approach performs in scenarios where some degree of leakage is necessary to solve the task. Such a setting could be simulated by systematically removing a subset of concepts (e.g., half) from the training set. This would better reflect real-world conditions, where annotating a complete set of concepts is often infeasible.
- The proposed metrics (AUC and NAUC) do not appear to represent a clear novelty. They essentially quantify what prior works have already assessed qualitatively (CBM responsiveness to concept interventions) by expressing it numerically rather than through visual plots.
- The claim of improved accuracy (Section 4.1) seems somewhat overstated. The observed gains are marginal and, arguably, accuracy should not be the primary focus of this work. It would be more appropriate to emphasise that the proposed regularisation maintains task performance while effectively reducing concept leakage, rather than suggesting accuracy improvements.

### Questions
- It would be interesting to explore how the proposed method behaves when the concept set is substantially reduced or incomplete (i.e., in cases where some degree of concept leakage becomes necessary to achieve good task performance). Such a setting would better approximate real-world conditions, where not all relevant concepts can be fully annotated.
- While the paper focuses on minimising information leakage from the input to the output, it does not address intra-concept leakage. Prior works [1,2] have shown that certain concepts can be predicted from others, which may undermine the independence assumption among concepts. It would be valuable to investigate whether the proposed method can also mitigate this form of leakage, thereby further improving intervention responsiveness.

---------------------------

[1] Gabriele Dominici, Pietro Barbiero, Mateo Espinosa Zarlenga, Alberto Termine, Martin Gjoreski, Giuseppe Marra, & Marc Langheinrich (2025). Causal Concept Graph Models: Beyond Causal Opacity in Deep Learning. In The Thirteenth International Conference on Learning Representations.

[2] Moritz Vandenhirtz, Sonia Laguna, Ričards Marcinkevi\vcs, & Julia E Vogt (2024). Stochastic Concept Bottleneck Models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4