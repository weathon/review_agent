# Perturb to Forget: Zero-Shot Machine Unlearning

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 0, 4, 6

## Abstract
Machine unlearning seeks to remove the influence of specific data from trained models, a requirement increasingly critical under modern privacy regulations. Yet most existing approaches either depend on costly retraining or require access to the original dataset, which may be unavailable or restricted. We propose Inversion-Guided Neuron Perturbation (IGNP), a zero-shot framework that performs unlearning entirely without the original data. IGNP begins by synthesizing class-representative samples through a model inversion-inspired process, enabling analysis of how different parameters encode forget and retain classes. By contrasting these sensitivities, IGNP identifies parameters that are especially critical for encoding the forget class, while being less influential for retain classes. This strategy erases targeted knowledge with precision while preserving model utility. Extensive experiments on multiple benchmarks demonstrate that IGNP achieves complete forgetting with minimal accuracy loss, outperforms state-of-the-art zero-shot and data-dependent baselines, and provides strong resistance to membership inference and inversion attacks. These results establish IGNP as a practical and efficient solution for data-free unlearning in compliance-driven machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a novel class unlearning approach, Inversion-Guided Neuron Perturbation (IGNP), without access to the original training data. With the observation that training data is often not disclosed, IGNP generates synthesizing class-representative data with fixed model parameters. This provides another data resource other than the original training data, but similarly influences the model. Synthesized data is analyzed by parameter sensitive algorithms to compute how each model parameter responds to forget class and retain class samples. The paper then proposes a selection algorithm to update the corresponding model parameters at target layers to remove the forget class.

### Strengths
1. The motivation is well justified. The paper addresses a significant and practical issue.
2. Comprehensive experiments with additional efforts to extend the method into different models. This provides a more comprehensive generalization of the method and emphasizes the importance of the finding. 
3. Detailed explanation in section 3 gives a clear roadmap and incentives on how and why the method is designed in the proposed way. 
Overall, the paper is insightful and well written. The paper provides a thorough explanation of the method and demonstrates its state-of-the-art performance empirically.

### Weaknesses
1. In section 3.2, the paper mentioned the forget class and the retain class samples are independently generated.However, there’s no further discussion and experiments on how the different ratio of synthetic samples between forgetting and retaining samples affects the unlearning performance. Based on section 3.3, sensitivity matrix S_forget and S_retrain depends on gradient computed on each class per sample. I am concerned that a disproportionate representation of the true training sample may affect the sensitivity ratio. 
2. I think more explanation is needed for the experimental results. Specifically, in Section 5.4, it is not clear why the MIA scores for the proposed method are lower than those for the retrained model. Based on the results in Table 4, the proposed method has an MIA accuracy of 0.02, which often indicates over-pruning or excessive forgetting, especially when compared to the MIA accuracy of the retrained model. I would appreciate further clarification on these results and a theoretical explanation for why the proposed method performs so well in terms of MIA.
3. Given the uncertainty in generating synthetic data, I will appreciate any discussion of mathematical proof on method consistency, specifically given that the synthetic data is randomly generated at first.

### Questions
I am curious how the target layer is chosen? Appendix A.3 provides detailed lists of the target layers. However, it seems like the paper only explains how the target parameters are selected, but not how the target layer is. Are we implicitly assuming the target layer is picked based on the target parameters?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper essentially presents a zero-shot version of SSD by constructing the retain and forget set via model inversion.

### Strengths
Using model inversion to turn non-zero-shot unlearning methods in zero-shot methods is a promising direction to overcome limitations of data availability.

### Weaknesses
I am seriously concerned about this paper in terms of scientific misconduct.
The entire method section does not cite a single prior work.
Figure 1 is clearly the SSD method with the added step 1 of creating the dataset via model inversion.

3.2 Model inversion does not cite any works. I suspect this is not a novel independent discovery.

3.3 passes of a prior ICLR paper as their own contribution. The exact same equations can be found in “Foster, J., Schoepf, S. and Brintrup, A., Loss-Free Machine Unlearning. In The Second Tiny Papers Track at ICLR 2024.”

3.4 is SSD with the addition of a hyperparameter search for alpha that I do not grasp. Dear authors, please elaborate on this selection process. Algorithm 1 is essentially SSD again.

The validity of the results are also questionable, as the authors do not perform hyperparameter tuning for the methods they benchmark against (see tables in the appendix). Also, no standard deviations etc.

Also, what is SSD model size scaler in Table 15? This is not present in the SSD paper and makes me suspect this work is heavily LLM generated as a related SSD paper has an auto scaler.

In conclusion: This could have been a great paper on how model inversion can help to turn any unlearning method into a zero-shot method but due to the above mentioned problems I can only recommend rejection due to serious breaches of scientific best practices.

### Questions
See weaknesses

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Inversion-Guided Neuron Perturbation (IGNP), a zero-shot machine unlearning method for classification models that operates without access to the original training data. 
The approach involves three main steps: (1) generating class-representative samples via an inversion process from the given forget class label, (2) computing sensitivity scores for model parameters, and (3) applying adaptive perturbations to weaken parameters associated with the forget class. The method is evaluated on image classification benchmarks, focusing on class unlearning scenarios where only the label to forget is provided. The authors claim advantages in efficiency and effectiveness compared to baselines like fine-tuning or other zero-shot methods.

### Strengths
- The method introduces an interesting use of inversion techniques to synthesize representative samples for unlearning, potentially useful in data-free settings.
- It demonstrates competitive performance on standard metrics like forget accuracy and retain accuracy in class unlearning tasks.
- The adaptive perturbation strategy based on sensitivity scores is a novel way to target forget-specific parameters without full retraining.

### Weaknesses
- The formulated unlearning problem is unconventional: instead of providing a forget set (data samples), only the class label is given as the unlearning request. While this setup is possible, in classification problems focused on class unlearning, a trivial solution exists when the label is known: simply zeroing out or significantly reducing the corresponding row vector in the classifier head would achieve similar effects. It is unclear what advantages the proposed method offers over such a simple baseline.

- the proposed method is inherently limited to classification tasks and cannot be extended to other unlearning scenarios (e.g. random forgetting) or non-classification tasks, which restricts its broader applicability.

- In Table 1, the fine-tuning baseline appears underperforming compared to reports in recent machine unlearning (MU) papers for ResNet-18 on CIFAR-10, where fine-tuning often yields forget accuracy of 0% and near-equivalent retain performance. This raises concerns about whether hyperparameters were selected to make the baseline appear weaker, potentially inflating the relative gains of IGNP.

- The Membership Inference Attack (MIA) evaluation lacks clarity on the specific setup. MIA in unlearning literature varies (e.g., confidence-based MIA-efficacy vs. MIA-privacy, or pairwise comparisons of prediction distributions across forget/retain/test sets)

- In Section 5.3, efficiency is claimed based on time costs, but the inclusion of an inversion process (which is computationally intensive) makes the reported speeds surprisingly low. It is unclear if the time for inversion-based data synthesis was excluded from these measurements.

- In Section 5.4's MIA results, the goal of class unlearning is typically to produce results similar to a retrained model. However, the reported MIA scores show substantial differences from the retrain baseline, 
which questions whether this can be considered a strong outcome for effective unlearning.

- As an ablation, it would be valuable to include results for a perturbation-based unlearning model using real forget data (instead of inversion-synthesized samples) to isolate and quantify the effect of the inversion step.

### Questions
- Could you provide comparisons to the trivial baseline of modifying the classifier head directly when the forget label is known? What unique benefits does IGNP offer in this setting?

- Please clarify the MIA setup: Which variant is used (e.g. test/forget/retain distribution comparisons)?

- For the importance scoring (Section 3.3), are there references for using squared partial derivatives? Would ablating with pruning-inspired criteria (e.g. Taylor importance criteria) change results?

- Can you provide ablation results comparing inversion-synthesized data vs. real forget data for perturbation? This seems essential to demonstrate the value of the zero-shot, data-free aspect.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes Inversion-Guided Neuron Perturbation (IGNP), a zero-shot (data-free) machine-unlearning method. IGNP (1) synthesizes class-representative pseudo-examples via a model-inversion-style optimization; (2) computes per-parameter sensitivity to the synthesized forget vs retain samples; (3) selects a small proportion of parameters (via an automatically calibrated threshold via binary search) that are disproportionately sensitive to the forget class; and (4) attenuates those parameters (multiplicative scaling) rather than zeroing them, with a designed scaling formula controlled by a perturbation strength λ. The authors evaluate IGNP on MNIST, SVHN, CIFAR-10 and CIFAR-100 across LeNet/ResNet9/ResNet18, comparing to retrain, finetune, SSD, GKT, UNSIR, etc., and report (i) forget-class accuracy near 0%, (ii) strong preservation of retain accuracy, (iii) robustness to membership inference and inversion attacks, and (iv) substantial runtime advantages over retraining. The paper contains ablations on coverage, λ, and shows continual unlearning (sequential erasure) capability.

### Strengths
1) Zero-shot unlearning is an important field of research which addresses the need for privacy compliance where original data is unavailable.

2) IGNP composes well-motivated components (inversion synthesis → sensitivity → calibrated perturbation). The binary-search calibration for thresholding coverage is an appealing practical touch that reduces manual tuning.

3) Paper evaluates across multiple datasets, architectures, and in sequential (continual) forgetting scenarios. The authors report outcomes of  effective forgetting (Df ≈ 0%), little loss in retained accuracy, resistance to MIAs, and favorable runtime. The ablations (λ, coverage, zeroing vs scaling) show the benefits of careful attenuation versus naive zeroing.

### Weaknesses
1) The paper lacks reporting of statistical variability (multiple seeds / runs / standard deviations). Given the stochasticity of inversion synthesis and binary search, the authors must report mean ± std over multiple independent runs (at least 3) for retain and forget accuracies and for MIA scores. This clarifies robustness and reproducibility.

2) The MIA results (e.g., reducing attack success from ~90% to ~0.02%) are impressive but surprising. The paper currently lacks sufficient methodological detail about the attack setup.

### Questions
1) Although overall Dr is preserved, table averages can hide specific failures. Provide per-class retain accuracies (before/after) and confusion matrices for tasks where retain performance degrades the most (especially CIFAR-100). Show whether some retained classes degrade systematically (e.g., semantically similar classes). This would reveal whether IGNP perturbs parameters that support multiple classes.

2) IGNP sets a target coverage interval (Cmin–Cmax ~ 4–5%) and uses binary search to find α. The manuscript argues proportion of perturbed parameters is stable, but provide empirical evidence (across datasets/architectures) that the optimal proportion is indeed stable (include plot or table). Explain how Cmin/Cmax were chosen and sensitivity to that choice; e.g., would 1% or 10% break things? Some plots are included in appendix but a concise summary and guidance are needed for practitioners

### Soundness
3

### Presentation
2

### Contribution
3
