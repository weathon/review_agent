# Fairness-Aware Multi-view Evidential Learning with Adaptive Prior

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Multi-view evidential learning harnesses diverse data sources to improve prediction performance and provide reliable uncertainty estimates. Recent advances have primarily focused on optimizing evidence fusion strategies, assuming that the evidence extracted from each view is naturally reliable for downstream integration. However, our empirical analysis reveals that samples tend to be assigned biased evidence to support data-rich classes, thereby rendering unfair uncertainty estimations. This motivates us to delve into a new Biased Evidential Multi-view Learning (BEML) problem. To this end, we propose Fairness-Aware Multi-view Evidential Learning (FAML) method to rectify biased evidence learning. Specifically, FAML introduces the training-trajectory-based adaptive prior into the construction of Dirichlet parameters, flexibly calibrating the initial support evidence assigned to each class during training. Furthermore, we incorporate a fairness constraint as a regularization term to alleviate bias in the evidence. In the multi-view fusion stage, we propose an opinion alignment mechanism to mitigate view-specific bias across views, thereby encouraging the integration of consistent and mutually supportive evidence. Theoretical analysis shows that FAML effectively achieves less biased evidence allocation. Extensive experiments on real-world multi-view datasets demonstrate the superiority of our FAML, in terms of prediction performance and uncertainty estimation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses a critical bias in multi-view evidential learning: existing methods assume reliable view-specific evidence learning, but real-world data shows samples allocate more evidence to data-rich classes, harming uncertainty estimation reliability. To solve this newly defined Biased Evidential Multi-view Learning (BEML) problem, it proposes the Fairness-Aware Multi-view Evidential Learning (FAML) framework, which integrates three key components: a training trajectory-based adaptive prior (to calibrate bias), a class-wise evidence variance fairness constraint (to balance allocation), and an opinion alignment mechanism (to reduce cross-view bias). Theoretical analysis confirms FAML enhances evidence learning fairness, while experiments on 6 real-world datasets demonstrate FAML outperforms state-of-the-art methods in balanced evidence distribution, prediction performance, and uncertainty estimation reliability. The work also contributes by highlighting this implicit unfairness, proving the adaptive prior expands minority class evidence margins (with improved generalization error bounds), and validating FAML’s superiority.

### Strengths
-  The paper acutely captures a neglected yet prevalent issue in multi-view evidential learning: implicit unfairness in evidence allocation caused by differences in class data volume, which undermines the reliability of uncertainty estimation. This finding directly challenges the flawed assumption of existing methods that "view-specific evidence learning is inherently reliable," clarifies key optimization directions for future research, and reflects a deep understanding of practical pain points in the field.
- The proposed FAML method forms a closed-loop optimization from three dimensions, with strong targeting and novel design:
Adaptive Prior: Dynamically adjusted based on training trajectories, effectively calibrating evidence learning bias caused by class imbalance.
Fairness Constraint: Directly promotes balanced evidence allocation across different classes through class-wise evidence variance control.
Opinion Alignment Mechanism: Reduces view-specific bias during multi-view fusion, ensuring the integrated evidence is consistent and mutually supportive.
- The paper features both solid theoretical foundations and sufficient experimental validation, enhancing the persuasiveness of its research conclusions. First, it proves through derivation that the adaptive prior can expand the evidence margin for minority classes and provides an improved factor for the generalization error bound, offering mathematical support for the method’s effectiveness. 
-  Extensive experiments on 6 real-world multi-view datasets not only verify FAML’s advantages in prediction performance but also demonstrate its ability to achieve fairer and more reliable uncertainty estimation, ensuring the universality of the results.

### Weaknesses
- The paper does not verify FAML’s performance under input noise (e.g., Gaussian noise on view features like GIST/HOG in Scene15), leaving unclear if its evidence fairness and uncertainty estimation hold for imperfect real-world data
- Though tested on datasets with 2-6 views, it lacks analysis on how view count changes (e.g., reducing Caltech-101’s 6 views to 3) affect FAML’s fairness (FD) and opinion alignment effectiveness

### Questions
- The paper updates the adaptive prior every 5 epochs starting from the 20th training epoch. Was there any preliminary experiment to confirm that a 5-epoch update interval is more suitable for maintaining training stability compared to other intervals (e.g., 3 or 10 epochs)?
- When defining the fairness degree (FD) based on class-wise evidence variance, did you observe how FD changes dynamically across training epochs (e.g., whether it drops faster in early or late stages)? And does this change trend correlate with the gradual increase of λ (the balancing coefficient) in the loss function?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel and important problem: Biased Evidential Multi-view Learning (BEML). The authors demonstrate through empirical analysis that existing multi-view evidential learning methods tend to allocate more evidence to data-rich (majority) classes, leading to biased and unreliable uncertainty estimates for data-poor (minority) classes.

### Strengths
To address this, the paper proposes a Fairness-Aware Multi-view Evidential Learning (FAML) framework. The method has three core components:

An Adaptive Prior based on training trajectories, which dynamically adjusts the Dirichlet prior to provide more support to classes that are under-represented or poorly performing.

An explicit Fairness Constraint, which penalizes high variance in evidence allocation across different classes, encouraging a more balanced evidence distribution.

An Opinion Alignment mechanism, which minimizes the dissonance between opinions from different views during fusion to mitigate view-specific biases.

The authors provide strong theoretical grounding for their adaptive prior, proving that it increases the evidence margin for minority classes and tightens the generalization error bound. Comprehensive experiments on six real-world datasets show that FAML significantly outperforms state-of-the-art methods in terms of accuracy (especially for tail classes), calibration error (ECE), and uncertainty reliability (AUROC, FPR-95).

### Weaknesses
While this is an excellent paper, the following minor revisions could further strengthen its clarity and impact:

Strengthen the "Related Work" on Fairness and Subpopulation Shift: The core problem FAML solves—evidence bias due to class imbalance—is deeply connected to the broader fields of AI fairness and subpopulation shift. To better position the paper's contribution, the "Related Work" section should be expanded to include and discuss key works from this area. For instance, the authors should cite foundational work like GroupDRO (Sagawa et al., 2019), which formalized the goal of improving worst-group generalization in subpopulation shifts. More importantly, citing a paper like UMIX (Han et al., 2022), which explicitly links Uncertainty-Aware methods to solving Subpopulation Shift, would be highly relevant. Discussing these papers would allow the authors to clearly articulate FAML's unique contribution: while GroupDRO tackles the problem via the loss function and UMIX uses uncertainty to guide data augmentation, FAML introduces a novel approach by manipulating the evidential learning framework itself (via adaptive priors and fusion) to achieve fairness in a multi-view setting.

Intuition for the Adaptive Prior (Eq. 5): The paper should more explicitly state the key intuition behind Equation 5. The current description is accurate but subtle. The authors should clearly emphasize that this formula creates an inverse relationship: the worse a class performs (i.e., the fewer samples are correctly classified, the smaller the denominator), the larger the adaptive prior (Beta_k) becomes. This "compensatory" mechanism is the core of the idea and should be stated plainly.

Explicit Formulation of L_fc and mu schedule: In Section 3.2.3, the paper introduces the fairness loss L_fc, stating it is based on Definition 1. For absolute clarity, the paper should explicitly write out the final loss term (e.g., L_fc = Var(...)). Furthermore, the balancing coefficient mu is described as "gradually changing from 0 to 1." Please specify the exact schedule used (e.g., linear, exponential) to enhance reproducibility.

Motivation for Dissonance Degree (Eq. 10): The "Dissonance Degree" used for opinion alignment is a novel metric. The authors should add a brief sentence justifying this specific choice (sum of absolute differences in variance) over other, more traditional divergence measures (e.g., KL or JS divergence on the Dirichlet means/probabilities).

### Questions
As shown in weaknesses.

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
5

### Summary
The paper highlights an overlooked bias in multi-view evidential learning, where evidence allocation tends to favor data-rich classes, leading to unreliable uncertainty estimates. Instead of relying on fixed uniform priors like traditional EDL methods, it adaptively adjusts class priors based on training trajectories in a principled manner. Additionally, a fairness constraint on class-wise evidence allocation and an opinion-alignment regularization across views ensure the consistent allocation of evidence across views. Experiments on six multi-view datasets demonstrate superior region accuracy and ECE compared to baselines, with ablation studies confirming the contributions of each component.

### Strengths
1. This paper is well organized, and the proposed methodology is enlightening.

2. The motivation behind the paper is clear, and the theoretical analysis is complete.

3. The proposed method offers novel insights, particularly in using training trajectories to adjust class priors, thereby mitigating view-specific bias throughout the multi-view fusion process

4. The proposed method shows a clear performance improvement in a series of experiments.

### Weaknesses
1. In this paper, the notion of fairness seems to focus on balancing the evidence allocation across different classes, rather than addressing fairness in terms of sensitive attributes like race or gender in a broader sense.

2. Is this approach intended as a general framework? Can other trusted multi-view fusion methods also adopt similar strategies to improve model performance even on balanced datasets?

3. Some implementation details seem to be missing. For instance: How does the hyper-parameter $\mu$ change during training? and How is the metric ECE calculated in your experiments.

If the authors can address my questions, I am willing to increase my score.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the issue of unreliable uncertainty estimation in multi-view evidence learning, which stems from biased evidence collection. The authors propose a framework called Fairness-Aware Multi-view Evidential Learning. This method uses a training-trajectory-based adaptive prior to calibrate the Dirichlet parameters, aiming to mitigate the evidence bias. The approach includes theoretical guarantees and is validated through experiments on six real-world datasets to demonstrate its performance.

### Strengths
1. The paper has a clear motivation and effectively solves the problems of biased evidence multi-view learning.

2. The paper offers a clear and well-grounded theoretical analysis that connects the adaptive prior design to margin theory, helping explain why the proposed approach could improve model's generalization.

3. The comparison experiments are comprehensive, including six representative multi-view datasets.

### Weaknesses
1. This work proposes an EDL-based multi-view classification method. However, the literature review for existing EDL-based multi-view methods is insufficient. The authors should provide a more comprehensive discussion of related work in this specific domain, such as, but not limited to, [1, 2].

2. The text in Figure 1 is too small, and there is no explanation of what the points, lines, and colors in the figure represent or why it is imbalanced. The blue in the legend of Figure 1 is different from that in the figure.

3. The phrase "most existing studies generally assume that view-specific evidence learning is inherently reliable" is ambiguous, especially the use of the word "reliable." What you want to convey is that the evidence learned from this view is unreliable, but it may lead readers to misunderstand that the view itself is unreliable.

4. Punctuation is also part of the formulas and needs to be added.

[1] Enhancing Testing-Time Robustness for Trusted Multi-View Classification in the Wild. CVPR 2025.

[2] Trusted multi-view classification with expert knowledge constraints. ICML 2025.

### Questions
1. What role did the opinion alignment play in promoting fairness? It seems irrelevant to fairness?

2. The paper positions fairness as a key design goal, yet there are no reported quantitative fairness evaluation metrics to determine this. Could the authors provide explicit metrics or quantitative analysis to support the fairness claims?

3. The degraded baseline model is introduced for the visualization. This baseline is described as FAML without the fairness-aware components. It is unclear if this is a re-run of an existing baseline (e.g. TMC) ? What are the hyper-parameters of the compared baseline?

4. In subjective logic, formulas rely on fixed priors to compute belief mass and uncertainty. When the prior becomes adaptive, do these formulations still hold as originally defined? Are the theoretical assumptions of subjective logic still satisfied after introducing the adaptive prior?

5. Could you discuss the robustness of FAML in the presence of potential noisy views. How does the adaptive prior perform in such scenarios?

6. Check for all possible typos in the manuscript. e.g., "bias exhibit view-specific pattern" should be "bias exhibits view-specific pattern" in Line 20.

### Soundness
3

### Presentation
3

### Contribution
3
