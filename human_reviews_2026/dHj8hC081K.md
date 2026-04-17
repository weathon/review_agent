# Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Mainstream Test-Time Adaptation (TTA) methods for adapting vision-language models, e.g., CLIP, typically rely on Shannon Entropy (SE) at test time to measure prediction uncertainty and inconsistency. However, since CLIP has a built-in bias from pretraining on highly imbalanced web-crawled data, SE inevitably results in producing biased estimates of uncertainty entropy. To address this issue, we notably find and demonstrate that Tsallis Entropy (TE), a generalized form of SE, is naturally suited for characterizing biased distributions by introducing a non-extensive parameter q, with the performance of SE serving as a lower bound for TE. Building upon this, we generalize TE into Adaptive Debiasing Tsallis Entropy (ADTE) for TTA, customizing a class-specific parameter q^l derived by normalizing the estimated label bias from continuously incoming test instances, for each category. This adaptive approach allows ADTE, even without hyperparameter tuning required by TE, to accurately select high-confidence views and seamlessly integrate with label adjustment strategy to enhance adaptation. Besides, our investigation reveals that both TE and ADTE can serve as direct, advanced alternatives to SE in TTA, without any other modifications. Experimental results show that ADTE outperforms state-of-the-art methods on ImageNet and its five variants, and achieves the highest average performance on 10 cross-domain benchmarks, regardless of the model architecture or text prompts used. Our code is available at https://anonymous.4open.science/r/TTA-Entropy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes replacing Shannon entropy (SE) with Tsallis entropy (TE) for selecting confident views in test-time adaptation (TTA), arguing TE better handles prediction bias in vision-language models. It further introduces Adaptive Debiasing Tsallis Entropy (ADTE), which assigns a class-specific parameter $q_l$, derived from an estimated class-prior/bias via a memory bank and pseudo-labels, then picks views with the smallest ADTE to aggregate predictions.

### Strengths
1. The paper offers a fresh perspective by replacing Shannon entropy with Tsallis entropy and further adapting a class-specific parameter $q_l$, giving a tunable lens on confidence that can better reflect skewed priors and domain shift during test time.

1. The entropy perspective is modality-agnostic and should, in principle, transfer to other test-time settings (e.g., VLMs, audio, or tabular) where confidence shaping under shift is crucial.

1. The experiments span multiple datasets and architectures, where ADTE outperforms most of them.

### Weaknesses
1. Table 3 show that “w/o ADTE and w/o LA” already achieves high accuracy; many reported deltas are modest ($\sim$0.5–2%); without confidence intervals or paired significance tests, it is unclear whether improvements are statistically or practically meaningful.

1. The core of ADTE is a min-max mapping from an estimated bias vector $\tilde{p}$ into $q_l \in [\alpha, \beta]$ (the paper uses $[0.01, 0.9]$) via Eq.~(13). There’s no theoretical link that this linear rescaling is optimal (or even monotone-calibrated) for view selection risk; it feels ad-hoc and dataset-dependent. Sensitivity is only probed by sweeping the lower bound, not the mapping itself.

1. Estimating class priors from pseudo-labels assumes pseudo-labels are reliable; errors in early iterations can feed back into the prior and $q_l$, potentially harming tail classes the method aims to help.

1. The paper did not specify the details of augmentation strategies; which must highly affect the TTA performance.

1. The use of large memory bank (1~10 samples per each of 1000 classes) might be unfair for comparison with baseline TTA methods.

### Questions
1. What is the design rationale of min-max mapping?

1. How sensitive is ADTE to pseudo-label noise in the early phase of adaptation? One experiment could be manually altering the class accuracy and see how ADTE behaves.

1. If we replace ADTE with a fixed q or with Shannon entropy but keep the same view-selection pipeline, how much of the reported gain would remain?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Adaptive Debiasing Tsallis Entropy (ADTE), a novel Test-Time Adaptation (TTA) method based on Tsallis Entropy (TE).
CLIP predictions exhibit bias, with certain classes being predicted more accurately than others.
Conventional CLIP TTA uses Shannon Entropy (SE), but SE treats all classes equally and thus cannot address the prediction bias problem.
Therefore, this paper proposes using TE, a generalization of SE, which is better suited for such biased predictions.
Furthermore, this paper proposes a method to automatically estimate q, the class-specific parameter required for TE, using a bias estimation technique that uses a memory bank.
Experiments on OOD benchmarks and cross-domain benchmarks demonstrate that ADTE achieves higher accuracy than existing TTA methods.

### Strengths
The motivation and proposed method are intuitive and interesting.
This paper adopts an intuitive approach of using parameter-free TE to address the prediction bias problem in CLIP. 
Despite its simple concept, the method proves to be effective.

Since TE is a generalization of SE, ADTE can be incorporated into existing SE-based TTA methods.
In an era where large-scale models trained on web-scale datasets with inherent prediction biases are mainstream, employing TE is a reasonable approach.

### Weaknesses
This paper lacks experiments with models other than CLIP, raising concerns about the applicability of ADTE.
For example, can ADTE be applied to unimodal models pretrained on ImageNet or subsequent models of CLIP, such as SigLIP?

ADTE is a method specialized for scenarios with predictive bias, but how common are such scenarios?
It is desirable to have a discussion about scenarios where ADTE is effective (such as model or dataset distributions).

### Questions
I was not aware of the details of the module used in Tables 3 and 4.
Is ADTE a method that includes LA?
What is the difference between LA-SE and SE-LA?

Minor comment.
Typo: Line120 Logit Adjustment (EM) -> Logit Adjustment (LA)

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Adaptive Debiasing Tsallis Entropy (ADTE), a test-time adaptation method for vision-language models such as CLIP. Conventional approaches rely on Shannon Entropy (SE), which is biased due to class imbalance in pre-trained data. ADTE replaces SE with Tsallis Entropy (TE), a generalized entropy with parameter $q$ that mitigates bias when $0<q<1$ and further introduces a class-specific adaptive parameter $q_l$ estimated from test data using a small memory bank. ADTE can be directly applied to existing TTA pipelines without retraining or hyperparameter tuning. Experiments on ImageNet variants and cross-domain benchmarks show consistent improvements over baseline methods.

### Strengths
- Effectively addresses the problem of biased predictions by generalizing SE to TE
- Plug-and-play design which can integrate into existing TTA frameworks without retraining
- Achieves consistent performance improvements across OOD and cross-domain benchmarks
- Computationally lightweight, minimal additional cost or tuning required

### Weaknesses
- In TTA settings like ZERO, where adaptation relies only on confident-view selection per instance, considering class-wise bias may be less relevant or overcomplicated.
- The paper lacks in-depth analysis explaining why ADTE particularly improves performance on cross-domain tasks, despite showing larger gains there in ablations.
- When LA is removed, performance becomes similar to competitive baselines, raising doubts about whether ADTE itself is a truly effective standalone TTA solution.
- The method’s novelty is limited, as several components (e.g., bias estimation, memory bank, LA) are borrowed or slightly modified from Frolic, making ADTE appear incremental rather than fundamentally new.
- The memory bank-based bias estimation is highly dependent on pseudo-label accuracy; early mispredictions can accumulate and distort class bias, especially for tail classes.
- In continual or gradual domain-shift settings, where test streams evolve over time, the memory-based adaptation may become inefficient or unstable, limiting its scalability to realistic online scenarios.

### Questions
see Weakness section

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of bias in vision-language models (VLMs) like CLIP during test-time adaptation (TTA). The authors argue that standard Shannon Entropy (SE) used in TTA methods (e.g., for selecting high-confidence augmented views) produces biased uncertainty estimates due to CLIP's pretraining on imbalanced data, leading to suboptimal view selection and adaptation. They propose using Tsallis Entropy (TE), a generalization of SE with a non-extensive parameter $q$, which better handles biased distributions. TE is further extended to Adaptive Debiasing Tsallis Entropy (ADTE), where class-specific parameters $q_l$ are adaptively computed via min-max normalization of estimated label biases from incoming test instances (using a memory bank and logit adjustment). ADTE integrates seamlessly with existing TTA pipelines like Zero, requiring no hyperparameter tuning. My detailed comments are as follows.

### Strengths
1. Extensive experiments on ImageNet, its variants, and 10 cross-domain datasets, demonstrate the effectiveness of the proposed method.
2. The approach is simple yet effective.

### Weaknesses
1. To further demonstrate the effectiveness of the proposed method, it would be better to conduct more experiments on the ImageNet-C, CIFAR10-C, and CIFAR100-C datasets.
2. The paper overlooks prior studies that have explored the use of Tsallis Entropy in test-time adaptation and domain adaptation. References [1–2] should be discussed in the related works section to better contextualize the novelty of ADTE.
3. The claim of requiring no hyperparameter tuning is somewhat misleading, as the method still depends on several implicit hyperparameters, such as [α,β] and the memory bank size.
4. ADTE relies on pseudo-labels to estimate class-wise bias; however, under severe domain shifts, pseudo-label noise may substantially affect the accuracy of bias estimation. The paper does not analyze how such errors impact ADTE’s performance.
5. The paper concludes that ADTE shows limited advantages when the model is nearly unbiased; however, it does not quantify the performance gap between ADTE and SE/TE under such low-bias conditions. Without empirical or theoretical analysis of these “unbiased” scenarios, it is difficult to assess ADTE’s applicability boundaries or to validate the claim that SE serves as its lower bound.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
