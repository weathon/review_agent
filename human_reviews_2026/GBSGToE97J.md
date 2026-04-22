# Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Collecting web data to train deep models has become increasingly common, raising concerns about unauthorized data usage. To mitigate this issue, unlearnable examples introduce imperceptible perturbations into data, preventing models from learning effectively. However, existing methods typically rely on deep neural networks as surrogate models for perturbation generation, resulting in significant computational costs. In this work, we propose Perturbation-Induced Linearization (PIL), a computationally efficient yet effective method that generates perturbations using only linear surrogate models. PIL achieves comparable or better performance than existing surrogate-based methods while reducing computational time dramatically. We further reveal a key mechanism underlying unlearnable examples: inducing linearization to deep models, which explains why PIL can achieve competitive results in a very short time. Beyond this, we provide an analysis about the property of unlearnable examples under percentage-based partial perturbation. Our work not only provides a practical approach for data protection but also offers insights into what makes unlearnable examples effective.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a new algorithm to create unlearnable data by inducing linearization to models through the crafted perturbations.

### Strengths
1. The proposed algorithm works well on making unlearnable datasets for CIFAR-10/100, SVHN, and ImageNet. 
2. PIL is the most time-efficient surrogate-based model.

### Weaknesses
1. The motivation and mechanism of the proposed algorithm are obfuscating. The combination and decomposition between $\delta$ and $\delta_1+\delta_2$, is problematic and not convincing. Equ (8) and Line 212 are very confusing. 
2. Compared to all baseline methods, the performance of the proposed method is not consistently best and shows incremental improvements. Although it's much efficient than other surrogate model-based methods, the contribution is not that significant to me. 
3. Theorem 1 is self-contradicted as a large $\alpha$ will lead to abrupt accuracy drop.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors propose that unlearnable examples work because they induce model linearity. Using existing unlearnable dataset methods, they show that models trained on unlearnable data are more linear. Model linearity is measured by the attack success rate of FGSM attacks. Using this hypothesis, they design a new loss function that optimizes effective unlearnable data perturbations in less time than other approaches due to the tiny number of surrogate parameters.

### Strengths
In my batch so far, this is the most well-written paper. The writing/math is very clear, and some sections are better than prior conference work which introduce unlearnable example methods (especially wrt the defense-attacker definition and motivation of the problem). I can tell the authors were careful in how they designed notation to explain their method. It was a fun read. 

There are a number of novel contributions: 
1. a simple (2 loss components optimized with SGD) loss function that optimizes effective perturbations (across datasets). See Table 1. The perturbations also work across common augmentations (Table 2).
2. a new interesting finding: all unlearnable methods make models more vulnerable to FGSM, which the authors use as a proxy for showing that models trained on unlearnable data become more linear. This motivates their approach. See Tables 6-7.
3. the authors tie together loose ends on why unlearnable examples work. Linear separability of perturbations [2] was initially thought to be the mechanism, but there were counter-examples like AR unlearnable examples. Their linear behavior hypothesis is backed up by their FGSM experiments, but they also consider alternatives like the possibility that networks simply learn a correspondence between perturbations and labels (Section 4.4 and Table 5). I could see their FGSM approach inspiring future work to check other unlearnable data methods.

The strengths of this work outweigh the weaknesses because I feel it gives a convincing argument on why unlearnable data works, and there continues to be work in this area.

### Weaknesses
1. For common defenses (Section 4.2.1), ISS [1] is likely the most important, reasonable, and cheap defense against unlearnable examples. I am particularly interested in an evaluation of *just ISS* (different JPEG compression qualities should be tried: 0.9, 0.8, 0.7, etc) instead of all the other "defenses" (cutout, cutmix, mixup, etc.) because JPEG has been shown to be so effective but the augmentations provided in Table 2 are a good start (and can still be in appendix). The ISS [1] paper broke a number of existing "unlearnable datasets" but this submission only considers augmentations outside of JPEG. Other augmentations can't really be considered a defense after the ISS paper. 
2. Additionally, being a linear perturbation, I wonder if the orthogonal projection [3] defense (Section 4.4 of [3]) would work against these perturbations. The authors of [3] argue linear perturbations can be easily broken. It would involve training a linear model on PIL data, then projecting data orthogonal to that learned linear model. 
3. One of the main claims is that this work reveals a fundamental property of unlearnable examples: "they cannot substantially reduce test accuracy when only part of the dataset is perturbed". But this has already been shown in [1] Table 2, where with only 20% clean data, training on the sample-wise poison gets 86.85% accuracy. Only at 100% of unlearnable data do we see the more than 70% drop. I'm not sure if that can be considered a contribution of this work.

[1] Unlearnable Examples: Making Personal Data Unexploitable. Huang et al., 2021

[3] What can we learn from unlearnable datasets?, Sandoval-Segura et al., 2023

I'd note that addressing 1. and 2. does not take away from Section 5's findings.

### Questions
1. If the unlearnable data (original image + delta) is linearly separable, is the model trained on the unlearnable data expected to be linear too? If so, the results here remind me of [2] (but I understand the claim here is different bc [2] is about separability of perturbations, and the argument here is about the learned function)
2. Was an ablation of the the loss components was considered? I wonder how much less effective the method would be without Eq. 3 (lambda  = 1) or without Eq. 4 (lambda = 0). Based on your analysis, I'd expect Eq. 4 loss to matter more, but it'd be great to get an answer on that. (Doesn't have to be on all datasets or models, just one example to give a sense of what the answer is)
3. Were there other proxies you considered that could test model linearity? 
4. Does it make sense to caveat a little the claim of Line 410? It seems like SEP doesn't have such a wide gap with clean/perturbed models on FGSM?

[2] Availability attacks create shortcuts. Yu et al., 2022

### Soundness
3

### Presentation
4

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
This paper introduces Perturbation-Induced Linearization (PIL), an efficient method that generates perturbations using only linear surrogate models. PIL achieves comparable or better performance than existing surrogate-based methods while reducing computational time (reported ~40s on CIFAR-10). It also uncovers a key mechanism behind unlearnable examples: they induce deep models
to behave more like linear models, which may reduce their capacity to learn meaningful
representations. This paper also analyzes why accuracy is harder to suppress when only a partial fraction of the training set is perturbed.

### Strengths
1. Simplicity. PIL only use only linear surrogate models. On CIFAR-10, the reported generation time is under one GPU-minute.

2. Efficiency. PIL remains effective under various data augmentation strategies and adversarial training.

### Weaknesses
1. Please include results for larger initial clean ratios (η), e.g., 0.8, to validate how perturbed samples contribute to accuracy improvements. 

2. This paper primarily focuses on methods developed up to 2022. It would be helpful to include comparisons with more recent work—such as CUDA [1] and UGEs [2] in effectiveness and generation time.

[1] Vinu Sankar Sadasivan, Mahdi Soltanolkotabi, and Soheil Feizi. Cuda: Convolution-based unlearnable datasets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3862–3871, 2023.

[2] Ye J, Wang X. Ungeneralizable examples[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 11944-11953.

### Questions
See weaknesses.

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
2

### Summary
This paper introduces Perturbation-Induced Linearization (PIL), a method for generating unlearnable examples, which means data samples intentionally perturbed to prevent unauthorized model training. Unlike prior approaches such as REM or TAP that require complex and computationally expensive deep proxy models, PIL generates effective perturbations using only a linear classifier, dramatically improving efficiency.

### Strengths
1.The paper provides a comprehensive theoretical analysis of why unlearnable examples work.
2.PIL’s use of a linear classifier as the generator is useful and computationally lightweight.
3.The method can be applied to a broad range of models, which shows the strong generalization.
4.Both experiments and theoretical analysis are comprehensive and clear.

### Weaknesses
1.The author's claim that PIL perturbations are ‘imperceptible’ to humans is not quantified by human evaluation results or statistical measurement.
2.The authors empirically show that gradients from unlearnable samples are nearly orthogonal to those from clean samples, implying training interference. However, this is discussed qualitatively. A more rigorous analysis could include “quantitative measures such as cosine similarity distributions between gradient vectors per class.”

### Questions
1.Could authors provide a more rigorous analysis about the quantitative analysis between unlearnable samples and clean samples?
2.Have the authors considered including the kernelized linearization to capture the richer structure compared with only a single-layer linear classifier?
3.Could authors measure Jacobian singular value spectra or Hessian eigenvalue distributions before and after PIL perturbations?

### Soundness
4

### Presentation
3

### Contribution
4
