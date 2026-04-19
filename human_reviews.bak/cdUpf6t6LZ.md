# Robust NAS under adversarial training: benchmark, theory, and beyond

- Decision: Accept (poster)
- Scores: 6, 6, 6

## Abstract
Recent developments in neural architecture search (NAS) emphasize the significance of considering robust architectures against malicious data. However, there is a notable absence of benchmark evaluations and theoretical guarantees for searching these robust architectures, especially when adversarial training is considered. In this work, we aim to address these two challenges, making twofold contributions. First, we release a comprehensive data set that encompasses both clean accuracy and robust accuracy for a vast array of adversarially trained networks from the NAS-Bench-201 search space on image datasets. Then, leveraging the neural tangent kernel (NTK) tool from deep learning theory, we establish a generalization theory for searching architecture in terms of clean accuracy and robust accuracy under multi-objective adversarial training. We firmly believe that our benchmark and theoretical insights will significantly benefit the NAS community through reliable reproducibility, efficient assessment, and theoretical foundation, particularly in the pursuit of robust architectures.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
For fast and standardized evaluation of Neural Architecture Search algorithms, it is important to develop various benchmarks that contain a large set of architectures and their quality measured with diverse performance metrics. In this vein, the paper builds and releases a new NAS benchmark that considers not only the standard accuracy of evaluated architectures but also their robustness against adversarial attacks. The authors extend the existing NAS-Bench-201 benchmark by adversarially training the NAS-Bench-201 architectures and evaluating them on clean and perturbed data. The proposed robust NAS benchmark is named NAS-RobBench-201. Additionally, the authors theoretically characterize the robust generalization performance of architectures using the Neural Tangent Kernel (NTK) framework.

### Strengths
- As the authors note, adversarially training takes considerably more time than standard training with the cross-entropy loss. Given the enormous computational cost of adversarial training and evaluating a large number of architectures, I believe the authors really have done an impressive set of experiments.
- I generally agree with the necessity to create a NAS benchmark that targets robustness and generalization beyond standard test data.
- I do not feel confident enough to review and comment on the more theoretical parts of the paper. The proofs were not reviewed thoroughly, and thus I cannot vouch for its correctness. I am, however, familiar with the NTK theory, and I find the idea of extending the NTK-based analysis to explain adversarial robustness interesting.

### Weaknesses
- Currently, the robust accuracy is measured only under FGSM and PGD attacks. I think the performance under stronger attack methods (e.g., AutoAttack) is necessary to make the benchmark more reliable.
- The authors observe that the standard and robust accuracies exhibit a meaningful level of correlation. If they are, do we really need a separate robustness benchmark that includes robust accuracies? Wouldn’t searching for the optimal architecture in a conventional sense work as a good proxy and naturally result in a more robust architecture?
- I believe that NAS-Bench-201 is too constrained of a search space to accurately reflect the vastness of the potential architecture pool. For instance, [1] shows that many NTK-based methods fail outside the boundary of NAS-Bench-201. While this is a good first step towards building a more comprehensive robustness benchmark, it would be great if the authors could show that the experimental and theoretical analyses hold outside the tested search space.
- It would be nice to include the performance of trained architectures on out-of-distribution data and/or datasets with distribution shift in the benchmark. Incorporating these other types of robustness would make the benchmark more comprehensive, especially considering that in real deployment scenarios, the architectures will encounter generic OOD data more often than adversarially-perturbed data.
- In section 4.4, the authors study the correlation between the minimum eigenvalue of the NTK matrix and robust accuracy. I understand that this NTK-score is derived from theoretical resultss, but have the authors studied the relationship between other NTK-based scores (e.g., condition number [2]) and robust accuracy?

[1] Mok, Jisoo, et al. "Demystifying the neural tangent kernel from a practical perspective: Can it be trusted for neural architecture search without training?." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
[2] Chen, Wuyang, Xinyu Gong, and Zhangyang Wang. "Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective." International Conference on Learning Representations. 2020.

### Questions
Please refer to the weaknesses section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper first presents an adversarially trained benchmark based on NASBench 201, and present NASRobBench 201.

### Strengths
There is not much literature that presents a unified robust NAS benchmark, in that sense the work targets a unique space of research that needs exploration.

The NTK score vs accuracy correlation sounds interesting.

### Weaknesses
1. It has been well known that adversarial robustness often may occur due to gradient obfuscation and is applicable for different sparse and dense model architectures [1,2]. Thus, a discussion on that would be necessary. In specific, is there a way to benchmark based on a subnet's susceptibility to be more prone towards obfuscation?

2. Di you use ImageNet or ImageNet-16-120? As the Fig. 2 and the contribution section has mention of each.

3. Please demonstrate the efficacy of the NASRobBench on Autoattack.

4. The related work of Adversarial example generation is not up to date, the author should discuss more about the auto attacks and other contemporary attacks.

[1] Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples, ICML 2018.

[2] DNR: A Tunable Robust Pruning Framework Through Dynamic Network Rewiring of DNNs, ASP-DAC 2021.

### Questions
1. Apart from the doubts in weakness, I have the following question:

a. How you add noise "twice"?

b. Please extend the NTK score vs accuracy correlation with more recent and stronger attack scenarios.


### Post rebuttal:
-------------------------

Thanks authors for rebuttal.

Despite the limited novelty of the work, which the work did not commit anyway, I believe the work is good enough with enough empirical evaluations. Thus, I increase the score to 6! Though a 7 might have been the right score for this. I refrained from giving it a 8, just due to limited novelty scope, however, I think this work could be a useful add to the adversarial network benchmarking community.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents two primary contributions. Firstly, the authors have developed a search space encompassing over 6,000 adversarially trained architectures, addressing the limitation of prior studies that lacked such a comprehensive search space. Secondly, the authors demonstrate that robust architectures can indeed be identified within this search space by employing robust NTK.

### Strengths
1. The paper's principal strength lies in its creation of an adversarially trained search space, which required 107k GPU hours to construct. This significant advancement is poised to benefit researchers focusing on robust architecture search in the future immensely.
2. The analyses conducted within this adversarial search space are considered to be novel. It is particularly noteworthy to observe cross-sectional evidence in the NAS search space, confirming the expectation that a 3x3 kernel size CNN layer contributes to robustness. Additionally, the findings concerning the correlation between clean accuracy and robustness in the face of adversarial attacks are intriguing.
3. The paper is well-written, offering clarity and ease of comprehension.

### Weaknesses
1. The paper lacks a detailed explanation and justification for the necessity of employing twice perturbation, as mentioned at the end of page 7. It remains unclear why robust NTK requires a double perturbation when conventional adversarial training typically examines generalization from a single perturbation.
2. While it is posited that robust accuracy is influenced by the adversarial term, the theoretical analysis provided appears to be a reiteration of what was presented by Zhu et al. and Cao et al. in the context of clean NTK. Consequently, the contribution in this area seems minimal. Notably, the robust term A*(x,W), which is contingent on W and x, was not accounted for in the theoretical framework and was instead treated as an independent input variable. This oversight could impact the applicability of the theoretical analysis.
3. The paper employs an evaluation metric that is not standard, raising questions about its normalization. The last part of page 4, in the dataset paragraph, should clarify whether the metric is normalized before the attack and then inputted into the model. Additionally, it would be beneficial if the authors specified whether the same perturbation radius (32x32) used for ImageNet was applied during training.

* All of the previous concerns have been resolved through the rebuttal. Thanks for the clear and explicit response to my initial concerns and questions. Therefore, I have revised my score to 6. Regarding normalization in evaluation, I would like the author to double-check whether normalization is also applied inside the attack functions, specifically after adding the perturbation and before forwarding it to the model.

### Questions
1. Could you please clarify what 'optimal' refers to in Table 2?
2. In the first column of Table 2, what is the definition of 'criterion' used for?
3. It would be interesting to learn about the correlation results when measured with the existing clean NTK, such as in previous works by Zhu et al. and Cao et al. (which could correspond to beta being 0).
4. Referring to Figure 3, it appears that there is a high correlation between clean and robust accuracy. However, I am curious about the implications of using just clean NTK values for adversarial robust search and how that might affect the interpretation of the results.


Minor Comments
1. In Table 2, the entries in the first column are not center-aligned. This misalignment could be corrected for improved readability and a more professional presentation of the data.
2. The interpretations of Figure 3, specifically parts (b) and (c), pose some difficulties. Enhanced specificity in the explanations accompanying these figures would greatly facilitate understanding.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
