# Poisoned Forgery Face: Towards Backdoor Attacks on Face Forgery Detection

- Decision: Accept (spotlight)
- Scores: 6, 8, 8

## Abstract
The proliferation of face forgery techniques has raised significant concerns within society, thereby motivating the development of face forgery detection methods. These methods aim to distinguish forged faces from genuine ones and have proven effective in practical applications. However, this paper introduces a novel and previously unrecognized threat in face forgery detection scenarios caused by backdoor attack. By embedding backdoors into models and incorporating specific trigger patterns into the input, attackers can deceive detectors into producing erroneous predictions for forged faces. To achieve this goal, this paper proposes \emph{Poisoned Forgery Face} framework, which enables clean-label backdoor attacks on face forgery detectors. Our approach involves constructing a scalable trigger generator and utilizing a novel convolving process to generate translation-sensitive trigger patterns. Moreover, we employ a relative embedding method based on landmark-based regions to enhance the stealthiness of the poisoned samples. Consequently, detectors trained on our poisoned samples are embedded with backdoors. Notably, our approach surpasses SoTA backdoor baselines with a significant improvement in attack success rate (+16.39\% BD-AUC) and reduction in visibility (-12.65\% $L_\infty$). Furthermore, our attack exhibits promising performance against backdoor defenses. We anticipate that this paper will draw greater attention to the potential threats posed by backdoor attacks in face forgery detection scenarios. Our codes will be made available at \url{https://github.com/JWLiang007/PFF}.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to attack face deepfake detection models via backdoor attack, especially for blending artifact detection methods. To resolve the scalable problem, the paper builds a trigger generator to generate the trigger adaptively. To ensure the stealthiness of the trigger, a relative pixel-wise embedding ratio is incorporated in the generation of the poisoned sample. In general, the style of this paper is clear and logical. However, some experiments are missing and hence make readers confused about the effectiveness of some modules.

### Strengths
1. The paper is the early exploration of backdoor attacks in face forgery detection and the description of the existing obstacles deployed trigger in face forgery detection is clear.
2. Besides, the paper subtly simplifies the complex forgery problem as linear translation transformation and hence makes the backdoor attack in the translation-sensitive scenario.
3. To make the trigger stealthy, the author further hid the generated noise by a simple weight parameter.

### Weaknesses
1.Some descriptions like formulation may be incorrect and may misunderstand the readers.
2.Some ablation studies are missing and make readers confused about the effectiveness of the proposed modules. 
3.The method lacks generalization to some extent. The main gains derive from blending artifact detection. However, in the real scenario, not all evidence of determining forgery is based on blending traces in face forgery detection. Therefore, I would like to see the feedback about the questions listed below and decide to give a reasonable rating.

### Questions
The current rating is not my final decision, I hope the authors can explan the questions listed below in detail:

1.In Eq. (2), as stated, the blending transformation needs two real samples, however, the variable in function T^b only has one sample instead of a pair of real samples. For example, in Face X-ray, the fake face image is generated via two different real ones. So I think the formulation should be adjusted to make it more reasonable.

2.In the section on ‘Stealthiness of Backdoor Attacks’, Table 2 shows the qualitative results among different backdoor attack methods. However, the paper does not mention the number of samples for evaluation.

3.The influence of the hyperparameter ‘a’ is ambiguous. Moreover, the comparison with/without ‘alpha’ is missing and hence makes readers confused about the effectiveness brought by the proposed relative embedding. It is better to show the qualitative results and the quantitative backdoor attack performance.

4.This attack method is built on the generated image should show a relatively distinct facial boundary. However, in real scenarios, fake images are generated from various transformations. If some fake images are generated only via non-linear transformation, such as blur, this method may not be optimal. This may be the reason why the attack performance is lower than ‘LC’ because not all training samples contain linear translation transformation.

5.Typo: In Resistance to Backdoor Defenses, ‘Efficient-b4’ should be ‘EfficientNet-b4’.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, a novel backdoor attack strategy is introduced for the FACE FORGERY DETECTION task. This approach deceives face forgery detectors by embedding specific triggers within the training data. The authors delve deeply into the internal structure of current face forgery detectors and propose a transformation-sensitive triggering mechanism to facilitate effective backdoor attacks. Extensive experiments validate the efficacy of the proposed method and highlight potential vulnerabilities in face forgery detection systems.

### Strengths
1. The paper is the first to uncover the vulnerabilities during the training phase of face forgery detection, introducing a practical clean-label attack technique.

2. The backdoor attack strategy presented is notably innovative, leveraging the unique architectural characteristics of face forgery methods.

3. The trigger designed in this study is highly covert, making it challenging for the human eye to detect.

4. The authors provide a comprehensive explanation of the motivations behind the proposed method, and the manuscript is articulated clearly.

### Weaknesses
1. The paper could provide a discussion on the limitations of the proposed method, ideally placed in the appendix.
2. The choice of different kernel sizes for the trigger can impact the attack performance on various forgery detection models. The rationale behind such an impact remains unclear. Additionally, the authors' decision to finalize a size of 5 for the kernel lacks a clear justification.
3. Table 4 indicates that existing defense mechanisms fail against the proposed attack. It remains ambiguous whether this failure is specific to the attack introduced by the authors or if it's a general shortcoming for other attacks as well. Furthermore, it would be valuable if the authors could analyze existing attacks to suggest potential defense strategies.
4. In the section on trigger generation, the authors should provide a comprehensive description of the generator's overall loss function and its basic architecture.

### Questions
Please refer to [Weakness].

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes the Poisoned Forgery Face clean label backdoor attack framework utilizing a mask for the inner face region and a stealthy translation-sensitive trigger pattern, which is suitable for attacking facial forgery detectors. By poisoning a small portion of the clean training data, the attacker can flip the label of deep fake images or videos from fake to real. This approach can overcome the label conflict problem, which happens when the detectors are trained with self-blending techniques with only real data. Experimental results demonstrated a significant improvement in the attack success rate and the improvement of the trigger stealthiness.

### Strengths
+ The paper addresses not only the traditional deepfake artifact detection, in which detectors are trained with deepfakes, but also the recent blending artifact detection, in which the detectors are trained with augmented self-blended real images. The proposed translation-sensitive trigger pattern is innovative and effective for dealing with the self-blending approach.
 
+ The paper also provides a comprehensive benchmark for backdoor attacks, which is not available in the literature.

### Weaknesses
+ This paper is not the first in the literature to address backdoor attacks in face forgery detection. Cao et al. [A] have already investigated this problem. Therefore, the first contribution is invalid.
 
+ The use of "b" in the equations in section 3 is confusing. b presents both "blending" and "remaining clean."
 
+ The benchmark should include some frequency-based backdoor attacks like [B].
 
+ The reported performance of Face X-ray is lower than that in the original paper and in the SBI paper. I am wondering if there is something wrong with the experiments.
 
+ The robustness of the proposed trigger patterns was not investigated. While sharing deepfake images or videos on social networks, these patterns may be destroyed by compression or some other image processing algorithms.
 
+ The paper should include a paragraph of ethics statement
 
+ The appendix section, which was referred from the main part, is missing.

References:

[A] Cao, Xiaoyu, and Neil Zhenqiang Gong. "Understanding the security of deepfake detection." In International Conference on Digital Forensics and Cyber Crime, pp. 360-378. Cham: Springer International Publishing, 2021.

[B] Wang, Tong, Yuan Yao, Feng Xu, Shengwei An, Hanghang Tong, and Ting Wang. "An invisible black-box backdoor attack through frequency domain." In European Conference on Computer Vision, pp. 396-413. Cham: Springer Nature Switzerland, 2022.

### Questions
Please refer to the comments in the weaknesses section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
