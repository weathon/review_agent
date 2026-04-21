# Training Diffusion Classifiers with Denoising Assistance

- Avg Score: 4.75
- Decision: Reject
- Scores: 5, 5, 6, 3

## Abstract
Score-matching and diffusion models have emerged as state-of-the-art generative models for both conditional and unconditional generation. Classifier-guided diffusion models are created by training a classifier on samples obtained from the forward-diffusion process (i.e., from data to noise). In this paper, we propose denoising-assisted (DA) classifiers wherein the diffusion classifier is trained using both noisy and denoised examples as simultaneous inputs to the model. We differentiate between denoising-assisted (DA) classifiers and noisy classifiers, which are diffusion classifiers that are only trained on noisy examples. Our experiments on Cifar10 and Imagenet show that DA-classifiers improve over noisy classifiers both quantitatively in terms of generalization to test data and qualitatively in terms of perceptually-aligned classifier-gradients and generative modeling metrics. We theoretically characterize the gradients of DA-classifiers to explain improved perceptual alignment. Building upon the observed generalization benefits of DA-classifiers, we propose and evaluate a semi-supervised framework for training diffusion classifiers and demonstrate improved generalization of DA-classifiers over noisy classifiers.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces Denoising-Assisted (DA) classifiers in the domain of classifier-guided diffusion models to enhance both conditional and unconditional generation tasks. The DA classifiers are trained using both noisy and denoised examples, unlike traditional diffusion classifiers trained only on noisy data. Through experiments on CIFAR10 and Imagenet datasets, the authors demonstrate that DA classifiers exhibit better generalization to unseen data and improved perceptual alignment of classifier-gradients, leading to enhanced image generation. The paper also theoretically analyzes the gradients of DA-classifiers to explain the observed improvements. Additionally, a semi-supervised framework is proposed to leverage the generalization strengths of DA-classifiers in scenarios with limited labeled data. The empirical and analytical discussions included in the paper provide a thorough understanding of the improvements DA-classifiers bring over noisy classifiers, showing promise in advancing the performance of generative models.

### Strengths
The paper unfolds an innovative notion of utilizing denoised samples as inputs to the classifier within a diffusion model framework. The simplicity of this idea is elegantly juxtaposed with its effectiveness, as substantiated by the authors through empirical evaluations and theoretical elucidations. The inner workings of the proposed method are also thoughtfully explained, shedding light on the mechanisms that contribute to its efficacy.

### Weaknesses
1. The presentation of the paper could be improved for better readability. Specifically, the formatting of equations (5) and (6) spanning across three rows appears to be cluttered and may benefit from a more concise representation.
  
2. The paper primarily focuses on one-step denoising, which raises the question of why multi-step denoising to reach time step t=0 was not explored. Utilizing the sample at time step t=0 as input to the classifier could potentially offer additional insights or improvements, and it would be beneficial for the authors to discuss or explore this aspect.
  
3. The core contribution of employing the denoised sample as input to a classifier may come across as straightforward. The paper could benefit from a clearer articulation of the motivation behind this choice and the specific problems it aims to address. While the one-step denoised sample is utilized, the rationale behind not exploring [x, one-step denoised sample, two-step denoised sample] as inputs could use further clarification. Although the authors provide some explanations, a more robust justification could enhance the perceived significance of the work.
  
4. The discussion on the semi-supervised learning framework introduces an idea of selecting pseudo-label data based on confidence thresholds during the diffusion process. However, the motivation behind this choice could be better elucidated. It may be worth exploring or explaining why traditional uncertainty measures in the raw data space were deemed insufficient, and how the proposed method addresses any identified limitations.

### Questions
See Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This manuscript proposed the denoising-assisted (DA) classifier that employs additional denoised image for better classification. The DA classifier has shown better performance on CIFAR-10 and ImageNet compared to original diffusion classifiers, and this imporvement is also verified by theoretical analysis.

### Strengths
1. A new method of DA classifier is proposed by employing the denoised image for better classification;

2. Emprical experiments on CIFAR-10 and ImageNet verifies the effectiveness of the proposed approach;

3. Some insightful observations in terms of the classifier gradients are presented to analyse the imporvement of DA classifier.

### Weaknesses
IMHO, the writing and organization of this manuscirpt can be greatly imporved for clearer illustration. For example, the motivation and the most related works w.r.t. diffusion classifiers are quilte unclear. Moreover, the writing are redundant and blunt, which blends the authors's contributions and existed works. Some technical questions are presented as below:

1. What is the motivation to develop DA classifier?  What is the advantages of DA classifier compared to existed classifiers including conventional deep models (VGG, ResNet, etc.) or large models (CLIP, etc.)?

2. What the authors do is to employ the denoised images as the additional input. To this end, the authors should provide more detail about how to get the denoised images and analysing the influence of different types of denoised images.

3. Why Theorem 1 explain the improvements of DA classifier?

Minors and typos:
1. "Cifar10" -> "CIFAR-10"
2. "Imagenet" -> "ImageNet"
3. Page 3, Sec. 2.2: $y\in [1,C]$ -> $y\in =\\{1, ..., C\\}$

4. Page 3, Sec. 3: "propose to use as input both ..." -> "propose to use both ... as input"
5. To many long sentences.

### Questions
See Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes an approach utilizing both noisy and denoised examples in the diffusion process as inputs for training a classifier (DA-classifier). Comparative analysis against the noisy classifier method demonstrates the effectiveness of the proposed approach. Additionally, by analyzing its generalization, gradients, and image generation quality, the study further explains the efficacy of the proposed approach.

### Strengths
1. The paper conducts both quantitative (generalization to test data) and qualitative (perceptually-aligned classifier gradients and generative modeling metrics) analyses, exhibiting the advantages of DA-classifiers. 
2. The study not only empirically examines the performance of DA-classifiers but also provides theoretical explanations for their gradient properties, theoretically supporting improved perceptual alignment. 
3. The proposed method, though quite simple, proves to be effective in empirical validation.

### Weaknesses
1. The numerical results are not good. In Table 2, there are no significant differences between the two methods, especially P and R on the two datasets and all metrics on ImageNet. The visualization in Figures 2, 3, and 4 is impressive.
2. The motivation behind using diffusion-based samples to train a classifier for semi-supervised learning is unclear and requires further clarification. It is okay to see that semi-supervised generative models perform much worse than those discriminative models. Can we see the comparisons in image generations under the semi-supervised settings? Table 5 only shows the results trained on labeled data.

### Questions
1. Why the results of Tables 2 and 4 are not consistent.
2. 'Cifar10' in abstract.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces Denoising-Assisted (DA) classifiers to improve the classifier guidance method in diffusion models. DA classifiers are time-dependent classifiers where the input includes denoised examples in addition to perturbed examples and timesteps. The effectiveness of DA classifiers is demonstrated through test classification results, gradient analysis, and the quality of the generated images. They also consider semi-supervised settings, and DA classifiers outperform noisy classifiers in terms of classification accuracy.

### Strengths
The proposed method is simple but effective. Incorporating denoised examples into the classifiers is a reasonable approach to improve performance. Additionally, it is worth noting that they address partial supervision settings, although they did not provide the result on image generation performance.

### Weaknesses
1. Baseline

The paper lacks sufficient baseline comparisons. To strengthen their argument, the authors should compare their method with other relevant classifier guidance methods such as DLSM, ED, and Robust-Guidance. While they mention that these methods are complementary, it would be beneficial to present their performance results, similar to how standard semi-supervised discriminative models are presented in Table 3. In addition, the authors should extend the application of DA classifiers to these methods and report the results to demonstrate the orthogonal nature of their work.

2. Missing generation results in semi-supervised settings

There is no FID and IS results for generated images in semi-supervised settings. They only said that "FID and IS metrics are similar to the results described in Table 2". Since the main task of this paper could be seen as conditional image generation, these experimental results are crucial evidence to assess the effectiveness of their method.

3. Classifier-free guidance in semi-supervised training

The authors mentioned that semi-supervised training of a classifier-free model is not easy to implement, citing the original paper's recommendation for the supervised settings. It is important to empirically verify this claim, as there is a lack of evidence on for the feasibility and effectiveness of classifier-free models in partial label settings.

4. Clarification of the sampling procedure

Since the primary objective of this paper is image generation, it would be beneficial to provide a detailed procedure or algorithm for sample generation using the proposed method. Actually, the utilization of DA classifiers in sample generation is not straightforward, given the difference in gradient computation between $\nabla_x \log p(y|x,t)$ and $\frac{d \log p(y|x,\hat{x},t)}{dx}$. A discussion of the implications of this difference would help clarify the approach.

### Questions
1. Please check the issues in the Weaknesses section.

2. Ablation study

It would be beneficial to include an ablation study to investigate the key factors contributing to the improvement of the DA classifier. Two specific experiments might be considered:
* Changing $\hat{x}$ to $s_\theta (x,t)$ in the DA classifier inputs: this would help to identify the essential elements, such as denoised examples or information of data scores.
* Use only denoised examples, i.e. $p_\phi (y|\hat{x}, t): this would evaluate the importance of using both perturbed and denoised examples.

3. Minor points

* Eq. (7) should explicitly state that this equation holds for the optimal score network.
* The figure in the middle of page 5 lacks a caption.
* In the paragraph "Classifier Gradients (Empirical Observation)", "... classifiers trained only with uncorrupted samples..." may be corrected to "... classifiers trained only with corrupted samples...".
* "Score-SSL" does not appear in the manuscript, except in Table 3.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
