# Quantile Activation: Correcting a failure mode of ML models

- Decision: Reject
- Scores: 5, 3, 8, 5

## Abstract
An established failure mode for machine learning models occurs when the same features are equally likely to belong to class $0$ and class $1$.. In such cases, any ML model cannot to correctly classify the sample. However, a solvable case emerges when the probabilities of class $0$ and $1$ vary with the "context distribution". To the best of our knowledge, standard neural network architectures like MLPs or CNNs are not equipped to handle this.

In this article, we propose a simple activation function, quantile activation (QACT), that addresses this problem without significantly increasing computational costs. The core idea is to "adapt" the outputs of each neuron to its *context distribution*. The proposed quantile activation, QACT, produces the "relative quantile" of the sample in its context distribution, rather than the actual values, as in traditional networks.

A practical example where the same sample can have different labels arises in cases of inherent distribution shift. We validate the proposed activation function under such shifts, using datasets designed to test robustness against distortions—CIFAR10C, CIFAR100C, MNISTC, TinyImagenetC. Our results demonstrate significantly better generalization across distortions compared to conventional classifiers, across various architectures. Although this paper presents a proof of concept, we find that this approach unexpectedly outperforms DINOv2 (small) under large distortions, despite DINOv2 being trained with a much larger network and dataset.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes a novel activation function, QACT, which computes its output based on the cumulative distribution function (CDF) of the pre-activation values within a minibatch. In the backward computation of QACT, the probability density function (PDF) of the CDF is estimated using kernel density estimation because the gradient of the CDF is given by the PDF.
The proposed activation function is evaluated using existing datasets with controlled levels of corruption, such as CIFAR-10C, demonstrating that DNN models with QACT achieve better classification accuracy and lower calibration error under strong distortion.

### Strengths
* Computing activation values based on the relative relationships between pre-activation values within a minibatch is interesting. This approach is somewhat similar to batch normalization (batchnorm) but does not include the trained coefficient and bias terms that batchnorm employs. I believe this is a key feature of the proposed activation function, enabling it to adapt to distribution shifts during inference.

### Weaknesses
* Terms such as "context," "context distribution," and "failure mode" are used in the manuscript, but their definitions are unclear. Specifically, I could not understand what "context" refers to in the toy example presented in Section 1. The authors should provide clear definitions for these terms, especially for "context," which is a key term in the paper.

* The concept of "context" in the paper appears to be similar to "distribution shift" as described in machine learning literature. Does it relate to distribution shift? If "context" in this paper is related to distribution shift, it should be compared with additional baseline methods developed for distribution shift, in addition to Dino-v2.

### Questions
* In Algorithm 1, do the indices in $z_i$, $q_i$, and $\tau_i$ refer to the same index? It seems that $z_i$ and $(q_i, \tau_i)$ should have different indices, if I understand correctly.

* In the experiments, why is the watershed loss used only with QACT models? Is there a reason why the watershed loss might not be suitable for non-QACT models? If there is no such reason, non-QACT models trained with the watershed loss should be included as baselines.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This manuscirpt reveals a failure mode cased by distribution shift in machine learning. Then the authors correspondingly propose a novel quantile activation fucntion of QACT to replace ReLU for addressing the failure model. The proposed QACT has a good performance on image datasets with distortions including CIFAR-10/100-C and TinyImageNet-C.

### Strengths
- Reveal a failure mode in machine learning

- Propose a novel activation fucntion of QACT to deal with classifcation with distribution shift

- The proposed QACT has a good performance on CIFAR-10/100-C and TinyImageNet-C

### Weaknesses
This paper does not have a good organization which makes the motivation unclear. The rationale of proposed quantile activation is also not well presented. In addition, only the small image datasets with distortion are not enough to demonstrate the effectiveness of QACT. Please see the details as below:

- **Q1:** This manuscript starts by showing a failure example in binary classification. For me, negative examples are distribution-shifted version of positive examples, thereby making this classification difficult. A formal definition should be provided to show this "failure" or "unlearnability". 

- **Q2:** What causes the "failure mode" and why quantile activation function can address this? Only some intuitive explanations are provided.  It will be much better to provide rigorous empirical or theoretically analysis w.r.t. these two questions.

- **Q3:** According to the manuscript, the computational complexity of QACT is $\mathcal{O} (n\log(n) +Sn_\eta)$, which means the method is hardly scalable to large datasets with significant sample size $S$.

- **Q4:** The experiments are not convincing for me: (1) few and narrow datasets: only the distorted image datasets are used. The proposed QACT should be at least verified on the original version of CIFAR-10/100 and TinyImageNet. Moreover, some other datasets with distirbution shift like ImageNet-Sketch [1] or ImageNet-A [2] can be used to make the results more convincing. (2) unsatisfied performance: even on the basic CIFAR-10/100-C, the proposed QACT does not show the superority when the severity is small (Figure 4(a) and 6(a)).




[1] Wang, Haohan, et al. "Learning robust global representations by penalizing local predictive power." Advances in Neural Information Processing Systems 32 (2019).

[2] Hendrycks, Dan, et al. "Natural adversarial examples." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a approach for dealing with a specific failure mode in machine learning : when the same features are equally likely to belonging to various classes. This occurs especially when context or distribution shifts occur. In order to do so, the authors propose using Quantile activation function, which act as « normalizers » to generalize through distribution shifts.

### Strengths
-Simplicity of the approach
-Thorough analysis (ex. computational complexity) and explanations of the approach
-Many explanations are provided as to why the idea makes sense in practice; think it is generally overlooked in machine learning, yet is key to grasping the inner workings of the approach.
-I don’t have much to say in the following sections: the work is straightforward, clearly explained, and well-supported by empirical evidence.

### Weaknesses
1. There might be a problem in the possibility to generalize the approach to more complex neural networks architecture. For instance, how would quantile activation be used in architecture involving transformers?

2. This isn’t quite a weakness in itself, but the idea is quite simple (almost naive), such that I’m surprised that this idea hasn’t been explored yet.

3. I find the toy examples quite interesting to understand the logic behind the approach, yet they describe quite unique situations that are unlikely to be met in practice. I think the toy examples would be more convincing if it described situations that are likely to be seen in practice.

### Questions
-Line 149 : « This work aims to address the failure mode described earlier. To the best of our knowledge, no existing literature specifically addresses this issue » But isn’t Challa et al. 2023 and the other related works addressing this issue?

-Do you think the approach could be used in the context of meta-learning? And if so, how would the quantile activation be used to gather information about a new context?

-Would the quantile approach be robust enough to handle unbalanced tasks?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a novel activation function called Quantile Activation (QAct) to enhance model robustness against context distribution shift. Specifically, it adapts the outputs from each neuron to its context distribution by considering the relative quantile of the sample in its context distribution. Experiments demonstrate the superiority of QAct compared with other activation functions.

### Strengths
1. This paper is well-written and easy to follow.
2. This paper considers an interesting scenario: existing ML models cannot correctly classify when several classes have the same probability.

### Weaknesses
1. The motivation and the experiments do not align. In Figure 1(a) and Figure 3(a), the authors emphasize the case where one class is the rotation version of the other. However, the datasets in the experiments are not the rotation version of each class, but the compression version[1]. 
2. In your experiments, the datasets (i.e., CIFAR-10C, CIFAR-100C, TinyImageNet-C, and MNISTC) are also considered by some papers [2, 3] as the covariate shift. Does your proposed method perform well on datasets with other distribution shifts?
3. Could you please re-explain how to obtain the weighted quantiles $\{q_i\}$? Do you augment $z_i$ by grounding the neuron and then re-wright them to obtain $q_i$, or grounding the neuron is the weighting process?

[1] Xie R, Wei H, Feng L, et al. On the importance of feature separability in predicting out-of-distribution error[J]. Advances in Neural Information Processing Systems, 2024, 36.

[2] Viviers C, Valiuddin A, Caetano F, et al. Can Your Generative Model Detect Out-of-Distribution Covariate Shift?[J]. arXiv preprint arXiv:2409.03043, 2024.

[3] Mallinar N, Zane A, Frei S, et al. Minimum-Norm Interpolation Under Covariate Shift[J]. arXiv preprint arXiv:2404.00522, 2024.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
3

### Contribution
2
