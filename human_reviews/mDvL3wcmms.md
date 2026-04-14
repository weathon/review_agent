# Classification-denoising networks

- Decision: Reject
- Scores: 5, 5, 1, 5

## Abstract
Image classification and denoising suffer from complementary issues of lack of robustness or partially ignoring conditioning information. We argue that they can be alleviated by unifying both tasks through a model of the joint probability of (noisy) images and class labels. Classification is performed with a forward pass followed by conditioning. Using the Tweedie-Miyasawa formula, we evaluate the denoising function with the score, which can be computed by marginalization and back-propagation. The training objective is then a combination of cross-entropy loss and denoising score matching loss integrated over noise levels. Numerical experiments on CIFAR-10 and ImageNet show competitive classification and denoising performance compared to reference deep convolutional classifiers/denoisers, and significantly improves efficiency compared to previous joint approaches. Our model shows an increased robustness to adversarial perturbations compared to a standard discriminative classifier, and allows for a novel interpretation of adversarial gradients as a difference of denoisers.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper unifies the tasks of image classification and denoising through a joint probabilistic model of (noisy) images and class labels to address complementary issues, such as insufficient robustness and partial neglect of conditional information in each task. The paper first introduces a framework that uses a single network to parameterize the joint distribution \( p(y, c) \) for performing classification, class-conditional, and unconditional denoising. Then, an architecture is proposed to parameterize the joint log-probability density of images and labels.

### Strengths
1. The derived discrepancy in the denoiser complements the previously connections between adversarial robustness and denoising.
2. The proposed architecture combines inductive biases suitable for both denoising and classification.

### Weaknesses
1. In joint training, the performance of both classification and denoising is inferior to existing standalone classification or denoising methods. 
2. Although joint training improves classification performance compared to separate training, the improvement is very limited and still falls short of ResNet18’s classification performance. 
3. The authors claim that the proposed method can be applied to out-of-distribution detection, but no experiments were conducted to verify the effectiveness of the method in this regard.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper integrates image denoising and image classification tasks, enabling the model to achieve enhanced robustness through joint learning. The architecture is based on ResNet-18, with modifications to improve performance by directly modeling the denoiser as the gradient of the neural network. The training objective combines cross-entropy loss with denoising score matching loss. Experimental results demonstrate that this approach outperforms previous jointly learned models and exhibits strong resilience to adversarial noise, along with a thorough analysis of its connection to score-based models.

### Strengths
+ Research on the joint learning of denoising and classification is limited; this paper highlights the potential of such integrated approaches.

+ The numerical results for classification surpass those of previous joint methods.

+ The relationship between the proposed method and the adversarial noise and energy-based models is analyzed in depth and discussed in detail.

### Weaknesses
- While this paper demonstrates superior performance in joint methods compared to others, it does not show significant advantages in standalone classification or denoising tasks. The denoising task under Gaussian noise can be viewed as a sampling process within a score-based diffusion model that uses Gaussian noise as a prior. In other words, the method presented in this paper sacrifices generative capabilities but learns a fixed noise-level denoiser in favor of enhanced classification performance. As a result, it outperforms JEM in classification but lacks the sampling ability of diffusion models and exhibits weaker resistance to adversarial noise compared to JEM.

### Questions
- Regarding the SVD decomposition of the Jacobian matrix, as described in line 427, it seems that any bias-free denoiser satisfies the condition \( D = \nabla_Y D Y \). Therefore, as long as a denoiser is effective and bias-free, the Jacobian will be decomposed to image feature information. I'm not sure what the significance of this experiment is, especially since Mohan et al. (2020) have already validated this.

-As mentioned in line 506, it seems that Guo et al. (2023) have already conducted a very similar study.


[1] Sreyas Mohan, Zahra Kadkhodaie, Eero P Simoncelli, and Carlos Fernandez-Granda. Robust and interpretable blind image denoising via bias-free convolutional neural networks. In International Conferenece on Learning Representations (ICLR), Addis Ababa, Ethiopia, April 2020.

[2] Qiushan Guo, Chuofan Ma, Yi Jiang, Zehuan Yuan, Yizhou Yu, and Ping Luo. Egc: Image generation and classification via a diffusion energy-based model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22952–22962, 2023.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
The paper propose a novel architecture that perform joint denoising and classification. It relies on tools from diffusion models such that the gradient of the denoiser is used to preform classification. The proposed algorithm and architecture are interesting and the mathematical formulation is sound. The authors perform experiments comparing to existing classifiers and denoisers showing on par performance. 
Yet, the authors ignore a large body of works that already use diffusion models to perform classification and bear great similarities to the proposed work. I will detail this below.

### Strengths
The authors propose a nice joint formulation for denoising and classification. The mathematical derivation is clear and the formulation is sound. If there were not many prior works that did very similar things I would have recommended accepting the paper. But a proper work should compare to prior works...

### Weaknesses
The paper elegantly (or may be not) ignores all the prior works that use diffusion models to perform classification. In fact, they cite one work (your diffusion model is secretly one shot classifier) but don't compare to it. Indeed, the work they already cite, deals less with robustness but there are many other works that perform joint classification and denoising and study robustness. For example:
(CERTIFIED!!) ADVERSARIAL ROBUSTNESS FOR FREE!, ICLR 2023
Robust Classification via a Single Diffusion Model, ICML 2024
Diffusion Models are Certifiably Robust Classifiers, NeurIPS 2024

Mentioning these works, explaining the difference and comparing to them (!!!) is a must. Right now this is the main problem with the paper and by ignoring the existing prior art it is a clear reject.

### Questions
Why did you ignore existing works?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work proposes a joint framework for classifying and denoising and also proposes a new network called GradResNet. This single network was trained for both denoising and classification tasks, yielding comparable results to the prior works such as ResNet, DnCNN and so on.

### Strengths
- The attempt to jointly classify and denoise looks interesting. One network can be effectively used for both classification and denoising.

### Weaknesses
- It is less convincing why one needs to combine the tasks of classification and denoising into a single network. Moreover, it is unclear what are the novel contributions of this work over prior works such as JEM.
- The evaluation look simplified by using too small networks for the problem like ImageNet classification, by denoising too small images with simple synthetic noises, and by  using too limited benchmarks.
- Diffusion models were mentioned in this work, but there is no experimental results about them.

### Questions
- Please see the weakness section.

### Soundness
2

### Presentation
2

### Contribution
2
