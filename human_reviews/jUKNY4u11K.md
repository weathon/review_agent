# WHAT YOU PAINT IS WHAT YOU GET

- Decision: Reject
- Scores: 3, 3, 6, 5

## Abstract
The two most prominent approaches for building adversary-resilient image classification models are adversarial training and input transformations. Despite significant advancements, adversarial training approaches struggle to generalize to unseen attacks, and the effectiveness of input transformations diminishes fast in the face of large perturbations. In general, there is a large space for improving the inherent trade-off between the accuracy and robustness of adversary-resilient models. Painting algorithms, which have not been used in adversarial training pipelines so far, capture core visual elements of images and offer a potential solution to the challenges faced by current defenses. This paper reveals a correlation between the magnitude of perturbations and the granularity of the painting process required to maximize the classification accuracy. We leverage this correlation in the proposed Painter-CLassifier-Decisioner (PCLD) framework, which employs adversarial training to build an ensemble of classifiers applied to a sequence of paintings with varying detalization. Benchmarks using provable adaptive attack techniques demonstrate the favorable performance of PCLD compared to state-of-the-art defenses, balancing accuracy and robustness while generalizing to unseen attacks. It extends robustness against substantial perturbations in high-resolution settings across various white-box attack methods under $\ell_\infty$-norm constraints.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
In this paper, a defense approach against adversarial examples based on adversarial training and image painting techniques is proposed. More specifically, at test time, the input images are first reconstructed by a painter using different stroke levels; then, a subset of the reconstructed images are fed into the classifier, and finally, its output scores are provided to a neural network that performs the decision.

### Strengths
- The proposed approach is very interesting and clearly presented.
- The addressed topic is relevant, as adversarial examples still represent an open challenge.
- The authors designed an adaptive attack to evaluate their defense, applying suggestions from related works, such as the use of BPDA to approximate the gradients of components that hinder the optimization process. Although the experimental evaluation still presents some issues, I appreciate this effort.

### Weaknesses
- The evaluation of the defense needs many improvements. First of all, the need for BDPA to attack the defense should be better experimentally proven. Using only 10 iterations for PGD and CW attacks makes them too weak (they are usually run with at least 100 iterations, but they might be increased even to 1000). An end-to-end attack should be performed by using the strongest attack (e.g., AutoAttack), applying random restarts and EOT steps to see if actually the defense cannot be attacked in this setting.
The authors initially mentioned applying EOT, but this doesn't seem to be the case after reading the rest of the paper and the provided source code. EOT should be used when the defense method induces randomization. The authors do not mention the choice and/or tuning of the attack hyperparameter (e.g., step size, number of random restarts, if performed). Finally, I noted that the experiments report some strange results: on the undefended model, AutoAttack performs worse than the other weaker attacks, and the same happens for trades. How can it be explained? Could it be due to the non-original implementation used by the authors?
- The competing methods considered for comparison do not report state-of-the-art results. Please consider better-performing models (for instance, from RobustBench) and compare the defense with them.
- Training/validation/test splits are built from the ImageNet training data (I assume that based on the number of images, as the ImageNet validation set contains only 50 images per class, while here, each of the seven considered classes has 1000 samples). However, the defended models are pre-trained on the entire ImageNet training set. In this way, images from training data are used to test the model. This approach is not methodologically correct and poses concerns about the reliability of the provided results.
- The best-performing classifier is chosen based on its robust accuracy under attack. This approach is also not methodologically correct, as it is similar to performing model tuning on test data.
- The computational complexity of the proposed approach seems huge and poses serious concerns about its practical applicability in real-world settings. The authors only report the inference time without specifying how it is measured, which is over 600x and 2200x with respect to undefended and standard adversarially-trained models for PCL_400 and PCLD, respectively. As it seems to be the main limitation of the defense, I would expect more discussion and evaluations about it (see questions).
- The authors categorize defenses against adversarial examples into adversarial training and input transformation methods. However, several works (even complementary to them) are based on detecting adversarial inputs. For the sake of completeness, this category of defenses should at least briefly be mentioned.

### Questions
- How is the inference time computed?
- How much does each defense's component impact on it?
- What is the memory footprint of the defense when it is deployed (also compared to standard models)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces PCLD, a novel defensive framework designed to mitigate adversarial attacks.
PCLD leverages the painter algorithm to generate images at varying levels of detail based on the number of strokes applied. The main assumption is that the painting algorithm would start building the image from the non-adversarial content, thus producing a "clean" image. These "painted" representations are then integrated with adversarial training to robustly train both a classifier and a decisioner, based on the classifier scores.

### Strengths
- usage of painting algorithm as a defensive transformation

### Weaknesses
- Core theoretical concepts missing
- Experimental evaluation should be enhanced
- Missing implementation details

**Clarity.** The authors omitted the definition of core concepts, that are essential for the reader, such as adversarial attacks and adversarial training. 
Their framework comprises adversarial training, however, the details behind the choice of the training loss are not discussed. 
There is no discussion about the optimizer and/or scheduler used, nor about their hyperparameters (except when they describe how they applied BPDA, using the encoder-decoders).
The state-of-the-art does not include methods that employ generative models that make use of style transfer to images to improve adversarial robustness [1].
Additionally, the authors should clearly discuss why the evaluation is limited at L-infinity attacks, and if this is a limitation of the defense.

**Evaluation should be improved.** All the accuracy-perturbation curves are made using PGD-10, while these results can be discarded, according to [2]. The results of PGD-100 are presented, but it is not clear if the adaptive strategy has been used or not. Additionally, as reported, having the decisioner (PCLD) does not really help improving the robust accuracy.
More details are given in the questions, and these are the main points to be addressed to improve the score of this review.

[1] Lin, Hubert, et al. "What can style transfer and paintings do for model robustness?." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
[2] Carlini, Nicholas, et al. "On evaluating adversarial robustness." arXiv preprint arXiv:1902.06705 (2019).

### Questions
- Would it be possible for an attacker to create an image for which painting the initial strokes produces the adversarial perturbation? If the painting is ML-based, probably there is a chance of having a successful adaptive attack in this way.
- What happens when an attacker targets the decisioner?
- In the abstract, what are "provable adaptive attack techniques?"
- The authors did not compare their approach with data augmentation involving another type of filtering, e.g., low-pass filter, to exclude high-frequency components of the images? 
- How is this approach different than, e.g., JPEG compression, which was proven ineffective as a defense?
- The authors should provide the result of the attack with an infinite budget, to show that the adaptive attack reaches 100% ASR as per the cited guidelines [2]

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed Painter-CLassifier-Decisioner (PCLD), an adversarial defense framework that introduces stroke-based painting to adversarial training. PCLD obtains a good generalization, able to defend against various adversarial attacks with different attack budgets. Compared with the baseline methods, PCLD helps improve both the clean accuracy and robust accuracy.

### Strengths
- The paper is well-written and easy to understand.
- The design of the defense pipeline is intuitive and makes sense to me. I think leveraging stroke-based painting to remove adversarial noise is an interesting idea.
- As a plug-and-play method, it obtains better robustness than some baseline methods using adversarial training.

### Weaknesses
- The inference time of PCLD is significantly longer than baseline methods. 
- Compared with many methods on RobustBench[1], there is still a non-negligible performance gap.

[1] https://robustbench.github.io/

### Questions
- In Fig. 2, does the decisioner learn the varying trend of the probability of each class and output a new probability or just choose one probability instance as the final output?
- Does the selection of the painting steps (including the total number of steps) affect the performance a lot? In this paper, based on what kind of metric do you set the painting steps?
- I am wondering if applying PCLD to a robust classifier (e.g. trained via TRADES) would further help improve the robustness. 
- Is PCLD still effective against $l_2$-norm attacks? In this paper, it is only evaluated on $l_\infty$-norm attacks. 
- For PCL without a decisioner, how do you get the final output? By manually setting a painting step and choosing this canvas as the input and get its output?
- According to Sec 4.3, $PCL_{B_p}$  stands for training on a set consisting of both clean and painting images. What is $PCL_{400}$ in Sec 4.4?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a reinforcement learning based painting algorithm framework to defend against adversarial perturbations in images. The framework includes a step to recreate the perturbed image in gradual steps using painting strokes and a classification-decision system to make decisions based on intermediate steps of painting.

### Strengths
This approach is a novel application of painting algorithms for transformation-based defenses against adversarial attacks.

### Weaknesses
The comparison of running times (0.67 vs 0.0003) with prior work is a huge downside which makes this method quite impractical to be deployed extensively.

### Questions
1. (Line 309) Some use of acronyms like CL for PCL and CLD for PCLD are not clear.

2. It wasn't entirely clear to me which part is being referred to as adversarial training.

3. Why is a non standard image size like 300x300 used for the ImageNet experiments. This would probably make intermediate activations in the model quite different to the standard 224x224 size used for ImageNet in other work.

### Soundness
2

### Presentation
2

### Contribution
2
