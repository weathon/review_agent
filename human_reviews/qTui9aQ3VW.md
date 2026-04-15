# How Robust Are Energy-Based Models Trained With Equilibrium Propagation?

- Decision: Reject
- Scores: 6, 6, 6, 6

## Abstract
Deep neural networks (DNNs) are easily fooled by adversarial perturbations that are imperceptible to humans. Adversarial training, a process where adversarial examples are added to the training set, is the current state-of-the-art defense against adversarial attacks, but it lowers the model's accuracy on clean inputs, is computationally expensive, and offers less robustness to natural noise. In contrast, energy-based models (EBMs), which were designed for efficient implementation in neuromorphic hardware and physical systems, incorporate feedback connections from each layer to the previous layer, yielding a recurrent, deep-attractor architecture which we hypothesize should make them naturally robust. Our work is the first to explore the robustness of EBMs to both natural corruptions and adversarial attacks, which we do using the CIFAR-10 and CIFAR-100 datasets. We demonstrate that EBMs are more robust than transformers and display comparable robustness to adversarially-trained DNNs on white-box, black-box, and natural perturbations without sacrificing clean accuracy, and without the need for adversarial training or additional training techniques.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper consider the adversarial robustness of EBM (trained with objective described in (1)), and compares with models trained with classic backpropagation. The results show that EBM are intrinsically more robust, and when we generate adversarial examples that fool the model, the "adversarial examples" also are more meaningful

### Strengths
A novel work about evaluating the adversarial robustness of EBM. The work seems pretty thorough,

### Weaknesses
First, regarding the adversarial robustness aspect of the paper -- the authors should make clear what the security model is early on. What I mean is that it is not enough to just state the words of "white-box, black-box" and so on, but one needs to have a discussion about what exactly these mean in the EBM case.

For example, it is somewhat clear from the early in the paper that the free phase of the inference may have gradient-masking behavior, but this was never discussed early on (and it is not enough to just say "we did white-box attacks", which in fact is actually already different from previous definitions of "white-box attacks"), and only until quite later in the experiments, we have an actual paragraph of "gradient obfuscation".

In other words, one should lead with a discussion specifically about how we attacked EBM, and be very clear about the inference stage issues. That will make the paper stronger.

Aside from that, while I do like the topic in this paper, the technical content seems to be a somewhat straightforward evaluation of EBM. They are good to know definitely, but the contributions are somewhat still limited.

### Questions
No specific question at this point. I have listed my concerns above, to me they are bigger concerns regarding the merits of the paper

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper evaluates the robustness of energy-based models (EBMs) via experiments. The authors provide experimental evidence for the EBM robustness against various types of noise. Specifically, EBMs are somewhat robust against both natural noise and adversarial perturbations. The authors first introduce the concept of EBM: some related works and the methodology in particular. Then, the authors present their empirical evaluation of the EBM robustness. Experimental results on the EBM robustness against various natural noise and adversarial examples (both white-box attacks and black-box attacks) are presented.

### Strengths
1. As new models are coming out in the machine learning community, it is worth investigating the adversarial robustness of those models.
2. The authors considered many different types of noise: 19 different types of natural noise and two different adversarial attacks (PGD and AutoAttack).
3. The presented result looks to be promising.

### Weaknesses
1. The description of how to attack EBM is not detailed enough.
    - If the adversary knows that EBM is used, the adversary will not try using the PGD attack on the traditional loss terms. (If PGD on the traditional loss terms is used for the experiment, it will damage the soundness of this paper.) What is the loss term that is used?
    - Is Temporal Projected Gradient Descent (TPGD) in Appendix 8.2 the attack used for EBM? If so, the authors should present this in the main body of the paper.
2. EBM itself should be described in more detail.
    - How do EBMs make predictions? This detail is needed to understand how to perform gradient-based adversarial attacks on EBM.
    - How costly are EBMs compared to the traditional models? This information is needed to justify the simple model architecture used in experiments.
3. The model architecture (four convolutional layers followed by a fully connected layer) is too simple. Is this because of the expensive computation cost of EBM? How do you know that the findings will generalize to more complex architecture?

### Questions
1. How do you measure the severity of each natural noise? The authors should describe how to measure the severity of each type of natural noise.
2. Figure 5 might not be enough to conclude that attacks on EP-CNN are more semantic. (Some people may not agree that the “deer” is induced by the attack.) Probably, it would be better to use a dataset that such semantic changes are more visible. For example, I’m curious about the changes from attacking EP-CNN trained on MNIST or a dataset containing face images.
3. Minor comments
    - To add an appendix, you should use `\appendix` command in LaTeX.
    - Consider making EP-CNN results more visible in the plot. For example, make other lines thinner in the plot, so that you can emphasize EP-CNN results. Also, it is better to render EP-CNN results after rendering other lines so that the other lines are now drawn over the EP-CNN results.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper empirically studies the energy based models especially the Equilibrium Propagation CNN (EP-CNN) in terms of adversarial robustness and natural noise on CIFAR. The results show that the EP-CNN is robust compared to ViT and adversarially trained CNN.

### Strengths
It’s an interesting direction to look into the robustness for EP models and the paper shows some interesting preliminary results on CIFAR. 

The paper is educative and interesting to read through, could possibly lead to many followups.

### Weaknesses
The paper claims that EP model is significantly more robust. However, the improvements are kind of marginal according to the line graphs in fig 2-4. Please quantify the improvements with specific numbers if the authors want to claim the significance.

The novelty is limited. This paper is mostly an empirical investigation on existing model architectures though I appreciate the new angle and experimental efforts.

The paper did not clearly state the novelty in Section3 Methods -  Are there any novelty proposed in this section or mainly repeat the approaches in existing works? If so it’s better to change the section title and make this clear.

It would be more convincing if the authors can theoretically prove or provide more solid insights why EP models are more robust.

EP models are harder to train and converge as discussed in Limitation, limiting the approach to scale up. It would be helpful to explore the benefits of EP models on other tasks besides image classification where they can show advantages over ViT and regular CNNs.

[post rebuttal]
Thank the authors for the response and additional results. Some of my concerns are addressed, while my last two concerns still remain. I am happy to raise my score to 6.

### Questions
See weakness above.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper analyzes the robustness of the Energy-Based Models (EBMs) trained with Equilibrium Propagation (PB) under natural corruptions and adversarial attacks. The experiments mainly focus on the PGD attacks, taking CIFAR-10 and CIFAR-100 as the datasets. There are three baseline models from different categories, namely, BP-CNN, AdvCNN, and ViT. Besides, at the end of the article, this paper discusses its limitations and future directions.

### Strengths
1. This paper is the pioneer in investigating the robustness of EBMs trained with EP under natural corruption and adversarial attacks, which may spark new insights and follow-up works.
2. This paper honestly discusses its limitations at present and provides several future directions.

### Weaknesses
1. Though the paper mentions that the high time cost limits EP models to relatively shallow architectures, I think the experiments that mainly focus on PGD are not enough. There are other attack methods, e.g., FGSM [1], C&W [2], AA [3], etc. The conclusion that EBMs trained with EP have better robustness will be more convincing if more attack methods are evaluated, even though the models have relatively simple architecture and the datasets are relatively small.
2. The superiority of the EP models in robustness is mainly shown by experimental results, which lack theoretical analysis and support. If rigorous mathematical analysis is too hard, then at least an intuitive analysis based on formulas that show why EP models can function well should be provided.

---
[1] Goodfellow, I. J.; Shlens, J.; and Szegedy, C. 2014. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.

[2] Athalye, A.; Carlini, N.; and Wagner, D. 2018. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In International conference on machine learning, 274–283. PMLR.

[3] Croce, F.; and Hein, M. 2020. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In International conference on machine learning, 2206–2216. PMLR.

### Questions
Including the concerns and questions mentioned in Weaknesses, I still have the following questions:
1. For the second term on the right side of Equation 1, I think it is possible that there are pooling layers following the FC layers. In other words, the second term can still contain $\mathcal{P}(\cdot)$?
2. According to my understanding, in Section 3.1, the training goal is to update the convolution weights $w_{n}$? However, I am unclear how the weights $w_n$ are updated based on Equations 4-6 related to the states $s^{n-1}$ and $s^{n+1}$. Can there be more explanation?
3. As mentioned in the limitation, ``EP is relatively sensitive to the hyperparameters used to train the model as well as the seed used to initialize the weights``. Are there some heuristic rules to set the hyperparameters and the weights' initialization?
4. Some editorial issues (just listed a few):
- Better not to use contractions such as ``didn't``, ``don't``, ``weren't``.
- It is a bit strange that there are the right brackets after BP-CNN and AdvCNN, respectively, but not the right bracket after ViT in Section 4.1.1.
- A missing period ``.`` after the second paragraph in Section 4.1.1.
- Inconsistent representations when referring to figures. For example, ``Fig. 2 and Fig. 10`` and ``Figure 1`` on page 8.
- On page 6, ``Figure 3,4`` should be ``Figures 3,4``.
- ...

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
