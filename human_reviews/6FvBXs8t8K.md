# Learn from the Past: A Proxy based Adversarial Defense Framework to Boost Robustness

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3, 6

## Abstract
In light of the vulnerability of deep learning models to adversarial samples and the ensuing security issues, a range of methods, including Adversarial Training (AT) as a prominent representative, aimed at enhancing model robustness against various adversarial attacks, have seen rapid development. However, existing methods  essentially assist the current state of target model to defend against parameter-oriented adversarial attacks with explicit or implicit computation burdens, which also suffers from unstable convergence behavior due to inconsistency of optimization trajectories. Diverging from previous work, this paper reconsiders the update rule of target model and corresponding deficiency to defend based on its current state. By introducing the historical state of the target model as a proxy, which is endowed with much prior information for defense, we formulate a two-stage update rule, resulting in a general adversarial defense framework, which we refer to as 'LAST' ($\textbf{L}$earn from the P$\textbf{ast}$). Besides, we devise a Self Distillation (SD) based defense objective to constrain the update process of the proxy model without the introduction of larger teacher models. Experimentally, we demonstrate consistent and significant performance enhancements by refining a series of single-step and multi-step AT methods (e.g., up to $\bf 9.2$% and $\bf 20.5$% improvement of Robust Accuracy (RA) on CIFAR10 and CIFAR100 datasets, respectively) across various datasets, backbones and attack modalities, and validate its ability to enhance training stability and ameliorate catastrophic overfitting issues meanwhile.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a framework called LAST which can be combined with existing adversarial training techniques to improve robustness without much increase in computational complexity.  The LAST framework utilizes a proxy model which stores a previous state of the target model that is being trained and uses this information when determining the parameter update of the model.  They empirically demonstrate that using this gradient update leads to smoother loss landscapes and improvements in robust performance.

### Strengths
- paper is well-written and ideas are clear
- good scope in experiments: authors experiment with multiple datasets (CIFAR-10, CIFAR-100) and combine LAST framework with multiple single step and multi-step AT training techniques to demonstrate improvements.  Also interesting experiments on transfer attacks between models.
- introduction of method is clear, authors give a good intuition for why LAST framework can help with robustness

### Weaknesses
- It would also be nice to see experiments with other model architectures and L2 threat model
- some improvements in robustness (especially combined with single step AT training) are very small (error bars overlap)

### Questions
See Weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes LAST, a framework for training adversarially robust classifiers. At each optimization step, the adversarial perturbations are computed on the target model, but, unlike in commonly used methods, the model parameters update is computed via a proxy model, which can be seen as a past version of the target model. Such framework has the advantage of reducing the instabilities of robust training like catastrophic overfitting, and can be used in combination with different popular variants of adversarial training. In the experimental evaluation on CIFAR-10 and CIFAR-100, LAST provides improvements in the adversarial robustness of the classifiers.

### Strengths
- The approach of using a proxy model for computing the updates for stabilizing adversarial training is novel.

- The models trained with LAST achieve higher robustness than the baselines at high perturbation radii.

### Weaknesses
- The presentation is not clear: there are non-standard expressions like "parameter-oriented attack" which are not defined, and sentences difficult to follow and interpret (e.g. first paragraph in Page 4).

- The motivation behind the proposed algorithm, and the steps followed to formulate it, is not clear. If I'm not missing something, I think that LAST can be summarized as computing the adversarial attack on the current model, transferring it to the model at the previous iteration, computing a gradient step on it, and updating the current model with such step (plus SWA). This has the effect of slowing down training until the current and previous model are similar, i.e. end of training with small learning rate, and prevent (or delay) overfitting. However, it is not clear why this should provide higher robustness in general. Additionally, the SD loss looks like a variant of the TRADES loss (Zhang et al., 2019) where cross-entropy is used on the adversarial instead of clean points.

- In Table 1, the difference in robustness at $\epsilon=8/255$, which is the target threat model seen during training, evaluated with AutoAttack, between the models with LAST and baselines is within the standard deviation, while LAST models have significantly worse clean performance. In Table 2 and Table 3, the robust accuracy at $\epsilon=8/255$ with AutoAttack is not reported. Then, the improvement provided by LAST in such threat model is not evident.

- For the comparison with multi-step methods, there should be a comparison with SOTA techniques (see e.g. [A]) to support the benefit provided by LAST in such case.

[A] https://robustbench.github.io/

### Questions
See details above.

Overall, I think the experiments suggest that LAST provides some improvement in robust accuracy when combined single-step methods and evaluated at larger radii than what used for training (which is not a common evaluation setup), but at the cost of worse standard accuracy. This should be more clearly conveyed when stating the contributions of the paper. Moreover, the presentation and motivation for the proposed method need improvement in my opinion.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a self-distillation based adversarial training framework that learns from past states to make models adversarially robust. In particular, the paper proposes a two-state weight update rule using the gradient of the past state to the current adversarial perturbation. Results on CIFAR-10 and CIFAR-100 show promise.

### Strengths
+ The idea of using past model states to improve the model’s defense against adversarial attacks is interesting.
+ Paper is in general easy to follow.

### Weaknesses
**Formulation**
* Considering the proposed method is a two-phase approach, what is its implication on the training time/computational complexity of the method, when compared to baselines? This may be very important to address, since the paper by itself refers to SAT as a computationally expensive approach. How does the proposed method compare on this?
* The paper lacks a deeper analysis of the proposed method. For e.g., in Algorithm 1, substituting L13 in L16 yields \tilde{w} as an additive component to \theta. This would imply gradient ascent in the direction updated in L12. What does this imply? 
* Why does the model state from the past expected to behave more robustly to perturbations? While an empirical motivation is provided, a more intuitive (or conceptual or theoretical) motivation is required.
* Sec 2.4 discusses briefly how the momentum may be different from the use of momentum. It would have been better to see empirical results with just momentum to indeed show that this is the case.

**Experiments**
* It is not clear how the baseline methods for picked for the experimental studies. There is no discussion or motivation for the choices. Compared to recent papers on adversarial training, the number of baselines studied seem fewer.
* All through the methodology section (Sec 2), there was no mention that the proposed method is an add-on to existing methods. The experiments however only study the method as an add-on, not as a separate stand-alone method. Why is this the case? Without a clear reason for this decision, this makes the contributions weak.
* Continuing with the previous point, how are the methods in the experiments – LF-AT, LF-AT-GA, LF-BAT, etc - implemented? Without this detail, it is difficult to follow the results.
* Results on CIFAR-10 and CIFAR-100 may not be conclusive – it is important to study datasets of the scale of ImageNet.
* Why was PARN-18 chosen as the backbone? Why not just ResNet-18 or any other backbone?
* In Table 2, using the \uparrow to show improvement across 3 random seeds seemed rather unconventional. Generally, papers report mean and std dev when there are multiple trials. It was not clear what was being conveyed here.

**Literature**
* The discussion of related work is limited and weak. Beyond not being comprehensive for adversarial attacks and defenses, the paper also misses discussing other earlier efforts that look back at past model states. E.g  Jandial et al, Retrospective Loss: Looking Back to Improve Training of Deep Neural Networks, KDD 2020 (Although this work did not explicitly look at adversarial robustness, it is important to discuss literature comprehensively)

**Clarity and Presentation**
* I found the plots in Fig 1 difficult to understand – what is loss gap? Between what quantities?

### Questions
Please see weaknesses above

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors present a two-stage update rule for adversarial training, resulting in a general adversarial defense framework.

### Strengths
The motivation is clearly explained.
The paper is well-written.
Introducing historical state of the target model as its proxy to construct a two-stage adversarial defense framework, is a novel idea.

### Weaknesses
It would be interesting to see results on out-of-distribution samples.

### Questions
Please see above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
