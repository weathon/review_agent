## Human Reviewer 1

### Summary
This paper proposes an innovative method for adversarial robustness in deep learning through multi-scale input processing and dynamic self-ensembling. By introducing multi-resolution input representations and CrossMax ensembling—a unique aggregation strategy that reduces susceptibility to adversarial perturbations—the authors effectively address key limitations in existing robustness approaches. This work is particularly notable for its ability to achieve competitive performance on CIFAR-10 and SOTA results on CIFAR-100 without relying on adversarial training, a common but computationally intensive technique.

### Strengths
The paper was well written and well structured. The flow of the paper was easy to follow. And I would like to thank the reviewers for their presentation. 

I really liked that the idea of multi-scaling was inspired by the real human biology. I also enjoyed the cleaver trick leveraged in the CrossMax ensembling (and also in self-ensembling) to avoid fitting to the outlier predictions. 

The results provided in the paper have significant value and are truly incredible. The idea is novel and the contributions are adequate.

### Weaknesses
The only weakness of the paper seems to be lack of evidence for adaptive attacks. I am not exactly sure if the following will be the best implementation, but just as an example the attacker could possibly optimize the input image with the objective of making a misclassification to happen across all layers in self-ensembled networks, or if the adaptive attack could optimize the input image across all resolutions, for white box attacks.

### Questions
1) As I previously mentioned, I think the paper could really benefit from some analysis regarding its resistance to some relative adaptive attack. 

2) I also believe that since their method can adapt quite rapidly from a naturally trained model into an adversarially robust classifier, it would be a missed opportunity if it did not provide robustness results on ImageNet (or any other larger / more complex datasets). 

3) In your setting, could the authors clarify if the attacker is aware of the parameters of the multi-resolution inputs? And also, for a single sample, is the process of generating different resolutions of the same image a randomized process or a fixed process for iterative PGD attacks? And also, how much the authors attribute the robustness to the randomization. If yes (and the process is randomized), could they provide similar results by fixing the random seed throughout the iterative generations to see if that effects the effectiveness of auto-attack. 

4) On 336, why was the batch-norm turned off for the setting with training from the scratch? 

5) Could the authors please enhance their representation of other methods? For instance,  mention the name of their method or name of the authors or some other representation that would give some sense of what the other baseline is. 

6) Not a question, only a comment, the material covered in the appendix was really interesting and I really enjoyed them. I would be really happy to increase my score if the authors can convince me that the robusntess is preserved, even with adaptive attacks.

### Soundness
2

### Presentation
4

### Contribution
4

### Rating
5

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper addresses the challenge of adversarial robustness in deep neural networks. To tackle this problem, the authors of the paper propose a mechanism called CrossMax, based on Vickrey auction, to ensemble predictions from different layers given a multi-resolution input. The proposed method achieves adversarial robustness by leveraging the inherent robustness of intermediate layer predictions. The authors of the paper demonstrate that this method achieves significant adversarial accuracy on CIFAR-10 and CIFAR-100 without adversarial training. With the addition of adversarial training, the results can be further improved. Furthermore, the authors also explore the connection between adversarial robustness and the hierarchical nature of deep representations, showing that gradient-based attacks yield interpretable images of target classes. Additionally, they demonstrate that this approach enables controllable image generation using pre-trained classifiers and CLIP models.

### Strengths
- The paper is very well written and easy to follow. I enjoyed reading it. 
- The proposed method is technically sound.
- The proposed idea is novel and generally applicable to a lot of different scenarios.
- The proposed method achieves impressive benchmark performance on various tasks. 
- The authors of the paper conduct ample experiments to support their hypothesis.

### Weaknesses
- Most of the claims made in the paper seem to be empirical. It is not immediately clear how generalizable the conclusions are to other datasets and problems. 
- The proposed multi-resolution approach can make the classifiers less computationally efficient. 
- Hyperparameters on how to choose resolutions and number of resolutions can be hard to find/optimize.

### Questions
- I find the observation of using multi-resolution input to the classifier makes it much more adversarially robust interesting and somewhat surprising, since all the different resolutions are derived from the single-resolution image. Would it be possible at all to encoder this prior implicitly into the classifier model instead, so that we don't need to explicitly feed in N images of different resolution? 
- Does the performance of the model get better as we feed in images of more and more resolutions? 
- While I understanding the rationale of using Vickrey auction for ensembling, I wonder what inspired the use of it for self-ensembling? Does the same approach work for ensembling in general?

### Soundness
3

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
A novel method for improving adversarial robustness is used that does not necessitate adversarial specific training, but can be complemented by it. Using biological inspiration, the authors device two strategies for improving adversarial robustness: the creation of a gaussian pyramid like view of an input image for a network as well as the CrossMax self-ensemble strategy.

### Strengths
- Clear description of gaussian pyramid image stack and CrossMax aggregation strategies
- Empirical validation of the increased adversarial robustness of intermediate network layers
- Strong adversarial robustness results in CIFAR-10/100 without adversarial training
- Verification of the Interpretability-Robustness Hypothesis through qualitative examples of generated perturbations

### Weaknesses
- Robustness method requires extra effort to generalize to new architectures as opposed to adversarial training
- Max operation in CrossMax may introduce discontinuities in training that may make learning more difficult
- While the performance on CIFAR is impressive, a large scale dataset like ImageNet may be required to fully understand the generalizability of this method

### Questions
- During the generation of whitebox attacks is the multi-resolution image expansion considered as a differentiable step that the attack method backpropagates through?
- How would this method be generalized to transformer architectures?
- Was there any fine tuning of the model for the multiple linear probes used to create the ensemble? In other words, were the linear probes only generated after the intermediate layers were fixed? If so, what do you think the effect would be of training each intermediate layer linear readout jointly?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper introduces a new method for adversarial defense, which improves the adversarial robustness of neural networks by utilizing multi-resolution inputs and robust integration methods. The authors assume that the existence of adversarial attacks is due to the differences between human and machine vision systems. To bridge this gap, they suggest using the dynamic self-ensemble of multi-scale input representations and intermediate layer predictions. They show that intermediate layer predictions exhibit inherent robustness against adversarial attacks aimed at deceiving the entire classifier, and propose a robust aggregation mechanism based on the Vickrey auction, called CrossMax, to dynamically integrate them. The proposed method can achieve good results on CIFAR-10 and CIFAR-100, without any adversarial training or additional data.

### Strengths
1. The observation in this study is very interesting. 
2. The design of CrossMax is inspired by the Vickrey auction mechanism, aiming to create an integrated model that is more robust against attackers by leveraging the top-ranked predictions. This is a meaningful design.
3. As we know, adversarial training is usually time consuming, while the proposed method can achieve good results without adversarial training or additional data.

### Weaknesses
1. The proposed scheme is interesting and it can also achieves good results on CIFAR-10/100. However, the experiments in this paper are not very convincing. Firstly, the images in the CIFAR dataset have very low pixels, and it's hard to determine the actual impact of multi-scale or multi-resolution approaches. Although the supplementary materials cover the size of $224\times 224$, there is no comparison, testing, and analysis with mainstream methods. It's better to evaluate the performance on ImageNet or its subset. 
2. The evaluation of the adaptive attack is missing. I only found the appendix A shows some visualization results of adaptive attack by the  multi-resolution attack. This must be evaluated in the main experiments and the statistical results should be reported. Otherwise, it cannot show that the proposed method can defend against the adaptive attack. Maybe other adversarial defense methods can achieve better results. 
3. What are the contents of A.8, C and D in the supplementary materials? Are they the corresponding figures that follow? I suggest that the author should provide a detailed description of them.
4. What would the effect be if training were started from scratch? Since the most adversarial training methods are trained from scratch, the authors fine-tune the model on a ImageNet pre-trained model. If the previous adversarial defense also follow the same experiment setting, what would the results be?

### Questions
My major concern refers to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
5