# Fine-Grained Iterative Adversarial Attacks with Limited Computation Budget

- Decision: Accept (Poster)
- Scores: 8, 8, 4, 4

## Abstract
This work tackles a critical challenge in AI safety research under limited compute: given a fixed computation budget, how can one maximize the strength of iterative adversarial attacks? Coarsely reducing the number of attack iterations lowers cost but substantially weakens effectiveness. To fulfill the attainable attack efficacy within a constrained budget, we propose a fine-grained control mechanism that selectively recomputes layer activations across both iteration-wise and layer-wise levels. Extensive experiments show that our method consistently outperforms existing baselines at equal cost. Moreover, when integrated into adversarial training, it attains comparable  performance with only 30\% of the original budget.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a timely approach to improving the computational efficiency of iterative adversarial attacks. To this end, authors replace the coarse-grained control of iteration count with a fine-grained, layer-wise spiking mechanism.

### Strengths
The topic this paper focused on is very interesting and novel. Besides, this paper is well-written.

### Weaknesses
1. Authors should report the magnitude of the error brought by using S_{\rho}. That is, authors should examine the approximation error between the left and right part of Figure 3.

2. Can the proposed method applied to the attack on large-scale dataset, such as the ImageNet dataset?

3. Can the proposed method be well generalized to more iterative attacks, such as C&W? It would be better authors add a discussion for the application to the black-box attacks.

minor:
1. The condition in Algorithm 2 "if $t=1$ or ..." seems to have a typo (likely should be if $t>1$ or ... for the second case). The notation $∂^(l)_t$ is also not explicitly defined before its use. A more polished and commented algorithm would improve clarity.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper tackles the efficiency problem of iterative adversarial attacks, which are computationally expensive due to repeated forward and backward passes. The authors propose Spiking-PGD, featuring Spiking Forward Computation, which skips redundant layer computations based on activation changes, and Virtual Surrogate Gradient, which restores gradient flow when activations are reused. Formulated as a layer–iteration joint optimization problem, this approach enables fine-grained computation allocation. Experiments on multiple datasets show that Spiking-PGD significantly improves attack success rates under the same computational budget and reduces adversarial training costs by up to 70% without sacrificing performance.

### Strengths
1.The work demonstrates strong originality and innovation.
2.The method is elegantly designed.
3.The paper is well-structured and clearly written.

### Weaknesses
1.The paper lacks a released code repository, which limits reproducibility.
2.My main concern is that since the proposed method reuses and stores previous activations, it is likely to increase memory consumption. As GPU memory is often a more critical bottleneck than time, corresponding experiments or analyses on memory overhead should be included.
3.The vision experiments are limited to ResNet-18; it is recommended to include and discuss more diverse architectures, especially Transformer-based models.
4.The evaluation should be extended to more diverse modalities, such as audio and NLP datasets.

### Questions
1. Please compare the GPU memory overhead with baselines.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Spiking Iterative Attacks, a method to make iterative adversarial attacks more efficient under limited computation. By observing that layer activations have minor changes across attack iterations, the authors introduce fine-grained, iteration and layer-wise control that recomputes activations only when necessary, and use a virtual surrogate gradient to maintain gradient flow when activations are reused. This approach achieves comparable or stronger attacks at significantly lower computational cost and also speeds up adversarial training.

### Strengths
- The paper is well-written and easy to read. 
- The contribution is novel and interesting.

### Weaknesses
**Missing important baselines induces vague lack of contemporaneity:** The paper compares against standard iterative attacks (PGD, MI-FGSM, I-FGSM) but omits stronger, widely-used robust evaluation suites such as AutoAttack (AA) [ext_ref_1]. Without AA (or similar), it’s hard to evaluate the actual impact of Spiking-PGD. AutoAttack is now a de-facto standard for robust evaluation, and since it only allows modifying iterations, skipping it weakens the claim that Spiking-PGD genuinely expands the efficiency–effectiveness frontier. Unfortunately, the same can be said for the choice of using only ResNet18 models and usual datasets and not extending to transformers. Even by accepting the choice of not using modern architectures or datasets, to broaden the impact of the proposed approach it is significant to extend beyond the single ResNet18 architecture with different Deep Neural Network architectures, such as VGG and others. 

**Slight lack of clarity in some aspects:** 
- They show activations quickly become similar across iterations, motivating their methodology. Yet, iterative attacks still improve with more iterations. The paper lacks a clear mechanistic explanation reconciling these two facts. If activations stabilize, why do further iterations still yield stronger adversaries?  
- Equation (2) is a discrete, combinatorial, non-convex optimization (NP-hard). The paper frames this but then presents a limited discussion on why other well-known strategies (continuous relaxations, differentiable masks, Gumbel-Softmax, integer programming relaxations, STE, or pruning literature methods) are unsuitable, or how they compare.

[ext_ref_1]: Croce, Francesco, and Matthias Hein. "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks." International conference on machine learning. PMLR, 2020.

### Questions
1. Why did the authors choose not to include AutoAttack (AA) [ext_ref_1] or other modern robust evaluation suites in their experiments, given that these are widely considered standard baselines for assessing adversarial robustness today? Can the authors integrate AA into their experiments during rebuttal?
2. Can the authors test the method on other architectures (e.g., VGG, WideResNet, or even better on transformer-based models) to validate whether the proposed approach generalizes across architectures?
3. The authors show that activations across iterations become highly similar, yet iterative attacks continue to improve as iterations increase. How do the authors reconcile this apparent contradiction? What mechanisms explain the continued effectiveness of attacks even when activations stabilize?
4. Can the authors elaborate more on why do standard relaxations or methods fail in solving equation 2?

Overall, I liked the way in which the paper was presented and the provided method/contribution. However, there are some important limitations from the experimental perspective, as well as concepts that need further elaboration from my point of view, thus justifying my score. I am open to increase my score, however, based on how the authors will conduct rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Spiking-PGD, a method for performing iterative adversarial attacks under a limited computational budget. The key idea is to skip layer computations in iterations where activations change little compared to previous iterations.

### Strengths
1 The claim addresses a practically important problem: iterative adversarial attacks (and adversarial training) are expensive, especially for larger models/datasets. 

2 The redundancy study is a good supporting piece: showing that intermediate activations across iterations become similar, giving plausibility to “skip some computations” idea.

### Weaknesses
1 Computation Budget Definition Ambiguity.
The proposed method claims reduced computational cost by selectively skipping layer computations based on activation similarity. However, to determine whether to skip a layer, one must first assess the change in activations, which itself appears to require computing the activation. The paper does not clarify how this comparison is implemented without incurring similar cost to a standard forward pass. Consequently, the reported budget may underestimate the true computational cost.
2 It is unclear whether the approach generalizes to other popular architectures (like Transformers in vision/NLP).

### Questions
1 Could the authors provide a more concrete breakdown of the computational cost in practical units, such as GPU FLOPs or runtime, instead of reporting it only as a proportion of full-precision operations? This would make it easier to evaluate the real-world efficiency of the proposed method.
2 If feasible, please include experiments on larger models and datasets (e.g., ImageNet with ResNet50 or Vision Transformers) to verify whether the claimed improvements under a fixed computational budget generalize to realistic, large-scale settings.

### Soundness
3

### Presentation
3

### Contribution
3
