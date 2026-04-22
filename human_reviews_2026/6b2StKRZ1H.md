# Efficient Parameter-Space Integrated Gradients for Deep Network Optimization

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 2

## Abstract
We explore previously unreported properties and practical uses of integrated gradients for training deep neural networks, primarily convolutional models, in the sense of averaging gradients over a continuous range of parameter values at each update step rather than relying solely on the instantaneous gradient. Our contributions are: (a) We show that, across multiple architectures, integrated gradients yield up to 53.5\% greater reduction in per‑batch loss compared to baseline optimizers. (b) We demonstrate that, for a fixed batch and models prone to ill-conditioned curvature, a single step can approximate more than four predicted updates. (c) We introduce an efficient approximation for ResNet-152 fine-tuning that integrates gradients over hundreds of past training iterations on a fixed batch at each parameter update. This variant is faster per step and easier to parallelize than a single step of a competitive Sharpness-Aware Minimization method, with only moderate memory overhead.

We validate the approach with first‑order optimizers (RMSProp, Adam) and a second‑order method (SOAP), showing consistent gains across settings. These results suggest that integrated gradients are a promising new direction for improving the generalization and potentially the test-time adaptation of deep models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a novel optimizer to train ANNs. Instead of calculating the gradient only at single points in the parameter space, the authors propose to integrate the gradient along a linear path between two parameter points. This gradient integration is approximated by individually integrating each factor of the backpropagation algorithm and assuming a linear dependence of each layer's activations on the parameters. The authors evaluate their proposed method on custom CNNs on the MNIST and Fashion-MNIST dataset as well as on ResNet-152 on Imagenet-OOD.

### Strengths
The core concept of gradient integration is interesting.

### Weaknesses
1. The paper is difficult to understand in many sections. The mathematical formalism is imprecise and is hard to follow. The lengthy discussion at the paper's conclusion doesn't help much to understand the experimental setup or results.
2. The motivation behind the method is unclear. What is the intuitive advantage of this approach compared to training with momentum? If the gradient is averaged over multiple parameter positions, why not also vary the data batch, similar to momentum? I fail to see an intuitive reason why the proposed method should outperform standard momentum-based training.
3. The specific approximations and assumptions made during the method's derivation, along with their justifications, need to be stated more clearly. For instance, factorizing the gradient averaging (Eq. 2) seems like a very harsh approximation to me that requires more thorough justification.
4. The empirical evaluation is insufficient. The chosen metrics (relative batch-loss minimization and relative sample efficiency) appear contrived, and their practical significance is not evident. Simply demonstrating improvement on these self-defined metrics is not enough to prove the advantages of the proposed method.
5. The authors do not provide  a comparison of the computational runtime of their proposed optimizer against baseline optimizers.
6. Tab. 2-4 are very messy and difficult to interpret. The criteria for bolding numbers are not explained. The meaning or significance of the "step length" column is unclear.
7. The training procedure for Fig. 1 is not described. Imagenet-OOD is not a well-established benchmark. You should give more details what you are doing here. 
8. The authors claim their method improves generalization but only provide test accuracies for a single model (ResNet-152) in Figure 1, which show only marginal gains. More extensive experiments on diverse and larger datasets are necessary. While experiments using models like ViT on ImageNet would be ideal, results on tasks such as WideResNet on CIFAR100 should be included at a minimum.
9. If the authors claim the reach better generalization than e.g. SAM, they should directly compare against this method.
10. The paper lacks any theoretical analysis of the proposed method.
11. The attached code is very messy and extremely difficult to comprehend.

### Questions
1. How did you choose the model for the LR-optimization in Tab. 2 and 3?
2. What is the exact improvement over your previous work?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an optimization approach that replaces the standard gradient update with an integrated gradient. Instead of using the instantaneous gradient, the method averages gradients computed along a virtual path between two or more weight states, claiming this approximates multiple gradient steps in one update. The paper introduced several variants, corresponding to integrating over one, two, or three previous parameter states. Experiments on small CNNs and ResNet-152 fine-tuning show lower per-batch losses and faster training compared to SGD and Adam, based on a custom metric called “Relative Batch–Loss Minimization.”

### Strengths
The paper may contain an interesting idea, but critical details are omitted or deferred to a non-public (non-peer-reviewed) reference in the supplementary material, making it difficult to judge the technical contribution.

### Weaknesses
The paper is poorly written and difficult to evaluate. The introduction fails to clearly describe the problem, motivation, or context, and key details are deferred to an anonymized, non-public paper that is cited as “under review” in the supplementary material. This is unprofessional and makes an assessment of the claimed contributions difficult. The paper frequently references prior “results” without providing credible sources other than a different manuscript in the supplementary material, leaving the reviewer unable to determine what is being extended or improved upon.

Technically, the proposed method, described as an “integrated gradient in parameter space”, reduces to a simple averaging of current and past gradients, offering no meaningful novelty beyond classical momentum or lookahead mechanisms. The presentation lacks reasoning, explanations, and substantive discussion. The writings are difficult to follow and overloaded with vague or loosely defined terms.

The experiments are limited to CNNs, and there is no clear reason why the proposed method can not be tested on different architectures on more standard benchmarks and tasks. The paper reports results using an unconventional metric (“Relative Batch Loss Minimization”) that does not align with standard evaluation practices. The overall formatting and clarity are poor, and the work does not meet the standards for a publishable submission.

### Questions
Beyond implementation details, what is the fundamental conceptual or algorithmic difference between the proposed “integrated gradient” update and classical momentum or lookahead methods, which also average gradients across iterations?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel and efficient method for optimizing deep neural networks by using integrated gradients in parameter space. Instead of relying on the instantaneous gradient at the current parameters, the core idea is to compute an averaged gradient over a continuous path between two parameter points (e.g., from past to current, or current to a forecasted future state) for each update.

### Strengths
The experimental results are good. AG-1 and AG-2 achieve up to 53.5% greater per-batch loss reduction compared to baseline optimizers at matched step lengths. AG-3 provides consistent test accuracy gains (~0.4%) on ResNet-152 fine-tuning, a notable improvement over a strong second-order baseline.

### Weaknesses
1. In section 3.2 the authors show the algorithm is based on the approximation and propagation of the average gradient according to  the paper [1], which is under review and cannot be found now. I believe this will have a significantly negative impact on the reviewers' assessment of this paper's contributions.

2. The need to modify the backpropagation graph for each layer type makes it non-trivial to port to new architectures.

3. The comparison with the baseline of the Transformer-based model is missing.

### Questions
It is unclear why the performance reported in Figure 1 remains below 60% for all methods. Additionally, the observed decline in accuracy before training reaches 10 epochs warrants clarification.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes an optimization framework called **Averaged Gradient (AG)** or **parameter-space integrated gradients**, which modifies backpropagation by replacing the instantaneous gradient with its average along a short trajectory in parameter space.  Unlike conventional momentum methods that aggregate past gradients with exponential weights, AG integrates the gradient field continuously (or approximately) between two or more parameter states, requiring no additional hyperparameters.  Several variants—AG-1 (forward-sign), AG-2 (full-magnitude), AG-3 (multi-step backward averaging), and AG-3-1 (one-step variant)—are introduced and evaluated across CNN, fully-connected, and ResNet architectures using optimizers such as Adam, RMSProp, and SOAP.  Experiments show that AG yields up to 50% faster batch-loss minimization and modest but consistent generalization gains, approximating the effect of multiple gradient updates at only twice the computational cost of a standard backward pass.

### Strengths
The topic of integrating gradients along a path in parameter space is conceptually important and connects to a fundamental idea in both optimization and interpretability.  The paper provides a computationally efficient extension of integrated gradients to the context of network training, showing that parameter-space integration can improve loss minimization and generalization with minimal overhead.

### Weaknesses
1. **Lack of technical contribution.** The core idea of integrated gradients is well-established, and the key innovation—the Averaged Gradient (AG) algorithm—was already introduced in the cited anonymous work. This paper mainly provides implementation variants (AG-2, AG-3, etc.) and additional empirical studies, rather than new theoretical or methodological advances.
2. **Obscure writing and weak presentation.** Many essential definitions and explanations (e.g., the definition of the averaging operator in Eq. (2)) are deferred to the anonymous citation or the appendix, making the paper hard to follow. The layout also has problems: some parts are overly dense (Table 1), while others are too sparse (Figure 1) or extend beyond column width (e.g., line 173). The overall presentation requires substantial revision before publication.
3. **Limited experimental scale.** The ResNet-152 evaluation is limited to fine-tuning rather than full end-to-end training, leaving open whether AG methods remain practical for large-scale optimization. The smaller models (A, B, C) contain only about 0.01 M parameters, which are too simple to convincingly demonstrate the claimed efficiency benefits.
4. **Relative reporting of results.** All numerical results in Tables 2–4 are given only as relative improvements, obscuring the absolute performance and making it difficult to assess convergence behavior or generalization. It is also unclear whether the baseline optimizers were sufficiently tuned, which could exaggerate the apparent advantages of the AG variants.

### Questions
1. Intuitively, the gradient at the current update should have greater significance. In previous methods such as momentum, there are hyperparameters to control this weighting, but in AG methods the integration is uniform with no weights. Would introducing weighted integration improve performance or stability?
2. Could you provide a direct quantitative comparison of time and memory consumption between the AG methods and conventional optimizers (e.g., Adam, SAM, or SOAP) to clarify the computational trade-offs?

### Soundness
2

### Presentation
2

### Contribution
1
