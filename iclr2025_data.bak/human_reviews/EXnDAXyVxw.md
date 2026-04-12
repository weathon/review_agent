## Human Reviewer 1

### Summary
The paper works on domain generalization problem through weight quantization, which can be an implicit regularize by inducing noise in model weights to guide the optimization process to flatter minima against domain shifts. The paper provides both theoretical and empirical evidence for quantization to encourage flatter minima. Experiments on several datasets demonstrate the generalization ability of the method across various datasets, architectures, and quantization algorithms.

### Strengths
1. The paper is well-written and easy to follow.

2. The idea that improves the domain generalization ability by weight quantization is interesting. 

3. The experimental results demonstrate the effectiveness of the proposed method.

### Weaknesses
1. Equation 4 shows that the weights in flatten minima should have lower losses on the quantization model. However, it is not clear how the loss function guarantees that lower loss can in turn lead to local minima. It is also not clear how the large noise dominate the optimization and lead to sub-optimal convergence.

2. As shown in the experiments, the model size will definitely be reduced through quantization, but how about the training and inference costs of the model? Especially when the quantization is conducted during the model training.

### Questions
1. How are the ensemble quantization models trained? The models are trained differently from the beginning or from the quantization step?

2. How does the quantization step (e.g., 2000 for DomainNet) affect the model performance? How do the author select such hyperparameter?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper investigates the impact of quantization noise on out-of-distribution (OOD) generalization. The authors observe that quantization noise appears to enhance OOD generalization capabilities in domain generalization tasks. While the observation is interesting, the methods employed for quantization and ensemble are standard and lack novelty.

### Strengths
This paper is well-organized in general, and the observation that quantization noise can improve OOD generalization is interesting.

### Weaknesses
1. The methods for quantization and ensemble are quite standard and not specific to domain generalization, providing limited novelty.
2. The theoretical justification linking flat minima to improved OOD generalization is weakened by recent research suggesting that the connection between flatness and generalization is questionable [1,2].
3. Quantization noise is similar to uniform weight noise, but a systematic comparison with the latter as well as other weight perturbation schemes is missing.

[1] Andriushchenko, Maksym, et al. "A modern look at the relationship between sharpness and generalization." arXiv preprint arXiv:2302.07011 (2023).
[2] Mueller, Maximilian, et al. "Normalization layers are all that sharpness-aware minimization needs." Advances in Neural Information Processing Systems 36 (2024).

### Questions
It would be interesting to see how different weight perturbation schemes can affect OOD generalization.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes quantization-aware training to improve models' domain generalization performance. Theoretical and empirical analysis show that quantization-aware training induces noise in model weights, which could guide the optimization process toward flatter minima that generalize better.

### Strengths
1. Introducing quantization to domain generalization, and drawing the potential link between quantization and flat minima is novel and helpful, as it could improve domain generalization performance as well as memory and computation efficiency. 
2. Extensive experiments on DomainBed and empirical analysis show the effectiveness of QT-DoG.

### Weaknesses
1. The effect of $s$ should be discussed in more detail, as it plays an important role in the training and the theoretical analysis. Specifically, choosing a different $s$ of each channel in a layer should be justified. Ablation studies regarding $s$ can also be conducted to give a better understanding of the effect of $s$.

3. Measuring sharpness by equation 5 needs more discussion. When the model weight under measurement has different scales, perturbing weight with the same $\gamma$ does not serve as a fair way to get $w'$ around the local area of $w$. Relative flatness (Definition 3 in [1]) could be considered to address the dependency of sharpness measurement over scale of model weight.

[1] https://arxiv.org/pdf/2001.00939

### Questions
1. How much does $w_q$ differ from $w$ in training?
2. Why each channel in a layer has a different scaling factor? How is the value of $s$ determined?
3. Given a fixed quantization bit $b$, how would $s$ influence the performance?
4. What is the scale of $w$ (e.g. norm) across different algorithms in section 3.3?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper proposes using quantization during model training as a strategy to enhance domain generalization. The authors demonstrate that the baseline ERM achieves competitive results when quantization is applied as an implicit regularizer, with quantization-induced noise guiding the model toward flatter minima in the loss landscape. Additionally, they introduce combining quantized models into a model soup, to further boost DG performance. Results on the DomainBed benchmark indicate that this approach performs comparably to state-of-the-art DG methods.

### Strengths
1. The paper is well-written, experimental results seem comparable against state-of-the-arts.

2. The design of using quantized models seem to be a good fit in weight ensembling, which has been proved effective for improving generalization.

### Weaknesses
1. The primary issue is the limited contribution. Their main idea is to replace the standard training in ERM with the existing quantized technique. That is hardly a new practice, since quantization itself is originally applied in ERM (given ERM is the basis of all training tasks), and they do not inroduce a DG-specific quantization technique. Additionally, the ensemble of qunatitized ERM is also an simple extension of the original model-soup. These can be called a trick for improving DG, but shouldn't be listed as a main contribution.

2. The overclaim is another problem. It is claimed that the theoretical connection between quantization and flatter minima is provided. To begin with, I cannot find a seriours definition for the flatter minima of a model in the manuscript. Moreover, even with the informal definition in Eq. (5), they do not provide any theoretical evidence to support that $\mathcal{F}\_{\gamma}(W^q) < \mathcal{F}\_{\gamma}(W)$. At last, I doubt it can be theoretically proved, given there are many quantization methods with different settings, it's hard to conclude a general framework to be applied in all.

3. According to Tab. 2, it seems different quantization methods can significantly affect the in-domain and out-of-domain accuracy, some even worse than the baseline ERM. Does this suggests that not all quantization leads to flat minima? If so, why does the adopted quantization method can lead to flat minima?

4. Experiments should be conducted in more data. The more realsitic WILDS dataset could be used to further show their effectiveness. Ablation studies shoube be conducted in more datasets rather than just PACS and TerraInc. Especially in Figure 3, where the upper two figures can barely support the claim.

5. Some unclear experimental settings. For example, how to choose the quantizer step size $s$? what is the quantizer step in Figure 3, is it a same thing with $s$. If no, how to decide it? does the quantization only applied in saving the model?

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4