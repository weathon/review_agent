# Understanding the Initial Condensation of Convolutional Neural Networks

- Avg Score: 4.00
- Decision: Reject
- Scores: 3, 3, 6

## Abstract
Previous research has shown that fully-connected neural networks with small initialization and gradient-based training methods exhibit a phenomenon known as condensation during training. This phenomenon refers to the input weights of hidden neurons condensing into isolated orientations during training, revealing an implicit bias towards simple solutions in the parameter space. However, the impact of neural network structure on condensation remains unknown. In this study, we study convolutional neural networks (CNNs) as the starting point to explore the distinctions in the condensation behavior compared to fully-connected neural networks. Theoretically, we firstly demonstrate that under gradient descent (GD) and the small initialization scheme, the convolutional kernels of a two-layer CNN condense towards a specific direction determined by the training samples within a given time period. Subsequently, we conduct a series of systematic experiments to substantiate our theory and confirm condensation in more general settings. These findings contribute to a preliminary understanding of the non-linear training behavior exhibited by CNNs.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper analyzes the phenomenon called initial condensation in simple CNN models. Initial condensation refers to the occurrence of weight grouping in the early stages of neural network (NN) training, which has been discussed in previous literature mainly with respect to fully connected NNs. A two-layer CNN is theoretically analyzed and it is shown that initial condensation occurs in one CNN layer with respect to convolutional kernels.

### Strengths
- Initial condensation in CNN models is analyzed theoretically, although the assumed model is restricted to a simple two-layer one.

### Weaknesses
- It is difficult to follow the technical details of the paper for the following reasons:
    - $W_{p,q,\alpha,\beta}^{[l]}$ is not well defined for $p=-\infty, \ldots, \infty$ and $q=-\infty, \ldots, \infty$, although it is required to define $x_{u,v,\beta}^{[l]}$.
    - I could not understand the meaning of $\frac{dW_{p,q,\beta}}{dt}$. What is $t$?
    - The description of the paper does not follow the notations defined in Section 3.1. For example, the matrix notation introduced in Section 3.1 is not used in the main technical discussion in Section 4. Also, the operator norm is defined in Section 3.1, but does not appear in the manuscript.

### Questions
- Why are only 500 samples used in the CIFAR10 experiment?

- What is the test accuracy of the final model in Figure 3? I wonder if the final model overfits the training set.

- It is better to use \citet{} for textual citation.

- It is better to avoid using multiple $:=$ in one line.

- In page 4: .. -> .

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This manuscript presents a study of condensation in convolutional neural networks.
The main theorem states that under certain assumptions on the data, activation function and initial weights, the following two things hold:
 1. the final weights go arbitrarily far away from the starting point 
 2. the final weights all point in the direction of the principal eigenvector of some data-dependent matrix.

The experiments confirm the theoretical results, even in settings where assumptions are broken.

### Strengths
- (significance) Insights in the learning dynamics of neural networks typically help to guide model development and speed up learning.
 - (originality) The condensation problem has been studied for fully-connected networks, but this appears to be the first work on convolutional networks.

### Weaknesses
- (clarity) The paper is quite chaotic and therefore hard to read.
   Especially the frequent notation changes and notation that is used only in one place make the paper hard to read.
   E.g.&nbsp;something like $$\boldsymbol{\theta}_{\boldsymbol{W}, \boldsymbol{v}_1} := \operatorname{vec}\bigl(\\{\boldsymbol{\theta}\_{\boldsymbol{W},\beta} \cdot \boldsymbol{v}\_1\\}\_{\\beta=1}^M\bigr)$$
   would be much clearer than the current formula above equation&nbsp;(13).
 - (clarity) The variables $\eta_0$ and $T_\mathrm{eff}$ come out of nowhere and no intuition or explanation is provided about what these variables represent.
 - (clarity) The experiment section mentions that every CNN has an additional 2 fully-connected layers to produce outputs.
   As a result I do not understand how it is possible to do experiments with the theoretical setting which only considers convolutional networks.
 - (significance) I am unable to distill from the manuscript whether condensation is a good thing or a bad thing.
   Figure&nbsp;1 seems to suggest condensation enables learning smaller networks.
   However, the main theorem implies that only two possible directions of weights survives, which intuitively feels like a bad thing that would hinder expressivity.
   As a result, I also do not quite understand why having experiments where the assumptions are violated is supposed to be a selling point (unless it is a good thing).
   On the other hand, if it were a good thing, I am concerned about the caption of Figure&nbsp;2 and&nbsp;3 where it is stated that the network attains less than 20% accuracy.
 - (originality) It is not clear which parts of the analysis are taken from prior work and what new insights are necessary to make this work for convolutions.
   The use of the eigenvectors seems to be one of the most obvious differences
   but it would be good to highlight where exactly the differences are.
 - (quality) I am unable to properly assess the derivatiations and proofs because I understand too little of what is going on.
   However, I shortly skimmed over the proof of the main theorem and noticed a transition where the norm of sums becomes the sum of norms without any comments.
   Also, some non-obvious statements seem to be planted without proper explanation.
 - (quality) The theoretical results seems to build on an analysis of the dynamics of gradient descent.
   However, the experiments make use of adaptive optimisers like Adam, which should lead to significantly different dynamics than plain GD.

 ### Minor Comments

 - The hyperlinks in the paper seem to be broken. 
   Reading the paper required more scrolling than I'm used to.
 - Assumption 4 seems to be more of a definition than an assumption.
 - I don't quite understand what the infinity norms are supposed to do in equation&nbsp;(6).

### Questions
1. Please, reduce the noise in the mathematical notation for the sake of readability.
    Note that there is more noise than the one example I provided (e.g. $\boldsymbol{x}_r$ and $\boldsymbol{w}_r$, $\boldsymbol{\theta}_\beta$, $\boldsymbol{U}$ and $\boldsymbol{V}$, ...)!
 2. Why is $\Big|\sum\_{\beta} \varepsilon \big\langle \boldsymbol{a}\_\beta, \sigma\bigl(\boldsymbol{x}\_\beta^{[1]}(i)\bigr)\big\rangle\Big| \leq M$ ?
 3. Where is the activation function in the time derivatives of the parameters?
    E.g. I would have expected the following for the derivative of the parameters in the last layer: $$\frac{\operatorname{d}\\!\boldsymbol{a}\_{u,v,\beta}}{\operatorname{d}\\!t} \approx \frac{1}{n} \sum_{i=1}^n y\_i \cdot \sigma\bigl(\boldsymbol{x}\_{u,v,\beta}^{[1]}(i)\bigr).$$
 4. Why is the supremum necessary in Theorem&nbsp;1?
    Is there a chance that condensation stops again before $T_\mathrm{eff}$?
 5. Do the assumptions correspond to the condensation regime from (Luo et al., 2021)?
    If not, what do the assumptions stand for?
 6. In which exact steps does this analysis differ from the analysis for fully-connected networks?
    It seems like the analyses have a lot in common.
 7. Is condensation a good or a bad thing?
 8. Why are the experiments conducted with networks that attain less than 20% accuracy?
 9. Does the theoretical setting in the experiment section also have 2 fully-connected layers at the end (as claimed in the experimental setup section)?
 10. Can the theory be directly applied to adaptive gradient methods, as used in the experiments?
 11. Does condensation also occur when the last layer is initialised with zeros?
 12. If the results apply when assumptions are violated, shouldn't it be possible to loosen the assumptions?

### Soundness
3 good

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the initial condensation phenomenon of training CNNs, supported by both experimental results and theoretical analysis.

### Strengths
Understanding the training dynamics of gradient-based method is a crucial theoretical issue. While previous research has primarily concentrated on fully connected networks (FCNs), this paper represents a significant advancement in comprehending the training dynamics of CNNs. It investigates the initial condensation dynamics in CNNs, supported by comprehensive experimental evidence. Furthermore, the authors provide a precise mathematical characterization and time estimation for this initial condensation phenomenon in CNN training.

### Weaknesses
Assumption 4, while somewhat strict, effectively explains the initial condensation phenomenon. As discussed in my first quenstion below, I think that by relaxing this assumption, we can gain a more profound insight into the condensation phenomenon.

### Questions
- If Assumption 4 becomes $\lambda_1=\lambda_2>\lambda_3$, how will the initial condensation phenomenon change? My guess is that there will be two condensation directions, corresponding to the first two eigendirections. Is that right?

- In the initial stage of training, if we decompose the dynamics in polar coordinates, the radial velocity is substantially smaller than the tangential velocity. Does the time estimation provided in Theorem 1 relate to this property of the training dynamics?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
