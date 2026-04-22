# Implicit Regularization Through Hidden Diversity in Neural Networks

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
A significant body of work has focused on studying the mechanisms behind the implicit regularization in neural networks. Recently, developments in ensemble theory have demonstrated that, for a wide variety of loss functions, the expected risk of the ensemble can be decomposed into a bias and variance term together with an additional term called *diversity*. By using this theoretical framework and by interpreting a *single* neural network as an ensemble, we expose a hidden diversity term in the decomposition of a neural network's expected risk. We argue that the additional diversity term regulates the variance error, thus identifying a new source of *implicit regularization* in neural networks. We demonstrate this regularization on regression and classification datasets by estimating the bias, variance, and diversity terms for both MLPs and CNNs. Using double descent as an example, we observe that diversity significantly increases for wide overparameterized neural networks. These results demonstrate a new perspective on implicit regularization in neural networks and open new possible avenues of research into their generalization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
- The paper proposes writing a single neural network with a dense final layer, as a weighted sum of subpredictors, where the weights are the rows of the logit layer scaled by a scalar multiplier depending on the initialization scheme.
- It applies Wood et al.'s decomposition to these subpredictors, viewing them as an ensemble of learners that exhibit a "diversity" term.
- The experiments suggest that, if I understood correctly, the risk of a single neural network can be estimated by computing the bias, variance, and diversity terms on the subpredictors.

### Strengths
If I understood correctly, this work presents a novel application of Wood et al.'s decomposition to a single neural network, offering interesting insights into the generalization properties of deep nets. This is particularly relevant given the widespread use of deep learning models and the lack of consensus on the mechanisms underlying good generalization. It provides an additional perspective on this important topic, using quantities that are simple to compute.

### Weaknesses
- The paper appears to be a shortened version of a journal article, which may be better suited for a journal submission. For an ICLR submission, more careful editorial choice could be applied to decide what content belongs in the main body versus the appendix.
- The main body of the paper is not fully self-contained, with missing details. For example, it is not clearly stated what is plotted in Figure 3, which I believe is the primary experimental result. As a result, I largely had to guess what exactly is the takeaway, what and how exactly are the quantities computed. 
- There is an excessive focus on irrelevant details, such as in section 3.2, where the general case for centroids is explained, despite only requiring a simple discrete weighted average  for both MSE loss and KL divergence in the rest of the paper.
- The paper lacks explicit actionable takeaways. For instance, it could be beneficial to directly incorporate diversity as a regularizer during training, making this concept more practical and accessible.

### Questions
1. In Section 3.3, Theorem 1: What is y, as defined in Section 3.1? What are q_s ? Are they densities as described in Section 3.2?
2. Line 261: Why is it necessary for the coefficients to satisfy \sum \beta = 1 ? From Section 3 alone, it is not clear why the ( p(q_{(i)}) ) coefficients should form a valid probability mass function rather than any arbitrary set of weights. Is it actually relevant in the context of the paper?
3. Line 276: h_{(1)} depends on the training set, but it is never explicitly mentioned how up to that point. I assume the models are trained, but this should be clearly stated. Are different training sets used since, in the expectations further below (Eq. 10), D is a random variable?
4. What exactly is plotted in Figure 3 (right)? How are all the terms computed? Is it (option 1) the same training subsets as in Figure 3 (left), or (option 2) are these estimated quantities over subpredictors? If option 2, are the subpredictors trained on the entire training set, while the estimators in the left figures are trained only on subsets? Is there a scaling factor between quantities on the left and their estimators on the right ?

Minor comments:
===============
5. Line 256: q^i versus q_{(i)} — why the change from subscript to superscript?
6. Section 3.1: The indices i are used both for training examples and subpredictors, which causes confusion (e.g., the formula for R_{emp} on line 117 does not make sense). It is also used in d_i, where I assume it refers to input space dimension.
7. Line 376: "It is known that width is the primary factor for good network performance" — this should be toned down, as for example, large language models often require increased depth rather than just increased width.
8. Line 849 (Appendix): "Importantly, we use the same seed to initialize the neural network weights on each trial set." If this is an important point, it should be discussed in the main body: what happens when seeds are not fixed?
9. Experiments are a bit toyish. Ideally, it would be interesting to include experiments on more realistic datasets, even just a ResNet on CIFAR-10 rather than fully convolutional neural networks.

### Soundness
4

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper finds that by interpreting single neural networks as implicit ensembles, one can use existing decompositions for the risk of ensembles into a bias, variance and diversity term to explain the expected risk of these single networks. They then argue that the diversity term in this decomposition of the risk of a neural network forms a new source of implicit regularization regulating the variance term. Finally, they empirically investigate this regularization in experiments using CNNs and MLPs.

### Strengths
The paper makes use of the elegant idea to interpret single neural networks as implicit ensembles, thereby being able to make use of already developed theory for ensembles to analyze single networks. Using this approach, they can in particular identify the diversity of the subpredictors of the model as a new source of implicit regularization, which is a relevant finding. 
More generally, the story of the paper was relatively clear and the paper was well-organized. Using the examples of the square loss and the KL divergence throughout the paper helped a lot in being able to understand the introduced concepts better.

### Weaknesses
One concern I have is that the paper does not provide substantially new insights beyond applying the existing theory from Wood et al. (2023) to subpredictors of a single neural network. I would have been interested in seeing slightly more discussion on where they provide new insights beyond the results from Wood et al. (2023) throughout the paper. 
Furthermore, most of the discussion in Section 4.4. seemed relatively speculative (e.g., multiple 'we hypothesize') and I would have been interested in seeing these hypotheses being empirically tested more explicitly.

### Questions
1. I assume that the Bias-variance-diversity decomposition (Theorem 1) does not need independence between the subpredictors (since this would not be fulfilled for the subpredictors in your model). Could you expand on why independence is not necessary here and how non-independence affects the different subterms?
2. How are the Bias and Variance terms in your decomposition in Equation (10) and (12) related to a more classical Bias-Variance Decomposition?
3. Could you clarify in more detail which points mentioned in the discussion in Section 4.4. you think have been validated by your experiments and how you would test any other hypotheses that remain?
4. Do you think the following is generally true (even for subpredictors in large networks):
> On their own, these subpredictors are relatively simple models (a hidden node multiplied by a weight) and each subpredictor will likely have a high bias error.
5. Could you explain what you mean by this sentence in more detail/why you made this choice:
> To minimize some of the implicit regularization effects due to mini-batch SGD (Smith et al., 2021), we make use of full batch gradient descent as an optimizer.
6. What is your explanation for why the diversity term is tracking the variance term so closely (e.g., in Figure 1)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work reinterprets a single neural network as an implicit ensemble, drawing upon existing literature on parameterizations that incorporate output (or logit) scaling factors, as well as on diversity theory.

### Strengths
- Section 3 provides a well-organized summary of the essential components from Wood et al. (2023), which helps readers grasp the necessary background.
- It is quite interesting to incorporate the output (or logit) scaling factors from MFP, SP, or muP into the ensemble combiner. This reminds me of Kirsch et al. (2025), where a slight connection between implicit ensembles and the NTK was discussed.
- I enjoyed the “ensemble” decomposition of a “single” model presented in Section 4.3. In the past, there have been discussions in the context of CNNs suggesting that the final average pooling layer could be interpreted as a form of implicit ensemble; the formulation here feels much more direct and follows the well-defined diversity decomposition of Wood et al. (2023).

---
- Kirsch et al. (2025), (Implicit) Ensembles of Ensembles: Epistemic Uncertainty Collapse in Large Models.

### Weaknesses
- The limitation, as also acknowledged by the authors in Appendix A.1, is that the analysis is restricted to networks whose final component is an MLP with ReLU activations. In practice, modern neural network architectures (yes, there’s really only one nowadays, transformers) do not typically fit this assumption, which makes this a clear weakness of the work. That said, it still offers a valuable perspective, so I wouldn’t consider this a major flaw.

### Questions
- wrong left quotation mark in line 89.

- One notable point in Wood et al. (2023) is that they consider (pre-softmax) logit ensembling for classification models. I’ve often felt that this differs somewhat from the common practice of performing (post-softmax) probability ensembling. While logit ensembling is sometimes used, from a Bayesian perspective, if we think of modeling the categorical predictive distribution, (post-softmax) probability ensembling would be the more appropriate formulation in the context of Bayesian model averaging. I wonder, though, whether a similar line of reasoning could be extended to justify post-softmax ensembling as well; what are your thoughts on that?

- One interesting aspect of neural network ensembles is that training individual members separately and then combining them often leads to different outcomes compared to training them jointly in an ensemble form from the start (Allen-Zhu and Li, 2023). The ensemble considered in this work corresponds to the latter case. I wonder whether similar results would still hold if the former approach were taken instead.

---
- Allen-Zhu and Li (2025), Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning.

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
3

### Summary
- This work combines two lines of work on understanding implicit regularization in deep neural networks and studying ensembles from the perspective of bias, variance and diversity terms. This work considers a single neural network as an ensemble of multiple neural networks and hence, breaks down the loss like ensembles, They connect the diversity term to an additional form of implicit regularization in neural networks. They also show that this diversity term is large for overparmeterized neural networks and can explain the double decent framework.

### Strengths
- The paper overall is well written.
- The paper combines two different lines of work on ensembles and implicit regularization in NNs. I find the study of implicit regularization through the ensemble loss decomposition interesting and novel.

### Weaknesses
- Although the work provides interesting connections, the main theorem 1 is taken from previous work and does not provide new theoretical contribution.
- In this work, the authors have considered one way of decomposing the model into multiple subnetworks along the last layer. But, would the results change if we use a different decomposition?

### Questions
- In this work, the authors have considered one way of decomposing the model into multiple subnetworks along the last layer. But, would the results change if we use a different decomposition?
- It seems like in all the experiments, diversity has the same behavior as variance. Can the authors suggest a case where this is not true?
- Does this diversity has any connections to any other form of implicit regularization that is classically studied in this literature?

### Soundness
2

### Presentation
3

### Contribution
2
