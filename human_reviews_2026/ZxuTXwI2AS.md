# KO: Kinetics-inspired Neural Optimizer with PDE Simulation Approaches

- Decision: Reject
- Scores: 4, 8, 2, 6

## Abstract
The design of optimization algorithms for neural networks remains a critical challenge, with most existing methods relying on merely heuristic adaptations of traditional gradient-based approaches. This paper introduces KO (Kinetics-inspired Optimizer), a novel neural optimizer gadget inspired by kinetic theory and partial differential equation (PDE) simulations. KO can be used with multiple types of base optimizers (e.g., Adam, SGD). In KO, the training dynamics of network parameters are perceived as the evolution of a particle system, where parameter updates are simulated via a numerical scheme for the Boltzmann transport equation (BTE) that models stochastic particle collisions. This physics-driven approach inherently promotes parameter diversity during optimization, mitigating the phenomenon of parameter condensation, i.e. collapse of network parameters into low-dimensional subspaces. Parameter condensation is proven harmful to model generalizability. We analyze KO's impact on parameter diversity, establishing both a strict mathematical proof and a physical interpretation. The convergence of the proposed optimizer can also be guaranteed. Extensive experiments on image classification (CIFAR-10/100, ImageNet) and text classification (IMDB, Snips) tasks demonstrate that KO consistently outperforms baseline optimizers, achieving accuracy improvements while remaining comparable computation cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a method to improve optimization of neural network trough a combination of the gradient with a mechanism that encourages neurons to learn different features. This mechanism is inspired by the kinetic theory of ideal gases from physics. Experiments were conducted on a synthetic problem, image and text datasets.

### Strengths
1) The authors present an interesting connection of neural network optimization to kinetic gas theory.

2) The authors present a small example on a synthetic dataset, which attempts to give some insights and illustration of their method beyond the usual evaluation on large-scale tasks. 

3) The paper states that there are improvements in the experiments through their method. However, I am unsure how valuable improvements of ~0.5% in classification accuracy are to computer vision or NLP communities.

### Weaknesses
1) For me, the paper was not easy to read and I think I would not be able to implement the method from the paper.
- More formal definitions of terms like "parameter diversity", "parameter condensation". "distribution of network parameters" (What are the different samples? Parameters of neurons, of layers?)
- Often the phrasing is vague "We introduce collisions at the gradient level" (L190)
- What's often helpful to explain analogies is a table that makes the connections explicit (e.g. weight vector of one neuron = particle) 


2) Some practical aspects should be discussed in more detail:
- Does the method require additional hyperparameters? Do they need additional tuning?
- How does the computational cost scale with the number of parameters?

3) Limited experimental evaluation: The authors should compare their method to other methods that also aim to manipulate the weights to improve generalization, such as weight decay. 

4) The goal of avoiding similar features in neurons to improve generalization, which is the basis of this work, appears to be in contradiction to more established views on generalization like the one in [1]. Can you discuss if and to what extent this is indeed true and what shortcomings you see in the perspective in [1]?

[1] R. S. Sutton, A. G. Barto "Reinforcement Learning, An Introduction" (Section 9.5.3 and Figure 9.7 Generalization in linear function approximation...)

### Questions
- See also my questions in "weaknesses"
- Why was the network on the synthetic dataset initialized with weights near zero? Does the experiment still produce the same results with the usual initilization schemes for gradient-based neural network optimization?
- Please fix equation references in L215

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a new method for training the weights of neural network architectures for tasks like image classification. The method aims to address the issue that standard gradient based methods can lead to layers with neurons in the same lower dimensional subspaces, and this leads to worse model generalization. The training method works by adjusting the gradient following an approach motivated by physics: particle collisions. Theoretical and numerical results support the finidings.

### Strengths
- The paper introduces a new training method that adjusts the gradients during updates, using ideas inspired from physics - particle collisions. The method is quite agnostic to the rest of the training process and can be employed in many training frameworks and optimizers.

- Numerically, the results are very convincing in improving the performance of other optimizers.

- Theory justifies these results - reducing the layer weight correlation.

- Computationally the method is not heavy.

### Weaknesses
- In terms of comparisons, I would have expected that the authors compare more clearly with regularization based approaches. In the introduction, the latter are described as post-hoc, but I am not convinced about this. In principle, regularization addresses the same issue, poor generalization.

- Convergence is proven, but it is not clear what is the impact of the proposed approach. I expect that the introduced 'collisions' cause the gradient directions to change randomly, and as a result, creating more variance in some cases?

### Questions
- I wonder how the 'collision' setup should change across the depth of the neural network, as well as during training. This aspect is not clear. Should more collisions happen earlier during training, or in earlier layers? 

- The method is argued to increase the entropy of the weights from one step to the next one. How does this compare with entropy being used as a regularization term? This relates to my comment earlier about unclear comparison with regularization based approaches

- The improvements are higher in less complex classification problems and also smaller neural networks. How is this justified?

- Are the obtained networks more susceptible to adversarial perturbations? Intuitively, maybe yes, since they have more entropy/randomness/high dimensional subspaces. This would be nice to show but not strictly a drawback

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a physics-inspired optimizer by considering that weights are particles. The idea is to use particle collision dynamics to modify gradients such that the weights/neuron condensation is reduced. The results show marginal improvements.

### Strengths
1. The idea of using particle collision dynamics to reduce weight condensation is interesting and might be impactful.
2. Since the idea only modifies the gradients, it could be used with existing optimizers.
3. The results on classification tasks show marginal but consistent improvements on both validation accuracy and weight condensation.

### Weaknesses
1. The method is proposed as a way to reduce neuron condensation; however, the modification is only applied to the gradients and not the updated weights. This is important as weights are initialized randomly, so they are all different, and just using gradient similarity to modify them does not make sense to me. Please clarify this.
2. Weight condensation and neuron condensation are used interchangeably; however, they might mean different things. Please clarify. Also, please clarify how cosine similarity is used as a metric for weight condensation because cosine similarity can only be computed between two vectors.
3. Weight decay might encourage the parameters to become low rank [1], which is contradictory to what is written in line 45.
4. The improvements are marginal, and this casts doubts on how useful the proposed physics-inspired modifications are.

[1] Súkeník, Peter, Christoph Lampert, and Marco Mondelli. "Neural collapse vs. low-rank bias: Is deep neural collapse really optimal?." Advances in Neural Information Processing Systems 37 (2024): 138250-138288.

### Questions
1. What is meant by: "Weight correlation is defined as the abstract sum of the network weights’ cosine similarity matrix."? Please provide the equation.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces KO (Kinetics-inspired Optimizer), a neural optimization method motivated by kinetic theory and PDE simulations. The authors provide theoretical analysis and empirical results that together support the effectiveness of the proposed approach.

### Strengths
1. The collision-inspired mechanism is conceptually interesting and plausibly bridges ideas from kinetic theory to practical optimization.
2. The paper includes both proofs and experiments, which helps substantiate the method’s validity.

### Weaknesses
There are shortcomings in the manuscript’s writing standards and in the completeness of the experiments. See the following questions.

### Questions
1. There are severe issues in writing and formatting, for example at lines 164, 177, 215, 246, 248, 295, and 363.

2. In Appendix Table 4, compared with the original network training, the relative time increase for ResNet-34 + H.C. is markedly higher than for ResNet-18 + H.C. and ResNet-50 + H.C., which appears unreasonable; please explain.

3. In Appendix Table 5, why is there no runtime comparison for the CIFAR-100 dataset?

4. How is coll_coef selected in the experiments, and on what basis? Please add ablation studies on hyperparameters, such as coll_coef.

5. In the Related Work section, the literature is discussed only up to 2022 and earlier; please add the latest works. The experiments should include comparisons with methods of the same type; if none are available, please state so.

6. Collision is an interesting idea. If, instead of collisions, one uses short-range repulsion and long-range attraction, how would this compare with collisions? No additional experiments are needed, but please provide a discussion.

7. Please add comparative experiments under initial learning rates of different orders of magnitude, not only the current initial learning rate of 0.1.

### Soundness
4

### Presentation
2

### Contribution
3
