# Flatter, Faster: Scaling Momentum for Optimal Speedup of SGD

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 3, 6

## Abstract
Commonly used optimization algorithms often show a trade-off between good generalization and fast training times. For instance, stochastic gradient descent (SGD) tends to have good generalization; however, adaptive gradient methods have superior training times. Momentum can help accelerate training with SGD, but so far there has been no principled way to select the momentum hyperparameter. Here we study training dynamics arising from the interplay between SGD with label noise and momentum in the training of overparametrized neural networks. We find that scaling the momentum hyperparameter $1-\beta$ with the learning rate to the power of $2/3$ maximally accelerates training, without sacrificing generalization. To analytically derive this result we develop an architecture-independent framework, where the main assumption is the existence of a degenerate manifold of global minimizers, as is natural in overparametrized models. Training dynamics display the emergence of two characteristic timescales that are well-separated for generic values of the hyperparameters. The maximum acceleration of training is reached when these two timescales meet, which in turn determines the scaling limit we propose. Our experiments in matrix-sensing, a 6-layer MLP on FashionMNIST and ResNet-18 on CIFAR10 validate this scaling for the time to convergence, and additionally for the momentum hyperparameter which maximizes generalization.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors present the novel theoretical result on how the momentum hyperparameter and learning rate balance the acceleration of the SGD Momentum (SGDM). The main result is derived from the stochastic process theory which describes the trajectory of the SGDM. Also, three models and datasets are considered to support the obtained roles of learning rate and momentum hyperparameter in moving along the longitudinal and transversal components of trajectory. The balance of this ingredient of fast convergence to flat local minimum is crucial for the runtime of the training process and the generalization of the obtained solution.

### Strengths
I would like to thank the authors for the clear and sufficiently detailed introduction section that provides all the necessary results and ideas. The interpretation of the convergence as a trade-off between two main scales (longitudinal and traversal) looks interesting and promising for further development in the case of adaptive step size optimizers. The presented experimental results in Section 4 confirm the obtained theoretical dependence of the momentum hyperparameter on the learning rate. The presented results can help practitioners to tune hyperparameters more efficiently and reduce the consumption of computational resources.

### Weaknesses
The weaknesses of the presented study are listed below
1) It is very hard to read Section 3 for non-experts in the stochastic process theory. I suggest the authors compress it and extend the section with experiment evaluation.
2) Figure 2 presents fitting line results that confirm the estimated dependence rule, however, I do not find the analysis of the variance of the derived estimate. I am not sure that if one adds more points to the plots, then the dependence is changed significantly.

### Questions
1) The authors consider simple models from computer vision tasks like ResNet18 and MLP. Could you please list the assumptions on the deep neural network that are necessary for the correctness of the derived relation between learning rate and momentum hyperparameter? These assumptions will be very helpful in the extension of the presented results to other models from different domains, for example, transformer-based LLMs.
2) Is it possible to extend the proposed approach from Heavy Ball to a Nesterov-like acceleration scheme? If so, please comment what are the potential obstacles to deriving similar results.
3) The question related to weakness 2:  how robust is the derived estimate w.r.t. the new points that may appear in the plots?
4) How the derived relations between the learning rate and momentum term can be interpreted from the perspective of loss surface properties? What the derived 2/3 power rule can highlight in the loss surface corresponding to the considered models?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper extends the analysis of (Li et al., 2022) to SGDM. Based on such an analysis, the optimal momentum hyperparameter $\beta$ to accelerate the training without hurting the generalization is studied. Experiments of matrix-sensing, a 6-layer MLP on FashionMNIST, and ResNet-18 on CIFAR10 are conducted to support the theoretical claims.

### Strengths
1. It is an important topic to study how the momentum hyperparameter is optimally picked in deep learning, as it allows for shrinking the search space of hyperparameters.

2. The theoretical results (at least the part I have understood) are solid.

### Weaknesses
1. The biggest concern I have is regarding the presentation of this paper. The quantities $\tau_1$ and $\tau_2$ are the focus of this paper. But it is only defined through informal descriptions (for example " Define $\tau_2$ to be the number of time steps it takes so that the displacement of $w_k$ along $\Gamma$ becomes a finite number as we take $\epsilon$ → 0 first, and $\eta$ → 0 afterward") in the introduction, and no (even informal) definition elsewhere. This makes these two terms extremely hard to interpret and understand.

2. This paper only considers optimization around the manifold of the global minima. Although this is inherited from (Li et al., 2022), I wonder whether this framework can characterize the convergence rate along the whole trajectory, which is the one of more interest.  For example, in Figure 2, the initialization is picked where a perfect interpolation has been already achieved. What happens if the initialization is chosen as, for instance, Kaiming's initialization?

3. The experiments are too few and toy considering this paper aims to provide modification for algorithms: there is only an experiment of 2-layer MLP and an experiment over CIFAR 10 with a very uncommon initialization (as discussed above).

### Questions
See weaknesses above.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper analyzes a noisy version of gradient descent (meant to be a proxy of SGD but simpler to analyze theoretically) in the presence of momentum. 
Given momentum parameter \beta and learning rate \eta,  two timescales are identified: 1) one corresponds  to the relaxation time to the Gaussian distribution in the traverse directions to zero loss manifold; 2) the other corresponds to the amount of time it takes to have finite displacements along the zero loss manifold.
The authors argue that the most efficient training is obtained when the two timescales are comparable, which implies a relation among eta and beta. 
In the limits of small noise, small learning rate and large times, the authors derive the SDE of the limit diffusion process. The process is driven toward flat minima (small hessian trace).

### Strengths
- The paper is well written and well organized.
- The paper provides numerical experiments on synthetic and natural settings.
- Both rigorous proofs and heuristics arguments are given.
- The theoretical analysis of the timescales provides the simple prescription gamma=2/3
  that is shown to speed up training and also give the best generalization in some settings.

### Weaknesses
- The work seems a relatively straightforward extension of Li et al. (2022)
- It is not clear if the optimality of the gamma=2/3 exponent applies to standard SGD as well.
- The theory does not give a prescription for the prefactor C. I think this makes it not so useful in practice.

### Questions
- What happens when using standard SGD instead of the label noise one? Can the author provide numerical evidence that gamma=2/3 is still a good exponent? My understanding is that all numerical experiments are carried out with SGD label noise.
- Could the author clarify the argument for the tau1 = tau2 criterium. In particular, it is not clear to me why we should care about traversal equilibration. What seems relevant is to move fast toward the flat region staying close enough to the zero loss manifold.
- What happens if also phase 1 is trained with SGD  instead of GD? The starting point for the following SGD label noise diffusion would be already in a flat region there wouldn't be much drift?
- Is Fig 1 obtained with a single value of \eta and varying \gamma? Could the author show multiple sets of points corresponding to different eta values?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
