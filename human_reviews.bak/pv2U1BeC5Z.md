# Spectral-Bias and Kernel-Task Alignment in Physically Informed Neural Networks

- Decision: Reject
- Scores: 6, 3, 6

## Abstract
Physically informed neural networks (PINNs) are a promising emerging method for solving differential equations. As in many other deep learning approaches, the choice of PINN design and training protocol requires careful craftsmanship. Here, we suggest a comprehensive theoretical framework that sheds light on this important problem. Leveraging an equivalence between infinitely over-parameterized neural networks and Gaussian process regression (GPR), we derive an integro-differential equation that governs PINN prediction in the large data-set limit--- the neurally-informed equation. This equation augments the original one by a kernel term reflecting architecture choices. It allows quantifying implicit bias induced by the network via a spectral decomposition of the source term in the original differential equation. Measuring this spectral bias in practice provides guidelines and insights for architecture design.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper derives the so-called neurally-informed equation (NIE) for a PINN, which characterizes the neural network predictions to the solution of a partial differential equation that takes a kernel term reflecting the prior physical knowledge into account. Based on the NIE, the paper proposes a spectral-bias statement that characterizes the training of an infinite-width PINN.

### Strengths
* The paper theoretically derives a spectral decomposition statement of the PINN output, which is promising in both understanding the dynamics of the PINN throughout training and designing techniques to improve training.

### Weaknesses
* The clarity of the paper needs to be improved. As a theoretical paper, one cannot tell the story by throwing out equations without explaining its assumptions. The two main results (i.e., NIE and spectral bias) need to be formalized into theorems.
* The paper needs more discussion about why the main results are important. In `section 2.1` and `section 2.3`, a lot of focus has been put on discussing how the main results are derived. However, an intuitive explanation of why they are useful is missing.
* As far as I understand, the spectral bias in this paper is based on the assumption that the PINN is infinitely wide. In practice, of course, every NN has a finite width. It is not clear how much things will break down when we look at a finite-width NN.
* Since the main text is 7-page right now, I recommend the authors add more introduction to the background of this work. The latter part of `section 1` goes a bit too fast and cannot be easily comprehended without a solid familiarity with theory of PINNs.

### Questions
* In your spectral bias statement, can you make any general statements about what the eigenvalues and eigenvectors will look like, or is it a very case-specific thing?

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper Leverages an equivalence between infinitely over-parameterized neural networks and Gaussian process regression (GPR) to derive the neurally-informed equation that governs PINN prediction in the large data-set limit. This equation augments the original one by a kernel term reflecting architecture choices. Spectral analysis is also performed.

### Strengths
Derive an alternative form for loss functions for PINN.
Use Several examples to demonstrate the accuracy of equivalent loss.
Perform spectral analysis.

### Weaknesses
1 experiments are too simple.
2 lack of reference, such as 
Neural-network-based approximations for solving partial differential equations. communications in Numerical Methods in Engineering, 1994
Training behavior of deep neural network in frequency domain.
On the spectral bias of deep neural networks.
The convergence rate of neural networks for learned functions of different frequencies.
A fine-grained spectral perspective on neural networks.
On the exact computation of linear frequency principle dynamics and its generalization.
3 writing can be improved, such as deepdxe-->deepxde
4 I cannot see a clear advantage of NIE form

### Questions
why experiments are only 1d simple equations?
what is the advantage of the proposed form?
Fig. 2 is hard to be understood. I also do not know how to use spectral analysis to understand the different results.

### Soundness
2 fair

### Presentation
2 fair

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
The paper considers the task of solving a given PDE via training a physically informed neural network (PINN) and aims at a theoretical description of the solution obtained in this way. To this end, they consider infinitely wide PINN trained until convergence by gradient descent with white noise added to the gradients. Under replica trick and first-order Taylor expansion, the authors show that PINN solution is described by linear integral equation - Neural informed equation (NIE). Then, the authors proceed with revealing a spectral picture behind NIE by identifying another operator whose eigendecomposition decouples PINN learning in independent spectral components. The authors demonstrate their formalism theoretically on a toy example of a simple first-order differential equation on a positive real line, as well as numerically for 1D heat equation with a specific source term.

### Strengths
The main results of the paper - NIE equation (5) and its spectral decomposition (15,16) look like a solid development that can become a basis for future analysis of the PINN behavior. Importantly, the approximations done when deriving NIE may indeed be negligible, as demonstrated by a good agreement between direct simulation and NIE solution for big dataset sizes in Figure 1. 

My score would be significantly higher if not for the presentation issues described below.

### Weaknesses
The main weakness of the paper is not clear enough presentation, making the accurate reading of the paper (e.g. by trying to verify theoretical results) quite difficult. Given the almost 2 unspent pages of the main paper and quite the small size of the current appendix, there is a lot of room for giving the necessary details.   

**The setting** is not described clearly and in full detail. For example, the authors mention that posterior (3,4) describes infinitely wide NN trained by Gradient Descent with white noise the gradients to the gradients weight decay. Apparently, this setting follows [Naveh2021], but this is not mentioned explicitly by the authors, and the respective reference is mentioned together with classical NTK work [Jacot2018]. Since the setting with added white noise to the gradient is much less common compared to that of [Jacot2018], it may take a significant time to figure out what the authors actually mean. Moreover, as mentioned by [Naveh2021], the considered setting is equivalent to Bayesian training with NNGP kernel, adding more options for confusion (see, for example, these two references at the end of page 3).

**Derivations** often lack comments and/or intermediate steps required for their understanding. A few examples with associated questions are
1. While it is expected that the result of [Naveh2021] for standard fully-connected networks carries over to the physically-informed networks and resulting in the posterior in eq. (3,4), the clarity of presentation would be improved if the authors would give the reference for the respective statement (if it was obtained previously) or formulate and prove a proposition (even if it turns out to be quite simple). 
2. Derivation of eq. (5) uses the replica trick to evaluate the average of possible choices of the training dataset $\mathcal{D}_n$. I would expect that the replica trick is a technique that a typical member of ICLR community does not known by default, so at least a reference or a short description of the technique is needed to make derivation accessible to a broader audience.  
3. Is the last transition in (31) correct? Somehow, it takes sum over replica modes out of the inner exponent in the action, thus decoupling the modes. It seems that decoupling is only possible after Taylor expands this exponent up to the first order (as also mentioned by the authors after eq. (33)).
4. After decoupling replica modes, the authors continue to work with the effective action (33) of a single mode. This raises a number of questions: (i) whether the replica limit $p\to0$ was taken implicitly at some point? (ii) why does this effective action describe the (data averaged) posterior distribution of $f$ we are interested in?   
5. The authors first state that they consider uniform data distribution in the bulk $\Omega$, but in the toy example, consider $\Omega=\mathbb{R}_+$ with $|\Omega|=\infty$ making uniform distribution on $\Omega$ not well defined. How is this issue resolved?
6. Calculation behind eq. (9) are not trivial and involve additional assumptions, e.g. vanishing of the boundary $x=\infty$ term during integration by parts.
7. Contour integration performed to obtain Green function $G(x,x')$ involves an approximation of taking into account only a single pole. This is mentioned in a few lines of text without even specifying which of the poles dominates the result. More detailed justification is required for this approximation to be acceptable (e.g. providing numerical evidence of the single pole dominating the integral). 
8. I could not understand how NIE equation (5) can be rewritten in the form (46) without integration by parts. Could you please provide more details?

As can be seen from the above, there are a number of assumptions and approximations performed in the paper. More careful commenting and/or validation of these assumptions would make the claims of the paper more verifiable. A separate example of that is the statement that the proposed approach can be extended to the nonlinear operators $L$. However, this claim is supported only by a single line at the end of Appendix B.2, which hints at a potential derivation strategy for nonlinear operators $L$. If not described in more detail (e.g. by providing execution of the derivation strategy suggested by the authors), this claim seems suitable only for the discussion section (as an example of a direction for future work). 

Regarding the plots, there is a small issue of visibility of axes and colormap ticks values on the inset heatmap plots from Figure 2, which are required to support the benefit of stronger concentration of cumulative spectral functions. At the moment, the only way to see them is with extremely high zoom and good eyesight. 

A few (potential) minor typos:
- Tilde is missing in the right-hand side of the reparametrization of variation from $h$ to $\widetilde{h}$ (eq. after eq. (35)).
- Is there an extra index $k$ in the definition of cumulative spectral function (e.g. it should be $A_k[Q,f]=\|P_k f\|^2 / \|f\|^2$)?

[Naveh2021] Gadi Naveh, Oded Ben David, Haim Sompolinsky, and Zohar Ringel. Predicting the outputs of finite deep neural networks trained with noisy gradients.

[Jacot2018] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks.

### Questions
Please refer to the above section.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
