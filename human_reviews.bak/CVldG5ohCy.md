# Adam through a Second-Order Lens

- Decision: Reject
- Scores: 1, 6, 3, 6

## Abstract
Research into optimisation for deep learning is characterised by a tension between the computational efficiency of first-order, gradient-based methods (such as SGD and Adam) and the theoretical efficiency of second-order, curvature-based methods (such as quasi-Newton methods and K-FAC). We seek to combine the benefits of both approaches into a single computationally-efficient algorithm. Noting that second-order methods often depend on stabilising heuristics (such as Levenberg-Marquardt damping), we propose AdamQLR: an optimiser combining damping and learning rate selection techniques from K-FAC (Martens and Grosse, 2015) with the update directions proposed by Adam, inspired by considering Adam through a second-order lens. We evaluate AdamQLR on a range of regression and classification tasks at various scales, achieving competitive generalisation performance vs runtime.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes AdamQLR, which is a modification of the Adam optimizer and tries to adapt some heuristics used in the K-FAC optimizer for Adam, such as Levenberg Marquardt damping (equation (2) in the paper) and learning rate selection (equation 3 in the paper), both based on a truncated second order Taylor expansion of the function computed by the neural network at the current parameters $\theta_t$ (equation 1 in the paper). The authors perform experiments on 6 tasks (Rosenbrock, UCI Energy/Protein, Fashion-MNIST, SVHN and CIFAR-10) and they compare two versions of their work (Adam QLR Tuned/Untuned) against a few popular optimizers in the literature (SGD Minimal/Full, Adam, K-FAC).

### Strengths
Below I enumerate the strengths of the paper:
1. clearly written and easy to understand
2. code is provided
3. optimal hyper-parameters are clearly stated in Table 2
4. ablation study for Levenberg-Marquardt heuristic

### Weaknesses
The paper lacks novelty and originality because it only combines Adam and K-FAC in a facile way and thus doesn't provide better results for most of the tasks mentioned, such as UCI Energi/Protein, Fashion-MNIST and CIFAR-10 (in these cases, K-FAC and/or original Adam are better than the proposed method because the generalization performance VS runtime is not competitive as stated in the abstract).

The evaluation was performed on small tasks and I believe that the usage of Rosenbrock function not adding any value to the paper since the tasks that involve Neural Networks are much more complicated. The paper does not have any tables that contains accuracies for classification tasks and it is extremely difficult to figure out what the final accuracies are only by looking at the plots. In the end, it is unfortunate to say that the paper does not meet the novelty and originality requirements for ICLR.

### Questions
1. how does AdamQLR behave on NLP tasks?
2. how do you compute the curvature matrix C that is used to update the learning rate $\alpha$ and the damping factor $\lambda$? In the manuscript you state that the overhead is only one additional forward pass, while we all know that computing Hessian-vector-products requires an additional backward pass (which, of course, implies a forward pass in the first place)

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work refers to Adam and proposes to adaptively adjust the learning rate. Specifically, the authors utilize $\rho$ to denote the ratio between the difference of true loss function $f()$ and the difference of second-order estimation $M()$. Then the authors refine the estimated Hessian matrix through $\lambda$ according to $\rho$. Finally, the learning rate is then computed by minimizing $M(\theta - \alpha d)$.

### Strengths
1.	The method makes sense.

2.	Extensive experiments show the effectiveness of the method.

### Weaknesses
1.	I wonder how to get the matrix $C$ in Eq. 1.

2.	What is the principle of setting $\omega_{dec}$ and $\omega_{inc}$, and why $\lambda$ is adjusted when $\rho$ larger than 3/4 or smaller than 1/4?

### Questions
Please see the weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper authors propose some symbiosis of two optimization methods: Adam and K-FAC. They combine damping and learning rate selection techniques from K-FAC and use it inside Adam algorithm. The resulting algorithm, called AdamQLR, is then evaluated on different regression and classification tasks.

### Strengths
1. Lots of numerical experiments.
2. Clear description of algorithm modification.
3. Good description of the motivation of the heuristics, adopted from K-FAC.
4. Description of the experimental setup and hyperparameter search space.

### Weaknesses
From theoretical point of view, the result seems insignificant. You took some heuristics, that improve the model, and moved it to another model. There is no evidence, that it should work better in theory. From practical point of view, as far as I understand, the number of hyperparameters increased: $\beta_1, \beta_2, \varepsilon$ for Adam vs $\beta_1, \beta_2, \varepsilon, \lambda$ for AdamQLR (or even $w_{dec}, w_{inc}$ instead of $\lambda$.

### Questions
Rosenbrock function example seems unfair, because you use Hessian there, what do you think?

### Soundness
1 poor

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper tried to combine the first-order method (such as Adam) with the second-order methods, such as K-FAC. More specifically, the authors propose a novel optimizer AdamQLR: combining damping and learning rate selection techniques of K-FAC. The experimental results illustrate that the proposed method AdamQLR can achieve competitive generalisation performance and training efficiency.

### Strengths
1. The idea of combining first-order and second-order methods is very interesting. In addition, the research direction is also very important. 
2. The proposed method is very easy to understand. I think we should pay more attention to second-order method and improve its efficiency.

### Weaknesses
1. I think the main results are from the figure 2. But the figure is not very clear for me, maybe you can list the training loss, test loss, convergence steps, and generalization gap ( |training_loss - test_loss| ) in a table. From this figure. I'm not very clear whether the proposed method can solve the overfitting issue and improve the generalization. So I think you can analyze the generalization gap. 

2. The experimental results are not very strong for me. Although the proposed method can achieve fast convergence and lower test loss, their performance is still too close. In addition, you try to analyze training loss and test loss in figure 2. But loss value is not a great metric for classification tasks and I think you should show the accuracy. 

3. The training task is too simple and the results on complex tasks (such as ImageNet) is not very strong.

### Questions
1. Loss value in figure 2 is not great enough to compare the performance of different methods and maybe you should provide the accuracy value.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
