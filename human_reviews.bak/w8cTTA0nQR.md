# Can Neural Networks Improve Classical Optimization of Inverse Problems?

- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 3, 6, 3

## Abstract
Finding the values of model parameters from data is an essential task in science. While iterative optimization algorithms like BFGS can find solutions to inverse problems with machine precision for simple problems, their reliance on local information limits their effectiveness for complex problems involving local minima, chaos, or zero-gradient regions.

This study explores the potential for overcoming these limitations by jointly optimizing multiple examples. To achieve this, we employ neural networks to reparameterize the solution space and leverage the training procedure as an alternative to classical optimization.
This approach is as versatile as traditional optimizers and does not require additional information about the inverse problems, meaning it can be added to existing general-purpose optimization libraries. We evaluate the effectiveness of this approach by comparing it to traditional optimization on various inverse problems involving complex physical systems, such as the incompressible Navier-Stokes equations. Our findings reveal significant improvements in the accuracy of the obtained solutions.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to use neural networks instead of iterative optimisation algorithms (such as limited-memory BFGS or Gauss-Newton) to solve general inverse problems.  The role of the neural networks is to infer the parameters of the system directly from observations. The paper formulates inverse problems as supervised machine learning problems.  The approach is experimentally validated on six different inverse problems.

### Strengths
The paper addresses an interesting topic.

### Weaknesses
The main weakness of this paper is its lack of novelty. 

The authors cite a number of related papers and claim: "However, since these approaches rely on loss terms formulated with neural network derivatives, they are not applicable to general inverse problems". 

I think the authors ignore a lot of work that is directly related to the question of solving inverse problems with neural networks. I mention some of them below, but there are many more.  Compared to this literature, it's hard to find any originality in the proposed method. In any case, if there is any originality, it should be specified on the basis of this (not cited) literature on inverse problems with neural networks.

Aggarwal, H.K., Mani, M.P., Jacob, M., 2019. MoDL: Model Based Deep Learning Architecture for Inverse Problems. IEEE Trans. Med. Imaging 38, 394–405. https://doi.org/10.1109/TMI.2018.2865356

Lucas, A., Iliadis, M., Molina, R., Katsaggelos, A.K., 2018. Using deep neural networks for inverse problems in imaging: Beyond analytical methods. IEEE Signal Process. Mag. 35, 20–36. https://doi.org/10.1109/MSP.2017.2760358

Mukherjee, S., Carioni, M., Öktem, O., Schönlieb, C.-B., 2021. End-to-end reconstruction meets data-driven regularization for inverse problems, in: NeurIPS.

Ongie, G., Jalal, A., Metzler, C.A., Baraniuk, R.G., Dimakis, A.G., Willett, R., 2020. Deep Learning Techniques for Inverse Problems in Imaging.

Peng, P., Jalali, S., Yuan, X., 2020. Solving Inverse Problems via Auto-Encoders. IEEE Journal on Selected Areas in Information Theory 1, 312–323. https://doi.org/10.1109/JSAIT.2020.2983643

### Questions
What is new about this approach compared to the literature on neural networks for inverse problems?

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents several algorithms for solving optimization of inverse problems using neural networks, including (1) supervised learning using simulated data; (2) reparameterization using untrained neural network as an implicit prior, and (3) neural adjoint that approximate the forward process with the neural network by training the network on simulated data, and then use the pre-trained network in the optimization problem solver.

### Strengths
Various proposed algorithms covering different usage of neural network in the optimization of inverse problems. Various of inverse problems in experiments.

### Weaknesses
My main concerns are about the contributions of this paper:

Form the method aspect:
- The supervised learning has been widely used for decades. 
- Reparameterization shares the same concept with well-known deep image prior. This paper claims that “these effects have yet to be investigated for general inverse problems or in the context of joint optimization”. Shouldn’t the imaging optimization belongs to the an instance of general inverse problem? Could the author also clarify the meaning of joint optimization, and if the proposed reparameterization has been applied to joint optimization in this work?
- For the neural adjoint, I cannot follow why do we need to approximate the forward process with the neural network as it is usually assumed to be known in the inverse problem.

From the application aspect: 
Experiments in this paper are all in small scale and both toy problems. How does the proposed method work in the real-world problem.

In short, I think this paper tries to use a unified notation to formulate the inverse problems, introduces existing works under these notations, and validate the existing works under various toy problems. I think the novelty of this paper is limited.

### Questions
See the weakness.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Iterative optimization algorithms can find solutions to inverse problems for simple problems, but suffer from reliance on local information. Thus their effectiveness for complex problems involving local minima, chaos, or zero-gradient regions is severely limited. In the study, the authors employ neural networks to reparameterize the solution space and leverage the training procedure as an alternative to classical optimization. Numerical experiments demonstrate that the neural networks can indeed improve the accuracy of solutions.

### Strengths
The idea of introducing a parameter \theta to reparameterize the problem is interesting. Involving neural networks to the procedure is also effective in optimizing the objective function. Extensive simulations illustrate the promising performance of the method.

### Weaknesses
In simulations, the graph Figure 2 (c) is not very informative, and in fact, a little confusing. From the graph, it seems that "neural adjoint" has larger loss than "BFGS", but the explanation below says that "The neural adjoint method finds better solutions than BFGS for about a third of examples for n = 256". For me, it looks like that the explanation is inconsistent with the graph. Please explain this or clarify it in the paper.

### Questions
No question.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This manuscript proposes a neural network-based framework for solving multiple inverse problems in a joint fashion. The core idea is to parameterize the inverse mapping as a neural network and then optimize the model parameters by minimizing a mismatch loss. This loss is defined based on $n$ pairs of data points and a known forward model. This method is referred to as the "reparameterized method," and the paper argues that leveraging information across multiple inverse problems can mitigate the challenging landscapes often encountered in these problems.

This approach shares similarities with amortized optimization methods widely studied in contexts like variational inference and stochastic control/reinforcement learning, as discussed in the recent review "Tutorial on Amortized Optimization" (arXiv: 2202.00665). This is the first work I have seen that applies this idea to inverse problems, although I am not sure whether existing work has already explored this straightforward extension in the context of amortized optimization.

However, even if we put aside the paper's novelty, it still suffers from various technical and presentation issues. Specifically, the paper's setting and methodology are not clearly explained, and the rationale for its comparisons with other neural network methods is not well justified.

### Strengths
The paper empirically demonstrates that the reparameterized approach, when augmented with BFGS refinement, delivers better solutions compared to the standard BFGS method for inverse problems.

### Weaknesses
Major Comments:
1. According to the introduction and conclusion, my understanding is that the paper's central thesis posits the reparameterized method as a more effective alternative to vanilla BFGS, particularly when enhanced with BFGS refinement, for solving multiple inverse problems. However, the paper, which has a very broad title, also delves heavily into comparisons with two other neural network approaches: supervised learning and the neural adjoint method. This leaves me uncertain: whether the authors aim to primarily promote the reparameterized method or to provide a broader discussion of various neural network-based strategies for inverse problems.

2. While I believe it is justifiable to compare the BFGS and reparameterized methods, I question the fairness of comparing the reparameterized approach with supervised learning and the neural adjoint method. This is because the latter two necessitate an additional dataset for pre-training, unlike the reparameterized method. Following this point, I am extremely confused by the meanings of subfigures (c) and (d) in the experiments.

   2(a) What exactly does "loss" mean on the y-axis of subfigure (c)? Does this term refer to specific error metrics (in terms of L?) related to the obtained solutions? Initially, the term "loss" would seem to indicate the objective functions associated with the different methods. However, supervised learning and the neural adjoint method have their own unique loss functions for training the neural networks, making direct comparison with the loss in the reparameterized method unclear.

    2(b) Similarly, what is meant by "dataset size" in subfigure (d) for the supervised learning and neural adjoint methods? Does it refer to the size of the training dataset for the networks, or the number of inverse problems to be solved? If it's the former, the numbers appear too small. If it's the latter, the confusion persists: during the problem-solving phase, methods like BFGS, supervised learning, and the neural adjoint solve each individual problem independently. The rationale for considering L/n across different n values is unclear, as these would merely represent the same statistics, recalculated from different sample sizes.

3. The paper introduces the reparameterized method under the assumption that $x_i$ is known. However, some experiments operate in settings where $x_i$ is unknown. I do not see a straightforward way to extend the method to this new context, given that the forward model relies on both $x_i$ and $\xi_i$ to produce $y_i$. This omission leaves a critical gap in the technical exposition.

4. On page 4, within the paragraph discussing supervised learning, it is inaccurately stated that "if we additionally have a prior on the solution space $P(\xi)$, we can generate synthetic training data." What is actually needed is knowledge of the joint distribution of $(x, \xi)$.

5. On page 5, in the first paragraph, the expression "$L(F(\xi|x_i), y_i)$" is used. Should $F$ actually be$\tilde{F}$? Otherwise, I do not understand how the neural adjoint method works.

Minor Issues:
1. In the second line below Figure 2, the mathematical expression appears to contain an error.

2. The conclusion asserts that this is "the only network-based method that does not require domain-specific information." The term "domain-specific information" is ambiguous here. To utilize the proposed reparameterized method, a forward model $F$ is necessary, which could reasonably be categorized as domain-specific information.

### Questions
See questions above

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
