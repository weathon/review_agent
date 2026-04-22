# On the Dynamics of Learning Linear Functions with Neural Networks

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 4, 4, 4, 4

## Abstract
This paper studies the gradient descent training dynamics of fitting a one-hidden-layer network with multi-dimensional outputs to linear target functions. That is, we focus on a realizable model where the inputs are drawn i.i.d. from a Gaussian distribution and the labels are generated according to a planted linear model with multiple outputs. This framework serves as a good model for a variety of interesting problems including end-to-end training in inverse problems and various auto-encoder models in machine learning. Despite the seemingly simple formulation, understanding training dynamics is a challenging unresolved problem. This is in part due to the fact that the training landscape contains multiple non-strict saddle points and it is completely unclear why gradient descent from random initialization is able to escape such bad stationary points. In this work, we develop the first comprehensive analysis of the gradient descent dynamics for learning linear target functions with ReLU networks. We show that gradient descent with moderately small random initialization converges to a global minimizer at a linear rate. To rigorously show that GD avoids non-strict saddle points, we develop intricate techniques to decompose the loss and control the GD trajectory, which may have broader implications for the analysis of non-convex optimization problems involving non-strict saddles. We corroborate our theoretical results with extensive experiments with various configurations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper is concerned with studying gradient descent on one-hidden layer ReLU networks, to learn linear functions.  The inputs are normally distributed, and the loss is mean square.  There are two main theoretical results, in both of which: the output dimension is 1; the width of the network is 2; the output layer is fixed, i.e., only the hidden layer is trained; and convergence to a global optimum with a linear rate is shown.  The two results differ in that the first is for population loss, and the second for empirical loss and also shows that a linear number of samples in the input dimension suffices.  The paper aso reports several experiments in similar or slightly extended settings compared to the two theoretical results.

### Strengths
The paper is generally well written.  The statements of the theoretical results and their discussion are clear.  The settings of the experiments are also clearly conveyed, and the plots are not difficult to understand.

### Weaknesses
The theoretical results are very restricted.  Learning a linear function by a width-2 one-hidden layer ReLU network does not seem substantially different from learning a single ReLU neuron by a single two-layer ReLU neuron.  In addition, only the hidden layer is trained, which substantially simplifies the dynamics.  And moreover, the gradient descent considered in the paper is non-standard due to the modification of multiplying by the reciprocal squares of the fixed output weights.

Most of the papers discussed as related work are several years old.  Also many relevant works are omitted, such as:
- Yehudai and Shamir "Learning a Single Neuron with Gradient Methods" COLT 2020;
- Vardi, Yehudai, and Shamir "Learning a Single Neuron with Bias Using Gradient Descent" NeurIPS 2021;
- Chistikov, Englert, and Lazic "Learning a Neuron by a Shallow ReLU Network: Dynamics and Implicit Bias for Correlated Inputs" NeurIPS 2023;
- Zhu, Liu, and Cevher "How Gradient descent balances features: A dynamical analysis for two-layer neural networks" ICLR 2025;
- Boursier and Flammarion "Simplicity bias and optimization threshold in two-layer ReLU networks" ICML 2025.

I also remark some typos etc.
- Grammatically correct is "optimum" for singluar and "optima" for plural.
- In the third last line of Theorem 1, $v_1$ and $v_2$ should not be bold.
- In section 4.1, there is a reference to Figure 1 (right), however Figure 1 has parts (a), (b), and (c).

### Questions
I do now see how the middle sentence in the paragraph after Theorem 1 shows that "two hidden nodes are necessary".

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work studies the learning of a linear teacher by a one hidden layer ReLU network, with exact parametrization (ie 2 neurons in the theoretically studied case of univariate output). For this problem, the authors show that both population GD and GD (with $n\gtrsim d$) with empirical data recovers the teacher function.

### Strengths
The question of whether a one hidden layer ReLU network can learn a linear teacher remains an open question from a general aspect. The authors provide a nice answer to this question in the case of exact parametrization and no label noise, by giving a nice characterization of the training dynamics.

### Weaknesses
My main concern about this work is its relation with the existing literature and how novel it is. In particular, the authors omit to mention the papers [1] and [2] which characterize the training dynamics of a two neurons two layer network when the teacher is itself a two neuron ReLU network. Since a linear function can be seen as the difference of two ReLUs, these two papers clearly fit the scope of learning a linear teacher. In consequence, they deserve to be mentioned, but also to be discussed in detail.

Even if the results of [1] and [2] might not be applied directly to the linear case, I believe that the proof techniques are very similar. In consequence, the authors would also need to discuss to what extent their proof techniques are new and what are the major challenges that were not addressed in previous works. 

My other concern is about the limited aspect of the problem. The authors consider strong assumptions that strongly simplify the analysis (and even denature the final message) : no label noise, exact parametrization, fixed second layer and $r=1$. In particular, the no label noise assumption allows a perfect recovery of the teacher possible even in the presence of a finite number of data points, which would clearly not be possible with label noise. In consequence, I find the abstract/introduction particularly overstating.

-----------------
# Minor remarks
- [3] also studied how a two layer ReLU newtork can learn from a linear teacher. Although their exact setting and analysis is somehow different, I think it deserves to be cited.
- Figures 1. (a) and (b) are not very clear. The axes are not labeled
- the multiplication by $\text{diag}(v)^-2$ in the GD step makes me think of the Hessian preconditioning. Is there any relation to it?
- I disagree that this work can be seen as a generalization of Xu and Du (2023) (line 413). Their paper precisely aims at studying the overparameterized case, not the exactly parameterized one
- line 405: you mention local optima, but in the problem here, it seems there is no suboptimal local optima, only saddle points


----------
# References
[1] Zhong, Kai, et al. "Recovery guarantees for one-hidden-layer neural networks." International conference on machine learning. PMLR, 2017.

[2] Zhang, Xiao, et al. "Learning one-hidden-layer relu networks via gradient descent." The 22nd international conference on artificial intelligence and statistics. PMLR, 2019.

[3] Boursier, Etienne, and Nicolas Flammarion. "Simplicity Bias and Optimization Threshold in Two-Layer ReLU Networks." Forty-second International Conference on Machine Learning, 2025.

### Questions
How novel are your result and analysis wrt [1] and [2] ?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the dynamics of the generalization error resulting from learning a linear function using a one-hidden layer neural network with two neurons and ReLU activation.

### Strengths
The paper addresses an important question by studying how the generalization error evolves with the number of steps for two-layer neural networks.

### Weaknesses
The setting appears too restrictive. First of all, if the target function is linear, does it even make sense to learn it using a neural network. Second, usually the width of the hidden layer is comparable with the input dimension, while in this case the hidden width is 2. Third, the last layer is not trainable and is just frozen instead. I would suggest adding much more motivation to explain why this setting is interesting.

### Questions
1) What is the justification for considering the setting presented in the paper?
2) What are the assumptions regarding c_7 in Theorem 2 and how does it depend on the other parameters from the statement? I find it counterintuitive that the generalization error would converge towards 0 for **any** c_7 without extra quantitative assumptions. 
3)  Can one generalize the results to an arbitrary number of neurons and / or more general activation functions?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper analyzes gradient-descent (GD) training dynamics of one-hidden-layer ReLU networks when the ground-truth mapping is linear. The authors mainly focus on the one dimensional case where the ground truth is a single vector and identify the existence of saddle points in this regime. They prove global convergence of GD for the small initialization and extend the result from population loss to empirical loss. Experiments are conducted to demonstrate the problem’s optimization landscape. They also briefly discuss the multi-dimensional output case in the appendix.

### Strengths
1. This paper proves global convergence of two layer ReLU networks for learning linear mappings. It also discusses how GD avoids saddle points along the optimization trajectory.  
2. Extensive experiments are conducted on the optimization landscape. The observed pairing behaviour in the multi-dimensional setting might be helpful for understanding feature learning of neural networks.

### Weaknesses
The main result considers learning a single linear vector as the teacher, which is overly simple. The single neuron learning problem is extensively studied in previous literature (e.g., see \[1\], \[2\]). Although these results cannot be directly transferred to the linear teacher setting, I believe a widely-used early-alignment technique (e.g., see \[3\], \[4\]) can still be applied. When initialization is small, the gradient (equation (8)) is almost parallel with the ground truth, forcing students to align with the teacher and converge globally. Therefore, the analysis techniques used in the paper seem not very novel to me. That being said, I am happy to change my mind if I missed anything.

References:

1. Yehudai, Gilad and Ohad Shamir. “Learning a Single Neuron with Gradient Methods.” ArXiv abs/2001.05205 (2020): n. Pag.  
2. Wu, Chenwei, Jiajun Luo, and Jason D. Lee. "No spurious local minima in a two hidden unit relu network." (2018).  
3. Soltanolkotabi, Mahdi. "Learning relus via gradient descent." Advances in neural information processing systems 30 (2017).  
4. Brutzkus, Alon, and Amir Globerson. "Globally optimal gradient descent for a convnet with gaussian inputs." International conference on machine learning. PMLR, 2017\.

### Questions
The multi-dimensional case in Appendix B seems interesting, but the assumption on initialization is restricted. What is the initialization scheme used in Theorem 6? Can a convergence proof be constructed for the random initialization setting?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper analyzes the gradient descent dynamics of training a single-hidden-layer ReLU network to fit linear target functions under Gaussian inputs. Despite the simplicity of the setup, the optimization landscape includes numerous non-strict saddle points, making convergence uncertain. Under this setting, the authors prove that gradient descent with small random initialization converges with a linear rate to the global minimum in two cases: population and empirical loss. Under the empirical loss, they recover the optimal sample complexity. The theoretical findings are supported by experiments.

### Strengths
This paper studies an interesting question of learning a simple function by a complex model that is over-expressive. This scenario arises, for example, in training autoencoders, where the end-to-end scheme should implement the identity transform. In such cases, the natural question is whether the learning algorithm can recover the simple function that we are after.

### Weaknesses
Under the setting considered in the paper, there is only one minimum, which corresponds to the underlying (ground truth) linear transform. Many works in the past showed that GD can minimize the loss while training neural networks, e.g., [R1]. So, it is not surprising that it happens here as well.

Moreover, the particular setting studied in the paper does not match the practice. For example:
1. Only the weights of the first layer are optimized.
2. The paper considers a preconditioner that uses the weights of the second layer as normalization.
3. The width of the hidden layer is exactly two

Finally, the two settings considered in the paper, empirical and population loss, don't add much more information and look repetitive to me (even though the proof technique is different, and we also get the sample complexity in the empirical setting).

**References**:
[R1] - Gradient Descent Finds Global Minima of Deep Neural Networks

### Questions
1. Could we derive **similar** results from prior work? For example, using the same architecture but with SiLU activation and applying [R1].
2. Can we extend the analysis to fully connected layers with bias?

### Soundness
3

### Presentation
3

### Contribution
2
