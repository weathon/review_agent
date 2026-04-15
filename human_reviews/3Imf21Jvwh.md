# Hard-Constrained Neural Networks with Universal Approximation Theorem

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 6, 6

## Abstract
Incorporating prior knowledge or specifications of input-output relationships into machine learning models has gained significant attention, as it enhances generalization from limited data and leads to conforming outputs. However, most existing approaches use soft constraints by penalizing violations through regularization, which offers no guarantee of constraint satisfaction---an essential requirement in safety-critical applications. On the other hand, imposing hard constraints on neural networks may hinder their representational power, adversely affecting performance. To address this, we propose HardNet, a practical framework for constructing neural networks that inherently satisfy hard constraints without sacrificing model capacity. Specifically, we encode affine and convex hard constraints, dependent on both inputs and outputs, by appending a differentiable projection layer to the network’s output. This architecture allows unconstrained optimization of the network parameters using standard algorithms while ensuring constraint satisfaction by construction. Furthermore, we show that HardNet retains the universal approximation capabilities of neural networks. We demonstrate the versatility and effectiveness of HardNet across various applications: fitting functions under constraints, learning optimization solvers, optimizing control policies in safety-critical systems, and learning safe decision logic for aircraft systems.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents a simple framework for imposing constraints on input-output relations in neural networks.
The approach consists in appending a final projection layer to the network, ensuring that the constraints are satisfied by construction. Moreover, the authors show (formally) that this projection operation does not hinder the expressivity of the network, and empirically evaluate the approach on various scenarios.

### Strengths
The paper is very clear in its presentation. It is well structured and reads well. The contributions of the paper are clearly presented and well summarized, both in text and by images and tables. The empirical evaluation include plenty of relevant and diverse scenarios.

### Weaknesses
I have some doubts regarding the novelty of the paper and the technical discussion. 

The idea of satisfying hard constraints using a final differentiable projection layer is mentioned in the works cited as reference, and many other works referencing the idea exist (i.e. https://arxiv.org/abs/2111.10785 and https://arxiv.org/abs/2307.10459). If the contribution is simply the extension of the idea to input dependent constraints, it should be more clearly stated.

As for the soundness of the approach, while the additional projection layer is differentiable, its derivative is not well behaved.
Claims such as "meeting the required constraints while allowing its output to be backpropagated through to train the model via gradient-based algorithms" and "This allows us to project the output $f_θ(x)$ onto the feasible set $C(x)$ and train the projected function via conventional gradient-based algorithms" are not substantiated by a proper discussion on the gradient properties of the resulting network.
In fact, as presented, the gradient is always orthogonal to the constraint. 
This observation is not novel, and from my understanding is the main motivation driving the development of alternatives to projection methods.

### Questions
I'd like to explain more in detail my doubts.
Consider the simple case of $1-d$ output and a single affine constraint. The projection layer reduces a simple rescaled ReLU, and the whole network has zero gradient where the constraint is not satisfied. 

This effect is true in general. In fact, if we evaluate the Jacobian of the projection layer when the constraint is not satisfied ($J_{\mathcal{P}} = I-a(x)a(x)^T$), we can see that the gradient of the network will be always orthogonal to the constraint vector $a(x)$. 

For this reason, if $f_\theta$ is initialized outside the feasible region, it should be impossible to "re enter" it by simply following the gradient. This means that the whole optimization would get "stuck" on the boundary of the feasible set, which might not be ideal. In stochastic gradient descent, this issue might be mitigated, however, i believe this is an important discussion to have in the paper.

The proposed projection (for the affine variant) works in two steps, reducing the dimensionality of the output space using the equality constraints, and performing a projection in the reduced space. Is this better than simply treating equalities as a pair of inequalities? This aspect should be investigated to justify the additional complexity of the method.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents HardNet, an approach to train neural networks that satisfy hard constraints by construction.
The core idea of the paper is to append a projection layer at the end of the network in order to bring the network output onto the feasible set.
Two different schemes are presented: one using a closed-form (non-orthogonal) projection for affine constraints, and one resorting to previous work presenting differentiable convex optimization solvers, in case of more general convex constraints.
Universal approximation theorems for the architectures are presented.
Experimental results on a variety of benchmarks are presented, demonstrating that HardNet attains good performance while satisfying the constraints.

### Strengths
The idea to enforce hard constraints by construction through a projection layer is simple and neat. 
Differently from previous work in the area, universal approximation theorems are provided.
The experiments show that, at least for affine constraints supported by HardNet-Aff, HardNet works quite well in practice (albeit at a small scale).
Finally, I found the related work section to be well-written and fairly comprehensive.

### Weaknesses
The main weaknesses of the paper are threefold: HardNet-Cvx, the assumptions behind HardNet-Aff, and the experimental section.

*HardNet-Cvx*: the idea to use differentiable optimizers to perform the projection does not appear to be completely novel. DC3 discusses it in related work, excluding it because of large computational cost (this large cost is definitely confirmed at inference in the experiments in Table 5). The rayen paper uses it as a baseline (named PP in their paper). I do not know if the authors are aware of this, but these points absolutely need to be acknowledged throughout the paper. Furthermore, the only example over HardNet-Cvx is used (Table 5) appears to nevertheless use affine constraints (albeit, as far as I understand, too many to be supported by HardNet-Aff). In this instance, its runtime is extremely large, questioning its practical applicability.

*HardNet-Aff assumptions*: the assumptions required for HardNet-Aff seem very strong to me. It seems to be that a simple interval constraint per network output coordinate would already be unsupported, hence incurring the large cost associated to HardNet-Cvx. Could the authors comment on this?

*Experiments*: my main concern over the experimental section is the surprisingly bad performance of DC3. In the original paper, all constraints appear to be satisfied in practice. Is there anything I am missing here? Was DC3 run for an insufficient number of iterations? I understand that for HardNet the constraints hold by construction, but DC3 appears to be fairly strong empirically, in the original paper. Important details such as training times for each scheme appear to be omitted (or at least, do not feature prominently). "DC3 + Proj" would also appear to be a missing, yet very interesting baseline. Further details are provided as questions.

------------------
Edit: I am decreasing my score to a 5 as I believe my concerns were not adequately addressed. For instance, I see the authors have acknowledged the existence of HardNet-Cvx in previous work within their updated section 4.2. This should clearly have been done from the original submission. The relative contribution is then only the proof, which is interesting yet quite overstated, as pointed out by reviewer ASwu. I also still find the performance of DC3 to be surprisingly bad with respect to the original papers, requiring clarifications.

### Questions
- Could the authors train DC3 for longer, or with more inner iterations to satisfy the inequality constraints? If this is deemed infeasible, can the authors provide an explanation on the discrepancy with the results in the original paper?
- Would it be possible to provide "DC3 + Proj" results?
- Why is DC3 absent from Table 5?
- In the toy example, training points are sampled from [-1.2, 1.2], but then the networks are evaluated on [-2, 2]. Aren't samples in that area OOD, in a sense? Couldn't that explain the performance of the baselines? I understand that guaranteed constraint satisfaction is an advantage of the proposed approach, but these points should be discussed. (e.g., by providing results on [-2, 2] training)
- What is lost by the fact that HardNet-Aff does not rely on an orthogonal projection? Does this imply anything concerning the hardness of learning the function through gradient-based method? An interesting ablation would be to compare HardNet-Aff with HardNet-Cvx on a setup where both are supported.
- It would be interesting to see some experiments on (even slightly) larger networks. Would some methods benefit more from the additional capacity than the others?

In general, I think the quality of the work would clearly increase if the authors were more honest on the limitations of the proposed approach (see weaknesses above).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a type of hard-constrained neural network by introducing differentiable projection layers. Specifically, if the constraints are affine and the number of constraints are no greater than the output dimension, the projection can be found in closed form. For other convex constraints, the authors propose to apply the differentiable optimization framework to compute the projection iteratively. The authors use experiments including learning an optimization solver and controlling a safety-critical dynamical system to demonstrate the effectiveness of the proposed work.

### Strengths
I have not worked on constrained neural networks, and hence I am unfamiliar with a lot of the cited literature. That being said, judged based on the content of this submission, the results are promising and meaningful, and the presentation is mostly clear.

### Weaknesses
- To use the closed-form projection algorithm Eq.(7), we need $n_{ineq} + n_{eq}$ to be no greater than the output dimension. Is this restrictive in practice? In the included experiments, which one of them uses closed-form projection?
- For Eq.(11), should $u$ also be a function of $t$, i.e., $u (t)$?
- In Table 4, why are rows 2 and 4 marked as red even though they are feasible?

### Questions
- I am a little confused about part iii) of Proposition 4.2. Namely, the projection preserves the distance from the boundary of the feasible set when $\bar{f}_\theta (x)$ satisfies the constraint. Would you mind sharing a geometric intuition?
- I am also confused about the $C_\leq (f (x))$ notation in line 359. What does $C$ denote? This is different from the $C (x)$ in Eq.(4), right?
- Regarding Figure 2, it looks like all models perform reasonably good in the region which the training data lie in, and the difference occurs outside of data coverage. I am confused why "Soft" seems much worse than others. If I understood it correctly, "Soft" penalizes when the model output violates the constraints. Since all training points are feasible, I intuitively expect "Soft" to behave similarly to "NN", but this is not the case. Could you please explain why such difference? Also, how would "Soft + proj" look like?
- For the "safe control policy" experiment in Section 5.3, what do you think is the biggest advantage of the proposed method compared with non-learning methods such as model predictive control?
- Line 500 mentions that the constraint in Eq.(12) can be conservative, leading to worse performance compared to "Soft" and "DC3". Is it possible to adjust the level of conservativeness by changing $\alpha$?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The proposes a new layer able to project the output of the neural network (which might be non-compliant with a set of hard constraints) back into a "safe space" where the constraints are guaranteed to be satisfied.

### Strengths
In general, I like this paper quite a lot and I think it has different strengths (listed below):

- The paper handles a very important problem. 
- The paper is technically sound 
- The paper is written very well

### Weaknesses
The paper has only one major flow: it is not well placed in the literature. 

Indeed, I do not think the authors are familiar with the Neuro-symbolic AI literature, where the problem of learning with hard constraints has already been studied. In particular, there is a research group that has worked a lot on creating layers that are make neural networks compliant by construction with a set of hard constraints [1,2,3]. [1] is the first work that proposed this kind of approach with constraints expressing hierarchies over the outputs. In the latest works they worked with hard constraints expressed in propositional logic [2] and as linear inequalities [3]. Obviously I believe [3] is particularly relevant to your paper and it would be nice to have a comparison between the two methods (at least in terms of discussion for the rebuttal phase and experimental only for the camera ready). Delving more on the logical side you have works like Semantic Probabilistic Layer that gives a probabilistic perspective to hard constraints expressed in propositional logic and can guarantee their satisfaction by construction [4].  Finally, you can find an entire line of work which maps the outputs of the neural network into logical predicates and allows reasoning on top of these predicates (see e.g., [5,6,7]) which then also guarantees the satisfaction of the constraint. 

The final rate is below the acceptance threshold because of this. However, I am fully aware that it is often very hard to keep up with the extensive literature available in ML, so I will be very open to increasing my score. 

References:

[1] Eleonora Giunchiglia and Thomas Lukasiewicz. Coherent hierarchical multi- label classification networks. In Proc. of NeurIPS, 2020.

[2] Eleonora Giunchiglia, Alex Tatomir, Mihaela Catalina Stoian, and Thomas Lukasiewicz. CCN+: A neuro-symbolic framework for deep learning with requirements. International Journal of Approximate Reasoning, 171, 2024.

[3] Mihaela C. Stoian, Salijona Dyrmishi, Maxime Cordy, Thomas Lukasiewicz, and Eleonora Giunchiglia. How Realistic Is Your Synthetic Data? Constraining Deep Generative Models for Tabular Data. In Proceedings of International Conference on Learning Representations, 2024.

[4] Kareem Ahmed, Stefano Teso, Kai-Wei Chang, Guy Van den Broeck, and Antonio Vergari. Semantic probabilistic layers for neuro-symbolic learning. In Proceedings of Neural Information Processing Systems, 2022.

[5] Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. DeepProbLog: Neural probabilistic logic programming. In Proceedings of Neural Information Processing Systems, 2018.

[6] Connor Pryor, Charles Dickens, Eriq Augustine, Alon Albalak, William Yang Wang, and Lise Getoor. Neupsl: Neural probabilistic soft logic. In Proceedings of International Joint Conference on Artificial Intelligence, 2023.

[7] Emile van Krieken, Thiviyan Thanapalasingam, Jakub M. Tomczak, Frank Van Harmelen, and Annette Ten Teije. A-neSI: A scalable approximate method for probabilistic neurosymbolic inference. In Proceedings of Neural Information Processing Systems, 2023.

### Questions
As I liked a lot the paper, here I also include a series of suggestions to further improve the paper:

- At page 4, shouldn't the sup-norm be defined as $||f||_\inf = sup_{x \in \mathcal{X}} |f(x)|$?
- At page 5, I think it would have been great to have a small example with just a neural network with two outputs $y_0$ and $y_1$ and the constraint $y_0 \ge y_1$. Then you could for example show that if $y_0 = 3$ and $y_1 = 4$ then $a(x)=[−1,1]$, $b(x) =0$ and 
$$
\mathcal{P}(f_\theta)(x) = f_\theta(x) - \frac{a(x)}{\||a(x)\||^2} \text{ReLU}(a(x)^\top f_\theta(x) - b(x)) = [3.5, 3.5].
$$
- At page 5, among the assumptions there is written that the constraints need to be feasible. Just to improve the readability of the paper and also make sure everything is well defined, it would help to add the meaning of the word feasible (i.e., "that there exists at least one solution or point within the domain of interest that satisfies all the constraints simultaneously")
- At page 5 the authors give the assumptions for which the number of constraints needs to be lower or equal than $n_{out}$. I think it would be really helpful to add a simple example with a set of constraints that cannot be captured (e.g., $x \ge 0, y \ge 0, x+y \ge 0$)
- At page 8, in the experimental analysis, the constraints you show clearly define a non-convex space. However, for your layer to work you need to have a set of constraints that defines a convex space. Are you simply applying different projections on the ground of the value of $x$? If that is the case, I personally find this experiment a bit misleading as this only works because $x$ is a known input. I do not think your layer would work in a setting where you have a constraint of the type if $y_1 > y_2$ then $y_2+ y_3 < 1$, tight?
- Finally, I think it would also be nice if you could extend on how this type of work is relevant for the SafeAI literature, as creating models that are complaint by design with a set of constraints obviously increases their safety.

### Soundness
3

### Presentation
3

### Contribution
3
