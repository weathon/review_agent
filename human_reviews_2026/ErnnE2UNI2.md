# Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6, 2

## Abstract
We study the implicit bias of sharpness-aware minimization (SAM) when training $L$-layer linear diagonal networks on linearly separable binary classification. For linear models ($L=1$), both $\ell_\infty$- and $\ell_2$-SAM recover the $\ell_2$ max-margin classifier, matching gradient descent (GD). However, for depth $L = 2$, the behavior changes drastically—even on a single-example dataset where we can analyze the dynamics. For $\ell_\infty$-SAM, the limit direction depends critically on initialization and can converge to $0$ or to any standard basis vector; this is in stark contrast to GD, whose limit aligns with the basis vector of the dominant coordinate in the data. For $\ell_2$-SAM, we uncover a phenomenon we call *sequential feature amplification*, in which the predictor initially relies on minor coordinates and gradually shifts to larger ones as training proceeds or initialization increases. Our theoretical analysis attributes this phenomenon to $\ell_2$-SAM’s gradient normalization factor applied in its perturbation, which amplifies minor coordinates early and allows major ones to dominate later, giving a concrete example where infinite-time implicit-bias analyses are insufficient. Synthetic and real-data experiments corroborate our findings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the implicit bias of SAM for linear diagonal networks for a single data point example. While previous works have studied the regression problem, this work focuses on classification and identifies some new phenomena:
- for $\ell_\infty$ SAM with depth at least $2$, the model can converge towards directions that do not maximize the margin
- for $\ell_2$ SAM with depth $=2$, the authors identify a novel "sequential feature discovery" phenomenon where the network most significant coordinate can switch during training

### Strengths
The paper is well written and highlights some new unknown phenomena for learning with SAM that are quite interesting. In particular, I found the sequential feature discovery quite cool

### Weaknesses
The main limitation of this work is the single data point assumption. This work indeed solely focuses on a single point dataset example, which is obviously very toyish. I presume extending this work to multiple points is challenging, but I would like to have at least discussions on the challenges when considering multiple points and also simulations illustrating similar phenomena on multiple training examples would be cool.

I also have a question of how surprising/interesting is this sequential feature discovery phenomenon. First, in practice, SAM is often used with very small $\rho$, so that we can expect the relevant initialisation regime is $\alpha \gtrsim \rho$, so that we would be in the Regime 3. Actually, this sequential feature discovery illustrates how complex and non-monotonic the dynamics can be as soon as we leave the linear parametrization, but does it really say more than that? And in particular, is this property really specific to SAM (and not happening for GD)?

It is not so clear what Section 4.2.4 (and Proposition 4.7) in particular really says about this sequential discovery feature. In particular, it focuses on **lower bounds of the rates** but it seems we cannot say anything about what coordinate is really the dominant one at some fixed time. In particular, I do not see how these results explain what we see in Figure 3.

I also have a concern/remark about the time reparametrization done in Equation (2). At first sight, I would say the trajectories would only match if $$\int_{0}^\infty \ell'(\langle \beta(\hat{\theta}(t)),\mu\rangle)\mathrm{d} t = -\infty.$$
So I guess a justification of that would be needed. Also, this time reparametrization leads to finite time blow ups as discussed line 240. However I am not sure how to interpret the result in case of blow-up: the trajectory can be considered until the first (smallest) blow up time and nothing happens afterwards? If that would be the case, this means that only one coordinate converges to $\infty$ in case of blow up, while all the other ones remain bounded as $t\to\infty$. This seems very weird to me, as in general when overfitting in classification, the coordinates would all converge to infinity (although potentially at different rates). Maybe this is made possible by the $\ell_\infty$-SAM regularization, but I guess it should at least be discussed as it seems surprising to me.

Another hidden assumption is that $L(\theta(0))< L(\mathbf{0})$ (eg assuming $\alpha$ has only positive coordinates) which actually seems to play a critical role in the analysis. I think it would be needed to discuss how necessary this assumption is and what would happen otherwise.

----------------------
# Other remarks:
- Theorem 4.2 assumes three points (a,b,c), but are these needed? Moreover, the final Proposition 4.7 does not totally explain it: it says that any $j$
- sequential feature discovery made me think about *incremental learning*. Although this is different, it might be worth to cite the works [1] and [2] that characterize it for diagonal linear networks
- Figure 3 is very nice. I would have liked a comparison with GD somewhere here.
- Maybe recall before Theorem 4.5 that $0<\mu_1<\ldots<\mu_d$, as I believe this is where it is really needed (and not before) in the paper


----------------
# References

[1] Berthier, Raphaël. "Incremental learning in diagonal linear networks." Journal of Machine Learning Research 24.171 (2023): 1-26.

[2] Pesme, Scott, and Nicolas Flammarion. "Saddle-to-saddle dynamics in diagonal linear networks." Advances in Neural Information Processing Systems 36 (2023): 7475-7505.

### Questions
see above

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work investigates the implicit bias of two versions of SAM related to the $L_2$ and $L_{\infty}$ norm in a classification setting whereas most previous work focusses on the finite mean squared loss regression setting. The analysis is based flow arguments for a data set of size one. They find that the SAM can learn features in different sequences whereas gradient descent would always prefer the major feature first. This provides a potential mechanism to explain the success of SAM. In the appendix their finding is substantiated by an experiment with Gradcam on a vision task.

### Strengths
Fully distinguishing the implicit bias induced by SAM and GF is definitely an important open problem. This work reveals a new phenomenon for the SAM optimizer: learning minor features first then major features first. The use of a single sample and the classification setting is original compared to previous works. The work gives good theoretical characterization what happens before reaching the margin. The theory is illustrated for small example in the appendix, indicating that the result might transfer to more complex settings.

### Weaknesses
1) Sequential feature discovery is a known phenomenon for gradient flow at least also known as saddle to saddle dynamics [1]. There are more works showing this for the gradient flow setting. It would be good to reposition the claim in this light as sequential feature discovery for SAM.

2) There is recent work on margin classification for mirror flow and steepest descent flow which cover more ground, how is the analysis for SAM more difficult? Can the authors contrast their work with [2,3]? What would we need to do for an extension or what prevents it?

3) The work is dense making it hard to value each part of the analysis. I would like to propose to put some parts in the appendix completely and provide more details of the current proofs in the main text and make clear where the text starts again and details of the proof are given. For instance highlight how Theorem 4.5 is proven and put the helper lemmas and propositions to the appendix while still highlighting the dynamical description as this important for intuition as well.

4) The experimental validation should be put in the main text as it helps substantiate the claims made. In addition the experimental validation is limited and could be improved to larger scale settings. Additional data analysis of the experiments is also needed. For example rescaling will most likely have changed the final performance of the model. Moreover, hyperparameters play an important role in these type of experiments and the architecture used these would need be mentioned in a table.

6) If SAM learns minor features first can we not see this in the loss curve trajectories in training i.e. GF should learn faster than SAM? Please show a loss plot illustrating this.

Minor points:
In appendix D the titles could be improved by using $L_{\infty}$, also in the text line 2194 it is said L2 sam is used but in the figures it is Linfty, which is it?

Typo line 364 coordinate

Formulation: line 451 "this is consistent with figure" should it not be an illustration of or this is substantiated by

[1] Pesme, Scott and Nicolas Flammarion. “Saddle-to-saddle dynamics in diagonal linear networks.” Journal of Statistical Mechanics: Theory and Experiment 2024 (2023): n. pag.

[2] Pesme, Scott et al. “Implicit Bias of Mirror Flow on Separable Data.” ArXiv abs/2406.12763 (2024): n. pag.

[3] Tsilivis, Nikolaos et al. “Flavors of Margin: Implicit Bias of Steepest Descent in Homogeneous Neural Networks.” ArXiv abs/2410.22069 (2024): n. pag.

While the work makes an interesting finding the current manuscript may need a substantial rewrite and more experimental illustrations given the limited theory setup.

### Questions
See weaknesses.

### Soundness
2

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
The paper analyzes the implicit bias of sharpness-aware minimization (SAM) in linearly separable binary classification using (L)-layer linear diagonal networks. For (L=1), both $\ell_\infty$-SAM and $\ell_2$-SAM recover the $\ell_2$ max-margin solution, matching gradient descent (GD). For depth (L=2), the behavior diverges sharply: (\ell_\infty)-SAM’s limit direction is highly initialization-dependent—converging to zero or to any standard basis vector—whereas GD aligns with the data’s dominant coordinate. In contrast, $\ell_2$-SAM exhibits **sequential feature discovery**, where the predictor first leans on minor coordinates and gradually shifts to larger ones as training progresses or initialization increases. The authors trace this to $\ell_2$-SAM’s perturbation normalization, which initially amplifies small coordinates before allowing major ones to dominate. Synthetic and real-data experiments corroborate these theoretical findings, revealing depth-sensitive and optimizer-specific implicit biases distinct from GD.

### Strengths
1. Exceptionally clear and easy to follow.
2. The theoretical analysis is solid and convincingly demonstrates how network depth affects the implicit bias of SAM.

### Weaknesses
It is unclear whether these theoretical findings can inform practical algorithmic improvements—for example, proposing a better SAM-style method.

### Questions
See weakness.

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
3

### Summary
The paper studies the bias (and training trajectory) sharpness-aware minimization in diagonal linear networks for a dataset consisting of one data point and logistic loss. In linear models, no difference between GD and SAM is found. For a larger number of layers, infinity-norm based SAM can exhibit different behavior depending on initialization and the perturbation radius of SAM. $\ell_2$ based SAM converges to the direction of the $\ell_1$ max-margin solution, but does so by first having the minor coordinates as the dominant direction and then sequentially moving, from coordinate to coordinate, to the major one.

### Strengths
The highlighted rich behaviors that SAM can exhibit is interesting and it is good to have more work studying logistic loss in this setting.  The paper is well written and the experiments on MNIST to some extend backup the theoretical insights.

### Weaknesses
Clearly, the model and data are very specialized. Maybe my main question would the impact of a non-linearity like ReLU would be. This is not covered by the experiments.

No code was provided, making it more difficult to reproduce results or checking up implementational details of the experiments. For example, I do not think the paper details how exactly the data for Figure 7 is generated (although from the picture one might guess that it could be points drawn from a Gaussian distribution with means (1,2) and (-1,-2)).

### Questions
What about more practical experiments for $\ell_2$ SAM and $L>2$? Do you still observe the sequential feature discovery? Or is this something that only happens in a very narrow regime?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper analyzes the implicit bias of sharpness aware minimization (SAM) in the setting of binary classification, linear separable data, and $L$-layer linear diagonal networks. The work shows the implicit bias changes with depth. In addition, unlike GD / GF, the limiting direction of $\ell_\infty$ SAM is dependent on initialization.

### Strengths
The paper is overall easy to read. In addition, the fact that the implicit bias of SAM changes with depth is interesting.
A nice result is that $\ell_\infty$-SAM's limiting direction is dependent on initialization unlike GD / GF. (That said, for non linear architectures, GD / GF is also dependent on initialization).

### Weaknesses
One weakness is the restrictive setting as the paper deals with L-layer diagonal linear networks and assumes the data is linear separable.

Theorem 4.2 assumes directional convergence of the inner and outer layer as well as their flows which is an extremely strong assumption and something that is highly nontrivial to prove in general.

In addition, the analysis is restricted to a dataset consisting of one point (that has a monotonic structure with respect to its entries). 

Finally, the analysis heavily uses the fact that the depth is 1 or 2 and it seems hard to generalize the analysis to the multi-layer case (L > 2).

### Questions
1. For Theorem 4.1 and Theorem 4.2, what are the conditions on $\rho$ (the radius of SAM)?

2. In Theorem 4.2, could the authors explain what the main obstructions are for proving directional convergence in the special case of one data point?

3. In theorem 3.2, what how do the coordinates of the coefficient $\beta(t)$ grow in the original flow? In a related vein, what is the rate of convergence of the loss using SAM in the original flow?

4. How would the theorems in section 4 change for $L > 2$?

5. Are the figures generated using rescaled flow or the original flow?

### Soundness
3

### Presentation
3

### Contribution
1
