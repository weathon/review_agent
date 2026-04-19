# Criteria and Bias of Parameterized Linear Regression under Edge of Stability Regime

- Decision: Reject
- Scores: 6, 5, 5, 8

## Abstract
Classical optimization theory requires a small step-size for gradient-based methods to converge. Nevertheless, recent findings (Cohen et al., 2021) challenge the traditional idea by empirically demonstrating Gradient Descent (GD) converges even when the step-size $\eta$ exceeds the threshold of $2/L$, where $L$ is the global smooth constant. This is usually known as the \emph{Edge of Stability} (EoS) phenomenon.  A widely held belief suggests that an objective function with subquadratic growth plays an important role in incurring EoS. In this paper, we provide a more comprehensive answer by considering the task of finding linear interpolator $\beta \in \mathbb{R}^{d}$ for regression with loss function $l(\cdot)$, where $\beta$ admits parameterization as $\beta =  w^2_{+} -  w^2_{-}$. Contrary to the previous work that suggests a subquadratic $l$ is necessary for EoS, our novel finding reveals that EoS occurs even when $l$ is quadratic under proper conditions. This argument is made rigorous by both empirical and theoretical evidence, demonstrating the GD trajectory converges to a linear interpolator in a non-asymptotic way. Moreover, the model under quadratic $l$, also known as a depth-$2$ \emph{diagonal linear network}, remains largely unexplored under the EoS regime. Our analysis then sheds some new light on the implicit bias of diagonal linear networks when a larger step-size is employed, enriching the understanding of EoS on more practical models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper explores the Edge of Stability (EoS) phenomenon. They study GD and focus on a constant step-size exceeding the typical threshold of $2/L$, As the contributions, they observe that EoS occurs even when the loss $l$ is quadratic under proper conditions while the existing works require a subquadratic $l $. Due to the close relationship between the quadratic $l$ and the  depth-2 diagonal linear network, the findings provide some explanations for the implicit bias of diagonal linear networks.

### Strengths
The work seems to be solid.

The paper is polished very clearly.

The paper focuses on understanding the convergence of the GD algorithm, breaking through the limitations of traditional smoothness analysis (the step-size exceeds the threshold of $2/L$),  and extending previous conclusions to more complex scenarios (from subquadratic growth objective functions to quadratic growth objective functions).

### Weaknesses
Does the conclusion of the more general $n$-sample case in the paper apply to SGD? Due to the importance of stochastic optimization, I expect to see similar conclusions under stochastic optimization as well.

I am sorry I am not very familiar with this topic.  I will revise my score based on the comments of other more senior reviewers.

### Questions
What is the behind insight in selecting a specific formula of $\beta$ as $w_+^2-w_-^2$?

Does the theoretical analysis hold in the more general $n$-sample case, or what are the difficulties in this analysis?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
In this paper, the authors consider the task of finding interpolators for the linear regression with quadratic parameterization and study the convergence of constant step-size GD under the large step-size regime. They focus on the non-trivial question of whether a quadratic loss can trigger the Edge of Stability phenomena. The authors show through both empirical and theoretical aspects that, when some certain condition is satisfied, EoS indeed occurs given quadratic loss.

### Strengths
In this paper, the authors consider running GD with a large constant step-size to find linear interpolators that admit quadratic parameterization for the one-sample linear regression task. They theoretically proved the on-sample case and extend the one-sample results by empirically finding conditions in the more general n-sample cases. Theses theoretical analysis and empirical results are presented with clear structures.

### Weaknesses
The major weakness of this submission is that the authors only provide the theoretical analysis and mathematical proof for the one sample case. The empirical results from the numerical experiments in section 3 shows the convergence of GD under the EoS regime when the loss function is quadratic. The authors only present the theorems characterizing the convergence of GD when the model considered has dimension $d = 2$. The authors should extend these theoretical analysis to the high dimension cases. The one sample analysis and teo dimensional proof is not enough to explain the empirical results from the numerical experiments.

### Questions
The authors argued that the model considered in this paper is the depth 2 diagonal linear network. Is that possible to extend the theoretical analysis to the model with higher depths?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper studies the implicit bias of GD on quadratic loss and linear models with diagonal linear network parameterization under the EoS regime. The paper first empirically shows that when data is in multi-dimension ($d\geq 2$), EoS can occur for quadratic loss, while prior works on EoS suggest that subquadratic loss is necessary for EoS to happen. The experiments show that different choices of step size lead to different oscillation types, from GF regime to EoS regime to chaos to divergence. The paper then theoretically studies the parameter convergence (directly related to generalization) in a sparse solution 2-D diagonal linear network setting and provides non-asymptotic rates in various settings. The results show that in the EoS regime when the step size is not too large ($\eta\mu<1$), smaller initialization ($\alpha$) yields a better generalization, while when step size is large ($\eta\mu>1$) there will be an error floor that cannot vanish as $\alpha\to 0$.

### Strengths
1. The paper is overall well-written and easy to follow. It introduces the settings clearly and provides sufficient illustrations to help present results in different conditions/regimes. The main findings are organized clearly. The proof overview is intuitive for grasping the essence of the proof technique. 

2. The paper studies implicit bias in the EoS regime for diagonal linear networks in multi-dimensions, which is a significant step forward as most of the prior works on EoS only deal with minimalist examples where the data is assumed to be 1-d, and it is interesting to see the empirical occurrence of EoS depends on the data dimension (non-degeneracy). Moreover, the diagonal linear network setting is closely related to the GD implicit bias literature, so it can potentially connected to wider works.

### Weaknesses
My major concern is regarding the "EoS regime". In the paper, there is no rigorous definition of the EoS regime. Instead, a flip sign condition $r_tr_{t+1}<0$ is used throughout the theoretical statements. However, I doubt whether the sign flips in residual indeed correspond to the commonly referred EoS phenomenon ($\eta L_t>2$ and oscillations in loss happen along the trajectory). For example, if we minimize $f(x)=\frac{1}{2}x^2$ with GD, the sharpness is $1$ and the residual $r_t=x_t$ is flipping its sign if we choose step size $\eta\in(1,2)$, but the loss is actually monotonically decreasing and the sharpness is always below $2/\eta$. 
In the proof overview (Section 5), it is discussed that $|r_{t+2}|\leq (1-\alpha)^2|r_t|$ is possible, but there is no statement comparing $|r_{t+1}|$ and $|r_t|$, and there is no statement on the relationship between step size $\eta$ and sharpness $L$, so we are not sure if EoS is happening or not. In my understanding, it might be the case that only the $\eta\mu>1$ case corresponds to the EoS that people refer to. 

Minor typos (not exactly weaknesses): 

1. In Figure 4 the x-axis has no label, which I guess is $\alpha$.

### Questions
1. For convergence in the regime $\mu\eta<1$, the current analysis assumes period-2 flip sign in residual, and the result essentially states ${\ell(w_{2t})}_{t\geq 0}$ converges. I am wondering if the analysis can be extended to deal with more general oscillations without specific periods, as they are closer to the EoS phenomenon in practice [1]. 

2. The paper assumes a specific scaling initialization (lines 218 and 238) for subsequent analysis and the authors claim this initialization is to align with the literature on diagonal linear networks. Regarding this initialization, I have a few questions: 
(a) Why in the theory $w_+$ and $w_-$ are of the same scale while in experiment $w_+$ has larger scale than $w_-$? (b) How important is this initialization for ensuring convergence and EoS? (c) How easy is the analysis to be extended to more general or random (e.g. Gaussian) initializations? Just for quick comparison, some existing works can handle general initializations as long as the initial weights satisfy certain conditions [2,3] while they mainly focus on the 1-d case. 

3. The result in the $\eta\mu>1$ regime (Figure 4, left) suggests that smaller step size allows smaller generalization errors. This seems to contradict a common belief that a large learning rate is beneficial to generalization as it forces the minimizer to be in a flat region. Could authors provide some insights about this difference?

[1] Cohen, Jeremy M., Simran Kaur, Yuanzhi Li, J. Zico Kolter, and Ameet Talwalkar. "Gradient descent on neural networks typically occurs at the edge of stability." arXiv preprint arXiv:2103.00065 (2021).

[2] Ahn, Kwangjun, Sébastien Bubeck, Sinho Chewi, Yin Tat Lee, Felipe Suarez, and Yi Zhang. "Learning threshold neurons via edge of stability." Advances in Neural Information Processing Systems (2023).

[3] Wang, Yuqing, Zhenghao Xu, Tuo Zhao, and Molei Tao. "Good regularity creates large learning rate implicit biases: edge of stability, balancing, and catapult." arXiv preprint arXiv:2310.17087 (2023).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors analyze the Edge-of-Stability (EoS) phenomenon for gradient descent on linear regression with the loss function $\ell(\langle x, \beta\rangle - y)$, where $(x, y)$ is the datapoint with $x\in \mathbb{R}^d$ and $y\in \mathbb{R}$. EoS phenomenon is observed when GD is run with a stepsize $\eta > \frac{2}{L}$ where $L$ is the smoothness constant of the loss. In the EoS regime, GD oscillates rapidly but still converges to the minima, under certain conditions.

Existing works (Ma et al 2022, Ahn et al 2022, Song & Yun 2023) show that for sub-quadratic $\ell$, GD can enter the EoS regime. This paper shows that for a particular quadratic parameterization, namely diagonal neural networks, GD can enter the EoS regime even for quadratic $\ell$. The parameterization is $\beta = \beta_{w} = w_{+}^2 - w_{-}^2$ and $w= [w_{+}^\top, w_{-}^\top]^\top$, with gradient updates on $w$.


From Claim 1, for quadratic $\ell(s)  = s^2/4$, they obtain EoS for GD under a single-sample regime for $d\geq 2, y\neq 0$ and $x$ non-degenerate. Further, for $d=2, x= (1,x')$ and sparse realizable model $\beta^\star = (\mu, 0), y= \mu$, with initialization scale $\alpha$, they obtain two separate regimes even for EoS. 

In the first regime, from Theorem 1, for $\mu \eta <1$ and constant $\alpha$, GD results in both the traditional Gradient Flow regime (GF), without oscillations, and the EoS regime. Here, EoS occurs with damped oscillations. Further, the final solution of GD has generalization error dependent on initialization $\alpha$.

In the second regime, from Theorem 2, $\eta \mu \in (1,2)$, with initialization, $x'$, and generalization error dependent on $\eta\mu$, GD in EoS might initially have diverging amplitude of oscillations, however, it eventually dies down and after a point converges at a linear rate.


Empirically, the authors show that their model requires overparameterization, as for $d=n$, EoS doesn't occur but for $d>n$, it does. For their case of single sample $n=1$, justifying their choice of $d=2$. 


**References**--
- (Ma et al 2022) Beyond the Quadratic Approximation: the Multiscale Structure of Neural Network Loss Landscapes. Arxiv.
- (Ahn et al 2023) Learning threshold neurons via the “edge of stability”. NeurIPS.
- (Song & Yun 2023) Trajectory Alignment: Understanding the Edge of
Stability Phenomenon via Bifurcation Theory. NeurIPS.

### Strengths
- **Novel insights**: There are several novel and non-trivial insights  -- i) quadratic losses can lead to EoS, ii) the diagonal NN model fulfils this requirement, iii) EoS can have regimes with oscillations with increasing magnitude but still converge later, iv) overparameterization might be necessary for EoS on quadratic losses. 

- **EoS on diagonal neural networks**: Diagonal neural networks serve as a simple to analyze but still expressive model. Using these models has 3 important advantages -- i) as these are real networks, their claimed phenomenon occurs not just on some theoretically well-crafted model, ii) diagonal NNs remain a good testbed for theoretical analysis of complicated deep learning phenomena, iii) they have provided a proof for EoS on diagonal neural networks, which was missing from existing works (Even et al 2023), and can now be used to verify claimed empirical phenomenon (Even et al 2023). Note that the proof of EoS on diagonal neural networks is highly non-trivial especially for the $\mu\eta > 1$ case.


- **Presentation**: The paper is easy to read inspite of the heavy notation. The key insights are clearly explained and the figures are very helpful in understanding them. Figure 6, in particular, is a good example to intuitively explain the proof sketch.




**References**--
- (Even et al 2023) (S)GD over Diagonal Linear Networks:
Implicit Bias, Large Stepsizes and Edge of Stability. NeurIPS.

### Weaknesses
- **Is subquadratic growth "necessary" or "sufficient" for observing EoS**? It might be beneficial to state the exact reference for the subquadratic condition. Note that the subquadratic condition was introduced in (Ma et al 2022), which the authors have not mentioned. Further, I'm not sure if (Ahn et al 2023) actually state that subquadratic growth is "necessary" for EoS. Assumption A2 and A3 in (Ahn et al 2023), show that subquadratic growth is sufficient for EoS. Similarly, the results in Section 4 in (Ma et al 2022), and assumptions 2.4 and 4.2 in (Song & Yun, 2023) are sufficient for EoS. If I'm missing some details and subquadratic condition is indeed necessary for EoS, can the authors should probably specify the exact theorem, assumption or an argument for this?

- **How large is $\mathfrak{t} - t_0$** ? In Theorem 2, there are $3$ phases for GD. In the first phase, it has not started oscillations, which lasts until $t_0$. From Lemma 7, $t_0 \geq \Omega_{\mu, \eta}(\log(1/\alpha^2))$, but only for $\mu\eta \in (0,2)$, which includes $\mu \in (1, \frac{3\sqrt{2} - 2}{2})$. In the second phase, which lasts from $t_0$ to $\mathfrak{t}$, the oscillations finally start decreasing in magnitude. Lemmas 14 and 15 establish that there exist such a $\mathfrak{t}$, but not how large it is. As after $\mathfrak{t}$, we see linear convergence, how long do we need to wait for it becomes an important question. If the authors cannot establish it theoretically, they might argue empirically that $\mathfrak{t}$ not very large.

- Typo in Line 1537: $|\lambda_{1,2}| < 1$.

**References** --
- (Ma et al 2022) Beyond the Quadratic Approximation: the Multiscale Structure of Neural Network Loss Landscapes. Arxiv.
- (Ahn et al 2023) Learning threshold neurons via the “edge of stability”. NeurIPS.
- (Song & Yun 2023) Trajectory Alignment: Understanding the Edge of
Stability Phenomenon via Bifurcation Theory. NeurIPS.

### Questions
- How does the level of overparameterization affect the EoS phenomenon? This might be an easy extension of Figure 5, by checking if increasing the level of overparameterization, i.e., the ratio $\frac{d}{n}$, changes the EoS phenomenon. Figure 5 contains $\frac{d}{n} = 2, 1$, but another experiment on a larger value of $\frac{d}{n}$ might show if large overparameterization helps in EoS. Note that this is not strictly required but might be interesting.

### Soundness
4

### Presentation
4

### Contribution
3
