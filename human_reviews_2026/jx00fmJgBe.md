# SPRINT: Stochastic Performative Prediction With Variance Reduction

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Performative prediction (PP) is an algorithmic framework for optimizing machine learning (ML) models where the model's deployment affects the distribution of the data it is trained on. Compared to traditional ML with fixed data, designing algorithms in PP converging to a stable point -- known as a stationary performative stable (SPS) solution -- is more challenging than the counterpart in conventional ML tasks due to the model-induced distribution shifts. While considerable efforts have been made to find SPS solutions using methods such as repeated gradient descent (RGD) and greedy stochastic gradient descent (SGD-GD), most prior studies assumed a strongly convex loss until a recent work established $\mathcal{O}(1/\sqrt{T})$ convergence of SGD-GD to SPS solutions under smooth, non-convex losses. However, this latest progress is still based on the restricted bounded variance assumption in stochastic gradient estimates and yields convergence bounds with a non-vanishing error neighborhood that scales with the variance. This limitation motivates us to improve convergence rates and reduce error in stochastic optimization for PP, particularly in non-convex settings. Thus, we propose a new algorithm called stochastic performative prediction with variance reduction (SPRINT) and establish its convergence to an SPS solution at a rate of $\mathcal{O}(1/T)$. Notably, the resulting error neighborhood is **independent** of the variance of the stochastic gradients. Experiments on multiple real datasets with non-convex models demonstrate that SPRINT outperforms SGD-GD in both convergence rate and stability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes SPRINT for performative prediction: an SVRG-style, snapshot-based variance-reduction method that tames noise from deployment-induced distribution shifts, proves fast and stable convergence without bounded-variance assumptions, and empirically outperforms SGD-GD on Credit, MNIST, and CIFAR-10—especially when feedback effects are strong.

### Strengths
1. The method converges faster and more stably than SGD-GD, reaching a final neighborhood that does not depend on gradient variance.

2. It operates under weaker assumptions by avoiding the bounded-variance requirement and explicitly addressing deployment-induced distribution shift.

3. It delivers stronger empirical performance, achieving lower loss and steadier training on Credit, MNIST, and CIFAR-10, with especially large gains when performative effects are strong.

### Weaknesses
Equation numbering is inconsistent (two equations labeled as Eq. 14).

The roles of $c_k$, $\beta_k$, and $\gamma_k$ are not clearly explained. Their definitions and interactions in the update rule should be clarified.

It is unclear whether Assumption 3.3 could be relaxed; this assumption seems restrictive for broader stochastic settings.

The paper should also discuss scenarios where SPRINT might underperform relative to SGD-GD (e.g., small population, low-variance cases) and analyze why.

Although hyperparameters are reported, the lack of details on the hyperparameter search process raises fairness concerns for baseline comparison.

The claimed “variance reduction” effect is not empirically validated—there is no report of gradient variance, convergence variance, or ablation removing the VR component.

 The experiments only report training loss; test accuracy should also be presented, especially given SPRINT’s claimed robustness under data distribution shifts.

Quantitative data on computational and memory costs are missing.

Experiments rely on small MLP and CNN architectures; results on more realistic models such as ResNet or VGG would strengthen the paper.

 The sensitivity of SPRINT to the learning rate remains unclear—this should be evaluated experimentally.

 It is also important to clarify whether SPRINT introduces additional computational or memory overhead compared to SGD-GC.

### Questions
see  Weaknesses

### Soundness
2

### Presentation
2

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
The paper proposes using the SVRG method for performative prediction, that is, minimizing a loss where the data distribution depends on the parameters being optimized. By adapting the SVRG proof to this setting, the author can establish a 1/T rate to a stationary point. Their proof also drops a previously used "bounded variance assumption".

### Strengths
One clear strength I see is that, as compared to Li & Tai 2024, these new convergence results do not rely on an explicit bounded variance assumption. Though implicitly the variance is bounded by the remaining assumptions. In any case, they use fewer assumptions.

### Weaknesses
The main weakness I see, and the one that my current stance hinges on, is the contradiction between the finite sum structure, and having a parameter dependent distribution.

**Finite sum breaks the data dependent assumption**.  How can you assume finite sum and that $z_i$ depends on $\theta$? By assuming the full loss has a finite sum structure, you loose the functional dependency on $\theta$, and you can no longer differentiate through a discrete sample. That is, having
$$ \nabla \mathcal{J}(\theta;\theta)  = \frac{1}{n} \sum_{I=1}^n \nabla \ell (\theta; z_i)$$ you essentially loose the dependence between the samples $z_i$'s and $\theta$. If not, how exactly are you now parametrizing the sampling of $z_t \in \{z_1, \ldots, z_n\}$ from this finite set so that it depends on $\theta_t$? This is why the related fields I mention below all have to contend with true expectations (infinite population). I read your Appendix G that briefly mentions how to avoid this issue, but it is not enough. You must contend with infinite samples, otherwise there is a contradiction between finite sum and samples depending on $\theta$.



**Limited related work.** This problem as stated 

$$ \min\_{\theta \sim \mathcal{D}\_{\theta}} \mathbb{E}\_{z \sim \mathcal{D}\_{\theta}}[\ell(\theta, z)] $$

Is verbatim the same problem stated in numerous fields, including RL, Optimal control, Variational Inference, and more. It has been an ongoing challenge in all of these fields to establish a non-asymptotic converges of SGD and variance reduced versions. These different fields have to deal with the exact same issues you address, such as having biased estimates of the gradient, assuming smoothness over a support of $z$, and analogous variance estimates. Because of this, you should extend your background section beyond predictive performance, and explain to the reader why this hasn't been done in RL/Control/VI before. A few standard references in convergence of SGD in RL below, but you should cast a wider net:

Lin Xiao, On the Convergence Rates of Policy Gradient Methods,  Journal of Machine Learning Research 23 (2022) 1-36

Alekh Agarwal, Sham M. Kakade, Jason D. Lee, and Gaurav Mahajan. On the theory of policy gradient methods: optimality, approximation, and distribution shift. J. Mach. Learn. Res. 22, 1, Article 98 (January 2021), 76 pages.

Rui Yuan, Robert M. Gower Alessandro Lazaric, A general sample complexity analysis of vanilla policy gradient, AISTATS 2022




**Variance independent bounds** This emphasis on not having "variance" appear in your bound is perhaps overplayed. This is because you assume the loss is $L_0$-Lipschitz, which bounds the gradient norm, and thus plays now the role of "variance" in your bounds. To be specific, using Lemma 5.2 from Li & Wai, assuming $L_0$-Lipschitz loss and $\epsilon$--sensitivity, you have immediately that Assumption 3.4 holds with $\sigma_0^2 = 2L_0^2 + 2\epsilon^2 L_0^2$ and $\sigma_1^2 =0.$ As such, I think this emphasis on not having dependency on variance is a bit confusing. A stronger message is that you simply rely on less assumptions (you do not explicitly assume  Assumption 3.4.
*Questions*

### Questions
Your experiments report training accuracy. Is there a corresponding notion of test accuracy here for  performative prediction? I'm guessing there is no such notion. Which make me think that online learning is a better paradigm for this problem.



**Minor issues**

1. Assumption 1: You should state $\forall z \in \mathcal{Z}.$

2. Definition 1. Because you evaluate the full batch gradient on the same $\theta$ as the one from which $z$ is sampled, this condition is exactly the standard $\delta$ stationarity condition, why has it been renamed to "$\delta$-Stationary Performative Stable Solution"? I find this confusing.


3. Formal statement of assumptions. You need to be crystal clear about your assumptions. In a formal Lemma, to be clear, you should always include the assumptions in the statement. For instance, does each Lemma 5.1 and 5.3 use all of your assumptions or only a subset? These things have to be clear for a theory paper.

4.  On a subjective note here, the name "Performative Prediction" is a very unfortunate name. It does not give any hint that this is an optimization problem where the data distribution is parameter dependent. This naming will make it harder for researchers in RL/control/VI working on this exact problem to find these works.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Strengths
1. Innovative integration of variance reduction into performative prediction: This paper is the first to systematically introduce variance reduction techniques into the performative prediction framework, extending the convergence theory to the non-convex setting.

2. Strong theoretical guarantees under nonconvex settings: Theoretical analysis establishes an O(1/T) convergence rate without assuming bounded gradient variance, representing a major improvement over prior stochastic PP methods which only achieved O(1/\sqrt{T}) rates with variance-dependent error bounds.

3. Empirical results align with theory: Experiments on both tabular (credit scoring) and vision (CIFAR-10, MNIST) datasets show consistent improvements in convergence speed and stability across varying performativity strengths, supporting the theoretical claims.

Weaknesses

1. Experimental scope is narrow: Evaluations are limited to small-scale networks and datasets. The method’s scalability and robustness under more realistic, high-dimensional or large-data performative environments remain untested.

2. Complexity results lack empirical confirmation: The complexity analysis is solid theoretically, but the paper does not include runtime or gradient-call comparisons to validate practical efficiency gains.

### Strengths
see Summary

### Weaknesses
see Summary

### Questions
NA

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
4

### Summary
This work considers the setting of Performative Prediction (PP), a novel risk minimization framework proposed by [Perdomo et al, 2021]. Unlike the tradietional ERM setting, PP assumes that the underlying data distribution also depends on the model parameters. This line of work gained much attention in the recent years, and much effort is devoted to finding a performative stable solution using variants of SGD, under convex/nonconvex loss function with bounded variance.

The authors observe that the nonconvex analysis in [Li and Wai., 2024] suffers from a critical reliance on a growth condition for the stochastic noise and an $O(1/\sqrt{T}) $ rate. To overcome this limitation, they propose a new algorithm and incorporate the variance-reduction technique. The authors further show that their method achieves an $O(1/T)$ convergence rate without assuming the previous noise condition.

### Strengths
The paper is clearly written and well-structured. The theoretical results appear sound. The proofs are well organized and generally convincing.

### Weaknesses
My concern majorly comes from a contribution perspective. The improvements appear to be restricted to a rather narrow setting of performative prediction. Moreover, the incorporation of variance reduction techniques follows a fairly standard approach. As a result, the paper does not seem to offer many new insights for the broader optimization community. Unless the authors can present more decisive evidence on the significance of their work, the paper in its current form falls short of the expectations for an ICLR publication.

### Questions
No further questions.

### Soundness
3

### Presentation
3

### Contribution
2
