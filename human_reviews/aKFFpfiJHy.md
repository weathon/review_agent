# Linear Bandits with Partially Observable Features

- Decision: Reject
- Scores: 3, 6, 3, 8

## Abstract
We introduce a novel linear bandit problem where a subset of features is latent, resulting in partial access to reward information and spurious estimates.
Without properly addressing the latent features, the regret grows linearly over the decision epoch $T$ while improving the regret bound is challenging because their dimension and relationship with rewards are not available.
We propose a novel analysis to handle the latent features and an algorithm that achieves a regret bound sublinear in $T$.
The core of the algorithm lies in (i) augmenting basis vectors orthogonal to the observable feature space, and (ii) developing an efficient doubly robust estimator that further improves the regret bound.
With these two ingredients, our algorithm achieves a regret bound of $\tilde{O}(\sqrt{(d + d\_h)T})$, where $d$ is the dimension of observable features, and $d_h$ is the _unknown_ dimension of the unobserved features that affects the reward. 
Crucially, our algorithm does not rely on prior knowledge of the unobserved feature space, which expands as more features become hidden.
Numerical experiments confirm that our algorithm outperforms both non-contextual multi-armed bandits and other linear bandit algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper studied linear bandits where the feature vectors of the actions are partially observable, specifically in the setting where there are no additional assumptions about the unobserved features. The authors observed that the reward of all actions lies in a subspace of $d$ dimensions, allowing for an assignment of the hidden vectors such that the corresponding unknown parameter $\mu_*$ is $d$-sparse. Based on this insight, the authors proposed a bandit algorithm that employs the Lasso estimator to leverage this sparsity. They showed that their algorithm achieves a regret bound of order $\sqrt{\tilde \sigma_{\min}^{-1} T d \log K}$, where $\tilde \sigma_{\min}$ denotes the minimum eigenvalue of the Gram matrix of the assigned feature vectors. Experiments are included to demonstrate the practical performance of the algorithm.

### Strengths
I think linear bandits with partially observed features is an interesting topic to study.

### Weaknesses
I feel there are critical theoretical flaws in the paper.

1. Even if $(I_K - P_X)U^{\top} \theta_*^{(u)}$ lies in a subspace of dimension $d$, it does not imply that $\mu_*^{(u)}$ is $d$-sparse. The sparsity only holds when the basis vectors $b_i$ align with the row space of the matrix. It seems to me that $b_i$ is arbitrarily chosen, so I do not think this claim generally holds.

2. The regret bound of the algorithm has a polynomial dependence on $\tilde \sigma_{\min}^{-1}$. However, since $\tilde \sigma_{\min}^{-1}$ is the minimum eigenvalue of the Gram matrix of a $K \times d$ matrix, it might still scale as $K$. Although the authors claim that the minimum eigenvalue is not always $O(1/K)$ because they assume $\|\tilde x_a\|_\infty \leq 1$ instead of $\|\tilde x_a\|_2 \leq 1$, I still cannot see the correspondence.

3. In Theorem 2, it is stated that $\mathcal{E}_t \subseteq [t]$, and it is also assumed that $|\mathcal{E}_t| \geq t$. I do not see how both conditions can hold unless $\mathcal{E}_t = [t]$.

### Questions
Please refer to the Weaknesses section above.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies a novel linear contextual bandits problem, which differs from the standard finite-arm linear contextual bandits problem in the following way. 
1. There are $K$ arms in total and the contextual vector of each arm is fixed through horizon $T$.
2. Among all dimensions $d$, there are a few of them are not observed by the player.
3. The contextual vector has bounded $\ell_\infty$-norm and the parameter $\theta^\star$ has bounded $\ell_1$-norm.

The paper presented an algorithm for the problem, gave matching lower and upper bounds up to logarithmic factors, and demonstrated the effectiveness of the paper in numerical experiments.

### Strengths
1. The paper gave a rather complete study on the proposed problem. As mentioned before, the paper included algorithm designs, proofs to the lower and upper bounds, and numerical experiments.
2. The lower and upper bounds are matching up to log factor.

### Weaknesses
1. The paper made a rather strong assumption that $(\mu,\theta)$ are bounded in $(\ell_\infty, \ell_1)$ which is different from the standard $(\ell_2, \ell_2)$ setup. This limits the comparison of this work over previous ones.

### Questions
1. Line 1325: "Cauchy-Schwartz" should be "Cauchy-Schwarz".
3. Is the observed part of the feature space really used? What would happen if we just pretend all features are not observable and apply the algorithm, i.e. replace $d,d_h$ by $d'=0$ and $d'_h=d+d_h$? It confused me a lot that the regret bound only depends on $d+d_h$.
2. I suggest the authors make more explicit about the regret bound's dependency on $\widetilde \sigma_{\min}^{-1}$ to avoid confusion.
3. At Line 475-480, I don't see why $\widetilde \sigma_{\min}$ is constant.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper consider a bandit setting with a fixed set of arms and with unobservable features. The authors propose to augment the feature space with basis vectors and propose algorithms to provide sublinear regret guarantee.

### Strengths
Designing algorithms to tackles unobservable confounding factors is important for bandit problems.

### Weaknesses
1. Assumption 1 requires features of all arms to be fixed over time, which is not common and very strong. I don't think this assumption matches practices. Besides, I didn't see how this can be "slightly modified" into a time-varying feature setting as claimed by the authors. If it's simple, please use the time-varying feature setting as the main setting, which is common in the bandit literature. 
2. The lower bound analysis doesn't seem right to me. The last step relies on the argument that the agent cannot identify $x_{a_*}$ from $x_{a'}$. However, the observed outcome $y$ for these two arms $a_*$ and $a'$ will be different since $\langle z_{a_*}, \theta_*\rangle \ne $\langle z_{a'}, \theta_*\rangle$, even though $x_{a_*}=x_{a'}$. Therefore, a linear regression will be able to distinguish these two actions. Intuitively, even if there are unobservable features, linear regression will still be able to tell one action from others since the unobservable values will be projected into the observable feature space. Thus, it's hard to believe the lower bound will scale as T. 
3. The coupling steps require more explanations and intuition as well. The authors only argue adaptation of the existing DR method is challenging but didn't explain the intuition. Why coupling can help with the analysis when arms are fixed?
4. The comparison in the experiments does not seem fair. The benchmarks in the experiments are not of the same type of bandit algorithm with the proposed ones. I would suggest the authors at least add DR Lasso Bandit (Kim & Paik, 2019).

### Questions
See my comments above.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors consider a regret minimizing linear bandits with K arms. Unlike the standard linear bandit setting, they assume that each arm has a set of unseen or latent features which affect the reward. As a result, standard linear bandit algorithms (eg OFUL) can’t be applied. The authors address this issue by projecting the problem to an appropriate space where an orthogonal complement to the span of the arms is augmented, and then performing an epsilon greedy style approach in this space. Critical to the solution is the use of Doubly robust estimation procedures. They show strong empirical results.

### Strengths
Overall I rather enjoyed this paper and thought the approach of the authors was clever. Intuitively, in an extreme case, if you do not observe any features, you can’t hope for better than O(\sqrt(KT)) regret coming from standard MAB results. However, by projecting the feature vectors appropriately, the authors can exploit linear dependencies between the unseen components and seen components to do better. 

What’s perhaps most impressive is Theorem 2, which provides an estimator that can effectively bound the estimation error using a doubly robust estimator.

### Weaknesses
I think there are some technical weaknesses and gaps:
a) It seems unfortunate that Theorem 2 has a burn in period that depends on K. Can the authors argue that this is necessary? (It seems so?)
b) I think some contextualization of Theorem 2 would be helpful - for example, which components of the error come from the bias of using 10 to estimate rewards? c) Forced uniform exploration seems critical to your method - do you think you could have avoided it?
D) It feels like the regret could be as bad as O(sqrt(KT)) - for example, if the seen features all have the same inner product with theta_s, and the unseen dimension encodes a basis. Can you clarify those cases?

Comments on Structure:
Overall I thought the paper was well written, but there were some gaps in exposition that would have helped. Here’s some concrete suggestions
a) Perhaps add a table contextualizing the regret in different ranges. Eg, how big can d_h get?
b) I think some concrete examples explaining your extrapolation between MAB and the linear bandit and how DR estimation helps this can help.

Finally I thought the discussion on sparsity was interesting…but I failed to understand it. d_h could be much smaller than d_u, but this does not imply sparsity?

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3
