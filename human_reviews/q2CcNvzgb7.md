# Learning Nash Equilibria in Normal-Form Games via Approximating Stationary Points

- Avg Score: 3.75
- Decision: Reject
- Scores: 3, 3, 6, 3

## Abstract
Nash equilibrium (NE) plays an important role in game theory. However, learning an NE in normal-form games (NFGs) is a complex, non-convex optimization problem. Deep Learning (DL), the cornerstone of modern artificial intelligence, has demonstrated remarkable empirical performance across various applications involving non-convex optimization. However, applying DL to learn an NE poses significant difficulties since most existing loss functions for using DL to learn an NE introduce bias under sampled play. A recent work proposed an unbiased loss function. Unfortunately, it suffers from high variance, which degrades the convergence rate. Moreover, learning an NE through this unbiased loss function entails finding a global minimum in a non-convex optimization problem, which is inherently difficult. To improve the convergence rate by mitigating the high variance associated with the existing unbiased loss function, we propose a novel loss function, named Nash Advantage Loss (NAL). NAL is unbiased and exhibits significantly lower variance than the existing unbiased loss function. In addition, an NE is a stationary point of NAL rather than having to be a global minimum, which improves the computational efficiency. Experimental results demonstrate that the algorithm minimizing NAL achieves significantly faster empirical convergence rates compared to previous algorithms, while also reducing the variance of estimated loss value by several orders of magnitude.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a novel training signal (function) for solving games that is amenable to unbiased estimation. It has lower variance than prior functions that have been proposed. Experiments show that an update rule derived from this function performs well on normal-form games; in these experiments players’ strategies are represented by overparameterized neural networks and updates can be performed more generally with arbitrary DL optimizers.

### Strengths
This paper tackles an interesting problem in computational game theory in trying to develop an improved unbiased estimator of exploitability. Lemma 4.1 in this paper is a key insight and interesting contribution that generalizes the idea of a "projected gradient" from prior work in Gemp et al. 2024. The authors use this to derive a learning algorithm. The proposed algorithm is competitive against baselines in several games from the literature (EFGs converted to NFG form). Also, the idea of sampling actions $a_i'$ from arbitrary distributions (lines 8 & 9 of Algorithm 1) for which to estimate gradients $\nabla_{x,a_i'}$ is novel. Prior work only considered sampling $a_i'$ from uniform.

### Weaknesses
I like the goal of this paper, but there are several technical issues: the proposed “loss” is not a loss in the traditional sense of ML; stop-grad operators ruin essential properties of gradients, especially those that suggest stationary points are easy to find; the proposed algorithm lacks novelty— it is similar to basic simultaneous gradient ascent and more closely to magnetic mirror descent; DL has been applied to NE in prior work (GAES [1], NES [2], NFGTransformer [3]).

1) A loss function is generally accepted to be a function whose value captures some measure of sub-optimality and whose global minimum expresses a desired solution. This is not the case with $\mathcal{L}^{\tau}\_{NAL}$. While $\\mathcal{L}^{\\tau}\_{NAL} = 0$ at a Nash equilibrium of the entropy regularized game, $\\mathcal{L}^{\\tau}\_{NAL}$ may be less than 0 for some other strategy profiles. This would suggest there exist profiles that are “closer” to a Nash equilibrium than actual equilibria, which is counter-intuitive. One can confirm this phenomenon with a simple example; consider the game where the row player’s payoff matrix is $\\begin{bmatrix} 0 & 0 \\\\ 1 & 1 \\end{bmatrix}$ and the column player’s is the transpose. If you plot $\\mathcal{L}^{\\tau}\_{NAL}$ for $\\tau=0.2$ and $\\hat{x}_i = \\mathbf{1} / \\vert \\mathcal{A}\_i \\vert = [1/2 ,1/2]$ while varying each player’s mixed strategy $x_r = [p, 1 - p]$ and $x_c = [p, 1 - p]$, you will see the loss is negative for a range of strategies. For this reason, please plot the actual loss function in experiments. I do not doubt the empirical results of Figure 3 that show the difference between the true expected loss and its estimated value; that is expected. But I would not be surprised if a plot of the true expected loss shows negative values at times.

2) Defining loss functions with “stop grad” operators in them generally leads to “pseudo-gradients”, i.e., their gradients no longer satisfy critical properties that gradients normally satisfy. For instance, it is true, as you state, that finding a stationary point of a loss function is easier than finding a global minimum (Jin et al. 2017). However, finding a stationary point of an arbitrary vector field (i.e., a point where a pseudo-gradient is zero) is nontrivial. Note that by searching for a Nash equilibrium of an entropy regularized game, you are computing a quantal response equilibirum (QRE). It is known that computing a QRE is PPAD-hard [4]. And there is another reason you can expect finding a stationary point to be hard which I discuss next.

3) The algorithm proposed is essentially simultaneous gradient ascent but with entropy regularized gradients and decaying temperature coefficients. This is quite similar to prior approaches, e.g., magnetic mirror descent [5]. Also, it has been proven that there exists no continuous update dynamics that converge to Nash equilibria in generic games [6,7], which includes the update proposed in this paper. Lastly, note that if one projects equation (3) onto the tangent space of the simplex, one obtains the projected gradient defined in Gemp et al. 2024: $\\Pi_{T\\Delta}[\\nabla_{x_i} \\mathcal{L}^{\\tau}\_{NAL}] = F_i^{\\tau,x} - \\langle F_i^{\\tau,x}, \\mathbf{1} / \\vert \\mathcal{A}\_i \\vert \\rangle \\mathbf{1} = -\\Pi_{T\\Delta}(\\nabla_x u_i^{\tau})$ regardless of the $\\hat{x}_i$ chosen. Based on this, I would expect the behavior of the update proposed in this paper to behave similarly to simultaneous gradient ascent (with regularization).

4) The proofs of Theorems 4.2 and 4.3 follow from minor modifications of results in Gemp et al. 2024, e.g., Lemmas 2 and 12 therein. Also, how does the duality gap of your approach in Thoerem 4.2 compare to Gemp et al. 2024? Presumably, since $\Pi_{T\Delta}(\\nabla_{x} \mathcal{L}^{\\tau}\_{NAL}) = -\Pi_{T\Delta}(\\nabla_x u_i^{\tau})$, then $||\Pi_{T\Delta}(\\nabla_x u_i^{\tau})|| \le ||\\nabla_{x} \mathcal{L}^{\\tau}\_{NAL}||$, i.e., the bound proposed here is not as tight. Do you agree?

Minor:
- The dot notation is unideal given it’s widely accepted to mean time derivative, $\\dot{x} = dx/dt$. Prior work uses $sg[\\cdot]$ for the "stop grad" operator which is clearer and more explicit. I would suggest writing equation (3) as $\\mathcal{L}^{\\tau}\_{NAL}(\\boldsymbol{x}) = \\sum\_{i \\in \\mathcal{N}} \\langle sg[ F^{\\tau,\boldsymbol{x}}\_i - \\langle F^{\\tau,\\boldsymbol{x}}\_i, \hat{x}_i \\rangle \mathbf{1} ], x_i \\rangle$ with the explicit $sg[\\cdot]$ operator or as $\\mathcal{L}^{\\tau}\_{NAL}(\\boldsymbol{x}) = \\sum\_{i \\in \\mathcal{N}} \\langle F^{\\tau,\\tilde{\\boldsymbol{x}}}\_i - \\langle F^{\\tau,\\tilde{\\boldsymbol{x}}}\_i, \hat{x}_i \\rangle \mathbf{1}, x_i \\rangle$ where the first term in the inner product evaluates $F$ at $\\tilde{\\boldsymbol{x}}$.
- In equation (6), please state the distribution that the expectation is taken w.r.t., i.e., $\\mathbb{E}_{s \\sim p_i(s)}$.
- Theorem 4.2 defines the utility function $u_i^{\\tau}(x) = u_i(x) - \\tau x_i \\log(x_i)$. The second term is currently interpreted as a vector. It is missing a sum or a transpose, i.e., it should be $u_i^{\\tau}(x) = u_i(x) - \\tau x_i^\\top \\log(x_i)$.

[1] Goktas, Denizalp, et al. "Generative Adversarial Equilibrium Solvers." The Twelfth International Conference on Learning Representations.

[2] Marris, Luke, et al. "Turbocharging solution concepts: Solving NEs, CEs and CCEs with neural equilibrium solvers." Advances in Neural Information Processing Systems 35 (2022): 5586-5600.

[3] Liu, Siqi, et al. "NfgTransformer: Equivariant Representation Learning for Normal-form Games." The Twelfth International Conference on Learning Representations.

[4] Milec, David, et al. "Complexity and algorithms for exploiting quantal opponents in large two-player games." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 6. 2021.

[5] Sokota, Samuel, et al. "A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games." The Eleventh International Conference on Learning Representations.

[6] Milionis, Jason, et al. "An impossibility theorem in game dynamics." Proceedings of the National Academy of Sciences 120.41 (2023): e2305349120.

[7] Vlatakis-Gkaragkounis, Emmanouil-Vasileios, et al. "No-regret learning and mixed nash equilibria: They do not mix." Advances in Neural Information Processing Systems 33 (2020): 1380-1391.

### Questions
Questions / Suggestions for improvement:
- Lemma 4.1 is interesting. Can you derive the $y \in \\Delta$ that minimizes variance of the norm of $b - \\langle b, y \\rangle \\mathbf{1}$?
- Why, in lines 8 & 9 of Algorithm 1, do you suggest sampling actions from a strategy that interpolates between uniform and the current strategy? Can you provide any analysis that motivates this decision? Note it appears you set $\\epsilon = 1$ in all experiments anyways, and this is precisely the setting that Gemp et al. 2024 already provided analysis for in their Table 2.
- How does your algorithm behave if you set $\\hat{x}_i=0$? How does it behave if you set $\\hat{x}_i = \\mathbf{1} / \\vert \\mathcal{A}_i \\vert$?
- DL is used as a primary motivation for the paper yet it is not discussed until the experiment section. Why does it make sense to overparameterize a strategy (e.g., 2-action game) with a neural network and then run gradient descent on the network parameters?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper investigates the problem of learning Nash equilibrium in large-scale normal-form games using gradient methods. Given that the game matrix can be enormous, estimating the loss function from samples is essential. By leveraging the fact that, at an interior Nash equilibrium, the deviations of all pure actions of a player should be equal, the authors introduce an additional entropy term into the game utility to ensure the modified game has an interior NE. They then construct a loss function whose first-order gradient can be unbiasedly estimated through importance sampling. Theoretical results demonstrate that the approximate NE of the modified game is also an approximate solution for the original game. Experimental results further showcase the effectiveness and low variance of this loss function.

### Strengths
1. The proposed approach is well-motivated, as it is often difficult to obtain the entire utility matrix, making loss estimation from sampling necessary.
2. Compared to previous relevant work (Gemp et al., 2024), the proposed loss function significantly reduces variance.
3. The authors conduct extensive experiments to demonstrate the efficacy of their approach, showing that the proposed loss function achieves better approximation and lower variance.

### Weaknesses
1. While the experimental results are promising and show a significant reduction in variance compared to Gemp et al. (2024), I find the optimization process of the approach to be unusual and lacking theoretical insights. The proposed loss function $\mathcal{L}_{NAL}^\tau(\mathbf{x}) $ is provided in Equation (2), 
but its gradient does not match the expression $ \langle F_i^{\tau, x} - \langle F_i^{\tau, x},\hat{x}_i \rangle \mathbf{1}, x_i\rangle $ in Equation (3). 
The authors derive the gradient by stopping the backpropagation of gradients in $F_i^{\tau, x}$.
While such an operation can be easily implemented, it lacks mathematical justification. 
The expression in Equation (3) is not necessarily the direction that maximizes the rate of increase for the proposed loss. 
Therefore, I do not believe that optimizing the loss using this incorrect gradient is theoretically sound.
2. When discussing Gemp et al. (2024), the authors state that "learning an NE through this loss function requires finding a global minimum in a non-convex optimization problem, which is widely acknowledged as significantly challenging." However, I did not find that the paper addresses this challenge. We still need to locate a global minimum for $ \mathcal{L}_{NAL}^\tau(\mathbf{x}) $.

### Questions
What is the input to the neural network in your experiments? Is it a constant 1024-dimensional vector? If so, why utilize a neural network? It seems that using a neural network may be redundant in this case.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper solves the problem of approximating Nash Equilibrium in multi-players normal form games (NFGs). The authors adopts a neural network architecture, that outputs the strategy profile of the players. The goal is to train the network, such that the network outputs are close to the solution concept of Nash Equilibrium, measured by Duality Gap.

The authors propose Nash Advantage Loss (NAL) to train the neural network. The advantage of NAL lies in two fold: 1) NAL has an unbiased estimator, while the only loss function that has this property in the literature is the loss function proposed by (Gemp et al., 2024); 2) compared with (Gemp et al., 2024), the unbiased estimator of NAL achieves lower variance.

### Strengths
**Technical Strengths:**

1. The proposed loss function and its corresponding training algorithm (mainly about extracting the unbiased estimators) are novel. This approach also has the potential to approximate the NE of large-scale games.
2. The analysis of the relation between NAL and the NE measure (Duality Gap) as well as the analysis of the variance comparison between NAL and (Gemp et al., 2024) are solid, providing theoretical guarantee of NAL.

**Other Strengths:**

1. The presentation of the paper is logically coherent and the derivation is also detailed, making the paper easy to follow. 
2. The authors conduct sufficient experiments and analysis, showing the applicability of the results.

### Weaknesses
1. It seems unfair to compare the variance of NAL with the variance of loss in (Gemp et al., 2024). It is because the gradient of NAL seems to have square root order magnitudes compared with the gradient of loss in (Gemp et al., 2024).
2. It also seems inappropriate to say this method solves the stationary point of NAL. It is essentially defining a "gradient" in equation (3) for back propagation, which is not the full gradient of the NAL function. The goal is equivalent to finding the zero point of equation(3). Therefore, the claim that this approach improves the tractability by solving stationary point instead of global minimum is questionable.
3. The Lemma 4.1 is in a slightly abrupt position.
4. In Figure 5, the duality gap of NashApr is too strange. Maybe the learning rate is not large enough when the optimizer is SGD.

### Questions
1. Does there exists a well-defined function whose gradient is exactly equation (3)?
2. I'm confused about whether the neural network takes game representation as inputs in the experiments. In Line 45, the authors mention that the game inputs is exponentially large, so direct computing methods (without estimate) suffer from large computational costs. Why do you train a neural network to output the strategy profile rather than optimize on strategy profile $x$ directly?
3. What is the problem size of the NFGs in the paper's experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper studies the problem of computing approximate Nash equilibria. In particularly large games, it is important to be able to run algorithms based on gradient samples similar to standard techniques applied for training NNs. This paper builds upon the recent work of Gemp et al that produced a specific loss function tailored for this task that allowed for unbiased stochastic optimization. This paper defines an alternative loss function that they call Nash Advantage Loss (NAL), which they use for learning Nash equilibria in normal-form games using deep learning. NAL is also unbiased and exhibits lower variance than the pre-existing methodology by Gemp et al.  Furthermore, the author claim that learning a NE by minimizing NAL is more tractable, as an NE is a stationary point of NAL rather than having to be a global minimum as it was required by the previous technique. The paper inncludes experimental results that show the effectiveness of the approach.

### Strengths
- The paper studies an exciting problem of solving for general games using combination of ideas from non-convex optimization, regularization and NNs.
- Compared to the previous methodology of Gemp et al the proposed technique shows reduced variance.
- The technique is validated with experimental results.

### Weaknesses
- The use of stop-gradients in going from the loss function defined in Equation 2 to Equation 3 does not feel perfectly theoretically justified. For example, it is not clear whether there exists a function whose gradient is exactly Equation 3? But if that is not the case, then it is not clear whether the proposed technique is akin to finding local optima of non-convex landscape or more like the more complex problem of finding zeros of more arbitrary vector field. For the second case, there exist impossibility results [1] that show that local learning rules although clearly sufficient for solving the first problem do not suffice to solve the second one. In any case, I believe that these connections need to be ironed out and expanded more formally.  
- The input, parametrization and usage of NNs should be expanded further.
- The paper lacks sufficient comparison/discussion to related prior work. E.g. [2-4]

[1] Milionis, et al. "An impossibility theorem in game dynamics." PNAS 2024
[2] Marris et al. "Turbocharging solution concepts: Solving NEs, CEs and CCEs with neural equilibrium solvers." NeurIPS 2
[3]  Goktas et al. "Generative Adversarial Equilibrium Solvers." ICLR 24
[4] Liu et al. "NfgTransformer: Equivariant Representation Learning for Normal-form Games." ICLR 24

### Questions
Is Equation 3 the gradient of some precise loss function? 
If not, why is it appropriate to consider your approach as finding stationary points?

### Soundness
2

### Presentation
3

### Contribution
2
