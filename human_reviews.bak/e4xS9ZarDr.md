# Lion Secretly Solves a Constrained Optimization: As Lyapunov Predicts

- Decision: Accept (spotlight)
- Scores: 8, 6, 8, 8

## Abstract
Lion (Evolved Sign Momentum), a new optimizer discovered through program search, has shown promising results in training large AI models. It achieves results comparable to AdamW but with greater memory efficiency. As what we can expect from the result of the random search, Lion blends a number of elements from existing algorithms, including signed momentum, decoupled weight decay,  Polayk and Nesterov momentum, but doesn't fit into any existing category of theoretically grounded optimizers. Thus, even though Lion appears to perform well as a general-purpose optimizer for a wide range of tasks, its theoretical basis remains uncertain. This absence of theoretical clarity limits opportunities to further enhance and expand Lion's efficacy. This work aims to demystify Lion. Using both continuous-time and discrete-time analysis, we demonstrate that Lion is a novel and theoretically grounded approach for minimizing a general loss function $f(x)$ while enforcing a bound constraint $||x||_\infty \leq 1/\lambda$. Lion achieves this through the incorporation of decoupled weight decay, where $\lambda$ represents the weight decay coefficient. Our analysis is facilitated by the development of a new Lyapunov function for the Lion updates. It applies to a wide range of Lion-$\phi$ algorithms, where the  $sign(\cdot)$ operator in Lion is replaced by the subgradient of a convex function $\phi$, leading to the solution of the general composite optimization problem $\min_x f(x) + \phi^*(x)$. Our findings provide valuable insights into the dynamics of Lion and pave the way for further enhancements and extensions of Lion-related algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper analyzes the recently introduced Lion optimizer, which was found previously in the literature through program search, and shows how it fits into a general family of optimizers, Lion-$\mathcal{K}$ which are shown to optimize a specific regularized/constrained version of the original cost function. This optimizer also recovers several known optimization methods from the literature such as mirror descent, momentum or weight decay. The analysis of this optimizer is done by proposing a new general of Lyapunov function yielded by the dynamics of Lion-$\mathcal{K}$.  Results are provided in the continuous time and discrete time setting. Additional experimental results are provided to illustrate the training dynamics.

### Strengths
### Originality

I believe the work is original, as it is, up to my knowledge, the first to propose such a general general Lion-$\mathcal{K}$ algorithm along with the corresponding Lyapunov function for its dynamics. In addition the derivation of such a Lyapunov function seems non-trivial, and I believe it will be of interest to the community.

### Quality

I believe the quality is good, as theorems are clearly stated with their assumptions, and full proofs are well detailed in appendix, and with an overview in the main body. Particular care was also taken in the experimental analysis to verify the theoretical results in practice.

### Clarity

The work is clear, theorems are clearly stated as well as their assumptions.

### Significance

I think this work is very significant, since Lion is currently a state of the art optimizer in deep learning, therefore analyzing its dynamics is of very high importance to the machine learning community. Additionally, the discussion on the relationship of Lion-$\mathcal{K}$ with other methods from the literature such as Mirror Descent, Momentum etc makes this analysis even more relevant and general.

### Weaknesses
I just have a question regarding the impact of assuming differentiability of $\mathcal{K}$ in the continuous-time analysis (see question below).

I also noticed just a few minor remarks/typos:
- “Going beyond Lion, difference $\mathcal{K}$” —> “Going beyond Lion, different $\mathcal{K}$”
- “One can think Lion” —> “One can think of Lion”
- “Gradient Enhacenment” —> “Gradient Enhancement”
- “Section 3 analysis” —> “Section 3 analyzes”
- Appendix B.6 : “Becomes a contraint optimization” —> “Becomes a constrained optimization”
- Although I think the Figures 1-3 are nice, unless I am mistaken those are not mentioned in the main text: I believe it could be good to mention them, for instance after the discussion on the different phases of training.

### Questions
I just have one question regarding the impact of the non-differentiability of $\mathcal{K}$: 

(a) In the continuous-time setting, it is assumed that $\mathcal{K}$ is differentiable. Then in such case the convergence to a stationarity point of the constrained/penalized function for the (continuous-time) algorithm is established in Theorem 3.1. 

(b) Then, a discrete-time analysis is provided, which allows $\mathcal{K}$ not to be differentiable, but however in such case I think the result is a little bit more difficult to interpret: indeed, the only result provided is  Thm 4.1, which is about $H(x_{t+1}, m_{t+1}) - H(x_{t}, m_t)$ (or, alternatively, about the sum of deltas), but unless I am mistaken, there is no result that looks similar to convergence to a stationary point of the constrained objective (there is also Theorem B.9 in Appendix but that one describes only the first phase of the dynamics). Therefore, I think it is a bit hard to verify from reading such theorem whether the conclusions of the discrete-time case (which correspond to the algorithm used in practice) will actually match the conclusions from the continuous-time case.

More precisely, those two theoretical results above are interesting in themselves, but I am just wondering to what extent they can indeed successfully predict the actual behaviour of the Lion-$\mathcal{K}$ method: indeed, as mentioned in the paper, Lyapunov analyses need $\mathcal{K}$ to be differentiable.  When the function $\mathcal{K}$ is not differentiable (as it is the case for the $\ell_1$ norm which is considered in the paper), analyses can break. But I believe this is more than just a technicality, i.e. I think the behaviour of algorithms for differentiable vs. non-differentiable $\mathcal{K}$ may be quite different: indeed, in the simple case of mirror descent (section 3.1), for instance, writing the discrete Mirror Descent update naively with $\mathcal{K}$ taken as the $\ell_1$ norm gives an algorithm of the form $x_{t+1} \leftarrow \text{sign}(x_t - \nabla f(x_t))$, which clearly does not converge in general to $min_x f(x)$ as iterates remain restricted in $\\{ +1, 0, -1\\}^{d}$ (the same problem would also happen for the continuous case Mirror Descent). But however, I think that having a non-differentiable $\mathcal{K}$ for the usual discrete-time Frank-Wolfe is OK.

Therefore, I believe it would be good to elaborate more (and/or give more references), in the continous case, on why the analysis of Lion-$\mathcal{K}$ would still somehow be valid or almost valid for non-differentiable $\mathcal{K}$, and/or, in the discrete-case setting, to give more details or a more intuitive reading of the result in Theorem 4.1. I think this would allow the reader to further confirm that what Lion-$\mathcal{K}$ is doing is indeed constrained/penalized minimization, without worrying that issues similar to the ones of Mirror Descent happen in the case of Lion-$\mathcal{k}$.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper provides a theoretical understanding of the Lion optimizer, a new optimizer discovered by program search. The authors introduced a general class of optimizers, Lion-$\mathcal{K}$, and showed that these optimizers minimize the training loss $f(x)$ with an additional constraint on $\mathcal{K}^*(x)$ via a Lyapunov based analysis. The Lion optimizer corresponds to the case where $\mathcal{K}
$ equals the $L_1$-norm of $x$, thereby imposing a constraint on $\|\|x\|\|_{\infty}$. Extending beyond Lion, the proposed Lion-$\mathcal{K}$ also encompasses algorithms such as Polyak/Nesterov Momentum, Singed Momentum, and (Accelerated) Mirror Descent.

### Strengths
The main idea of this paper is novel and interesting. It offers a fresh perspective on the theoretical understanding of Lion and sheds light on the role of the key components, such as the (decoupled) weight decay and the gradient enhancement (the Nesterov trick). The proposed family, Lion-$\mathcal{K}$, also encompasses a wide range of optimizers.

### Weaknesses
1. This manuscript appears to be incomplete. Notably, there is a lack of discussion on related works, and the empirical results presented in Figures 1-3 are neither interpreted nor mentioned, even though there seems to be sufficient space for such discussions. I strongly recommend that the authors enhance the completeness and rigor of this paper via:
   - Interpretation of the empirical results, 
   - Ablation studies on the key components of Lion to justify the theory, 
   - Empirical validation of (at least some) algorithms listed in Table 2. 

2. This paper does not connect the superior performance of Lion or other Lion-$\mathcal{K}$ optimizers to their main finding. How does this additional constraint help optimization or generalization?


3. (Minor) Some typos:
    - In the definition of the conjugate function below eq (4), the subscript should be $z$ instead of $x$. 
    - Above Lemma 2.1: crucial rule -> crucial role. In Lemma 2.1, $\nabla \mathcal{K},\nabla \mathcal{K} \to \nabla \mathcal{K},\nabla \mathcal{K^*} $.

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Authors use the analysis of the Lyapunov function to prove the theoretical guarantee for a broader family of Lion-K algorithms, in which the lion algorithm is part of. Authors also discuss connections of Lion-K with existing algorithms. They also emphasize it is still lack on the physical intuition on this algorithm.

### Strengths
This paper is very well written and super clear. Authors make every effort to have all the points stated out in a very concise way, even for the audience without too much background. The key contribution is they found the Lyapunov function for this family which is completely non-trivial. The guessing requires lots of trial and error. With the Lyapunov function, the optimization schemes are theoretically guaranteed

### Weaknesses
To demonstrate efficiency in the conclusion, authors didn't compare with other methods. At least, it is necessary to include other benchmark algorithms in empirical evaluations.

### Questions
1. Can you comment on the convergent rate of lion and lion-K algorithms, based on the current results? Why it performs comparable or favorably to AdamW? 

2. Can you comment the possible convergence on lion-K vs interesting decomposition (12)? It seems they may both work well.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the Lion optimizer, which is discovered through program search. It is shown that Lion indeed minimizes the loss while constraining the $\ell_\infty$ norm of the parameters. Specifically, the authors proposed a general framework for analyzing a general class of Lion-type algorithms, which lead to solutions of general composite optimization problems determined by reshaper function. Theoretical analyses revealed the two-phase dynamics of such algorithms: In the first phase, the iterates convergences to the constrained domain exponentially fast; then in the second phase, the dynamics minimizes the objective.

### Strengths
The paper is well written and very easy to follow. The proposed framework is very interesting, not only demestifying the Lion optimizer, but also encompassing many other algorithms. The authors have clearly explained the idea and discussed thoroughly the related background. The results seem solid and novel.

### Weaknesses
1. The current analyses apply to only algorithms with full gradient. It would be interesting to see results for Lion with stochastic gradients.
2. It would be helpful if the authors can discuss and comment on the implications of Theorem 4.1. Also, it is worth explaining why it is necessary to use a different implicit scheme.
3. Typo:
    - In the first sentence in the paragraph above Figure 4, "difference $\mathcal{K}$" -> "different $\mathcal{K}$"
    - The second paragraph above Lemma 2.1, "a systematic introduce to ..." -> " a systematic introduction to ..."

### Questions
In Section 3.1, the authors discuss the connection with existing algorithms. Do the corresponding convergence results also reduce to those classic ones?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent
