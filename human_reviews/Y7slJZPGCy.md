# Efficient Interpolation between Extragradient and Proximal Methods for Weak MVIs

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
We study nonmonotone games satisfying the weak Minty variational inequality (MVI) with parameter $\rho \in (-\tfrac{1}{L}, \infty)$, where $L$ is the Lipschitz constant of the gradient operator. An error corrected version of the inexact proximal point algorithm is proposed, with which we establish the first $\mathcal O(1/\epsilon)$ rate for the entire range $\rho \in (-\tfrac{1}{L}, \infty)$, thus removing a logarithmic factor compared with the complexity of existing methods. The scheme automatically selects the needed accuracy for the proximal computation, and can recover the relaxed extragradient method when $\rho > -\tfrac{1}{2L}$ and the relaxed proximal point algorithm (rPPA) when $\rho > -\tfrac{1}{L}$. Due to the error correction, the scheme inherits the strong properties of the _exact_ rPPA. Specifically, we show that linear convergence is automatically achieved under appropriate conditions. Tightness for the range of $\rho$ is established through a lower bound for rPPA. Central to the algorithmic construction is a halfspace projection, where the key insight is that the allowed error tolerance can both be used to correct for the proximal approximation and to enlarge the problem class.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the nonmonotone games satisfying the weak Minty variational inequality (MVI). The authors propose an inexact hybrid proximal extragradient method. The method converges with the rate of $O(1/\epsilon)$ in the maximal range of $\rho$. Compared with Alacaoglu et al. (2024), the method in this work removes a $O(\log(1/\epsilon))$ factor. The authors also show that linear convergence is automatically achieved under their proposed inexact conditions.

### Strengths
The contribution and improvement over previous methods are clear. The analysis in this paper also seems interesting to me. 

If I understand correctly, using exact proximal point methods to extend the range of $\rho$ to $(-1/L,\infty)$ seems straightforward. 
An additional $O(1/\epsilon)$ naturally appears if using inexact proximal point methods. Therefore the result in Alacaoglu et al. (2024) seems not very surprising. 
But the result without such a log factor seems much more challenging and interesting to me.

### Weaknesses
1. This paper only considers the deterministic case, which seems to be easier than the stochastic case considered by some other works such as [1,2].

2.  Alacaoglu et al. (2024) also consider the cohypomonotone case, with an accelerated rate. The acceleration seems more interesting and challenging.

[1] Alacaoglu, Ahmet, Donghwan Kim, and Stephen J. Wright. "Extending the Reach of First-Order Algorithms for Nonconvex Min-Max Problems with Cohypomonotonicity." arXiv preprint arXiv:2402.05071 (2024).

[2] Chen, Lesi, and Luo Luo. "Near-optimal algorithms for making the gradient small in stochastic minimax optimization." arXiv preprint arXiv:2208.05925 (2022).

### Questions
Most of my questions are about related works.
1. Have the authors considered the stochastic case like [1,2]. Is it possible to remove the log factor using your framework?
2. It seems that a similar idea (using proximal methods) also appears in [2]. Although the result in [2] writes the assumption $\rho > - \frac{1}{2L}$, it seems their proposed algorithm can also be used in the range $\rho > - \frac{1}{L}$ (due to the use the proximal methods) just by changing the constants in their algorithm. Also, [1] applies very a similar MLMC technique as used in [2]. Could the author please tell me if my understanding is correct and compare these related works for me?
3. The method in this paper looks similar to [3]. What are the differences between the method in this paper and [3]?

[1] Alacaoglu, Ahmet, Donghwan Kim, and Stephen J. Wright. "Extending the Reach of First-Order Algorithms for Nonconvex Min-Max Problems with Cohypomonotonicity." arXiv preprint arXiv:2402.05071 (2024).

[2] Chen, Lesi, and Luo Luo. "Near-optimal algorithms for making the gradient small in stochastic minimax optimization." arXiv preprint arXiv:2208.05925 (2022).

[3] Monteiro, Renato DC, and Benar F. Svaiter. "Iteration-complexity of a Newton proximal extragradient method for monotone variational inequalities and inclusion problems." SIAM Journal on Optimization 22.3 (2012): 914-935.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Non monotone games, which are operators that are non convex and non concave appear in several applications and are proven to be hard to solve. One important class of problems within this is the class of problems satisfying the weak Minty Variational Inequality. Thiis was one of the counter examples proving the hardness of the broader family of problems. The weak minty condition is a relaxation of monotonicity, i.e., roughly for some $\rho$, we require for all solutions $z^{\star}$ of an operator S, 
$$ \langle S(z) , z-z^{\star}\rangle \geq \rho \|S(z)\|_2^2.$$

There has been a flurry of work that has tried to give algorithms for different values of $\rho$. When the operator is $L$-Lipschitz, the relaxed extragradient method allows $rho > -1/2L$ and the relaxed proximal point algorithm allows a larger range, $\rho > -1/L$ but loses an additional $log 1/\epsilon$ factor. This paper considers the range $-1/L<\rho <-1/2L$ and gives an algorithm that gets rid of this extra $log 1/\epsilon$ factor. Their method also unifies the two methods mentioned above. The additionally also obtain some kind of linear convergence under additional assumptions that I dont fully follow.

Their algorithm follows an inexact proximal point algorithm, where they require the errors of every prox step to be small in order to obtain convergence. Normally this would involve increasing the number of iterations, but the authors get away with this using “half-space projections” inspired by Solodov and Svaiter’99.

### Strengths
The paper is able to unify a few methods to get one coherent method for solving the Minty Variational Inequality problem to an optimal rate. They also improve a $\log 1/\epsilon$ factor for an important range of parameters. It is overall reasonable well written, though I would hope that the next version has background which is easier to parse for a newcomer in the field.

### Weaknesses
I think the paper would improve if there is a more detailed summary of what the previous methods do. I understand the space limitations in the main text but maybe something in the appendix would help. Currently there is a brief overview in the introduction but since the method unifies certain methods, it would be nice to know in somewhat more detail what the unification is.

### Questions
Some suggestions to the authors:
1. Please check the introduction, it has several grammatical errors.
2. If possible please introduce the halfspace method. It seems to show up without introduction and it is not clear what it means in Section 4
3. It is also not clear to me why the halfspace method works in getting better accuarcy? Also is it the same as the Ellipsoid method? I can imagine that the ellipsoid method would work the same way for convex functions.
4. I dont follow what the linear convergence means. Is it for the inner loop?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces an inexact proximal point algorithm with an error correction mechanism, which guarantees convergence for minimax problems that satisfy the weak Minty variational inequality (MVI) with a parameter  $\rho>-1/L$. The proposed method achieves a convergence rate of  $\mathcal O(1/\epsilon)$ without the logarithmic complexity factor found in previous approaches, making it a promising solution for weak MVIs in constrained settings.

### Strengths
* The proposed algorithm achieves better iteration complexity by avoiding the logarithmic factor present in previous methods.

* The performance results shown in Figure 2 are impressive and highlight the algorithm's effectiveness.

* The scheme unifies the analysis of the relaxed extragradient method, the relaxed proximal point algorithm (rPPA), and classical methods for monotone problems.

### Weaknesses
* It would be helpful if the authors elaborated on the intuition behind the necessity of multiple inner loops when $\rho<-1/2L$. Further insight into why these loops are essential could enhance understanding of the algorithm’s design choices.

* In Section 8, the authors mention the possibility of extending the range of $\rho$ beyond $1/L$. Including experimental results to illustrate this potential would strengthen the paper’s claims and provide evidence for the method’s flexibility under various settings of $\rho$.

*  The paper would benefit from additional empirical evaluation to demonstrate its applicability to complex real-world problems. Alternatively, a more explicit discussion on the practical relevance of these theoretical results could help clarify their significance for challenging problem settings.

* The paper’s notation is somewhat confusing, as the authors sometimes use the range of $\gamma$ to imply constraints on 
$\rho$ and other times directly specify the range of $\rho$. Since $\rho$ and $\gamma$ are related, a unified notation throughout would make the technical content easier to follow, especially given the paper’s density.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

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
This paper investigates variational inequalities that satisfy the weak Minty variational inequality (MVI) condition and proposes a hybrid proximal extragradient method. The authors establish improved complexity bounds for both the implicit and computationally efficient explicit versions of this method, extending previous results to a broader range of the MVI parameter $\rho$. The tightness of this range is confirmed through a lower bound analysis for the exact version of the method. Additionally, linear convergence is shown to be achievable under appropriate conditions.

### Strengths
1. **Theoretical Results**: This paper achieves superior complexity and convergence over a broader range of $\rho$. In particular, the authors establish an upper bound for the number of inner iterations that is independent of target accuracy. The theoretical results are further enriched by a lower bound analysis, thoroughly addressing the questions posed in the introduction.
2. **Algorithm Design**: The authors construct the algorithm based on a halfspace projection operator, yielding a hybrid method that combines the inexact proximal point method with the extragradient method. This novel perspective offers a systematic approach to determining the step size and recovers several algorithms from prior work under specific conditions.
3. **Presentation**: The paper is well-organized, easy to follow, and contains extensive content.

### Weaknesses
1. **Significance (major concern)**: While this paper addresses the questions raised in line 47, the practical significance may be limited. Previous work has achieved $O(1 / \epsilon \log (1 / \epsilon) )$ complexity for $\rho > -1/L$; the marginal gain from removing the logarithmic term (which is much smaller than $1/\epsilon$) is unclear. Additionally, the $O(1/\epsilon)$ complexity is already achievable for $\rho > -1/(2L)$, so the practical value of studying $\rho \in (-1/L, -1/(2L)]$—a narrow range—seems limited.

2. **Presentation Complexity**: The relationship between the proposed algorithm and existing methods is somewhat complex. A diagram could help clarify this relationship. Additionally, the placement of Algorithm 1 feels distant from Section 6.

### Questions
1. The authors refer to the halfspace projection as an *error correction* step. When comparing Eqs. 9 and 10, if we substitute the first equality condition from Eq. 8 into Eq. 9, we get $z^{k+1} = (1-\lambda_k) z^k + \lambda_k (z^k - \bar{v}^k - \epsilon^k)$. Can we interpret the error correction as eliminating the *explicit* error term $\epsilon^k$ in Eq. 10 through an additional operator evaluation $\bar{v}^k \in \gamma S \bar{z}^k$? Consequently, although the calculation of $\bar{z}^k$ involves this error term, it only appears *implicitly*, and with the error condition, it ensures Fejér monotonicity.

2. Why is the parameter $\lambda_{k}$ considered within $(0,2)$? The analysis suggests that $\lambda_{k}$ primarily affects the step size $\lambda_k \alpha_k$, while the theoretical results indicate that $\lambda_k=1$ yields the best convergence. What are the benefits of considering $\lambda_k < 1$ or $\lambda_k > 1$?

3. Does $\alpha_k$ have an upper bound?

### Soundness
3

### Presentation
3

### Contribution
2
