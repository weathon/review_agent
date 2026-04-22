# Near-Optimal Convergence of Accelerated Gradient Methods under Generalized and $(L_0, L_1)$-Smoothness

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4, 4

## Abstract
We study first‐order methods for convex optimization problems with functions $f$ satisfying the recently proposed $\ell$-smoothness condition $\|\|\nabla^{2}f(x)\|\| \le \ell\left(\|\|\nabla f(x)\|\|\right),$ which generalizes the $L$-smoothness and $(L_{0},L_{1})$-smoothness. While accelerated gradient descent (AGD) is known to reach the optimal complexity $\mathcal{O}(\sqrt{L} R / \sqrt{\varepsilon})$ under $L$-smoothness, where $\varepsilon$ is an error tolerance and $R$ is the distance between a starting and an optimal point, existing extensions to $\ell$-smoothness either incur extra dependence on the initial gradient, suffer exponential factors in $L_{1} R$, or require costly auxiliary sub-routines, leaving open whether an AGD‐type $\mathcal{O}(\sqrt{\ell(0)} R / \sqrt{\varepsilon})$ rate is possible for small-$\varepsilon$, even in the $(L_{0},L_{1})$-smoothness case. We resolve this open question. Developing new proof techniques, we achieve $\mathcal{O}(\sqrt{\ell(0)} R / \sqrt{\varepsilon})$ oracle complexity for small-$\varepsilon$ and virtually any $\ell$. For instance, for $(L_{0},L_{1})$-smoothness, our bound $\mathcal{O}(\sqrt{L_0} R / \sqrt{\varepsilon})$ is provably optimal in the small-$\varepsilon$ regime and removes all non-constant multiplicative factors present in prior accelerated algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a new accelerated gradient descent method for convex optimization under a generalized $\ell$-smoothness assumption, establishing an AGD-type complexity that improves upon existing convergence guarantees.

### Strengths
1. **Originality:** Building on a previously proposed assumption and existing methods, the paper establishes an improved complexity bound.

2. **Quality:** While we have concerns about the writing clarity (discussed in *Weaknesses*), the overall structure is sound and the presentation is generally clear.

3. **Clarity:** The paper clearly explains its motivation and main ideas.

4. **Significance:** The new result may advance $\ell$-smooth optimization theory. However, given the limited empirical advantage over GD-based methods, it remains unclear whether the method achieves practical acceleration.

### Weaknesses
1. The proof technique is not novel. The proposed method is a discretization of a rotated heavy-ball flow introduced in [1]. With a simple change of variables, the Lyapunov function coincides with those in [2], [3], or [4]. Modern continuous-flow–based proofs, such as [5] and [6], could simplify the argument substantially and shorten Appendix B.

2. Lines 272–274 state: “While for $L$–smooth functions the proof technique from (Wei & Chen, 2025) does not offer any advantages over, for example, (Nesterov, 1983) because the result in (Nesterov, 1983) is optimal.” However, (Nesterov, 1983) is not optimal in the information-theoretic sense. There is extensive work on optimal gradient methods (OGM); see, for example, [7].

3. The numerical experiments in Appendix A do not demonstrate the efficiency of the proposed method. The AGD curve shows no improvement over GD, contradicting the theoretical claims. In addition, AGD exhibits severe oscillations, suggesting that it may not achieve genuine acceleration—or even reliable convergence.

**References**

[1] Wei, J., & Chen, L. (2024). Accelerated Over-Relaxation Heavy-Ball Method: Achieving Global Accelerated Convergence with Broad Generalization. *ICLR*, 2025.

[2] Alvarez, F., & Attouch, H. (2001). An inertial proximal method for maximal monotone operators via discretization of a nonlinear oscillator with damping. Set-Valued Analysis, 9(1–2), 3–11. https://doi.org/10.1023/A:1011203001547

[3] Su, W., Boyd, S., & Candès, E. J. (2016). A differential equation for modeling Nesterov’s accelerated gradient method: Theory and insights. *Journal of Machine Learning Research*, 17(153), 1–43. http://jmlr.org/papers/v17/15-084.html

[4] Wibisono, Wilson, & Jordan (2016). A variational perspective on accelerated methods in optimization. *Proceedings of the National Academy of Sciences*, 113(47), E7351–E7358. https://doi.org/10.1073/pnas.1614734113

[5] Chen, L., & Luo, H. (2019). First-order optimization methods based on Hessian-driven Nesterov accelerated gradient flow. arXiv preprint arXiv:1912.09276.

[6] Luo, H., & Chen, L. (2022). From differential equation solvers to accelerated first-order methods for convex optimization. *Mathematical Programming*, 195(1), 735-781. https://doi.org/10.1007/s10107-021-01713-3

[7] Kim, D., Fessler, J.A. (2015). Optimized first-order methods for smooth convex minimization. *Mathematical Programming*. https://10.1007/s10107-015-0949-3

### Questions
1. Beyond the generalized smoothness assumption, is there an analogous assumption that extends strong convexity? Under such a condition, can the method achieve linear convergence?

2. What convergence guarantees can be established when applying gradient methods to $\ell$-smooth functions without assuming convexity?

### Soundness
3

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
3

### Summary
The paper propose first-order methods for convex optimization problems with $\ell$-smoothness condition: $\|\nabla ^2 f(x)\| \leq \ell(\|\nabla f(x)\|)$, where $\ell$ is a non-decreasing, positive, locally Lipschitz function. When $\psi(x) = \frac{x^2}{2\ell(4x)}$ is strictly increasing, the proposed algorithms achieve the oracle complexity of $O(\sqrt{\ell(0)}R/ \sqrt{\varepsilon})$ for $R = \|x_0 - x^*\|$ and small $\varepsilon$. In particualr, with $(L_0, L_1)$-smoothness, the oracle complexity is $O(\sqrt{L_0}R/{\sqrt{\varepsilon}})$.

### Strengths
-  The proposed algorithms improve the the oracle complexity of accerated gradient methods on a class of convex optimization problems over $\ell$-smoothness condition. 

- The proofs of the main theorems are sound and well-discussed.

### Weaknesses
- The assumption that $\psi(x)$ is strictly increasing restricts the results mainly to $(L_0, L_1)$-smoothness.


- The algorithms requires sophiticated choices on parameters which are usually unknown or difficult to estimate; while the optimal convegence region relies on these parameters.

### Questions
- What is the motivation of considering the function $\psi(x)$? 

- Beyond $(L_0, L_1)$-smoothness, what are the functions $\ell$ such that the assumption $\psi(x)$ is strictly increasing holds?


- For Algorithm 1, what is the convegence guarantee after GD and before $\bar k$ itertations?


- There is a non-accelerated phase in Algorithm 1 and results in addtional constant factors in the oracle complexity. The acclerated region relies on $\delta$ and $\bar R$.  Should this be viewed as local accelerated convergence (requiring initial guess close to the solution)?

- The paper claims the convergence rate is a significant improvement over previous works. It would be nice to have some numerical experiments compared with other AGD methods to validate the performance of the proposed algorithms.

- Is the set $Q$ defined in line 441 non-empty?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes accelerated gradient methods under $(L_0, L_1)$-smoothness condition. It presents two algorithms, one with a brief GD warm start and another that adapts step sizes without a warm start, and establish the best-known oracle complexity $\sqrt{l(0)}R/\sqrt{\epsilon}$ in the small $\varepsilon$ regime. The analysis also extends to the generalized $(L_0, L_1)$-smoothness setting.

### Strengths
They proposed algorithm which established the best-known oracle complexity in the small $\epsilon$ regime with tailored Lyapunov function. The results align with optimal complexity under $l$-smoothness condition and are empirically validated on a toy problem. The proof sketch is properly presented and Table 1 clearly exhibits the contribution.

### Weaknesses
To be honest, I’m uncertain that the paper’s contribution meets the bar for acceptance at this venue. While the paper establishes a best-known bound, the guarantee is confined to the small $\epsilon$ regime, and the constant factor improvement over Li et al. (2024a) seems somewhat incremental.

### Questions
Is a lower-bound result established under this $(L_0, L_1)$-smoothness condition?

What are the technical challenges in extending the analysis to arbitrary $\epsilon$?

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
This paper considers problems that satisfy $\ell$-smoothness conditions, which genearlize both standard and $(L_0, L_1)$-smoothness. For this setting, they achieve a $O(\sqrt{\ell(0)}R/\sqrt{\varepsilon}) + L_1^2R^2$ first-order oracle complexity to find an $\epsilon$-approximate solution.

### Strengths
The primary strength of the paper is that it improves the previous $\ell(||\nabla f(x^0)||)$ dependence to $\ell(0)$ (up to considerations of additive terms, discussed below in "Weaknesses"), and this helps better place the result in the context of classic lower bound in smooth convex optimization.

### Weaknesses
One issue is that the algorithm needs $\Gamma_0$, $\bar{R}$. Do the other algorithms in Table 1 require these? If not, then the results are not directly comparable, and it would then be important to explain these caveats as an additional part of the table. The authors claim (erroneously) the complexity is optimal (line 250). The authors should specify the range of $\varepsilon$ where they claim optimality, and should emphasize the additive term which prevents them from actually being optimal. The additive term is $L\_1^2R^2$, which can dominate for some ranges of $\varepsilon$, $L\_1$, $ R $, $L\_0$. This needs clarifying for the exact range of improvements, and to point out when previous works dominate this work, for proper comparison.

Following this, the work could benefit from providing a clearer description of why this result is important in the face of previous work, since the overall improvement seems quite slight in that it only affects the smoothness parameters (and furthermore at the cost of a potentially worse additive term in some cases), and the techniques resemble those in Vankov et al. Because of these concerns, there is some hesitance felt about whether these results are significant enough to warrant acceptance, especially in view of the caveats above yet to be completely addressed.

### Questions
What precisely is $\nu$ in Table 1? (Can its dependence on parameters of $f$ be elaborated on?)

### Soundness
2

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
4

### Summary
This paper proposes two new Accelerated GD variants designed to optimize $\ell$-smooth functions (the generalized version of $L$-smoothness and $(L_0, L_1)$-smoothness). The authors provide convergence guarantees for their proposed methods, showing that in the small accuracy regime their methods attain near-optimal convergence of $\mathcal{O}(\sqrt{\ell(0)}R/\sqrt{\varepsilon})$, where $R =\\|x_0 - x^{\star}\\|$, thus imporving upon prior approaches. Between the two of them, these algorithms cover both the case of sub- and superquadratic $\ell$. Preliminary experiments are provided.

### Strengths
The topic is relevant to the research community, and the paper is well-structured and well-written. To the best of my knowledge, the related literature is appropriately covered, and significant parts of the technical approach are novel. The contribution is significant for both theory and practice, since it helps delineate the reach/limitations of classical methods under generalized smoothness, and can provide practitioners in, e.g., scientific computing fields, with potentially improved tools.

### Weaknesses
1. **Technical approach**
	* Assumption 2.3 states that "$ f : \mathbb{R}^d \to \mathbb{R} \cup \{\infty\} $ [...] attains its minimum at a (non-unique)
$x^\ast \in \mathbb{R}^d $ [...]". However, none of the motivating examples in line 051 do satisfy this over $\mathbb{R}^d$ (for the case of $x^p$, consider $p$-odd).
	* The method addresses contrained optimization, yet the proof of Lemma B.3 uses the result of Lemma B.1 by replacing $\nabla f(y)$ with $\nabla f (x^\star)$ which is set to zero. The constrained optimum $x^\star$ does not necessarily satisfy $\nabla f (x^\star) = 0$, so the result is problematic. Could you please address this and the possible ramifications of it?

2. **Presentation**
	* A comparison between the stepsize's dependence on problem constants of Alg2 vs. [1,2,3] is missing, and would help in understanding the tradeoffs between this and previous methods.

3. **Experiments**
	* The experiments are too simplistic, and only compare Algorithm 2 with Tyurin's GD variant (unaccelerated). The method should be compared with the Accelerated versions of [1, 2, 3]. For a fair comparison where auxiliary subroutines are concerned, convergence in terms of wall clock time should be considered. This is useful for understanding the practical behaviours of these methods relative to each other, since it is likely that despite the worse convergence upper bounds, the prior algorithms are still competitive in practice in the small $\varepsilon$ regime.
	* Experiments with various degrees of overestimation for $\bar{R}$ and $\Gamma_0$ should be conducted to understand Alg. 2's sensitivity to hyperparameter tuning (even if the $\varepsilon$-dependent term only depends on $R$, and not $\bar{R}$).
	* Less pressing: ideally, an experiment should be included on a practically-relevant (empirically determined) $(L_0, L_1)$-loss, in order to assess the method's sensitivity to tuning in practice
	

[1] Haochuan Li, Jian Qian, Yi Tian, Alexander Rakhlin, and Ali Jadbabaie. Convex and non-convex optimization under generalized smoothness. Advances in Neural Information Processing Systems, 36, 2024a.

[2] Eduard Gorbunov, Nazarii Tupitsa, Sayantan Choudhury, Alen Aliev, Peter Richt´arik, Samuel Horv´ath, and Martin Tak´aˇc. Methods for convex (L0,L1)-smooth optimization: Clipping, acceleration, and adaptivity. In International Conference on Learning Representations, 2025.

[3] Daniil Vankov, Anton Rodomanov, Angelia Nedich, Lalitha Sankar, and Sebastian U Stich. Optimizing (L0,L1)-smooth functions by gradient methods. arXiv preprint arXiv:2410.10800, 2024.

### Questions
Please see comments above.

### Soundness
2

### Presentation
3

### Contribution
3
