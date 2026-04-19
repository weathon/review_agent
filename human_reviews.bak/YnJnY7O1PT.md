# Private Stochastic Convex Optimization with Tysbakov Noise Condition and Large Lipschitz Constant

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 6, 3

## Abstract
We study Stochastic Convex Optimization in Differential Privacy model (DP-SCO). Unlike previous studies, here we assume the population risk function satisfies
the Tysbakov Noise Condition (TNC) with some parameter $\theta>1$, where the Lipschitz constant of the loss could be extremely large or even unbounded, but the $\ell_2$-norm gradient of the loss has bounded $k$-th moment with $k\geq 2$. 
For the Lipschitz case with $\theta\geq 2$, we first propose an $(\epsilon, \delta)$-DP algorithms whose utility bound is $\tilde{O}\left(\left(\tilde{r}_{2k}(\frac{1}{\sqrt{n}}+(\frac{\sqrt{d}}{n\epsilon}))^\frac{k-1}{k}\right)^\frac{\theta}{\theta-1}\right)$
 
in high probability, where $n$ is the sample size, $d$ is the model dimension, and $\tilde{r}_{2k}$ is a term that only depends on the $2k$-th moment of the gradient. 

It is notable that such an upper bound is independent of the Lipschitz constant. We then extend to the case where 
 $\theta\geq \bar{\theta}> 1$ for some known constant $\bar{\theta}$. Moreover, when the privacy budget $\epsilon$ is small enough, we show an upper bound of $\tilde{O}\left(\left(\tilde{r}_{k}(\frac{1}{\sqrt{n}}+(\frac{\sqrt{d}}{n\epsilon}))^\frac{k-1}{k}\right)^\frac{\theta}{\theta-1}\right)$ 

even if the loss function is not Lipschitz. For the lower bound, we show that for any $\theta\geq 2$, the private minimax rate for $\rho$-zero Concentrated Differential Privacy is lower bounded by $\Omega\left(\left(\tilde{r}_{k}(\frac{1}{\sqrt{n}}+(\frac{\sqrt{d}}{n\sqrt{\rho}}))^\frac{k-1}{k}\right)^\frac{\theta}{\theta-1}\right)$.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work studies DP-SCO under Tysbakov Noise Condition (TNC, parameterized by $\theta>1$) where the population loss function's gradient is only assumed to have bounded $k>2$-th moment i.t.o. $\ell_2$-norm, which relaxes the common assumption that gradient is uniformly bounded by a Lipschitz constant.
The authors propose several localized noisy clipped gradient methods for different cases with varying $\theta$s. 
They show that when applied to convex and smooth functions, these algorithms can achieve faster high-probability convergence rates than that a convex function can achieve.
Moreover, a lower bound under zCDP is derived to confirm the near optimality of the proposed algorithms.
This work is generally interesting and expands our understanding of DP-SCO under TNC.
However, I believe more effort should be put into polishing the manuscript before it is ready for publication.

### Strengths
1. the authors considered a setting of TNC that expands our understanding of DP-SCO. The proposed algorithms and theoretical results are promising.
2. a matching minimax lower bound is derived by considering a hard loss function. This function might be of independent interest and may provide insights into the privacy-accuracy tradeoff under TNC settings.

### Weaknesses
1. The manuscript seems unfinished. The authors are highly encouraged to conclude the paper with a Conclusion section that summarizes your contributions.
2. There are many typos. Some theorems in the Appendix are not self-contained. This makes readers hard to follow, and they might not be able to fully appreciate the contributions. I will provide more details in Questions.

### Questions
1. Some undefined notations & typos:

	(1) Lemma 2 line 244, what is $r^{(k)}$? In the same line, $\tilde{e}_1^k$ should be $\tilde{e}_1^{(k)}$.

 	(2) algorithm 5 line 388, return $w_k$? should be $w_l$

 	(3) line 393, there are two 'algorithm 5'

 	(4) algorithm 7 line 475, what is AC-SA? this algorithm is not mentioned anywhere else.

 	(5) please be as precise as possible with the usage of per-sample loss $f$ and population loss $F$, see lines 807, 863, 873. 

2. Theorem 8, 9, 10 are not self-contained. Please properly define all variables used in Theorems.

	(1) In theorem 8, What are $w_T$ and $\hat{w}$? I presume this theorem is associated with algorithm 2, so $w_T$ is the output of algorithm 2. But $\hat{w}$ is still not well-defined. The definition of $b_t$ should appear earlier, as you need it for the eq in line 746.

	(2) In theorem 9, What are $w_l$ and $\hat{w}_l$? In algorithm 3, $w_l$ is used for the output, but in algorithm 4, this output is assigned to $\hat{w}_l$. In this sense, $w_l$ and $\hat{w}_l$ are the same. Please be clear on the meanings of notations used in Theorems. Additionally, please be accurate with the usage of per-sample loss $f$ and population loss $F$.

	(3) similar issues in theorem 10. Please define $w_l$ and $w^*$. Or at least associate it with a specific algorithm

3. Theorem 8 line 805, do you have the assumption $\eta\leq \frac{2}{\lambda}$? is this an implicit result of a properly chosen $p$? I am curious because in algorithm 3 step 6 when you invoke algorithm 2, the strong convexity $\lambda_i$ is parameterized by $p$, it is not immediate to see whether this assumption holds.

4. Equation in line 863 seems incorrect? Even if you replace $f$ with $F$ on the rhs, it seems the $w^*$ term is still missing on the rhs. Do I miss something?

5. A clarification question: I notice the lower bounds are proved for $(\theta, 1)$-TNC while the upper bounds are for $(\theta, \lambda)$-TNC. So there still exists a gap in terms of parameter $\lambda$? To be specific, it seems upper bounds and lower bounds differ in a term of $(\frac{1}{\lambda})^{\frac{1}{\theta-1}}$, if $\lambda\neq1$.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper explores DP-SCO with large Lipschitz constants under TNC conditions. Key findings include achieving theoretical bounds that are independent of the Lipschitz constant. The paper also provides lower bounds in some settings.This research contributes significantly to the domain of privacy-preserving machine learning, particularly in managing complex data distributions with rigorous privacy requirements.

### Strengths
The paper is well-written, with algorithms clearly delineated and accompanied by thorough commentary.

An upper bound that is independent of the Lipschitz constant and a lower bound were derived, demonstrating the algorithm's effectiveness and providing a valuable theoretical tool for further research.

A comparison with previous results on DP-SCO under varying assumptions is provided, positioning this work effectively within the related literature.

### Weaknesses
The TNC combined with Assumption 1 (bounded moments and diameter), while justified by related works, may limit the broader interest from the ICLR audience, potentially making it more suitable for a journal. Further explanation of related applications could help address this issue.

The novelty of the algorithm should be more distinctly explained to clarify how it differs from standard "sample, clip, SGD" methodologies.

The lower and upper bounds are not presented within the same framework, understandably due to the absence of amplification results under CDP.

### Questions
Could you provide some experimental results regarding the efficiency (speed and actual accuracy) of the proposed method?

The lower and upper bounds are not presented within the same framework. Could you use some conversion between DP, GDP, and CDP to allow for a comparative analysis to some extent?

### Soundness
3

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
3

### Summary
This paper studies the problem of Stochastic Convex Optimization under the lens of Differential Privacy. Previous papers usually assume the loss is Lipschitz so the sensitivity of the gradient is bounded, thus making it easier to apply Gaussian Mechanism to guarantee privacy. This paper relaxes this Lipschitz assumption and derives a high probability bound for the population risk.

### Strengths
- Designing differentially private algorithms without requiring Liptchitz loss is a pretty interesting problem. Though the gradient norm in practice tends to be bounded with high probability, it is usually not uniformly bounded. Thus, it is necessary to develop theories that can cover this realistic scenario.

- The algorithm has good theoretical guarantees. The algorithm basically recovers the optimal asymptotic rate of the previous works. The author also proves a lower bound to show that their guarantee is optimal up to some constants.

- The analysis is sound and correct

- The Tysbakov Noise condition is novel and hasn't been extensively studied in the literature.

### Weaknesses
- I think the paper would greatly benefit from some high-level discussion of the algorithm. The author doesn't have to go through every steps of the algorithm but they should highlight some key ideas of the algorithm (they did highlight a few like shuffling, and separating algorithm into phases but it is not clear how those technical ideas contribute to the final results).

- The author claims that their locality algorithm as being novel but as far as I know, this is a pretty common method for Stochastic optimization for strongly convex losses (in their cases, losses that satisfy the Tysbakov Noise Condition). See [1]. Obviously, the actual parameters used and the partitions are different so I would recommend the authors focus on that.

- The paper needs a bit of discussion on the TNC. Seems like it's some generalization of strong convexity but the paper does not provide strong arguments on why we should care about this settings. Some more discussion on this would be greatly appreciated.

- I'm not too convinced that the new constant (that basically replaces the Lipchitz constant in the final bound) is that much smaller than the constant from doing clipped SGD.

- To improve the bound in section 6, the authors use privacy amplification by sampling which results in $\epsilon$ being tiny just for this algorithm to work. This is not exactly a huge surprise since privacy amplification by sampling is more suited for the settings where we take multiple passes over the dataset.

[1] Feldman, Vitaly, Tomer Koren, and Kunal Talwar. "Private stochastic convex optimization: optimal rates in linear time." Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing. 2020.

### Questions
- Can the author compare the results of this paper to the results of [1]. [1] also does not require Lipschiz loss. I think the main reason why the proposed algorithm is able to handle non-Lipschitz loss is basically through clipping. 

- Why do we need to switch to $\rho-$zCDP for lower bounds?

- What's the role of the constant $p$?

[1] Chen, Xiangyi, Steven Z. Wu, and Mingyi Hong. "Understanding gradient clipping in private sgd: A geometric perspective." Advances in Neural Information Processing Systems 33 (2020): 13773-13782.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Stochastic optimization has been studied under the privacy constraint by a large body of recent work. Many of the works assume that the loss functions are lipschitz, and some recent works have studied this problem under the weaker assumption of bounded moments of gradient norms. This problem has also been studied under both the convexity assumption as well as a strong convexity assumption.

The current work extends this to losses satisfying a different growth condition known as the Tsybakov noise condition. Intuitively, it says that the function grows at least as fast as a \theta power of the distance from the optimum.  \theta = 1 gives convexity and \theta=2 gives strong convexity. This work considers a range of theta and proposes algorithms which lead to rates that interpolate between these two, and extrapolate beyond theta=2 as well.

I think the problem is interesting. This work shows the first results under this noise condition, and shows that the results are tight.

Unfortunately, I have some concerns about the correctness of the results. Theorem 1 in the paper relies on stability of clipped SGD, which will need a new argument. The authors seem to build on a result about stability of SGD, but clipping of gradients as done in their algorithm may break this stability.

### Strengths
- Combines Tsybakov assumption and gradient moment assumptions, and shows new upper and lower bounds for private optimization under these assumptions.

### Weaknesses
Weaknesses:
- It is unclear to me why Theorem 1 is true. The proof of privacy in the appendix makes claims about stability. However, this will need a proof: while gradient descent steps are contractive (or strongly contractive in the case of strongly convex losses), the steps taken in Alg 2 are not gradient steps but rather clipped gradient steps. Clipped gradients do not in general correspond to gradients of a convex function, and so it is not clear to my why this argument works.  In fact it seems to me that there is counter-example showing that for strong convexity, strong contractivity does not hold even in one dimension: if f(x)=x^2/2, and you clip gradient norm at 1, then a gradient step at y or y+1 for y> 1 will move to (y-\eta) and (y+1-\eta) respectively. Thus the desired strict contractivity does not seem to hold. Please also see “Characterizing Private Clipped Gradient Descent on convex generalized linear problems” by Song, Thakkar and Thakurta, which shows that convexity can break down.
- Relatedly, If I understand correctly, Thm. 3.9 in HRS has an L^2 in the numerator that seems to be missing in your restatement (Lemma 3). Perhaps there is some normalization difference that I am missing?
- Tsybakov is misspelled in the title and abstract.
- Private Iterative localization based on stability was previously used in [14]. Could you say more on how your use is different from their?
- second line in table seems to be under the same conditions as the first. Is that supposed to be for the strongly convex case? Is there a gap between upper and lower bounds for that case?

Nits:
- Many of the citations of fairly old published work is to the arxiv version. While it may be useful to point to an arxiv version, I would recommend citing the published version.
- Table text is much too small to read.

### Questions
See weaknesses above. In particular, I would be curious if you can give a full proof of privacy in Thm. 1.

### Soundness
1

### Presentation
2

### Contribution
1
