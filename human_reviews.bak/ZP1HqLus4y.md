# CONTINUAL FINITE-SUM MINIMIZATION UNDER THE POLYAK-ŁOJASIEWICZ CONDITION

- Decision: Reject
- Scores: 3, 3, 3

## Abstract
Given functions $f_1,\ldots,f_n$ where $f_i:\mathcal{D}\mapsto \mathbb{R}$, \textit{continual finite-sum minimization} (CFSM) asks for an $\epsilon$-optimal sequence $\hat{x}_1,\ldots,\hat{x}_n  \in \mathcal{D}$ such that

$$\sum_{j=1}^i f_j(x_i)/i - \min_{x \in \mathcal{D}}\sum_{j=1}^if_j(x)/i \leq \epsilon$$

 In this work, we develop a new CFSM framework under the Polyak-Łojasiewicz condition (PL), where each prefix-sum function $\sum_{j=1}^i f_j(x)/i$ satisfies the PL condition, extending the recent result on CFSM with strongly convex functions.  We present a new first-order method that under the PL condition producing an $\epsilon$-optimal sequence with overall $\mathcal{O}(n/\sqrt{\epsilon})$ first-order oracles (FOs), where an FO corresponds to the computation of a single gradient $\nabla f_j(x)$ at a given $x \in \mathcal{D}$ for some $j \in [n]$. Our method also improves upon the  $\mathcal{O}(n^2 \log (1/\epsilon))$ FO complexity of state-of-the art variance reduction methods as well as upon the $\mathcal{O}(n/\epsilon)$ FO complexity of $\mathrm{StochasticGradientDescent}$. We experimentally evaluate our method in continual learning and the unlearning settings, demonstrating the potential of the CFSM framework in non-convex, deep learning problems.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors consider a continual version of finite-sum optimization (CFSM), where the loss functions arrive sequentially, and the ERM updates by including a newly received loss function in each iteration. This work extends the CFSM to the non-convex scenario by proposing a novel algorithm called CSVRG-PL. Specifically, when the loss functions in the sequence satisfy the PL property and a rather novel diminishing return assumption on the optimal points' distances, the authors demonstrate that CSVRG-PL achieves an improved rate of $O(n/\epsilon^{1/2})$, outperforming existing algorithms under the same setting.

### Strengths
Overall, the paper is well-structured and easy to follow. The contributions are clear, and the results offer an improvement over previous work under the CFSM framework and the PL condition. The comparison with existing methods is complete and allows readers to understand the background without effort.

### Weaknesses
However, a primary concern is the validity of the diminishing return assumption over distances between optimal points. This assumption is crucial to the improvement of the rate and appears unavoidable within the proof. Nevertheless, it is not justified in the previous literature including [Mavrothalassitis et al.,  2024], and may be too assertive. Specifically, in the non-convex setting, the optimal points of different loss functions could be arbitrarily distant from each other, making this diminishing return assumption quite unpractical. Besides, it seems that the authors want to defend this assumption by referencing the uniform stability condition in generalization theory. Indeed, uniform stability is often used as a powerful strategy for generalization error. However, it is typically a quantity that should be bounded by controlling the incremental error of gradient steps, rather than being assumed directly. As a result, this argument seems to be not supportive. It would be nice if the authors could provide more convincing arguments for this assumption.

### Questions
* It would be great if the authors could do more proofreading to promote the consistency of notations and referencing style. 
* The discussion on the PL condition may be unnecessary, as it is a fairly standard assumption in non-convex optimization.
* In line 128, $O(n/\epsilon^{1/4})$ should be $\Omega(n/\epsilon^{1/4})$ because it is a lower bound result.

### Soundness
3

### Presentation
2

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
This paper studies continual finite-sum minimization (CFSM) problems, introducing a novel first-order method, CSVRG-PL, and establishing its computational complexity under the PL condition. Numerical experiments are conducted to demonstrate the superiority of CSVRG-PL over SGD.

### Strengths
The paper is well-organized and easy to follow, with new theoretical contributions to the literature.

### Weaknesses
1. **Similarity to Prior Work**: The proposed algorithm is very similar to the method in [1], as both are inspired by the well-known SVRG method and follow the same algorithmic structure. As the authors point out, the primary difference lies in line 5 of Algorithm 2, with the rest of the method remaining nearly identical. This minor modification limits the novelty of the proposed approach.

2. **Limited Theoretical Contributions**: Although the paper presents three theorems, they essentially convey a single result—an upper bound for FO complexity under the PL condition. In contrast, [1] provides both upper and lower bounds, though it focuses only on the strongly convex setting and there remains a gap between the bounds. To enrich the content, it would be beneficial to explore a more general setting (e.g., the convex case without strong convexity or the non-convex case without the PL condition). Additionally, the authors could attempt to derive an improved lower bound or examine whether the gap between upper bounds in the strongly convex and PL settings is intrinsic to the problem structure. This would deepen the theoretical contribution of the paper.

[1] Ioannis Mavrothalassitis, Stratis Skoulakis, Leello Tadesse Dadi, and Volkan Cevher. Efficient continual finite-sum minimization. In The Twelfth International Conference on Learning Representations, 2024.

### Questions
1. In Corollary 1, the number of FO calls for lines 6 and 15 in Algorithm 1 is upper bounded by $n$, though the actual count is $i$. Could a finer analysis improve the result in Corollary 1?
2. In line 529, the authors mention achieving $O(n/\epsilon^{1/3})$ complexity for the strongly convex case. Could the authors clarify where this result appears?

[Typos]
Line 240: Remove the repeated "that."
Line 532: The complexity for variance reduction methods should be $O(n^2 \log (1/\epsilon))$.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper focuses on Continual Finite-Sum Minimization (CFSM) and proposes and analyzes a new algorithm, CSVRG-PL, for solving the problem when the functions satisfy the POLYAK-LOJASIEWICZ (a generalization of strong convexity that captures some structured non-convex problems as well). The paper includes experiments on deep learning problems.

### Strengths
The paper is well written, and under the strong assumption made in this work (see weaknesses below), the theoretical result presented in the paper looks correct. The experiments evaluate the proposed method CSVRG-PL in continual learning settings, which is not related to the assumptions made in the paper, but show the practicality of the proposed method.

### Weaknesses
### On restrictive setting:
The paper heavily relies on the paper of Mavrothalassitis et al., the first to propose algorithms in the Continual Finite-Sum Minimization for the strongly convex regime. The main contribution is the extensions of these ideas in the PL setting. 

Extending strongly convex results to PL functions is a reasonable extension, but I do not consider this by itself to be enough of a contribution to justify a paper acceptance at NeurIPS. For this paper to make more sense, it would have been much more exciting and appealing if the theory was extended to convex problems or general non-convex (under potential growth conditions). From an optimization viewpoint, simply extending a strongly convex theory to PL is, most of the time, an easy task. 

Despite the above comment, I appreciate the clearness in explaining the differences between the CFSM-PL and the work of Mavrothalassitis et al., as presented in point 3 of the main contributions, and I fully agree with this.
I understand the challenge of extensions to PL scenarios, but in my opinion, by comparing the two papers (the submitted one and Mavrothalassitis et al.), I do not see a difficulty in putting things together as this paper did.

### Issues with Theoretical assumptions

Let me highlight three two major flaws of the analysis:

1.  Assumption 1 is extremely strong, and even if it is used in previous/older papers, it does not mean that it should continue being used. I believe having it restricts the theorems substantially, and most likely, it does not hold for smooth PL functions in unconstrained setting (the D in the assumption can be extremely large for this to make sense in several practical examples, which will make the bounds on the theorem impractical)

2. Same holds for Assumption 2. In many practical scenarios without extra assumptions on interpolation (most likely the ones of the experiments in the paper as well), the K is extremely large and cannot really be quantified. The K appears in the upper bound of the theorems, which allows potentially very loose final results.

### On presentation and parts of the main paper

The whole paper has one single theorem (Theorem 2). I appreciate the detail on the theoretical justification of the paper, and I like the presentation of it, but in my opinion, all of this discussion can easily move to the appendix. I see a clear benefit in including all the lemmas in the main paper (considering the work of Mavrothalassitis et al., these lemmas are not difficult to prove to be worth part of the main paper). In addition, I am not sure why Theorems 1 and 3 are called Theorems (these are just a proposition and a corollary, respectively).
\end{itemize}

The experiments are not related to what the theory suggests. The papers directly solve DNNs, but this is not the theoretical focus of this work (PL functions). Not all ML papers should be about DNNs. Do the authors have any setting that their assumptions are actually satisfied and which is not toy example and not strongly convex?

### Questions
Please see my points and questions in the Weaknesses box

### Soundness
2

### Presentation
3

### Contribution
2
