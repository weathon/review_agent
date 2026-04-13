## Human Reviewer 1

### Summary
The paper studies the convergence of SAM and USAM in stochastic settings. It proves these properties for a newly proposed algorithm, Unified-SAM, which includes SAM and USAM as special cases. The analysis relaxes popular assumptions like bounded variance (BV) and gradients (BG), replacing them with expected residual (ER) condition. The proof provides convergence guarantees for SAM under different step sizes in non-convex functions and Polyak-Lojasiewicz (PL) functions. The theory holds under arbitrary sampling paradigms, including importance sampling. The authors also demonstrate Unified-SAM's performance compared to SAM in practical settings.

### Strengths
- The paper is well-written and easy to follow.
- The authors challenge existing assumptions on stochastic noise, such as bounded gradients and bounded variance, in proving SAM's convergence, replacing them with an expected residual condition that encompasses both as special cases.
- The proof is a slight improvement over existing studies.

### Weaknesses
- The empirical results from the original SAM paper Foret et al. [2020] are established using a constant $\rho$. This is a crucial point for aligning theoretical and empirical results. To my knowledge, there are no existing works that establish convergence results on the constant $\rho$ for non-convex functions. However, there are some theoretical papers that use conditions closely related to the constant $\rho$, but these have not been discussed in the paper under review, for example,  Nam et al. [2023], Khanh et al. [2024], Xie et al. [2024]. The assumption in this paper regarding $\rho$ is $\rho = \min ( \frac{1}{2t+1}, \rho^{\star} )$, which is less general than the assumption in Nam et al. [2023], where $\rho$ is defined as $\sum^{\infty}_{t=0} \rho_t^2 < \infty$. Additionally, the assumption on $\rho$ in Khanh et al. [2024] for the full-batch setting is even more general, as it allows the perturbation radius $\rho$ to decrease at arbitrary slow rates, which nearly captures the constant $\rho$.
- In the asymptotic setting, this paper's result in Theorem 3.7, $\min_{t=0,...,T-1} \mathbb{E} || \nabla f(x^t) || \leq \epsilon$, is weaker than the convergence result in the stochastic, non-convex setting in Nam et al. [2023], where the gradient norm approaches zero almost surely. Furthermore, if Theorem 3.7 is considered in the full-batch setting, it is also weaker than the result in Khanh et al. [2024.
- In Theorem 3.5, why do you write $O\left( \frac{1}{t} + \frac{1}{t^2} \right)$ instead of $O\left( \frac{1}{t} \right)$, since they are equivalent?
- Compare your Assumption 3.1 (Expected Residual Condition) with Assumption A.4 in Nam et al. [2023].
- The name of the algorithm, Unified-SAM, may not be suitable, as it only covers USAM, SAM, and the variant that transitions between SAM and USAM. There are many other SAM-like variants that this algorithm does not cover.
- As shown in Tables 2 and 3, the proposed Unified-SAM does not show significant improvement over SAM. This is a point that diminishes the importance and contribution of the paper.
- Line 152-153: “This is the first result that drops the bounded variance assumption for both USAM and SAM." I suggest rewriting this sentence to clarify that, while the bounded variance assumption is removed, an Expected Residual condition is still required.

References:
- Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization
for efficiently improving generalization. ICLR 2021. URL https://arxiv.org/pdf/2010.01412.
- Pham Duy Khanh, Hoang-Chau Luong, Boris S Mordukhovich, and Dat Ba Tran. Fundamental convergence analysis of sharpness-aware minimization. NeurIPS, 2024. URL https://arxiv.org/pdf/2401.08060.
- Kyunghun Nam, Jinseok Chung, and Namhoon Lee. Almost sure last iterate convergence of sharpness-aware minimization. Tiny Papers ICLR, 2023. URL https://openreview.net/forum?id=IcDTYTI0Nx.
- Wanyun Xie, Thomas Pethick, and Volkan Cevher. Sampa: Sharpness-aware minimization parallelized. NeurIPS, 2024. URL https://arxiv.org/pdf/2410.10683v1.

### Questions
See the Weakness part.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper studies the convergence of sharpness-aware minimization (SAM) for smooth functions. The authors proposed a new unified notion of normalized SAM and unnormalized SAM by linearly interpolating the perturbed point used to take the gradient for the next update. The main contribution of the paper is to establish convergence rates to first-order stationary points for unified SAM under a more general noise assumption called "Expected Residual (ER) Condition" and arbitrary sampling methods. For non-convex but PL functions, the authors prove the loss value converges in $O(1/\epsilon)$ steps.

### Strengths
This paper is overall well written (except some typos) and the presentation of the main result is good and easy to understand. The theoretical results for convergence of SAM under the Expected Residual (ER) Condition is new. The authors also provide numerical evaluation for the new algorithm proposed in this paper, that is, unified SAM with $\lambda_t = 1-1/t$.

### Weaknesses
I have concerns both about the correctness and significance about the main result in this paper. Given these concerns, I do not think the current draft meets the bar of ICLR.

**Correctness**: 
- In Theorem 3.7, the authors wrote "choose $\rho\le\bar \rho$ and $\gamma\le \bar\gamma$...", the gradient norm will be smaller than $\epsilon$ in at most $1/\epsilon^4$ setps. It is not clear if the authors mean there exists $\rho\le\bar \rho$ and $\gamma\le \bar\gamma$, or for any $\rho\le\bar \rho$ and $\gamma\le \bar\gamma$. The former interpretation make the result trivial because the optimal choice is $\rho=0$ and unified SAM becomes SGD. Howver, i think the later interpretation makes the current results wrong. As a quick sanity check, learning rate $\gamma=0$ means the gradient should not change at all. (see elaboration in the next point below)

- In the formal statement of Theorem 3.7, which is Theorem C.4, the authors indeed interpret the condition as "for any $\rho\le\bar \rho$ and $\gamma\le \bar\gamma$" (line 1100). However, this does not make sense because it does not exclude the case of $\gamma=0$. A direct cause of this could be that the authors forgot to include Equation (13) into the restrictions that they need to satisfy, which says $T\ge \frac{12\delta_0}{\gamma \epsilon^2}$.

- However, I do not think the above issue can be fixed, unless $\rho(1-\lambda)=0$. The authors tried to replicate the analysis by Khaled \&
Richtarik (2020) for SGD to unified SAM, including how to deal with the seemingly exponential explosion. However, in the case of unified SAM, the term before $[f(x^))-f^{inf}]$ in line 1067 is essentially $\frac{(1+\Theta(\gamma))^T}{T\gamma}$ when $\rho(1-\lambda)\neq 0$.  Because $\frac{(1+\Theta(\gamma))^T}{T\gamma} \ge \frac{(1+\Theta(T\gamma)}{T\gamma} = \Theta(1)$, the right-hand side of line 1066 is at least a constant.

**Significance of Theory**: This paper only talks about optimization of SAM, but indeed, SAM is proposed to improve the generalization of SGD. It is not clear to me what the ultimate goal here by studying these convergence bounds of unified SAM. Both from a intuitive sense or from the bounds presented in this paper, SGD (or unified SAM with $\rho=0$) has the best optimization performance. To me the real problem for SAM is how to show they can efficiently minimize the sharpness-aware loss (either the origial notion of maximal loss after certain perturbation, or the hessian-regularized version proposed in Wen et al. (2023) and Bartlett et al. (2023)), rather than viewing them as a tool to minimize the original training loss and analyze that behavior.

**Significance of Experiments**: The performance of the proposed $1-1/t$ schedule for unified SAM does not beat normalized SAM by a margin which is larger than the standard deviation in most experiments. Sometimes unified SAM even has better generalization. I understand this does not contradict with theory because theory does not try to say anything about generalization. However, it is not clear to me if the main benefit of unified SAM is better optimization or generalization. 

**References**:

- Wen, Kaiyue, Tengyu Ma, and Zhiyuan Li. "How Sharpness-Aware Minimization Minimizes Sharpness?." The Eleventh International Conference on Learning Representations. 2023.
- Bartlett, Peter L., Philip M. Long, and Olivier Bousquet. "The dynamics of sharpness-aware minimization: Bouncing across ravines and drifting towards wide minima." Journal of Machine Learning Research 24.316 (2023): 1-36.


# Minor comments:
1. line 271. "our results is the first to demonsteate linear convergence in the fully stochastic regime". I would suggest that the authors do not call this linear converge to avoid confusion, when the final bound for loss is still at least constantly large.
2. line 334, "The results in Proposition 3.6 and Theorem 3.7 are tight, as setting ρ= 0 Unified SAM reduces in SGD and these simplify to the step sizes and rates (up to constants) of Theorem 2 and Corollary 1 from Khaled & Richtarik (2020)." The whole point of the analysis is for the regime $\rho\neq 0$. The fact that the analysis in this paper recovers previous result when $\rho=0$ does not indicate the tightness of the result in the main setting when $\rho\neq 0$, especially the dependence on $\rho$.

# Typos:
1. line 224, missing l2 norm on $g(x)$
2. section B in appendix, line 866-890. Missing norm and square over the norm.
3. line 1121. "From Theorem 3.7" should be from "Proposition C.3"

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper extends and combines previous analyses on the convergence of SAM and unnormalized SAM (U-SAM) by considering a generalized update rule called Unified SAM. Under the Expected Residual Condition, they prove convergence for Unified SAM for loss functions satisfying PL conditions and generalized non-convex loss functions. The convergence bound holds for a wide range of sampling strategies. Intriguingly, they show that importance sampling can minimize this convergence bound. Empirically, they show that Unified SAM matches and sometimes outperforms the original SAM.

### Strengths
[Quality] This paper is well-written and contains solid theoretical and empirical results.
[Clarity] This paper carefully discusses previous convergence bounds on SAM and U-SAM and shows clearly how the current convergence results generalize previous ones.

### Weaknesses
The connections and distinctions between this work and previous analyses of SGD require clearer articulation. The paper’s central assumption, the Expected Residual Condition, is adopted from [1] and is also employed in [2,3]. In the limiting case where $\rho = 0$, the results presented here converge to those found in the SGD analyses in [2,3]. While the authors make this reduction explicit, the paper does not address how the implications of the current results and proof techniques differ from those of earlier studies.
For instance, the authors suggest that this work "provides a theoretical justification for applying importance sampling in SAM." However, this argument is based on the quantity $\max_i L_i/p_i$ within the bound, which also appears in previous analyses, such as [1]. Thus, this justification extends to SGD as well and is not specifically related to SAM, a point that should be conveyed more directly.
The connection between the current theoretical and empirical results could be further strengthened. The computer vision experiments in this paper demonstrate that Unified SAM can achieve improved validation performance. However, since the primary focus of this paper is on the convergence properties of these methods, training loss would serve as a more relevant metric for linking the theory with empirical findings. This metric, however, is not included in the paper or its appendix.
The presentation can be improved.
The current paper contains typos that may obscure understanding. For example, the $g(x)$ in equation (ER) in Assumption 3.1 should be the norm of $g(x)$. Further, given the complexity of the current theoretical bounds, an intuitive interpretation of the current bounds can improve the paper.

[1] SGD: General Analysis and Improved Rates. arxiv.org/abs/1901.09401
[2] SGD for Structured Nonconvex Functions: Learning Rates, Minibatching and Interpolation arxiv.org/abs/2006.10311 
[3] Better Theory for SGD in the Nonconvex World. arxiv.org/abs/2002.03329

### Questions
Questions
[Relationship of convergence speed and $\lambda$] In the logistic regression experiments, the convergence is better with larger $\lambda$, which is coherent with the theory as here $C = 0$. Is there a case that the bound will be minimized at a non-zero $\lambda$ when $C \neq 0$ and in general, is there a setting where Unified SAM’s convergence speed will strictly improve over U-SAM?
[Empirical Studies] In the case of computer vision experiments, is the training loss of Unified SAM or USAM lower than SAM? As the best practice so far seems to be using SAM in the later phase of training, how to disentangle the sharpness reduction benefit of SAM over U-SAM [1,2]?


[1] The Crucial Role of Normalization in Sharpness-Aware Minimization, arxiv.org/abs/2305.15287 
[2] How Does Sharpness-Aware Minimization Minimize Sharpness? arxiv.org/abs/2211.05729

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The authors provide a Unified framework (Unified SAM) as a convex combination of SAM ascend and USAM ascend. They provide convergence guarantees for Unified SAM via PL condition. In special cases, the bounds reduce to that of SGD, which is known. The sampling they consider in their method is arbitrary, not restricted, and includes importance sampling. The paper concludes with experiments.

### Strengths
- interesting and well-motivated problem
 - very well written
 - clean theoretical contributions and supporting experiments

### Weaknesses
- the paper would benefit from an informal statement of results (in math) in terms of convergence rates at the begining

### Questions
This is an interesting paper. I have a question: how does the rate you prove depend on the parameter $\lambda$? I understand that you derive the results in the paper, but can you explain in words how the convergence rate is changed when $\lambda$ varies? Another question is, are you making the previous bounds tighter, or do you present a proof that works for new regimes?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
2