# Differentially Private SGD Without Clipping Bias: An Error-Feedback Approach

- Decision: Accept (poster)
- Scores: 6, 6, 6, 6

## Abstract
Differentially Private Stochastic Gradient Descent with Gradient Clipping (DPSGD-GC) is a powerful tool for training deep learning models using sensitive data, providing both a solid theoretical privacy guarantee and high efficiency. However, existing research has shown that DPSGD-GC only converges when using large clipping thresholds that are dependent on problem-specific parameters that are often unknown in practice. Therefore, DPSGD-GC suffers from degraded performance due to the {\it constant}  bias introduced by the clipping. In our work, we propose a new error-feedback (EF) DP algorithm as an alternative to DPSGD-GC, which offers a diminishing utility bound without inducing a constant clipping bias. More importantly, it allows for an arbitrary choice of clipping threshold that is independent of the problem. We establish an algorithm-specific DP analysis for our proposed algorithm, providing privacy guarantees based on R{\'e}nyi DP. And we demonstrate that under mild conditions, our algorithm can achieve the same utility bound as DPSGD without gradient clipping. Our empirical results on standard datasets show that the proposed algorithm achieves higher accuracies than DPSGD while maintaining the same level of DP guarantee.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
A novel algorithm for stochastic differentially private model training is proposed that eliminates the bias caused by clipping. The essential idea is to maintain the clipped component in a separate accumulator that is also added (with its own clipping) to the update at each iteration. A convergence proof is provided (for strongly convex loss) showing that the algorithm does not suffer from the O(1) term caused by clipping in typical DP-SGD. Experimental results show that the method closes the gap to non-private SGD on several tasks.

### Strengths
The algorithm is novel and, while not terribly surprising in hindsight (like many good ideas) is a useful contribution. More original is the DP analysis, which is complicated due to the non-privatized hidden clipping-error-accumulator state. The experimental results (on non-convex objectives) are encouraging, and it is in particular very exciting that the dependence on the clipping threshold appears to be weak. If future results continue to show such a pattern, that would be particularly advantagous, as DP-SGD can be quite sensitive to the clipping threshold.

### Weaknesses
It is unfortunate that the convergence result depends on strong convexity. This limits the usefulness of the result. Fortunately the DP guarantee does not require the strong convexity assumption, and the experimental results use non-convex models.

The experiments don't seem to be very careful about how $C$ is chosen. For example, in Table 3, it is just stated that C=1. Considering the whole point of the paper is a better method for clipping, I would like to see more exploration into this. In Table 2, we just look at two points C=1.0 and C=0.1 for each dataset. But there could be an interesting curve there because there is a tradeoff between clipping error (which is hopefully alleviated by DiceSGD) and the amount of noise you have to add for DP.

### Questions
* It is stated that DiceSGD requires a constant multiplicative factor more noise than DPSGD. What is that factor?
* Given that DiceSGD maintains a private state anyway, and the convergence analysis assumes strong convexity, could the privacy guarantee be strengthened by NOT releasing the model after every step, similar to Feldman et al. "Privacy Amplification by Iteration" (which is cited)? The analysis you already have -- for non-convex objectives and releasing all iterates -- seems to me primarily useful because it bounds the privacy loss for non-convex objectives *without* releasing all iterates. I can't imagine a scenario where you can protect the private state but nevertheless want to release all model iterates. On the other hand, if it were possible, an analysis of the convex case where all model iterates are private could give you a tighter bound.
* In the NLP experiment, it is stated that GPT-2 is fine-tuned. By default I would assume that means all-parameter fine-tuning, but Hu et al. (2021) is cited twice (referencing the metrics and in the table) which makes me wonder if it is Lora fine tuning. Which is it?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new variant of DP-SGD algorithm that eliminates the clipping bias during training. The key idea is to borrow the error-feedback mechanism for compressed SGD. Theoretically, under some common assumptions, this paper provides novel convergence analysis and privacy analysis. Finally, the effectiveness of the proposed algorithm is demonstrated on empirical experiments.

### Strengths
1. This paper is well-written.
2. The proposed algorithm is relatively simple and scalable. The algorithm itself does not require fine-tuning the clipping threshold.
3. Proofs of utility and privacy both require non-trivial effort and novel technique.

### Weaknesses
Compared to DP-SGD with a properly chosen clipping threshold, the error rate is larger, and the convergence is slower. This is due to the larger noise needs to be added. I was wondering if there is any practical implication in the experiments.

### Questions
Please see weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper to add a state accumulating the error caused by clipping in DP-SGD and adding it back later, which is called DiceSGD. It is designed to reduce the clipping bias of DP-SGD

### Strengths
1. This is an interesting attempt at analyzing and mitigating the clipping bias in DP-SGD.
2. The convergence analysis and privacy analysis are non-trivial.

### Weaknesses
1. My major concern is that although clipping bias can cause privacy degradation in DP-SGD, the major drop is still caused by the added noise. Thus I am not sure how useful it is to trade noise variance for clipping bias. The empirical study partially answers the question but might not be generalizable.
2. I think DiceSGD should be compared with DP-FTRL and matric mechanism as well since they are all stateful algorithms. In terms of implementation, DP-SGD is more efficient than all of them since it is stateless.

### Questions
Is it possible to get a rigorous bias-variance trade-off to confirm that DiceSGD is better than DP-SGD?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a differentially private optimization algorithm with error feedback as a means to combat the clipping bias. The resulting algorithm is shown to converge without a bias at all values of the clip norm (whereas DP-SGD analyses show convergence at particular clip norm values). Experiments show that the proposed approach outperforms DP-SGD. The theoretical analysis requires a strong assumption that amounts to requiring that all per-example gradients are almost surely bounded.

### Strengths
- The result is quite significant. Clipping is a known problem in the literature both in theory (where existing analyses are inadequate) and in practice (where the clip norm needs to be tuned carefully). This paper addresses this important problem.
- The main paper is reasonably well-written and clear to understand. The fixed point analysis used to motivate the method does a great job of explaining the key ideas. The proofs could use some rewriting to improve clarity (details below).
- The optimization proofs look right to me, although I have not verified them line-by-line. I was unable to parse the privacy proofs which could use some rewriting.
- I would rate the originality as moderate since the paper is the application of a well-known technique from optimization (and signal processing before that) to DP optimization. This application introduces some technical challenges in the proof which the paper handles well.

### Weaknesses
**Theory**:
- _Discussion of assumptions_: The assumption that $\|\nabla f(x) - g_i\|_2 \le \sigma$ everywhere is rather strong. It needs to be better and more transparently justified. Its ramifications need to be discussed more carefully. For instance, under such an assumption, it is not inconceivable that DPSGD with [adaptive clipping](https://arxiv.org/abs/1905.03871) also converges favorably (see e.g. [Varshney et al.](https://arxiv.org/abs/2207.04686) who used this trick for linear regression). A comparison to this approach both theoretically and empirically is missing.
- _Convergence rates_: I believe that $\eta = O(T^{-(1-\nu)})$ is a suboptimal choice and the right rate should be something like $\exp(-c\sqrt{t}) + \log t /t$. See e.g. [Section 3 of Stich](https://arxiv.org/pdf/1907.04232.pdf) for notes on tuning the step size carefully to get the best rates.
- _Setting of parameters: How does the convergence rate vary with $C_1$ and $C_2$? What are their optimal values? How much does the rate suffer when these constants differ from their optimal values?
- _Potentially incorrect statements_: Page 8 says "the result does not rely on the specific values of [$\sigma$ and $G$]". I do not think this is true: $\sigma_1$ depends on $\tilde G$, which depends on $G'$ which depends on $G$ and $\sigma$.
- _Effect of the clip norm on the learning rate_: Suppose the clip norm is so small that all gradients are effectively clipped. Then, reducing the clip norm any further has the role of reducing the learning rate. Do the bounds capture this dependence? More generally, how does the clip norm determine the optimal learning rate?

**Inadequate experiments**:
- The experimental results are promising but many more experiments are needed to paint a convincing picture. It is fine to run some detailed experiments on smaller-scale settings that can be tested more extensively (I would argue that these have more value for this particular paper than inadequate experiments on larger models but the authors can feel free to disagree). 
- Standard deviations across multiple repetitions need to be reported as the algorithms are noisy
- How is the clip norm of DPSGD tuned? That is, is the optimal choice one of $C \in \{0.1, 1\}$. If the claim is that DiceSGD converges regardless of the clip norm, one needs to show the results for a broad range of clip norms (including the best possible for DPSGD) ranging over several orders of magnitude.
- How do these results translate to different values of $\epsilon$?

**Clarity (main paper)**:
- The distinction between "DPSGD" (without clipping) and "DPSGD-GC" (with clipping) is confusing. DPSGD is always presented in the literature with clipping and theoretical analyses use Lipschitz functions (=> bounded gradients) to avoid the technicalities induced by clipping. It would be much more natural to refer to both as DPSGD and make a distinction in the setting (i.e., whether the clipping is active or not).
- Section 2.1: why even mention $(\epsilon, \delta)$-DP? It would be much cleaner to describe the background in terms of RDP and the proofs use this as well.
- $\mathbb{E}_i$ and $\mathbf{g}_i$ are not defined but are used in various places (.e.g Assumption 3.5). This is also poor notation because it implicitly hides the dependence on $\mathbf{x}$.

**Clarity (proofs)**:
- The definitions of key quantities such as $\Delta^t$ and $\alpha^t_e$ should be featured more prominently. Right now, they are hidden somewhere amid the proof and are hard to find. E.g. bottom of p.18, $\alpha^t_e$ is used but it is defined in the proof of Theorem 3.7.
- Lemma A.6: $\mathcal{A}_1^t, \mathcal{A}_2^t, \mathcal{H}^t$: I do not understand this notation. Some examples will be helpful.
- p.18, Step II: This proof is very hard to parse and the quantities do not make sense to me. Please consider standardizing notation across various proofs for the reader's convenience and simplifying the notation in this proof in particular.

**Missing refs**:
- Using normalized gradients instead of clipping: https://arxiv.org/pdf/2106.07094.pdf
- Section 2.3: there is some literature on joint DP + compression e.g. https://arxiv.org/abs/2111.00092

**Minor**:
- Why is $\tilde g_i^t$ not boldfaced in Algorithm 1?
- There is $\mathbb{E}$ missing in the "e" terms in the first display equation of page 6.
- Assumption 3.1: typo $\mathbb{R} -> \mathbb{R}^d$. Btw, Assumption 3.1 always holds under strong convexity (which is assumed), so it does not have to be explicitly specified
- Assumption 3.2: reads like a definition more than an assumption
- Theorem 3.7: $\frac{1}{2\mu} \ge \frac{3}{32 L}$ always, so that condition on $\eta$ can be dropped.
- Theorem 3.7: $\kappa$ is typically used to refer to the condition number and can be quite misleading here. Besides, there is no need to define a new variable if it is used only once.
- $\nu$ and $\tilde G$ are not defined in Table 1
- Theorem 3.8: $\sigma_1$ is not a noise multiplier, it is the noise standard deviation. See [here](https://arxiv.org/pdf/2303.00654.pdf) for details on the terminology.
- Formating math: proper-sized brackets (using \left, \right), etc.
- Some references need to be updated (citing arxiv versions instead of the published ones, etc.)
- Typo on p.18, step 1: $\sigma -> \sigma_1$
-Typos in Lemma A.11: do you need $\sigma > 4$ and $\epsilon = 2p^2 \alpha / \sigma^2$.

### Questions
I've given several questions in the "weaknesses" section. Some more questions:
- p. 18, Step III: In the definition of $\Delta^t_g$, are the $g_i^t$ under both $D$ and $D'$ assumed to be the same? If the differing example was sampled in iteration $k < t$, then the iterates $\mathbf{x^t}, \mathbf{x'^t}$ are both distinct. Then, their corresponding gradients in iteration $t$ are also distinct. 

Overall, I think this idea is great. If the authors can develop the theory further, run comprehensive experiments, and improve the clarity, I think this paper can be really impactful.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
