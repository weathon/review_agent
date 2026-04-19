# Convergence Analysis of Adaptive Gradient Methods under Refined Smoothness and Noise Assumptions

- Decision: Reject
- Scores: 5, 5, 6, 6

## Abstract
Adaptive gradient methods, such as AdaGrad, are among the most successful optimization algorithms for neural network training. While these methods are known to achieve better dimensional dependence than stochastic gradient descent (SGD) under favorable geometry for stochastic convex optimization, the theoretical justification for their success in stochastic non-convex optimization remains elusive. In fact, under standard assumptions of Lipschitz gradients and bounded noise variance, it is known that SGD is worst-case optimal (up to absolute constants) in terms of finding a near-stationary point measured by the $\ell_2$-norm, making further improvements impossible. Motivated by this limitation, we introduce refined assumptions on the smoothness structure of the objective and the gradient noise variance, which better suit the coordinate-wise nature of adaptive gradient methods. Moreover, we adopt the $\ell_1$-norm of the gradient as the stationarity measure, as opposed to the standard $\ell_2$-norm, to align with the coordinate-wise analysis and obtain tighter convergence guarantees for AdaGrad. Under these new assumptions and the $\ell_1$-norm stationarity measure, we establish an **upper bound** on the convergence rate of AdaGrad and a corresponding **lower bound** for SGD. In particular, for certain configurations of problem parameters, we show that the iteration complexity of AdaGrad outperforms SGD by a factor of $d$. To the best of our knowledge, this is the first result to demonstrate a provable gain of adaptive gradient methods over SGD in a non-convex setting. We also present supporting lower bounds, including one specific to AdaGrad and one applicable to general deterministic first-order methods, showing that our upper bound for AdaGrad is tight and unimprovable up to a logarithmic factor under certain conditions.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper explores whether the iteration complexity of AdaGrad surpasses that of SGD in stochastic non-convex optimization. By introducing refined assumptions that align with the coordinate-wise nature of adaptive gradients, the authors use the L1-norm of the gradient as a stationarity measure and demonstrate that AdaGrad's iteration complexity outperforms SGD by a factor of d.

### Strengths
1. The paper successfully bridges a gap in the theoretical understanding of AdaGrad compared to SGD in non-convex settings.
2. The comparison between the upper bound for AdaGrad and the lower bound for SGD is well-executed, clearly highlighting the advantage of AdaGrad.

### Weaknesses
1. The refined assumptions used are not particularly novel; similar approaches have been explored in previous works. Additionally, the use of the L1-norm for the theoretical analysis has been discussed in recent literature, limiting the paper's overall novelty.
2. The authors have already highlighted the main differences between this paper and [1], noting that [1] examines the upper bound for SGD, while this paper focuses on the lower bound. However, despite this distinction, the two papers remain quite similar. A more in-depth discussion of the unique techniques or specific analytical steps that differentiate this work would enhance the clarity and contribution of the paper.
3. The lack of experimental validation is a significant shortcoming. While the paper emphasizes theoretical analysis, some empirical results would be beneficial. Demonstrating that AdaGrad reduces the bound of SGD by a factor of d through experiments—especially by varying the value of d—would greatly support the theoretical claims.

[1] Yuxing Liu, Rui Pan, and Tong Zhang. Large batch analysis for adagrad under anisotropic smoothness. arXiv preprint arXiv:2406.15244, 2024.

### Questions
See the Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper provided a convergence analysis of the adaptive gradient (AdaGrad) in the nonconvex setting. Relaxing standard assumptions of Lipschitz gradients and bounded noise variance in the $\ell_2$ metric, where the SGD is worst-case optimal in terms of finding a near-stationary point, the paper proposed a coordinate-wise analog conditions under which the AdaGrad can improve the dimension $d$ dependence in the iteration complexity over the SGD under the $\ell_1$ norm. In particular, the paper proved: (i) an upper bound of AdaGrad; (ii) a lower bound of SGD; and (iii) a lower bound of AdaGrad (matching AdaGrad upper bound for the noiseless term). The paper showed that for certain problem configurations, the gap between the AdaGrad upper bound and the lower bound of SGD is non-trivial. More precisely, the iteration complexity of AdaGrad outperforms SGD by a factor of $d$, thus arguing the improved theoretical guarantee of AdaGrad over SGD in certain nonconvex settings.

### Strengths
The results of this paper are mainly theoretical. This paper compares the convergence rate of AdaGrad with SGD under coordinate-wise
assumptions (smoothness and noise) and $\ell_1$-norm stationarity measure. The paper demonstrates a provable advantage of AdaGrad over SGD in certain nonconvex settings.

The paper is well-written and the intuition for introducing the $\ell_1$-norm of the gradient as the stationarity measure is well-explained.

The paper mainly compared the upper bound for AdaGrad with the lower bound of SGD, while the derivation of the latter result is missing in the concurrent work [1], as stated by the authors.

[1] Yuxing Liu, Ru Pan, and Tong Zhang. Large batch analysis for Adagrad under anisotropic smoothness. arXiv preprint arXiv:2406.15244,2024.

### Weaknesses
It would be better if the comparison between SGD and AdaGrad in convex settings is provided, for both $\ell_2$-norm measure (I guess it belongs to the existing literature) and $\ell_1$-norm stationary measure. Moreover, even though the paper is theoretical, it would still be beneficial to demonstrate the gain of dimension dependence using some numeric experiments.

### Questions
The improvement of AdaGrad over SGD is demonstrated with abstract assumptions on smoothness and noise. Can the paper give a concrete example or more primitive assumptions (such as in the Abstract motivation of training neural networks) when those proposed assumptions are satisfied?

In the non-convex setting, this paper utilizes the $\ell_1$-norm of the gradient as the stationarity measure and compares the convergence rate. When comparing different optimization algorithms, they might converge to different stationary points (such as due to initialization and implicit bias). How does the initialization impact the AdaGrad dynamics? Is there any discussion on the works for the implicit bias of SGD and AdaGrad in non-convex settings?

As raised in the Weakness, it would be beneficial to demonstrate the gain of dimension dependence using some numeric experiments, i.e., the gap between the AdaGrad upper bound and SGD lower bound, and their difference in iteration complexity when changing dimensions.

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
4

### Summary
The paper provides expected average $\ell_{1}$-norm convergence upperbound and lowerbound for AdaGrad under coordinate-wise bounded variance gradient noise and coordinate-wise smoothness assumptions for non-convex objectives. Comparisons between coordinate-wise AdaGrad and SGD are performed to demonstrate dimensional advantage of coordinate-wise AdaGrad over scalar SGD under favorable conditions.

### Strengths
• Interesting fine-grained analysis that shows advantage of coordinate-wise methods over scalar versions via better dimensional dependency. 

• The lower bound for AdaGrad is interesting where the implication is that the log(T) factor in AdaGrad is shown to be not improve-able.

### Weaknesses
- Comparisons are not sound. 
    - When comparing with SGD, especially when the lower bound utilizes new coordinate-wise smoothness, one should also consider a coordinate wise version of SGD as a baseline i.e. SGD with stepsize $\eta_{i}=\Theta(1/L_{i})$ for each $i\in[d]$. In that case, both algorithms are allowed $O(d)$ memory (rather than vanilla SGD only uses O(1) memory). There, I think the benefit of adaptive methods will be more of automatically adapting to these coordinate wise smoothness rather than any dimensional improvement in the convergence rate.
    - The worst-case comparison on lines 358-362 is not clear: For example, if $\sigma_{i}=1$ for all $i$, then $\|\|\sigma\|\|_2=\sqrt{d}$ and $\|\|\sigma\|\|_1=d$. There, the dimensional dependency is still $d$ and not $\sqrt{d}$. What am I missing?
    - Previous work [Liu et al 2023] provides $\ell_{1}$-norm convergence in high probability for sub-Gaussian coordinate-wise noise for AdaGrad. There, I don't think it's fair to compare results in high-probability with results converge in expectation (in the related works section and Appendix A). 
    - The comparison between AdaGrad and SGD on lines 512-516 is not too convincing. The authors should also comment on the setting when AdaGrad is worse than SGD and discuss the reason behind that. 

- Upperbound analysis for AdaGrad is not novel and upperbound results seem incremental. 
    - The analysis seems quite similar to Ward et al 2020 with the generalization to coordinate-wise version of the variance and smoothness. Furthermore, $\ell_{1}$-norm convergence for non-convex objective under coordinate-wise noise is not new (e.g. see Liu et al. 2023), and the extension of Liu et al. 2023's proof to coordinate-wise smoothness seems not too difficult. 
    - Could the authors comment on the main difficulties needed to generalize existing in expectation results to use the new assumptions?

- Need to verify new assumptions in practical settings: 
    - The authors should show that these assumptions are observed in real world tasks (e.g. training transformers, CNNs, etc.) where adaptive methods converge faster than SGD, Adagrad-norm. 
    - The authors should perform controlled synthetic experiments to verify that coordinate-wise adaptive methods outperform scalar SGD and AdaGrad-norm under the conditions posed in the comparison.

### Questions
The concerns from weakness section should be addressed. I am willing to raise my score.

To make the paper's contribution more significant, I think the authors should consider performing some empirical validation of the new assumptions (coordinate noise, sparsity rate, coordinate smoothness, etc.), especially in modern DNNs settings (e.g. LLMs) where dimension is large and perform similar analysis on a wider range of algorithms like Adam or RMSProp.

### Soundness
3

### Presentation
3

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
The paper studies the convergence of AdaGrad under: 1) coordinate-wise smoothness and noise variance, and 2) convergence of the $L_1$ norm. In the noiseless setting, AdaGrad is proven to be superior (not inferior) to SGD in terms of constant dependence. Moreover, in the stochastic setting where the noise is sparse and aligns with the smoothness parameter, AdaGrad also exhibits better constant dependence. This is done by also establishing a lower bound for SGD when using the $L_1$ norm. Furthermore, the paper presents lower bounds for AdaGrad and all deterministic first-order algorithms.

### Strengths
- The paper is well-written and easy to follow, with a well-organized language. 
- The technical contribution is solid, presenting a novel perspective on showing the provable advantage of AdaGrad under coordinate-wise smoothness and $L_1$-norm convergence. 
- Additionally, the lower bounds provided are novel in the context of $L_1$-norm convergence.

### Weaknesses
- The provable advantage is clear in the deterministic setting, while for the stochastic setting, the benefit depends, it could be even worse. In real-world applications, it is unknown whether the noise is sparse and aligns with the smoothness parameters.

### Questions
- It’s unclear why $L_1$-norm convergence is considered a better measure, if not to artificially distinguish between SGD and AdaGrad. Since adaptive methods generally exhibit faster convergence in practice, is there a rationale behind why the $L_1$-norm result aligns more closely with the observations?
- Can we achieve similar advantages if SGD employs a coordinate-wise stepsize, such as $1/(L_i \sqrt{T})$ for the $i$-th coordinate?

### Soundness
4

### Presentation
4

### Contribution
3
