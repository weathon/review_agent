# Alignment-Sensitive Minimax Rates for Spectral Algorithms with Learned Kernels

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
We study spectral algorithms in the setting where kernels are learned from data. We introduce the effective span dimension (ESD), an alignment-sensitive complexity measure that depends jointly on the signal, spectrum, and noise level $\sigma^2$. The ESD is well-defined for arbitrary kernels and signals without requiring eigen-decay conditions or source conditions. We prove that for sequence models whose ESD is at most $K$, the minimax excess risk scales as $\sigma^2 K$. Furthermore, we analyze over-parameterized gradient flow and prove that it can reduce the ESD of a sequence model, which in turn moves the problem into an easier ESD class and lowers the corresponding minimax risk. This analysis suggests a general route to study how adaptive feature learning can improve generalization through signal-kernel alignment: adaptive learning procedures reshape the kernel so that the ESD decreases and the problem enters an easier ESD class. We also extend the ESD framework to linear models and RKHS regression, and we support the theory with numerical experiments. This framework provides a novel perspective on generalization beyond traditional fixed-kernel theories.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Effective Span Dimension (ESD), a new alignment-sensitive complexity measure for spectral learning algorithms. The ESD jointly depends on the signal, the kernel spectrum, and the noise level, and is shown to characterize the minimax excess risk in sequence models, as well as in their linear and RKHS counterparts. The authors further analyze over-parameterized gradient flow dynamics, proving that adaptive eigenvalue learning can reduce the ESD, and thereby improve generalization. The paper concludes with numerical experiments that visualize ESD evolution and empirically support the theoretical claims.

### Strengths
The paper offers an original and technically sound contribution to the theory of spectral learning. The introduction of the ESD provides a novel lens through which to understand generalization in adaptive kernel and spectral methods. The ESD allows the analysis of signal–spectrum alignment, bias–variance trade-offs, and minimax optimality within a single framework. The theoretical results are clearly presented, mathematically sound, and supported by illustrative experiments.

### Weaknesses
The connection to neural networks is somewhat overstated. Most theoretical results hold only in idealized sequence, linear, or RKHS models, while genuine neural networks involve evolving eigenfunctions, which the paper explicitly leaves for future work. Consequently, the practical relevance to real NN training remains speculative. In addition, while the paper contrasts ESD with classical notions such as the effective dimension, this comparison is not entirely fair: traditional kernel learning theory (e.g., Caponnetto & De Vito, Zhang) already incorporates signal–kernel alignment through the RKHS norm of the target function or source condition constants, albeit indirectly. While the ESD formalism clarifies and extends this dependence, the importance of target-kernel alignment was already recognized as a critical aspect of learning in the past literature.

### Questions
1. _Applicability to Neural Networks:_ The paper’s framing emphasizes relevance to neural network generalization, yet all formal results are derived in simplified sequence or RKHS models. Could the authors clarify what specific aspects of neural network training (beyond the stylized over-parameterized gradient flow) they expect the ESD framework to capture? For instance, do they anticipate ESD reductions under standard SGD dynamics in finite-width networks?

2. _Estimating the ESD in Practice:_ Since the ESD is defined at the population level and depends on the unknown signal coefficients, it is unclear how it could be estimated from data. Do the authors foresee practical estimators or proxies for the ESD that could be applied in empirical studies?

3. _Relation to Classical Source Conditions:_ Classical RKHS learning rates already depend on the target’s regularity (e.g., through the source condition constant or RKHS norm). Could the authors elaborate on how their framework fundamentally differs from these forms of target dependence? Is the ESD intended to replace such constants, or does it complement them?

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
In the context of kernel regression, the paper introduces a new characteristic, Effective Span Dimension (ESD), discusses its properties, and uses it to analyze the spectral alignment between the target (represented by its spectral expansion coefficients) and the learning algorithm (represented by the kernel eigenvalues). First, the original regression problem is reduced to a spectral "sequence model" setting. Then, the ESD is introduced, motivated by the Principal Component algorithm. It is then argued that the ESD is relevant for any estimator (Theorem 3.3) and captures properties not captured by alternative alignment measures and effective dimensions. Then, a span profile reflecting the dependence of ESD on the noise level is introduced and discussed. A minimax rate result analogous to the earlier Theorem 3.3 is established. Then, Section 5 considers gradient flow in a deep linear network-type models and shows, under certain assumptions, a dynamic signal-spectrum alignment in terms of ESD. Finally, section 6 provides numerical demonstrations of spectral alignment and associated evolutions of span profiles.

### Strengths
The paper is carefully and thoughfully written. The mathematical level of the paper is high, and the explanations are good. The authors are very familiar with spectral algorithms and respective literature.   

The paper has conceptual novelty in introducing a new charactiristic, Effective Span Dimension (ESD), applicable to general spectral distributions in contrast to the standard power-law source/capacity framework. This characteristic is relevant for target-spectrum alignment studies and might be useful in future research (though I'm not convinced in its importance; see Weaknesses below). In particular, the authors demonstrate, in terms of this characteristic, an alignment under gradient flow in some special setting (Theorem 5.2). 

The paper contains both rigorous results and experiments. Several lengthy appendices provide additional discussion, numerical illustrations and the proofs. It appears that the authors have put a significant effort into this work.

### Weaknesses
I don't see why the Effective Span Dimension - the key concept proposed and studied in the paper - is as fundamental and important as the authors claim ("*the ESD is a fundamental measure for signal-spectrum alignment*").

1. ESD is introduced essentially as the critical value of the parameter $d$ in the PC method at which the squared bias matches the variance. A critical value of the regularization parameter can presumably also be considered for any other spectral method, e.g. previously mentioned Ridge and GF. For other methods the connection between the critical parameter and the risk may be not so explicit, but one can nevertheless define and study profiles analogous to the span profile of definiton 3.4. Each method would then have a different profile; there is no obvious connection between profiles of different methods.

2. The authors argue that ESD is fundamental because, while it is introduced using a specific estimator (PC), by Theorem 3.3 it quantifies the best performance of any estimator. However, this argument is misleading, since PC plays a preferred role in Theorem 3.3. Namely, the class $\mathcal F_{K,\lambda}^{(n)}$ is effectively defined there as the class of all problems on which the risk of the PC estimator is sufficiently low. The theorem then states that no other estimator can uniformly outperform PC on this class. But this is hardly surprising, since different estimators are efficient on different classes of problems. Likewise, I would expect analogs of theorem 3.3 to hold for other classes of problems, introduced using baseline estimators other than PC. In any case, Theorem 3.3 does not imply that the ESD is as relevant for other estimators as for PC.   

3.  I don't see how ESD can be used to efficiently compare different spectral algorithms and clearly reveal their important theoretical properties. In the standard source/capacity framework one can, for example, identify a specific range of exponents where GF achieves the minimax rate, but Ridge fails to do that due to saturation (Math´e, 2004; Bauer et al., 2007; Pillaud-Vivien et al., 2018). While limited to the standard source/capacity setting, this result provides a concise, sharp and contrastive characterization of the two algorithms and so has a clear theoretical value. I don't see any comparably interesting results in the present paper. Theorems 3.3 and 4.3 basically say that PC cannot be uniformly improved on the problems where PC has an upper bounded risk. Theorem 5.2 does demonstrate a signal-spectrum alignment by GF in terms of ESD. However, I don't find this theorem very illuminating. It is established in a very special linear network setting and under some complicated assumptions. The effect is likely non-universal, and it's not very clear which properties of the algorithm and the problem are essential for it. The theorem only tells us something about the ESD, which is an auxiliary quantity, rather than the risk which is the primary object of interest. In particular, the theorem doesn't demonstrate acceleration of learning thanks to the spectral alignment.

### Questions
N/A

### Soundness
3

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
3

### Summary
This paper introduces a novel, signal-aware complexity measure for kernel regression, the Effective Span Dimension (ESD). This measure, depends on the interplay between the true signal, the kernel's spectrum and the noise level. The authors first approximate the RKHS regression problem with a sequence model. They then analyze this sequence model and prove that their proposed ESD is a fundamental measure of the problem's difficulty by showing that the risk of an oracle Principal Component estimator is tightly bounded by the ESD and proving a matching minimax lower bound. Finally, the authors connect this framework to the algorithm of Li & Lin (2024), showing that throughout training, this adaptive process actively reduces the ESD.

### Strengths
The authors correctly identify the shortcomings of "signal-agnostic" measures which fail to capture the critical interplay between the signal and the kernel. This idea of the EDS is sound. Aside minor typos, the paper is well-written.

### Weaknesses
While the premise of the paper is sound, the execution is not entirely convincing. My biggest criticism is that the entire framework is built upon an "oracle" estimator (the PC estimator in the sequence model and the KPCE estimator in the RKHS model) that assumes access to unknowable, population-level quantities (the eigenvalues and eigenvectors of the integral operator). It makes it unclear if we can use the ESD to study the standard kernel ridge regression estimator or the kernel PC ones (the non oracle version of Eq. (22)). The paper characterizes the difficulty of an idealized problem that no practitioner ever gets to solve.

### Questions
Typos/notations: 
- Eq. (3) isn't the equation with the covariance exact? Why did you use \approx? 
- The use of psi_j for eigenfunctions and psi_nu for filter functions is confusing
- Eq (8) it should be theta and not theta^\star in the set
- The use of sigma^2 for sigma^2_0/n is confusing
- Example 4.2: where is L_k defined? 
- On page 17 Gkj (d×d matrix) is referred to as the "empirical Gram matrix." In kernel literature, "Gram matrix" has a standard n×n meaning.

### Soundness
2

### Presentation
3

### Contribution
2
