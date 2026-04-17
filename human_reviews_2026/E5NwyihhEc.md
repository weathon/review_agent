# Why Variance Reduction Hurts Noisy Zeroth-Order Hard-Thresholding?

- Decision: Reject
- Scores: 6, 4

## Abstract
The hard-thresholding gradient descent approach is a primary method for solving $\ell_0$-constrained optimization problems to achieve sparsity. In the black-box setting, where only function outputs are accessible, recent work has introduced stochastic and variance-reduced zeroth-order hard-thresholding algorithms to establish both algorithmic and theoretical feasibility, specifically addressing the inherent conflict between zeroth-order and hard-thresholding. However, in practice, function outputs often contain noise, which exacerbates their conflict and undermines the robustness of these algorithms' guarantees. In this work, we investigate the performance for noisy zeroth-order hard-thresholding algorithms, providing convergence analysis for its stochastic version. Furthermore, we theoretically demonstrate the zeroth-order hard-thresholding variance reduction algorithms leveraging historical gradients inherently lowers the tolerable noise upper bound. Contrary to usual presumptions, our findings reveal that variance reduction techniques fail to enhance performance in this setting and even lead to worse feasibility compared to simpler methods without such techniques. These theoretical insights are validated through experiments on a sparse regression problem, black-box adversarial attacks, and biological gene expression.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work considers the problem of zeroth-order hard-theshlding in presence of noise, that is when 1) approximating gradients by computing finite differences of a random subset of components, 2) setting parameters to zero when they are small in magnitude, to satisfy a L0 penalty on the loss, and 3) The loss is stochastic, perhaps because of data batching.
In particular, the paper studies theoretically convergence rates of known algorithms (SZOHT), including variance reduction (VR-SZHT), and provides some empirical evaluations.
It shows that increasing the noise requires increasing the number of components used to estimate the gradient, and that, while variance reduction improves convergence, it also increases the algorithm’s sensitivity to noise.

### Strengths
- Paper is well written and clear

- Results appear to be correct and novel

- Theorem 2 is interesting. It shows that, while variance reduction improves convergence, it fails if the noise is too large, and the amount of variance reduction needs to be carefully adjusted to avoid this failure.

### Weaknesses
- Theorem 1 is quite unsurprising. It’s kind of obvious that increasing the noise requires increasing the number of components used to estimate the gradient (that effectively reduces the noise). However I am not able to comment on the technical contribution.

- There is no explanation on how to interpret experimental results. I didn’t quite understand them. Is the point that variance reduction VR-SZHT is worse than SZOHT (which is supposed to be the baseline)? 

Minor:

- Broken reference on line 249

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This submission studies the behavior of stochastic and variance-reduced zeroth-order hard-thresholding (ZOHT) algorithms under noisy conditions. The authors extend the stochastic ZOHT (SZOHT) framework to handle noise, providing convergence and complexity analysis under standard assumptions of restricted strong convexity and smoothness. They then generalize to variance-reduction variants—VR-SZHT (SVRG-style) and pM-SZHT (q-memorization-style)—and theoretically show that, contrary to intuition, variance reduction actually lowers the tolerable noise upper bound, thereby worsening performance in noisy environments. Experiments on synthetic regression, black-box adversarial attacks, and biological gene expression data confirm that simpler SZOHT is more robust than its variance-reduced counterparts.

### Strengths
Three strong points are as follows:

1. The paper demonstrates both analytically and empirically that variance-reduction techniques, though typically beneficial, can be harmful in noisy zeroth-order sparse optimization, offering a new understanding of the interplay between noise, gradient estimation, and sparsity.

2. The authors provide complete convergence proofs for SZOHT and its variants, establishing explicit bounds on the required number of random directions and the tolerable noise level.

3. Experimental results across three different tasks (synthetic regression, adversarial attacks, and noisy gene expression) consistently support the theoretical findings.

### Weaknesses
Three weak points are as follows:

1. The empirical section, though diverse in task type, remains small-scale; the algorithms are not tested on high-dimensional or large-data scenarios where zeroth-order methods are especially appealing.

2. The convergence results rely on restricted strong convexity/smoothness and bounded additive noise assumptions, which may not hold in highly nonconvex or structured real-world problems (the gene expression dataset used in this case).

3. The applicability of the proposed algorithms is very limited in the modern ML community. I was wondering if the authors could elaborate it on this.

### Questions
Q1. The discussion in the conclusion, suggesting that the results may inspire progress in “zeroth-order fine-tuning of Large Language Models,” seems overstated. The connection between the current theoretical setting (sparse recovery under additive noise with restricted strong convexity) and practical LLM fine-tuning is unclear. Could the authors discuss this in more detail?

Please see others in the weak points.

### Soundness
3

### Presentation
3

### Contribution
2
