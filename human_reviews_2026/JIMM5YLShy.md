# On the Benefits of Weight Normalization for Overparameterized Matrix Sensing

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 4

## Abstract
While normalization techniques are widely used in deep learning, their theoretical understanding remains relatively limited. In this work, we establish the benefits of (generalized) weight normalization (WN) applied to the overparameterized matrix sensing problem. We prove that WN with Riemannian optimization achieves linear convergence, yielding an $\textit{exponential}$ speedup over standard methods that do not use WN. Our analysis further demonstrates that both iteration and sample complexity improve polynomially as the level of overparameterization increases. To the best of our knowledge, this work provides the first characterization of how WN leverages overparameterization for faster convergence in matrix sensing.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the advantages of a generalized weight normalization (WN) technique in the context of overparameterized matrix sensing, where the goal is to recover a low-rank PSD matrix from linear measurements. The authors prove that WN, combined with Riemannian gradient descent (RGD), achieves linear convergence with random initialization, offering an exponential speedup over standard gradient descent (GD) methods without WN. They demonstrate that increased over-parameterization reduces both iteration and sample complexity polynomially. The analysis reveals a two-phase convergence behavior (initial saddle-escape phase followed by linear convergence) and is supported by experiments on synthetic data and image reconstruction tasks.

### Strengths
The paper's primary strength lies in its novel theoretical insights, providing the first characterization of how WN leverages overparameterization for faster convergence in matrix sensing, with a rigorous proof of linear convergence that exponentially outperforms sublinear rates in non-WN methods. It quantifies the benefits clearly, showing polynomial improvements in iteration and sample complexity as overparameterization increases, which contrasts positively with prior work where overparameterization hinders performance. Empirically, the experiments robustly validate the theory, including comparisons under varying conditions like condition numbers, overparameterization levels, and noise, with real-world image reconstruction adding practical value. The manuscript is well-organized and clear, featuring helpful tables, figures, and a reproducibility statement with full proofs and setups in appendices.

### Weaknesses
1. While the analysis is focused and insightful, its scope is limited to symmetric PSD matrix sensing. It would be helpful if the authors could briefly explain why it is reasonable to consider only the symmetric PSD case.
2. Experimentally, while solid, the studies are somewhat constrained in scale (e.g., small matrix dimensions in synthetic tests).
3. The RIP condition required for the main theorem scales as $\delta = O((r - r_A)^6 / (\kappa^2 m^3 r^4 r_A))$, which appears to be rather stringent. A brief analysis or discussion of this condition's implications would be valuable.

### Questions
Overall, I don't see major issues that undermine the core contributions—the theory is sound, and the claims are well-supported. 
That said, here are a few constructive suggestions:
1. I am curious about how the proposed algorithm would perform in more challenging matrix sensing settings, such as those with specially designed structures or nontrivial optimization landscapes (e.g., as discussed in arXiv:2110.10279).
2. A short note on potential applications (e.g., in signal processing) could emphasize the work's relevance to ICLR's audience.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides a theoretical characterization of how Weight Normalization (WN) accelerates convergence in overparameterized matrix sensing. The authors prove that Riemannian gradient descent (RGD) with WN achieves linear convergence and improved sample complexity, while standard gradient descent (GD) may exhibit exponential slowdown. Theoretical findings are supported by synthetic experiments that align well with the analysis.

### Strengths
1.Provides the first clear theoretical explanation of how WN accelerates convergence under overparameterization.

2.Solid and rigorous analysis using Riemannian optimization tools.

3.Well-aligned experiments that validate the theory.

4.Overall writing and presentation are clean and accessible.

### Weaknesses
1.The paper identifies a two-stage convergence pattern with a transition at $r_A-\frac{1}{2}$ yet this boundary is not theoretically justified or intuitively explained.

2.The “Full-rank case (r = m)” in Sec. 5.3 is conceptually an extension of Sec. 5.2 (“ON THE BENEFIT OF OVERPARAMETERIZATION”) and could be merged there for better logical flow.

3.The title seems somewhat broader than the actual technical scope, which is limited to PSD matrix sensing.

### Questions
1.See Weakness. 1.

2.Since $A$ is PSD, one might consider reformulating (1) as 

$\min_{X, \Theta} f(X, \Theta) = \frac{1}{4}\|M(X\Theta X^{\top})-y\|^2$

with $\Theta$  being diagonal and nonnegative. Would this restriction simplify the analysis or improve interpretability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies a matrix factorization similar to the weight normalization that can be helpful for the matrix sensing problem. The author proves that this matrix factorization with Riemannian optimization can achieve a linear convergence rate, which is an exponential improvement over the previous lower bound for symmetric matrix sensing.

### Strengths
1. The paper is well-organized and easy to follow. The contributions of this paper is proposed in a clear way, and the main technical idea is also clear.  The author also provides solid theoretical proof and some discussions about the main result. 

2. The authors also further study the initial increment learning phase in the optimization process, which is good for understanding the optimization dynamics of WN with Riemannian optimization. 

3. The empirical part is comprehensive. Beyond the simulated optimization problems, the authors also provide experiments on image reconstruction problem. The performance of WN with Riemannian optimization is better compared to GD, which also matches the theoretical discovery of this paper.

### Weaknesses
1. My main concern is that after relaxing the PSD constraint, the problem no longer corresponds to symmetric matrix sensing. Specifically, when the constraint $\Theta \in S^r_{+}$ is relaxed to $\Theta \in S^r$, the term $X^T \Theta X$ may no longer be written as $YY^T$. Thus, the proposed method is not a true solution to symmetric matrix sensing but rather an approach that accelerates matrix sensing in general. This weakens the paper’s contribution, as several existing methods [1,2] also focus on improving the efficiency of matrix sensing.

2. The paper introduces Riemannian optimization equations without providing sufficient background or basic explanations. It would be helpful if the authors could include a high-level introduction and some intuitive discussion of Riemannian optimization to improve readability.

[1]. Xu et al. 2023. The power of preconditioning in overparameterized low-rank matrix sensing

[2]. Xiong et al. 2024. How over-parameterization slows down gradient descent in matrix sensing: The curses of symmetry and initialization

### Questions
Lemma 4.1 states that a point $(X, \Theta)$ is a saddle point if certain conditions are satisfied. However, this does not necessarily imply that all saddle points satisfy these conditions. Could the authors clarify how this lemma connects to the saddle-to-saddle dynamics and the subsequent incremental learning process?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors address the problem of overparameterized matrix sensing and present an analysis of the benefits of applying weight normalization in this context. They reformulate the classical matrix sensing problem by decoupling the magnitude and direction components of the matrix, and they demonstrate that using Riemannian gradient descent with weight normalization improves the convergence rate, particularly when the level of overparameterization is large. Experiments on both synthetic and real-world datasets are provided to support the analysis.

### Strengths
The paper is well written, clearly structured, and the ideas are well explained. The theoretical motivation and experimental design are both sound.

### Weaknesses
I have a major concern regarding the positioning of the paper within the existing literature. While the paper emphasizes the proposed weight normalization approach, its main benefit—improved convergence rate—falls within a broader line of research on optimization techniques for matrix sensing. In particular, related approaches such as preconditioned gradient descent and their follow-up works have already been proposed in this domain. Although some of the related (e.g. preconditioned gradient descent) papers are cited, they are not discussed thoroughly. These works should be more deeply reviewed in the related work section, and, ideally, comparisons should be included in the experimental evaluation to better situate the contribution and highlight the advantages of the proposed approach over existing methods.

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
3
