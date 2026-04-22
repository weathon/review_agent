# Global Convergence of Four-Layer Matrix Factorization under Random Initialization

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 2, 6, 8

## Abstract
Gradient descent dynamics on the deep matrix factorization problem is extensively studied as a simplified theoretical model for deep neural networks. Although the convergence theory for two-layer matrix factorization is well-established, no global convergence guarantee for general deep matrix factorization under random initialization has been established to date. 
To address this gap, we provide a polynomial-time global convergence guarantee for randomly initialized gradient descent on four-layer matrix factorization, given certain conditions on the target matrix and a standard balanced regularization term. 
Our analysis employs new techniques to show saddle-avoidance properties of gradient decent dynamics, and extends previous theories to characterize the change in eigenvalues of layer weights.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the 4-layer matrix factorization problem, with regularization, under gradient flow and infinitesimal learning rate gradient descent. It proves the convergence with probability around 1/2 to the global minimum and around 1/2 to saddles, under a special balanced initialization.

### Strengths
The theory seems technically rigorous and the analysis of deeper networks is important even just linear. It extends the results of matrix factorization to deeper layers, although under some strong requirements (see Weaknesses).

### Weaknesses
The analysis relies on a special initialization that seems to reduce the four-layer model to the one with approximately two-layer freedom. It also depends on strong regularization. The writing is a little technicle and lacks high-level intuitive explanations of the terms used in the analysis.

### Questions
- What is $c_1,c_2$?
- What are the requirements on the width $d$?
- The regularization coefficient is very large (polynomial in width). Could the authors comment on which parts of the analysis would fail if this coefficient were reduced or even set to zero? Is the exact balance between weights essential to the proof?
- The initialization seems very special in at least the following two senses. 1) It could with probability at least 1/2 make the dynamics fall into a degenerate submanifold such that GD/GF will converge to saddles, whose stable manifold is a measure-zero set, i.e., very rare case. 2) Adjacent weights share the same randomness up to canonicalization, and all weights share the same Gaussian randomness. This seems to reduce the model to a two-layer-like case. What, then, is special about the 4-layer setting under this initialization? Which parts of the analysis would fail under a more general (e.g., Gaussian for all weights) initialization?
- Reduce to diagonal target: based on my understanding, the dynamics is nonlinear and the transformation is not symmetric on all the four factors.  Also, it seems the analysis relies on the alignment between $U$ and $V$ as stated around line 280-290.  Please show this reduction does not affect the GD trajectory, beyond the isotropic matrix case.
- Please explain the origin or intuition behind the term $W+(WW^H)^{1/2}$ in the analysis
- Please explain why terms like $(U\pm V)\Sigma$ is useful in the convergence analysis instead of something like $U\Sigma\pm \Sigma V^\top$
- The analysis seems to rely a lot on even layers. What is the reason of bounding even layers and the thresholds of extending it to odd layers? Is this due to the initialization? Also, does this work for deeper layer? If not, what is the threshold of the extension?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors study convergence of full-batch gradient descent (GD) on 4-layer matrix factorization with random initialization, i.e., GD on the loss

$\mathcal L (W_1,W_2,W_3,W_4) = 1/2 \Vert W_4W_3W_2W_1 - \Sigma \Vert_F^2 +  \mathcal{L}_{reg} (W_1,W_2,W_3,W_4)$

To prove convergence in this setting they let $\mathcal{L}_{reg}$ be a balancedness-regularization.

Their main result, Theorem 1, shows convergence of GD to a global minimum in polynomial time (depending on the ambient dimension and the operator norm of $\Sigma$).

### Strengths
+ Proving global convergence of GD in the general $L$-layer case is an interesting problem

### Weaknesses
- Quality of the writing makes it really hard to parse statements
- Quantities are used before they are defined
- Relevant literature is not taken into account
- 4-layer case with balancedness regularization is a very special setting and main theorem only applies to $\Sigma$ with flat singular spectrum, i.e., all singular values are identical

### Questions
I do not recommend this paper to be accepted. There are several reasons for this decision:

1.) Due to the noticeable low quality of writing, it is very difficult to fully parse mathematical statements and to check their correctness. Quantities are often used before they are defined, e.g., $\Delta_{j,j+1}$ used in line 161, but defined in line 187, or  balanced Gaussian initialization used in line 207, but defined in lines 221ff

2.) Relevant literature such as 

Nguegnang, G.M., Rauhut, H. & Terstiege, U. Convergence of gradient descent for learning linear neural networks. Adv Cont Discr Mod 2024, 23 (2024). https://doi.org/10.1186/s13662-023-03797-x

has not been taken into account. This makes it hard to assess the novelty of the presented results.

3.) The main result only applies to a very restricted special case (4-layer, identical singular values of $\Sigma$, $\mathcal L$ including balancedness regularization). 

Since the combination of these points requires substantial changes, I strongly recommend a thorough revision of the full manuscript and extension of the results before resubmission.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides a rigorous polynomial-time global convergence analysis for gradient descent on a four-layer linear network (matrix factorization), given the inclusion of a standard balanced regularization term and the condition that the target matrix has identical singular values. Previous results addressed only two-layer networks or required balanced initialization; this work achieves the first provable global convergence guarantee for a deeper architecture (N=4) under small random Gaussian initialization. The proof proceeds through a novel three-stage decomposition: alignment, saddle avoidance, and local convergence. The authors introduce two monotonic quantities (a non-increasing skew-Hermitian error and a non-decreasing Hermitian main term) that track training progress and ensure singular values remain bounded away from saddle points. The result holds with high probability in the complex setting and with probability close to 1/2 in the real setting.

### Strengths
1. Strong theoretical novelty: This work establishes the first polynomial-time global convergence guarantee for gradient descent on a deep matrix factorization problem (N>2) under random initialization, specifically targeting the four-layer (N=4) architecture. This result addresses a longstanding open question regarding general deep matrix factorization and Deep Linear Networks. The paper is technically impressive and represents a meaningful step forward in understanding deep gradient dynamics.

2. High mathematical rigor and completeness: The paper exhibits high mathematical rigor, supported by detailed proofs that are presented in full in the Appendix, allowing a reader to verify each step independently. The analysis requires complex technical developments, including the utilization of tools from random matrix theory, such as the Circular Ensembles, for analyzing the minimum singular value at initialization. Furthermore, to rigorously translate continuous-time intuition (Gradient Flow) into guarantees for the discrete Gradient Descent algorithm, the authors developed new perturbation bounds for eigenvalues. These auxiliary lemmas and novel technical approaches are considered to be of independent interest for deep linear networks analysis.

3. Conceptually structured framework: The three-phase view of training provides an intuitive narrative for how deep networks self-align. The construction of the skew-Hermitian and Hermitian terms offers a clean way to formalize this intuition, and could serve as a basis for analyzing even deeper or nonlinear architectures.

### Weaknesses
1. Limited scope and overstated generality: The analysis applies specifically to the four-layer case with identical singular values of the target matrix. The extension to general depth or non-isotropic targets is only conjectural. The paper’s framing (“global convergence for deep networks”) slightly overstates the reach of the results.

2. Dense presentation and navigational difficulty: Despite occasional intuitive remarks, the exposition remains heavy, with long theorem sequences spanning most of the paper. Assumptions vary subtly across theorems, and the logical dependencies are hard to track. A table summarizing the different assumptions made in each theorem and their dependencies would help improve readability. I also suggest moving some of the technical results (especially those that serve primarily as supporting bounds or spectral perturbation lemmas) to the appendix would free space for a higher-level discussion of the main theorems and their implications. This would help readers grasp the overarching logic without losing rigor and would also make room for intuitive commentary or schematic illustrations of the convergence stages.

3. Notation overload: The paper suffers from significant notation overload and inconsistency, which poses a challenge to readability despite the rigorous content. For instance, $\delta$ serves both as a failure probability parameter and as the minimal singular value in theorem 7. M serves as a scalar bound for maximum singular values and again as a placeholder for a matrix sum (M=S+D) in perturbation lemmas in the appendix.

### Questions
1. The discrete-time bounds include a higher-order term $a^2 \eta^2$ that appears to control the deviation between gradient flow and gradient descent. However, $\eta$ is defined as a function of $a$. Could the authors clarify the intended scaling between the initialization magnitude $a$ and step size $\eta$? 

2. Some results (e.g., the main convergence theorem) explicitly assume $N=4$, while other constants and inequalities treat $N$ as variable. Could the authors clarify which theorems hold only for the four-layer case and which, if any, extend to general even depth? A brief discussion of whether the proof technique could plausibly generalize beyond N=4 would be very helpful.

3. $\delta$ denotes both a failure probability parameter and the minimal singular value in theorem 7. Could the authors clarify whether these are distinct quantities and consider using distinct notation or a notation summary to reduce ambiguity?

4. The paper establishes a deep linear convergence result under strong structural assumptions. Could the authors elaborate briefly on what aspects of the proof they believe might carry over to nonlinear or more general architectures?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies gradient flow under the balancedness condition and gradient descent without the balancedness assumption in a four-layer linear network, proving global convergence under random initialization. The authors consider both complex-valued and real-valued matrices and show that, with high probability under complex initialization and with probability $1/2$ under real initialization, the product matrix converges to a global minimum in polynomial time. The paper provides a proof sketch that proceeds through three distinct phases: first, the layers become balanced; second, the minimum eigenvalue of the Hermitian term remains positive; and finally, the loss decreases monotonically to zero.

### Strengths
- The paper is well-organized and, although technically challenging, is written in a way that makes the content as accessible as possible to readers. The problem settings and notations are clearly presented, which helps readers follow the subsequent sections. In addition, Section 4 (Gradient Flow under Balanced Gaussian Initialization) effectively serves as a warm-up, while Section 5 (Gradient Descent under Unbalanced Gaussian Initialization) presents the main results and provides a well-structured proof sketch.
- This paper is the first to provide results on four-layer matrix factorization under random Gaussian initialization beyond the NTK regime, which represents a significant contribution.It presents important and novel findings not only for the analytically tractable (though nontrivial) gradient flow case but also, building on this foundation, for gradient descent under unbalanced initialization, offering a comprehensive analysis of both settings.

### Weaknesses
- The paper lacks a clear explanation of the convergence rate derived in Theorems 1 and 2. A more detailed discussion of this rate, including how tight the bound is, would strengthen the analysis. It would also be helpful to compare the convergence rate with that of the depth-2 case (e.g., [1]). In addition, although the authors note that it is difficult to analyze odd factorizations theoretically, it would still be valuable to include an empirical comparison of the convergence behavior across depth-2, depth-3, depth-4, and even deeper networks.
- The paper does not demonstrate the incremental learning of singular values in matrix factorization, where the model gradually learns features one by one. This limitation may stem from the assumption that all target singular values are equal in most theorems. However, Figure 1 (identity target) suggests that, possibly due to random initialization, such incremental learning behavior still emerges. Could the authors provide a theoretical explanation or justification for this phenomenon under their setting?
- The paper does not address cases with different target singular values or rank-deficient targets. While this omission is understandable given the analytical difficulty, it would be helpful if the authors could briefly explain what makes these cases challenging to analyze.

[1] Global Convergence of Gradient Descent for Asymmetric Low-Rank Matrix Factorization

### Questions
- Can the analysis be extended to deeper (even) linear networks? If not, please clarify the main obstacles or limitations that prevent such an extension.
- It is also unclear why the complex domain is considered alongside the real domain. Typically, analyses are conducted only in the real domain, so it would be helpful to explain the specific motivation or advantage of including the complex case.
- Could the authors clarify why the convergence rate and initialization scale involve the term $\sigma_1(\Sigma)^{1/4}$? In particular, and how does it compare to the two-layer results?

### Soundness
3

### Presentation
3

### Contribution
4
