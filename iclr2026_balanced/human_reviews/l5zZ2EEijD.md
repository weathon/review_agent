## Human Reviewer 1

### Summary
The paper studies the classic ℓp regression problem, which generalizes least squares (p = 2) and appears in robust regression, clustering, and semi-supervised learning. The authors propose a new Iteratively Reweighted Least Squares (IRLS)–based algorithm for ℓp regression that: Achieves the same theoretical iteration bound as the complex SODA 2019 / JACM 2024 algorithm of Adil–Kyng–Peng–Sachdeva,
But retains the simplicity and practical efficiency of the p-IRLS (NeurIPS 2019) method.

### Strengths
The IRLS method is widely used in practice but typically lacks strong theoretical guarantees. This paper proposes a variant that offers both rigorous theoretical foundations and favorable computational efficiency. The authors reformulate IRLS within a primal–dual invariant framework, deriving the update rule from a monotonic invariant of the dual objective. The algorithm adaptively rescales weights to preserve this invariant, achieving provably fast convergence for all values of p. The proposed approach is novel and potentially impactful beyond this specific setting, given the broad applicability of IRLS across various domains.

### Weaknesses
There should be more discussion on the intuition behind why this reformulation leads to improved convergence, as the underlying mechanism is not immediately clear—especially in contrast to the standard IRLS method, which is more intuitive.

The case of p<1 is particularly important due to its relevance in applications such as robust regression. However, this scenario is not addressed in the main paper and is only briefly mentioned in Appendix C, where the conclusions remain unclear. Given its significance, this case should be discussed in the main text, with the key findings and implications explicitly stated.

### Questions
The authors should include a comparison with existing IRLS algorithms and their variants, along with an explanation of why the proposed approach outperforms these methods.

The paper presents two versions of the algorithm—one for the low-accuracy regime and another for the high-accuracy regime. A discussion on their practical usage would be valuable. In particular, it would be helpful to clarify under what conditions each version is preferable, and whether a hybrid strategy (e.g., using the low-accuracy regime for initialization followed by the high-accuracy regime for refinement) would offer additional benefits.

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper presents new algorithms for solving ℓₚ regression problems using the Iteratively Reweighted Least Squares (IRLS) framework. The proposed methods achieve state-of-the-art iteration complexity while maintaining practical simplicity.

### Strengths
The paper establishes a solid theoretical foundation by matching the best-known asymptotic complexity of Adil et al. (JACM 2024) with a simpler iterative structure. Its primal-dual, invariant-based design offers a clean and principled acceleration of IRLS, effectively bridging theory and practice through provable guarantees and strong empirical performance. Experiments on synthetic and real datasets confirm notable gains in speed and accuracy, supported by clear and well-organized mathematical exposition.

### Weaknesses
None

### Questions
The algorithm’s performance under modern hardware acceleration is not discussed. I'm more interested in the parallel algorithm.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 3

### Summary
In this paper, the authors present an algorithm for solving $\ell_p$ regression problems using the IRLS method. The algorithm is based on a clever update rule motivated by the dual formulation of an equivalent problem, and it achieves better complexity than other versions. In particular, the number of calls to the linear system solver matches the state-of-the-art theoretical order.

### Strengths
The paper is very well written in general. Although my expertise lies far from this area, I was still able to understand the main ideas and appreciate the contributions. The experimental results are very good.

### Weaknesses
This is not a weakness of the paper itself, but rather of this type of work in general. I believe that one of the strongest contributions of the paper lies in Theorems 1.1 and 1.2, but the proofs are relegated to the appendix (and I did not check them). I have always found that such papers are more appropriate for journal publication, where these key aspects can be better appreciated.

### Questions
I don't understand why to include the number of iterations of cvx sedumi/SDPT3 in Fig. 2. The nature of the iterations is quite different. I think it would be better to include them only in the time comparison.

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
2

---

## Human Reviewer 4

### Summary
The paper gives an algorithm for the p-norm regression problem for $p> 2$. The known algorithms include the algorithm by AKPS’ JACM 2024 which gives the best theoretical guarantees but a complicated algorithm, and a practical algorithm by APS’19 which outperforms all existing implementations at the cost of a worse iteration complexity. This paper bridges the gap by giving an algorithm that matches the guarantees of AKPS as well as outperforms the algorithm by APS’19. 

Their algorithms is an IRLS style algorithm similar to the approach of APS’19 but their algorithm and analysis are quite different. The analysis follows a more primal dual approach similar to the works of Ene-Vladu ICML’2019. They first use an approach similar to EV’19 to give a low accuracy solver and then use the iterative refinement framework from AKPS to boost the accuracy. In order to do so they require modifying their algorithms to work for a regularized version of the problem.

### Strengths
1. Interesting theoretical result which can be of independent interest.
2. Bridges the gap between theory and practice for this problem.
3. They also test their algorithm on some real-world data sets which I dont think has been done before for this problem.

### Weaknesses
See questions. No major weaknesses!

### Questions
1. The algorithm looks more complicated than APS’19 and also of a similar level of complexity as AKPS JACM’24. It looks like a different approach, and maybe the authors could say something about the fundamental difference with previous works. I am not sold on the simplicity of the algorithm, regardless I think it is interesting as a standalone algorithm.
2. Also the difference in iteration counts in the experiments seems to be some fixed constant factor of maybe 2 or something. I am wondering if this is because APS’19 is not implemented well enough or this algorithm is genuinely faster due to the improved rates of convergence?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
5