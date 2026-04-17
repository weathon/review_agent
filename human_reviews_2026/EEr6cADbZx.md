# Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 10

## Abstract
Matrix factorization mechanisms for differentially private training have emerged as a promising approach to improve model utility under privacy constraints. In practical settings, models are typically trained over multiple epochs, requiring matrix factorizations that account for repeated participation. Existing theoretical upper and lower bounds on multi-epoch factorization error leave a significant gap. In this work, we introduce a new explicit factorization method, Banded Inverse Square Root (BISR), which imposes a banded structure on the inverse correlation matrix. This factorization enables us to derive an explicit and tight characterization of the multi-epoch error. We further prove that BISR achieves asymptotically optimal error by matching the upper and lower bounds. Empirically, BISR performs on par with the state of the art factorization methods, while being simpler to implement, computationally efficient, and easier to analyze.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the problem of adding correlated noise to elements in a data stream with a momentum structure in a differentially private manner under continuous observation, while maintaining low approximation error. The authors propose a novel factorization mechanism for injecting correlated noise at each iteration, providing formal differential privacy guarantees even against an adversary with access to all entries of the stream.

Their approach improves upon existing methods by factorizing the workload matrix, extending the square-root factorization framework studied in prior work. This refined factorization yields tighter approximation error bounds. Furthermore, the authors establish matching lower bounds on the achievable approximation error for general classes of stream structures, demonstrating the optimality of their approach.

The proposed mechanism has clear practical relevance, particularly for differentially private stochastic gradient descent (DP-SGD) with momentum. It enables more efficient and accurate private training of deep learning models, highlighting the work’s potential impact on privacy-preserving machine learning.

### Strengths
This paper is very strong in terms of results:
1. The paper presents rigorous theoretical results based on novel techniques that effectively close a significant gap in the existing literature on matrix factorization mechanisms within the context of streaming differential privacy covering a large variety of tasks.

2. The paper synthesizes concepts from several related domains to develop a generalized formulation of DP-SGD with momentum, extending beyond existing approaches in the literature.

3. The paper proposes a computationally efficient method to compute correlated noises. It provides a practical technique to implement these correlated noise techniques for settings which are not simple SGD and attempt to break into the practical scenario in existing optimization techniques used for differentially private deep learning.

4. Practically the algorithm performs very well compared to existing matrix factorization methods for DP-SGD on classification datasets.

### Weaknesses
The main weakness of this paper lies in its presentation. While the technical results appear sound and potentially impactful, the exposition makes it difficult for readers to fully grasp the scope and implications of the work (especially from the lower bound perspective). Below are some specific comments and suggestions:
1. *Dependence on Prior Work for Context*: The paper relies heavily on Kalinin & Lampert (2024) to set up the background for its lower bound results. For instance, the “multi-participation setting” introduced in Theorem 3 is not defined anywhere in the paper and an appropriate reference has not been given as well, which makes it hard to understand the exact conditions under which the stated bounds apply.

*Suggestion*: It would be helpful if the authors could briefly explain or restate the definition of the multi-participation setting, even if it was introduced in prior work. A short contextual description would make the paper more self-contained and easier to follow.

2. *Ambiguity in Notation*: Some notations are not clearly introduced. In particular, the terms $\Omega_\alpha$ and $\Omega_{\alpha, \beta}$ in Theorem 3, along with their Big-O counterparts in Theorem 4 and Corollary 1, are not defined.

*Suggestion*: Please clarify the meaning of these notations, either directly in the text or in a concise notation table.

Including short background explanations—perhaps in an appendix if not in the main text—would greatly improve the paper’s readability. While it is understandable that some results depend on established frameworks, providing minimal context would help readers appreciate the contributions without needing to refer extensively to other works.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
1

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a new algorithm banded inverse square root (BISR), which is an extension to BSR from (Kalinin & Lampert 2024) by imposing a banded structure on the inverse matrix $C^{-1}$ rather than $C$ itself.

By making such modification, the authors are allowed to give a more explicit expression error with respect to the bandwidth $p$ and reduce the overall computational complexity. The authors also show a tight error bound on the approximation error with matching lower bounds.

### Strengths
I find the idea of bounding the bandwidth of matrix on matrix $C$ to reduce computational complexity interesting and the optimal error bound a solid contribution. Also, I appreciate the authors' efforts make to make a comprehensive discussion and comparison with prior work.

### Weaknesses
1. The algorithmic modification are relatively minimal. The general structure of the algorithm directly follow that in (Kalinin & Lampert 2024) which slightly weaken the contribution of this work.
2. More discussion is needed on the approximation error defined in equation (1) . In particular, how is this error related to the convergence error? Would achieving an optimal bound on this error also imply an optimal convergence rate? It would be great if the authors can provide explicit convergence rates for certain loss function classes (e.g., convex-Lipschitz loss) and make comparisons with existing DP algorithms like DP-SGD.
3. The algorithm, claimed as computational efficient, may not be suitable for modern model training. To achieve optimal rate, $p$ needs to be set as $\tilde{O}(b)$ where $b$, as far as I understand, can be approximated as the number of update steps per epoch. In large-scale training, this number can reach tens of thousands or even millions. Therefore, combining $p$ different $Z_i$'s in each update step could become prohibitively expensive in practice.

### Questions
Questions are included in the "Weaknesses" section.

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
The paper investigates the matrix factorization mechanism for differentially private stochastic gradient descent (DP-SGD) under multi-epoch/multi-participation settings.
The matrix factorization mechanism is used to inject correlated noise into gradients during training by means of a correlation matrix.
The authors propose to enforce a banded structure on the _inverse_ of the correlation matrix, to derive explicit upper bounds on factorization errors and to prove asymptotic optimality.
The authors compare their approach with existing techniques, showing that their proposed approach achieves higher or comparable accuracy for large matrices.
Moreover, the paper proposes an efficient, low-memory method which matches the performance of state-of-the-art approaches while being more efficient.

### Strengths
* **Theoretical contribution.**
The paper introduces and discusses a matrix factorization technique with provable optimality, and refines prior existing bound (Kalinin and Lampert, 2024).
Unlike related work, you provide an explicit dependence on the bandwidth $p$ and on the participation $b$, which leads to more useful guarantees.
The idea to consider the inverse correlation matrix instead of the matrix itself is, as far as I can tell, novel and elegant.
Moreover, the discussion on an efficient implementation reinforces the practical utility of your approach.
The problem discussed is very relevant and well placed within related literature.

* **Empirical validation.**
The empirical validation you present is limited but consistent, and qualitatively supports your claims.
In particular in low-resource regimes, the presented approach performs better than existing ones.

### Weaknesses
* **Clarity and accessibility.** The paper has dense notation and long proofs: intuition could be introduced earlier.
For instance, the benefits of inverse banding are not intuitively clear, and visualizations could help here.

* **Low privacy regime.**
In your empirical evaluation, you only present results in a arguably low privacy regime $\epsilon=9$.
While this specific value for the privacy budget seems to be common in related literature, it is generally understood to be at the edge of what is considered to be differentially private at all.
If my understanding is correct, the benefits of one specific factorization technique against another, is particularly important when the amount of noise added is small.
While this justifies the chosen privacy regime, it may diminish the contribution for more strict, and therefore relevant from a privacy perspective, privacy regimes ($\epsilon <= 1$) where DP constraints are more meaningful.

* **Empirical relevance.**
Following from my previous point, I am not convinced of the practical relevance of the approach.
While I understand that your contribution is, first of all, theoretical, a comparison with more recent DP-SGD variants would further justify its practical relevance.
The plots do not show any measure of dispersion (e.g., error bars).
The empirical setup does not seem to reflect a real-world application with realistic privacy requirements, and no investigation of the effectiveness of the approach with different privacy budgets is performed.
From the plots, the correspondence between RMSE and accuracy is difficult to grasp.

### Other remarks

* The nomenclature is at times confusing.
Both $C$ and $C^{-1}$ are referred to as "correlation matrix" throughout the introduction (e.g., compare lines 45 and 59/61).

### Questions
* How practically (or theoretically) relevant is your approach for stricter privacy regimes, i.e., small values of $\epsilon$?
* What is the standard deviation/variability of the results reported in Figure 1/2? Are the results significantly different if you include error bands in the plots/results?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper introduces and analyzes a novel matrix factorization scheme, called BISR, which can be applie to DP-SGD and relies on (i) computing the inverse square root of the workload's matrix, (ii) imposing a band structure on it, and (iii) inverting the resulting matrix again.
A novel lower bound on factorization error for SGD's workload matrix is derived, with matching upper bound for the BISR method, which relies on band structure of the inverse square root.
Numerical experiments show that the resulting factorizations are consistently on par than the existing BSR mechanism, and strongly outperform it for some choices of bandwidth.

### Strengths
1. The proposed method, relying on imposing structure on the inverse of SGD's workload matrix squared root, is original and very different from existing ideas.
2. A refined theoretical lower bound on the factorization of SGD's workload matrix is provided.
3. The BISR method is shown to match this upper bound.
4. BISR is shown numerically to provide better factorization in many setting that existing methods, and this improved factorization precision results in improved accuracy over existing method in multiple private machine learning tasks.
5. The paper is very well written and easy to read.

### Weaknesses
The paper is very interesting, and I mostly remark strengths about the contributions. Nonetheless, there are some minor weaknessses:
1. No theoretical guarantees for DP-SGD under the BISR matrix factorization are provided.
2. While the theoretical claims hint for a large improvement over the BSR method, this does not always show in practice; studying more precisely (i.e., non-asymptotically) the respective behaviour of the two methods may reveal more subtle compromises.
3. The experiments on CIFAR-10 and IMDB are performed in the small bandwidth and low privacy regime: it is not clear whether the same conclusions would hold in high-privacy/large bandwidth regimes.

### Questions
1. Is it possible to derive theoretical convergence guarantees for DP-SGD in simple settings (e.g., strongly-convex functions)? In this setting, is there a chance to observe the true metrics on matrix factorization approximation that impact the final privacy-utility trade-off?
2. Authors claim that the RMSE may not be a good proxy for approximation error. Are there other candidates for better proxies?
3. Experiments showcase the low privacy regime: how would the result change in a high privacy regime? Would BISR still be better? Would the difference increase/decrease?
4. What is the intuition why the inverse square root should be closer to a band structure than the square root itself?

### Soundness
4

### Presentation
4

### Contribution
4
