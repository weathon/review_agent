# Computing Exact Shapley Values in Polynomial Time for Product-Kernel Methods

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Kernel methods are widely used in machine learning due to their flexibility and expressiveness. However, their black-box nature poses significant challenges to interpretability, limiting their adoption in high-stakes applications. Shapley value-based feature attribution techniques, such as SHAP and kernel method-specific adaptation like RKHS-SHAP, offer a promising path toward explainability. Yet, computing exact Shapley values is generally intractable, leading existing methods to rely on approximations and thereby incur unavoidable error. In this work, we introduce PKeX-Shapley, a novel algorithm that utilizes the multiplicative structure of product kernels to enable the exact computation of Shapley values in polynomial time. The core of our approach is a new value function, the *functional baseline value function*, specifically designed for product-kernel models. This value function removes the influence of a feature subset by setting its functional component to the least informative state. Crucially, it allows a recursive thus efficient computation of Shapley values in polynomial time. As an important additional contribution, we show that our framework extends beyond predictive modeling to statistical inference. In particular, it generalizes to popular kernel-based discrepancy measures such as the Maximum Mean Discrepancy (MMD) and the Hilbert–Schmidt Independence Criterion (HSIC), thereby providing new tools for interpretable statistical inference.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a novel value function for feature attribution values via Shapley values. The proposed value function is model-specific for product-kernels, i.e. it is defined based on the internal model structure instead of predictions of chosen data points, similar to methods like path-dependent TreeSHAP. The authors demonstrate that for this value function the Shapley values can be computed efficiently in $\mathcal O(d^2)$ time. The key observation is that due to the product-kernel structure, the marginal contributions $\nu(S \cup \{j\}) - \nu(S)$ can be factored out, such that the remaining common terms need to be summarized. The authors provide a recursive algorithm for this, and further extend this result to kernel-based statistical measures, namely MMD and HSIC. The authors empirically show that exact computation outperforms approximation via KernelSHAP, and their chosen value function extract synthetic ground-truth feature importance values. Finally, they illustrate the MMD explanation, and showcase an application within feature selection.

### Strengths
- The baseline value function is well motivated, and occurs naturally in product-kernels.
- The paper solves model-specific computation of Shapley values for product-kernels, which is non-trivial
- The results can be directly extended to other kernel-based statistics

### Weaknesses
- All figures seem to be missing in the paper, so I cannot judge their quality and validate the empirical results. In this current state, I cannot accept the paper. From the theoretical contribution, this works seems interesting and worth accepting.
- A similar "trick" was used in TreeSHAP, since the tree-based value function is also a sum (over leaves) of products of (weighted) decision rules. A comparison to efficient methods for trees, or a comparison between these two approaches would be desirable to grasp the common aspects of both computation methods, see e.g. [1].


[1] Bifet, Albert, Jesse Read, and Chao Xu. "Linear tree shap." Advances in Neural Information Processing Systems 35 (2022): 25818-25828.

### Questions
- Is there a direct link from product-kernels and your value function to e.g. interventional or path-dependent perturbations used in TreeSHAP? It seems that product-kernels can be viewed as a special case of the value function in [1]. How does the computation differ?
- Can the current result be generalized to value functions that admit a form $\nu(S) = \sum_q \nu_q(S)$, where $\nu_q$ has the property: $\nu_{q}(S \cup \{i\}) = b(\{i\}) \nu_q(S)$ for some common factor $b(\{i\})$? In your case, $q$ seems to be the data points, whereas for TreeSHAP $q$ is the leaf nodes. Do there exist other methods, where these types of value functions occur? How does your computation differ from computations used by TreeSHAP applied to this general case?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper considers the problem of computing Shapley values for a value function $v$ defined on $d$ elements. For an input $\mathbf{x}$, they introduce a *product kernel* value function that describes the prediction when only given the features in $S$. They then use the product structure to *exactly* compute the Shapley values in polynomial time in $d$. (Estimating the Shapley values is less accurate, and a naive, exact computation would take exponential time.) They also propose value functions for the *maximum mean discrepancy* and *Hilbert-Schmidt Independence Criterion*, and apply similar ideas to exactly compute the Shapley values.

They run experiments on how *useful* the Shapley values of their value functions are to the more standard value functions. However, none of the figures are actually present in the submitted PDF.

### Strengths
* Computing Shapley values is a very popular problem. For general value functions, we generally need approximation methods which can be imprecise. Exactly computing Shapley values is always preferable, but we require specific structure in the value function for exact computation to be fast. Therefore, their contribution of a (reasonable) value function, with fast exact algorithms is quite nice.

* I appreciate the experimental comparison of how useful the Shapley values of this value function are. I'll give the benefit of the doubt and *assume* the missing figures show your approach is comparable to others. But please corroborate with the figures in the rebuttal.

* The key insight of their approach is that the *product* structure allows them to factor $v(S \cup \{i\} ) - v(S)$. This is a nice (simple) insight, and I appreciate how well explored this is in your paper e.g., considering the technique applied to MMD, HSIC, and experiments.

### Weaknesses
Please see the questions I ask below. If inadequately addressed in the rebuttal, these could be weaknesses.

Minor:

* You're missing the figures. Please upload a new PDF that includes them.

* The name "PKeX" is quite ugly. Can you change this to something nicer please?

* When introducing MMD and HSIC, you do so in a paragraph or so with lots of in line equations. Can you please make these sections nicer by a) putting each equation on its own line so it's easier for the reader to parse, and b) adding more discussion?

* Your Experiment 1 on computing the shapley values via your method and estimating them via Kernel SHAP is quite silly to me. If you're using your method as the ground truth (this is totally fine because you *show* your approach exactly recovers shapley values), of course you'll get much better accuracy than Kernel SHAP. But, I guess some readers might want to see this clearly explored.

* There are several typos. Maybe run the whole main doc through an LLM and ask it to list them for you.

### Questions
When substituting the "least informative information", could you please justify why 1 is the right choice?

For an elementary symmetric polynomial, how is the "base case" $e_0$ defined?

Could you please provide a time complexity analysis of your recursive algorithm? I see you claim it's quadratic in the conclusion, but I'd like more details please.

If the figures are included and my questions are adequately addressed, I will increase my score to a 6 or 8.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Overall, this work targets an important issue with SV computation and provides theoretical guarantees; however, the results are not persuasive enough due to the missing figures and other issues (see my comments below).

All five figures in the main text of the submitted paper are blank, which significantly impacted my ability to fully understand and evaluate the work, and this flaw is reflected in my scores. Therefore, I can only provide feedback based on the current text and cannot guarantee my comments are 100% aligned with the authors' original intention. Still, I hope the following suggestions are helpful.

### Strengths
The math notations are clear, well-written, and easy to follow. The integration of MMD and HSIC provides the possibility for future non-prediction tasks (statistical inference) and may be a foundation for future research. Both synthetic and real-world datasets are used, and several baselines are claimed to be compared.

### Weaknesses
1. Although the authors claim that PKeX-Shapley is solvable in polynomial time, it would be beneficial to see a formal analysis of its runtime complexity in big-O notation, in terms of the feature dimension and other relevant parameters.

2. The core idea of this paper is the novel value function defined in Equation 3. To fully validate this proposed function, its properties should be discussed more thoroughly. In addition to the additivity property (Lemma 4), its adherence to other key Shapley axioms, such as Symmetry and the Dummy Player property, should also be formally demonstrated.

3. PKeX-Shapley is a model-specific method for computing Shapley values, which limits its application to kernel-based machine learning models. Consequently, it is not suitable for CV and NLP tasks, where neural networks are the primary predictors and traditional sampling methods remain applicable. Future work should focus on addressing this limitation.

4. In Experiment 1, the authors use their exact PKeX-Shapley result as the 'ground truth' to evaluate sampling-based methods. However, this only measures the sampling error relative to their own value function. A more informative comparison, especially for the synthetic cases, would be to compute the ground-truth Shapley values by brute-forcing the definition in Equation 1 with a standard, model-agnostic value function. The comparison should then be based on this 'true' agnostic ground truth.

### Questions
In the definition of the value function in Equation 3, the authors claim that they remove a feature’s influence by factoring out its corresponding functional component and setting it to one. Can this be viewed as a special imputation strategy in SHAP? This is relevant since Equation 3 is also a kind of prediction game in terms of the product-kernel model. If we just impute the removed features with a fixed value, a lot of computation time can also be saved in SHAP or KernelSHAP. On the other hand, the conditional distribution can be quickly estimated by a surrogate model, i.e., a neural network (see FastSHAP). Could the authors better illustrate this gap with theory or experiments?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper studies how to compute the Shapley value, where the value function is defined in Eq. (3), for feature attribution when a product kernel is employed. The paper proposes a polynomial-time algorithm, as shown in Theorem 3, and further demonstrates that the technique can be straightforwardly applied to the value function defined via either the Maximum Mean Discrepancy (MMD) or the Hilbert–Schmidt Independence Criterion (HSIC), as shown in Propositions 6 and 8.

### Strengths
NA

### Weaknesses
- In Eq. (3), there appear to be two different definitions of $k_{\mathcal{S}}(\cdot, \mathbf{x})$.  Note that it is used to define the value function $v\_{\mathbf{x}}(\mathcal{S})$, whose Shapley value is the main focus of this paper. The first definition is $\Pi_{\mathcal{S}}k(\cdot,\mathbf{x})$ (line 787), which makes proposition 2 hold, where $\Pi_{\mathcal{S}}$ denotes the orthogonal projection onto the subspace $\mathbb{H}\_{\mathcal{S}}$. The second definition is $\prod_{i\in\mathcal{S}} k_i(\cdot, x_i)$, which is the one used in Theorem 3 and Propositions 6 and 8. **The main issue is that the authors implicitly assume that these two definitions coincide, but they do not.**

    - While the result of Theorem 9 is non-trivial in genenel, it is straightforward to see that, for every $ \mathcal{S} \subsetneq \mathcal{D} $, $\mathbb{H}\_{\mathcal{S}}$ defined in Proposition 2 is potentially the trivial zero subspace $\\{\mathbf{0}\\}$ for product kernels. **Consequently, if $k_{\mathcal{S}}(\cdot, \mathbf{x}) \coloneqq  \Pi\_{\mathcal{S}}k(\cdot,\mathbf{x})$, then $v\_{\mathbf{x}}(\mathcal{S}) = 0$ for every $\mathbf{x} \in \mathbb{R}^{d}$ and every $ \mathcal{S} \subsetneq \mathcal{D} $. As a result, the Shapley value of $v\_{\mathbf{x}}$ would be simply $\frac{f(\mathbf{x})}{d}\mathbf{1}\_{d}$, which is meaningless. However, this is not the case when $k_{\mathcal{S}}(\cdot, \mathbf{x}) \coloneqq  \prod\_{i\in\mathcal{S}} k\_{i}(\cdot, x_i) $.**

    - Write $\mathbf{x} \coloneqq (\mathbf{x}_1, \mathbf{x}_2) \in \mathbb{R}^{d_1 + d_2}$ where $\mathbf{x}_1 \in \mathbb{R}^{d_1}$ and $\mathbf{x}_2 \in \mathbb{R}^{d_2}$, and the product kernel is defined as $k(\mathbf{x}, \mathbf{y}) = (\mathbf{x}\_{1}\cdot \mathbf{y}\_{1})(\mathbf{x}_2 \cdot \mathbf{y}_2)  = k\_1(\mathbf{x}\_1, \mathbf{y}\_1) k\_2(\mathbf{x}\_2, \mathbf{y\_2})$. Then, the feature map is $\phi(\mathbf{x}) = \mathbf{x}_1  \mathbf{x}\_{2}\^{\mathsf{T}} \in \mathbb{R}^{d_1 \times d_2}$. The induced reproducing kernel Hilbert space $\mathbb{H}$ of $k$ is isometrically isomorphic to $\mathbf{\Phi} \coloneqq \mathrm{span}\\{\phi(\mathbf{x})\colon \mathbf{x} \in \mathbb{R}^{d_1 + d_2}\\}$. In other words, any $f \in \mathbb{H}$ corresponds to a matrix $\mathbf{F} \in \mathbf{\Phi}$ such that $f(\mathbf{x}) = \langle \mathbf{F}, \mathbf{x}\_1 \mathbf{x}\_{2}\^{\mathsf{T}} \rangle = \mathrm{Tr}(\mathbf{F} \mathbf{x}\_1 \mathbf{x}\_{2}\^{\mathsf{T}}) = \mathbf{x}\_{2}\^{\mathsf{T}}\mathbf{F}\mathbf{x}\_{1}$. Let $\mathbb{H}\_{1}$ be the subspace of $\mathbb{H}$ that contains all $f$ such that $f(\mathbf{x}) = f(\mathbf{y})$ if $ \mathbf{x}\_{1} = \mathbf{y}\_{1}$, which is how $\mathbb{H}\_{\mathcal{S}}$ is defined. Given $f \in \mathbb{H}_1$ and $\mathbf{x} \in \mathbb{R}\^{d\_{1} + d\_{2}}$, there must be $f(x) = \mathbf{z}\^{\mathsf{T}}\mathbf{F}\mathbf{x}_1 \equiv C$ for every $\mathbf{z} \in \mathbb{R}\^{d\_2}$, which leads to $\mathbf{Fx}\_{1} = \mathbf{0}$. Since $\mathbf{x}_1$ also varies, it yields $\mathbf{F} = \mathbf{0}$. In other words, $f(\mathbf{x}) = 0$ for every $\mathbf{x} \in \mathbb{R}\^{d_1 + d_2}$, and thus $\mathbb{H}\_{1} = \\{\mathbf{0}\\}$. This argument holds true for the product kernel $k(\mathbf{x}, \mathbf{y}) = \prod\_{j \in \mathcal{D}}[(2x\_{j}\^{2}-x\_{j})(2y\_{j}\^{2}-y\_{j}) + (4x\_{j} - 4x\_{j}\^{2})(4y\_{j} - 4y\_{j}\^{2})] = \prod\_{j \in \mathcal{D}}k\_{j}(x\_j, y\_j)$, the structure of which is used throughout the paper. The feature map $x \mapsto (2x^2 - x, 4x-4x^2)$ is constructed to have $0 \mapsto (0, 0)$, $0.5 \mapsto (0, 1)$ and $1 \mapsto (1, 0)$, which are sufficient to derive $\mathbb{H}\_{\mathcal{S}}=\\{\mathbf{0}\\}$ for every $ \mathcal{S} \subsetneq \mathcal{D} $. 

    - After all, the authors fail to justify the use of $\prod_{i\in\mathcal{S}} k_i(\cdot, x_i) $ in defining the value functions used throught the paper.


- For $v_{\mathbf{x}}$ defined using $k_{\mathcal{S}}(\cdot, \mathbf{x}) \coloneqq  \prod\_{i\in\mathcal{S}} k\_{i}(\cdot, x_i) $, it has a multiplicative structure, i.e., $v_{\mathbf{x}}(\mathcal{S}) = \prod_{i\in\mathcal{S}} v_i$ where $v_i = \mathbf{\alpha}\^{\mathsf{T}}k_i(\mathbf{X}_i, \mathbf{x}_i)$. **For this kind of value functions, it is already known how to compute the corresponding Shapley value in polynomial time back in 2020, see [1, Algorithm 2]**. 

   - According to [2, Eq. (6)], $f\_{\mathbf{x}}(\mathcal{S}) = \sum_{v\in L(T_{f})} R\_{\mathbf{x}}\^{v}(\mathcal{S})$. By setting some values to zero, it would reduce to  $f\_{\mathbf{x}}(\mathcal{S}) = R\_{\mathbf{x}}\^{v}(\mathcal{S})$. By [2, Eq.(5)], $R\_{\mathbf{x}}\^{v}(\mathcal{S}) = \prod\_{i \in \mathcal{S}}q_i$, from which the multiplicative structure of it is clear. In other words, $v_{\mathbf{x}}$ is just a much simpler case of $f\_{\mathbf{x}}$. 

    - Although the time complexity of [1, Algorithm 2] is $\Theta(LD^2)$ for computing the Shapley value of $f_{\mathbf{x}}$, it reduces to $\Theta(d^2)$ for $v_{\mathbf{x}}$ as $L=1$ in this case. **By contrast, the time complexity of their Algorithm 1 is $\Theta(d^3)$ as it contains three loops (one is abbreviated as a sum), which is clearly worse.**

    - The idea behind [1, Algorithm 2] is very simple. Suppose we have only encountered all players in $\mathcal{S}$ and have computed a vector $\mathbf{p} \in \mathbb{R}\^{|\mathcal{S}|+1}$ such that $p_i = \frac{i!(|\mathcal{S}|+1-i)!}{(\mathcal{S} + 1)!}\sum\_{S\subseteq\mathcal{S}\colon |S|=i}\prod\_{j\in S}v_j$, it costs $\Theta(|\mathcal{S}|)$ time to update $\mathbf{p}$ when a new player is introduced. Also note that this update is inversible. It costs $\Theta(d^2)$ time to have the $\mathbf{p}$ that includes all features. Then, for each feature $i$, it takes $\Theta(d)$ time to remove this feature in $\mathbf{p}$; after that, the sum of $\mathbf{p}$ is exactly the Shapley value for that feature. So, the total time is $\Theta(d^2) + d \times \Theta(d) = \Theta(d^2)$.

- Since the value function defined via either the Maximum Mean Discrepancy (MMD) or the Hilbert–Schmidt Independence Criterion (HSIC) also possesses this multiplicative structure, it is expected that the Shapley value of them can be computed in polynomial time. In particular, their proposed algorithms do not offer any advantages compared to the existing techniques in terms of time and space complexity. 

Overall, this paper offers no useful contributions to the study of Shapley value computation, and the theoretical development is problematic. 

**References**

[1] Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., ... & Lee, S. I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature machine intelligence, 2*(1), 56-67.

[2] Yu, P., Xu, C., Bifet, A., & Read, J. (2022, November). Linear TreeShap. *In Proceedings of the 36th International Conference on Neural Information Processing Systems* (pp. 25818-25828).

### Questions
Please refer to the weaknesses.

### Soundness
1

### Presentation
3

### Contribution
1
