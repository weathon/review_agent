# Federated ADMM from Bayesian Duality

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
We propose a new Bayesian approach to generalize the federated Alternating Direction Method of Multipliers (ADMM). We show that the solutions of variational-Bayesian (VB) objectives are associated with a duality structure that not only resembles the structure of ADMM's fixed-points but also generalizes it. For example, ADMM-like updates are recovered when the VB objective is optimized over the isotropic-Gaussian family, and new non-trivial extensions are obtained for other exponential-family distributions. These extensions include a Newton-like variant that converges in one step on quadratic objectives and an Adam-like variant that yields up to 7% accuracy boosts for deep heterogeneous cases. Our work opens a new Bayesian way to generalize ADMM and other primal-dual methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel method that combines Bayesian variational inference with the Alternating Direction Method of Multipliers (ADMM) for federated learning scenarios. Based on the concept of "Bayesian Duality," the authors extend traditional parameter optimization to distribution optimization, developing the BayesADMM algorithm. This method not only encompasses standard federated ADMM as a special case (when using fixed-variance Gaussian distributions) but also naturally extends to more complex distributions, producing update rules similar to Newton's method. The authors further derive the IVON-ADMM variant, a computationally efficient implementation suitable for deep learning models. Experimental results show that this method outperforms existing baselines on multiple datasets (MNIST, FashionMNIST, and CIFAR-10), particularly excelling in heterogeneous data distribution scenarios.

### Strengths
1. The paper proposes a novel theoretical framework that integrates Bayesian variational inference with the ADMM algorithm, providing a unified perspective for federated learning. This approach of connecting optimization algorithms with probabilistic inference has theoretical depth.
2. The authors clearly demonstrate how traditional federated ADMM serves as a special case of BayesADMM under specific conditions. This theoretical connection provides a new perspective for understanding existing methods and naturally leads to more powerful extensions.
3. The implementation details of IVON-ADMM showcase a practical and efficient algorithm variant. Compared to existing federated learning optimizers (such as FedAvg and FedAdam), it incurs limited additional computational overhead while delivering significant performance improvements.
4. The experimental design is comprehensive, covering both homogeneous and heterogeneous data distribution scenarios, and validates the method's effectiveness across multiple standard datasets. Particularly, Figure 3(b) demonstrates BayesADMM's property of converging in just one communication round for certain loss functions, which is a compelling empirical result.
5. The authors situate their method within the broader context of optimization and Bayesian inference literature, clearly highlighting connections and distinctions with prior works such as Partitioned Variational Inference (PVI) and Bregman ADMM.

### Weaknesses
1.Although the paper claims BayesADMM is theoretically superior, it lacks detailed analysis of computational complexity. Specifically, how do the algorithm's computational and communication costs scale with model parameter size when using more complex distributions (such as full-covariance Gaussians)? This is crucial in practical federated learning scenarios.
2.The experimental section only reports average performance without providing standard deviations or statistical significance tests, making it difficult to assess result reliability. Given the stochastic nature of federated learning (such as client selection and data partitioning), such statistical analysis is particularly necessary.
3.Table 4 shows IVON-ADMM's performance across multiple rounds but does not analyze the trade-off between convergence speed and communication rounds. In practical applications, early stopping may be more useful, but the paper does not explore performance comparisons of different methods under limited communication budgets.
4.While the authors mention the concept of "Bayesian Duality," their explanation of its theoretical foundation is not sufficiently deep. The derivation of the BayesADMM algorithm in Section 3.3 is relatively brief, and certain key steps (such as the transition from Equation 26 to 27) lack detailed explanation, which may affect reader comprehension.
5.The paper does not adequately discuss the impact of hyperparameter selection, particularly the sensitivity of regularization parameter ρ and temperature parameter τ. Although the appendix mentions hyperparameter search, it lacks systematic analysis of how these parameters affect algorithm performance.
6.The comparison with recent state-of-the-art federated learning methods (such as Scaffold and FedProx) is not comprehensive. While the paper compares with FederatedADMM and BregmanADMM, these methods are relatively outdated, and more comparisons with recent works should be included.

### Questions
1.In practical federated learning scenarios, clients often have different computational capabilities and communication bandwidths. How does BayesADMM adapt to this system heterogeneity? Specifically, what is the algorithm's robustness when certain clients cannot complete the full variational inference update?
2.From Figure 3(a), it can be seen that PVI diverges without damping while BayesADMM remains stable. Could you provide a detailed analysis of BayesADMM's convergence guarantees, particularly theoretical guarantees for non-convex optimization problems?
3.IVON-ADMM uses diagonal covariance approximation in its implementation, which may lead to information loss in deep learning models. How much performance improvement do you think using low-rank approximation or other covariance structures would bring? How much would the computational overhead increase?
4.In Algorithm 1, you mention "implementation details in Appendix G," but the provided PDF excerpt does not contain this section. Could you briefly explain the key implementation differences between IVON-ADMM and standard ADMM, particularly the techniques used when handling high-dimensional parameter spaces?
5.The paper states that BayesADMM can converge in just one communication round for certain loss functions. Is this property limited to specific types of loss functions? Could you provide more general conditions that specify when the algorithm can converge quickly?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a general Bayesian perspective on federated ADMM. Under this framework, the classical ADMM emerges as a special case corresponding to isotropic Gaussian posteriors, while more expressive exponential-family posteriors yield new variants. One such variant called _IVON-ADMM_ is derived using diagonal Gaussian covariance and is claimed to perform better in heterogeneity and uncertainty cases of federated learning.

### Strengths
__Conceptual novelty:__ The paper establishes a novel Bayesian duality perspective that unifies ADMM and VB under a single framework. This is an interesting connection that could inspire extensions of primal-dual optimization methods.

__Clear motivation and exposition:__ The introduction and backgrounds are well written and clearly position the work relative to prior ADMM and PVI approach (Swaroop et al., 2025).

__Framework generality:__ The proposed Bayesian duality formulation provides a principled way to derive new ADMM-like algorithms by changing the exponential-family posterior.

__Readable presentation:__ For the most part, the paper is well structured and logically progress from classical ADMM to its Bayesian interpretation and finally to _IVON-ADMM_.

### Weaknesses
__Soundness of the Formulation:__ While the high-level idea is promising, the derivation in section 3.3 raises concerns about mathematical consistency:
- The "Bayesian ADMM" updates (Eqns. 12-14) are expected to follow from alternating optimization of the Lagrangian in Eqn. 11. However, the replacement of the dual update term $\mu_k - \bar{\mu}$ with $\lambda_k - \bar{\lambda}$ lacks justification within the Lagrangian formulation. The reasoning provided in Appendix E.2, appealing to Bayesian intuition, seems heuristic rather derivational.
- Equation 11 itself may need reconsideration: shifting the linear term $<\hat{\lambda}_k, \mu_k - \hat{\mu}>$ between the sub problems while keeping others fixed breaks symmetry between local and global updates, potentially undermining the claimed equivalence to ADMM.
- Overall, the theoretical grounding of the "Bayesian duality" remains somewhat fragile: the proposed updates look reasonable by analogy but are not rigorously shown to correspond to valid saddle-point dynamics of the stated objective.

__Relation to Existing Work:__
- The method appears to be a straightforward extension of _PVI_ with modified update equations. The novelty over _PVI_ is mainly the introduction of the step size $\rho$ and reinterpretation of dual variables. The paper should make a stronger argument for why this constitutes a _fundametal_ new framework rather  than a variant of _PVI_ with heuristic scaling.
A direct empirical or theoretical  comparison with _PVI_ (as Eqn. 4) is missing. Including  such results would make the claimed advantages more credible.


__Experimental Evaluation__:  The experiments, while broad, are not yet conclusive about the claimed benefits in heterogeneity and uncertainty.
- Figure 3-4 provide illustrative  but small-scale toy examples; they show qualitative improvement but not a clear quantitative advantage.
- The key claim that _IVON_ handles heterogeneity better by leveraging posterior covariance is not substantiated with ablation or analysis showing the role of uncertainty.
- Comparisons with both BayesADMM (without _IVON_) and _PVI_ are missing. Including them would help isolate what _IVON_ adds.
- The computational overhead relative to FedDyn should be quantified to see the computation gain compared to the performance gain.

__Clarity and Notation:__ Several presentation issues reduce readability and reproducibility.
- Step 3 of Fig. 2 is valid only for $\alpha=\frac{1}{1+\rho K}$.
- Many symbols and methods are used before being introduced:
    + BLR first appears on lines 294 and 297 without citation.
    + $q_{1:K}, \hat{t}_{1:K},\bar{q}$ are used before definition in line 215.
- The discussion of natural vs. ordinary gradients is confusing. The paper should use distinct and properly defined notations for both.
- Section 3.2  could be organized better: the correspondence between Eqns. 2 and 10 is conceptually interesting but presented unclearly, with inconsistent references to $\hat{t}_k, \lambda_k, \text{ and } \mu_k$.

### Questions
1. What specific mechanism makes BayesADMM or IVON-ADMM handle heterogeneity and uncertainty better than existing method? Can you clarify the role of the posterior covariance in this improvement and demonstrate it experimentally?
2. In Fig. 4, how are the gray uncertainty contours generated? Are they derived from posterior covariance?
3. Why does the exposition (Eqns. 1, 2, 10, 11) rely on the plain Lagrangian, while the ADMM's implementation (Eqns. 3, 12-14) uses the augmented Lagrangian version?
4. The notation $\hat{\lambda}_k$ seems to play dual roles -- as natural parameters of site functions in Sec. 3.2 and as dual variables in Sec. 3.3. Are both interpretations valid? If so, explain their precise connection.

### Soundness
2

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
2

### Summary
Authors propose a new Bayesian approach to derive and extend the federated Alternating Direction Method of Multipliers (ADMM). We show that the solutions of variational-Bayesian objectives are associated with a duality structure that not only resembles ADMM but also extends it， which opens a new direction to use Bayes to extend ADMM and other primal-dual methods.

### Strengths
Authors introduced a Bayesian duality, from which an extension of ADMM that optimizes over distributions naturally follows. For Gaussians with fixed variance, they recover regular ADMM and general Gaussians give Newton-like methods and IVON-ADMM. These show good performance when compared to recent baselines. Other approximating distributions may lead to new interesting splitting algorithms, and more generally, which opens up new research paths to extend and improve primal-dual algorithms using Bayesian ideas.

### Weaknesses
In the federated learning ADMM framework, there are theoretical guarantees for communication complexity and iterative complexity. Can the author briefly discuss the communication complexity and iteration complexity of Bayesian ADMM.

### Questions
In the federated learning ADMM framework, there are theoretical guarantees for communication complexity and iterative complexity. Can the author briefly discuss the communication complexity and iteration complexity of Bayesian ADMM.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose an analogous extension to federated ADMM in the context of variational inference. This provides an extension of ADMM-like federated procedures based on the duality of various exponential distributions.

### Strengths
The paper proposes a novel ADMM-like extension to federated learning, with good experimental results.

### Weaknesses
It seemed that the argument was more by analogy than exact equality. The claim that ADMM is recovered exactly is misleading because it requires an approximation and therefore is not necessarily recovered exactly.

A couple minor points:
- In equation 3, a subscript k is missing
- On line 195 A* should be defined in the main text rather than just in the appendix.
- In figure 4, it would be helpful to mention that each line is numbered with the iteration number.

### Questions
Why is it valid to just switch back and forth between variational inference and MLE? I was not convinced that equation 5 was equivalent to equation 1. Along this line, the notation in the paper blurred distinction between parameters and distributions (see equation 4 for example and line 976).
In equation 12-14, $\bar{\lambda}$ is not defined. Perhaps include an update equation for it. Is that because it is a deterministic mapping from $\bar{\mu}$? Similarly for $\bar{q}$, is the update from equation 4?
Under what circumstances does BayesADMM converge?

### Soundness
2

### Presentation
2

### Contribution
3
