# Learning from Linear Algebra: A Graph Neural Network Approach to Preconditioner Design for Conjugate Gradient Solvers

- Decision: Reject
- Scores: 5, 6, 5, 3

## Abstract
Large linear systems are ubiquitous in modern computational science and engineering. The main recipe for solving them is the use of Krylov subspace iterative methods with well-designed preconditioners. Deep learning models can be used as nonlinear preconditioners during the iteration of linear solvers such as the conjugate gradient (CG) method. Neural network models require an enormous number of parameters to approximate well in this setup. Another approach is to take advantage of small graph neural networks (GNNs) to construct preconditioners with predefined sparsity patterns. Recently, GNNs have been shown to be a promising tool for designing preconditioners to reduce the overall computational cost of iterative methods by constructing them more efficiently than with classical linear algebra techniques. However, preconditioners designed with these approaches cannot outperform those designed with classical methods in terms of the number of iterations in CG. In our work, we recall well-established preconditioners from linear algebra and use them as a starting point for training the GNN to obtain preconditioners that reduce the condition number of the system more significantly. Numerical experiments show that our approach outperforms both classical and neural network-based methods for an important class of parametric partial differential equations. We also provide a heuristic justification for the loss function used and show that preconditioners obtained by learning with this loss function reduce the condition number in a more desirable way for CG.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
introduces a method for improving preconditioners in the Conjugate Gradient (CG) method using Graph Neural Networks (GNNs). The key contribution is a new scheme, called PreCorrector, which trains a GNN to modify and improve classical preconditioners like incomplete Cholesky (IC). The paper demonstrates that this approach outperforms traditional preconditioning techniques and previously proposed neural network-based preconditioners.

### Strengths
1. The method generalizes well to unseen linear systems, indicating robust applicability.
2. The integration of GNNs  seems reasonable.

### Weaknesses
1. The novelty of this work appears to be limited, as it closely resembles existing methods. Moreover, there is a lack of thorough discussion and comparison with relevant prior works, such as [Azulay, Yael, and Eran Treister. "Multigrid-augmented deep learning preconditioners for the Helmholtz equation"], [Li, Kangan, Sabit Mahmood Khan, and Yashar Mehmani. "Machine learning for preconditioning elliptic equations in porous microstructures: A path to error control"], and [Li, Kangan, Sabit Mahmood Khan, and Yashar Mehmani. "Machine learning for preconditioning elliptic equations in porous microstructures: A path to error control"]. A more detailed comparison would help to better contextualize the contribution of this work within the existing literature.

2. The problems for validation are relatively esay and small, which are Poisson 2D or diffusion equation.

### Questions
1. Could the authors provide more specific details on how the computation time was measured, including the tools or setup used for timing and any relevant conditions that might affect the results?

2. The idea presented seems similar to the method proposed by [Li, Yichen, et al. "Learning preconditioners for conjugate gradient PDE solvers." International Conference on Machine Learning. PMLR, 2023]. However, their method failed with cases involving varying coefficients, as seen in Tables 4 and 5. Could you provide a more detailed explanation of why their approach might have failed, considering differences in the architecture, loss function, and training method? The current explanation in the paragraph "Experiments with neural preconditioner design" is not clear, and it is difficult to understand what sets PreCorrector apart and how it addresses these challenges.

3. Any explanation why the factor $L$ depends on $b$ from L(θ) = GNN(θ, A, b)? Does it mean that in the solving phase, the factor $L$ would vary across iterations in the sense that L(θ) = GNN(θ, A, r), where r is the residual.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes to design CG preconditioners with GNNs. The proposed PreCorrector routine takes legacy preconditioners as input and learns corrections to these legacy preconditioners (e.g., ILU). The proposed method maintains the sparsity pattern of the system matrix and provides better convergence properties in CG iterations. The effectiveness of the proposed method is tested on large-scale diffusion and Poisson's equations. Experimental results show significant speedup over legacy preconditioners.

### Strengths
- **Convergence**: The proposed PreCorrector routine does converge faster than traditional methods; 
- **Generalization**. The proposed method performs well when dealing with high-contrast coefficients in PDEs. This is considered a challenge to all neural preconditioners/solvers; 
- **Sparsity Preservation**. The underlying GNN architecture preserves the sparsity pattern, yielding better performance for succeeding GEMV operations;
- **Experimental Results**. The experiments are comprehensive and the results are convincing, showing significant speedup.

### Weaknesses
- **Theoretical study of the loss function**, as explained by the authors themselves; 
- **Generalization**. The proposed methods are tested over diffusion and Poisson's equations, which are symmetric positive definite. This is acceptable provided that the CG iterator itself requires this property, yet this limits the potential applications of the proposed method. 
- **Limited Comparison**. The paper compares the PreCorrector against classical methods and (Li et al, 2023). The scope is somewhat limited.

### Questions
- Have the authors tried to extend the method to non-symmetric indefinite systems? BiCGSTAB is already available for these systems and how does PreCorrector work with BiCGSTAB?
- How does PreCorrector perform when preconditioning different sparsity patterns beyond C(0) v.s. ICt(5)?
- Ablation on the $\alpha$ parameter?
- What specific steps or techniques were used to stabilize the training process of PreCorrector?
- Computational Trade-off: For large systems, does the time saved by reduced CG iterations outweigh the precomputation and GNN inference time?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes an approach to computing a preconditioner for the conjugate gradient method for solving large linear systems. The approach improves Li et al. (2023) in that rather than learning the nonzeros of the incomplete Cholesky factor, it learns a correction to the factor. The authors show that their approach can outperform the standard incomplete Cholesky (IC) preconditioner, addressing a drawback of Li et al. (2023), which is less performant than IC.

### Strengths
- The idea of learning a correction rather than learning the nonzeros of a preconditioner is novel.

- The learned preconditioner can generalize to different grid sizes and different parameters of the PDE.

- The proposed approach can outperform the standard incomplete Cholesky (IC) preconditioner, addressing a drawback of Li et al. (2023), which is less performant than IC.

### Weaknesses
- The paper is incremental in that a majority of the basis of the work comes from Li et al. (2023).

- The work stresses the use of a different loss function (Eqn (3) rather than Eqn (2)), but the argument is dubious. The authors reformulate Eqn (3) to Eqn (5) in practice, which, however, can also be obtained directly from Eqn (2). Hence, the authors' argument that they use a better loss function ((3) against (2)) does not hold. For more details, see a related question in the following "Questions" section.

### Questions
- While the authors have made clear how the training loss (3) is reformulated to (5), one should note that (2) can also be reformulated to (5), because $b_i = A x_i$. The subtlety lies in whether $x_i$ or $b_i$ is drawn from the standard normal. When one starts with the loss (3), $b_i$ is drawn from the standard normal, which causes the trouble of computing $x_i$ (which requires solving linear systems). In contrast, when one starts with the loss (2), one may draw $x_i$ from the standard normal, which causes no difficulty in computing $b_i$. In this regard, the loss (2) is even better than (3) in practice.

- How does training time compare with solution time? Does the Pre-time column in Tables 1 to 4 mean the evaluation of the GNN or the training of the GNN?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper titled "Learning from Linear Algebra: A Graph Neural Network Approach to Preconditioner Design for Conjugate Gradient Solvers" presents a novel method for designing preconditioners using graph neural networks (GNNs). It aims to improve the efficiency of solving large linear systems that arise in computational science and engineering, particularly those characterized by parametric partial differential equations. The authors argue that their approach, termed "PreCorrector," leverages existing linear algebra techniques and demonstrates superior performance in numerical experiments compared to traditional and other neural network-based methods.

### Strengths
- **Innovative Approach**: The use of GNNs to enhance traditional preconditioners is a fresh perspective in the domain of numerical linear algebra.
- **Theoretical Foundation**: The paper provides a strong theoretical basis for the loss function and discusses its implications on the preconditioner's performance.
- **Numerical Validation**: Extensive experiments validate the proposed method, showcasing its effectiveness in reducing the condition number of systems, which is crucial for the performance of iterative solvers.

### Weaknesses
- **Limited Scope**: While the paper presents promising results, the experiments are confined to a specific class of parametric PDEs. The generalizability of the results to broader contexts remains unclear.
- **Comparison with State-of-the-Art**: The comparison with existing methods could be more comprehensive. Many contemporary techniques in preconditioner design and GNN applications are not adequately addressed.
- **Complexity of Implementation**: The proposed method may introduce additional complexity in practical implementations, which could deter its adoption in industry settings.

### Questions
1. Can the authors provide more insights into how the proposed GNN architecture can be adapted or generalized to other types of linear systems?
2. How do the computational costs of training and implementing the GNN compare to those of traditional preconditioner design methods?
3. Are there plans to evaluate the proposed method on a wider array of problems, particularly those encountered in real-world engineering applications?

### Soundness
2

### Presentation
3

### Contribution
2
