

{0}------------------------------------------------

# A BLOCK COORDINATE DESCENT METHOD FOR NONSMOOTH COMPOSITE OPTIMIZATION UNDER ORTHOGONALITY CONSTRAINTS

Anonymous authors

Paper under double-blind review

## ABSTRACT

Nonsmooth composite optimization with orthogonality constraints is crucial in statistical learning and data science, but it presents challenges due to its nonsmooth objective and computationally expensive, non-convex constraints. In this paper, we propose a new approach called **OBCD**, which leverages Block Coordinate Descent (BCD) to address these challenges. **OBCD** is a feasible method with a small computational footprint. In each iteration, it updates  $k$  rows of the solution matrix, where  $k \geq 2$ , while globally solving a small nonsmooth optimization problem under orthogonality constraints. We prove that the limiting points of **OBCD**, referred to as (global) block- $k$  stationary points, offer stronger optimality than standard critical points. Furthermore, we show that **OBCD** converges to  $\epsilon$ -block- $k$  stationary points with an ergodic convergence rate of  $\mathcal{O}(1/\epsilon)$ . Additionally, under the Kurdyka-Lojasiewicz (KL) inequality, we establish the non-ergodic convergence rate of **OBCD**. We also extend **OBCD** with breakpoint searching methods for subproblem solving and greedy strategies for working set selection. Comprehensive experiments demonstrate the superior performance of our approach across various tasks.

## 1 INTRODUCTION

We consider the following nonsmooth composite optimization problem under orthogonality constraints ( $\triangleq$  means define):

$$\min_{\mathbf{X} \in \mathbb{R}^{n \times r}} F(\mathbf{X}) \triangleq f(\mathbf{X}) + h(\mathbf{X}), \text{ s.t. } \mathbf{X}^\top \mathbf{X} = \mathbf{I}_r. \quad (1)$$

Here,  $n \geq r$  and  $\mathbf{I}_r$  is a  $r \times r$  identity matrix. We do not assume convexity of  $f(\mathbf{X})$  and  $h(\mathbf{X})$ . For brevity, the orthogonality constraints  $\mathbf{X}^\top \mathbf{X} = \mathbf{I}_r$  in Problem (1) is rewritten as  $\mathbf{X} \in \text{St}(n, r) \triangleq \{\mathbf{X} \in \mathbb{R}^{n \times r} \mid \mathbf{X}^\top \mathbf{X} = \mathbf{I}_r\}$ , where  $\mathcal{M} \triangleq \text{St}(n, r)$  is the Stiefel manifold in the literature (Edelman et al., 1998; Absil et al., 2008; Wen & Yin, 2013; Hu et al., 2020). We impose the following assumptions on Problem (1) throughout this paper. (Asm-i) For any  $\mathbf{X}$  and  $\mathbf{X}^+$ , where  $\mathbf{X}$  and  $\mathbf{X}^+$  only differ at most by  $k$  rows with  $k \geq 2$ , we assume  $f : \mathbb{R}^{n \times r} \mapsto \mathbb{R}$  is  $\mathbf{H}$ -smooth with  $\mathbf{0} \preceq \mathbf{H} \in \mathbb{R}^{nr \times nr}$  such that:

$$f(\mathbf{X}^+) \leq \mathcal{Q}(\mathbf{X}^+; \mathbf{X}) \triangleq f(\mathbf{X}) + \langle \mathbf{X}^+ - \mathbf{X}, \nabla f(\mathbf{X}) \rangle + \frac{1}{2} \|\mathbf{X}^+ - \mathbf{X}\|_{\mathbf{H}}^2, \quad (2)$$

where  $\|\mathbf{H}\|_{\text{sp}} \leq L_f$  for some constant  $L_f > 0$  and  $\|\mathbf{X}\|_{\mathbf{H}}^2 \triangleq \text{vec}(\mathbf{X})^\top \mathbf{H} \text{vec}(\mathbf{X})$ <sup>1</sup>. Here,  $\|\mathbf{H}\|_{\text{sp}}$  is the spectral norm of  $\mathbf{H}$ . Notably, when  $\mathbf{H} = L_f \cdot \mathbf{I}_{nr}$ , this condition simplifies to the standard  $L_f$ -smoothness (Nesterov, 2003). (Asm-ii) The function  $h(\mathbf{X}) : \mathbb{R}^{n \times r} \mapsto \mathbb{R}$  is closed, proper, and lower semicontinuous, and potentially non-smooth. Additionally, it is coordinate-wise separable, such that  $h(\mathbf{X}) = \sum_{i,j} h(\mathbf{X}_{ij})$ . Typical examples of  $h(\mathbf{X})$  include the  $\ell_p$  norm function  $h(\mathbf{X}) = \|\mathbf{X}\|_p$  with  $p \in \{0, 1\}$ , and the indicator function for non-negativity constraints  $h(\mathbf{X}) = \mathcal{I}_{\geq 0}(\mathbf{X})$ . (Asm-iii) The following small-sized subproblem can be solved exactly and efficiently:

$$\min_{\mathbf{V} \in \text{St}(k, k)} \mathcal{P}(\mathbf{V}) \triangleq \frac{1}{2} \|\mathbf{V}\|_{\mathbf{Q}}^2 + \langle \mathbf{V}, \mathbf{P} \rangle + h(\mathbf{VZ}), \quad (3)$$

<sup>1</sup>Given any symmetric matrices  $\mathbf{C} \in \mathbb{R}^{n \times n}$  and  $\mathbf{D} \in \mathbb{R}^{r \times r}$ , we let  $\mathbf{H} = \mathbf{D} \otimes \mathbf{C}$ . The function  $f(\mathbf{X}) = \frac{1}{2} \text{tr}(\mathbf{X}^\top \mathbf{C} \mathbf{X} \mathbf{D}) = \frac{1}{2} \|\mathbf{X}\|_{\mathbf{H}}^2$  satisfies (2) with equality, as  $f(\mathbf{X}^+) = \mathcal{Q}(\mathbf{X}^+; \mathbf{X})$  holds for all  $\mathbf{X}$  and  $\mathbf{X}^+$ .

{1}------------------------------------------------

for any given  $\mathbf{Z} \in \mathbb{R}^{k \times r}$ ,  $\mathbf{P} \in \mathbb{R}^{k \times k}$ , and  $\mathbf{Q} \in \mathbb{R}^{k^2 \times k^2}$ . Here, we employ a notational simplification by defining  $h(\mathbf{VZ}) \triangleq \sum_{i,j} h([\mathbf{VZ}]_{ij})$ , given the coordinate-wise separability of the function  $h(\cdot)$ .

Problem (1) is an optimization framework that plays a crucial role in a variety of statistical learning and data science models, such as sparse Principal Component Analysis (PCA) (Journée et al., 2010; Shalit & Chechik, 2014), nonnegative PCA (Zass & Shashua, 2006; Qian et al., 2021), deep neural networks (Cogswell et al., 2016; Cho & Lee, 2017; Xie et al., 2017; Bansal et al., 2018; Massart & Abrol, 2022; Huang & Gao, 2023), electronic structure calculation (Zhang et al., 2014; Liu et al., 2014), Fourier transforms approximation (Frerix & Bruna, 2019), phase synchronization (Liu et al., 2017), orthogonal nonnegative matrix factorization (Jiang et al., 2022),  $K$ -indicators clustering (Jiang et al., 2016), and dictionary learning (Zhai et al., 2020).

### 1.1 RELATED WORK

We now present some related algorithms in the literature.

► **Minimizing Smooth Functions under Orthogonality Constraints.** One difficulty in solving Problem (1) arises from the nonconvexity of the orthogonality constraints. Existing methods for handling this issue can be divided into three classes. (i) Geodesic-like methods (Abrudan et al., 2008; Edelman et al., 1998; Absil et al., 2008; Jiang & Dai, 2015). Since calculating geodesics involves solving ordinary differential equations, which may cause computational complexity, geodesic-like methods iteratively compute the geodesic logarithm using simple linear algebra calculations. The work of (Wen & Yin, 2013) develops a simple and efficient constraint preserving update scheme and achieves low computation complexity per iteration. They combine the feasible update scheme with the Barzilai-Borwein (BB) nonmonotonic line search for optimization with orthogonality constraints. (ii) Projection-like methods (Absil et al., 2008; Golub & Van Loan, 2013). These methods preserve the orthogonality constraints by projection. They decrease the objective value using its current Euclidean gradient direction or Riemannian tangent direction, followed by an orthogonal projection operation. This can be calculated by polar decomposition or approximated by QR factorization. (iii) Multiplier correction methods (Gao et al., 2018; 2019; Xiao et al., 2022). Since the Lagrangian multiplier associated with the orthogonality constraint is symmetric and has an explicit closed-form expression at the first-order optimality condition, multiplier correction methods update the multiplier after achieving sufficient reduction in the objective function. This leads to efficient first-order feasible or infeasible approaches.

► **Minimizing Nonsmooth Functions under Orthogonality Constraints.** Another difficulty of solving Problem (1) comes from the nonsmoothness of the objective function. Existing methods for addressing this problem can be classified into three categories. (i) Subgradient methods (Hwang et al., 2015; Li et al., 2021; Cheung et al., 2024). Subgradient methods are analogous to gradient descent methods. Most of the aforementioned geodesic-like and projection-like strategies can be incorporated into the subgradient methods. However, the step size in subgradient methods needs to be diminishing to guarantee convergence. (ii) Proximal gradient methods (Chen et al., 2020; Li et al., 2024). They solve a strongly convex minimization problem over the tangent space using a semi-smooth Newton method to find a descent direction. Subsequently, they maintain the orthogonality constraint through a retraction operation. (iii) **Block Majorization Minimization (BMM) or BCD on Riemannian manifolds** (Li et al., 2024; 2023; Breloy et al., 2021; Gutman & Ho-Nguyen, 2023; Cheung et al., 2024). This class of methods iteratively constructs a tangential majorizing surrogate for a block of the objective function, takes an approximate descent step in the resulting direction within the tangent space, and then applies retraction to project back onto the manifold. Notably, their subproblems are often solved approximately, whereas our method can solve them exactly due to the small size of the subproblems. (iv) Operator splitting methods (Lai & Osher, 2014; Chen et al., 2016; Zhang et al., 2019). Operator splitting methods introduce linear constraints and decompose the original problem into simpler subproblems, which can be solved separately and exactly. Alternating Direction Methods of Multipliers (ADMM) (He & Yuan, 2012) and Smoothing Penalty Methods (SPM) (Chen, 2012) represent two prominent variants of operator splitting methods.

► **Block Coordinate Descent Methods.** (Block) coordinate descent is a classical and powerful algorithm that solves optimization problems by iteratively performing minimization along (block) coordinate directions (Tseng & Yun, 2009; Xu & Yin, 2013). The BCD methods have recently gained attention in solving nonconvex optimization problems, including sparse optimization (Yuan, 2024),  $k$ -means clustering (Nie et al., 2022), structured nonconvex minimization (Yuan, 2023), recurrent neural network (Massart & Abrol, 2022), and multi-layer convolutional networks (Bibi et al.,

{2}------------------------------------------------

2019; Zeng et al., 2019). BCD methods have also been used in (Shalit & Chechik, 2014; Massart & Abrol, 2022) for solving optimization problems with orthogonal group constraints. However, their column-wise BCD methods are limited only to solve smooth minimization problems with  $k = 2$  and  $r = n$  (Refer to Section 4.2 in (Shalit & Chechik, 2014)). Our row-wise BCD methods can solve general nonsmooth problems with  $k \geq 2$  and  $r \leq n$ . The work of (Gao et al., 2019) proposes a parallelizable column-wise BCD scheme for solving the subproblems of their proximal linearized augmented Lagrangian algorithm. Impressive parallel scalability in a parallel environment of their algorithm is demonstrated. We stress that our **row-wise** BCD methods differ from the two **column-wise** counterparts.

► **Summary.** Existing solutions have one or more of the following limitations: (i) They rely on full gradient information, incurring high computational costs per iteration. (ii) They cannot handle general nonsmooth composite problems. (iii) They lack descent properties, even worse, they are infeasible methods, achieving solution feasibility only at the limit point. (iv) They often lack rigorous convergence guarantees. (v) They only establish weak optimality at critical points. ★ To our knowledge, this represents the first application of BCD methods to solve nonsmooth composite optimization problems under orthogonality constraints, demonstrating strong optimality and convergence guarantees.

### 1.2 CONTRIBUTIONS

This paper makes the following contributions. (i) Algorithmically: We propose a Block Coordinate Descent (BCD) algorithm tailored for nonsmooth composite optimization under orthogonality constraints (Section 2). (ii) Theoretically: We provide comprehensive optimality and convergence analyses of our methods (Sections 3 and 4). (iii) Side Contributions: We introduce breakpoint searching methods for solving subproblems when  $k = 2$  (Section 5), and present two working set selection greedy strategies to improve the computational efficiency of our methods (Section D in the Appendix). (iv) Empirically: Extensive experiments demonstrate that our methods surpass existing solutions in terms of accuracy and/or efficiency (Section 6).

## 2 THE PROPOSED OBCD ALGORITHM

In this section, we introduce **OBCD**, a Block Coordinate Descent algorithm for solving general nonsmooth composite problems under Orthogonality constraints, as defined in Problem 1.

We start by presenting a new update scheme designed to maintain the orthogonality constraint.

► **A New Constraint-Preserving Update Scheme.** For any partition of the index vector  $[1, 2, \dots, n]$  into  $[B, B^c]$  with  $B \in \mathbb{N}^k$ ,  $B^c \in \mathbb{N}^{n-k}$ , we define  $U_B \in \mathbb{R}^{n \times k}$  and  $U_{B^c} \in \mathbb{R}^{n \times (n-k)}$  as:  $(U_B)_{ji} = \begin{cases} 1, & B_i = j; \\ 0, & \text{else.} \end{cases}$ ,  $(U_{B^c})_{ji} = \begin{cases} 1, & B^c_i = j; \\ 0, & \text{else.} \end{cases}$ . Therefore, we have the following variable splitting for any  $X \in \mathbb{R}^{n \times r}$ :  $X = I_n X = (U_B U_B^\top + U_{B^c} U_{B^c}^\top) X = U_B X(B, :) + U_{B^c} X(B^c, :)$ , where  $X(B, :) = U_B^\top X \in \mathbb{R}^{k \times r}$  and  $X(B^c, :) = U_{B^c}^\top X \in \mathbb{R}^{(n-k) \times r}$ .

In each iteration  $t$ , the indices  $\{1, 2, \dots, n\}$  of the rows of decision variable  $X \in \text{St}(n, r)$  are separated to two sets  $B$  and  $B^c$ , where  $B$  is the working set with  $|B| = k$  and  $B^c = \{1, 2, \dots, n\} \setminus B$ . To simplify notation, we use  $B$  instead of  $B^t$ , as  $t$  can be inferred from the context. We only update  $k$  rows of the variable  $X$  via  $X^{t+1}(B, :) \leftarrow V X^t(B, :)$  for some appropriate matrix  $V \in \mathbb{R}^{k \times k}$ . The following equivalent expressions hold:

$$X^{t+1}(B, :) = V X^t(B, :) \Leftrightarrow X^{t+1} = (U_B V U_B^\top + U_{B^c} U_{B^c}^\top) X^t \quad (4)$$

$$\Leftrightarrow X^{t+1} = X^t + U_B (V - I_k) U_B^\top X^t. \quad (5)$$

We consider the following minimization procedure to iteratively solve Problem (1):

$$\min_V F(\mathcal{A}_B^t(V)), \text{ s.t. } \mathcal{A}_B^t(V) \in \text{St}(n, r), \text{ where } \mathcal{A}_B^t(V) \triangleq X^t + U_B (V - I_k) U_B^\top X^t. \quad (6)$$

The following lemma shows that the orthogonality constraint for  $X^+ = X + U_B (V - I_k) U_B^\top X$  can be preserved by choosing suitable  $V$  and  $X$ .

**Lemma 2.1.** (Proof in Appendix E.1) We let  $B \in \{B_j\}_{t=1}^{C_n^k}$ , where the set  $\{B_1, B_2, \dots, B_{C_n^k}\}$  denotes all possible combinations of the index vectors choosing  $k$  items from  $n$  without repetition. We let

{3}------------------------------------------------

$\mathbf{V} \in \text{St}(k, k)$ . We define  $\mathbf{X}^+ \triangleq \mathcal{X}_\mathbb{B}(\mathbf{V}) \triangleq \mathbf{X} + \mathbf{U}_\mathbb{B}(\mathbf{V} - \mathbf{I}_k)\mathbf{U}_\mathbb{B}^\top \mathbf{X}$ . (a) For any  $\mathbf{X} \in \mathbb{R}^{n \times r}$ , we have  $|\mathbf{X}^+|^\top \mathbf{X}^+ = \mathbf{X}^\top \mathbf{X}$ . (b) If  $\mathbf{X} \in \text{St}(n, r)$ , then  $\mathbf{X}^+ \in \text{St}(n, r)$ .

Thanks to Lemma 2.1, we can now explore the following alternative formulation for Problem (6).

$$\bar{\mathbf{V}}^t \in \arg \min_{\mathbf{V}} F(\mathcal{X}_\mathbb{B}^t(\mathbf{V})), \text{ s.t. } \mathbf{V} \in \text{St}(k, k). \quad (7)$$

Then the solution matrix is updated via:  $\mathbf{X}^{t+1} = \mathcal{X}_\mathbb{B}^t(\bar{\mathbf{V}}^t)$ .

The following lemma offers important properties for the update rule  $\mathbf{X}^+ = \mathbf{X} + \mathbf{U}_\mathbb{B}(\mathbf{V} - \mathbf{I}_k)\mathbf{U}_\mathbb{B}^\top \mathbf{X}$ .

**Lemma 2.2.** (Proof in Appendix E.2) We define  $\mathbf{X}^+ = \mathbf{X} + \mathbf{U}_\mathbb{B}(\mathbf{V} - \mathbf{I}_k)\mathbf{U}_\mathbb{B}^\top \mathbf{X}$ . For any  $\mathbf{X} \in \text{St}(n, r)$ ,  $\mathbf{V} \in \text{St}(k, k)$ ,  $\mathbb{B} \in \{\mathcal{B}_i\}_{i=1}^{C_n^k}$ , and symmetric matrix  $\mathbf{H} \in \mathbb{R}^{nr \times nr}$ , we have:

- (a)  $\frac{1}{2} \|\mathbf{X}^+ - \mathbf{X}\|_{\mathbf{H}}^2 = \frac{1}{2} \|\mathbf{V} - \mathbf{I}_k\|_{\mathbf{Q}}^2$ , where  $\mathbf{Q} \triangleq (\mathbf{Z}^\top \otimes \mathbf{U}_\mathbb{B})^\top \mathbf{H}(\mathbf{Z}^\top \otimes \mathbf{U}_\mathbb{B})$ , and  $\mathbf{Z} \triangleq \mathbf{U}_\mathbb{B}^\top \mathbf{X} \in \mathbb{R}^{k \times r}$ .
- (b)  $\frac{1}{2} \|\mathbf{X}^+ - \mathbf{X}\|_{\mathbf{F}}^2 = \langle \mathbf{I}_k - \mathbf{V}, \mathbf{U}_\mathbb{B}^\top \mathbf{X} \mathbf{X}^\top \mathbf{U}_\mathbb{B} \rangle$ .
- (c)  $\frac{1}{2} \|\mathbf{X}^+ - \mathbf{X}\|_{\mathbf{F}}^2 \leq \frac{1}{2} \|\mathbf{V} - \mathbf{I}_k\|_{\mathbf{F}}^2 = \langle \mathbf{I}_k, \mathbf{I}_k - \mathbf{V} \rangle$ .

► **The Main Algorithm.** The proposed algorithm **OBCD** is an iterative procedure that sequentially minimizes the objective function along block coordinate directions within a sub-manifold of  $\mathcal{M}$ .

Starting with an initial feasible solution, **OBCD** iteratively determines a working set  $\mathbb{B}^t$  using specific strategies. It then solves the small-sized subproblem in Problem (7) through successive majorization minimization. This method iteratively constructs a surrogate function that majorizes the objective function, driving it to decrease as expected (Mairal, 2013; Razaviyayn et al., 2013; Sun et al., 2016; Brelot et al., 2021), and it has proven effective for minimizing complex functions.

We now demonstrate how to derive the majorization function for  $F(\mathcal{X}_\mathbb{B}^t(\mathbf{V}))$  in Problem (7). Initially, for any  $\mathbf{X}^t \in \text{St}(n, r)$  and  $\mathbf{V} \in \text{St}(k, k)$ , we establish following inequalities:  $f(\mathcal{X}_\mathbb{B}^t(\mathbf{V})) - f(\mathbf{X}^t) \stackrel{\textcircled{1}}{\leq} \langle \mathcal{X}_\mathbb{B}^t(\mathbf{V}) - \mathbf{X}^t, \nabla f(\mathbf{X}^t) \rangle + \frac{1}{2} \|\mathcal{X}_\mathbb{B}^t(\mathbf{V}) - \mathbf{X}^t\|_{\mathbf{H}}^2 \stackrel{\textcircled{2}}{=} \langle \mathbf{U}_\mathbb{B}(\mathbf{V} - \mathbf{I}_k)\mathbf{U}_\mathbb{B}^\top \mathbf{X}^t, \nabla f(\mathbf{X}^t) \rangle + \frac{1}{2} \|\mathbf{V} - \mathbf{I}_k\|_{\mathbf{Q}}^2 \stackrel{\textcircled{3}}{\leq} \langle \mathbf{V} - \mathbf{I}_k, [\nabla f(\mathbf{X}^t)(\mathbf{X}^t)^\top]_{\mathbb{B}\mathbb{B}} \rangle + \frac{1}{2} \|\mathbf{V} - \mathbf{I}_k\|_{\mathbf{Q}+\alpha\mathbf{I}}^2$ , where step  $\textcircled{1}$  uses Inequality (2); step  $\textcircled{2}$  uses Claim (a) of Lemma 2.2; step  $\textcircled{3}$  uses  $\alpha > 0$  and  $\mathbf{Q} \preceq \mathbf{Q}$ , which can be ensured by choosing  $\mathbf{Q}$  using one of the following methods:

$$\mathbf{Q} = \mathbf{Q} \triangleq (\mathbf{Z}^\top \otimes \mathbf{U}_\mathbb{B})^\top \mathbf{H}(\mathbf{Z}^\top \otimes \mathbf{U}_\mathbb{B}), \text{ with } \mathbf{Z} \triangleq \mathbf{U}_\mathbb{B}^\top \mathbf{X}^t, \quad (8)$$

$$\mathbf{Q} = \mathbf{c}\mathbf{I}, \text{ with } \|\mathbf{Q}\|_{\text{sp}} \leq c \leq L_f. \quad (9)$$

Then, we construct the function  $\mathcal{K}(\mathbf{V}; \mathbf{X}^t, \mathbb{B})$  that majorizes  $F(\mathcal{X}_\mathbb{B}^t(\mathbf{V})) = f(\mathcal{X}_\mathbb{B}^t(\mathbf{V})) + h(\mathcal{X}_\mathbb{B}^t(\mathbf{V}))$ :

$$\begin{aligned} F(\mathcal{X}_\mathbb{B}^t(\mathbf{V})) &\leq f(\mathbf{X}^t) + \langle \mathbf{V} - \mathbf{I}_k, [\nabla f(\mathbf{X}^t)(\mathbf{X}^t)^\top]_{\mathbb{B}\mathbb{B}} \rangle + \frac{1}{2} \|\mathbf{V} - \mathbf{I}_k\|_{\mathbf{Q}+\alpha\mathbf{I}}^2 + h(\mathbf{V}\mathbf{U}_\mathbb{B}^\top \mathbf{X}^t) \\ &\leq \underbrace{\frac{1}{2} \|\mathbf{V} - \mathbf{I}_k\|_{\mathbf{Q}+\alpha\mathbf{I}}^2 + \langle \mathbf{V}, [\nabla f(\mathbf{X}^t)(\mathbf{X}^t)^\top]_{\mathbb{B}\mathbb{B}} \rangle + h(\mathbf{V}\mathbf{U}_\mathbb{B}^\top \mathbf{X}^t)}_{\mathcal{K}(\mathbf{V}; \mathbf{X}^t, \mathbb{B})} + \tilde{c}, \end{aligned} \quad (10)$$

where  $\tilde{c} = f(\mathbf{X}^t) + h(\mathbf{U}_\mathbb{B}^\top \mathbf{X}^t) - \langle \mathbf{I}_k, [\nabla f(\mathbf{X}^t)(\mathbf{X}^t)^\top]_{\mathbb{B}\mathbb{B}} \rangle$  is a constant. Here, we use the coordinate-wise separable property of  $h(\cdot)$  as follows:  $h(\mathcal{X}_\mathbb{B}^t(\mathbf{V})) = h(\mathbf{U}_\mathbb{B}\mathbf{U}_\mathbb{B}^\top \mathbf{X}^t + \mathbf{U}_\mathbb{B}\mathbf{V}\mathbf{U}_\mathbb{B}^\top \mathbf{X}^t) = h(\mathbf{U}_\mathbb{B}^\top \mathbf{X}^t) + h(\mathbf{V}\mathbf{U}_\mathbb{B}^\top \mathbf{X}^t)$ . We minimize the upper bound of the right-hand side of Inequality (10), resulting in the minimization problem that  $\bar{\mathbf{V}}^t \in \arg \min_{\mathbf{V} \in \text{St}(k, k)} \mathcal{K}(\mathbf{V}; \mathbf{X}^t, \mathbb{B})$ , which can be efficiently and exactly solved due to our assumption.

Three strategies to find the working set  $\mathbb{B}$  with  $|\mathbb{B}| = k$  can be considered. (i) Random strategy:  $\mathbb{B}$  is randomly selected from  $\{\mathcal{B}_1, \mathcal{B}_2, \dots, \mathcal{B}_{C_n^k}\}$  with equal probability  $1/C_n^k$ . (ii) Cyclic strategy:  $\mathbb{B}^t$  takes all possible combinations in cyclic order, such as  $\mathcal{B}_1 \rightarrow \mathcal{B}_2 \rightarrow \dots \rightarrow \mathcal{B}_{C_n^k} \rightarrow \mathcal{B}_1 \rightarrow \dots$  (iii) Greedy strategy: We propose two novel greedy strategies to find a good working set. Due to space limitation, we have included them in Appendix D.

The proposed **OBCD** algorithm is summarized in Algorithm 1. Importantly, **OBCD** is a partial gradient method with low iterative computational complexity as it only assesses  $k$  rows of the Euclidean gradient of  $\nabla f(\mathbf{X}^t)$  and the solution  $\mathbf{X}^t$  to compute the linear term  $\langle [\nabla f(\mathbf{X}^t)(\mathbf{X}^t)^\top]_{\mathbb{B}\mathbb{B}}, \mathbf{V} \rangle = \langle [\nabla f(\mathbf{X}^t)]_{:, [\mathbf{X}^t]_{:, \mathbb{B}}}, \mathbf{V} \rangle$ , as shown in Equation (10).

► **Solving the General OBCD Subproblems.** The following lemma outlines key properties of the **OBCD** subproblems.

{4}------------------------------------------------

**Algorithm 1:** OBCD, The Proposed Block Coordinate Descent Algorithm for Problem (1).

**Input:** an initial feasible solution  $\mathbf{X}^0$ . Set  $k \geq 2$ ,  $t = 0$ .

**for**  $t$  from 0 to  $T$  **do**

(S1) Use some strategy to find a working set  $\mathbb{B}^t$  for the  $t$ -th iteration with

$\mathbb{B}^t \subseteq \{1, 2, \dots, n\}^k$ . Let  $\mathbb{B} = \mathbb{B}^t$  and  $\mathbb{B}^c = \{1, 2, \dots, n\} \setminus \mathbb{B}$ .

(S2) Choose a suitable matrix  $\mathbf{Q} \in \mathbb{R}^{k^2 \times k^2}$  using Equation (8) or Equation (9):

(S3) Find a **global (or local)** optimal solution  $\bar{\mathbf{V}}^t$  for the following problem:

$$\bar{\mathbf{V}}^t \in \arg \min_{\mathbf{V} \in \text{St}(k, k)} \mathcal{K}(\mathbf{V}; \mathbf{X}^t, \mathbb{B})$$

satisfying  $\mathcal{K}(\bar{\mathbf{V}}^t; \mathbf{X}^t, \mathbb{B}) \leq \mathcal{K}(\mathbf{I}_k; \mathbf{X}^t, \mathbb{B})$ , where  $\mathcal{K}(\cdot; \cdot, \cdot)$  is defined in Inequality (10).

(S4)  $\mathbf{X}^{t+1}(\mathbb{B}, :) = \bar{\mathbf{V}}^t \mathbf{X}^t(\mathbb{B}, :)$

**end**

**Lemma 2.3.** (Proof in Appendix E.3) We define  $\mathbf{P} \triangleq [\nabla f(\mathbf{X}^t)(\mathbf{X}^t)^\top]_{\text{BB}} - \text{mat}(\text{Qvec}(\mathbf{I}_k)) - \alpha \mathbf{I}_k$ , and  $\mathbf{Z} = \mathbf{U}_{\mathbb{B}}^\top \mathbf{X}^t$ . We have: (a) The subproblem  $\bar{\mathbf{V}}^t \in \arg \min_{\mathbf{V} \in \text{St}(k, k)} \mathcal{K}(\mathbf{V}; \mathbf{X}^t, \mathbb{B})$  in Algorithm 1 is equivalent to Problem (3). (b) Assume that Formula (9) is used to choose  $\mathbf{Q}$ . Problem (3) further reduces to the following problem:  $\bar{\mathbf{V}}^t \in \arg \min_{\mathbf{V} \in \text{St}(k, k)} \mathcal{P}(\mathbf{V}) \triangleq \langle \mathbf{V}, \mathbf{P} \rangle + h(\mathbf{VZ})$ . In particular, when  $h(\mathbf{X}) \triangleq 0$ , we obtain:  $\bar{\mathbf{V}}^t = -\mathbb{P}_{\mathcal{M}}(\mathbf{P})$ . Here,  $\mathbb{P}_{\mathcal{M}}(\mathbf{P})$  is the nearest orthogonality matrix to  $\mathbf{P}$ .

**Remark 2.4.** (a) By Claim (b) of Lemma 2.3, when  $k > 2$ ,  $h(\mathbf{X}) = 0$ , and  $\mathbf{Q}$  is chosen to be a diagonal matrix as in Equation (9), the subproblem  $\bar{\mathbf{V}}^t \in \arg \min_{\mathbf{V} \in \text{St}(k, k)} \mathcal{K}(\mathbf{V}; \mathbf{X}^t, \mathbb{B})$  in Algorithm 1 can be solved exactly and efficiently due to our assumption, see Remark 2.6. (b) For general  $k$  and  $h(\cdot)$ , the subproblem may not be solved globally, but a local stationary solution  $\bar{\mathbf{V}}^t$  satisfying  $(\bar{\mathbf{V}}^t; \mathbf{X}^t, \mathbb{B}) \leq \mathcal{K}(\mathbf{I}_k; \mathbf{X}^t, \mathbb{B})$  can be achieved. Although strong optimality may be compromised, convergence to a critical point (as discussed later) for the final solution  $\mathbf{X}^\infty$  remains achievable.

► **Smallest Possible Subproblems When  $k = 2$ .** We now discuss how to solve the subproblems exactly when  $k = 2$  and  $h(\cdot) \neq 0$ . The following lemma reveals an equivalent expression for any  $\mathbf{V} \in \text{St}(2, 2)$ .

**Lemma 2.5.** (Proof in Appendix E.4) Any orthogonal matrix  $\mathbf{V} \in \text{St}(2, 2)$  can be expressed as  $\mathbf{V} = \mathbf{V}_\theta^{\text{rot}}$  or  $\mathbf{V} = \mathbf{V}_\theta^{\text{ref}}$  for some  $\theta \in \mathbb{R}$ , where  $\mathbf{V}_\theta^{\text{rot}} \triangleq \begin{pmatrix} \cos(\theta) & \sin(\theta) \\ -\sin(\theta) & \cos(\theta) \end{pmatrix}$ ,  $\mathbf{V}_\theta^{\text{ref}} \triangleq \begin{pmatrix} -\cos(\theta) & \sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{pmatrix}$ . We have  $\det(\mathbf{V}_\theta^{\text{rot}}) = 1$  and  $\det(\mathbf{V}_\theta^{\text{ref}}) = -1$  for any  $\theta$ .

Using Lemma 2.5, we can reformulate Problem (3) as the following one-dimensional problem:  $\bar{\theta} \in \arg \min_{\theta} \mathcal{P}(\mathbf{V}_\theta)$ , s.t.  $\mathbf{V} \in \{\mathbf{V}_\theta^{\text{rot}}, \mathbf{V}_\theta^{\text{ref}}\}$ . The optimal solution  $\bar{\theta}$  can be identified even if  $h(\cdot) \neq 0$  using a novel breakpoint searching method, which is discussed later in Section 5.

**Remark 2.6.** (i)  $\mathbf{V}_\theta^{\text{rot}}$  and  $\mathbf{V}_\theta^{\text{ref}}$  are called Givens rotation matrix and Jacobi reflection matrix respectively in the literature (Sun & Bischof, 1995). Previous research only considered  $\{\mathbf{V}_\theta^{\text{rot}}\}$  for solving symmetric linear eigenvalue problems (Golub & Van Loan, 2013) and sparse PCA problems (Shalit & Chechik, 2014), while we use  $\{\mathbf{V}_\theta^{\text{rot}}, \mathbf{V}_\theta^{\text{ref}}\}$  for solving Problem (1). (ii) We show the necessity of using  $\{\mathbf{V}_\theta^{\text{rot}}, \mathbf{V}_\theta^{\text{ref}}\}$  in the following two examples of  $2 \times 2$  optimization problems with orthogonality constraints:  $\min_{\mathbf{V} \in \text{St}(2, 2)} F(\mathbf{V}) \triangleq \|\mathbf{V} - \mathbf{A}\|_F^2$ , and  $\min_{\mathbf{V} \in \text{St}(2, 2)} F(\mathbf{V}) \triangleq \|\mathbf{V} - \mathbf{B}\|_F^2 + 5\|\mathbf{V}\|_1$ , where  $\mathbf{A} = \begin{pmatrix} 1 & 0 \\ -1 & -1 \end{pmatrix}$  and  $\mathbf{B} = \begin{pmatrix} 1 & 0 \\ 1 & 0 \end{pmatrix}$ . The use of the reflection matrix  $\mathbf{V}_\theta^{\text{ref}}$  is essential in these examples because it results in lower objective values. See Section C.1 in the Appendix for more details.

## 3 OPTIMALITY ANALYSIS

This section provides some optimality analysis for the proposed algorithm.

► **Basis Representation of Orthogonal Matrices.** The following theorem is used to characterize any orthogonal matrix  $\mathbf{D} \in \text{St}(n, n)$  and  $\mathbf{X} \in \text{St}(n, r)$ .

**Theorem 3.1.** (Proof in Appendix F.1, Basis Representation of Orthogonal Matrices) Assume  $k = 2$ . For all  $i \in [C_n^k]$ , we define  $\mathcal{W}_i \triangleq \mathbf{I}_n + \mathbf{U}_{\mathcal{B}_i}(\mathcal{V}_i - \mathbf{I}_k)\mathbf{U}_{\mathcal{B}_i}^\top = \mathbf{U}_{\mathcal{B}_i}\mathcal{V}_i\mathbf{U}_{\mathcal{B}_i}^\top + \mathbf{U}_{\mathcal{B}_i^c}\mathbf{U}_{\mathcal{B}_i^c}^\top$ , where

{5}------------------------------------------------

$V_i \in \text{St}(2, 2)$ . We have: **(a)** Any matrix  $\mathbf{D} \in \text{St}(n, n)$  can be expressed as  $\mathbf{D} = \mathcal{W}_{C_n^k} \dots \mathcal{W}_2 \mathcal{W}_1$  using suitable  $\mathcal{W}_i$  (which depends on  $V_i$ ). Furthermore, if  $\forall i, V_i = \mathbf{I}_2$ , then  $\mathbf{D} = \mathbf{I}_n$ . **(b)** Any matrix  $\mathbf{X} \in \text{St}(n, r)$  can be expressed as  $\mathbf{X} = \mathcal{W}_{C_n^k} \dots \mathcal{W}_2 \mathcal{W}_1 \mathbf{X}^0$  using suitable  $\mathcal{W}_i$  and any fixed constant matrix  $\mathbf{X}^0 \in \text{St}(n, r)$ .

**Remark 3.2.** (i) We use both Givens rotation and Jacobi reflection matrices to compute  $\mathbf{D} \in \text{St}(n, n)$ . This is necessary since a reflection matrix cannot be represented through a sequence of rotations. (ii) The result in Claim (b) of Theorem 3.1 indicates that the proposed update scheme  $\mathbf{X}^+ \leftarrow \mathbf{X} + \mathcal{U}_B(\mathbf{V} - \mathbf{I}_k)\mathcal{U}_B^\top \mathbf{X}$  as shown in Formula (5) can reach any orthogonal matrix  $\mathbf{X} \in \text{St}(n, r)$  for any starting solution  $\mathbf{X}^0 \in \text{St}(n, r)$ .

► **First-Order Optimality Conditions for Problem (1).** We provide the first-order optimality condition of Problem (1) (Wen & Yin, 2013; Chen et al., 2020). We use  $\partial F(\mathbf{X})$  to denote the limiting subdifferential of  $F(\mathbf{X})$  (Mordukhovich, 2006; Rockafellar & Wets., 2009), which is always non-empty since  $F(\mathbf{X})$  is closed, proper, and lower semicontinuous. Given  $f(\mathbf{X})$  is differentiable, we have  $\partial F(\mathbf{X}) = \partial(f + h)(\mathbf{X}) = \nabla f(\mathbf{X}) + \partial h(\mathbf{X})$ . We extend the definition of limiting subdifferential to introduce  $\partial_{\mathcal{M}} F(\mathbf{X})$  as the Riemannian limiting subdifferential of  $F(\mathbf{X})$  at  $\mathbf{X}$ , defined as  $\partial_{\mathcal{M}} F(\mathbf{X}) \triangleq \partial F(\mathbf{X}) \ominus (\mathbf{X}[\partial F(\mathbf{X})]^\top \mathbf{X})$ , where  $\ominus$  is the element-wise subtraction between sets.

Introducing a Lagrangian multiplier matrix  $\Lambda \in \mathbb{R}^{r \times r}$  for the orthogonality constraint, we define the following Lagrangian function of Problem (1):  $\mathcal{L}(\mathbf{X}, \Lambda) = F(\mathbf{X}) + \frac{1}{2} \langle \mathbf{I}_r - \mathbf{X}^\top \mathbf{X}, \Lambda \rangle$ . Notable, the matrix  $\Lambda$  is symmetric, as  $\mathbf{X}^\top \mathbf{X}$  is symmetric. We state the following definition of first-order optimality condition.

**Definition 3.3.** Critical Point (Wen & Yin, 2013; Chen et al., 2020). A solution  $\tilde{\mathbf{X}} \in \text{St}(n, r)$  is a critical point of Problem (1) if:  $\mathbf{0} \in \partial_{\mathcal{M}} F(\tilde{\mathbf{X}}) \triangleq \partial F(\tilde{\mathbf{X}}) \ominus (\tilde{\mathbf{X}}[\partial F(\tilde{\mathbf{X}})]^\top \tilde{\mathbf{X}})$ , where  $(\partial F(\tilde{\mathbf{X}}) \ominus \tilde{\mathbf{X}}[\partial F(\tilde{\mathbf{X}})]^\top \tilde{\mathbf{X}}) \triangleq \{\mathbf{G} - \tilde{\mathbf{X}} \mathbf{G}^\top \tilde{\mathbf{X}} \mid \mathbf{G} \in \partial F(\tilde{\mathbf{X}})\}$ . Furthermore,  $\Lambda \in [\partial F(\tilde{\mathbf{X}})]^\top \tilde{\mathbf{X}}$ .

**Remark 3.4.** The critical point condition in Lemma 3.3 can be equivalently expressed as (Absil et al., 2008; Jiang & Dai, 2015; Liu et al., 2016):  $\mathbf{0} \in \mathbb{P}_{T_{\tilde{\mathbf{X}}}\mathcal{M}}(\partial F(\tilde{\mathbf{X}}))$ . Here,  $T_{\tilde{\mathbf{X}}}\mathcal{M}$  is the tangent space to  $\mathcal{M}$  at  $\tilde{\mathbf{X}} \in \mathcal{M}$  with  $T_{\tilde{\mathbf{X}}}\mathcal{M} = \{\mathbf{Y} \in \mathbb{R}^{n \times r} \mid \mathbf{X}^\top \mathbf{Y} + \mathbf{Y}^\top \mathbf{X} = \mathbf{0}\}$ .

► **Optimality Conditions for the Subproblems.** The Euclidean subdifferential of  $\mathcal{K}(\mathbf{V}; \mathbf{X}^t, \mathbf{B}^t)$  w.r.t.  $\mathbf{V}$  can be computed as follows:  $\tilde{\mathbf{G}}(\mathbf{V}) \triangleq \tilde{\mathbf{\Delta}} + \mathbf{U}_B^\top [\nabla f(\mathbf{X}^t) + \partial h(\mathbf{X}^{t+1})](\mathbf{X}^t)^\top \mathbf{U}_B$ , where  $\tilde{\mathbf{\Delta}} = \text{mat}((\mathbf{Q} + \alpha \mathbf{I}_k)\text{vec}(\mathbf{V} - \mathbf{I}_k))$ , and  $\mathbf{X}^{t+1} = \mathbf{X}^t + \mathcal{U}_B(\mathbf{V} - \mathbf{I}_k)\mathcal{U}_B^\top \mathbf{X}^t$ . Using Lemma 3.3, we set the Riemannian subdifferential of  $\mathcal{K}(\mathbf{V}; \mathbf{X}^t, \mathbf{B}^t)$  w.r.t.  $\mathbf{V}$  to zero and obtain the following first-order optimality condition for  $\tilde{\mathbf{V}}^t$ :  $\mathbf{0} \in \partial_{\mathcal{M}} \mathcal{K}(\tilde{\mathbf{V}}^t; \mathbf{X}^t, \mathbf{B}^t) \triangleq \tilde{\mathbf{G}}(\tilde{\mathbf{V}}^t) \ominus \tilde{\mathbf{V}}^t \tilde{\mathbf{G}}(\tilde{\mathbf{V}}^t)^\top \tilde{\mathbf{V}}^t$ .

► **Optimality Conditions and Their Hierarchy.** We introduce the following new optimality condition of block- $k$  stationary points.

**Definition 3.5.** (Global) Block- $k$  Stationary Point, abbreviated as  $\text{BS}_k$ -point. Let  $\alpha > 0$  and  $k \geq 2$ . A solution  $\tilde{\mathbf{X}} \in \text{St}(n, r)$  is called a block- $k$  stationary point if:  $\forall B \in \{B_i\}_{i=1}^{C_n^k}$ ,  $\mathbf{I}_k \in \arg \min_{\mathbf{V} \in \text{St}(k, k)} \mathcal{K}(\mathbf{V}; \tilde{\mathbf{X}}, \mathbf{B})$ , where  $\mathcal{K}(\cdot, \cdot, \cdot)$  is defined in Equation (10).

**Remarks.**  $\text{BS}_k$ -point states that if we globally minimize the majorization function  $\mathcal{K}(\mathbf{V}; \tilde{\mathbf{X}}, \mathbf{B})$ , there is no possibility of improving the objective function value for  $\mathcal{K}(\mathbf{V}; \tilde{\mathbf{X}}, \mathbf{B})$  across all  $\mathbf{B} \in \{B_i\}_{i=1}^{C_n^k}$ .

The following theorem establishes the relation between  $\text{BS}_k$ -points, standard critical points, and global optimal points.

**Theorem 3.6.** (Proof in Appendix F.2) We establish the following relationships:

- (a) {critical points  $\tilde{\mathbf{X}}$ }  $\supseteq$  { $\text{BS}_2$ -points  $\tilde{\mathbf{X}}$ }.
- (b) { $\text{BS}_2$ -points  $\tilde{\mathbf{X}}$ }  $\supseteq$  {global optimal points  $\tilde{\mathbf{X}}$ }.
- (c) { $\text{BS}_k$ -points  $\tilde{\mathbf{X}}$ }  $\supseteq$  { $\text{BS}_{k+1}$ -points  $\tilde{\mathbf{X}}$ }, where  $k \in \{2, 3, \dots, n-1\}$ .
- (d) The reverse of the above three inclusions may not always hold true.

**Remark 3.7.** The optimality of  $\text{BS}_2$ -points is stronger than that of standard critical points (Wen & Yin, 2013; Chen et al., 2020; Absil et al., 2008).

{6}------------------------------------------------

## 4 CONVERGENCE ANALYSIS

This section presents the ergodic and non-ergodic (or last-iterate) convergence rates of the proposed **OBCD** algorithm.

We denote any point of the limit point set of **OBCD** (which is not necessarily a singleton) as  $\tilde{\mathbf{X}}$ . For the case where a random strategy is used to find the working set, **OBCD** generates a random output  $(\tilde{\mathbf{V}}^t, \mathbf{X}^{t+1})$  with  $t = 0, 1, \dots, \infty$  which depends on the observed realization of the random variable:  $\xi^t \triangleq (\mathbb{B}^1, \mathbb{B}^2, \mathbb{B}^3, \dots, \mathbb{B}^t)$ .

### 4.1 ERGODIC CONVERGENCE RATE

Initially, we introduce the notation of  $\epsilon$ -BS $_k$ -point as follows.

**Definition 4.1.** ( $\epsilon$ -BS $_k$ -point) Given any constant  $\epsilon > 0$ , a point  $\tilde{\mathbf{X}}$  is called an  $\epsilon$ -BS $_k$ -point if:  $\frac{1}{C_n^k} \sum_{t=1}^{C_n^k} \text{dist}(\mathbb{I}_k, \arg \min_{\mathbf{V}} \mathcal{K}(\mathbf{V}; \tilde{\mathbf{X}}, \mathbb{B}_t))^2 \leq \epsilon$ , where  $\mathcal{K}(\cdot, \cdot, \cdot)$  is defined in Equation (10).

Using the optimality measure from Definition 4.1, we establish the ergodic convergence rates of **OBCD**.

**Theorem 4.2.** (Proof in Appendix G.1) We define  $\tilde{c} \triangleq \frac{2}{\alpha} \cdot (F(\mathbf{X}^0) - F(\tilde{\mathbf{X}}))$ . We have:

(a) The following sufficient decrease condition holds for all  $t \geq 0$ :

$$\frac{\alpha}{2} \|\mathbf{X}^{t+1} - \mathbf{X}^t\|_F^2 \leq \frac{\alpha}{2} \|\tilde{\mathbf{V}}^t - \mathbb{I}_k\|_F^2 \leq F(\mathbf{X}^t) - F(\mathbf{X}^{t+1}).$$

(b) If the  $\mathbb{B}^t$  is selected from  $\{\mathbb{B}_i\}_{i=1}^{C_n^k}$  randomly and uniformly, **OBCD** finds an  $\epsilon$ -BS $_k$ -point of Problem (1) in at most  $T$  iterations in the sense of expectation, where  $T \geq \lceil \frac{\tilde{c}}{\epsilon} \rceil$ .

(c) If the  $\mathbb{B}^t$  is selected from  $\{\mathbb{B}_i\}_{i=1}^{C_n^k}$  cyclically, **OBCD** finds an  $\epsilon$ -BS $_k$ -point of Problem (1) in at most  $T$  iterations deterministically, where  $T \geq \lceil \frac{\tilde{c}}{\epsilon} + C_n^k \rceil$ .

**Remark 4.3.** Theorem 4.2 shows that **OBCD** converges to  $\epsilon$ -block- $k$  stationary points with an ergodic convergence rate of  $O(1/\epsilon)$ , which is typical for general nonconvex optimization.

Apart from Definition 4.1, another common optimality measure relies on the Riemannian subgradient. To this end, we present the following lemma. For simplicity, we assume that a random strategy is employed to determine the working set in the remainder of this paper.

**Lemma 4.4.** (Proof in Appendix G.2, **Riemannian Subgradient Lower Bound for the Iterates Gap**) Assume  $\|\nabla f(\mathbf{X})\|_{\text{sp}} \leq l_f, \|\partial h(\mathbf{X})\|_{\text{sp}} \leq l_h$  for all  $\mathbf{X} \in \text{St}(n, r)$  with  $l_f, l_h > 0$ . The Riemannian subdifferential of  $\mathcal{K}(\mathbf{V}; \mathbf{X}^t, \mathbb{B}^t)$  at the point  $\mathbf{V} = \mathbb{I}_k$  can be computed as:  $\partial_{\mathcal{M}} \mathcal{K}(\mathbb{I}_k; \mathbf{X}^t, \mathbb{B}^t) = \mathbb{U}_{\mathbb{B}^t}^{\perp} (\mathbb{D} \oplus \mathbb{D}^T) \mathbb{U}_{\mathbb{B}^t}^{\perp}$ , where  $\mathbb{D} = [\nabla f(\mathbf{X}^t) + \partial h(\mathbf{X}^t)][\mathbf{X}^t]^T$ . (a) It holds that:  $\mathbb{E}_{\xi^{t+1}}[\text{dist}(\mathbf{0}, \partial_{\mathcal{M}} \mathcal{K}(\mathbb{I}_k; \mathbf{X}^{t+1}, \mathbb{B}^{t+1}))] \leq \phi \cdot \mathbb{E}_{\xi^t}[\|\tilde{\mathbf{V}}^t - \mathbb{I}_k\|_F]$ , where  $\phi \triangleq 4(l_f + l_h + L_f) + 2\alpha$ . (b)  $\mathbb{E}_{\xi^t}[\text{dist}(\mathbf{0}, \partial_{\mathcal{M}} F(\mathbf{X}^t))] \leq \gamma \cdot \mathbb{E}_{\xi^t}[\text{dist}(\mathbf{0}, \partial_{\mathcal{M}} \mathcal{K}(\mathbb{I}_k; \mathbf{X}^t, \mathbb{B}^t))]$ , where  $\gamma \triangleq (C_n^k/C_n^{k-2})^{1/2}$ .

**Remark 4.5.** The important class of nonsmooth  $\ell_1$  norm function  $h(\mathbf{X}) = \|\mathbf{X}\|_1$  (Chen et al., 2020; 2024) satisfies the assumption made in Lemma 4.4.

We establish the ergodic convergence rates of **OBCD** using the optimality measure of Riemannian subgradient (Chen et al., 2020; Cheung et al., 2024; Li et al., 2024).

**Theorem 4.6.** (Proof in Appendix G.3) We define  $\tilde{c} \triangleq \frac{2}{\alpha} \cdot (F(\mathbf{X}^0) - F(\tilde{\mathbf{X}}))$ , and  $\{\phi, \gamma\}$  as in Lemma 4.4. **OBCD** finds an  $\epsilon$ -critical point of Problem (1) satisfying  $\mathbb{E}_{\xi^t}[\text{dist}^2(\mathbf{0}, \partial_{\mathcal{M}} F(\mathbf{X}^{t+1}))] \leq \epsilon$  in at most  $T + 1$  iterations in the sense of expectation, where  $T \geq \lceil \frac{\gamma^2 \phi^2 \tilde{c}}{\epsilon} \rceil$ .

### 4.2 NON-ERGODIC CONVERGENCE RATE UNDER KL ASSUMPTION

We establish the non-ergodic convergence rate of **OBCD** using the Kurdyka-Łojasiewicz inequality, a key tool in non-convex analysis (Attouch et al., 2010; Bolte et al., 2014; Liu et al., 2016).

Initially, we make the following additional assumption.

**Assumption 4.7.** The function  $F^\circ(\mathbf{X}) = F(\mathbf{X}) + \mathcal{I}_{\mathcal{M}}(\mathbf{X})$  is a KL function.

{7}------------------------------------------------

**Remark 4.8.** *Semi-algebraic functions are a class of functions that satisfy the KL property. These functions are widely used in applications, and they include real polynomial functions, finite sums and products of semi-algebraic functions, and indicator functions of semi-algebraic sets (Attouch et al., 2010; Xu & Yin, 2013).*

We present the following useful proposition, due to (Attouch et al., 2010; Bolte et al., 2014).

**Proposition 4.9. (Kurdyka-Łojasiewicz Property).** For a KL function  $F^\circ(\mathbf{X})$  with  $\mathbf{X} \in \text{dom } F^\circ$ , there exists  $\sigma \in [0, 1)$ ,  $\eta \in (0, +\infty]$ , a neighborhood  $\Upsilon$  of  $\tilde{\mathbf{X}}$ , and a concave continuous function  $\varphi(t) = ct^{1-\sigma}$ ,  $c > 0$ ,  $t \in [0, \eta)$  such that for all  $\mathbf{X}' \in \Upsilon$  and satisfies  $F^\circ(\mathbf{X}') \in (F^\circ(\tilde{\mathbf{X}}), F^\circ(\tilde{\mathbf{X}}) + \eta)$ , the following inequality holds:  $\text{dist}(\mathbf{0}, \partial F^\circ(\mathbf{X}'))\varphi'(F^\circ(\mathbf{X}') - F^\circ(\tilde{\mathbf{X}})) \geq 1$ .

Utilizing the Kurdyka-Łojasiewicz property, one can establish a finite-length property of **OBCD**, a result considerably stronger than that of Theorem 4.2.

**Theorem 4.10. (Proof in Appendix G.4, A Finite Length Property).** We define  $e^{\ell+1} \triangleq \mathbb{E}_{\mathcal{E}^\ell} \|\tilde{\mathbf{V}}^\ell - \mathbf{I}_k\|_F$ , and  $d^\ell \triangleq \sum_{j=\ell}^{\infty} e^{j+1}$ . Based on the continuity assumption made in Lemma 4.4, We have:

- (a) It holds that  $(e^{\ell+1})^2 \leq \kappa e^\ell (\varphi^\ell - \varphi^{\ell+1})$ , where  $\varphi^\ell \triangleq \varphi(F(\mathbf{X}^\ell) - F(\tilde{\mathbf{X}}))$ ,  $\kappa \triangleq \frac{2\gamma\phi}{\alpha}$  is a positive constant,  $\gamma \triangleq (C_n^k/C_{n-2}^k)^{1/2}$ ,  $\phi$  is defined in Lemma 4.4, and  $\varphi(\cdot)$  is the desingularization function defined in Proposition 4.9.
- (b) It holds that  $\forall t \geq 1$ ,  $d^t \leq e^t + 2\kappa\varphi^t$ . The sequence  $\{e^\ell\}_{\ell=1}^{\infty}$  has the finite length property that  $d^\ell \triangleq \sum_{j=\ell}^{\infty} e^{j+1}$  is always upper-bounded by a certain constant.

Finally, we establish the last-iterate convergence rate for **OBCD**.

**Theorem 4.11. (Proof in Appendix G.5).** Based on the continuity assumption made in Lemma 4.4, there exists  $t'$  such that for all  $t \geq t'$ , we have:

- (a) If  $\sigma = 0$ , then the sequence  $\mathbf{X}^\ell$  converges in a finite number of steps in expectation.
- (b) If  $\sigma \in (0, \frac{1}{2}]$ , then there exist  $\dot{c} > 0$  and  $\dot{\tau} \in (0, 1]$  such that  $\mathbb{E}_{\mathcal{E}^{\ell-1}} \|\mathbf{X}^\ell - \mathbf{X}^\infty\|_F \leq \dot{c}\dot{\tau}^\ell$ .
- (c) If  $\sigma \in (\frac{1}{2}, 1)$ , then there exist  $\dot{c} > 0$  such that  $\mathbb{E}_{\mathcal{E}^{\ell-1}} \|\mathbf{X}^\ell - \mathbf{X}^\infty\|_F \leq \mathcal{O}(t^{-(1-\sigma)/(2\sigma-1)})$ .

**Remark 4.12.** *When  $F(\mathbf{X})$  is a semi-algebraic function and the desingularising function is  $\varphi(t) = ct^{1-\sigma}$  for some  $c > 0$  and  $\sigma \in [0, 1)$ , Theorem 4.11 shows that **OBCD** converges in finite iterations when  $\sigma = 0$ , with linear convergence when  $\sigma \in (0, \frac{1}{2}]$ , and sublinear convergence when  $\sigma \in (\frac{1}{2}, 1)$  for the gap  $\|\mathbf{X}^\ell - \mathbf{X}^\infty\|_F$  in expectation. These results are consistent with those in (Attouch et al., 2010).*

## 5 SOLVING THE SUBPROBLEM WHEN $k = 2$

This section presents a novel Breakpoint Searching Method (**BSM**) to find the global optimal solution of Problem (3) when  $k = 2$ .

Initially, Problem (3) boils down to the following one-dimensional subproblem:  $\min_{\theta} \frac{1}{2} \|\mathbf{V}\|_Q^2 + (\mathbf{V}, \mathbf{P}) + h(\mathbf{VZ})$ , s.t.  $\mathbf{V} \in \{\mathbf{V}_\theta^{\text{opt}}, \mathbf{V}_\theta^{\text{ref}}\}$ , which can be further rewritten as:  $\theta \in \arg \min_{\theta} \frac{1}{2} \text{vec}(\mathbf{V})^\top \mathbf{Q} \text{vec}(\mathbf{V}) + (\mathbf{V}, \mathbf{P}) + h(\mathbf{VZ})$ , s.t.  $\mathbf{V} \triangleq \begin{pmatrix} \pm \cos(\theta) & \sin(\theta) \\ \mp \sin(\theta) & \cos(\theta) \end{pmatrix}$ , where  $\mathbf{Q} \in \mathbb{R}^{4 \times 4}$ ,  $\mathbf{P} \in \mathbb{R}^{2 \times 2}$ , and  $\mathbf{Z} \in \mathbb{R}^{2 \times r}$ . Given  $h(\cdot)$  is coordinate-wise separable, we have the following equivalent optimization problem:

$$\min_{\theta} h(\cos(\theta)\mathbf{x} + \sin(\theta)\mathbf{y}) + a \cos(\theta) + b \sin(\theta) + c \cos^2(\theta) + d \cos(\theta) \sin(\theta) + e \sin^2(\theta), \quad (11)$$

where  $a = \mathbf{P}_{22} \pm \mathbf{P}_{11}$ ,  $b = \mathbf{P}_{12} \pm \mathbf{P}_{21}$ ,  $c = 0.5(\mathbf{Q}_{11} + \mathbf{Q}_{44}) \pm \mathbf{Q}_{14}$ ,  $d = -\mathbf{Q}_{12} \pm \mathbf{Q}_{13} \mp \mathbf{Q}_{24} + \mathbf{Q}_{34}$ ,  $e = 0.5(\mathbf{Q}_{22} + \mathbf{Q}_{33}) \mp \mathbf{Q}_{23}$ ,  $\mathbf{r} = \pm \mathbf{Z}(1, :)$ ,  $\mathbf{s} = \mathbf{Z}(2, :)$ ,  $\mathbf{p} = \mathbf{Z}(2, :)$ ,  $\mathbf{u} = \mp \mathbf{Z}(1, :)$ ,  $\mathbf{x} \triangleq [\mathbf{r}; \mathbf{p}] \in \mathbb{R}^{2r \times 1}$ , and  $\mathbf{y} \triangleq [\mathbf{s}; \mathbf{u}] \in \mathbb{R}^{2r \times 1}$ .

Our key strategy is to perform a variable substitution to convert Problem (11) into an equivalent problem that depends on the variable  $\tan(\theta) \triangleq t$ . The substitution is based on the trigonometric identities that  $\cos(\theta) = \pm 1/\sqrt{1 + \tan^2(\theta)}$  and  $\sin(\theta) = \pm \tan(\theta)/\sqrt{1 + \tan^2(\theta)}$ .

The following lemma provides a characterization of the global optimal solution for Problem (11).

{8}------------------------------------------------

**Lemma 5.1.** (Proof in Appendix H.1) We define  $\tilde{F}(\tilde{c}, \tilde{s}) \triangleq a\tilde{c} + b\tilde{s} + c\tilde{c}^2 + d\tilde{c}\tilde{s} + e\tilde{s}^2 + h(\tilde{c}\mathbf{x} + \tilde{s}\mathbf{y})$ , and  $w \triangleq c - e$ . The optimal solution  $\tilde{\theta}$  to (11) can be computed as:  $[\cos(\tilde{\theta}), \sin(\tilde{\theta})] \in \arg \min_{[c,s]} \tilde{F}(c, s)$ , s.t.  $[c, s] \in \{[c_1, s_1], [c_2, s_2], [0, 1], [0, -1]\}$ , where  $c_1 \triangleq \frac{1}{\sqrt{1+(\tilde{t}_+)^2}}$ ,  $s_1 = \frac{\tilde{t}_+}{\sqrt{1+(\tilde{t}_+)^2}}$ ,  $c_2 \triangleq \frac{-1}{\sqrt{1+(\tilde{t}_-)^2}}$ , and  $s_2 \triangleq \frac{-\tilde{t}_-}{\sqrt{1+(\tilde{t}_-)^2}}$ . Furthermore,  $\tilde{t}_+$  and  $\tilde{t}_-$  are respectively defined as:

$$\tilde{t}_+ \in \arg \min_t p(t) \triangleq \frac{a+bt}{\sqrt{1+t^2}} + \frac{w+dt}{1+t^2} + h\left(\frac{\mathbf{x}+t\mathbf{y}}{\sqrt{1+t^2}}\right), \quad (12)$$

$$\tilde{t}_- \in \arg \min_t \tilde{p}(t) \triangleq \frac{a-bt}{\sqrt{1+t^2}} + \frac{w+dt}{1+t^2} + h\left(\frac{\mathbf{x}-t\mathbf{y}}{\sqrt{1+t^2}}\right). \quad (13)$$

We describe our **BSM** to solve Problem (12); our approach can be naturally extended to tackle Problem (13). **BSM** first identifies all the possible breakpoints / critical points  $\Theta$ , and then picks the solution that leads to the lowest value as the optimal solution  $\tilde{t}$ , i.e.,  $\tilde{t} \in \arg \min_t p(t)$ , s.t.  $t \in \Theta$ .

We assume  $\mathbf{y}_i \neq 0$ . If this is not true and there exists  $\mathbf{y}_i = 0$  for some  $i$ , then  $\{\mathbf{x}_i, \mathbf{y}_i\}$  can be removed since it does not affect the minimizer of the problem.

We now show that how to find the breakpoint set  $\Theta$  for  $h(\mathbf{x}) = \lambda \|\mathbf{x}\|_0$ , where  $\lambda \geq 0$ . We also provide additional examples of **BSM** for other different  $h(\mathbf{x})$ . Due to space limitation, we have included them in Appendix B.

### ► Finding the Breakpoint Set for $h(\mathbf{x}) \triangleq \lambda \|\mathbf{x}\|_0$

Since the function  $h(\mathbf{x}) \triangleq \lambda \|\mathbf{x}\|_0$  is scale-invariant and symmetric with  $\|\pm t\mathbf{x}\|_0 = \|\mathbf{x}\|_0$  for all  $t > 0$ , Problem (12) reduces to the following problem:

$$\min_t p(t) \triangleq \frac{a+bt}{\sqrt{1+t^2}} + \frac{w+dt}{1+t^2} + \lambda \|\mathbf{x} + t\mathbf{y}\|_0. \quad (14)$$

Given the limiting subdifferential of the  $\ell_0$  norm function can be computed as  $\partial \|\mathbf{t}\|_0 \in \{\mathbb{R}_{\{0\}}, \mathbb{R}_{\{0\}, \text{else}}\}$  (see Appendix C.5), we consider the following two cases. (i) We assume  $(\mathbf{x} + t\mathbf{y})_i = 0$  for some  $i$ . Then the solution  $\tilde{t}$  can be determined using  $\tilde{t} = \frac{\mathbf{x}_i}{\mathbf{y}_i}$ . There are  $2r$  breakpoints  $\{\frac{\mathbf{x}_1}{\mathbf{y}_1}, \frac{\mathbf{x}_2}{\mathbf{y}_2}, \dots, \frac{\mathbf{x}_r}{\mathbf{y}_r}\}$  for this case. (ii) We now assume  $(\mathbf{x} + t\mathbf{y})_i \neq 0$  for all  $i$ . Then  $\lambda \|\mathbf{x} + t\mathbf{y}\|_0 = 2r\lambda$  becomes a constant. Setting the subgradient of  $p(t)$  to zero yields:  $0 = \nabla p(t) = [b(1+t^2) - (a+bt)t] \cdot \sqrt{1+t^2} \cdot t^\circ + [d(1+t^2) - (w+dt)(2t)] \cdot t^\circ$ , where  $t^\circ = (1+t^2)^{-2}$ . Since  $t^\circ > 0$ , we obtain:  $d(1+t^2) - (w+dt)2t = -(b-at) \cdot \sqrt{1+t^2}$ . Squaring both sides, we obtain the following quartic equation:  $c_4 t^4 + c_3 t^3 + c_2 t^2 + c_1 t + c_0 = 0$  for some suitable  $c_4, c_3, c_2, c_1$  and  $c_0$ . Solving this equation analytically using Lodovico Ferrari's method (WikiContributors), we obtain all its real roots  $\{\tilde{t}_1, \tilde{t}_2, \dots, \tilde{t}_j\}$  with  $1 \leq j \leq 4$ . There are at most 4 breakpoints for this case. Therefore, Problem (14) contains at most  $2r + 4$  breakpoints  $\Theta = \{\frac{\mathbf{x}_1}{\mathbf{y}_1}, \frac{\mathbf{x}_2}{\mathbf{y}_2}, \dots, \frac{\mathbf{x}_r}{\mathbf{y}_r}, \tilde{t}_1, \tilde{t}_2, \dots, \tilde{t}_j\}$ .

## 6 EXPERIMENTS

This section provides numerical comparisons of **OBCD** against state-of-the-art methods on both real-world and synthetic data. We describe the application of  $L_0$  norm-based Sparse PCA (SPCA) in the sequel, while additional applications for nonnegative PCA and  $\ell_1$  norm-based SPCA can be found in Appendix J.

► **Application to  $L_0$  Norm-based SPCA.**  $L_0$  norm-based Sparse PCA (SPCA) is a method that uses  $\ell_0$  norm to produce modified principal components with sparse loadings, which helps reduce model complexity and increase model interpretability (d'Aspremont et al., 2008; Chen et al., 2016). It can be formulated as:  $\min_{\mathbf{X} \in \text{St}(n, r)} \frac{1}{2} \|\mathbf{X}, \mathbf{C}\mathbf{X}\| + \lambda \|\mathbf{X}\|_0$ , where  $\mathbf{C} = \mathbf{A}^\top \mathbf{A} \in \mathbb{R}^{n \times n}$  is the covariance of the data matrix  $\mathbf{A} \in \mathbb{R}^{m \times n}$  and  $\lambda > 0$ .

► **Data Sets.** To generate the data matrix  $\mathbf{A}$ , we consider 10 publicly available real-world or random data sets: 'w1a', 'TDT2', '20News', 'sector', 'E2006', 'MNIST', 'Gisette', 'Caltech', 'Cifar', 'randn'. We randomly select a subset of examples from the original data set. The size of  $\mathbf{A} \in \mathbb{R}^{m \times n}$  are chosen from the following set  $(m, n) \in \{(2477, 300), (500, 1000), (8000, 1000), (6412, 1000), (2000, 1000), (60000, 784), (3000, 1000), (1000, 1000), (500, 1000)\}$ .

► **Compared Methods.** We compare with two existing operator splitting methods: Linearized ADMM (LADMM) (Lai & Osher, 2014; He & Yuan, 2012) and Smoothing Penalty Method (SPM)

{9}------------------------------------------------

| data-m-n | $F_{min}$ | LADMM (id) | SPM (id) | LADMM (rnd) | SPM (rnd) | OBCD-R (id) |
|-|-|-|-|-|-|-|
| $r = 20, \lambda = 1000$ , time limit=30 |  |  |  |  |  |  |
| w1a-2477-300 | 1.5e+04 | 2.0e+03 | 3.90e+03 | 1.48e+03 | 8.02e+03 | 0.00e+00 |
| TDT2-500-1000 | 2.0e+04 | 4.00e+03 | 6.71e+03 | 2.00e+03 | 7.05e+03 | 0.00e+00 |
| 20News-8000-1000 | 2.0e+04 | 3.00e+03 | 3.00e+03 | 5.00e+03 | 6.00e+03 | 0.00e+00 |
| sector-6412-1000 | 2.0e+04 | 1.01e+03 | 3.00e+03 | 1.02e+03 | 1.30e+04 | 0.00e+00 |
| E2006-2000-1000 | 2.0e+04 | 2.10e+03 | 4.00e+03 | 1.16e+01 | 1.20e+04 | 0.00e+00 |
| MNIST-60000-784 | 6.7e+04 | 6.38e+04 | 8.68e+04 | 2.28e+03 | 4.30e+04 | 0.00e+00 |
| Gisette-3000-1000 | 2.1e+05 | 4.11e+05 | 2.02e+05 | 1.19e+05 | 8.65e+04 | 0.00e+00 |
| CoreCaltech-3000-1000 | 1.9e+04 | 9.05e+03 | 2.46e+03 | 3.09e+04 | 0.00e+00 | 0.00e+00 |
| Cifar-1000-1000 | 1.6e+04 | 1.80e+04 | 9.99e+02 | 2.40e+04 | 1.10e+05 | 0.00e+00 |
| randn-500-1000 | 1.4e+04 | 2.53e+04 | 5.81e+04 | 2.22e+04 | 4.92e+04 | 0.00e+00 |

Table 1: Comparisons of relative objective values ( $F(\mathbf{X}) - F_{min}$ ) for  $L_0$  norm-based SPCA across all the compared methods. The 1<sup>st</sup>, 2<sup>nd</sup>, and 3<sup>rd</sup> best results are colored with red, green and blue, respectively.

![Figure 1: Convergence curves for four datasets: (a) w1a-2477-300, (b) TDT2-500-1000, (c) 20News-8000-1000, and (d) sector-6412-1000. Each plot shows the objective value (log scale) over 30 seconds for LADMM(id), SPM(id), LADMM(rnd), SPM(rnd), and OBCD-R(id). OBCD-R(id) consistently achieves the lowest objective value across all datasets, while other methods get stuck in poor local minima.](f519a5be118c846f631c992412353fb9_img.jpg)

Figure 1: Convergence curves for four datasets: (a) w1a-2477-300, (b) TDT2-500-1000, (c) 20News-8000-1000, and (d) sector-6412-1000. Each plot shows the objective value (log scale) over 30 seconds for LADMM(id), SPM(id), LADMM(rnd), SPM(rnd), and OBCD-R(id). OBCD-R(id) consistently achieves the lowest objective value across all datasets, while other methods get stuck in poor local minima.

Figure 1: The convergence curve of the compared methods for solving  $L_0$  norm-based SPCA with  $\lambda = 100$ . No matter how long the algorithms run, the other methods remain trapped in poor local minima.

(Lai & Osher, 2014; Chen, 2012), initialized differently with random and identity matrices, resulting in four variants: LADMM(id), SPM(id), LADMM(rnd), and SPM(rnd). We use a random strategy to find the working set for **OBCD**, initializing it with the identity matrix, resulting in **OBCD-R**(id).

► **Implementations.** All methods are implemented in MATLAB on an Intel 2.6 GHz CPU with 32 GB RAM. However, our breakpoint searching procedure is developed in C++ and integrated into the MATLAB environment<sup>2</sup>, as it requires inefficient element-wise loops in native MATLAB. Code to reproduce the experiments can be found in the **supplemental material**.

► **Experiment Settings.** We compare objective values ( $F(\mathbf{X}) - F_{min}$ ) for different methods after running for 30 seconds, where  $F_{min}$  represents the smallest objective among all methods. For numerical stability in reporting the objectives, we use the count of elements with absolute values greater than a threshold of  $10^{-6}$  instead of the original  $\ell_0$  norm function  $\|\mathbf{X}\|_0$ . We set  $\alpha = 10^{-5}$  for **OBCD**. Full-gradient methods have higher per-iteration complexity but require fewer iterations, while **OBCD**, as a partial-gradient method, has lower per-iteration costs but needs more iterations. Thus, we compare based on CPU time rather than iteration count.

► **Experiment Results.** Table 1 and Figure 1 display accuracy and computational efficiency results for  $L_0$  norm-based SPCA, yielding the following observations: (i) **OBCD-R** delivers the best performance. (ii) Unlike other methods where objectives fluctuate during iterations, **OBCD-R** monotonically decreases the objective function while maintaining the orthogonality constraint. This is because **OBCD** is a greedy descent method for this problem class. (iii) While other methods often get stuck in poor local minima, **OBCD-R** escapes from such minima and generally finds lower objectives, aligning with our theory that our methods locate *stronger stationary points*.

## 7 CONCLUSIONS

In this paper, we introduced **OBCD**, a new block coordinate descent method for nonsmooth composite optimization under orthogonality constraints. **OBCD** operates on  $k$  rows of the solution matrix, offering lower computational complexity per iteration for  $k \geq 2$ . We also provide a novel optimality analysis, showing how **OBCD** exploits problem structure to escape bad local minima and find better stationary points than methods focused on critical points. Under the Kurdyka-Lojasiewicz (KL) inequality, we establish strong limit-point convergence. Additionally, we present two extensions: efficient subproblem solvers for  $k = 2$  and new greedy strategies for working set selection. Extensive experiments demonstrate that **OBCD** outperforms existing methods.

<sup>2</sup>Though we prioritize accuracy over speed, the comparisons remain fair despite using different programming languages. The other methods, based on matrix multiplication and SVD, utilize highly optimized BLAS and LAPACK libraries for the computational platform and compilation architecture.

 Rest of paper (reference and Appendix) is removed.