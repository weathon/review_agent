

{0}------------------------------------------------

# MEMORY-AUGMENTED TRANSFORMERS CAN IMPLEMENT LINEAR FIRST-ORDER OPTIMIZATION METHODS

Anonymous authors

Paper under double-blind review

## ABSTRACT

We show that memory-augmented Transformers (**Memformers**) can implement linear first-order optimization methods such as conjugate gradient descent, momentum methods, and more generally, methods that linearly combine past gradients. Building on prior work that demonstrates how Transformers can simulate preconditioned gradient descent, we provide theoretical and empirical evidence that Memformers can learn more advanced optimization algorithms. Specifically, we analyze how memory registers in Memformers store suitable intermediate attention values allowing them to implement algorithms such as conjugate gradient. Our results show that Memformers can efficiently learn these methods by training on random linear regression tasks, even learning methods that outperform conjugate gradient. This work extends our knowledge about the algorithmic capabilities of Transformers, showing how they can learn complex optimization methods.

## 1 INTRODUCTION

In-context learning (ICL) allows large language models (LLMs) to generate contextually appropriate outputs based solely on examples and queries provided in a prompt, without requiring any parameter adjustments (Brown, 2020; Liu et al., 2021; Lu et al., 2021; Wei et al., 2022; Wu et al., 2022). This remarkable ability has spurred research into understanding how Transformers can implement algorithms (Achiam et al., 2023; Touvron et al., 2023), with recent studies focusing on their capability to simulate optimization algorithms (Dai et al., 2022; Von Oswald et al., 2023a; Garg et al., 2022; Akyürek et al., 2022). Transformers have been shown to implement gradient-based optimization during their forward pass, such as preconditioned gradient descent for linear regression tasks (Dai et al., 2022; Mahankali et al., 2023; Ahn et al., 2024).

More recently, studies have demonstrated that Transformers can learn even more advanced optimization methods. For instance, Fu et al. (2023) showed that Transformers exhibit convergence rates comparable to Iterative Newton’s Method, a higher-order optimization technique that converges exponentially faster than gradient descent for in-context linear regression. Additionally, Vladymyrov et al. (2024) proved that Transformers can, in fact, learn a variant of gradient descent that approximates second-order methods, such as GD<sup>++</sup>, achieving convergence rates similar to Newton’s method. These findings lead to the central question of our paper:

*Can Transformers efficiently “learn” more advanced gradient-based optimization methods?*

We aim to address this question by revealing some of the representational power of Transformers as “algorithm learners,” further motivating the use of machine learning for discovering new optimization algorithms. To make our investigation more precise, we focus on learning the class of gradient-based algorithms obtained by linearly combining past gradients, known as *Linear First-Order Methods (LFOMs)* (Goh, 2017), where the  $(k + 1)$ st iterate is

$$w^{k+1} = w^0 + \sum_{i=0}^k \Gamma_i^k \nabla f(w^i), \quad (1)$$

and where  $\{\Gamma_i^k\}_{i=0}^k$  are diagonal matrices. Model (1) is quite general, as it includes, as special cases, standard methods such as gradient descent (GD), momentum GD, Nesterov’s accelerated gradient, conjugate gradient, and in a stochastic setting, AdaGrad, ADAM, among others.

{1}------------------------------------------------

By “**learning**” an algorithm like CGD or LFOM, we mean two key things:

1. *The Memformer, in its forward pass, under certain internal parameter settings, can perform iterations of CGD and/or LFOM.* This means that its architecture and parameterization are sufficiently expressive to execute these optimization methods as part of its computation.
2. *The Memformer’s learnable parameters can be trained on linear regression tasks. When using these learned parameters, which are shared across all in-context data samples in a batch, the Memformer can execute “CGD-like” and “LFOM-like” iterations during a forward pass.* The surprising aspect lies in the Memformer’s ability to achieve competitive—and in some cases even superior—performance compared to CGD, despite using a relatively small number of learned parameters shared across all test samples drawn independently of the training data.

Our key insight for efficiently learning LFOMs is to leverage memory-augmented Transformers, known as *Memformers* (Wu et al., 2020; Xu et al., 2021), which retain intermediate attention values across layers. This memory enables Memformers to store past gradients, facilitating the execution of advanced first-order methods such as conjugate gradient descent and momentum methods. The same mechanism allows Memformers to implement more general LFOMs.

While unconditional learning of gradient methods remains out of reach, we build on related work demonstrating that Transformers can learn gradient descent in the context of linear regression tasks (Garg et al., 2022; Akyürek et al., 2022; Von Oswald et al., 2023a; Ahn et al., 2024; Zhang et al., 2024). Inspired by these findings, and extending the work of Ahn et al. (2024), we conduct a theoretical analysis of the loss landscape for memory-augmented linear Transformers that omit softmax activation (Schlag et al., 2021; Von Oswald et al., 2023a; Ahn et al., 2024).

In the Appendix, we also include our experiments that Memformers can outperform Nesterov Accelerated Gradient (NAG) and momentum GD. In summary, our main contributions are as follows:

## MAIN CONTRIBUTIONS

- (1) **Theoretical justification that Memformers can implement LFOM iterations, including CGD.** We provide a rigorous theoretical framework showing that Memformers, when trained on linear regression tasks, can be configured to perform iterations of LFOMs in their forward pass, encompassing advanced algorithms like CGD. By leveraging their memory mechanisms, Memformers can store and effectively combine past gradients, enabling them to implement these sophisticated optimization methods within their architecture.
- (2) **Empirical evidence of Memformers “learning” optimization algorithms.** Through extensive experiments, we demonstrate that Memformers can learn LFOMs, in a general sense, by training on random linear regression tasks. *Remarkably, a Memformer utilizing a shared set of learned parameters is able to process batches of in-context data samples and perform competitively with, and in some cases even outperform, the CGD (and NAG) algorithm that is individually optimized for and run separately on each data sample in the test batch.* This finding is particularly **surprising and significant** because CGD tailors its optimization individually for each data sample, whereas the Memformer applies a general optimization strategy learned from the training data across all samples. The ability of Memformers to generalize optimization strategies across data samples using shared parameters highlights their generalization capabilities, which have not been fully recognized in prior research.
- (3) **Enhanced performance through multi-headed attention with theoretical insights.** We show empirically that multi-headed attention improves Memformers’ test performance and offer a heuristic explanation for why increasing attention heads enhances loss performance on test data.

*Our main objective in this paper is to investigate the potential of memory-augmented Transformers to learn advanced optimization algorithms in a general sense. We are not advocating for Transformers as replacements for established optimization methods in practical applications.* Instead, we aim to shed light on the algorithmic capabilities of Transformers, inspiring further exploration into how these architectures can learn and generalize complex algorithms. We believe our results contribute to a deeper understanding of how augmented Transformers can facilitate optimization, which may ultimately lead to the discovery of new and practical gradient-based algorithms.

{2}------------------------------------------------

### 1.1 RELATED WORK

Research on Transformers is extremely active, and we cannot hope to fully capture the breadth of the related literature. Below, we summarize the most immediately relevant topics.

**In-Context Learning.** The ability of Transformer models to perform in-context learning (ICL) has been extensively studied since its introduction by Brown (2020). Subsequent works have explored how these models adapt to new tasks without requiring parameter updates (Xie et al., 2021; Von Oswald et al., 2023b; Hahn and Goyal, 2023; Liu et al., 2021; Lu et al., 2021; Wei et al., 2022; Wu et al., 2022). This foundational research has paved the way for studies investigating how Transformers can implement specific algorithms, such as gradient-based methods.

**Gradient-Based Methods in Transformers.** Garg et al. (2022) analyze the learning of gradient descent within Transformers, particularly in the context of ICL for linear functions. Empirical studies (Garg et al., 2022; Akyürek et al., 2022; Von Oswald et al., 2023a) have shown that Transformers can learn gradient descent after being trained on random linear regression tasks. Expanding on these results, Von Oswald et al. (2023a); Ahn et al. (2024) demonstrate that Transformers can implement preconditioned gradient descent for solving linear regression problems presented in input prompts. Notably, these works—as well as ours—utilize Linear Transformers as discussed in (Schlag et al., 2021; Von Oswald et al., 2023a; Ahn et al., 2023).

**Higher-Order Optimization Methods in Transformers.** Transformers have also been shown to learn higher-order optimization techniques, such as Newton’s method, expanding their capabilities beyond first-order methods (Fu et al., 2023; Giannou et al., 2024; Vladymyrov et al., 2024).

**Memory-Augmented Transformers (Memformers).** Memformers were introduced by Wu et al. (2020); Xu et al. (2021). These models retain intermediate attention values across layers through memory registers, enabling more complex computations and optimization methods to be learned. While significant progress has been made in understanding how Transformers can learn gradient descent, their potential for learning more sophisticated LFOMs remains largely unexplored. Our work addresses this gap by showing how Memformers can efficiently implement a wide range of advanced first-order and quasi-second-order optimization techniques, including CGD and momentum methods, thereby pushing the boundaries of Transformer-based architectures.

## 2 BACKGROUND AND PROBLEM SETUP

### 2.1 LINEAR TRANSFORMERS ON RANDOM LINEAR REGRESSION

We follow the setup of training Transformers on random instances of linear regression, following the prior works (Garg et al., 2022; Akyürek et al., 2022; Von Oswald et al., 2023a; Ahn et al., 2024). We largely use the notation and formal setup of (Ahn et al., 2024), which we now proceed to recall.

**Data Distribution.** Let  $\mathbf{x}(i) \in \mathbb{R}^d$  represent covariates drawn independently from a distribution  $\mathcal{D}_X$ , and let  $\mathbf{w}^* \in \mathbb{R}^d$  be drawn from  $\mathcal{D}_W$ . The matrix of covariates  $\mathbf{X} \in \mathbb{R}^{(n+1) \times d}$  contains rows  $\mathbf{x}(i)$ . The responses are  $\mathbf{y} = [\langle \mathbf{x}(1), \mathbf{w}^* \rangle, \dots, \langle \mathbf{x}(n), \mathbf{w}^* \rangle] \in \mathbb{R}^n$ . Define the input matrix  $\mathbf{Z}_0$  as:

$$\mathbf{Z}_0 = \begin{bmatrix} \mathbf{x}(1) & \mathbf{x}(2) & \dots & \mathbf{x}(n) & \mathbf{x}(n+1) \\ \mathbf{y}(1) & \mathbf{y}(2) & \dots & \mathbf{y}(n) & 0 \end{bmatrix} \in \mathbb{R}^{(d+1) \times (n+1)}, \quad (2)$$

where the zero corresponds to the unknown response for  $\mathbf{x}(n+1)$ . The task is to predict  $(\mathbf{w}^*)^\top \mathbf{x}(n+1)$  using  $\mathbf{Z}_0$ . The training data consists of pairs  $(\mathbf{Z}_0, (\mathbf{w}^*)^\top \mathbf{x}(n+1))$  for  $\mathbf{x}(i) \sim \mathcal{D}_X$  and  $\mathbf{w}^* \sim \mathcal{D}_W$ .

**Self-Attention Without Softmax.** We focus on the linear self-attention layer, building on (Schlag et al., 2021; Von Oswald et al., 2023a). Let  $\mathbf{Z} \in \mathbb{R}^{(d+1) \times (n+1)}$  be the input matrix of  $n+1$  tokens in  $\mathbb{R}^{d+1}$ . Standard self-attention layer is defined as

$$\text{Attn}_{\text{smax}}(\mathbf{Z}) := W_v \mathbf{Z} M \cdot \text{smax}(\mathbf{Z}^\top W_k^\top W_q \mathbf{Z}), \quad (3)$$

where  $W_v, W_k, W_q \in \mathbb{R}^{(d+1) \times (d+1)}$  are weight matrices, and  $\text{smax}(\cdot)$  denotes the column-wise softmax. The masking matrix  $M$  ensures that the label for  $\mathbf{x}(n+1)$  is excluded is given by

$$M = \begin{bmatrix} \mathbf{I}_n & 0 \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{(n+1) \times (n+1)}. \quad (4)$$

{3}------------------------------------------------

Omitting softmax, the attention mechanism becomes

$$\text{Attn}_{P,Q}(\mathbf{Z}) := PZM(\mathbf{Z}^\top QZ), \quad (5)$$

where  $P = W_v$  and  $Q = W_k^\top W_q$ . This simplified form, as shown in Ahn et al. (2024), can implement preconditioned gradient descent, and it is the one we also use.

**Transformer Architecture.** As in the related work, we also simplify the Transformer to consider only attention layers, using  $L$  layers of linear self-attention with a residual connection. Therefore, for each layer  $\ell$ , the output is updated as

$$\mathbf{Z}_{\ell+1} = \mathbf{Z}_\ell + \frac{1}{n} \text{Attn}_{P_\ell, Q_\ell}(\mathbf{Z}_\ell), \quad \ell = 0, 1, \dots, L-1. \quad (6)$$

Using updates (6), with the input  $\mathbf{Z}_0$ , the final transformer output is

$$\text{TF}_L(\mathbf{Z}_0; \{P_\ell, Q_\ell\}_{\ell=0}^{L-1}) = -[\mathbf{Z}_L]_{(d+1), (n+1)}. \quad (7)$$

The set of parameters  $\{P_\ell, Q_\ell\}_{\ell=0}^{L-1}$  is then learned by minimizing the following training objective:

$$f(\{P_\ell, Q_\ell\}_{\ell=0}^{L-1}) = \mathbb{E}_{(\mathbf{Z}_0, \mathbf{w}^*)} \left[ (\text{TF}_L(\mathbf{Z}_0) + (\mathbf{w}^*)^\top \mathbf{x}(n+1))^2 \right]. \quad (8)$$

Here, the scaling factor  $\frac{1}{n}$  is used only for ease of notation and does not influence the expressive power of the Transformer.

We will utilize the following lemma from Ahn et al. (2024), which demonstrates that multi-layer Transformers simulate preconditioned gradient descent under suitable parameterization. We have provided the full proof of this Lemma 1 in the Appendix for completeness.

$$P_\ell = \begin{bmatrix} \mathbf{B}_\ell & 0_{d \times d} \\ 0 & 1 \end{bmatrix}, \quad Q_\ell = - \begin{bmatrix} \mathbf{A}_\ell & 0 \\ 0 & 0 \end{bmatrix}, \quad \mathbf{A}_\ell, \mathbf{B}_\ell \in \mathbb{R}^{d \times d}. \quad (9)$$

**Lemma 1** (Lemma 1, Ahn et al. (2024)). *Consider an  $L$ -layer linear transformer parameterized by  $\mathbf{A}_0, \dots, \mathbf{A}_{L-1}$ , as in (9). Let  $y_\ell^{(n+1)}$  be the  $(d+1, n+1)$ -th entry of the  $\ell$ -th layer output, i.e.,  $y_\ell^{(n+1)} = [\mathbf{Z}_\ell]_{(d+1), (n+1)}$  for  $\ell = 1, \dots, L$ .*

$$y_\ell^{(n+1)} = -\langle \mathbf{x}^{(n+1)}, \mathbf{w}_\ell^{\text{gd}} \rangle, \quad (10)$$

where the sequence  $\{\mathbf{w}_\ell^{\text{gd}}\}$  is defined as  $\mathbf{w}_0^{\text{gd}} = 0$  and for  $\ell = 1, \dots, L-1$ :

$$\mathbf{w}_{\ell+1}^{\text{gd}} = \mathbf{w}_\ell^{\text{gd}} - \mathbf{A}_\ell \nabla R_{\mathbf{w}^*}(\mathbf{w}_\ell^{\text{gd}}), \quad (11)$$

with the empirical least-squares loss (with  $\mathbf{X} := [\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(n)}] \in \mathbb{R}^{d \times n}$ ):

$$R_{\mathbf{w}^*}(\mathbf{w}) := \frac{1}{2n} \|\mathbf{X}^\top \mathbf{w} - \mathbf{X}^\top \mathbf{w}^*\|^2 = \frac{1}{2n} (\mathbf{w} - \mathbf{w}^*)^\top \mathbf{X} \mathbf{X}^\top (\mathbf{w} - \mathbf{w}^*). \quad (12)$$

### 2.2 LINEAR FIRST-ORDER METHODS

Linear First-Order Methods (LFOMs) (Goh, 2017) are a class of optimization algorithms that linearly combine past gradients for minimizing smooth objective functions. They iteratively update a parameter vector  $\mathbf{w}$  using the gradient of the objective function. The general update rule is

$$\mathbf{w}^{k+1} = \mathbf{w}^k + \alpha_k \mathbf{d}^k, \quad (13)$$

where  $\alpha_k$  is the step size and  $\mathbf{d}^k$  is the update direction, typically related to the gradient  $\nabla f(\mathbf{w}^k)$ . Algorithms within this family differ in how they compute  $\mathbf{d}^k$  and choose  $\alpha_k$ .

LFOMs can be expressed in a cumulative form. For gradient descent, unrolling (13) we get

$$\mathbf{w}^{k+1} = \mathbf{w}^0 - \alpha \sum_{i=0}^k \nabla f(\mathbf{w}^i), \quad (14)$$

while common momentum methods need an additional term incorporating past gradients, yielding

$$\mathbf{w}^{k+1} = \mathbf{w}^0 + \sum_{i=0}^k \gamma_i^k \nabla f(\mathbf{w}^i), \quad (15)$$

where the coefficients  $\gamma_i^k$  weight previous gradients. More advanced methods, or general LFOMs, use diagonal matrices  $\Gamma_i^k$  to coordinate-wise scale each gradient component, i.e.,

$$\mathbf{w}^{k+1} = \mathbf{w}^0 + \sum_{i=0}^k \Gamma_i^k \nabla f(\mathbf{w}^i). \quad (16)$$

{4}------------------------------------------------

**Momentum Methods and Conjugate Gradient Descent (CGD)** Momentum methods accelerate convergence by incorporating a momentum term, modifying the gradient to account for past updates and achieving faster convergence in relevant directions. Conjugate Gradient Descent (CGD), on the other hand, is a first-order method optimized for quadratic minimization, serving as a benchmark for large-scale, sparse linear systems. After an initial steepest descent, CGD generates directions conjugate to previous ones, leading to faster convergence than standard gradient descent. Both are core methods within the LFOM class, summarized below:

#### Momentum Methods

- 1: Initialize  $\mathbf{w}_0, \mathbf{v}_0 = 0$
- 2: **for**  $n = 1, 2, \dots$  **do**
- 3:   Compute the gradient:
 
$$\nabla f(\mathbf{w}_n)$$
- 4:   Update the velocity:
 
$$\mathbf{v}_n = \beta \mathbf{v}_{n-1} - \eta \nabla f(\mathbf{w}_n)$$
- 5:   Update the iterate:
 
$$\mathbf{w}_{n+1} = \mathbf{w}_n + \mathbf{v}_n$$
- 6: **end for**
- 7:  $\beta$ : Momentum coefficient (controls the influence of past gradients)
- 8:  $\eta$ : Learning rate (scales the gradient step size)

#### Conjugate Gradient Descent (CGD)

- 1: Initialize  $\mathbf{w}_0, \mathbf{s}_0 = -\nabla f(\mathbf{w}_0)$
- 2: **for**  $n = 1, 2, \dots$  **do**
- 3:   Compute the steepest descent direction:
 
$$\Delta \mathbf{w}_n = -\nabla f(\mathbf{w}_n)$$
- 4:   Compute the conjugacy coefficient:
 
$$\gamma_n = \frac{\|\nabla f(\mathbf{w}_n)\|^2}{\|\nabla f(\mathbf{w}_{n-1})\|^2}$$
- 5:   Update the search direction:
 
$$\mathbf{s}_n = \Delta \mathbf{w}_n + \gamma_n \mathbf{s}_{n-1}$$
- 6:   Perform a line search:
 
$$\alpha_n = \arg \min_{\alpha} f(\mathbf{w}_n + \alpha \mathbf{s}_n)$$
- 7:   Update the iterate:
 
$$\mathbf{w}_{n+1} = \mathbf{w}_n + \alpha_n \mathbf{s}_n$$
- 8: **end for**

Momentum methods provide fast convergence by accumulating gradient history and are widely used in modern optimization. CGD converges in at most  $N$  iterations for quadratic functions, where  $N$  is the number of variables, and is effective for ill-conditioned problems.

## 3 MEMFORMERS CAN IMPLEMENT LFOMS IN-CONTEXT

Memformers can “learn” LFOMs in the specific sense discussed earlier in Section 1. Each layer  $\ell$  of the Memformer has learnable parameters such as  $\mathbf{A}_\ell, \mathbf{B}_\ell$  (9), and  $\alpha_\ell, \gamma_\ell$  (18) or  $\Gamma_\ell$  (20).

Theoretically, in Propositions 1 and 2 below, we show that in their forward pass, under certain parameter configurations, Memformers can implement exact CGD and LFOM iterations. This is indicative of the algorithmic capacities of these architectures. **In experiments, using a small number of learned parameters that are shared across a batch of in-context test data samples, the Memformer can then perform “CGD-like” (3.1) or “LFOM-like” (3.2) iterations that are competitive with, and in some cases even outperform, CGD.**

As noted in (Ahn et al., 2024, Subsection C.1), the term  $\text{Attn}_{P_\ell, Q_\ell}(\mathbf{Z}_\ell)$  in the update for  $\mathbf{Z}_{\ell+1}$  (6) corresponds to the preconditioned gradient  $\mathbf{A}_\ell \nabla R_{\mathbf{w}^*}(\mathbf{w}_\ell^{\text{gd}})$  of the in-context loss (12) in the update for  $\mathbf{w}_{\ell+1}^{\text{gd}}$ .

We will henceforth call the class of algorithms that the following architecture (18) can implement as “**CGD-like**”, and the class of algorithms that architecture (20) can implement as “**LFOM-like**”.

### 3.1 DYNAMIC MEMORY FOR CGD-LIKE ALGORITHMS

**Proposition 1.** *A memory-augmented Transformer can implement Conjugate Gradient Descent (CGD) in its forward pass through a dynamic memory mechanism that recursively refines search*

{5}------------------------------------------------

directions, where the update rules are:

$$\mathbf{R}_\ell = \text{Attn}_{P_\ell, Q_\ell}(\mathbf{Z}_\ell) + \gamma_\ell \mathbf{R}_{\ell-1}, \quad (17)$$

$$\mathbf{Z}_{\ell+1} = \mathbf{Z}_\ell + \alpha_\ell \frac{1}{n} \mathbf{R}_\ell, \quad (18)$$

where  $\gamma_\ell$  and  $\alpha_\ell$  control the influence of past updates and the step size, respectively.

*Proof Sketch.* Here  $\mathbf{R}_\ell$  denotes the state of a single memory register  $\mathbf{R}$  at different layers  $\ell$  during a forward pass. CGD refines search directions using current gradients and previous directions. The Transformer simulates this by using  $\text{Attn}_{P_\ell, Q_\ell}(\mathbf{Z}_\ell)$  as the current update, analogous to the gradient in CGD, and  $\gamma_\ell \mathbf{R}_{\ell-1}$  to refine the previous search direction, corresponding to the recursive update of  $\mathbf{s}_n$  in CGD.

The recursive update for  $\mathbf{R}_\ell$  thus mimics  $\mathbf{s}_n$ , the search direction in CGD. The update for  $\mathbf{Z}_{\ell+1}$  uses  $\mathbf{R}_\ell$ , scaled by  $\alpha_\ell$ , similar to how CGD iterates are updated using  $\mathbf{s}_n$ . With  $\mathbf{A}_\ell = \mathbf{I}$ , this process matches CGD applied to the loss  $R_{w^*}(\mathbf{w})$  (12), using both current and previous gradients to refine the search direction. (A full proof of Proposition 1 is provided in Appendix A.)  $\square$

### 3.2 IMPLEMENTING $k$ STEPS OF LFOM WITH MEMORY REGISTERS

We extend our analysis to show how Transformers can simulate  $k$  steps of Linear First-Order Methods (LFOMs). This is achieved by maintaining a memory register at each layer, which stores accumulated updates from previous layers, simulating iterative optimization.

**Proposition 2.** *A memory-augmented Transformer can implement  $k$  steps of LFOM in its forward pass by maintaining memory registers across layers, where the update rules are:*

$$\mathbf{R}_\ell = \text{Attn}_{P_\ell, Q_\ell}(\mathbf{Z}_\ell), \quad (19)$$

$$\mathbf{Z}_{\ell+1} = \mathbf{Z}_\ell + \frac{1}{n} \sum_{j=0}^{\ell} \Gamma_j^\ell \odot \mathbf{R}_j, \quad (20)$$

where  $\Gamma_j^\ell$  governs the contribution of previous layers, and  $\odot$  is the Hadamard product for scaling.

*Proof Sketch.* Here each  $\mathbf{R}_\ell$  denotes a separate memory register for each layer  $\ell$ . Memformers with this architecture simulate iterative optimization by refreshing the memory register  $\mathbf{R}_\ell$  at each layer with  $\text{Attn}_{P_\ell, Q_\ell}(\mathbf{Z}_\ell)$ , capturing the current update. The cumulative update to  $\mathbf{Z}_{\ell+1}$  incorporates past layers through a weighted sum of previous memory registers  $\mathbf{R}_j$ , with weights  $\Gamma_j^\ell \in \mathbb{R}^{(d+1) \times (n+1)}$ , mimicking LFOM’s cumulative iterative process. We will henceforth refer to this architecture (20) as “**LFOM Memformer**”.

The Hadamard product  $\odot$  modulates the influence of  $\mathbf{R}_j$ , analogous to gradient preconditioning. This setup subsumes the case of diagonal preconditioners  $\Lambda_i^k$  acting on gradients  $\nabla R_{w^*}(\mathbf{w}_i^{\text{gd}})$ , which in the general form looks like:

$$\mathbf{w}_{k+1}^{\text{gd}} = \mathbf{w}_0 + \sum_{i=0}^k \Lambda_i^k \nabla R_{w^*}(\mathbf{w}_i^{\text{gd}}). \quad (21)$$

The matrices  $\Gamma_j^\ell \in \mathbb{R}^{(d+1) \times (n+1)}$  and  $\Lambda_i^k \in \mathbb{R}^{d \times d}$  serve similar roles, but their dimensions differ. We expect this Hadamard product memory architecture to be able to perform richer algorithms than LFOMs, though a formal characterization of its full potential remains to be done.

The full proof follows from the cumulative memory structure and the connection between attention and preconditioned gradients, as discussed in the proof steps of Lemma 1. (A full proof of Proposition 2 is provided in Appendix A.)  $\square$

**Remark.** The update (20) could be interpreted as a type of *gated memory*, related to gating in LSTMs and GRUs that also use the Hadamard product to modulate information flow through gates. This similarity suggests that principles from these architectures could help refine memory mechanisms in Transformers, potentially enhancing their ability to handle long-term dependencies in optimization tasks. However, further exploration is needed to fully understand this relationship.

{6}------------------------------------------------

### 3.3 EXPERIMENTAL RESULTS: MEMFORMER PERFORMANCE VS. CGD

In this section, we present our empirical results for Memformers “learning” conjugate gradient descent (CGD), general linear first-order methods (LFOMs), and general LFOMs with  $\text{GD}^{++}$ . The method  $\text{GD}^{++}$  is a quasi-Newton method where the inverse Hessian in Newton’s method is approximated by a truncated Neumann series; for more details on  $\text{GD}^{++}$ , refer to Section A.10 of Von Oswald et al. (2023a).

We consider the in-context loss function (12) for linear regression. The input dimension is set to  $d = 5$ , and the number of training observations in the prompt is  $n = 20$ . Both the inputs  $\mathbf{x}^{(i)}$  and the target weight vector  $\mathbf{w}^*$  are sampled from Gaussian distributions:  $\mathbf{x}^{(i)} \sim \mathcal{N}(0, \Sigma)$  and  $\mathbf{w}^* \sim \mathcal{N}(0, \Sigma^{-1})$ , where  $\Sigma = \mathbf{U}^\top \mathbf{D} \mathbf{U}$ . Here,  $\mathbf{U}$  is a uniformly random orthogonal matrix, and  $\mathbf{D}$  is a fixed diagonal matrix with entries  $\text{diag}(1, 1, 1/2, 1/4, 1)$ .

We optimize the function  $f$  (8) for a three-layer linear transformer using the ADAM optimizer. The matrices  $\mathbf{A}_0$ ,  $\mathbf{A}_1$ , and  $\mathbf{A}_2$  (as in (9)) are initialized with independent and identically distributed (i.i.d.) Gaussian entries. Each gradient step is computed using a batch of size 1000, and we resample the batch every 100 steps. We clip the gradient of each matrix to have a maximum norm of 0.01. All plots are averaged over five runs, each with a different randomly sampled  $\mathbf{U}$  (and thus different  $\Sigma$ ).

Figure 1 illustrates the implementation of a CGD-like algorithm under the architecture given by (18). In Figure 1a, the line-search parameters  $\alpha_\ell$  and deflection parameters  $\gamma_\ell$  for each layer  $\ell$  are obtained by training using ADAM. By “CGD-like,” we mean that upon training the Memformer using ADAM, the Memformer layers learn general parameters  $\alpha_\ell$  and  $\gamma_\ell$  which, while they may not match the exact CGD parameters for individual observations, perform well enough on each observation to be comparable to, if not competitive with, CGD. We further explain the important issue of learning general parameters in Section 4.

Figure 1b presents the same experiment as Figure 1a, using the architecture in (18), but with the parameters  $\mathbf{A}_\ell$  for each layer not restricted to scalars. Thus, past gradients are accounted for, similar to CGD, but with preconditioners  $\mathbf{A}_\ell$ . This is therefore not a “CGD-like” algorithm. We aim to demonstrate that once we allow preconditioned gradients, a Memformer implements a certain “LFOM-like” algorithm that distinctly outperforms CGD.

Figure 2 presents the performance of LFOM Memformer under the architecture in (20), where the matrix parameters  $\Gamma_j$  for each layer  $j$  are obtained by training using ADAM. In our experiments, we consider the special case of  $\Gamma_j^\ell = \Gamma_j \forall \ell$ , which is more natural, if we consider that each layer  $j$  of the Memformer has an associated  $\Gamma_j$ . Figure 2a shows the results on non-isotropic data, and Figure 2b shows the results on isotropic data. Note that this algorithm is quite similar in nature to the previous case in Figure 1b. Here, the  $\Gamma_j$ ’s essentially act as preconditioners of the gradients computed in each layer. Consequently, the graphs of Figures 1b and 2a are nearly identical. In the isotropic data experiment (Figure 2b), we observe that the Memformer does not perform better than a linear transformer. In quadratics with isotropic data, there is no significant variation in curvature across directions; thus, incorporating past gradients via momentum offers little advantage. Momentum is more beneficial in cases with non-isotropic data.

Figure 3 presents LFOM Memformer with  $\text{GD}^{++}$  under the architecture in (20), where the  $\mathbf{B}_\ell$  blocks in the  $\mathbf{P}_\ell$  matrices for each layer  $\ell$  (9) are allowed to be non-zero. Once again, the matrix parameters for each layer  $\ell$  are obtained by training using ADAM. In this case, the  $\mathbf{B}_\ell$  matrices resemble a heavily truncated Neumann series of the inverse  $\mathbf{X}\mathbf{X}^\top$  (Hessian of (12)), resulting in a quasi-Newton method. The experiments are conducted on both non-isotropic data (Figure 3a) and isotropic data (Figure 3b).

## 4 EXPERIMENTS: INFLUENCE OF BATCH SIZE ON PERFORMANCE

We emphasize here that the results presented in Section 3.3 compare the performance of Transformers and Memformers (which learn shared generic parameters upon training) against CGD that runs on fresh observations of batch size  $B = 1000$ , independently resampled from the same distribution. But unlike CGD that computes specific parameters for each observation, the Transformer and Memformer models learn shared parameters  $\mathbf{P}_\ell$ ,  $\mathbf{Q}_\ell$  (and  $\alpha_\ell$ ,  $\gamma_\ell$ , or  $\Gamma_\ell$ ) for each layer  $\ell$ , and these parameters are applied uniformly across all 1000 observations in the batch. In contrast, CGD is executed individually on each of the 1000 observations in the batch, and the average log-loss versus layers is plotted.

{7}------------------------------------------------

![Figure 1: Comparison of Linear Transformer and CGD Memformer (18) with general CGD-like parameters to actual CGD running separately on each test observation. (a) Without Preconditioning: log(Loss) vs Number of Layers/Steps (1 to 4). (b) With Preconditioning: log(Loss) vs Number of Layers/Steps (1 to 4).](3121afa7ca030b22ee0345864ca6f38b_img.jpg)

Figure 1 consists of two line plots, (a) and (b), showing the log(Loss) on the y-axis against the Number of Layers/Steps on the x-axis (ranging from 1 to 4). Both plots compare three methods: Conjugate Gradient Descent (blue line), Memformer with CGD (red line), and Linear Transformer (green line).

(a) Without Preconditioning: The y-axis ranges from -1.5 to 0.5. The Linear Transformer (green) starts at approximately 0.5 and decreases to about -0.1. The Memformer with CGD (red) starts at approximately 0.5 and decreases to about -0.4. The Conjugate Gradient Descent (blue) starts at approximately 0.5 and decreases much more sharply to about -1.5.

(b) With Preconditioning: The y-axis ranges from -2.0 to 0.5. The Memformer with CGD (red) starts at approximately 0.2 and decreases to about -2.0. The Conjugate Gradient Descent (blue) starts at approximately 0.5 and decreases to about -1.5. The Linear Transformer (green) starts at approximately 0.2 and decreases to about -1.1.

Figure 1: Comparison of Linear Transformer and CGD Memformer (18) with general CGD-like parameters to actual CGD running separately on each test observation. (a) Without Preconditioning: log(Loss) vs Number of Layers/Steps (1 to 4). (b) With Preconditioning: log(Loss) vs Number of Layers/Steps (1 to 4).

Figure 1: Comparison of Linear Transformer and CGD Memformer (18) with general CGD-like parameters to actual CGD running separately on each test observation.

![Figure 2: LFOM Memformer (20) performance on non-isotropic vs. isotropic test data (Pre = with non-trivial preconditioners). (a) Non-Isotropic Data: log(Loss) vs Number of Layers/Steps (1 to 4). (b) Isotropic Data: log(Loss) vs Number of Layers/Steps (1 to 4).](d864789b0d8384da1d22fd6a5d76bbdf_img.jpg)

Figure 2 consists of two line plots, (a) and (b), showing the log(Loss) on the y-axis against the Number of Layers/Steps on the x-axis (ranging from 1 to 4). Both plots compare three methods: Conjugate Gradient Descent (blue line), LFOM Memformer (red line), and Linear Transformer (Pre) (green line).

(a) Non-Isotropic Data: The y-axis ranges from -2.0 to 0.5. The LFOM Memformer (red) starts at approximately 0.2 and decreases to about -2.0. The Linear Transformer (Pre) (green) starts at approximately 0.2 and decreases to about -1.1. The Conjugate Gradient Descent (blue) starts at approximately 0.5 and decreases to about -1.5.

(b) Isotropic Data: The y-axis ranges from -6 to 0. The LFOM Memformer (red) starts at approximately 0.2 and decreases to about -1.8. The Linear Transformer (Pre) (green) starts at approximately 0.2 and decreases to about -1.8. The Conjugate Gradient Descent (blue) starts at approximately 0.2 and decreases much more sharply to about -6.

Figure 2: LFOM Memformer (20) performance on non-isotropic vs. isotropic test data (Pre = with non-trivial preconditioners). (a) Non-Isotropic Data: log(Loss) vs Number of Layers/Steps (1 to 4). (b) Isotropic Data: log(Loss) vs Number of Layers/Steps (1 to 4).

Figure 2: LFOM Memformer (20) performance on non-isotropic vs. isotropic test data (Pre = with non-trivial preconditioners). Test data is independently sampled from the same distribution as the training data.

The strength of LFOM Memformers (20) (with matrices  $\Gamma_\ell$  restricted to scalar multiples of the identity) becomes even more pronounced when tested on training data with small batch sizes, such as  $B = 1$  and  $B = 10$ . In these scenarios, the Memformers learn parameters that significantly outperform CGD running in parallel on each of the observations in those small batches. Figure 4 demonstrates this comparison. We further provide an experimental comparison of LFOM Memformer performance vs. Nesterov Accelerated Gradient Method and Momentum GD in the Appendix.

## 5 EXPERIMENTS: IMPACT OF USING MULTI-HEADED ATTENTION

Our experiments show that increasing the number of attention heads improves test loss performance. Multi-head attention enables Transformers to learn diverse preconditioning matrices, better adapting to varying data covariance structures. In our architecture (17), attention values from each head are summed into the memory register  $\mathbf{R}_\ell$  at each layer. Heuristically, each head captures different aspects of the data, estimating gradients from multiple perspectives. This ensemble-like behavior reduces variance in gradient updates by averaging out individual noise and biases, leading to faster

{8}------------------------------------------------

![Figure 3: LFOM Memformer (20) GD++ performance on non-isotropic vs. isotropic test data. The figure contains two line plots. Plot (a) for Non-Isotropic Data shows log(Loss) vs. Number of Layers/Steps (1 to 4). Plot (b) for Isotropic Data shows log(Loss) vs. Number of Layers/Steps (1 to 4). Both plots compare Conjugate Gradient Descent (blue), Linear Transformer (Pre) GD++ (green), and LFOM Memformer with GD++ (red). In both cases, the Memformer (red) achieves the lowest loss, followed by the Linear Transformer (green), and then Conjugate Gradient Descent (blue).](b93cbfb52e37619e688175a6aad9edd9_img.jpg)

Figure 3: LFOM Memformer (20) GD++ performance on non-isotropic vs. isotropic test data. The figure contains two line plots. Plot (a) for Non-Isotropic Data shows log(Loss) vs. Number of Layers/Steps (1 to 4). Plot (b) for Isotropic Data shows log(Loss) vs. Number of Layers/Steps (1 to 4). Both plots compare Conjugate Gradient Descent (blue), Linear Transformer (Pre) GD++ (green), and LFOM Memformer with GD++ (red). In both cases, the Memformer (red) achieves the lowest loss, followed by the Linear Transformer (green), and then Conjugate Gradient Descent (blue).

Figure 3: LFOM Memformer (20) GD++ performance on non-isotropic vs. isotropic test data (Pre = with non-trivial preconditioners). Test data is independently sampled from the same distribution as the training data.

![Figure 4: LFOM Memformer (20) with scalar preconditioners Γℓ vs. CGD performance on small batch training data. The figure contains two line plots. Plot (a) for Batch Size B = 1 shows log(Loss) vs. Number of Layers/Steps (1 to 4). Plot (b) for Batch Size B = 10 shows log(Loss) vs. Number of Layers/Steps (1 to 4). Both plots compare Conjugate Gradient Descent (blue) and Scalar Preconditioning Memf (red). In both cases, the Scalar Preconditioning Memf (red) achieves a significantly lower loss than Conjugate Gradient Descent (blue).](bedcca5cdf168e3508ef511d94ec514c_img.jpg)

Figure 4: LFOM Memformer (20) with scalar preconditioners Γℓ vs. CGD performance on small batch training data. The figure contains two line plots. Plot (a) for Batch Size B = 1 shows log(Loss) vs. Number of Layers/Steps (1 to 4). Plot (b) for Batch Size B = 10 shows log(Loss) vs. Number of Layers/Steps (1 to 4). Both plots compare Conjugate Gradient Descent (blue) and Scalar Preconditioning Memf (red). In both cases, the Scalar Preconditioning Memf (red) achieves a significantly lower loss than Conjugate Gradient Descent (blue).

Figure 4: LFOM Memformer (20) with scalar preconditioners  $\Gamma_\ell$  vs. CGD performance on small batch training data ( $B = 1$  and  $B = 10$ ). The Memformer demonstrates superior performance on the training data.

convergence and more stable optimization. Acting as implicit regularization, it prevents overfitting and enhances generalization on test data. This phenomenon is also supported by recent studies. Chen et al. (2024) showed that multi-head attention is essential for effective context preprocessing in sparse linear regression, aligning with our findings. Similarly, Cui et al. (2024) provided theoretical and empirical evidence that multi-head attention outperforms single-head attention in in-context learning.

Figure 5 compares models with 1-head and 5-head attention, illustrating the benefits of multiple heads on convergence speed and test loss performance.

## 6 DISCUSSION AND FUTURE WORK

This work demonstrates the capability of memory-augmented Transformers (Memformers) to implement a broad range of first-order optimization methods, opening several research directions. We briefly comment on some of these aspects below.

{9}------------------------------------------------

![Figure 5: Comparison of LFOM Memformer (with scalar preconditioners Γℓ) performance using 1-head and 5-head attention, relative to CGD. The figure contains two line plots. Plot (a) shows 1-Head Attention Performance with log(Loss) on the y-axis (ranging from -1.5 to -0.7) and Number of Layers/Steps on the x-axis (1 to 4). Plot (b) shows 5-Head Attention Performance with log(Loss) on the y-axis (ranging from -2.2 to -0.8) and Number of Layers/Steps on the x-axis (1 to 4). Both plots compare Conjugate Gradient Descent (blue line) and Scalar Preconditioning Memf (red line). In both cases, the red line (Memf) shows a more significant decrease in log(Loss) as the number of layers/steps increases, indicating better performance relative to CGD.](4e0ade2f41b66d5602160da5cc978274_img.jpg)

Figure 5: Comparison of LFOM Memformer (with scalar preconditioners Γℓ) performance using 1-head and 5-head attention, relative to CGD. The figure contains two line plots. Plot (a) shows 1-Head Attention Performance with log(Loss) on the y-axis (ranging from -1.5 to -0.7) and Number of Layers/Steps on the x-axis (1 to 4). Plot (b) shows 5-Head Attention Performance with log(Loss) on the y-axis (ranging from -2.2 to -0.8) and Number of Layers/Steps on the x-axis (1 to 4). Both plots compare Conjugate Gradient Descent (blue line) and Scalar Preconditioning Memf (red line). In both cases, the red line (Memf) shows a more significant decrease in log(Loss) as the number of layers/steps increases, indicating better performance relative to CGD.

Figure 5: Comparison of LFOM Memformer (20) (with scalar preconditioners  $\Gamma_\ell$ ) performance using 1-head and 5-head attention, relative to CGD.

- (i) **Architectural Flexibility:** Small modifications, such as (gated) memory registers, significantly enhance Transformers’ ability to learn and implement diverse optimization algorithms. Future research could explore further architectural innovations to unlock even greater capabilities.
- (ii) **General Function Classes:** While our approach successfully makes Transformers implement LFOMs on quadratic functions, future work should extend this to more general objective functions. Doing so may require novel training strategies, and possibly architectural adjustments to handle non-quadratic functions. The role of nonlinear attention and the MLP component of Transformers may also prove to be useful here.
- (iii) **Efficiency vs. Generalization:** Attention-based methods require more computation than directly implementing conjugate gradient descent or momentum GD. However, Transformers excel in learning general parameters, enabling LFOMs to generalize across new data without needing per-instance optimization. Exploring practical use of such “learned optimizers” to either warmstart a solver, or to potentially even bypass it, is a tantalizing research topic.
- (iv) **Theoretical Foundations and Convergence Analysis:** Strengthening the theoretical basis of Transformers’ optimization capabilities, including convergence analysis and their alignment with classical optimization theory, is another important direction for future research.
- (v) **Meta-learning and Transfer Learning:** The ability of Transformers to learn and generalize optimization algorithms offers exciting potential for meta-learning and transfer learning, providing new opportunities in areas where traditional optimization methods fall short.

### 6.1 LIMITATIONS

We briefly remark on some limitations of our current framework. For instance, while Memformers are quite versatile, our experiments (Figures 1, 2) indicate they do not radically outperform preconditioned GD on general quadratic problems as in (12), where the preconditioner matrix  $\Gamma_\ell$  (and likewise,  $\mathbf{A}_\ell$ ) for the current layer  $\ell$  is the main contributor to loss performance at each update step  $\ell$  (17). On the other hand, this behavior is likely due to the task being quadratic, and a future study that tackles more general ICL formulations will likely shed light here.

Transformers can implement second-order methods like Newton’s method (Fu et al., 2023; Giannou et al., 2024), which typically outperform LFOMs in convergence speed and accuracy. However, we reiterate that the main focus of our paper is to explore the space of first-order optimization algorithms that augmented Transformers can learn, as opposed to looking for “the best” algorithm.

 Rest of paper (reference and Appendix) is removed.