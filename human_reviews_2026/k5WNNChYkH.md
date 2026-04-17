# Training Overparametrized Neural Networks in Sublinear Time

- Decision: Reject
- Scores: 8, 6, 4, 6, 4

## Abstract
The success of deep learning comes at a tremendous computational and energy cost, and the scalability of training massively overparametrized neural networks is becoming a real  barrier to the progress of artificial intelligence (AI). Despite the popularity and low cost-per-iteration of traditional backpropagation via gradient decent, stochastic gradient descent (SGD) has prohibitive convergence rate in non-convex settings, both in theory and practice.  To mitigate this cost, recent works have  proposed to employ alternative (Newton-type) training methods with much faster convergence rate, albeit with higher cost-per-iteration. 
For a neural network with $m=\mathrm{poly}(n)$ parameters and input batch of  $n$  datapoints in $\mathbb{R}^d$, the previous work of [Brand, Peng, Song and Weinstein, ITCS 2021] requires $\sim mnd + n^3$ time per iteration. In this paper, we present a novel training method that requires only $m^{1-\alpha} n d + n^3$ amortized time in the same overparametrized regime, where $\alpha \in (0.01,1)$ is some fixed constant. This method relies on a new and alternative view of neural networks, as a set of binary search trees, where each iteration corresponds to modifying a small subset of the nodes in the tree. We believe this view would have further applications in the design and analysis of deep neural networks (DNNs). Finally, we give a lower bound for the dynamic sensitive weight searching data structure we make use of, showing that under ${\sf SETH}$ or ${\sf OVC}$ from fine-grained complexity, one cannot substantially improve our algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work introduces an algorithm for Newton-type neural network training that reduces the time per iteration in an over-parametrized regime from linear-time in the network size to sub-linear time while maintaining the same convergence rate.

### Strengths
The presented algorithm improves prior bounds by leveraging an alternative, yet simple view of neural networks as binary trees. This view may have further applications in neural network design and analysis.

### Weaknesses
Comments:
- As far as I can tell, the key assumption here is an 2-layer overparametrized DNNs for which the Jacobian of the loss is sparse (L248). I might have missed it but where does this observation come from, in particular the assumed concrete value range c=[0.1, 1]? Being unfamiliar with the context of this work, the manuscript could benefit from elaborting on this assumption.
- I suggest adding an accesible illustration of the binary tree structure as it relates to the neural network as it appears to be the key insight underpinning the presented approach. 
- L035 states that success "is approaching its limit and is largely compromised by the computational complexity of these resource-hungry model" - I suggest qualifying this statement or elaborate; true, resources are an issue but it is far from clear that we have reached a structural limit at this point.

### Questions
- Can you think of any numerical toy experiments that could give some empirical illustration of a case where the CPI scales sub-linearly in the network size? Or do you have any intuition on the conditions that make sub-linear time feasible; it does seem counter-inuitive.
- The abstract teases further applications of the binary tree view, but I am unsure if this is ever further discussed in the paper. Could you please elaborate on this?

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
This paper proposes a new theoretical framework for training large neural networks more efficiently by achieving sublinear time per iteration. The authors reinterpret neural networks as collections of binary search trees, where only a small subset of nodes requires updates due to the sparsity of activations, drastically reducing computational overhead. Using this representation and a specialized dynamic data structure for managing weight updates, the algorithm attains an amortized cost-per-iteration of $O(\tilde{m}^{1-\alpha}nd + n^3)$, improving upon the prior best linear-time methods. The approach integrates sketch-based regression and implicit weight maintenance to maintain convergence guarantees comparable to Gauss-Newton optimization. Theoretical analysis shows linear convergence of the loss and establishes a near-optimal lower bound under standard fine-grained complexity assumptions, suggesting the proposed method approaches the fundamental computational limits of neural network training.

### Strengths
1. The paper introduces the framework achieving sublinear-time per iteration for training overparameterized neural networks, marking a notable improvement over existing linear-time methods.

2. Reinterpreting neural networks as binary search trees is a creative and powerful abstraction.

3. The method maintains provable convergence at a linear rate comparable to second-order (Gauss-Newton) methods while reducing computational cost.

### Weaknesses
1.The paper presents only theoretical analysis without any experimental results or empirical benchmarks

2.  The guarantees rely heavily on random initialization and activation sparsity, which may not hold consistently in all data or model settings.

### Questions
1. Have you implemented the proposed sublinear-time training algorithm in experiments? Can it be evaluated in ImageNet dataaset?

2.The results rely on random initialization and sparsity of activations. How sensitive is the algorithm to the choice of initialization or the ReLU bias parameter b?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors have studied 2-layer neural networks in overparameterized settings. They propose a binary search for training the network sublinear runtime through observing the sparsity in the Jacobian.

### Strengths
One of the key strengths is to improve cost-per-iteration and establishing lower bounds for the dynamic sensitive weight searching data structure.

### Weaknesses
The authors have claimed that this is a natural assumption that the parameters of the second layer of the neural network are fixed during training. However, this is not a natural assumption and it is not sufficient to claim that this is a natural assumption because some papers have made such an assumption. For example:

In Section 2.3 of [Du et al., 2019], the authors have extended to settings when both layers are trained simultaneously. However, the authors have included [Du et al., 2019] in Section 3.1 after Definition 3.1. As an example, the following related work have not made such assumptions.  

S.S. Du, X. Zhai, B. Poczos, and A. Singh. Gradient descent provably optimizes over-parameterized neural networks. ICLR 2019.

C. Song, A. Ramezani-Kebrya, T. Pethick, A. Eftekhari, and V. Cevher. Subquadratic overparameterization for shallow neural networks. NeurIPS 2021.

The lack of numerical experiments is a major weakness. The improved overall runtime should be demonstrated by computing wall-clock timing results when training a neural network under a meaningful and practical setting.

### Questions
Please check the weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposed treating the training of a NN as a binary tree search in sublinear time by taking advantage of the sparsity in the Jacobian because weights change very slowly and the initial partitions largely stay the same, hence can be exploited.  
The paper's analysis focuses on two-layer ReLU networks, with the second layer fixed.  
With a shift parameter $b=\sqrt{0.48\log m}$, each input activates only $O(m^{0.76})$ neurons, leading to an amortized sublinear cost per iteration while preserving Gauss--Newton--type convergence.  
The $m^{0.76}$ exponent is not fundamental but a result of the chosen parameterization and proof technique, and could vary with tighter analysis or adaptive thresholds.

The main contribution is the connection between neural-network training and dynamic
data-structure problems.  
Casting wide-network optimization as maintaining binary trees of activation thresholds is clever,
and combining it with sketching and regression methods yields an interesting sublinear-time bound.
The lower-bound result strengthens the theoretical side.
The practical impact is uncertain—this remains a clean piece of theory rather than something directly
usable in training large models—but the conceptual insight is worthwhile.

### Strengths
-- Clear and rigorous proofs; every claim has a formal statement and reference.

-- An elegant integration of NTK analysis with ideas from fine-grained complexity and geometric data structures.

 -- The conditional lower bound adds credibility that the speedup is close to the theoretical limit.

-- Very readable and organized; notation and assumptions are transparent.

-- The paper may stimulate further work on algorithmic aspects of training dynamics.

### Weaknesses
--  The reliance on shifted ReLU ($\phi(x)=\max\{x,b\}$) makes the theory less aligned with
        practical ReLU networks.

--  The second-layer weights are fixed, which simplifies but narrows the results.

-- No empirical demonstration or timing experiment, even small synthetic ones.

-- The novelty of the binary-tree idea is moderate, as similar data-structure tools exist,
        though the integration here is fresh.

-- Limited discussion of how such methods might actually be implemented or approximated
        in modern training pipelines.

### Questions
-- Can the same sparsity argument work with unshifted ReLU or smooth activations?

-- Could the $n^3$ term be reduced by using iterative solvers with Jacobian–vector products?

-- Would small experiments (even synthetic) help confirm that the constants are reasonable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a theoretically motivated algorithm for training overparameterized neural networks in sublinear time. By introducing a set of binary trees to encode the relationship between model's weights and training data, the proposed algorithm improve the cost-per-iteration (CPI) from $O(mnd + n^3)$ to $O(m^{1-\alpha}nd + n^3)$, where $\alpha \in (0.01,1)$. Convergence, running time and lower bound analyses are given to support the effectiveness of the training algorithm.

While the motivation is interesting and the attempt to connect algorithmic efficiency with theoretical guarantees is valuable, the manuscript suffers from several critical weaknesses in both theoretical clarity and presentation accuracy. The main issues lie in the lack of experimental validation, some mathematical inconsistencies, and conceptual confusion regarding the model setup.

### Strengths
Overall, the paper is clearly and fluently written, not only proposing a novel training algorithm but also providing a comprehensive theoretical foundation. The strengths are listed as follows:
1. Interesting theoretical motivation: the goal of achieving sublinear-time training for overparameterized models is theoretically meaningful.
2. Innovative insight: introduce a binary-tree based dynamic data structure to accelerate the training of overparameterized neural networks effectively.
3. Complete theoretical analyses: the loss function is proved to converges linearly with high probability using the proposed algorithm. Shortest running time (CPI: $O(m^{1-\alpha}nd + n^3)$) is proved to reached comparing to existing studies. The lower bound discussion linking algorithmic runtime with conditional lower bounds (e.g., SETH-based hardness) adds a valuable perspective from computational complexity.

### Weaknesses
The weaknesses are listed as follows:
1. No experimental validation: The paper presents no empirical experiments or numerical simulations to support the theoretical claims. Given that the paper introduces a new data structure and derives a new training algorithm, an empirical demonstration is necessary to validate the practicality of both the new structure and the superiority of the algorithm. For example, report on the comparison of the CPI and the total running time of the proposed algorithm versus the existing ones.
2. Misleading use of “Deep Neural Networks”: The paper’s algorithm and analysis are restricted to a two-layer neural network. Despite this, the authors repeatedly refer to “deep neural networks (DNNs)” and even title the algorithm “A Fast DNN Training Algorithm” (Line 280). This terminology is misleading and scientifically inaccurate because no multi-layer (deep) structure is analyzed or empirically demonstrated. The generalization from 2-layer to multi-layer settings is nontrivial and not theoretically justified in the manuscript.
3. Theoretical results only apply to two-layer shifted ReLU networks with a single output and fixed second layer, which limits their application in many practical cases.
4. Mistakes in mathematical expressions:
    1. Line 215: The dimension of Jacobi matrix seems wrong. It should be $J_t\in\mathbb{R}^{n\times md}$ there since $x\in\mathbb{R}^{d}$.
    2. Line 226: There seems to be a typo of definition of $(G_t)_{i,j}$ : "$\partial$" is missing. 
    3. Line 293&309&314&365: Since the paper uses various vector and matrix norms, it should specify which norm is used, e.g. $\ell_2$-norm $\\|\cdot\\|_2$.

### Questions
Questions and suggestions are given as follows:
1. More discussions and explanations of the choice of parameter $\alpha$ and shifted parameter $b$.
    1. The constraints on $\alpha$ are inconsistent in Theorem 1.1 ($\alpha\in(0.01,1)$) (Line 085), Theorem 6.1 ($\alpha\in[0.1,0.24]$) (Line 392) and Theorem F.1 ($\alpha\in[0.1,0.24)$) (Line 1694).
    2. $b$ is chosen as $b=\sqrt{0.48\log m}$ in Corollary A.17 without explanation. Why is this choice the optimal one?
    3. The choice of $b$ are closely related to the width of networks ($m$) as stated in Corollary A.17. Further discussions are therefore needed to show how to choose $b$ in pratical. For example, From Theorem 5.1, Theorem A.4 and Corollary A.17, one can get $m=O(\lambda^{-4}n^4)$, $\lambda \in [e^{-b^2/2}\cdot\frac{\epsilon}{100n^2},e^{-b^2/2}]$ and $b=\sqrt{0.48\log m}$. Let $\lambda \approx e^{-b^2/2}$, one can get $\lambda^{-1} \approx e^{0.24 \log m} = m^{0.24}$, which implies that $m = O(m^{0.96}n^4)$, that is, $m=O(n^{100})$. Is this acceptable in practice?
2. Can the algorithm and convergence results be extended to the following general cases: multi-layer networks; ReLU without shift or other activation functions; and two-layer networks with a non-fixed second layer?
3. It's better to give the definition of notations and parameters before using them. For example, the use of notations $O,o,\Omega$, the parameter $R$ in Line 335 and abbreviation "TS" in Line 297.

### Soundness
3

### Presentation
2

### Contribution
2
