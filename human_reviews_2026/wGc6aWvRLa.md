# Convergence of Near-Linear Width ReLU Networks with Unbalanced Initialization

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
The optimization of neural networks is fundamental in machine learning. While the conjecture that linear width suffices for convergence has been confirmed in some restrictive settings, a significant gap remains for non-smooth ReLU networks, for which prior works require substantially wider, polynomial-width networks. We significantly narrow this gap by developing an analysis that simultaneously achieves near-linear width and accelerated convergence for two-layer ReLU networks with shared first layer and vector-valued outputs. Our results are enabled by a novel unbalanced Gaussian initialization that tightly controls the kernel shift for the non-smooth ReLU activation. We prove that gradient descent (GD) achieves linear convergence for networks with only $\tilde{\Omega}(Nn/\lambda)$ neurons, where $N$ is the sample size, $n$ is the output dimension, and $\lambda$ denotes the smallest eigenvalue of the (limiting) neural tangent kernel (NTK), which is standard in prior analyses operating in the NTK regime. Within the same framework, Nesterov's accelerated gradient (NAG) attains a provable speedup without sacrificing near-linear width, improving the iteration complexity from $O(n\kappa\log\frac{1}{\epsilon})$ to $O(\sqrt{n\kappa}\log\frac{1}{\epsilon})$, where $\kappa$ is the NTK condition number. Finally, our analysis establishes low-rank adaptivity: by introducing a sketching step at initialization and a subspace analysis, the width requirement reduces to $\tilde{\Omega}(Nr/\lambda)$ for responses of rank $r \ll n$. By tackling the key analytical hurdles of non-smoothness and vector output with a shared first layer, our work substantially tightens the required width for provable convergence in ReLU networks and brings theory closer to long-standing conjectures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proves a conjecture that has been existed for a long time, which states that probably there need only $\Omega(N)$ neurons exist to prove that neural networks are trainable, where $N$ is the number of data. More specifically, they discuss the vector-output setting, and show that $\Omega(Nn/\lambda)$ neurons are enough to guarantee that gradient descent converges to a global optimum, where $N$ is the number of data, $n$ is the dimension of the output, and $\lambda$ is the minimal eigenvalue of the NTK. The proof is quite standard, where they bound the eigenvalue of the NTK and show that along training, the NTK does not change that much. The main innovation comes from not relying on the Frobenius norm bound like the other papers did, but using matrix concentration inequalities directly. Another innovation is assymetric initialization. The authors extend their result to accelerated gradients and obtain an acceleration result, and gave a better lower bound when the $Y$ is low rank by sketching.

### Strengths
One apparent strength is, if the proof is correct, it solves a conjecture that was yet remained to be solved (of which I have doubts - see weaknesses). 

Lemma 3.6 seems correct and identifies an important inefficiency in proofs of other papers. Other papers have had similar ideas, e.g. see Theorem B.4. in [1]. 

At last, the paper is nice to read because it solves some additional natural questions that one might have in these line of results - e.g. can the proof be extended to accelerate gradients? Can it be extended to low-rank structure (because it is intuitive that for low rank responses we won't need that much neurons). The sketching idea is very interesting too.

[1] Kim, Sungyoon, and Mert Pilanci. "Convex relaxations of relu neural networks approximate global optima in polynomial time." arXiv preprint arXiv:2402.03625 (2024).

### Weaknesses
The biggest suspicion that I have is about Lemma 3.10. I have read [2] and [3] and I believe both papers use Lemma 3.10, provided that **they are only training the first-layer weight**. In this paper the authors discuss the problem where they are training both first and second later weights. So I don't understand how Lemma 3.10. can be directly applied. Maybe it could be applied, or maybe it holds - at least some clarification is needed. Also, the form is not identical to the statement in [2]; there is no statement about weight bounds and there is only one parameter $R_1$ in [2]. So I think it would be good if the authors derive the Lemma again.

Some minor comments:

- In Table 1, certain results use random initialization (e.g. [4]), while others (especially this paper) uses a different initialization. I think it would be good if clarified. 
- I think it would be good to add results that discuss loss landscapes in similar regime, e.g. [5]. [5] shows that if $m > N$ for regularized neural networks, all local optima are global. Does this imply that gradient descent will find a global optimum always? Why or why not?
- [6] also has coupled initialization: how is this different from the initialization in this paper? I think discussion is needed, because they seem very similar to me.

[2] Jun-Kun Wang, Chi-Heng Lin, and Jacob D Abernethy. A modular analysis of provable acceleration via polyak’s momentum: Training a wide relu network and a deep linear network. In International Conference on Machine Learning, pp. 10816–10827. PMLR, 2021.

[3] Song, Z. and Yang, X. Quadratic suffices for over-parametrization via matrix chernoff bound

[4] Du, Simon S., et al. "Gradient descent provably optimizes over-parameterized neural networks." arXiv preprint arXiv:1810.02054 (2018).

[5] Haeffele, Benjamin D., and René Vidal. "Global optimality in tensor factorization, deep learning, and beyond." arXiv preprint arXiv:1506.07540 (2015).

[6] Munteanu, Alexander, et al. "Bounding the width of neural networks via coupled initialization a worst case analysis." International Conference on Machine Learning. PMLR, 2022.

### Questions
How feasible would it be to extend the result to random initialization?

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper significantly narrows the theoretical gap for training two-layer ReLU networks by proving that a near-linear network width suffices for global convergence, moving closer to the long-standing conjecture that linear width is sufficient. The authors achieve this through a novel unbalanced Gaussian initialization scheme that biases training towards the Neural Tangent Kernel (NTK) regime, enabling tight control over kernel dynamics despite the non-smooth ReLU activation.

### Strengths
1.  The paper makes a contribution in closing the gap between theory and practice for ReLU networks. By proving convergence with a near-linear width of $\tilde \Omega (Nn/\lambda)$, it improves upon the best-known prior quadratic and polynomial bounds, addressing a long-standing conjecture in the field. 

2. The introduction of the unbalanced initialization scheme biases the training dynamics into a regime where the kernel shift can be tightly controlled. This core idea enables the entire analysis and is a creative solution to the fundamental hurdle of non-smoothness in ReLU networks.

### Weaknesses
1. The near-linear width bound scales with $1/\lambda$, where $\lambda$ is the smallest eigenvalue of the limiting NTK.  In practice, $\lambda$ can be extremely small for real-world datasets, potentially making the required width large again and limiting the practical relevance of the theoretical guarantee. 

2. While the analysis is a major step for two-layer ReLU networks, the modern deep learning landscape is dominated by deeper architectures and losses like cross-entropy. The paper does not provide a clear pathway for extending its novel techniques (unbalanced initialization, subspace analysis) to these more complex and practical settings, which limits the immediate significance of the contributions for broader applications.

### Questions
1. The width requirement's dependence on $1/\lambda$ is a significant bottleneck, as $\lambda$ can be exceedingly small. Do you have any empirical evidence or theoretical intuition suggesting that this dependence is unavoidable for your method, or is it an artifact of the analysis? 

2. The paper's analysis is a landmark for two-layer networks. What do you perceive as the primary obstacles in extending your core techniques—specifically the unbalanced initialization and the tight control of kernel shift—to deep ReLU networks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work studies the convergence rate and required network width of ReLU networks in the NTK regime. Under the assumption of unbalanced initialization, they proved that the required network width scales linearly with the number of data samples, up to a logarithmic factor.
Furthermore, they provided an accelerated convergence result under the Nesterov’s Accelerated Gradient (NAG) setting.

### Strengths
To the best of my understanding, the result for non-differentiable activation functions is novel. Although the reviewer has not fully verified the details of the proof, no obvious errors were found. The convergence result of the NTK with ReLU activation clearly contributes to our understanding of neural networks. In addition, the extension to the multidimensional case is a valuable and meaningful result.

### Weaknesses
I believe that the justification for the unbalanced initialization is rather insufficient. 
There are two major conditions that must be satisfied under this assumption. 
First, the matrices $W$ and $V$ are required to satisfy specific *symmetry* and *anti-symmetry* conditions. 
Second, the coefficients $c_1$ and $c_2$ must satisfy three inequalities given in Equation (5). 
In my opinion, both assumptions are somewhat artificial and overly restrictive in order to make the theoretical result hold. 

Regarding the inequality conditions, the requirement $c_1 c_2 = \Omega(\sqrt{N})$ significantly deviates from the initialization schemes commonly used in practice (e.g., He initialization). 
Moreover, the paper provides insufficient discussion on the plausibility of the symmetry and anti-symmetry constraints imposed on $W$ and $V$. 
Are these assumptions realistic? Do real-world neural network initializations actually satisfy them? (I would argue that they clearly do not.) 
If they do not, how sensitive are the main results to deviations from these assumptions? 

The paper does not appear to provide a satisfactory answer to these important questions.

When the data $Y$ exhibits a low-rank structure, assuming that the initialization already aligns with $Y$ is an overly strong assumption. 
The authors should discuss how the training dynamics behave---specifically, whether and how the network converges to the desired subspace---when the initialization is not already within that subspace.

### Questions
1. Can the unbalanced initialization be theoretically or empirically justified?

2. If not, how could the result be improved by relaxing or weakening this assumption?

3. Could the authors provide a more appropriate explanation for the initialization associated with $Y$?

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
This work analyzes convergence for near-linear width two-layer ReLU networks using unbalanced initialization. With the same initialization scheme, the paper proves Nesterov accelerated gradient enjoys the usual quadratic speedup while still only needing near-linear width unlike prior work which needed much larger width. In addition, the paper extends the convergence analysis to low-rank kernel regimes via a subspace analysis.

### Strengths
The unbalanced initialization scheme is interesting as in practice (or at least in pytorch default) the initialization scheme is also unbalanced (namely for a two layer ReLU network with width m, the outer layer is scaled by $\frac{1}{\sqrt{m}$ while the inner layer is scaled by $\frac{1}{\sqrt{d}}$).

The extension to the low rank setting as well as the techniques used to deal with this setting are interesting.

### Weaknesses
The weaknesses are readability of the results and some correctness concerns regarding the proof
(see questions).

### Questions
1. What are the constants $C_1$ and $C_2$ (as used in lemma 3.12)?

2. In Theorem 3.1 and 3.2, can you clarify the dependence of $m$ on $c_1$ and $c_2$? 

3. Why is there no upper bound constraint on $c_1$ in Theorem 3.1 and 3.2? To be more concrete,
the gradients of the risk with respect to the inner layer $W$ scales with $c_1$. Hence, the amount that the inner layer moves (namely $R_2$) should scale with $c_1$. If $c_1$ is allowed to be arbitrarily large, every activation can change and hence the kernel shift can be arbitrarily bad. Am I missing something here?

### Soundness
3

### Presentation
2

### Contribution
2
