# A Fast and Scalable Extented Hoyer Projection for Structured Neural Network Sparsity

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
Deep networks require sparsity mechanisms that are both scale-invariant and computationally efficient. Existing approaches based on the Hoyer score rely on non-convex projections, resulting in unstable heuristics and potential convergence issues.

In this paper, we introduce a new Cone Alignement Index (CAI), a convex constraint whose level sets form a Lorentz hypercone. This geometric structure enables the first Closed-Form Projection (CFP) onto such a cone, requiring only a single interpolation step and enjoying guaranteed convergence. We derive analytical expressions for:
(i) computing the active set through a provably correct threshold rule, and
(ii) performing the final projection using a closed-form interpolation coefficient.

Building on this result, we propose a fast bilevel projection method, consisting solely of successive Closed-Form Projection (CFP) algorithms, with guaranteed convergence and naturally inducing hardware-friendly column (or row)-wise sparsity.

Thanks to these Closed-Form Projection (CFP) algorithms, our method is up to 6.5 times faster than the original Hoyer projection on the vector. Our bilevel Closed-Form Projection (CFP) algorithm is 2r times faster than the HALS algorithm on matrices.

 Applied to transformer attention matrices on biomedical and NLP dataset (GLUE benchmark), it achieves up to $96 \% $ sparsity with negligible accuracy degradation, outperforming state-of-the-art “universal  Big bird " masks.

Overall, this work provides a principled, convex, and scalable alternative to Hoyer-based sparsification, opening the door to energy-efficient LLMs with controllable structured sparsity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper resolves the non-convex optimization issue associated with achieving structural sparsity with Hoyer-based regularizations. The paper makes theoreitical analysis on the space of Hoyer score and proposes a fast projection algorithm with a single hypercone projection onto the Hoyer score surface. The proposed Hoyer projection leads to improved efficiency-performance tradeoff on autoencoder and transformer tasks.

### Strengths
1. This paper makes novel observation and analysis on the surface of Hoyer score. This leads to the first projection-based optimization method with the Hoyer measurement, which is a significant contribution
2. The paper makes solid theoreitical derivations to show the correctness and effectiveness of the proposed projection. 
3. The overall presentation is good. The paper is easy to follow and proposed method is well-motivated.

### Weaknesses
The major weakness of this work comes with the emperical evaluations. Though the paper claims to resolve the non-convex optimization issue of the ogirinal Hoyer regualrization, whether this is a practically meaningful in deep learning remains doubtful. Deep learning itself is known to be non-convex in most cases, where having a non-convex regualrization added to the loss may not bring much addiitonal difficulty to the optimization. The paper should make a thorough comparison between the Hoyer regularization optimization and the proposed Hoyer projection to show if the proposed method is more effective and/or easier to optimize.

Furthermore, there lacks adequate comparison against related work on sparisity-inducing regualrizations, projections, and neural network pruning methods. The only baseline used in the experiments is L1-based method, which is very basic and has shown to be performing worse than more recent projection and regualrization methods. Adding more baselines, and potentially adding more neural network models into the experiment setting can better justify the method emerically.

### Questions
1. Please do a direct coparison of the cost and performance between the proposed Hoyer projection and directly performing gradient descent with Hoyer regularization
2. Please add more baselines, especially traditional projection-based sparisification methods like SCAD and MDP

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
4

### Summary
This paper introduces a projection method for neural network sparsity based on an "extended Hoyer score." The authors describe the geometric properties of this score, showing that its level sets form a hypercone, and leverage this to derive a "one-shot," linear-time projection algorithm. This single-vector method is further extended to a bi-level $\ell_{H,\infty}$ projection to induce structured, column-wise sparsity. The approach is evaluated on autoencoder and transformer architectures, where the authors report improvements in the accuracy–sparsity trade-off.

### Strengths
Clear Motivation and Problem Framing: The paper does a good job motivating the need for a scale-invariant, differentiable sparsity-inducing method.

### Weaknesses
**Weakness:**

**1. Fundamental lack of novelty leading to re-solving previously solved problems:**
    The major weakness of this paper is a fundamental lack of novelty. Both the Hoyer score and sparsity criterion have been well-studied [1, 2, 3, 4] and in better depth. Specifically, [2, 3] (following from [1] and Hoyer's original work [5]) solve the exact problems this paper addresses  (single-vector and grouped projection, respectively) with far more rigor, theoretical depth, and experimental coverage. The failure to cite or compare against this seminal works fundamentally fails to represent state of the problem the paper is trying to solve.
    
**2. Algorithms are unproven heuristics with significant theoretical gaps:**
    The paper’s algorithmic framing, based on an "extended Hoyer" score and "hypercone geometry," is a reframing of the standard orthant-projection trick already used in prior work [3]. The algorithms derived from this framing lack theoretical justification: the "naive" alternating projection (Algorithm 1) has no convergence guarantee, and the "fast" algorithms are unproven heuristics for problems that have already been solved *exactly* with rigor.

**2.1 For the single-vector projection (Algorithm 2):**  
The paper proposes an iterative hard-thresholding/active-set heuristic without proofs of uniqueness or global optimality. The claimed $O(n)$ complexity is also unsupported by a formal bound on the *while* loop's iterations. Closely related subproblems were already handled by a well-established line of work the authors do not cite: *Potluru et al.* (2013) [1] provided an exact $O(n\log n)$ routine ("Sparse-opt") for the fixed $\ell_1/\ell_2$ constraint, and *Thom et al.* (2015) [2] subsequently developed a provably exact $O(n)$ Euclidean projector via a soft-thresholding Representation Theorem.

**2.2 For the structured-sparsity projection (Algorithm 3):** The proposed bi-level $l_{H,\infty}$ extension (Algorithm 3) is presented as a novel method for structured sparsity. However, it appears to be a simple heuristic composition of a single-vector Hoyer projection and an $l_{\infty}$ projection. The paper provides no proof of feasibility or equivalence to a well-defined constrained optimization problem and fails to compare against rigorous, jointly-optimized group Hoyer projectors like GSP [3].
    
**Weak Empirical Validation:** The experiments are insufficient to support the paper's claims. The datasets used are either very small (like the 779-sample HIF2 dataset) or are not the most representative benchmarks for Transformer sparsification (like the tabular Otto dataset). They are small-scale, lack statistical rigor (e.g., confidence intervals), and most importantly, omit a direct runtime and accuracy comparison to the most relevant baselines [2, 3]. The reported runtime improvements over the old Hoyer (2004) [5] baseline are modest, and the accuracy differences are not shown to be statistically significant. 

Typo/error: 
1. Main title on openreview says "extented", do the authors mean extended?
2. Pdf title makes the typo: "EXTENDEDED",  instead of "Extended" again?

**References:**
1. Potluru, Vamsi K., Sergey M. Plis, Jonathan Le Roux, Barak A. Pearlmutter, Vince D. Calhoun, and Thomas P. Hayes. "Block coordinate descent for sparse NMF." _arXiv preprint arXiv:1301.3527_ (2013).
2. Thom, Markus, Matthias Rapp, and Günther Palm. "Efficient dictionary learning with sparseness-enforcing projections." _International Journal of Computer Vision_ 114, no. 2 (2015): 168-194.
3. Ohib, Riyasat, Nicolas Gillis, Niccolo Dalmasso, Sameena Shah, Vamsi K. Potluru, and Sergey Plis. "Explicit Group Sparse Projection with Applications to Deep Learning and NMF." _Transactions on Machine Learning Research_.
4. Yang, Huanrui, Wei Wen, and Hai Li. "Deephoyer: Learning sparser neural network with differentiable scale-invariant sparsity measures." _arXiv preprint arXiv:1908.09979_ (2019).
5. Hoyer, P. O. (2004). Non-negative matrix factorization with sparseness constraints. Journal of Machine Learning Research, 5, 1457– 1469.

### Questions
1. Could the authors clarify the theoretical or empirical advantages of their proposed iterative hard-thresholding heuristic (Algorithm 2) over the existing, provably-exact soft-thresholding $O(n)$ algorithm from [1, 2]?
1.1 Specifically, is this heuristic guaranteed to converge to the same optimal solution?
2. This is not an experimental benchmark paper and the paper introduces three new algorithms (Algorithms 1, 2, and 3) as mathematical optimization methods. Could the authors provide the corresponding formal guarantees for these algorithms, such as proofs of convergence, solution uniqueness, and global optimality?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes a fast and scalable projection method for inducing structured sparsity in neural networks. It introduces an extended Hoyer score that transforms the projection space into a hypercone, allowing efficient computation. Experiments on synthetic data, autoencoders, and transformer models show faster convergence and better accuracy–sparsity trade-offs.

### Strengths
The paper introduces an extended Hoyer score that turns the projection space into a hypercone, enabling an efficient one-shot linear-time projection. It further extends to structured sparsity via a novel bi-level projection.

### Weaknesses
1. Only a few datasets (HIF2, Otto, SST-2) are tested; results could be strengthened by larger and more diverse benchmarks (e.g., ImageNet-scale, vision transformers).
2. Training setup lacks details (e.g., number of epochs, batch size, sparsity target l, and hyperparameter tuning).
3. The paper focuses on geometric intuition but lacks convergence guarantees or formal proof.
4. The paper does not compare with other sparse training methods.
5. The paper does not present computational advantage, like speed.

### Questions
1. Can the authors provide wall-clock runtime comparisons or FLOPs analysis on real GPUs to support the claimed linear complexity improvement?
2. Since the bilevel projection involves nested optimization, how stable is the gradient flow during end-to-end training? Any divergence observed?

### Soundness
3

### Presentation
2

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
The paper tackles the question of inducing sparsity in nonconvex optimization problems using variants of the Hoyer sparsity measure. Efficient algorithms are proposed and validated on real-world datasets and deep learning architectures.

### Strengths
(i) Considers the Hoyer sparsity criterion in which the l1 norm is relaxed to a sum and the resulting score is analyzed to provide an efficient algorithm.
(ii) A structured sparsity setting is also briefly considered in the main paper and an algorithm is proposed to optimize for it.
(ii) Experiments are provided to show the benefits of the proposed approach on the sparse attention matrices for transformers on the OTTO group classification dataset and GLUE.

### Weaknesses
(a) It is unclear how the new score that is proposed satisfies desirable properties of sparsity (*)
(b) Prior works on analyzing and proposing efficient algorithms for Hoyer sparsity are not considered (*)
(c) Experiment results do not compare to other pruning/distillation approaches in the literature. 

* Comparing measures of sparsity. https://arxiv.org/abs/0811.4706
* First results  on uniqueness of SPARSE NMF https://www.eurasip.org/Proceedings/Eusipco/Eusipco2005/defevent/papers/cr1658.pdf
* Block Coordinate descent for Sparse NMF https://arxiv.org/abs/1301.3527
* Sparse Activity and Sparse Connectivity in Supervised Learning. JMLR 
* Explicit Group Sparse Projection with Applications to Deep Learning and NMF  https://arxiv.org/abs/1912.03896

### Questions
(1) What is the stated novelty of the proposed approach over the prior literature?

### Soundness
2

### Presentation
2

### Contribution
2
