# Tensor Power Methods: Faster and Robust for Arbitrary Order

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Tensor decomposition is a fundamental method used in various areas to deal with high-dimensional data. Among the widely recognized techniques for tensor decomposition is the Canonical/Polyadic (CP) decomposition, which breaks down a tensor into a combination of rank-1 components. In this paper, we specifically focus on CP decomposition and present a novel faster robust tensor power method (TPM) for decomposing arbitrary order tensors. Our approach overcomes the limitations of existing methods that are often restricted to lower-order ($\leq 3$) tensors or require strong assumptions about the underlying data structure. By applying the sketching method, we achieve a running time of $\widetilde{O}(n^{p-1})$ per iteration of TPM on a tensor of order $p$ and dimension $n$. Furthermore, we provide a detailed analysis applicable to any $p$-th order tensor, addressing a gap in previous works. Our proposed method offers robustness and efficiency, expanding the applicability of CP decomposition to a broader class of high-dimensional data problems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a theoretical study on tensor power methods for finding eigenvectors and eigenvalues of tensors of order p \ge 3. The authors propose an algorithm (Algorithm 1) and provide rigorous proofs for the approximation errors. The paper aims to generalize existing power method analyses, which have been primarily limited to third-order tensors (e.g., Wang & Anandkumar, 2016).

### Strengths
The main strength of this work lies in its theoretical completeness. The proposed results extend the scope of tensor power methods beyond the well-studied third-order case, filling a notable gap in the literature. The proofs are detailed, and the technical rigor contributes to the theoretical understanding of tensor eigenvalue computations.

### Weaknesses
However, I have several concerns regarding the clarity and significance of the contribution. First, the paper is difficult to follow in several parts due to unclear definitions, missing explanations of notations, and imprecise mathematical exposition. I will discuss these issues in detail in the “Questions” section.

Second, while the contribution is well stated, I am not fully convinced of its overall importance for the following reasons.

(1) The theoretical results rely on a strong assumption that A^* must be constructed using orthonormal basis vectors (see the fourth condition in Theorem D.1, which is also used in Theorem 1.1). This assumption, although standard in CPD analyses for third-order tensors, seems overly restrictive for higher-order tensors due to the over-determinacy of tensor eigenvectors—there are typically many more eigenvectors than the ambient dimension. I could be convinced if the authors provided numerical evidence showing that this assumption is not too strong in practice, or additional theoretical arguments addressing this concern.

(2) The complexity analysis in Theorem 1.1 shows that the computational cost of the proposed algorithm grows exponentially with the tensor order p. This raises doubts about the claim of having a “fast” algorithm, especially since the main focus is on tensors of order greater than three.

(3) I could not clearly identify what new techniques were introduced to handle the higher-order case compared to prior works. The proofs appear to rely on more algebraic derivations rather than fundamentally new ideas. If the authors can clearly highlight the non-trivial extensions and explain how the proof differs from the third-order case, this concern could be easily addressed.

### Questions
1. In line 44, regarding the citation of CP decomposition, why not cite the original work by Frank (1927)? Is there a specific reason for omitting it?
2. For Theorem 1.1, could you elaborate on how the proposed algorithm achieves computational efficiency compared to existing methods?
3. In lines 119–122, the phrase “and this combination is the only… the rank-1 tensor that… is not possible” is unclear. Does this refer to the uniqueness property of CP decomposition?
4. In line 178, the notation v_j is not clearly explained. What does the subscript j represent?
5. In line 197, what does “part 1” refer to, and in line 200, what does V on the right-hand side of the inequality denote? These should be clarified in the main text, not only in the appendix, to improve readability.
6. In the algorithm section, what is the relationship between A and A_i for i \in [n]?
7.  In Lemma 4.2, the definition of \delta should also be presented in the main paper for completeness.
8. In line 996, what is meant by an “orthogonal tensor”? Please provide a formal definition.


⸻

LLM Usage Disclosure:
Parts of this review were refined using a large language model (OpenAI GPT-5) to improve grammar and readability. All substantive assessments, critiques, and questions reflect my own independent judgment.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a generalization of the robust Tensor Power Method (TPM) for Canonical Polyadic (CP) tensor decomposition to arbitrary tensor orders $p \geq 3$. Previous work (e.g., Anandkumar et al., 2014; Wang & Anandkumar, 2016) provided robust guarantees and convergence proofs for third-order tensors under bounded noise. The authors extend these results by developing a theoretical framework for higher-order tensors, establishing formal convergence and error bounds, and proposing a fast sketch-based implementation using randomized Hadamard transforms (RHT). The algorithm achieves a claimed runtime of $\tilde{O}(n^{p-1})$ per iteration and maintains robustness guarantees under noise. The paper provides detailed proofs and lemmas supporting the generalization.

### Strengths
**Originality:**

The paper addresses a formal generalization of the robust TPM from third-order tensors to arbitrary-order tensors, filling a theoretical gap in tensor decomposition literature.

While the idea itself is incremental, the generalization requires careful handling of tensor algebra, noise bounds, and convergence proofs, which the authors provide comprehensively.

**Quality:**

The mathematical derivations are technically sound and complete, with proofs and lemmas that appear consistent and rigorous.

The authors clearly define all assumptions, norms, and quantities, ensuring reproducibility of the theoretical work.

**Clarity:**

Key results (Lemma 3.1, Theorem 4.9) are stated cleanly, and the appendices provide formal proofs.

Despite its technical density, the paper is generally self-contained.

**Significance:**

The generalization to arbitrary-order tensors contributes to the theoretical completeness of tensor power methods.

The work may serve as a reference for theoreticians studying high-order tensor decompositions, even if it has limited immediate practical impact.

### Weaknesses
**Limited Novelty Beyond Extension**

The core algorithmic idea follows directly from prior robust TPM frameworks (Anandkumar et al., 2014; Wang & Anandkumar, 2016).

The generalization to arbitrary order relies mainly on extending existing proofs by induction, without introducing fundamentally new insights or techniques.

Sketching via randomized Hadamard transforms is also well-established and applied here in a standard way.

**No Empirical Validation**

The paper makes claims of being “faster” and “robust,” but presents no experiments or synthetic evaluations to substantiate these claims.

Runtime scaling (Õ(n^{p-1})) and robustness under realistic noise conditions remain theoretical and unverified.

**Exposition is overly technical and lacks insight.**

### Questions
1) Can you include small synthetic experiments (for example p = 3,4,5) showing actual runtime scaling or robustness under controlled noise? This would strengthen the “faster” and “robust” claims. More empirical experiments (synthetic or real) would also be a good addition to the paper.

2) For what practical range of tensor order p and dimension n does your algorithm remain computationally feasible?

3) How sensitive is convergence to initialization? Does the generalization change the probability of successful recovery compared to p=3?

4) Could this framework handle non-orthogonally decomposable or overcomplete tensors? Is that theoretically tractable within your current proof structure?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors propose a tensor power method (TPM) for CP decomposition of arbitrary-order tensors. Theoretical analysis demonstrates that the proposed method can achieve a running time of $\mathcal{O}(n^{p-1})$ per iteration on a tensor of order 
$p$ and dimension $n$, overcoming limitations of the previous method (Wang & Anandkumar, 2016) restricted to lower-order tensors ($p\leq 3$). Theoretical analysis is provided for arbitrary including convergence guarantees and error bounds under bounded noise.

### Strengths
1. A new TPM method is proposed that can deal with CP decomposition of arbitrary-order tensors.
2. Theoretical analysis is given to guarantee the recovery, robustness to noise, and the advantage in time complexity per iteration.

### Weaknesses
1. There are no experiments that verify the advantage (efficiency and robustness) of the proposed method.
2. It is not clear what the difficulty is that extends the analysis in (Wang & Anandkumar, 2016) to the case with $p>3$.

### Questions
1. Please clarify what the difficulty is that extends the analysis in (Wang & Anandkumar, 2016) to the case with $p>3$.
2. As the author states that the proposed method can handle higher-order tensors with lower complexity, no experimental results are given to support the claims. 
3. In the technique overview and conclusion, the author said that the proposed method can be applied to tensor data such as images and videos. Therefore, the author should carry out experiments on real-world image/video data along with the running time comparison to show the efficiency and robustness of the proposed method.

### Soundness
2

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
5

### Summary
The paper studies a power method for computing low-rank CP decompositions of higher-order tensors.  It works with arbitrary tensor orders, unlike some previous related works.  The authors use randomized sketching to accelerate the tensor power method.  They provide thorough convergence analysis for perturbations of orthogonally decomposable tensors.

### Strengths
- The paper accelerates the tensor method power with randomized sketching to evaluate inner products more quickly.

- The paper has a strong theoretical contribution, with its Theorem 4.9.

- The work applies to tensors of any order.

### Weaknesses
- The paper is missing important references.

- The paper has no numerical experiments.

- The presentation in the paper is suboptimal, and the main body pushes too many key points to the appendices.

### Questions
1) General remark: The authors missed a very relevant line of work in the literature:
- "Landscape analysis of an improved power method for tensor decomposition", Joe Kileel, Timo Klock, Joao Pereira, NeurIPS 2021
- "Subspace power method for symmetric tensor decomposition", Joe Kileel, Joao Pereira, Numerical Algorithms, 2025.
There is also the recent preprint:
"Multi-subspace power method for decomposing all tensors", Kexin Wang, Joao Pereira, Joe Kileel, Anna Seigal, arXiv:2510.18627
This line of work develops a power method for CP decomposition that is applicable not only when the tensor is a noisy perturbation of an orthogonally decomposable tensor.  The method applies beyond the ranks where whitening is possible, and with tensors of any order.  Briefly, it works by constructing an auxiliary tensor $\tilde{T}$ to the input tensor $T$ such that the eigen/singular vectors of $\tilde{T}$ are the CP components of $T$.  The works also include an analysis of SS-HOPM and various noise robustness and convergence analyses.  I would suggest the authors cite these relevant works and comment very briefly on differences between their own work and these.

2) Line 170-171: As part of their stated breakthrough, the authors write ``Moreover, we have created a strong and adaptable algorithm that can handle a variety of tensor data: natural language corpora, images, videos, etc."  This adaptability to different data structures is something I didn't catch.  Can the authors explain more?

3) Line 267: It seems the authors are assuming the tensors $A_i$ have common CP components $x_j$.  This seems to be a strong assumption.  My guess is that the $A_i$'s are meant to be order-$(p-1)$ slices of a single low-rank order-$p$ tensor $A$ which would indeed yield this situation, but it wasn't explained in the "sketching technique" section.

3) Line 270: Can the authors comment on how the complexity improves over a non-sketching based implementation?  It seems the savings is through evaluation of $\langle A_i, u^{\otimes (p-1)} \rangle$ for $i=1, \ldots, n$ dropping from $\mathcal{O}(n^p)$ to $\mathcal{O}(\epsilon^{-2} n^{p-1})$. 

4) Lines 294-311, Algorithm 1: This algorithm is not easily understood in a self-contained way.  I would suggest the authors include pseudo code for the subfunctions ds.QUERY and ds.QUERYVALUE within the body of the paper to increase readability.

5) Line 415: It is not necessary to include the fact that $(ab)^2 = a^2b^2$ for real numbers $a$ and $b$ in an ICLR submission.

6) Line 433, Theorem 4.9: Is $A^*$ orthogonally decomposable?  Please include this in the theorem statement if so.

7) General comment: The paper and appendices have no numerical experiments.  It would be nice to include some basic ones, for example to illustrate the speed-up brought by the sketching approach or the convergence rates proven by the authors.  

8) General comment: The authors should be more explicit that they consider symmetric tensors and symmetric CP tensor decomposition.

### Soundness
3

### Presentation
2

### Contribution
3
