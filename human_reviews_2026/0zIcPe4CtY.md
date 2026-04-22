# Sublinear Time Quantum Algorithm for Attention Approximation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Given the query, key and value matrices $Q, K, V\in \mathbb{R}^{n\times d}$, the attention matrix is defined as $\mathrm{Att}(Q, K, V)=D^{-1}AV$ where $A=\exp(QK^\top/\sqrt{d})$ with $\exp(\cdot)$ applied entrywise, $D=\mathrm{diag}(A{\bf 1}_n)$. The attention matrix is the backbone of modern transformers and large language models, but explicitly forming the softmax matrix $D^{-1}A$ incurs $\Omega(n^2)$, motivating numerous approximation schemes that reduce runtime to $\widetilde O(nd)$ via sparsity or low-rank factorization.

We propose a quantum data structure that approximates any row of $\mathrm{Att}(Q, K, V)$ using only row queries to $Q, K, V$. Our algorithm preprocesses these matrices in
$\widetilde{O}\left( \epsilon^{-1} n^{0.5} \left( s_\lambda^{2.5} + s_\lambda^{1.5} d + \alpha^{0.5} d \right) \right)$
time, where $\epsilon$ is the target accuracy, $s_\lambda$ is the $\lambda$-statistical dimension of the exponential kernel defined by $Q$ and $K$, and $\alpha$ measures the row distortion of $V$ that is at most $d/{\rm srank}(V)$, the stable rank of $V$. Each row query can be answered in
$\widetilde{O}(s_\lambda^2 + s_\lambda d)$
time.

To our knowledge, this is the first quantum data structure that approximates rows of the attention matrix in sublinear time with respect to $n$. Our approach relies on a quantum Nystr{\"o}m approximation of the exponential kernel, quantum multivariate mean estimation for computing $D$, and quantum leverage score sampling for the multiplication with $V$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies quantum algorithms (in the QRAM model) to approximate attention in transformers. Given query-key-value matrices $Q,K,V$ of sizes $n\times d,d\times n,$ and $n\times d$, respectively, the goal is to approximate the attention matrix $A=softmax(QK^\top)V$. This is achieved by constructing a data structure that, given any index $i\in[n]$, it returns $\widetilde r_i\in\mathbb{R}^{d}$, which approximates the  $i$-th row of $A$. The total complexity to construct the data structure  depends on $n,d$ and other parameters such as the accuracy, statistical dimension, and row-distortion, and Frobenius norms (I expand further below). If the additional parameters besides $n,d$ are treated as constants, then the proposed approach provides quadratic speed-up with respect to the large dimension $n$, against known algorithms for the problem at hand. Importantly, the complexity guarantees are also accompanied with approximation bounds.

### Strengths
1) The paper is well-written and concise, which makes it enjoyable to read. 
2) It targets one of the currently most popular computational problems in deep learning, the "quadratic curse" of attention.
3) It provides a novel approach to the problem, combining (non-trivially) techniques from random Linear Algebra and quantum computing.
4) It achieves a complexity that scales as $\sqrt{n}$, with respect to the large dimension $n$. This is impressive, even if it only holds for certain parameter regimes. To my knowledge, classical algorithms require $\Omega(n^2)$ time to achieve sharp element-wise approximations (ref [2] below), or $\Omega(n)$ for less strict approximations. 
5) The related work discussion is thorough.
6) The mathematical analysis is very rigorous. I did not read all the details in the proofs, but I was not able to "break" any of them, they seem correct, and well-written.

### References
- [1] Demmel, James, Ioana Dumitriu, and Olga Holtz. "Fast linear algebra is stable." Numerische Mathematik 108.1 (2007): 59-91.
- [2] Alman, Josh, and Zhao Song. "Fast attention requires bounded entries." Advances in Neural Information Processing Systems 36 (2023): 63117-63135.

### Weaknesses
I have only two "major concerns" to raise at this stage. Below in "Questions" I provide specific questions that would help me understand the details better and clarify these points.
1) **Classical model of computation**: From what I understand, there are subroutines in the main algorithm that rely on classical computations. For example, line 274 assumes a subroutine to compute the pseudoinverse in $O(n^\omega)$. How? From what I know, finite precision algorithms can only return approximate solutions (see e.g. ref [1] below). Things might be easier in "exact arithmetic", but I am not sure that infinite-precision is compatible with QRAM.
2) **Approximation/complexity trade-off**: I am a bit sceptical about the complexity / approximation trade-off. In Theorem 3.1, line 362, there is a $\sqrt{n}$ term, and two more "hidden" in the Frobenius norms. The former can be absorbed by setting e.g. $\lambda=1/\sqrt{n}$, but it is a bit unclear how this affects complexity. Now, if we were to upper bound the $||\cdot||_F$-norms with $||\cdot||_2$-norms, which is commonly the desired type of bound, they would intruduce another $\sqrt{n}$ factor. This factor would have to be absorbed inside $\epsilon$, e.g., by setting $\epsilon'=\epsilon/\sqrt{n}$. But they this would introduce an additional $\sqrt{n}$ factor in the complexity of line 365, and therefore it would no longer be sublinear in $n$.

**Minor concerns:**
I have the following two minor comments (but they did not influence my recommendation).

- The QRAM model is mostly of theoretical interest (at least at the time of this writing). This might be a limitation for practical implementations in the future. 
- The authors recognize in line 377 that the reported approximation guarantee is for a "symmetrized version" rather than the classic norm-wise approximation. I think this is fine, I am more concerned with the Frobenius-versus-spectral norm topic, as I mentioned above. But it certainly speaks in favor of the authors that they explicitly mention this topic.

### Questions
My current assessment is slightly leaning towards reject, due to the two main concerns that I raised above. At this stage it is not clear to me if the final, end-to-end complexity achieves the reported sublinear time, or, if it does, what are the corresponding parameter regimes. My recommendation is not final. I will take into consideration the authors responses as well as the comments from the other reviewers.
Here I mention some questions that I would like answered to help me clarify my understanding of the paper and provide the additional evidence for my final assessment.
### Questions
1) Regarding concern 1): What is the "classical" model of computation followed here? Is it "compatible" with QRAM? Could you provide references/discussion on the precise complexity / approximation guarantees of the assumed classical subroutines?
2) Regarding concern 2): Could you provide a small paragraph discussing further how the choice of the different parameters affects the total complexity? If someone wants spectral-norm bounds, how can they be achieved?
3) Can we replace QRAM with something simpler (e.g., QROM)? Which parts are currently the "bottleneck"? 
4) Are there any quantum / classical lower bounds for Frobenius-norm type of approximations, e.g., in similar spirit to [2] below. I do not expect the authors to prove lower bounds at this stage, but a relevant discussion would be helpful.
5) If we were to use the proposed algorithms to  approximate the entire attention matrix, what would be the complexity and how does it compare with existing attention algorithms? I think that the $\Omega(n^2)$ lower bounds of [2] leave quite some room for improvements. E.g., if the final complexity of the proposed algorithm is $O(n^{1.5})$ to achieve the same (or similar) bound as [2], then this would already be a nice improvement, and would significantly strengthen the presentation. Could you provide some insights? 
6) Could the authors comment on how to choose the $\lambda$ parameter?

### Additional Feedback
Here I provide additional feedback with the aim to improve the paper. These points are here to help, and not necessarily part of the decision assessment.

- The main result, Theorem 3.1, is in page 7. It would be nice to either move it earlier, e.g., in the introduction, or at least a more explicit statement of the main result in the introduction.
- A table with the main result compared to existing algorithms could be helpful. E.g., to compare complexity, approximation guarantees (if any), the model of computation, or other properties that the authors consider important (again, this is not a request for the rebuttal, just potentially nice-to-have)
- Some paragraphs are a bit long, e.g., the first paragraph of Section 3.3. 
- Using colored references can be helpful, and I think it is allowed by ICLR template.
- In line 206, $O(s^2)\cdot \mathcal{T}_K+s^\omega$ should be $O(s^2\cdot \mathcal{T}_K+s^\omega)$. The hidden constants in fast matrix product are quite large. There might be other places in the paper where this applies.
- When mentioning algorithms/subroutines with $O(n^\omega)$  complexity, a reference or proof should be given. I know that some theory papers tend to take them for granted, but often they are highly non-trivial to prove, or even to find the corresponding bibliography.
- Between lines 368-369: The sentence "...achieving a quadratic speedup over any classical algorithm" should probably be "...achieving a quadratic speedup with respect to $n$ over any classical algorithm that we know of". 
- In line 372, it is mentioned "when $a=o(n)$". Based on Definition C.2 in the Appendix, I think $a(V)$ is always at most $d$. Take the SVD of $V=U\Sigma W^\top$, and let $v_i$ be the $i$-th row of $V$. It holds that 
$||v_i||_{2}^{2}=||e_i^{\top}U\Sigma W^{\top}||_2^2\leq  ||e_i^{\top} U||_2^2||\Sigma W^\top||_2^2=\tau_i||V||_2^2.$ Replacing this in Definition C.2 gives 
$\frac{d}{||V||_F^2}\cdot \max_i\frac{||v_i||_2^2}{\tau_i} \leq d$.

### Soundness
3

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
3

### Summary
This paper proposes a quantum data structure for approximating rows of the attention matrix in sublinear time with respect to the sequence length n. The method combines quantum Nyström approximation, multivariate mean estimation, and leverage score sampling to approximate the components of the attention mechanism. This is the first quantum algorithm to achieve sublinear dependence on n in the row-query model without structural assumptions.

### Strengths
1. The work is the first to achieve sublinear-in-n row queries for attention approximation using quantum methods.
2. The approach makes no  structural assumptions making it widely applicable.

### Weaknesses
1. Parameter dependence: The runtime depends on s  and α , which may be large in practice, limiting practical speedups.
2. Norm of D−1 assumption: The guarantee requires ∥D−1∥<(ϵ∥E∥+λn)−1, which may not hold in all settings.

### Questions
1. Can you give numberical experiments to show the time cost, errors and the assumptions on parameters.
2. How does the statistical dimension s behave in practice for typical transformer inputs, and does it remain small enough to yield meaningful speedups?
3. Is the row distortion parameter α bounded in real-world value matrices, and are there cases where it becomes large enough to negate the sublinear advantage?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The goal of the paper is to study approximation algorithms for self-attention computation in the transformer architecture. The inputs to self-attention are $Q,K,V \in \mathbb{R}^{n\times d}$ and the goal is to output $Att(Q,K,V) = D^{-1}A V$ where $A= exp(QK^T/\sqrt{d})$ and $D^{-1} = diag(A\mathbb{1})$. Past works for provable attention approximation need at least $\Omega(nd)$ time, which is the input and output size, and the paper focuses on quantum algorithms that can achieve a better runtime. If one insists on outputting the entire $Att(Q,K,V)$ matrix, $\Omega(nd)$ time is inevitable, however this can be avoided by formulating the problem as a data structure problem. In particular the goal is preprocess $Q,K,V$ into a data structure that then allows, for any index $i\in [n]$, the return an approximation to the $i^{th}$ row of $Att(Q,K,V)$. Even then since each row of $Att(Q,K,V)$ is a convex combination of rows of $V$, achieving sublinear in $n$ time is hard. 

Their main contribution is a quantum data structure that access the input matrices only using row queries, performs preprocessing in $\widetilde{O}(\epsilon^{-1} n^{0.5} poly(d,s_{\lambda},\alpha)$ time, and answers output row queries in time $\widetilde{O}(s_{\lambda}^2 + s_{\lambda}d)$ (here $s_{\lambda}$ is the statistical dimension of $exp(QK^T/\sqrt{d})$. Their approach uses techniques such as Grover search, Quantum Nystrom approximation, and Quantum multivariate mean estimation.

### Strengths
The main strength of the paper is to present a sublinear time algorithm that answers row queries for attention approximation in the quantum model. The techniques are very interesting and conceptually simple.

### Weaknesses
Perhaps one minor weakness is that there are few previous works on attention approximation that achieve spectral norm approximation guarantees and it would be to prove such a guarantee here as well.

### Questions
The first question is that the authors make a statement that achieving sublinear in $n$ dependence for the row query model seems intractable for classical algorithms since each row of the output is a convex combination of rows of $V$. Is there a formal claim to show this ? If yes then since there are past works on attention approximation that make structural assumptions on the input matrices, is it possible to prove classical sublinear in $n$ guarantees under plausible assumptions ?

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
The paper proposes a quantum data structure for approximating the Transformer attention mechanism under the row query model, where only individual rows of the attention output are queried. The key contribution is a theoretical framework that achieves sublinear preprocessing time complexity of $\tilde{O}(\epsilon^{-1} n^{0.5}(s_\lambda^{2.5} + s_\lambda^{1.5} d + \alpha^{0.5} d))$, providing a quadratic speedup over the best known classical algorithms.
The method embeds the non-symmetric attention matrix $A = \exp(QK^\top / \sqrt{d})$ into a larger symmetric exponential kernel matrix over the combined dataset $(Q, K)$, and applies a combination of quantum Nyström approximation, quantum multivariate mean estimation, and quantum leverage score sampling to approximate the attention normalization factor, kernel matrix, and value multiplication components, respectively.
The resulting data structure allows approximating any attention row in time $\tilde{O}(s_\lambda^2 + s_\lambda d)$, without assumptions on $Q, K, V$. This is, to the authors’ knowledge, the first quantum algorithm achieving sublinear dependence on sequence length $n$ for attention approximation. The authors also provide theoretical guarantees in Frobenius norm for the symmetrized attention matrix $(A + A^\top)/2$, along with detailed parameter dependence on the kernel’s statistical dimension $s_\lambda$ and value distortion factor $\alpha$.

### Strengths
S1: High Originality and Theoretical Significance: This work, to the best of my knowledge, is the first to propose a sublinear-time quantum algorithm for approximating the standard Transformer attention mechanism in the row-query setting. Achieving a preprocessing complexity of $\tilde{O}(n^{0.5})$, the method provides a potential quadratic speedup over classical algorithms. This represents a meaningful theoretical advance and offers a new perspective on overcoming the quadratic bottleneck in large-scale attention computation.

S2: Sophisticated Theoretical Framework: The paper demonstrates strong technical depth by systematically combining several advanced quantum tools—Nyström kernel approximation, multivariate mean estimation, and leverage score sampling—into a coherent data structure for attention approximation. The approach of embedding the non-symmetric attention matrix into a symmetric exponential kernel over the joint query–key space is both elegant and conceptually novel. Moreover, the framework is general, requiring no structural assumptions on $Q$, $K$, or $V$, which enhances its theoretical robustness and potential applicability.

### Weaknesses
W1: Lack of Empirical Validation: The paper is entirely theoretical and does not provide any numerical simulation or small-scale experiment to illustrate the potential practical impact of the proposed method. While this is acceptable for a theoretical contribution, even a simple empirical demonstration (e.g., simulated quantum runtime scaling or synthetic kernel approximation) would help substantiate the claimed sublinear advantages.

W2: Symmetrization Limitation: Because the algorithm approximates the attention matrix through a symmetric kernel on the combined $(Q, K)$ dataset, it effectively provides guarantees only for the symmetrized form $(A + A^\top)/2$. This design choice limits its direct interpretability as an approximation to the true attention matrix, and it remains unclear whether the same speedup can be achieved without this symmetrization.

### Questions
Q1:Regarding Empirical Validation (W1):
Could the authors provide any empirical or simulated evidence to illustrate the practical implications of the proposed algorithm? For example, could a small-scale classical simulation or synthetic experiment demonstrate the expected sublinear scaling behavior or approximation quality?

Q2:Regarding Symmetrization Limitation (W2):
The current framework provides guarantees only for the symmetrized attention matrix $(A + A^\top)/2$. Do the authors believe the same quantum speedup could be achieved without this symmetrization? If not, could they elaborate on the fundamental technical barriers that make direct approximation of the asymmetric attention matrix more challenging?

### Soundness
2

### Presentation
3

### Contribution
2
