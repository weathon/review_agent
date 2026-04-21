# Fast RoPE Attention: Combining the Polynomial Method and Fast Fourier Transform

- Avg Score: 4.80
- Decision: Reject
- Scores: 3, 5, 5, 3, 8

## Abstract
The transformer architecture has been widely applied to many machine learning tasks. A main bottleneck in the time to perform transformer computations is a task called attention computation. [Alman and Song, NeurIPS 2023] have shown that in the bounded entry regime, there is an almost linear time algorithm to approximate the attention computation. They also proved that the bounded entry assumption is necessary for a fast algorithm assuming the popular Strong Exponential Time Hypothesis.

A new version of transformer which uses position embeddings has recently been very successful. At a high level, position embedding enables the model to capture the correlations between tokens while taking into account their position in the sequence. Perhaps the most popular and effective version is Rotary Position Embedding (RoPE), which was proposed by [Su, Lu, Pan, Murtadha, Wen, and Liu, Neurocomputing 2024]. 

A main downside of RoPE is that it complicates the attention computation problem, so that previous techniques for designing almost linear time algorithms no longer seem to work. In this paper, we show how to overcome this issue, and give a new algorithm to compute the RoPE attention in almost linear time in the bounded entry regime. (Again, known lower bounds imply that bounded entries are necessary.) Our new algorithm combines two techniques in a novel way: the polynomial method, which was used in prior fast attention algorithms, and the Fast Fourier Transform.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents an algorithm, along with its complexity analysis, to efficiently approximate a variant of attention called RoPE attention, which is used in recent LLMs. The algorithm is based on approximating softmax with a rational function, computing efficiently polynomials of circulant matrices, and showing that attention can be recasted in this framework. The complexity of the algorithm matches the lower-bound for this problem.

### Strengths
The paper is very clearly-written and the algorithm seems correct. The algorithm is novel and combines nicely several ideas from applied mathematics (approximation of exp by a polynomial, FFT to compute the product of a circulant matrix and a vector). It tackles the crucial question of efficiently computing attention matrices, and the algorithm could therefore have good impact. I stand in favour of acceptance, and encourage authors to complement their contribution with pseudo-code, open-source implementation, and experiments (see Weaknesses).

### Weaknesses
[POST-REBUTTAL EDIT: Reviewer Q6JF found a very significant issue in the paper. Namely, some matrix central to the analysis is supposed to be circulant, whereas in the case of standard RoPE attention, this matrix is only constant on each diagonal, which is weaker than circulant. Thus, the proposed algorithm probably does not apply to standard RoPE attention. The authors did not provide an answer during the rebuttal (see discussion below the review of Reviewer Q6JF). Thus I cannot recommend acceptance of the paper, and lowered my score.]

- Section 5.3 is a bit hard to follow and could be more detailed. For instance, a self-contained algorithm in pseudo-code would be beneficial both for the overall understanding of the paper, and for checking correctness of the proof of Section 5.3.
- There is no experiment. While I don’t think that experiments are an absolute prerequisite for a ML paper, it would be nice to provide an open-source implementation of the algorithm and to compare empirically the performance in terms of speed and accuracy with a naive computation of RoPE attention (and perhaps other baselines if existing), and with the computation of standard attention with the polynomial method, for various hyperparameter values. Given the nature of the contribution, I am still in favor of acceptance despite the absence of experiments, but encourage authors to provide experiments in the next version of the paper.

Minor remarks and typos (no influence on the rating):
- line 67: I believe $B$ is not defined at this point.
- line 304: please find a proper reference instead of “Folklore”.
- line 380 up to the end of Section 4: there are $f$ several times in indices of $W$. Is this a typo?
- in Lemma 5.1, specify again that the $A^{\ell_1, \ell_2}$ are rescaled circulant matrices.
- In Lemma 5.2, rephrase the definition of $\mathcal{M}$: “let $\mathcal{M}$ be the set of functions …”
- In the proof of Lemma 5.2, recall again why $N^{(m)}$ are rescaled circulant matrices (due to Lemma 3.6).
- In the proof of Lemma 5.2, explain where the estimate of $|\mathcal{M}|$ comes from.
- line 482: $M^{(m)}$ should be $N^{(m)}$.
- line 497: $d$ should be $\tilde{d}$?

### Questions
- To the best of your knowledge, is the polynomial method actually used in practice to approximate the attention computation in large-scale Transformer models?
- The number of circulant matrices increases from $|S|$ to $|\mathcal{M}|$ due to the element-wise product by the polynomial $p$. This means that, if we apply the algorithm several times (across several layers of Transformer), the number of matrices to handle grows at each iteration. Thus the complexity of applying recursively, say $L$ times, this algorithm, could grow super-linearly with respect to $L$. Do you know this complexity?

### Soundness
1

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper discusses an approach to approximating RoPE attention, building on previous work that approximated standard attention using the polynomial expansion of exp. The work deals with Rope using FFT and shows how to extend previous results relatively easily, including results on approximation performance.

### Strengths
Approximating attention is an important topic and these results go some way to state how well RoPE attention can be approximated quickly. As far as I know, these results are new and do add to knowledge about how well we can approximate Rope attention.

### Weaknesses
The paper isn't particularly clear in places, with some definitions out of sequence. The results also seem quite straightforward though there are also places where more justification or explanation seems required. The lack of any empirical performance is a shame since I think most practitioners would like to see whether the theory actually translates into a practically useful approximation.

### Questions
*** page 1

The attention should be normalized by \sqrt(d), not d.

*** page 2

B is not defined

*** page 3

equation 1 has no normalization, yet definition 1.1 does have normalization (though this should be sqrt(d), not d)

The notation [n] is perhaps well used in combinatorics, but is not common elsewhere. Please help the reader and explain this notation.

After reading through, the authors define this on page 6, but this is too late for the reader.

*** page 4

The argument that A cannot be low rank isn't very clear to me and I feel the explanation could be improved. For example, one can choose a rotation R that is the identity, recovering the standard attention (which can have low rank).

I find the presentation based on discussing whether M "were indeed circulant a matrix" confusing. Clearly, M cannot in general be circulant (since the exponent in equation 1 is not circulant) and this discussion I feel confuses rather than helps the reader.

*** page 6

My immediate thought is that the exponential can only be well approximated for small x. Lemma 3.1 seems to confirm that the degree of the polynomial (and hence complexity of the approximation) scales with the range of x. The immediate question is then whether this would be practical in real-case scenarios.

Fact 3.3. This is called "folklore". Please cite a reference to this result. "Folklore" suggests that it may not be true, whilst this is indeed a fact.

By definition a circulant matrix multiplied by a vector \sum_j A_{i-j}v_j is a convolution and therefore can be calculated using the FFT in O(n\log n).

line 323 -- typos

*** page 7

The notation in line 351 is confusing. This W isn't the same W as in the definition of the original attention

Definition 4.1 introduces S but doesn't use it in the definition.

In general I found the presentation in section 4 quite confusing. If one wants to compute \sum_j A_{i,j}v_j where we define the linear attention A_{i,j} = vec(q_i)\trans R^{j-i} vec(k_j) for the ith query vector vec(q_i) and jth key vector vec(k_j) and rotation matrix R raised to the power j-i, then trivially we can write \sum_j A_{i,j}v_j= vec(q_i)\trans\sum_j R^{j-i}vec(k_j)v_j. Since \sum_j R^{j-i}vec(k_j)v_j is a convolution, then it directly follows that Av is computable in O(n\log n) time. It would be useful if the authors could therefore explain why the circulant results are required.

*** page 10

The justification for theorem 5.3 isn't fully clear to me. The previous results relate to approximating the unnormalised attention. However, we need to divide the approximated unnormalised attention by the approximated normalised attention. I don't immediately see why the division doesn't affect the approximation error results. This could at least be better explained.

The hardness result seems essentially trivial.

*** General comments

It's disappointing that there is no empirical support for the method including, for example, a simple experiment to show how well the method works.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents a sublinear algorithm for efficient Rotary Position Embedding (RoPE) attention computation with provable guarantees. The main technique combines the polynomial method with the Fast Fourier Transform (FFT). The paper also presents a lower bound on this problem.

### Strengths
- Rotary positional encoding is very popular in practice. It is relevant and timely to consider efficient algorithms for RoPE-attention.

- The paper presents the main technique clearly.

### Weaknesses
- **Limited algorithmic novelty.** The proposed method relies on the polynomial method to handle $\exp$ in attention, which has been studied in existing work [1,2]. Fast Fourier Transform has also been introduced in prior works to efficiently handle positional information in attention [3,4,5]. While this paper studies attention with a different positional encoding variant (RoPE), it would enhance the paper quality if the authors can discuss more on the new algorithmic insights in the proposed method.

[1] On The Computational Complexity of Self-Attention

[2] Fast attention requires bounded entries

[3] Stable, Fast and Accurate: Kernelized Attention with Relative Positional Encoding

[4] Toeplitz Neural Network for Sequence Modeling

[5] From block-Toeplitz matrices to differential equations on graphs: towards a general theory for scalable masked Transformers

- **Weak lower bound.** The lower bound only works for the "_general_" RoPE Attention Computation. The hard instance is constructed by setting all the $W_i$'s to identity matrices, falling back to the vanilla attention. This results does not lead to any new insight on the hardness of approximating the true RoPE attention in practice. It is not clear whether the exact RoPE attention (with $R_{j-i}$ defined as in line 111) satisfies similar hardness results.

- **Lack of empirical evaluations.** The authors do not provide any implementation of the proposed algorithm and run experiments on any datasets. It may help the practitioners if the implementation and empirical analysis on the speed-up rate, modeling quality, and comparisons with other efficient attention methods can be provided. Empirical evaluations can also better justify the theoretical results.

- **Minor issues.**
  - `\citep` and `\citet` are not used correctly in the paper, e.g., in lines 58, 88, 89, 90.
  - Line 116: The word "about" is unnecessary and makes the statement vague. In RoPE, $\alpha=10^4$ exactly.
  - Def. 1.1 is about "A General Approximate RoPE Attention Computation", but in its statement the terminology changes to "Rotated attention computation". Please make it consistent.
  - Def 4.2 should have been stated before Def. 4.1 to make the exposition logical. Also, should $\mathrm{supp}(W_i)=S$ be $\mathrm{supp}(W_i)\subset S$ instead?

### Questions
- In Def. 1.1, should the scaling factor be $1/\sqrt{d}$ instead of $1/d$?

- Can you discuss whether the assumptions on $B$ (the upper bound on queries, keys, and values) hold in practice based on empirical evidence?

- Do you have evidence to show whether the assumption $B=o(\sqrt{\log n})$ holds in practice? Or, under what condition is this assumption more likely to be violated?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper, essentially, establishes the following result: insertion of a specific (i,j) position-dependent square dxd matrix **R** between queries matrix **Q** and keys matrix **K** in the attention mechanism **A**  such that  $A_{ij}=exp(Q_i R_{i-j} K_j^\top)$ leads to an algorithm which under assumption of bounded norm in attention parameter matrices **Q, K, V** computes the approximate attention in $O(n \log n)$ time. They call $R_{i-j}$ *general RoPE* matrix and consider specific constructions which makes the resulting **A** matrix a *rescaled circulant matrix* which could be constructed and multiplied by any other vector in $O(n \log n)$ by means of Fast Fourier Transform.

### Strengths
* The paper reveals interesting and promising direction of work on optimization of runtime of Transformers by injecting specific types of structured matrices.
* Their approach is inspiringly novel and non-trivial and, if studied further, could lead to novel augmented Transformer architectures having the same quality as the original but computable in $O(n \log n)$ time.
* For general setting when the proposed generalized RoPE matrix can't be reformulated as entry-wise transformation of **Q** and **K**, the method is sound as it seems to be capable of computing approximate attention in sub-quadratic run-time complexity with guaranteed level of precision under some assumptions.

### Weaknesses
Unfortunately, several key claims and assumptions in the paper are mistaken.

1. Theorems 1.3 and 5.3 are not correct. The statements clearly articulate the $O(n^{1+o(1)})$ runtime of the proposed algorithm while the proofs directly show that the algorithm requires $O(n \log n)$ operations which is greater than $O(n^{1+o(1)})$.
2. It invalidates one of  the main claims of the paper (lines 155-159) that their algorithm is in the same complexity class as the less complicated previous algorithms [1, 2] for approximate softmax attention computation.
3. Theorems 1.5 and 5.2 are not complete. The proof considers only the case when $R_{i-j}=I$ (identity matrix). What about more general case, can it be reduced to the identity matrix?
4. The claims (lines 182-188) that in standard RoPE attention "the underlying matrix which exp is applied to no longer needs to have low rank" and the prior fast approximation techniques "fundamentally cannot apply to RoPE attention"  seem to be poorly worded and mistaken. Attention matrix $A \in \mathbb{R}^{n \times n}$ is a product of  **Q** and **K** of size (n, d) each, thus having rank at most d (d<<n). Applying the matrix projection $R \in \mathbb{R}^{d \times d}$ to either $Q$ or $K^\top$ does not change their dimensions, so **A** still maintains its rank. The same is true when a vector  $Q_i$ or $K_j$ is projected onto $R_{i-j}$, and then all $n$ vectors are stacked to form the entire **Q** or **K** matrix.
5. Moreover, usually RoPE are applied efficiently in entry-wise fashion with $O(nd)$ time as opposed to $O(nd^2)$ time when $R_{i-j}$ is a (d, d)-size matrix. In conjunction with previous point, it weakens the cause for implementing the algorithm, at least for standard RoPEs.
6. The authors state that standard RoPE are a special case of their proposed general RoPE matrices. They also state that any sequence of $R_{i-j}$ realizations can be constructed in such a way that the resulting pre-softmax attention becomes a rescaled circulant matrix (lines 186-188). They expand on it in Claim 4.8-4.9 as equation in line 415 can characterize  a rescaled circulant matrix if $C^{l_1, l_2}$ is a (possibly rescaled) circulant matrix. 

However, it's not possible to choose underlying elements $l_1, l_2$ of $R_{i-j}$ in case of standard RoPE in such a way that $C^{l_1, l_2}$ is a (rescaled) circulant matrix. To see that, construct matrix $C^{1, 1}$ with $C_{i,j}^{1, 1}=\cos (\theta (i-j))$. Its elements depend only on difference of positions (i-j). It's a symmetric matrix with an additional structure which differs from circulant matrix structure. Neither its possible to express this matrix as a rescaled circulant matrix for any sequence length n in general.

Thus, standard RoPE cannot be viewed as circulant-matrix inducing structure, and we get a contradiction.

7. In line 187 authors make an even stronger claim that "by picking the Rj−i entries appropriately, one can choose M to be any circulant matrix" (not merely *some rescaled* circulant matrix  but *any fully* circulant matrix) which is not further proved or discussed.

References:
[1] Josh Alman and Zhao Song. Fast attention requires bounded entries. In NeurIPS, 2023.

[2] Feyza Duman Keles, Pruthuvi Mahesakya Wijewardena, and Chinmay Hegde. On the computational complexity of self-attention. In International Conference on Algorithmic Learning Theory, pp. 597–619. PMLR, 2023.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a method to compute RoPE attention in nearly-linear time. 
More formally, given a set of matrices $W_i$, for integer $i \in [-(n-1), (n-1)]$, and matrices $Q$, $K$ and $V$, the matrix $A$ defined as 

$$A_{i,j} := \exp\left( Q_{i,\*} W_{i-j} K_{j,\*}^T / d  \right)$$

and $D := diag(A \mathbb{1}_n)$.
The goal is to approximate the matrix $D^{-1} A V$.

This is done through a decomposition of the matrix $A$ into a sum of rescaled circulant matrices $A^{\ell_1, \ell_2}$. This allows to compute a product of $t$ such matrices and a vector in time $O(t n \log n)$ using Fast Fourier Transform. This fact is used to compute the desired matrix using the polynomial method

### Strengths
- This paper adapts the algorithm proposed by Alman & Song, (2023; 2024a;b) for a wider variety of attention mechanisms.
This is a fairly important step for the overall optimization of general transformer algorithms.
- The presented method is applicable to a large variety of RoPE-style attention mechanisms.

### Weaknesses
The overall structure of the presentation is good. However, there are several minor issues that make the paper difficult to understand. I suggest the following actions:
- In introduction, define what $B$ is before discussing it.
- Replace "We define" in line 410 with "Recall that".
- Move the definition of $A^{\circ t}$ from the beginning of preliminaries inside Lemma 5.1, since it is the only place where it is used.

### Questions
All of my questions got resolved during the discussion process. It also allowed me to correctly understand the proofs inside the paper, which altered my recommendation.

### Soundness
3

### Presentation
3

### Contribution
3
