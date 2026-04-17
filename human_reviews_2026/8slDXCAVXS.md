# Sublinear Time Quantum Sensitivity Sampling

- Decision: Reject
- Scores: 4, 8, 6, 2

## Abstract
We present a unified framework for quantum sensitivity sampling, extending the advantages of quantum computing to a broad class of classical approximation problems. Our unified framework provides a streamlined approach for constructing coresets and offers significant runtime improvements in applications such as clustering, regression, and low-rank approximation. Our contributions include:

* **$k$-median and $k$-means clustering:** For $n$ points in $d$-dimensional Euclidean space, we give an algorithm that constructs an $\epsilon$-coreset in time $\widetilde O(n^{0.5}dk^{2.5}~\mathrm{poly}(\epsilon^{-1}))$ for $k$-median and $k$-means clustering. Our approach achieves a better dependence on $d$ and constructs smaller coresets that only consist of points in the dataset, compared to recent results of [Xue, Chen, Li and Jiang, ICML'23].
    
* **$\ell_p$ regression:** For $\ell_p$ regression problems $\min_{x\in \mathbb{R}^d}\\|Ax-b\\|_p$ where $A\in \mathbb{R}^{n\times d}$ and $b\in \mathbb{R}^n$, we construct an $\epsilon$-coreset of size $\widetilde O_p(d^{\max\\{1, p/2\\}}\epsilon^{-2})$ in time $\widetilde O_p(n^{0.5}d^{\max\{0.5, p/4\}+1}(\epsilon^{-3}+d^{0.5}))$, improving upon the prior best quantum sampling approach of [Apers and Gribling, QIP'24] for all $p\in (0, 2)\cup (2, 22]$, including the widely studied least absolute deviation regression ($\ell_1$ regression).
    
* **Low-rank approximation with Frobenius norm error:** We introduce the first quantum sublinear-time algorithm for low-rank approximation that approximates the best rank-$k$ solution to a matrix $A\in \mathbb{R}^{d\times n}$ and does not rely on data-dependent parameters, and runs in $\widetilde O(nd^{0.5}k^{0.5}\epsilon^{-1})$ time. Additionally, we present quantum sublinear algorithms for kernel low-rank approximation and tensor low-rank approximation, broadening the range of achievable sublinear time algorithms in randomized numerical linear algebra.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a unified quantum framework to perform sensitivity-based coreset sampling in sublinear time. Applications include low-rank approximations for the Frobenius norm, kernel functions, and tensors, with extensions to subspace approximation, regression, and clustering. Their methods rely mostly on a method where approximate sensitivities are recursively computed and resamples are made based sensitivity estimates and where sampling is aided by Grover search to produce quantum speedups.

### Strengths
1. The paper provides quantum speedups for a variety of tasks. They provide the first sampling-based algorithms for several low-rank approximation tasks and algorithmic improvements for clustering and regression. 
2. The theoretical work and proofs are solid and written in great detail.
3. Query lower bounds are provided for spectral approximations.

### Weaknesses
1. The complexity terms in Tables 1 and 2 seem a bit too generous in some cases. For example, in the first row in Table 2, for the runtime cost for low rank approximations, the table states $$nd^{0.5}k^{0.5},$$ while Theorem C.5 states $$\epsilon^{-1}nd^{0.5}k^{0.5} + nk^{\omega-1}+\epsilon^{-1.5}n^{0.5}k^{1.5}+\epsilon^{-2}d^{0.5}k^{1.5}+\epsilon^{-3}kd.$$ Putting in the assumption that $\epsilon \in \mathcal{O}(1)$ and $k \le \min(n,d)$, we should get the reduction 
$$nd^{0.5}k^{0.5} + nk^{\omega-1}+kd.$$
Another example is the missing $d^\omega$ in the subspace approximation compared to Theorem E.2.
2. Minor point, but it might be better to separate classical and quantum prior algorithms into separate columns.
3. Further see questions below.

### Questions
1. In the paper, many runtime estimates are presented with the assumption that $\epsilon \in \mathcal{O}(1)$. I wonder how this asssumption is justified? To my knowledge, a lot of Grover-type algorithms in optimization (e.g. SDP solvers) have a decrease in dimension dependency but have an increase error dependence. How does this part compare to classical methods? 
2. Are there any theoretical proofs that overestimates can indeed suffice as a replacement for the exact estimate (or indeed, an $\epsilon$-close estimate)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a quantum computing approach to sensitivity sampling for coreset creation aimed at a range of machine learning tasks including clustering, linear regression, and low rank matrix approximation.

### Strengths
The paper builds on quantum leverage score sampling approach based on Grover’s search introduced by Apers and Gribling, and extends it to sensitivity sampling. Then, they show that the resulting coreset selection method leads to improvements in complexity for k-median and k-mean clustering, linear regression, and low-rank matrix approximation, offering a clear advanced over the current state-of-the-art. The fact that the proposed algorithm’s complexity does not depend on data-specific parameters is another strong point.

### Weaknesses
The paper operates within the QRAM-based model of computation, which, while common in the field, is likely to limit its practical applicability on medium-term hardware.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors study quantum speedups for *coreset construction problem*, i.e., given a set $A$ and a cost function $\text{cost}(\cdot,\cdot)$ on $A\times X$, find a subset $B$ such that $\sum_{b\in B} w_b f(b, x) =(1\pm \epsilon) \sum_{a\in A}f(a,x)$, which is a significant step in downstream optimization and data selection tasks. The standard approach performs a weighted sampling procedure where each point is drawn with probability $s_i = \max_{x\in X} \frac{cost(A_i,x)}{cost(A,x)}$, called *sensitivity*, and assigns weight $1/s_i$ to the drawn point which has sample complexity $O(\epsilon^{-2}(\sum_{i=1}^n s_i) \text{dim X} \log(1/\delta))$ to find an $\epsilon$ coreset with probability at least $1-\delta$. Since the computation of $s_i$ requires a pass over each element of $A$, this process has $O(n)$ dependency. The goal of the paper is to reduce $n$ dependency to sub-linear by using quantum computing techniques, mainly quantum amplitude amplification/Grover's search. The authors give hybrid algorithms to construct coreset for various problems including clustering, regression and low rank approximation.

The main ingredient seems to be the recursive procedure to do weighted sampling originally used by Apers &Gribling in the context of quantum linear programming. This procedure is used as a submodule in the various classical algorithms and the authors obtained speedups either in $n$ or problem dimension $d$ under QRAM model compared to both classical and quantum algorithms. The given algorithms achieve runtimes independent of data-specific parameters unlike the previous work.

### Strengths
The studied problems are relevant and important for modern machine learning algorithms. As the bottleneck to obtain sublinear runtimes is due to sensitivity computation, it is natural to consider quantum algorithms which are known to give speedups for similar problems. Although the quantum part of the algorithm is previously known, the paper has still include novelty as the previous work did not address the problems considered here. Some of submodules such as quantum weighted sampling could be seen as an independent tool that could be used for different problems in the future although it was developed in previous works.

Each algorithm has been given in detail in the appendix as pseudocode and the analysis seem to be thorough and correct. The paper does not provide empirical evidence, yet at this stage the proposed algorithms cannot be implemented on the existing hardware because of error overheads. However, the level of the rigor in the analysis is sufficient to compensate this.

The authors also proposed open questions for further directions which would be appreciated by the community for further research on the subject.

### Weaknesses
One major weakness is that the paper assumes access to quantum read/classical-write (QRAM) which could in general change the over all runtime of the quantum algorithms significantly, however the paper does not discuss the effect of this structure. Without this input model, the quantum state needs to be prepared from scratch which adds additional loop to each of these algorithms. It seems that the previous paper by Apers & Gribling also have this input model, and the remark in that paper states that this assumption can be removed by paying poly cost in *time* which can potentially remove the claimed speedup over classical algorithms. Note that this is not  very important in Apers & Gribling paper as their claim is on the *query complexity* not time. If this remark is not applicable in this paper, this has to be discussed and justified properly. 

It is also known that realizing QRAM structure might be challenging. From the theoretical perspective, it is completely valid to assume this structure; however this also needs to be highlighted in the main text. The paper also claims that the runtime is independent of data set parameters. However, the loading to QRAM initially might have data dependent overhead. Similarly input representation depends on how QRAM is implemented, therefore it becomes important whether we can load this set to QRAM. 


It is also not clear how Grover’s algorithm is incorporated into different classical algorithms. Although the main body introduces the Grover’s algorithm, it is not clear at intuitive level how each application makes use of QSample routine,. Along the lines of my previous comment, it seems that QRAM model also makes the algorithms somewhat straightforward (not to trivialize the contribution of the paper but I would think without QRAM, the algorithms might need significant modification).

### Questions
Questions/Suggestions

1. The runtime $T_\text{sensitivity}$ does not seem to be instantiated for the specific problems in the main text and details seem to be postponed. Can authors give explicit bounds for this for different applications?
2. Based on my comment above, how would QRAM assumption change the runtime and the algorithm overall?
3. Is $\delta$ around lines 60 failure probability of the algorithm? 
4. Around lines 212, I believe the algorithm is to repeat Grover’s algorithm m times where each run takes $O(\sqrt{n/m})$ calls?
5. Lemma A.14 is the result for main quantum procedure, but it is not referenced properly from the main text. The runtime does not seem to be trivial by itself.
6. Around line 221, if p_i does not form a distribution, the paper should not say probabilities but it might say a measure proportional to weights p_i. Otherwise, it is a confusing statement.
7. The paper claims sublinear runtime but some results are sublinear in $n$ and some of them are in $d$. This should be clarified at the beginning.

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
4

### Summary
This submission presents a sublinear-time quantum algorithm for constructing a coreset via sensitivity sampling, which can be used as a subroutine to speed up other problems such as k-median and k-means clustering, $\ell_p$ regression, and low-rank approximation. The main source of quantum speedup comes from a variant of Grover's algorithm for sampling a subset, which relies on an oracle for computing the sensitivity (or Lewis weights in some applications).

### Strengths
This manuscript contains a lot of results that improve on previous ones in a certain regime. The main technical contribution is from the classical side, where the authors focused on how to efficiently estimate sensitivity or weights in different applications so that the quantum algorithm can use them. The results are summarized in theorems which are supported by proofs.

### Weaknesses
First of all, this submission is difficult to read. Maybe this is natural since it contains too many results. I appreciate the high-level technical overview in Section 2; however, I found that the technical overview is still very technical. Reading Section 2 doesn't help with grasping the intuition; I would rather directly read the appendices. I think the authors should work on improving the presentation in this section.

In addition, the presentation lacks clarity. For example, in Line 57, is the number of samples here the size of the coreset? In Line 84, does the final sample size mean the same thing? In Lines 164-165, is the $s$ (upper bound on the total weight) the same as the $s$ in Line 84, and also the $s$ in Line 315? In Lines 220-222, the description of the sampling process is rather confusing. Why not just use the language of "sampling a subset" as in Apers & Gribling (2024)? In Line 233, the matrix $S$ is not defined. In Lines 243-244, "uniformly sample half of the points from $A'$" should be stated more precisely here.

Besides the presentation issues, there's a more critical technical flaw. In the statement of the theorems, a classical oracle that outputs an estimation of sensitivity (or weights) is assumed. However, in Grover's algorithm, one needs a quantum oracle. If a classical subroutine (a classical white-box) is given, additional work needs to be done: one first needs to turn the classical subroutine into a quantum circuit, and then clean the garbage during the conversion by uncomputing the ancillas. It could be more complicated when the classical subroutine generates approximations of the true value with a success probability (like Algorithms 4, 5, 6). In this case, the corresponding quantum oracle will produce a superposition of an estimate (with bounded approximation ratio) and some garbage state. Also, the portion corresponding to the desired estimate itself is a superposition of different approximations. Maybe I am wrong, but I don't think directly plugging this oracle into Grover's algorithm will produce the correct state. Some additional steps need to be done, including concentrating the estimate using the median trick. Also, a careful error analysis of the quantum algorithm with this concentrated oracle should be done to prove the correctness of the entire approach.

There are other minor things:
1. In the second and third bullet points of the abstract, $n, d, k$ are not defined.
2. In Theorem 1.1, a quantum algorithm is randomized, so there's no need to say a "randomized" quantum algorithm
3. Line 202, further extend -> we further extend

### Questions
I would be willing to raise my score once my concerns in the previous section can be addressed without a substantial revision.

### Soundness
2

### Presentation
2

### Contribution
3
