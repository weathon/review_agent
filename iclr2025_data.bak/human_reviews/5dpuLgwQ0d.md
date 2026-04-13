## Human Reviewer 1

### Summary
This paper introduces a nearly linear time algorithm designed to determine the number of clusters in a graph using the eigengap heuristic. The primary contribution is an algorithm that counts the eigenvalues of the normalized adjacency matrix within a specified interval [a,b]. The author then leverages graph sparsification and this eigenvalue-counting algorithm to estimate the number of clusters $k$.

### Strengths
The paper makes a valuable contribution to the scalability of spectral clustering for large graphs. Combined with [Peng et al., 2017], it demonstrates that both the eigen-gap heuristic to infer $k$ and spectral clustering to infer the clusters can be executed in nearly-linear time.

### Weaknesses
* The datasets used in the experimental evaluation are limited in size. Testing on larger graphs is necessary to conclude the algorithm's empirical runtime, especially as the paper emphasizes scalability as a key benefit. For example, testing on graphs with 10^4, 10^5, 10^6, and more vertices (or edges) would show how runtime empirically scales with graph size.

### Questions
* Remark 2: This section is somewhat unclear. In line 191, $\beta$ appears with a factor of 2 and requires $\beta > 2$, while line 194 does not include the factor of 2 and suggests $\beta > 1$. This may be a typo, as line 194 may need to specify $\beta > 2$ as well. Additionally, it seems this assumption primarily applies to the interval sizes in step (iii) of the algorithm. For clarity, it might be preferable to assume something like $\lambda_k(M) \ge (2+\epsilon) \lambda_{k+1}(M)$ for some $\epsilon > 0$ and proceed from there.

* It seems the authors' proposed sparsification step requires prior knowledge of $k$, given that the definitions of $p_{u}(v)$​ and $p_v(u)$​ on lines 177 and 180 rely on the quantity $ \lambda_{k+1}(N_{G}) $. Could the authors clarify if the method can avoid this dependency on $k$ or if alternative sparsification approaches, such as spectral sparsification by Spielman, Srivastava, or Teng, might also be suitable?

* Line 215: what is $h_{ab}(M)$? If $h_{ab}$ is applied entrywise, I am not sure to understand where the equality comes from. 

* Line 321: What is the justification for $\|T_k(M)\|_2 \le 1$? Additionally, the third inequality in this line also assumes $\|T_k(M)\|_F^2 \le n$, which is not explicitly shown. Further explanation would be helpful.

* Line 494: If $p$ and $q$ are fixed while $n$ increases, should the number of edges grow at a rate of $n^2$ instead?


Minor Points:

Line 38: “graph with n”: missing the word “vertices.”

Line 42: “nearly-liner” should be corrected to “nearly-linear.”

Line 303: The symbol should likely be $\bar{M}$ rather than $\bar{A}$.

Line 464: The sentence in this line lacks clarity and may need rephrasing for coherence.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
3

---

## Human Reviewer 2

### Summary
Given a graph $G(V,E)$ with $|V|=n$ and $|E|=m$ and $k$-many well-defined clusters, the paper aims to determine $k$ efficiently. The authors work under the assumption that the graph has a large eigengap, that is, the $k$-th eigenvalue of the normalized adjacency matrix $N_G$ is much larger than the $k+1$-th eigenvalue. In this setting, the authors provide an algorithm that guesses $k$ correctly with probability $0.9$  in $\mathcal{O}(m \cdot {\sf poly} \log (n) )$   (or $\tilde{\mathcal{O}}(n)$ ) time.

Below, I describe the steps of the algorithm and point out inconsistencies in the algorithm/proof arguments that I have observed. 

---



**Step 1 (Section 3.1):** First, they use a "cluster preserving sparsifier" from existing work (Sun and Zanetti, 2019) that can sparsify the graph $G$ to $H$ with $\tilde{\mathcal{O}}(n)$ edges while ensuring the aforementioned eigengap is maintained. 


> **First inconsistency:**  The sparsifier method itself needs the knowledge of $k$ (which the overall algorithm is trying to determine). 

The sparsifier decides two values $p_u(v)$ and $p_v(u)$ for each edge $e(u,v)$. Then each edge is sampled with probability $p_u(v)+p_v(u)-p_u(v)*p_v(u)$. However, as described in lines *176* to *182* of the paper, $p_u(v)$ requires the knowledge of $\lambda_{k+1}(N_G)$, that is, the $k+1$-th eigenvalue of $N_G$ (where $k$ is the underlying number of clusters that the authors are trying to determine). This makes for a *circular* argument. One needs to know $k$ to obtain $\lambda_{k+1}(N_G)$, and the paper's goal is to determine $k$ itself. 
This implies that either the algorithm is incorrect (if Sun and Zanetti themselves need the knowledge of $k$) or that Sun and Zanetti's paper can determine $k$ in linear time already, which makes this paper redundant. 

----


**Step 2 (Section 3.2):** Then, the authors use recent analysis from [1] in a straightforward manner (with some modifications) to design Algorithm 1 of the paper, which given a matrix $M$, and two reals $a$ and $b$, counts the number of eigenvalues in the range $[a,b]$ in $\tilde{\mathcal{O}}(n)$ time.


----



**Step 3 (Section 3.3):** They want to count the number of eigenvalues of $N_H$ in different ranges using the previous algorithm, and they want to exploit the fact that if there is a large gap between $k$ and the $k+1$-th eigenvalue, the counting algorithm will return the same output ($k$) around this point even when the search range is expanded. 

To do this, they search in the range $[1-(\beta)^i/n^2,1]$ for increasing $i$, starting from $i=1$. They first keep increasing $i$ until the returned count is greater than $1$ (to ensure they at least get the second eigenvalue in their range) and then keep increasing $i$ until they get the same count for some $[1-(\beta)^{i'}/n^2,1]$ and $[1-(\beta)^{i'+1}/n^2,1]$ (that is, from the time their search range found the second eigenvalue and up until this point every increase in $i$ results in more eigenvalue being found in the range), which indicates they have encountered the assumed eigengap. At this point, they stop their algorithm and return the number of eigenvalues counted in this range as the number of clusters $k$. 

> **Second inconsistency (correctness):**  The authors do not prove that this algorithm cannot terminate for some $k'<k$ value. 

The paper assumes that $\lambda_{k+1}(N_G)-\lambda_{k}(N_G)$ is large, and therefore $\lambda_{k+1}(N_H)-\lambda_{k}(N_H)$ is also large. Therefore, the counting algorithm would return the same count in this neighborhood. However, the paper does not argue why $\lambda_{j+1}(N_H)-\lambda_{j}(N_H)$  is sufficiently small for all $2\le j<k$ (so that count keeps increasing for every increasing $i$ until the range reaches the eigengap). Without such a guarantee, the correctness of the algorithm is not complete. 

p.s. Here I note that if all the clusters are of the same size and are somewhat uniform (as in the SBM experiments by the author), then it is reasonable to believe that $ \lambda_{j+1}(N_H)-\lambda_{j}(N_H)$ is very small for $2\le j<k$. However this does not seem to hold in the general case. 

 
---

[1] Vladimir Braverman, Aditya Krishnan, and Christopher Musco. Sublinear time spectral density estimation. In 54th Annual ACM Symposium on Theory of Computing (STOC’22), pp. 1144–1157, 2022.

### Strengths
Fast algorithms for unsupervised inference are always a welcome addition to the literature. The authors have also implemented the algorithms and showed some experimental results, which is nice. Given the mathematical inconsistencies I have observed, I cannot further comment on the paper's strengths. Overall it seems that the paper needs significant revisions before it can be published.

### Weaknesses
The primary weakness is the lack of clarity on the correctness of the algorithm that I have mentioned in the summary. To re-state:

1) The authors use a sparsifier from the work by Sun and Zanetti, 2019 which itself requires the value of the $k+1$-th eigenvalue of the original graph. This makes the current algorithm incorrect as the sparsifier itself needs the knowledge of $k$ (to obtain $k+1$-th eigenvalue.

2) The authors also do not provide a complete proof of the correctness in the last step of the algorithm (described in detail in the Summary).

### Questions
Could the authors comment on the inconsistencies in the algorithm/proof I mentioned in the Summary? I look forward to reading the author's explanations.

### Soundness
1

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
5

---

## Human Reviewer 3

### Summary
Given an undirected graph that has a significant eigen gap between k-th and (k+1)-th largest eigenvalues, this paper gives a near-linear (in the number of edges) time algorithm for identifying k. 

The algorithm is built on several known results, but the combination of these is natural and elegant. The first step is to reduce the problem to counting the number of eigenvalues in a given range [a, b]. For this task, this paper makes use of Chebyshev’s polynomial and expansion, to reduce this to a trace estimation problem. This problem is then solved by employing Hutchinson’s estimation, and this paper gives a new concentration bound of this estimation (specifically for their trace approximation problem).

Experiments are conducted to validate the performance of the proposed algorithm, on  synthesized datasets by stochastic block model and some other synthesized datasets generated by scikit-learn. The real world running time is reported.

### Strengths
- The result is clean, and the eigen gap assumption seems to be natural (and was also justified in previous works)

- The near-linear time algorithm is very efficient

- The presentation is very clear

### Weaknesses
- Technical contribution seems to be limited, since almost all steps are using known/standard techniques. However, I do find the combination of them natural and elegant

- The experiments did not compare with any other baselines

- The experiments did not use real datasets and only small-scale synthesized dataset is used

### Questions
- In the statement of Lemma 14, epsilon only appears in the running time of CountEigenvalues, and I don’t see how it affects the accuracy. Actually, I think your CountEigenvalue has this epsilon as input. I’m not saying the proof has a problem; I’m just saying the statement loos strange.


- Section 3.3, in the description of the main algorithm, do you also need to specify the epsilon for CountEigenvalues? In this context, I think you need to say epsilon is an input parameter?

- Actually, Theorem 6 does not have the parameter epsilon, whereas CountEigenvalues need a parameter epsilon. However, how this epsilon is picked in order to prove Theorem 6 is not clearly discussed. Does it depend on the universal constant C? If so, then maybe it’s good to state this dependence, since perhaps your algorithm is efficient even when C is not constant.

- In the statement of Theorem 6, k is defined as the number to satisfy \Upsilon_G(k) \geq C k. However, is k well defined, in particular, is the k that satisfies this unique? If it is not unique, then what’s the guarantee of your algorithm then?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper introduces a novel algorithm designed to determine the number of clusters (k) in a graph based on the eigen-gap heuristic, which historically requires high computational effort due to its dependence on calculating all eigenvalues of the graph's normalized adjacency matrix. This research addresses one of the bottleneck of spectral clustering, presenting a new methodology that computes k in nearly-linear time. The algorithm, with high probability, accurately computes the number of clusters by analyzing the normalized adjacency matrix of the graph through a sparse representation.
Central to this approach is the strategic use of orthogonal polynomials, which facilitate an efficient approximation of the graph's spectral properties. This method avoids the computationally expensive task of direct eigenvalue computation by instead approximating the trace of the graph normalized adjacency matrix.

### Strengths
The strategy of sparsifying the graph and using orthogonal polynomials to efficiently approximate the trace seems new and constitute a nice contribution.

### Weaknesses
1) The effectiveness of the algorithm heavily relies on specific assumptions about the graph's structure, particularly the presence of well-defined clusters and clear spectral gaps.
While the algorithm presented in the paper represents a substantial theoretical advancement in the field of graph clustering, its practical applicability might be limited by its stringent reliance on specific graph conditions,. A critical oversight is the lack of a mechanism within the algorithm to pre-assess whether a given dataset meets these conditions. Without preliminary testing to verify these prerequisites, users might apply the algorithm to unsuitable datasets, leading to poor performance or invalid clustering results. Incorporating a diagnostic test as an initial step in the algorithm could significantly enhance its utility. This modification would make the algorithm more robust and adaptable, extending its relevance and effectiveness across a broader range of practical scenarios where data conditions are not ideal or well-understood in advance.

2) The paper may not provide extensive comparative analysis with other state-of-the-art clustering algorithms (with model selection), particularly those that might use different approaches such as density-based clustering or machine learning models that do not rely on spectral properties. This limits understanding of where the presented algorithm stands in the broader landscape of model selection techniques.

### Questions
1) Given the critical reliance of the algorithm on specific structural conditions within the graph, could the authors elaborate on any potential methods or strategies to integrate a diagnostic test within the algorithm to pre-assess whether these conditions are met in a given dataset? This addition would be crucial for ensuring the algorithm's adaptability and effectiveness.

2) Could you run your algo on difficult synthetic data and compare it to some state of the art and simple algorithms (like cross validation of some unsupervised criteria)?


Minor questions:
- Define the notation  h_{a,b}(M), T_i(M)...

- Are the Chebyschev polynomials used for a specific reason or other orthogonal polynomials could be considered as well?

- In Lemma 14, the probability obtained should not depend also on epsilon? 

- It could maybe more insightful to state the main result with a probability that depend on the parameters of the problem (rather than 9/10).

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
5

### Confidence
3