# On Differentially Private String Distances

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 2

## Abstract
Given a database of bit strings $A_1,\ldots,A_m\in \{0,1\}^n$, a fundamental data structure task is to estimate the distances between a given query $B\in \{0,1\}^n$ with all the strings in the database. In addition, one might further want to ensure the integrity of the database by releasing these distance statistics in a secure manner. In this work, we propose differentially private (DP) data structures for this type of tasks, with a focus on Hamming and edit distance. On top of the strong privacy guarantees, our data structures are also time- and space-efficient. In particular, our data structure is $\epsilon$-DP against any sequence of queries of arbitrary length, and for any query $B$ such that the maximum distance to any string in the database is at most $k$, we output $m$ distance estimates. Moreover,

- For Hamming distance, our data structure answers any query in $\widetilde O(mk+n)$ time and each estimate deviates from the true distance by at most $\widetilde O(k/e^{\epsilon/\log k})$;
- For edit distance, our data structure answers any query in $\widetilde O(mk^2+n)$ time and each estimate deviates from the true distance by at most $\widetilde O(k/e^{\epsilon/(\log k \log n)})$.

For moderate $k$, both data structures support sublinear query operations in the combined size of the query and its output. We obtain these results via a novel adaptation of the randomized response technique as a bit flipping procedure, applied to the sketched strings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies the following problem: given binary strings $A_1, \ldots, A_m$ of length $n$, efficiently compute an $\varepsilon$-DP data structure that can take any new length $n$ binary string $B$ and approximate its distance to each of $A_1, \ldots, A_m$. It provides time- and space-efficient solutions for both Hamming and edit distance. The Hamming distance algorithm works by incorporating randomized response into an existing algorithm [PL07] for the non-private algorithm and, given $B$ within distance $k$ of each $A_i$, outputs estimates with error $\tilde O(k/e^{\varepsilon/\log(k)})$. The edit distance algorithm is only partially described in the main body and, given $B$ within distance $k$ of each $A_i$, outputs estimates with error $\tilde O(k/e^{\varepsilon / [\log(k) \log(n)]})$.

### Strengths
String distance is a natural problem, and I am not aware of past work on doing it with DP. The approach for Hamming distance is intuitive.

### Weaknesses
1) For Hamming distance, unless $\varepsilon$ is very large, the error bound $k/e^{\varepsilon/\log(k)})$ is close to the (trivial) error bound $k$ baked into the theorem assumption. The same problem holds (to a greater degree) for edit distance. This might be OK with even partial lower bounds, but no lower bounds are provided. Since $\varepsilon$-DP is a fairly meaningless privacy guarantee unless $\varepsilon$ is a small constant (say, $\varepsilon \ll 10$), these are very weak utility results (and this doesn't attempt to reason about whatever constants and log factors are being hid here).

2) The notion of neighboring distances is restrictive: a neighboring databases can differ in one bit of one string. Using the motivating example from the intro (~Line 58), this is record-level (whether a person has condition X) rather than person-level (whether a person is in the database) privacy.

3) The Hamming distance algorithm is just a pre-existing algorithm + randomized response, and the edit distance algorithm isn't described in the main body. This means the algorithmic novelty of the main paper is low. (I also don't see why the edit algorithm has to be postponed for space -- there are many instances of unnecessary pseudocode and spacing, so I'm sure there are a few paragraphs of space to be had with some light editing.) There are also some possible technical problems (see next section).

4) The discussion of boosting (Remark 3.8) doesn't observe that $\varepsilon$ would also need to be split to maintain a fixed privacy level, and the effect of splitting on boosting success probability is unclear to me.

5) I have previously been involved in two review processes for this paper. Both times, multiple reviewers asked about issues like those above (and others), and the authors silently withdrew the paper without responding to comments and resubmitted it to another conference with minor cosmetic changes. I understand that it is challenging to incorporate reviewer feedback, and not every reviewer is correct all the time, but it is disappointing to see authors just ignore many similar reviews and keep resubmitting essentially the same paper.

### Questions
In addition to the weaknesses mentioned above, here are some more concrete questions for the authors:

1) What does $A[p:q]$ mean?

2) What are the base cases of $D(i, j)$? What happens when the set used to define $F(r, d)$ is empty? What are the actual domains of these functions and $EXTEND$ and $LCP$? The paper quickly writes out some recursive definitions, but base cases are glossed over.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the following problem: given a dataset of $m$ binary strings, each of length $n$, the goal is to construct a data structure that satisfies $\epsilon$-differential privacy (under the notion where two neighboring datasets differ in a single bit of a single string). Given a query string of length $n$, the data structure should output estimated Hamming or edit distances between the query and each string in the dataset.

The paper assumes that the distance between the query string and each dataset string is at most $k$. Under this assumption, the proposed data structure can be constructed in time $\tilde{O}(mn)$, and supports queries in $\tilde{O}(mk + n)$ time for Hamming distance and $\tilde{O}(mk^2 + n)$ time for edit distance. The resulting approximation error is on the order of $\tilde{O}(k)$.

### Strengths
The paper is generally well written and easy to follow.  

A notable feature of the proposed data structure is that it supports an unbounded (potentially infinite) number of queries while maintaining differential privacy.

### Weaknesses
The assumptions and resulting bounds in the paper do not appear to be meaningful (please correct me if I am mistaken).  


Let $A_1, \ldots, A_m$ denote the dataset strings and $B$ the query string.

1. The assumption that $D(A_i, B) \le k$ for all $i \in [m]$ implies that $D(A_i, A_j) \le 2k$ for all pairs $(i, j)$. 
         Consequently, all dataset strings must be highly similar to each other if $k$ is small. 


2. On page 6, the reported Hamming distance error bound is  
   $$
   \frac{k \log^3 k}{1 + e^{\epsilon / \log k}}.
   $$
   This bound is not meaningful for typical values of $\epsilon$, e.g., $\epsilon = O(1)$, in which case it simplifies to $\Theta(k \log^3 k)$.  

3. In comparison, a trivial estimator that always outputs $D(A_i, B) = 0$ already incurs an error of at most $k$, which is asymptotically smaller. Moreover, this baseline requires no preprocessing and achieves $O(1)$ query time.  

4. To achieve an error bound of $o(k)$, one would need $\epsilon = \omega((\log k) \cdot \log \log^3 k)$, which can be unrealistically large.  

5. If $k \in o(n)$, the assumption $D(A_i, B) \le k$ implies that all dataset strings are nearly identical—differing in only an $o(n)/n = o(1)$ fraction of positions.  
   This already covers the case where $k = \Theta(n^c)$ for some $c \in (0, 1)$, in which achieving meaningful accuracy requires $\epsilon = \Omega(\log n)$, again much larger than typical privacy budgets.  
   Of course, if $k = \Omega(n)$, we also have $\epsilon = \Omega(\log n)$.

### Questions
1. I would suggest introducing the definition of the neighboring relation earlier in the paper, rather than deferring it to the preliminaries section. While reading the introduction, I was looking for this definition for a while before eventually finding it later in the paper.

2. Line 262: the probability expression appears to contain a typo. It should be  
   $$
   \Pr[ \forall j \in [M_2], |T_j| \le 10 \log k ]
   $$
   instead of  
   $$
   \Pr[ \forall j \in [M_2], |  |T_j| \le 10 \log k ].
   $$

### Soundness
2

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
3

### Summary
This paper studies the problem of differentially private approximating bit strings in a database. In particular, given $m$ strings datasets, the task is to output a sketch while ensuring privacy that can be used to answer any query. This is one of the most fundamental data structure problem, 

The authors consider two different natural error metric: Hamming and edit distance and for each of them they provide algorithm that has error better than just using the Laplace mechanism.

### Strengths
They study one of the most fundamental data structure problems, and the error metric is fine.

Their results are clearly stated, which makes it easy to understand the merit of the paper. I really want to thank the authors for that.

### Weaknesses
The error on approximating Hamming distance between each query and database string scales as $k \log k$ (for typical small $\varepsilon$. I am confused as to why the algorithm's accuracy is in any way meaningful. 

The algorithmic novelty for the Hamming distance data structure is unclear. It is described as an adaptation of a non-private approach, followed by the use of a randomized response.

### Questions
Is the paper doing something more than just randomized response on non-private algorithm? 

What other notion of neighboring would make sense when studying string distance with DP? I am not from the string distance community, so this might sound naive,

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper considers differentially private algorithms for string matching. Specifically, consider a sensitive data set that is a list of $m$ strings of length $n$. The sensitive units are individual symbols of the strings. (So I guess in a database of DNA strings, one would want to protect individual alleles/mutations.)

The paper's algorithms output a data structure that takes as input a string $B$ and outputs estimates of the Hamming or edit distance to each of the $m$ sensitive strings. More precisely, the data structure takes a promise on the maximum distance $k$ between $B$ and any string in the database; it outputs a distance estimate with additive error $\tilde O(k/e^{\epsilon/\log k})$. (That is, it is gives a nontrivial accuracy guarantee for $\eps \gg \log k$.)

These algorithms are fairly fast—they run in quasilinear time $\tilde O(mn)$.

### Strengths
The paper addresses a well-defined mathematical problem and appear to involve nontrivial analysis of perturbed versions of string sketches (e.g. LSH ouputs).

### Weaknesses
* Motivation: The paper does not clearly list plausible settings where this notion of privacy (and accuracy) make sense. In the DNA application, for example, the privacy protection seems very weak.

* Significance: The algorithm provides nontrivial accuracy guarantees only for very large values of $\epsilon$. I couldn't understand a setting where this guarantee would be useful/important. (The paper mentions that $\epsilon = k \log k$ is "too large for most applications". Why is $\log k$ ok? Both are much larger than 1.)

### Questions
Could the utility guarantee be rephrased as a mixed additive/multiplicative error guarantee? Even having additive error larger than $k$ could be useful if the goal is to distinguish large distances from small ones.

### Soundness
4

### Presentation
3

### Contribution
2
