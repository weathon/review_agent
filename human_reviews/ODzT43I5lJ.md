# Differentially Private One Permutation Hashing

- Decision: Reject
- Scores: 5, 5, 3, 5

## Abstract
Minwise hashing (MinHash) is a standard hashing algorithm for large-scale search and learning with the binary Jaccard similarity. One permutation hashing (OPH) is an effective and efficient alternative of MinHash which splits the data into K bins and generates hash values within each bin. In this paper, to protect the privacy of the output sketches, we combine differential privacy (DP) with OPH, and propose DP-OPH framework with three variants: DP-OPH-fix, DP-OPH-re and DP-OPH-rand, depending on the densification strategy to deal with empty bins in OPH. Detailed algorithm design and privacy and utility analysis are provided. The proposed DP-OPH methods significantly improves the DP minwise hashing (DP-MH) alternative in the literature. Experiments on similarity search confirm the effectiveness of our proposed algorithms. We also provide an extension to real-value data, named DP-BCWS, in the appendix.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper explores differentially private one permutation hashing (DP-OPH) methods and develop three variants based on the densification procedure of OPH.  This paper conduct privacy and utility analyses for these algorithms and demonstrate that DP-OPH outperforms the DP MinHash alternative previously suggested for hashing Jaccard similarity across various privacy settings.

### Strengths
[S1] The motivation of this paper is clear, which is to protect the privacy of one permutation hashing (OPH) to meet differential private definition.

[S2] This paper is easy to read with necessary introduction of background and preliminaries.

[S3] This paper provides extensive theoretical analysis with formalized proof. This paper also provides fair experiments with clear results.

### Weaknesses
[W1] The technical depth and contribution are limited. This paper introduces randomized response technique to randomize output, which is a common idea that has been studied in various previous works. Although this combination successfully outputs a differential private OPH, as claimed by the authors, the technical challenge and depth are not clear shown to highlight contribution. 

[W2] The assumption “each data vector contains at least f non-zeros” in Sec 3.2 remains to be discussed more seriously. The number of non-zeros should obey some probability distribution. Although the probability might be small enough to be negligible if the f is set properly, one should quantificationally capture the probability or provide necessary reference to make the assumption more rigorous. In addition, to lead following contribution, the authors claim K is usually around hundreds in Sec 3.2, which also should be provided with a reference to support the claim.

[W3] The language level is sub-standard, which requires further polishment. Some conceptual information is missing. For example, the definition and procedure of SortedIndex and MapToIndex in Alg.3 are not introduced, making readers confused about the procedure of Alg.3.

### Questions
[Q1] The authors could consider to provide more discussions, including but not limited to the limitation and potential applications of the proposed schemes.

[Q2] This paper could benefit from language improvements.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces and analyzes a few algorithms for differentially private binary sketches. The proposed setting features binary databases of size $D$, and the goal is to compute Jaccard similarity between databases privately using sketches of size $\ll D$. The paper's proposed algorithms are primarily variants of one permutation hashing. One variant does not use densification (rand), one uses fixed densification (fix), and one uses re-randomized densification (re). The paper also introduces a private baseline based on minhash and provides both formal and empirical utility results demonstrating that, roughly, re dominates fix dominates minhash. Finally, empirical results show that rand in turn dominates re unless $\epsilon \approx 10$ or larger.

### Strengths
Private sketching is a natural problem, and the genomics example is well-motivated. Studying private versions of oph and minhash is also a natural approach, so even somewhat independently of whether they produce good algorithms or not, it's useful to have a study written down.

### Weaknesses
I think there is a potentially decent but smaller version of the paper contained in the current submission. Some specific comments:

1) I found the empirical results in Figures 3 and 4 surprising. After all the effort to set up and analyze fix and re, the simpler (and pure DP) rand clearly obtains the highest precision outside of extremely large values $\epsilon > 10$. Even when $\epsilon$ is large, the gap between rand and re is small. Given the effort the authors have expended on fix and re, I think it is valid to include them *somewhere* in the paper, but I suggest that they make more sense discussed in the Appendix as natural solutions that turn out to be less useful. Perhaps the authors think it's valuable to include fix and re because they have formal utility results that can be compared to minhash? If so, I understand this argument, but I still think the gap in performance is large enough that most or all of the focus should be on rand.

2) The lower bound $f$ required for fix and re is another disadvantage relative to rand. Also, how does "filtering" for $f$ (Line 418) affect the privacy guarantee -- what does the algorithm do if given a data points that fails $f$?

3) The introduction highlights that, unlike some past work, this approach does not assume private hash function randomness. I get that this is more flexible, but I don't understand why it's so meaningful. These algorithms still require that the bit flip randomness is kept private -- why is the hash function assumption so different?

4) A utility analysis for rand would help complete the paper structure suggested above. Have the authors attempted this, or do they have reason to believe it's hard?

### Questions
(See Weaknesses above.)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper studies the problem of similarity search based on the measure of Jaccard similarity, under the setting of differential privacy.

Jaccard similarity between two "documents" are defined as follows:
* Documents are represented as binary vectors $\{0, 1\}^D$, e.g. these can be the indicator vectors of the bag of words.
* $J(\mathbf{u}, \mathbf{v}) := \frac{\sum_{i=1}^D u\_i v\_i}{\sum_{i=1}^D \max\\{u\_i, v\_i\\}}$ measures the ratio of the size of the intersection and the union of the two documents.

Given a collection of $n$ documents, $\mathbf{u}^{(1)}, \ldots, \mathbf{u}^{(n)}$, and a query document $\mathbf{v}$, we would like to recover the top documents $\mathbf{u}^{(i)}$ that have highest value of $J(\mathbf{u}^{(i)}, \mathbf{v})$.

The `MinHash` is a celebrated techique to construct a sketch of the given documents in order to efficiently estimate the Jaccard similarity against any given query document.

The goal in the paper is to construct differentially private versions of this sketching technique, where the notion of differential privacy holds against adding / removing one word from any given document.

The paper considers a $b$-bit version of `MinHash`, as well as methods involving "re-randomization" (there were introduced in prior work). These sketches are made differentially private by applying the randomized response mechanism.

The privacy analysis is performed that gives tighter bounds than basic composition, and some experimental results are provided on some practical datasets (the Leukemia gene expression dataset, and Webspam dataset), showing that the DP versions of the `MinHash` variants with re-randomization achieve better precision.

### Strengths
The paper introduces DP variants of `MinHash` sketching technique and provides a detailed analysis.

### Weaknesses
I felt the presentation in the paper is quite poor. I had to re-read several parts of the paper, and I still don't fully understand many of the details. (See some of my questions below.)

Presentation aspects aside, I am also confused about the motivation or potential application of these methods.
In the experiments, the precision values are reported, by which I think means that the DP hashes to recover the documents with most similarity. The evaluation of precision itself is not a private operation though, as it requires comparing against the true Jaccard similarity.
I understand that this is done just to demonstrate the utility of the approach, so it is okay that computing precision is non-private.
But how do we expect to use these DP methods in any case? Sure, we can compute the approximate Jaccard similarity of a query against documents in a database, but then if we actually retrieve the most similar document and look at its contents then it is not private anymore.

### Questions
The description of `OPH-re` is confusing to me. I read it several times and I still don't understand what the method is doing. In particular, I did not get what $SortedIndex$ and $MapToIndex$ mean and what $\pi^{(k)} : \pi(\mathcal{B}_{k'}) \mapsto MapToIndex$ means. It might help to work through some example to explain the algorithm.

In `DP-OPH-rand` (Algorithm 5), is there some typo on Line 4 in that should the $\epsilon$ be $\epsilon'$ ?

Is it possible to simplify Theorem 3.6 in some way to highlight the main take-away as opposed to have detailed calculations within the theorem statement that take half a page to even state?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies the differentially private hashing and sketching, for the task of computing Jaccard similarity index. It starts by reviewing several classical, non-private hashing schemes for this task, including MinHash and One Permutation Hashing (OPH) from prior works.

It then designed new privatized versions of these hash schemes, including DP-OPH-re, DP-OPH-fix, DP-OPH-rand, etc. The paper proved that these algorithms satisfy either pure-DP or approximate-DP, and conducted a number of experiments comparing all these methods. The general conclusion seems to be that DP-OPH-re is the most effective approach.

### Strengths
* Designing differentially private sketching/hashing schemes is a well-motived and important problem in differential privacy and massive data processing. This paper studies one of the most fundamental problems in this direction.
* The algorithms are clearly explained and easy to follow.

### Weaknesses
* The main privatization technique used in this paper seems to be the randomized response mechanism. It is hard to believe that this can be effective.
* To privatize the OPH-re/fix schemes, this paper proposes to consider the relaxation of approximate privacy to avoid paying a huge price on "epsilon". I have some concerns on the choice of $\delta$ here.
* The algorithms proposed in this paper try to protect the privacy of a single bit in a binary feature vector $u\in \{0, 1\}^D$. In a more realistic setting, though, I would imagine $u$ being the feature vector of a client, and it must be protected as a whole ---- I understand this sounds very hard. But it needs to be justified to choose $u_i$
* The theoretical discussion in the introduction is almost impossible to follow -- I would suggest emphasizing the asymptotic behavior and deferring equations to the appendix when possible.
* Some issues with the experiments need to be clarified (see below)

### Questions
* I think the standard setup of $\delta$ in DP theory/practice is to make it "cryptographically" small (smaller than inverse data set size, at least). It seems you ran experiments within the regime that $\delta \approx 10^{-6}$. What happens if you scale it further?
* Can you comment on your definition of "adjacent inputs" in defining privacy? In other words, how to interpret the implication of protecting a single bit in the feature vector?
* Could you clarify the interplay between $eps,b,K$ and the variance of your hashing scheme? In particular, I think the quality of the DP-OPH would degrade significantly once $b$ goes moderately large. On the other hand, setting a smaller $b$ makes the hashing itself less accurate. Can you compare your results with non-private algorithms? I apologize if these are written in the text (and would appreciate it if you can provide a pointer)
* What is your definition of "precision" in the experiment? Also, why not include a non-private algorithm for comparison?

### Soundness
2

### Presentation
2

### Contribution
2
