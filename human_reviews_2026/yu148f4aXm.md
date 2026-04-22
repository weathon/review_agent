# Universal Ordering for Efficient PAC Learning

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 2, 6, 4, 4

## Abstract
We initiate the study of the \emph{universal ordering} problem within the PAC learning framework: given a set of $n$ samples independently drawn from an unknown distribution $\mathcal{D}$, can we order these samples such that every prefix of length $k \le n$ yields a near-optimal subset for training a PAC learner? 
This question is fundamentally motivated by practical scenarios involving incremental learning and adaptive computation, where guarantees must hold uniformly across varying data budgets. 
We formalize this requirement as achieving anytime-valid PAC guarantees. 
As a warm-up, we analyze the simple random ordering baseline using classical concentration inequalities. 
Through a careful union bound over a geometric partitioning of prefixes, we establish that it provides a surprisingly strong universal guarantee, incurring at most an $O(\log\log n)$ overhead compared to a random subset of size $k$. 
We then present a more powerful analysis based on the theory of test martingales and Ville's inequality, demonstrating that a random permutation achieves PAC guarantees for all prefixes that match the statistical rate of a random subset of size $k$, without the logarithmic overhead incurred by naive union-bound techniques. 
Our work establishes a conceptual bridge between universal learning on fixed datasets and the broader field of sequential analysis, revealing that random permutations are efficient and provably robust anytime-valid learners but opening the door to further improvements.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Let H be a class of finite VC dimension and D be a realizable distribution. Standard PAC bounds imply that if we take a uniformly random sample S_n from D of length n, then with probability 1 - delta, every S_n-consistent hypothesis will have D-error at most O(log(1/delta)/n).

In the same setting, let S_k denote the k-length prefix of S_n for k = 1, ..., n.

The paper asks: for which sequences {eps_k}_{k = 1} one can show that with probability 1 - delta, for every k = 1, ..., n, we have that all S_k-consistent hypotheses have D-error at most eps_k?

A trivial observation is that we can guarantee that for eps_k = (log n + log(1/delta)/k (I'm skipping O()-notation for better readibility) by applying standard bounds for every k = 1, ..., n with delta' = delta/n.

A slightly less trivial but also a simple observation implies that instead of the additional log n term in the numerator, one can get log log n term. Indeed, let's use the standard bound for k that are powers of 2 and with delta' = delta/log n. By the union bound, we have with probability 1 - delta that for every k = 1, ..., n which is a power of 2, every S_k-consistent hypothesis has D-error at most eps_k = (log log n + log(1/delta)/k. What about non-powers of 2? If k is not a power of 2, take k_0 to the biggest power of 2, smaller than k. Observe that all S_k-consistent hypotheses are also S_{k_0}-consistent, which means that they all have D-error at most eps_{k_0} \le 2 eps_k.

The main result of the paper is that this bound on eps_k is also true with no additional term that depend on n at all -- using martingale theory.

### Strengths
The paper motivates this problem by dynamic learning -- where the size of the data set is now known in advance. We might want to have good pac guarantees for all possible prefixes of the sample set (as good as they can get for such prefix size).

### Weaknesses
Most of the paper is mathematically confusing and needs careful re-writing (although results about log n and log log n overhead seem simple exercises and are restorable, see Summary).

Definition 3.4 is mathematically unclear. It is an ordering of what? What is the quantification over A? To be honest, it is not clear which object is defined to be ``Universally PAC-valid''. What makes more sense is the formulation from the Summary, so we can define something like ``a sequence {eps_k} of errors is universally pac-achiebable with delta if with probability 1 - delta...''

In Theorem 4.2, 4.3, what is eps? Why bounds on the error have n in the numerator, which makes that larger than 1? In the proof of Theorem 4.3 you don't need Lemma 4.2, see a simpler proof in the Summary section.

I did not understand the proof of Theorem 5.1. Already in the definition H_0^h, H_1^h, these are random events, this is either true or false for a given hypothesis h, so what sense does it have to consider conditional probability involving it? The definition of M_k does not seem to depend on D, how do you deduce that M_k is large if some S_k-consistent hypothesis has large D-error?

I think this paper has some potential, but it definitely needs a thorough rewritting.

### Questions
no questions

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the universal ordering problem in the PAC learning framework. The central question is whether, given $n$ i.i.d. samples from an unknown distribution, one can order these samples so that every prefix of length $k \leq n$ forms a near-optimal subset for PAC learning. This can help with more efficient use of data in computation and memory constrained settings. The authors formalize this requirement using anytime-valid inference techniques and study how random permutations of the dataset can provide such guarantees. The authors first analyse a naive union-bound argument, showing that random orderings incur a  $O(\log n)$ overhead for boosting the success probability. This is later improved to only a $O(\log \log n)$ overhead, since the authors exploit the fact that training on similar length prefixes (when their ratio is a small constant) is highly correlated and therefore suffices to only ensure a good performance at prefix lengths that grow exponentially. 
For their main result, they leverage martingale-based techniques and Ville’s inequality to eliminate this overhead entirely, demonstrating that random permutations achieve optimal PAC rates uniformly across all prefix sizes.

### Strengths
The paper formalizes a previously unexplored yet practically relevant problem and provides a starting point for its theoretical analysis. Conceptually, the paper builds a bridge between universal data ordering, sequential analysis, and safe anytime-valid inference, showing that random shuffling is not only convenient but also efficient.

### Weaknesses
However, the paper’s focus is limited to random orderings, leaving important practical and algorithmic questions open. Furthermore, even though the stated motivation is to find near optimal orderings, this is not reflected in the results, as no lower bounds are provided to support that. Presentation wise, I would like to see more details of the paper’s contributions/results in the introduction.

### Questions
- Do you conjecture that using random permutations is actually optimal, or could a deterministic or data-dependent ordering achieve strictly tighter bounds?
- The analysis assumes i.i.d. data and a consistent learner in the realizable setting. How sensitive are the results to label noise, covariate shift, or approximate consistency? Could similar anytime-valid results be extended to the agnostic PAC setting?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes and studies a problem called the ``universal ordering'' problem that is defined under the framework of classical PAC learning. That is, while traditional PAC learning characterizes the sample complexity as a fixed size of samples drawn from an unknown distribution, universal ordering considers a given large sample set and aims to achieve PAC guarantees on any prefix $k$ samples. The paper shows that random permutation can achieve this goal and provide satisfying universal guarantees. The paper first establish a baseline analysis of union bound on concentration inequalities and proves a $O(\log\log n)$ overhead. It then introduces a supermartingale analysis on the sequence of data that avoids taking union bound on all $n$ prefixes.

### Strengths
This paper proposes a very interesting problem of studying universal guarantees under the PAC learning framework, which is conceptually related to other fields of sequential analysis and safe testing. It could be of interest in many real-world applications when data sets are obtained and fixed, while the budget for learning is varying from time to time. It also provides a theoretical guarantee for the method of random permutation for data sets, which may often be used as a data preparation step in learning tasks.

### Weaknesses
Comparing to its conceptual contribution and the statistical analysis for the universal learning guarantees, the technical contribution seems not so compelling. The traditional PAC learning theory already assumes i.i.d. sampling from the underlying distribution, hence, any $k$ leading samples are already satisfying the universal ordering criteria. Given this, a random permutation seems just a renascent of the i.i.d. assumption of PAC learning. From this perspective, the study could be more application driven, i.e. many real-world data sequences lack the necessary randomness in it, and random permutation is a valid resolution for inserting such randomness, benefiting learning guarantees.

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes the novel **universal ordering** problem, which asks whether $n$ i.i.d. samples from an unknown distribution $\mathcal D$ can be arrange in a fixed sequence so that every prefix subsequence of length $k \geq n$ forms an approximately optimal subset for PAC learning. Using two different arguments, it establishes baseline results for the case when the samples are randomly arranged and the hypothesis class is either finite, or it has a finite VC dimension. The second argument (based on Ville's inequality) yields a strictly better bound than the first argument (based on a union bound). Both bounds provide PAC guarantees in comparison to the optimal statistical rate of a random subset of size $k$.

### Strengths
1. The proposed **universal ordering problem** is a novel and potentially significant idea: arrange the training data so that we can directly compare models that fit any $k$-prefix of the dataset against each other.
2. The second argument establishes a technique to work with randomly arranged data using anytime-valid inference and demonstrates that it allow us to obtain stronger results than using union bounds.
3. The paper is clear and concise. The proofs are well-written and seem to be correct.

### Weaknesses
1. Despite the promising premise of ordering training data in an optimal manner, the paper only deals with the case when we're arranging them in a uniformly random order.
2. The paper lacks a demonstration, be it theoretical or empirical, of how models training with different dataset sizes can be compared against each other.

### Questions
1. The statements of Theorem 4.1, 4.3, 4.4 should precisely state which terms are being referenced by the word *error*.
2. There should be a brief explanation on why the bound on $\epsilon_k$ on line 566 leads to the bound in Theorem 4.1. A similar suggestion can be made to the proof of Theorems 4.3.
3. Assuming i.i.d. and finite training data, is there any way to arrange them that would meaningfully differ from a uniformly random arrangement? If the answer to the question is no then the paper would benefit from taking into consideration the non i.i.d. training data scenario.
4. Given that random permutations already achieve such a strong universal guarantee (achieving the optimal statistical rate), what could be expected from another permutation of the training data? Why do you think this ordering is unlikely to be optimal?
5. How was Figure 2 created?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors introduced the universal ordering problem within the PAC learning framework. As a baseline, they analyzed random orderings using standard techniques. They then refined this analysis via Ville’s inequality, which yields a tighter bound that unexpectedly removes all logarithmic factors.

### Strengths
- The paper presents original contributions.
- While I have not verified every proof in full detail, the arguments appear correct.
- I specifically like the Ville’s inequality idea.
- The paper is well-written and easy to follow. I particularly appreciated the step-by-step progression of improvements.
- The paper includes a comprehensive discussion section.

### Weaknesses
- The main concern I have is that I’m not convinced of the importance of the problem. In particular, while I understand it on a mathematical level, the stated motivations did not resonate with me.
- The scope of the paper is limited by its exclusive focus on random orderings, which leaves algorithmic aspects unexplored. In a setting where the goal is to minimize the expected error, random ordering is clearly optimal. The work would be substantially more impactful if the high-probability optimal ordering differed from the random ordering.
- The discussion of related works could be strengthened. For example, the following paper appears relevant: https://arxiv.org/pdf/2202.05246

### Questions
- Could you elaborate on the key motivations for studying this problem?
- Is there any reason to consider non-random orderings in your formulation?

### Soundness
4

### Presentation
4

### Contribution
2
