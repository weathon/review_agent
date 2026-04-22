# Index2Sort: Sorting Algorithm Using Static Index Data Structure

- Avg Score: 3.67
- Decision: Reject
- Scores: 2, 2, 8, 4, 4, 2

## Abstract
We introduce Index2Sort, a general framework for deriving sorting algorithms from static indexes.
Index2Sort treats the index as an opaque box that exposes only two operations: index construction and rank queries.
This abstraction allows Index2Sort to be applied to various index structures, including classical and learned indexes.
Our theoretical analysis shows that the computational guarantees of the index transfer directly to Index2Sort.
If the index can be constructed in expected time $\mathcal{O}(nC(n))$ and can answer rank queries in expected time $\mathcal{O}(Q(n))$, then Index2Sort sorts the input in expected time $\mathcal{O}(nC(n) + nQ(n))$.
In particular, when using a state-of-the-art learned index with $C(n)=Q(n)=1$, this yields an expected complexity of $\mathcal{O}(n)$, which is a strictly tighter bound than those of existing learned sorting algorithms.
In contrast to recent theoretical works on learned sorting, which derive complexity guarantees by analyzing the internal structure of a learned index and designing a sorting algorithm with a similar structure, Index2Sort achieves stronger guarantees without requiring any inspection or modification of the index internals.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the algorithm Index2sort to derive sorting algorithms from various indexing structures, classical or learned indexes, with computational guarantees based on the complexity of the index structures (construction and querying). In particular when using the best known learned index, the result is a sorting algorithm with an expected complexity of O(n). The same strategy works even if the index structures provide approximations for rank queries under certain constraints. Learned indexes and learned sorts approximate the cumulative distribution function of the data using a machine learning algorithm. The expected O(n) sort beats the best known learned sort of complexity (nloglogn).

### Strengths
The paper is an interesting work in combinatorial algorithms and provides strong asymptotic results for learned sorting.

### Weaknesses
The paper has virtually nothing to do with representation learning or machine learning. The algorithm uses a learned index as a black box and 
nothing else uses learning. The results are theoretical with no clear practical advantages to the current best sorting algorithms.

### Questions
Why is this work related to representation learning? 

Do the errors in handling approximate rank queries accumulate through the recursive algorithm?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces Index2Sort, a general framework that derives sorting algorithms from static index data structures by treating them as opaque boxes. The framework automatically inherits index complexity guarantees: if an index has construction time O(nC(n)) and query time O(Q(n)), Index2Sort achieves O(nC(n) + nQ(n)). When instantiated with the ESPC learned index, it achieves O(n) expected time under distributional assumptions—claimed to be strictly better than existing O(n log log n) learned sorts—creating a future-proof paradigm where indexing advances automatically transfer to sorting.

### Strengths
**S1-** Novel theoretical framework: The conceptual inversion of using indexes for sorting is elegant and the opaque-box abstraction is intellectually appealing.

**S2-** Rigorous theoretical analysis: The proofs appear technically sound with careful treatment of expected complexity, worst-case bounds, and handling of approximate rank queries.

**S3-** Generality: The framework applies to classical indexes (B-trees) and learned indexes uniformly, with automatic guarantee transfer.

### Weaknesses
**W1 - Limited Theoretical Novelty and Overstated Claims**  
The O(n) result follows directly from plugging ESPC-index properties (C(n)=Q(n)=1) into the framework, essentially wrapping existing complexity without fundamental innovation. The "strictly tighter" claim is misleading since O(n) holds only under specific assumptions (Xρf or XC) that are not proven weaker than prior work's Xρ1,ρ2. No formal comparison demonstrates that these distributional assumptions are less restrictive. The "opaque box" abstraction is overstated—the algorithm requires a specific interface (construction on sorted arrays, rank queries), not truly opaque.  

**W2 - Unresolved Worst-Case Complexity with R(n) Dependence**  
The worst-case bound O(nC(n) + nQ(n) + R(n)) includes an explicit R(n) term for sorting range buckets. In standard settings this is O(n log n), but pathologically large range buckets remain unaddressed. Section 6 inadequately discusses when this dependence becomes problematic or provides algorithmic modifications to remove it. Distribution shift analysis assumes δ is known in advance with no practical guidance for determining it. The initial shuffle requirement assumes random permutability, limiting applicability to streaming or ordered data.  

**W3 - Poor Practical Performance Despite Theoretical Advances**
Authors acknowledge Index2Sort is "primarily theoretical" and **slower than IS4o and recent learned sorts** (BLS, ULS, Learned Sort 2.1), significantly undermining practical impact. Missing critical analyses include space complexity (index stores sorted u′ requiring O(n) space), cache complexity crucial for real performance, and guidance on when theoretical advantages materialize. No analysis of constant factors that may make O(n log log n) algorithms faster than O(n) for realistic n values.  

**W4 - Selective Experimental Presentation and Misleading Comparisons**  
Main paper shows only 4 datasets; full results show mixed performance. Some datasets have <3.2% unique values producing atypical behavior. Highlights Learned Sort 2.0's O(n²) failure but compares against older algorithm (2021) without guarantees—fairer comparison would be PCF Learned Sort (2024) with O(n log log n) guarantees and O(n log n) worst-case. No discussion of whether O(n log log n) algorithms have better constant factors or ablation studies on hyperparameters (α, τ, ε).  

**W5 - Incomplete Proofs and Incremental Technical Contributions**
Theorem C.1 proof deferred to "similar approach" without full details. Lemma A.1 adapts Frazer & McKellar (1970)—adding point buckets for duplicates is incremental. These incomplete proofs weaken confidence in theoretical claims.  

**W6 - Overclaiming and Insufficient Limitations Discussion**
Phrases like "strictly tighter bound," "future-proof paradigm," and "stronger guarantees" overstate conditional results. Results depend on specific distributional assumptions not proven weaker than prior work. Limitations section is brief and doesn't adequately address failure modes, the R(n) dependence problem, or when the "future-proof paradigm" would provide significant practical benefits. Notation inconsistencies (x′ usage, subscript conventions) reduce clarity.

**W7 - Old Leaned Index Methods**  The referenced methods PGM-Index and RMI are outdated, and many advanced learned index methods were not referenced, compared, or used (e.g., DILI, DobLIX).

### Questions
**Q1-** On distributional assumptions: Can you provide a formal comparison showing that Xρf or XC are strictly weaker (or at least comparable to) Xρ1,ρ2? What real-world distributions satisfy one but not the other?

**Q2-** On the R(n) term in worst-case: Can you provide tighter analysis or algorithmic modifications to remove the R(n) dependence? You mention this as future work, but it seems fundamental to the approach.

**Q3-** On practical performance: Given that Index2Sort is slower than existing algorithms in practice, what is the target use case? Is this purely theoretical, or do you envision scenarios where this would be deployed?

**Q4-** On constant factors: The O(n) bound hides constant factors. How do these compare to O(n log log n) algorithms for realistic n (say, 10⁶ to 10⁹)?

**Q5-** On the "opaque box" abstraction: Could you clarify what you mean by "opaque"? You still require a specific interface (rank queries on sorted data). How is this different from prior work that uses learned indexes for sorting?

**Q6-** On distribution shift: How would one choose δ in practice without knowing the data distribution in advance?

**Q7-** On the shuffle: Is the initial shuffle truly necessary, or can it be removed with stronger assumptions?

**Q8-** On space complexity: What is the space overhead compared to in-place algorithms like IS4o?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper designs a theoretical framework for analyzing sorting methods based on the performance of indexing methods. It generalizes various pervious results, and allows for applying other existing results in learned indexing to learned sorting.

### Strengths
- The framework itself is interesting, providing a plug-and-play framework for learned sorting that inherits theoretical properties of learned indexing

- The algorithm to perform sorting is also interesting, sorting and building an index on part of the data and applying the index to sort the rest

- There are some neat theoretical insights. Specifically,  that for fast sorting you only need a good static index that answers queries well from the same distribution as the data it was built on. The latter is a nice sorting specific insight that allows application of the results from Croquevielle et al.

### Weaknesses
- In general there is a trivial relationship  between sorting and indexing: if a *dynamic* index can be built on n elements in time C_n then we can also sort the data in O(C_n). From that perspective all results from dynamic indexing trivially  apply to sorting. This paper shows that this can be done from *static indexes* as well, and even if the index is only good at answering queries from the same distribution as the data. This is an interesting result, but the intro should specifically emphasize this. Without this context, the answer to the question "can results on indexing transfer to sorting?" is trivially yes, because building a dynamic index is equivalent to sorting  the data. 

- As the paper notes, the theoretical constructs with better asymptotic bounds do not necessarily perform better than practical methods with worse asymptotic bounds. Although one factor is optimized implementation (as the paper notes), the constant factors in time complexity also play a big role (difference between nlogn and nloglogn is very small and for many practical datasizes cnloglog n can be larger than nlog n depending on the constant c). It would be good to discuss non-asymptotic bounds and comparison---to know at what data size which method performs better and not asymptotically.

### Questions
NA

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
4

### Summary
This paper studies the sorting problem with static indexes. Their algorithm, Index2Sort, requires a static index built on a sorted array A and supports rank queries on A.

Let the construction time and the per-query time of the static index be $O(n C(n))$ and $Q(n)$, respectively. Index2Sort can sort an array of n elements in $O(n C(n) + n Q(n))$ time. If the index answers approximate rank queries with error $\varepsilon$, the running time becomes $O(n C(n) + n Q(n) + n \log(\varepsilon+1))$. Combining these bounds with an index for which $C(n)=O(1)$ and $Q(n)=O(1)$, they obtain a linear-time sorting algorithm under certain input assumptions.

### Strengths
1. The work provides theoretical time bounds for distributions with shifts, approximate rank queries, and worst-case inputs. These results show that Index2Sort is not restricted to ideal cases and is applicable in various practical scenarios.

2. The algorithm is simple and efficient; advances in learned indexes may further improve its performance.

### Weaknesses
1. The Index2Sort algorithm and its analysis resemble existing CDF-based learned sorts. A notable difference is that Index2Sort needs a recursive phase to grow a sufficiently large sorted sample for bucketing. Consequently, the contribution is more like an application of state-of-the-art learned indexes than a technical novelty.

2. Although the experiments demonstrate potential—performance is comparable to several existing algorithms—the implementation is not fully optimized for cache efficiency or other low-level factors, and it remains slower than the state-of-the-art sorting implementations.

### Questions
The experimental results for Index2Sort with the RMI-index appear incomplete on some datasets (e.g., Stocks [Volume]). 

Nearly half of several figures are almost empty, while the remaining half contain 14 lines, making them hard to recognize.

### Soundness
3

### Presentation
2

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
The paper proposes Index2Sort, a general framework that derives sorting algorithms from static indexes by treating the index as an opaque box supporting only two operations: index construction and rank queries. The core idea is that any index capable of rank queries can be reused for sorting, and its computational guarantees automatically transfer to the sorting process. Specifically, if the index can be constructed in time O(nC(n)) and each rank query can be answered in time O(Q(n)), the resulting sorting algorithm runs in O(nC(n) + nQ(n)) time. Using a state-of-the-art learned index under certain distributional assumptions, the overall complexity can be as fast as O(n). The authors perform experiments to analyze the running time of their framework on both synthetic and real data, including cases with approximate query-answer oracles.

### Strengths
1. The proposed algorithm and framework are general and can be applied to many existing learned indexes, as well as future ones.
2. The authors conduct experiments on both real and synthetic datasets, and the results align well with the theoretical complexity analysis.

### Weaknesses
1. The algorithm and its analysis are relatively simple and conceptually close to the standard divide-and-conquer framework, which makes the novelty somewhat limited.
2. The theoretical guarantees for learned indexes depend heavily on distributional assumptions, which may limit their applicability in practice.

### Questions
n/a

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 6

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a sorting algorithm that extends the bucket sort. The core idea is that, given an array, it is first randomly permuted, followed by a portion of it being sorted, then a learned index is created using the sorted subset, and finally, the unsorted portion is bucket sorted using the learned index. Finally, merging both sorted parts. In essence, instead of performing a bucket sort throughout, initially, a learned index is created over a subset to utilize it for expected linear-time sorting of the unsorted part.

### Strengths
This is a simple yet interesting idea to employ the learned indices in the hierarchy of the bucket sort.

Additionally, it is attractive to present a framework that can utilize any learned index for such a sorting algorithm.

### Weaknesses
This work, though an application of shallow machine learning, does not offer any new insight. Even from an algorithm design perspective, the contributions are only marginal. For the core ML community, such as ICLR, the submission is of little interest. 

For someone from the core DBMS community, a natural question to ask would be, if you need only a static index, what prevents you from using HistTree [1], a non-learned index in place of a B-tree, and then critically analyze the performance. Even for a learned index, there are better candidates than the PGM-index, such as LIPP/ALEX (yes, ALEX offers dynamic updates; still, it outperforms the PGM-index for static updates), see [2]. Then, why not use them in your experiments?

The claimed algorithmic innovations in the submission may be of greater interest to communities such as ESA and ICALP. For innovations from a DBMS perspective, ICDT, CIDR, and similar conferences may find more alignment.

[1] Crotty, Andrew. "Hist-Tree: Those Who Ignore It Are Doomed to Learn." CIDR. 2021.

[2] Sun, Zhaoyan, Xuanhe Zhou, and Guoliang Li. "Learned index: A comprehensive experimental evaluation." Proceedings of the VLDB Endowment 16.8 (2023): 1992-2004.

### Questions
See the weaknesses. Address those points.

### Soundness
2

### Presentation
2

### Contribution
1
