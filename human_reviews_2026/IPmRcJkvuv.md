# Evolutionary Architecture Search Through Grammar-based Sequence Alignment

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Neural architecture search (NAS) in expressive search spaces is a computationally hard problem, but it also holds the potential to automatically discover completely novel and performant architectures. To achieve this we need effective search algorithms that can identify powerful components and reuse them in new candidate architectures. In this paper, we introduce two adapted variants of the Smith-Waterman algorithm for local sequence alignment and use them to compute the edit distance in a grammar-based evolutionary architecture search. These algorithms enable us to efficiently calculate a distance metric for neural architectures and to generate a set of hybrid offspring from two parent models. This facilitates the deployment of crossover-based search heuristics, allows us to perform a thorough analysis on the architectural loss landscape, and track population diversity during search. We highlight how our method vastly improves computational complexity over previous work and enables us to efficiently compute shortest paths between architectures. When instantiating the crossover in evolutionary searches, we achieve competitive results, outperforming competing methods. Future work can build upon this new tool, discovering novel components that can be used more broadly across neural architecture design, and broadening its applications beyond NAS.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, authors introduce constrained and recursive Smith-Waterman style sequence alignment algorithms, CSWX and RCSWX, as efficient, grammar-based tools for NAS applications in expressive grammar-defined spaces. The proposed methods serve as both architecture distance metrics and as crossover operators for evolutionary-population-based search. The authors claim to vastly reduce the computational complexity of crossover and distance computation compared to prior methods, and retaining or even improving search efficiency. Applications shown include efficient diversity measurement, loss landscape analysis, and practical evolutionary search in large grammar-based NAS benchmarks, with empirical results supporting the methods’ computational scalability and competitive search performance.

### Strengths
1. The adaption of constrained Smith-Waterman alignment to grammar-based NAS is both novel and well-motivated. The recursive variant to handle permutation invariance demonstrates sophisticated algorithmic thinking.

2. The formal analysis on the metric properties, runtime complexity, and permutation invariance, provide relatively solid theoretical foundations. 

3. The experiments span multiple dimensions, e.g., search performance, scalability, landscape analysis, offer clear and quantitative head-to-head comparisons across five datasets and several search strategies, e.g., STX, SEPX and no crossover.

### Weaknesses
1. While the authors acknowledge their focus is on introducing tools rather than achieving SoTA performance, the experiments use only 1000 architecture evaluations and relatively small datasets, e.g., CIFAR-10. Besides, there is no direct comparison with the most recent SoTA NAS approaches beyond the baselines. These weakens the empirical claim of broader impact and applicability.

2. As shown in Table 1, the proposed RCSWX shows less robust results compared to CSWX, specifically, with lower mean accuracy and higher variance on some benchmarks. The authors’ explanation of the difference between 'perfect interpolation' and 'noise injection' is plausible but lacks rigorous analysis. Additionally, only three random seeds are used, which is limited for high-variance evolutionary methods, and statistical significance is not analysed.

3. Another critical weakness is the absent of ablation studies for critical scoring function choices, e.g., the cost matrix or arbitrary constants in SubstitutionCost. I wonder how sensitive are performance and search landscape properties to these choices? Additionally, the selection function and skewness hyper-parameter’s  impact is not explored. 

4. While the proposed methods are well-motivated for grammar-based spaces, they are only shown for the 'einspace' family and similar grammars. Potential obstacles for spaces with more irregular grammar rules, complex parameter sharing, or multi-input/-output components are not discussed. Authors claim 'any search space that can be represented as a sequential set of tokens', but there is not critically assessed. These make the extensibility and generalisation of proposed methods more like an assertion.

### Questions
1. What’s the impact of different choices for the cost matrix in SubstitutionCost?

2. Given the high variance in the evolutionary search, why are only three seeds used? Have you conduct significant testing? 

3. Can the proposed methods be readily adapted for use in non-evolutionary NAS?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces two algorithms (CSWX and RCSWX) based on the Smith-Waterman sequence alignment method, to enable efficient evolutionary search in expressive, grammar-based Neural Architecture Search (NAS) spaces. These methods provide a computationally tractable way to calculate an edit distance between neural architectures. This distance is then leveraged not only as a crossover operator to create hybrid offspring, but also as a formal distance metric to analyze the architectural loss landscape and population diversity. The authors demonstrate experimentally that their approach is orders of magnitude faster than prior graph-edit-distance methods and achieves competitive performance in evolutionary searches.

### Strengths
+ The paper clearly shows that prior methods based on Graph Edit Distance (e.g., SEPX) are NP-hard and become intractable for even moderately sized graphs. The proposed (R)CSWX methods, by contrast, are highly efficient, effectively scaling to large architectures. This is a significant practical contribution.

+ The proposed method is valuable as both a search operator and an analysis tool. Using $d_{RCSWX}$ to perform a large-scale analysis of the architectural loss landscape may be a compelling application.

+ The paper is well-written and clearly structured.

### Weaknesses
+ As is well known, the primary computational cost of NAS is the performance estimation (i.e., architecture evaluation), not the search strategy. Therefore, the main contribution of this paper, which enhances the efficiency of the search strategy, seems less significant in the context of the overall NAS pipeline.

+ In Table 1, RCSWX shows the lowest average performance, even underperforming the "No Crossover" baseline. Its result on AddNIST, in particular, is very poor. This undermines the central motivation for developing the more complex, permutation-invariant operator.

+ It would have been highly valuable to include a comparison on a smaller benchmark where SEPX is tractable (e.g., NAS-Bench-101). This would have established whether (R)CSWX, which approximates the edit path, produces offspring of comparable or superior quality to the true shortest edit path crossover.

+ The "skewness" parameter (Algorithm 1) appears to be important for guiding the sampling of operations. The paper mentions it can be set based on parent performance, but it is not discussed in the experimental setup. The sensitivity of the search to this parameter is unknown.

### Questions
+ Given the poor search performance of RCSWX compared to the simpler CSWX, could the author elaborate on the hypothesis that "perfect interpolation" is detrimental?

+ How was the "skewness" parameter for operation sampling set during the search performance experiments in Section 5.1? How sensitive is the performance of CSWX and RCSWX to this hyperparameter?

### Soundness
2

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
4

### Summary
This paper introduces (R)CSWX, an algorithm for computing distances and performing crossover in grammar-based NAS search spaces, based on the Smith-Waterman algorithm. The proposed algorithms scales well to large and complex spaces, and CSWX performs well compared to the baseline (no crossover) but RCSWX offers no empirical improvement on average.

### Strengths
1. (R)CSWX is an interesting algorithm, building on the Smith-Waterman algorithm, for NAS.
2. Runtime analysis presents good upper bound computational complexity.

### Weaknesses
W1. RCSWX which is the more tractable and practical algorithm underperforms the baselines. 
W2. The authors offer insufficient analysis or explanation for W1.
W3. The authors state that the focus of this work is to "introduce a theoretically sound, computationally efficient crossover operator for grammar-based NAS, intended as a tool for further research rather than as a benchmark for state-of-the-art performance." but this is not a sufficient reason to not compare CSWX and RCSWX with other optimisation strategies .
W4. In light of W3, the experiments were also lacking, with no comparison to similar NAS algorithms.

### Questions
1. How does Figure 2 demonstrate good exploration/exploration trade-off?
2. How does CSWX empirically scale further than 200 nodes?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces two variants of the Smith-Waterman (i.e., Constrained Smith–Waterman Crossover - CSWX, and recursive CSWX) to compute the edit distance in grammar-based evolutionary architecture search, and perform crossover for the selected parent. CSWX first converts each parent architecture into a simplified sequence, then computes the minimum-cost alignment path between them, and finally generates an offspring along that path. Experiments are performed on Unseen NAS benchmark, demonstrating the effectiveness of CSWX and RCSWX.

### Strengths
- The proposed variants seem reasonable.
- As shown in Table 1, crossover shows better performance than mutation only.

### Weaknesses
- As stated in the conclusion, CSWX sometimes outperforms RCSWX,  which requires a deep investigation.
- In Figure 2, no crossover shows better validation performance on Chesseract and Isabella, but achieves lower test performance on test set as shown in Table 1. It is unclear why this happens.
- The performance gain is not consistent across datasets, which raises concerns about the stability of the method when applied to different datasets. Sometimes CSWX does not better than STX (see Chesseract and MulTNIST in Table 1).

### Questions
My main concern about the paper is the experimental results. The performance gain is not significant (compared to STX) and is sometimes not stable across datasets.

### Soundness
3

### Presentation
2

### Contribution
2
