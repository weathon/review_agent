# TailorPiece: Tailoring Linear Models for Joint Representation

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
The need to represent a long data series using a sequence of line segments abiding by a maximum error threshold arises in various domains. This problem, known as Piecewise Linear Approximation (PLA), has a long history and has recently gained attention with the rise of applications dealing with time-stamped data. State-of-the-art PLA methods achieve space savings over lossless compression techniques with tolerable precision loss by quantizing starting points and representing similar line segments jointly. However, these methods do not tailor line segments for their eventual joint representation and do not minimize the number of segments either. In this paper, we present TailorPiece, a suite of algorithms for lossy PLA-based compression that explicitly tailor linear segments for both small sequence length and joint representation under a given error threshold and starting-value quantization. Our first algorithm, TailorPieceDP, optimizes a mergeability criterion of PLA segment descriptions; in a degenerate form, it reduces to an algorithm that represents the data series by the minimum number of PLA segments. Our second algorithm, TailorPieceGD, greedily selects the endpoint of each PLA segment within a tunable search space that allows the subsequent segment to extend farther, thereby balancing compression and runtime. Through experimentation, we show that TailorPieceDP achieves improvements of up to 34% over prior art in compression ratio and TailorPieceGD gains similar savings with a runtime reduced by two orders of magnitude.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper addresses the problem of lossy time-series compression under a fixed error bound with quantized starting values. It identifies limitations in the previous MIXPIECE method, which greedily maximizes segment length but may yield suboptimal segment counts and poor mergeability. They propose three variants: MINSEGMENTS, a dynamic programming approach for globally minimal segmentation under quantization; TAILORPIECEDP, which further optimizes for mergeability by maximizing slope interval width; and TAILORPIECEGD, a greedy lookahead version balancing compression quality and runtime.

### Strengths
- The paper clearly articulates the practical need to improve MIXPIECE’s segmentation under quantization and mergeability constraints.

- The DP formulations are logically consistent and correctly defined for global optimality. The overall structure and explanations are straightforward, making the method easy to reproduce and understand.

- The approach provides a tunable trade-off between accuracy and efficiency, which is valuable for real-world compression systems.

### Weaknesses
- While technically correct, the paper’s contributions are incremental extensions of existing piecewise linear approximation (PLA) frameworks rather than fundamentally new ideas, which limits its novelty. No new theoretical model, loss formulation, or probabilistic insight into PLA is introduced. The paper essentially re-optimizes an existing heuristic with better parameterization. From a research standpoint, this positions the contribution as an engineering refinement, not a methodological breakthrough.

- The paper offers no formal complexity analysis, approximation guarantees, or theoretical characterization of how the new objectives affect global optimality. For example, while MINSEGMENTS claims “globally minimal” segmentation, the proof is implicit. There is no formal definition of optimality under quantized constraints or derivation of time/space complexity. Similarly, for TAILORPIECEDP, the trade-off between segment count and interval width is handled empirically but never quantified analytically.

- The evaluation focuses on synthetic or benchmark datasets (UCR) with standard metrics and lacks deeper diagnostic analysis.

- No discussion on integration into full compression or streaming pipelines.

### Questions
- How does the algorithm scale with very long or streaming sequences—can the DP version handle data in the order of millions of points?

- Can the authors provide theoretical or empirical bounds linking segment count and slope interval width?

- How sensitive is performance to quantization granularity or dataset smoothness?

- Could the approach extend to multidimensional or irregularly sampled time series?

### Soundness
2

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
4

### Summary
The paper proposes a library of algorithms for lossy-PLA compression of time-stamped data. The aim of the work is to reduce the storage requirement, and thereby, the cost, by achieving better compression rates through improving the joint representation of similar data. The algorithms were tested against 8 baselines using 41 datasets. The experiments demonstrated improvements in compression rate in addition to runtime improvements for some algorithms.

### Strengths
- The paper attempts to address an important research area regarding storing big time-stamped data with the lowest possible storage cost, which is applicable to many sectors.
- The experiments were conducted against 8 baselines.
- The results show improvements compared with the baselines.
- The algorithms presented in the paper are well-explained both in text and mathematically.

### Weaknesses
- Very brief presentation and discussion of related work. The paper lacks a proper “related work” section, where the work from the literature is typically presented, discussed, and research gaps are listed. Without such section, it is not clear where the contribution of this paper stands in relation to the related literature, and whether it addresses an actual gap or not. There are scattered information about related methods across the paper; however, they do not properly replace a proper “related work” section.
- The contribution of the paper is not clear. No specific research questions, clear setup of experiments, nor experiments goals are clearly presented.
- When comparing the algorithms presented in the paper with the baselines, the results are presented in numerical forms without testing whether the improvements are statistically significant or not. The statistical significance of the claimed improvements in comparison with the baseline methods needs to be tested (using a Friedman test followed by Nemenyi post-hoc test, for example).
Relatively weak benchmark. The experiments were run on a subset of datasets from the UCR Archives (41 out of 128 datasets). The authors mention that only datasets that do not contain undefined values were used. This is an issue for two reasons: (i) this is not always the case for real-world data, especially the sectors mentioned by the authors in the introduction (line 033); it is better to test the algorithms using all the datasets in the UCR Archive by passing the data as it is or preprocessing the data to handle the undefined values, or both. (ii) - The authors of the UCR Archive explicitly discourage against cherry-picking datasets from the archive [1].
- The implications of lossy compression are not properly discussed. The paper proposed lossy compression as the solution for storing time-stamped data in a more efficient way compared with lossless compression. However, the implications of such choice are not properly discussed. For example, how accurate the original data can be reconstructed from the compressed representation and how severe do the lost information affect the usability of the data in any downstream tasks. In this end, it is important to be able to use the data after storing it.
- Wording issues in the text that lead to ambiguity:
Lines 061–062: “TAILORPIECEDP, which, building on top of TAILORPIECEDP,”

- Note: I understand that some sections might have been omitted due to the page limit. However, the authors could have made a better use of the appendix regarding the distribution of the content instead of omitting crucial details.

[1] Dau HA, Bagnall A, Kamgar K, Yeh CC, Zhu Y, Gharghabi S, Ratanamahatana CA, Keogh E. The UCR time series archive. IEEE/CAA Journal of Automatica Sinica. 2019 Nov 8;6(6):1293-305

### Questions
- Why is the complexity of the time series data sample not considered when attempting to extract compressed representations? It might be beneficial to consider the complexity of the data when calculating the minimum-length PLA in Algorithms 3.2, for example (See, e.g. [2])
- In all the algorithms presented in the paper, there is an error threshold (ε) as an input. However, it is not clear from the paper nor the appendix how this error threshold is calculated in the experiments and how can any future users of the algorithms calculate the error threshold. There is one brief explanation about this in lines 315–317, but it is still not clear how the values of the range are selected. Can you please elaborate on this? (the authors are encouraged to add a section about this in the paper or the appendix in any future versions).

[2] Nagaraj, N., Balasubramanian, K. & Dey, S. A new complexity measure for time series analysis and classification. Eur. Phys. J. Spec. Top. 222, 847–860 (2013). https://doi.org/10.1140/epjst/e2013-01888-9

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes TAILORPIECE, a method that performs piecewise linear approximation (PLA) under a maximum error constraint, while balancing the minimal number of segments and segment mergeability. By combining dynamic programming and greedy strategies, the approach effectively improves the compression ratio while maintaining approximation accuracy.

### Strengths
S1. The framework explicitly optimizes for segment mergeability, allowing similar line segments to be grouped and jointly represented, thereby improving compression efficiency.

S2. Two hyperparameters (p and q) are introduced to flexibly balance compression accuracy and runtime, enabling users to tune the method 
according to application needs.|

S3. TailorPiece advances beyond the previous state-of-the-art O(nlogn) complexity by introducing algorithms with O(Rn).

### Weaknesses
W1. The TAILORPIECEGD algorithm is a heuristic greedy approach that does not guarantee global optimality. Moreover, its performance depends on two hyperparameters p and q, which require manual tuning to achieve the desired trade-off between compression accuracy and runtime.

W2. TailorPiece quantizes the segment starting value $v$ into discrete levels defined by the error bound $\varepsilon$, using
$$
b^- = \lfloor v / \varepsilon \rfloor \times \varepsilon, \quad 
b^+ = \lceil v / \varepsilon \rceil \times \varepsilon.
$$
when $v$ is an exact multiple of $\varepsilon$, the lower and upper bounds collapse ($b^- = b^+$), leaving no feasible interval for numerical tolerance.

W3. The baseline HIRE (Barbarioli et al., 2023) explicitly defines its reconstruction constraint in terms of L-infinity rather than L2. Therefore, classifying HIRE under L2-based methods could be misleading.

### Questions
Q1. The compression ratios of some datasets in Table 2 (such as Rock and CinCECGTorso) are very high. Is this related to the data characteristics?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper describes a suite of techniques for the Piecewise linear approximation. 
The techniques are well described with definitions and problems.
The performances are compared with MIXpiece algorithm for the same task. Multiple experiments on compression and approximation are presented.

The proposed techniques are a slight variation on similar optimization
For example, in Figure 11, the TailorPieceGD shows a very similar compression ratio of TailorPieceDP with a reduced compression time. The best compression of TailorPieceDP is better but the improvement is less than 10%.
The comparisons with other techniques than MiXPiece are limited. Also other techniques base on euristics could be used for the same problem.

### Strengths
The approximation with sequence of values with Piecewise linear approximation achieves good results and the proposed techiques are better than MIxPiece technique

### Weaknesses
The comparison with other techniques also from other optimization paradigms is limited

### Questions
Can the algorithm of the proposed suite be integrated into a single technique that is optimized adaptively according to the approximation error?

### Soundness
3

### Presentation
3

### Contribution
3
