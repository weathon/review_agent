# Isolation-based Spherical Ensemble Representations for Anomaly Detection

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Tabular Anomaly detection is a critical task in data mining and management with applications spanning fraud detection, network security, and log monitoring. Despite extensive research, existing unsupervised anomaly detection methods still face fundamental challenges including conflicting distributional assumptions, computational inefficiency, and difficulty handling different anomaly types. To address these problems, we propose ISER (Isolation-based Spherical Ensemble Representations) that extends existing isolation-based methods by using hypersphere radii as a monotonic transformation of local density characteristics while maintaining linear time and constant space complexity. ISER constructs ensemble representations where hypersphere radii encode local sparsity through a monotonic transformation of density: smaller radii correspond to dense regions while larger radii correspond to sparse areas. We introduce a novel similarity-based scoring method that measures pattern consistency by comparing ensemble representations against a theoretical anomaly reference pattern. Additionally, we enhance the performance of Isolation Forest by using ISER and adapting the scoring function to address axis-parallel bias and local anomaly detection limitations. Comprehensive experiments on 22 real-world datasets demonstrate ISER's superior performance over 11 baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ISER (Isolation-based Spherical Ensemble Representations), a new unsupervised anomaly detection method designed to overcome the limitations of existing approaches such as conflicting distribution assumptions, inefficiency, and poor handling of diverse anomaly types. ISER extends isolation-based methods by introducing hypersphere radii as proxies for local density, maintaining linear time and constant space complexity. A similarity-based scoring function compares ensemble representations with a theoretical anomaly reference pattern, enabling more consistent anomaly identification. The method also enhances the Isolation Forest framework to mitigate axis-parallel bias and improve local anomaly detection. Experiments on 22 real-world datasets show that ISER achieves superior performance over 11 baselines, demonstrating both efficiency and robustness across multiple domains.

### Strengths
1. Extensive experiments were conducted to analyze ISER across a wide range of datasets, with comprehensive comparisons against numerous baseline methods.
2. The proposed method demonstrates high computational efficiency and good scalability.

### Weaknesses
1. The motivation behind the proposed method is not clearly articulated. Although the paper introduces minor modifications to iNNE and IDK to address their respective challenges, the rationale for combining the two approaches remains unclear.
2. The method design relies heavily on empirical estimations and lacks solid theoretical justification for its key assumptions.
3. The paper’s overall exposition lacks coherence and contains several obvious errors and imprecise statements, giving the impression of being hastily written.

### Questions
1. The key assumption of the proposed method that the radius of a hypersphere can serve as a proxy for the local density of a sample point is questionable. Intuitively, local density should be reflected by the ratio of the number of sample points inside the hypersphere to its volume. Since the paper directly uses the hypersphere radius as a proxy for density, a stronger theoretical justification is necessary.
2. In Equation (4), the choice of the function $\frac{1}{r[\hat{z}_i(x)]}$ lacks explanation. Could other functions that are monotonically decreasing with respect to $r[\hat{z}_i(x)]$ and take values within $[0,1]$ be used instead? Would such alternatives affect the method’s performance, and should a comparative analysis be conducted?
3. The proposed method uses similarity or mean-based strategies to compute anomaly scores. Although this reduces computational cost, both strategies are based on empirical observations rather than rigorous theoretical guarantees like those provided in IDK [2].
4. In Figure 1, the notation $Di$ is clearly incorrect, as the index $i$ is not shown as a subscript. Moreover, the bold points in the figure, which seem to represent sampled points, are not explicitly explained, and there is no legend provided. Is it necessary to make these additions?
5. In Table 5, the %Ano value for the “fault” dataset is clearly wrong, which raises concerns about the reliability of the experimental code. Additionally, in the authors’ anonymous repository, the “Avg.” row in the README’s version of Table 5 is inconsistent with the version presented in the paper.
6. The abstract mentions the “difficulty in handling different anomaly types,” yet the introduction contains no discussion of this issue. This inconsistency should be addressed.
7. The introduction does not make the logic behind combining the two methods clear. It only shows small modifications made to iNNE [1] and IDK [2] to overcome their respective challenges, but why should these two methods be combined at all?

[1] Bandaragoda, Tharindu R., et al. "Isolation‐based anomaly detection using nearest‐neighbor ensembles." _Computational Intelligence_ 34.4 (2018): 968-998.
[2] Ting, Kai Ming, et al. "Isolation distributional kernel: A new tool for kernel based anomaly detection." _Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining_. 2020.

### Soundness
3

### Presentation
2

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
The paper introduces isolation-based spherical ensemble representations which extends isolation-based anomaly detection through hypersphere-based partitioning and ensemble scoring. The approach aims to approximate local density using hypersphere partioning while. The authors report performance improvements over several baselines on synthetic and real-world datasets. The work is clearly written and demonstrates the proposed approaches in experimental evaluation. However, several conceptual and analytical aspects remain underdeveloped, particularly regarding theoretical grounding and the interpretation of experimental results.

### Strengths
- The paper provides an accessible overview of existing isolation-based methods and motivates the need for improving density awareness.
- ISER maintains the desirable linear-time and constant-space properties of traditional isolation methods. The scalability analysis is presented and demonstrates practical efficiency.
- Code for reproducing the results is provided.

### Weaknesses
- Incremental novelty. Hypersphere partitioning-based isolation approach was used in iNNE and IDK.
- Lack of formal theoretical grounding. The definitions of global, local, and dependency anomalies rely on qualitative descriptions such as "deviate significantly" or "sparse regions," without mathematical/statistical formulations. Consequently, it is unclear how the proposed mechanism theoretically distinguishes among these anomaly types. The role of hypersphere radii as density proxies is intuitive but not formally justified.
- Incremental perofrmance improvement. Figure 2 aggregates results from heterogeneous datasets with different baseline performances, making it challenging to interpret relative improvement. Figure 3 shows that ISER is statistically comparable to several existing methods, suggesting only marginal differences. While visually clear, these figures provide limited insight into mechanism or consistency.
- Moreover, the authors provide limited analysis of why the method performs better or under what data conditions it is expected to succeed.
- Although ISER has linear-time complexity wrt $n$, this advantage is shared with existing methods. The reported runtime gains are minor and largely reflect constant-factor efficiency improvements rather than new theoretical insights into computational scalability.

### Questions
- The proposed method does not seem to involve representation learning in the usual sense but rather constructs a fixed mapping from random hypersphere partitions but more like a randomized geometric feature transformation. Since the mapping $\Phi$ is not learned, I wonder if this fits within unsupervised representation learning. Please correct me if I’m mistaken.
- The authors do not provide any results/analyses about how the hypersphere partitioning behaves in high-dimensional settings, where nearest neighbor become less informative. It is unclear whether the proposed method remains meaningful as the dimension increases.
- It is unclear how the hyperparameters were selected. If I understood correctly, this work focuses on the unsupervised anomaly detection and there's no (known) anomalies in the validation set.

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
3

### Summary
This work studies on the problem of anomaly detection and proposes a new method based on isolation-driven spherical ensemble representations.

While the topic is important and the paper presents a method that is conceptually reasonable, I have significant concerns on the motivation and the experimental evaluation. As a result, I assign a score of 4. 

Notably, I am not familiar with this research area.  While I have spend considerable time on reviewing this work, my comments may lack field-specific depth.

### Strengths
1.	This work studies on an important problem.
2.	The proposed method appears to be reasonable.
3.	Experiments on 22 datasets have been conducted to verify the efficacy of the proposed method.

### Weaknesses
1.	My major concern lies on the unclear motivation. Some important questions have not been addressed, like: 1) deep-learning-based anomaly detection methods have been widely explored by recent work, and have demonstrated high effectiveness. The paper’s stated drawbacks --- such as longer training time and reduced interpretability --- are not convincingly shown to be critical limitations in practice.  The authors should discuss the actual severity of these issues and their impact on real-world deployments. 2) The introduction of hypersphere radii is interesting, but the underlying intuition and its specific benefits for the proposed method remain unclear. More discussions should be provided.

2.	The proposed method is largely heuristic. Note that the proposed method is relatively traditional. Without a solid theoretical justification and analysis, the technical contributions appear limited.


3.	There are also some significant limitations on the experiments: 1) The baselines are relatively old (all before 2023). More state-of-the-art methods (e.g., [a1]) should be included.  2) From the table 7, the improvements of the proposed method over the best baseline are modest, and the proposed method does not achieve state-of-the-art performance on many datasets. This raises questions about the practical significance of the approach.

4.	Some concerns on the presentation: 1) Many important contents (e.g., the experimental setup and comparable results) are presented in the appendix, making them less accessible to readers. 2) Tables 6 and 7 are formatted horizontally, which heavily affects readability. 

[a1] cvpr’25: Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
2

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
This paper introduces ISER, a novel unsupervised anomaly detection algorithm. The method builds upon existing hypersphere-based partitioning techniques (like those in iNNE and IDK) but introduces a new way to encode and score data points. The core of ISER is to construct an ensemble representation for each data point. This is done by repeatedly sampling subsets of data, defining hyperspheres based on nearest-neighbor distances within these subsets, and then encoding based on its nearest hypersphere. The paper proposes two scoring mechanisms for this representation, ISER-A and ISER-S. Furthermore, the paper proposes ISER-IF, which uses the ISER ensemble representation as a transformed feature space for the standard iForest. The authors correctly identify that this transformation inverts iForest's core assumption and propose a modified scoring function to address iForest's axis-parallel bias.

### Strengths
1. The paper provides a clear theoretical justification for why this absolute density proxy is more robust than iNNE's relative (ratio-based) score and IDK's point-count-based embedding. This new representation successfully avoids the failure modes of its predecessors.
2. The paper introduces two scoring methods. The similarity-based ISER-S is particularly innovative.
3. The ISER-IF contribution is also a major strength. 
4. The paper is backed by a large-scale, comprehensive empirical study. The use of 22 real-world datasets + 3 synthetic datasets against 11 baselines provides strong evidence.

### Weaknesses
1. The paper claims that it follows the same hypersphere partitioning as iNNE/IDK, and then contrasts scoring qualitatively without providing formal conditions where ISER is better.
2. Assigning a flat score of 1 to all points outside their nearest hypersphere (Eq. 4) is a questionable design, as this binary assignment discards information about the degree of isolation. The authors should provide a justification for this.
3. The paper claims that normal data and anomalies cluster in $\Psi$-space, leading to longer anomaly path lengths and a flipped scoring rule. This is plausible but needs more clarifications. Please supply theory or empirical evidence showing when this ordering holds.
4. The paper states that points in overlapping regions are assigned only to the nearest hypersphere center. Is this strategy still valid when a point lie at the junction of a "sparse" large sphere and a "dense" small sphere. 
5. The core design choices (e.g., the $1 - 1/r$ form for $\phi_i(x)$) are justified by intuition rather than formal arguments. The authors should provide theoretical analysis to support their method.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
