# Dual Alignment for Covariate Shift: A Principled Framework for Graph Domain Adaptation

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
\textit{Graph Domain Adaptation} (\textit{GDA}) is fundamentally challenged by \textit{Covariate Shift} (\textit{CS}), a pervasive discrepancy between source and target graph distributions. We decompose CS into two complementary components: \textit{Feature Shift} (\textit{FS}), arising from mismatched node feature distributions, and \textit{Feature-Conditional Structure Shift} (\textit{FCSS}), reflecting structural variations conditioned on features. Both FS and FCSS distort \textit{Graph Neural Network} (\textit{GNN}) representations, thereby hindering reliable cross-domain transfer. To overcome these issues, we propose \textit{Dual Alignment for Covariate Shift} (\textit{DACS}), a framework that jointly addresses FS and FCSS through adversarial feature alignment for domain-invariant embeddings and adaptive reweighting to enforce structural consistency. Extensive experiments on benchmark datasets demonstrate that DACS effectively bridges domain gaps and consistently outperforms state-of-the-art baselines, highlighting its strong cross-domain generalization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the challenging Covariate Shift (CS), a pervasive discrepancy between source and target graph distributions. The authors decompose CS into two complementary components: Feature Shift (FS), arising from mismatched node feature distributions, and Feature-Conditional Structure Shift (FCSS), reflecting structural variations conditioned on features. Both FS and FCSS distort Graph Neural Network (GNN) representations, thereby hindering reliable cross-domain transfer. To overcome these issues, the authors propose Dual Alignment for Covariate Shift (DACS), jointly addressing FS and FCSS through adversarial feature alignment for domain-invariant embeddings and adaptive reweighting to enforce structural consistency.

### Strengths
1. The authors find a pervasive discrepancy between source and target graph distributions, including Feature Shift (FS) and Feature-Conditional Structure Shift (FCSS).
2. The authors propose Dual Alignment for Covariate Shift (DACS), jointly addressing FS and FCSS through adversarial feature alignment for domain-invariant embeddings and adaptive reweighting to enforce structural consistency.

### Weaknesses
1. The experiment is insufficient, and unable to verify the validity of the method. 
2. Datasets in the paper are not commonly used in graph domain adaptation, Citation networks generally test on ACMv9, DBLPv7, and Citationv1, and Airport network lacks attributes, the authors address Feature Shift (FS), arising from mismatched node feature distributions, and Feature-Conditional Structure Shift (FCSS) on Airport is of little significance. 
3. The proposed method include feature alignment, representation reweighting, and conditional alignment of reweighted source and target representations. Ablation study lacks verification for feature alignment. 
4. The paper lacks parameter analysis.

### Questions
Please refer to weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of Graph Domain Adaptation under Covariate Shift. The authors propose a principled decomposition of CS into two components: (1) Feature Shift mismatch in the marginal node feature distributions, and (2) Feature-Conditional Structure Shift, a mismatch in the conditional probability of the graph structure given the node features (i.e., $P_S(A|X) \neq P_T(A|X)$). Based on this decomposition, the paper derives a theoretical upper bound on the target risk (Theorem 3.1) which explicitly separates the risk into terms corresponding to source risk, FS, and FCSS. To mitigate these shifts, the paper introduces DACS (Dual Alignment for Covariate Shift), a framework with three main components:

1. Adversarial Feature Alignment: A standard DANN-style adversarial loss on an MLP encoder's output ($H^1$) to align the marginal feature distributions and reduce the FS term. 
2. Adaptive Reweighting for FCSS: A novel, layer-wise reweighting strategy for the source graph's message passing. This reweighting (Theorem 4.1) aims to align the conditional expectations of aggregated representations by applying a weight $w^l \approx \frac{P_T(A|H^l)}{P_S(A|H^l)}$ at each GNN layer. This ratio is estimated using a distance-binning scheme on node-pair representations.

### Strengths
1. The method is well-supported by theory. Theorem 3.1 provides a clear upper bound on the target risk that directly motivates the model's architecture, with Term (II) mapping to the FS-alignment module and Term (III) mapping to the FCSS-alignment module.

2. The experimental results are comprehensive and convincing. DACS shows state-of-the-art performance across all datasets. The synthetic experiments (Table 1) are particularly strong, as they effectively demonstrate the model's robustness to different, isolated types of shift.

3. The ablation analysis (Table 5, Figure 6) clearly demonstrates that both the FS and FCSS components are crucial and complementary. The full DACS model significantly outperforms variants that only tackle one of the two shifts.

### Weaknesses
1. The FCSS reweighting module (Sec 5.2) appears to be computationally expensive. It requires estimating conditional probabilities $P(A_{uv} | ...)$ using a "distance binning scheme." This seems to require computing $O(N^2)$ pairwise distances $\{d(H_{w,u}^{l},H_{w,v}^{l})\}$ at each layer $l$ during training to populate the bins.

The paper mentions a "subsampling strategy" for the large Arxiv dataset (Sec 6), but this is vague. How many pairs are sampled? How does this subsampling affect the stability and accuracy of the probability ratio estimation? A complexity analysis is missing.

2. The reweighting estimation in Section 5.2 relies on two strong approximations:
$P_T(A_{uv}|H_{w}^{l},H^{1}) \approx P_T(A_{uv}|H_{w,u}^{l},H_{w,v}^{l})$
$P_T(A_{uv}|H_{w,u}^{l},H_{w,v}^{l}) \approx P_T(A_{uv}|d(H_{w,u}^{l},H_{w,v}^{l}))$
The second approximation, in particular, is a significant information bottleneck, reducing two high-dimensional embedding vectors to a single scalar distance. This might fail to capture more complex, non-metric relationships that determine connectivity. Some justification or analysis of this approximation's validity is needed.

3. The method introduces new and sensitive hyperparameters, most notably the reweighting coefficient $\lambda$ (Fig 4c) and the number of distance intervals $J$ (Fig 4d). The performance seems to drop off significantly if these are not set correctly (e.g., $J=100$ is optimal, but $J=10$ or $J=1000$ performs worse). This could make the model difficult to tune and apply to new datasets.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies graph domain adaptation under covariate shift by decomposing the shift into Feature Shift (FS) and Feature-Conditional Structure Shift (FCSS). The authors derive a target-risk upper bound that isolates contributions from initial (feature) representation mismatch and final (structure-conditioned) representation mismatch, and propose DACS — a dual-alignment algorithm that (i) adversarially aligns initial features, (ii) applies adaptive layer-wise edge reweighting to correct FCSS, and (iii) performs final-layer conditional alignment. Experiments on synthetic datasets and several real benchmarks show consistent improvements over prior GDA methods.

### Strengths
1. Strong and transparent theory. The decomposition of the target risk into three interpretable terms (source risk, initial representation discrepancy, and final representation conditional discrepancy) is clear and useful. Theorem 3.1 and the following discussion give an intuitive, actionable view on why feature alignment and structure correction are both necessary. The proposed modules (adversarial feature encoder, layer-wise reweighting, and final conditional adversarial alignment) follow directly from the bound: each component is motivated by a specific term in the bound. This one-to-one mapping strengthens the paper’s conceptual coherence.

2. Comprehensive experiments. The paper provides both controlled synthetic tests (separate FS, SS, and combined cases) and multiple real-world tasks. Results show that DACS improves robustness when feature and structural shifts co-exist, supporting the paper’s central claim empirically. The ablation study is informative.

3. Scalability considerations. The authors discuss and implement pragmatic strategies for large graphs (ArXiv), showing the method can be adapted to scale.

### Weaknesses
1. Robustness to semantic shifts and homophily mismatch. The focus is on covariate shift. However, a common and arguably more challenging scenario in graph DA is semantic shift — the conditional distribution P(Y|X,A) changes (labels behave differently across domains) — or cases where homophily patterns differ dramatically (homophilic ↔ heterophilic). It is not clear whether the reweighting approach can handle substantial changes in label–structure coupling or when source/target differ in homophily patterns. More advanced strategies, such as graph rewiring or graph structure learning, might be required to effectively address these semantic or homophily-related discrepancies. The paper briefly contrasts FCSS with label-conditional structure shift in the related work, but does not evaluate these scenarios.

I suggest that the authors discuss the potential effectiveness of the proposed method under semantic shift and, if possible, include additional experiments on graph domain adaptation between homophilic and heterophilic graphs to further validate the method’s robustness.

2. Limited reweighting visualization. The adaptive edge reweighting is central, but the paper provides only high-level descriptions and a few embedding visualizations. There is no direct visualization showing which edges are upweighted or downweighted and whether these correlate with label boundaries or structural motifs. A heatmap visualization of subgraph edge weights is suggested to compare the original source graph and the reweighted source graph, thereby providing an intuitive illustration of which edge patterns are being reweighted.

3. Complexity analysis is light. While the authors propose subsampling/top-k heuristics for large graphs, the paper lacks a formal time and memory complexity analysis.

### Questions
See weaknesses.

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
This paper presents Dual Alignment for Covariate Shift (DACS), a framework aimed at improving Graph Domain Adaptation by addressing covariate shift, particularly the Feature Shift and Feature-Conditional Structure Shift. These two types of shifts are common when transferring knowledge between source and target graphs, and they distort the learned representations, impairing cross-domain transfer. The proposed DACS method aligns both FS and FCSS using a combination of adversarial feature alignment, adaptive reweighting, and final-layer representation alignment. The authors show that DACS outperforms previous methods through extensive experiments across multiple datasets, demonstrating that the approach can effectively mitigate the challenges of domain shift in graph learning.

### Strengths
1. The paper introduces a comprehensive method to address two crucial aspects of covariate shift (FS and FCSS), which have been underexplored in prior work.
2. This empirical evidence supports the claims made by the authors and demonstrates that DACS can be applied successfully across different domains in graph neural networks.
3. DACS is grounded in a solid theoretical understanding of covariate shift and its impact on graph learning. The decomposition of covariate shift into Feature Shift (FS) and Feature-Conditional Structure Shift (FCSS) is well-justified, and the paper provides mathematical justification for why the proposed alignment techniques improve transferability.

### Weaknesses
1. Limited Novelty in Methodology – While the proposed method is a valuable contribution, the use of adversarial alignment and feature reweighting is not entirely new. These techniques have been explored in several other papers on domain adaptation and graph domain adaptation. The paper could provide a more detailed comparison to recent methods such as other MoE-based methods to better position DACS within the current landscape of domain adaptation methods.
2. The writing and presentation of the paper could benefit from further polishing. The logical flow is not always clear, and the use of notations is dense, which might hinder reader comprehension. Additionally, the figures and tables could be enhanced to improve clarity and better illustrate key concepts and experimental results.

### Questions
1. See weaknesses.
2. The algorithm does not show significant improvement on the Airport dataset. It would be useful to provide a more detailed analysis of the reasons behind this.
3. The experimental setup for rotation in Table 1 lacks sufficient explanation. It would be helpful to clarify what specific settings or transformations were applied during this experiment, as well as its relevance to the overall analysis.
4. While the experiments demonstrate the effectiveness of the method, the analysis is not comprehensive enough. The paper could be improved by including additional robustness or sensitivity experiments to assess the method’s performance under varying conditions.

### Soundness
2

### Presentation
4

### Contribution
2
