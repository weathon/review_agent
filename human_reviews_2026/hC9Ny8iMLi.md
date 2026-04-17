# SAGA: Structural Aggregation Guided Alignment with Dynamic View and Neighborhood Order Selection for Multiview Graph Domain Adaptation

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Graph domain adaptation (GDA) transfers knowledge from a labeled source graph to an unlabeled target graph to alleviate label scarcity. In multi-view graphs, the challenge of mitigating domain shift is constrained by structural information across various views. Moreover, within each view, structures at different hops capture distinct neighborhood levels, which can lead to varying structural discrepancies. However, existing methods typically assume only a single-view graph structure, which cannot effectively capture the rich structural information in multi-relational graphs and hampers adaptation performances. In this paper, we tackle the challenging Multi-view Graph Domain Adaptation (MGDA) problem by proposing Structural Aggregation Guided Alignment (SAGA) that aligns multi-view graph data via dynamic view and neighborhood order selection. Specifically, we propose the notion of Structural Aggregation Distance (SAD) as a dynamic discrepancy metric that jointly considers view and neighborhood order, allowing the dominant view–order pair to vary during training. Through empirical analysis, we justify the validity of SAD and show that domain discrepancy in MGDA is largely governed by the dominant view–order pair, which evolves throughout training. Motivated by this observation, we design SAGA, which leverages SAD to dynamically identify the principal view-order pair that guides alignment, thereby effectively characterizing and mitigating both view- and hop-level structural discrepancies between multi-view graphs. Experimental results on various multi-relational graph benchmarks verify the effectiveness of our method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the challenging problem of Multi-view Graph Domain Adaptation (MGDA), where both the source and target domains consist of multiple relational graph views. The authors introduce a novel framework, SAGA (Structural Aggregation Guided Alignment), which dynamically aligns source and target graphs by selecting the most transferable *view–hop* pair based on a newly proposed discrepancy metric, Structural Aggregation Distance (SAD). The SAD quantifies view- and hop-level structural disparities between source and target graphs, and SAGA leverages it to guide dynamic alignment during training. Experimental results on ACM and MAG benchmarks demonstrate notable performance improvements over a variety of GDA baselines.

### Strengths
1. The paper introduces the MGDA setting and explicitly formulates the structural discrepancy problem in multi-view graphs, which extends the conventional GDA formulation and captures richer relational dependencies.
2. The proposed SAD metric provides an interpretable and quantitative tool to assess cross-view and cross-hop structural differences, showing a clear empirical correlation between smaller SAD values and reduced domain performance gaps.
3. Extensive experimental validation across ACM (two-view) and MAG (multi-relation) datasets demonstrates consistent and significant performance gains, supported by solid ablation and visualization analyses.
4. The paper is well-organized and clearly describes how dynamic hop and view selection improve over static designs.

### Weaknesses
1. **Clarity on MGDA vs. GDA definition** 
   While the paper emphasizes MGDA as a more general setting, it would benefit from a clearer formal distinction between standard GDA (single-view) and MGDA (multi-view). Specifically, it remains unclear whether MGDA is simply a multiple relation extension of GDA or a fundamentally different original graph domain adaptation (GDA) paradigm.

2. **Interpretation of SAD and performance correlation** 
   The empirical finding that a smaller SAD corresponds to smaller source–target performance gaps (Fig. 2) is interesting but under-explained. It would strengthen the paper to theoretically justify why minimizing SAD could implies improved transferability, possibly linking SAD to existing domain discrepancy measures (e.g., MMD or spectral distance).

3. **Dataset construction details** 
   Although the paper provides dataset statistics (Table 4), the construction pipeline of multi-view graphs should be more clearly described. For example: How exactly are multiple views (e.g., PAP, PSP, PP) extracted and synchronized between source and target? Are the node sets shared across views, or partially overlapping? How does this multi-view setup differ from standard GDA datasets where only single-view adjacency is considered?  

4. **Comparative coverage and positioning** 
   The relation between SAGA and recent homophily-aware GDA frameworks such as HGDA could be more explicitly discussed. It remains unclear whether SAGA’s view-hop aggregation mechanism could be unified with or compared against homophily–heterophily decomposition strategies.

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes SAGA, a framework for multi-view graph domain adaptation (MGDA). The key idea is to introduce Structural Aggregation Distance (SAD) to quantify cross-domain and cross-hop structural discrepancies and dynamically select dominant view–hop pairs to guide alignment. The method aims to jointly mitigate view-level and hop-level shifts, achieving strong empirical results on ACM and MAG benchmarks.

### Strengths
- This paper proposes a new Multiview Graph Domain Adaptation (MGDA) setting and introduces a new benchmark that extends the original GDA scenarios to more realistic and diverse environments.

- The paper identifies an important problem domain adaptation across multi-view graphs and provides an intuitive metric (SAD) to quantify multi-hop structural discrepancies.

- The dynamic view–hop selection is a novel perspective that integrates both inter-view and intra-view adaptation.

- The empirical results across multiple benchmarks show promising performance improvements over prior GDA and MGDA baselines.

### Weaknesses
- The definition of SAD is based on a fixed formulation (Eq. 3–4), yet the paper claims to dynamically mitigate this fixed metric during training. It remains unclear how the method achieves mitigation rather than mere re-weighting of a static discrepancy. More discussion is needed to connect the empirical observation that smaller SAD correlates with the mitigation of the performance gap (Fig. 2) to the optimization mechanism that drives SAD reduction. Does the model directly minimize SAD, or is SAD only used to select dominant views? Clarifying this link is crucial to understanding the role of SAD in SAGA.

- While the experiments include many graph-domain adaptation baselines, some common GDA baselines (e.g., [1], [2]) are absent. Including or discussing these would strengthen the empirical validation and confirm SAGA’s advantage.

- SAGA involves multiple loss components ($L_R$, $L_{IA}$, $L_{CA}$, $L_S$, $L_T$) and dynamic SAD computation. How does this computational cost compare to simpler baselines such as HGDA or PA in practice? 

- Could you further clarify the relative importance of View and Neighborhood Order in the selection process? Specifically, under what circumstances does the View play a more critical role, and in what cases do different Neighborhood Orders become more influential? 

[1] Chen W, Ye G, Wang Y, et al. Smoothness Really Matters: A Simple yet Effective Approach for Unsupervised Graph Domain Adaptation[J]. AAAI, 2025.

[2] Yang L, Chen X, Zhuo J, et al. Disentangled Graph Spectral Domain Adaptation[C]. ICML, 2025.

### Questions
- Can the authors elaborate on how SAD dynamically changes during training? Is it re-estimated per mini-batch, or per epoch?

- Does the model ever backpropagate through SAD values (Eq. 3), or are they detached and used only for view/hop selection?

- Could SAGA be applied when the number of views differs between source and target graphs?

  **Some minor concern**：

- Figures 2–3 captions could explicitly indicate that SAD is *not* a direct loss but an auxiliary metric.

- Typo issue: in line 213 ''viws'' should be ''views''.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes the practical multi-view graph domain adaptation (MGDA) problem and introduces the effective SAD metric and SAGA framework with clear structure and strong motivation. However, it lacks theoretical analysis of SAD (e.g., comparison with MMD), convergence guarantees for dynamic selection, and complexity analysis of SAD computation. Numerous formatting issues are also present.

### Strengths
The paper clearly points out the limitation of most existing graph domain adaptation methods, which are largely confined to single-view graphs, and systematically proposes the multi-view graph domain adaptation (MGDA) problem setting, which aligns well with practical application needs. The concept of Structural Aggregation Distance (SAD) is a major highlight of the paper. Through empirical analysis, the effectiveness of dynamically selecting dominant view-order pairs is demonstrated, providing strong motivation for the methodological design. The SAGA framework has a clear structure, incorporating scalable graph encoding, dynamic neighborhood selection, and intra-/inter-domain alignment modules, forming a complete and coherent solution.

### Weaknesses
The paper currently lacks theoretical analysis of the SAD metric. For instance, what are the theoretical connections or advantages of SAD compared to classical distribution discrepancy measures such as MMD or Wasserstein distance? Is there any theoretical guarantee on the convergence of the dynamic selection strategy? Although Section 5.4 mentions that the computational cost is "practically acceptable" and Appendix C (Table 5) compares training time and memory usage, there is still a lack of quantitative complexity analysis specifically for the SAD computation itself. Additionally, the paper contains numerous formatting issues, such as missing spaces, improper placement of Figure 3 and Table 2, and punctuation errors (e.g., using commas at the end of sentences).

### Questions
The paper mentions that the computational overhead of SAD is "practically acceptable" as the terms do not participate in backpropagation. However, a more detailed complexity analysis of the SAD computation itself is needed for readers to fully assess the scalability of the proposed method. Could the authors please provide a quantitative analysis? 
The dynamic selection of the principal view-order pair is a central contribution of SAGA. To gain a more intuitive understanding of how this mechanism guides the alignment process during training, a visualization analysis would be highly insightful.

### Soundness
3

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
5

### Summary
This work proposes a novel research area, i.e., multi-view graph domain adaptation (MGDA), which focuses on learning node embeddings from source multi-view graph and then adapting the learned label information to a target multi-view graph. To address the cross-view, cross-hop, and cross-domain knowledge shift problem, this work introduces Structural Aggregation Distance (SAD) and proposes a framework, SAGA, for MGDA.

The experiments show SOTA results across two datasets and 12 source-target domain pairs, which demonstrate the power of SAD and the effectiveness of SAGA for MGDA.

### Strengths
1. [Interesting Problem] The proposed multi-view graph adaptation (MGDA) problem is both novel and appealing. This line of research has strong potential for generalization to other domains. The authors also claim to be the first to formally investigate the MGDA problem.
2. [Well Motivated] The proposed framework is well motivated, grounded in insightful observations about the relationship between SAD and the domain gap. The overall design of SAGA and the research questions are logically coherent and well articulated.
3. [Attractive Presentation] The paper is clearly organized and well presented. The authors begin with an empirical study that effectively highlights their motivation, followed by a method grounded in optimization loss derived from that study. This logical flow enhances the overall clarity and readability of the paper.

### Weaknesses
However, there are several important issues regarding the **soundness** and presentation details.

1. The empirical study is not convincing.
   * What is the meaning of $L_s - L_t$. This does not make sense if they denote the equation 14 and equation 15, since Eq. (15) is actually the entropy of predictions, and Eq (14) is cross entropy with labels, and their difference cannot be used to represent domain gap.
   * Moreover, these plots cannot support the claim "a smaller Structural Aggregation Distance (SAD) generally indicates small cross-network performance gap between domain" which is one of the main idea of the whole framework, if the x-axis is the training epoch that is optimized by Eq (16). Under the supervision of Eq. (16) that minimizes SAD and cross entropy simultaneously, it is very natural that the SAD and $L_s-L_t$ will decrease, but this does not mean SAD will *indicate* domain gaps.
   * Suggestions: in the empirical study, use a more concise loss instead of the complete loss of SAGA. For example, it might be better if the x-axis denotes the minimization of SAD single. Or just show this experiment as an analysis.

2. The problem is not clearly defined and introduced, although the authors claim that problem is one of the important contributions.
   * The authors introduce MGDA. However, they do not **formally** and clearly define this problem. This may confuse others for the experimental settings and optimization objectives.
   * The importance of MGDA is not explicitly and intuitively shown. Why MGDA is an important research direction, what specific problem MGDA could address, and how it benefits to down-stream tasks (applications) is not illustrated. Therefore, the **significance** of MGDA is not clear, weakening the overall contributions.

3. Other issues regarding presentations:
   * Related work is not well organized. It should categorize the literature, and describe their difference and relations to this work.
   * Notation part is not complete, consistent, and clear. For example, $K_T$ and $K_s$ should be defined in their first use; $G$ and $G^S$ ($G^T$) are defined differently. Why do $V$ and $\varepsilon$ not involved in $G^S$ ($G^T$)?
   * Section 4.1 is not complete. Please check Line 282.
   * Experimental settings are not clear. What are the train set, validation set, and test set?
   * The current version still has a few typos and format issues.

### Questions
Please try to address weaknesses.

### Soundness
1

### Presentation
3

### Contribution
2
