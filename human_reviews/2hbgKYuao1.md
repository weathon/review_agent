# Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality

- Decision: Reject
- Scores: 6, 6, 5, 6

## Abstract
We propose an expressive and efficient approach that combines the strengths of two prominent extensions of Graph Neural Networks (GNNs): Subgraph GNNs and Structural Encodings (SEs). Our approach leverages walk-based centrality measures, both as a powerful form of SE and also as a subgraph selection strategy for Subgraph GNNs. By drawing a connection to perturbation analysis, we highlight the effectiveness of centrality-based sampling, and show it significantly reduces the computational burden associated with Subgraph GNNs. Further, we combine our efficient Subgraph GNN with SEs derived from the calculated centrality and demonstrate this hybrid approach, dubbed HyMN, gains in discriminative power. HyMN effectively addresses the expressiveness limitations of Message Passing Neural Networks (MPNNs) while mitigating the computational costs of Subgraph GNNs. Through a series of experiments on synthetic and real-world tasks, we show it outperforms other subgraph sampling approaches while being competitive with full-bag Subgraph GNNs and other state-of-the-art approaches with a notably reduced runtime.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The proposed work mainly addresses concerns regarding the temporal complexity of Subgraph GNNs. It proposes a mechanism based on subgraph centrality to sample the graphs that will be part of the bag. Furthermore, it demonstrates that adding centrality encoding to nodes can enhance the discriminative power of sampled Subgraph GNNs while limiting the added computational overhead.

Main Contributions:
1. Adoption of centrality-based sampling for Subgraph GNNs.
2. Clear statement and results regarding incorporating structural encoding as means to improve expressivity of sampled Subgraph GNNs without much computational overhead.

### Strengths
1. **Relevance and Contextualization**: The proposed work addresses an issue (computational complexity of Subgraph GNNs) that is extremely relevant for the community. It clearly identifies gaps in existing methods and situates the research well within the current state of the art.
2. **Centrality-based Sampling**: The decision of sampling based on centrality measures has some theoretical backing. The analysis based on perturbations is well founded and creative.
3. **Structural Encoding**: The decision of incorporating structural encodings with subsampling is well motivated and has sufficient theoretical backing.

### Weaknesses
1. **Theoretical Concerns**: The analysis based on perturbation is not sufficient to justify using the highest centrality nodes to guide the sampling procedure. Even though they will tend to produce the highest perturbation in the graph representation, this alone does not fully justify their selection for the sampling procedure. These nodes may cause the highest perturbation in graph representations, but their practical value in enhancing graph learning remains uncertain, there is no guarantee on the usefulness and attainability of such modifications. Additionally, the assumption that high-centrality nodes capture the most relevant information is limited by the specific centrality metric used, which may not capture all essential features.
2. **Comparison of Centrality Measures**: The comparison between the adopted subgraph centrality, SC, and other centralities is not sufficient to prove SC's superiority. Moreover, centrality measures often complement each other rather than subsumming one another. SC may be more effective for certain tasks but not necessarily for all graph learning tasks. Rather than claiming SC's superiority, it would be better to show in which tasks SC excels and acknowledge that other centrality measures may outperform it in different contexts. Otherwise, stronger results linking the superiority of SC over every other centrality for graph learning, would be necessary.
3. **RWSE vs CSE**: While the proposed method (CSE) shows some advantages over RWSE, particularly from a sampling perspective as seen in Figure 7, its benefits appear limited. The experiments focus primarily on counting substructures, a relevant task but one that may not fully demonstrate CSE's broader applicability due to its inherent predisposition toward this type of task.
4. **Inconsistencies in Experiments**: The experimental results lack consistency, as the models used for comparison vary across datasets. It is understandable that in many cases the results for all datasets are not available in the original works. However, this inconsistency can raise confusion and concerns of cherry-picking. It would strengthen the results to ensure uniformity in model comparisons.
5. **Missing Runtime Comparisons**:  The efficiency of the proposed method is emphasized throughout the paper, yet runtime comparisons are not provided for all datasets. Since computational efficiency is a key focus, these comparisons should be included in every experiment to give a more comprehensive view of the method’s benefits.
6. **Confusing Proof**: The presentation of Theorem 4 lacks clarity. Specifically, in line 910, it is stated that Lemma 1 will be used to demonstrate that the multiset of values in the centrality encoding for each graph will be distinct. However, unless I am overlooking something, Lemma 1 does not seem to establish this. Rather, it appears to indicate that the global node is consistently selected for top-1 sampling, which does not sufficiently ensure distinguishability between the centrality multisets of the two graphs. This interpretation seems supported by the statement in line 963. Moreover, the topic of centrality encoding does not reappear until line 966.

Furthermore, considering that $A$ represents the adjacency matrix and $A^k$ denotes its $k$th power, the assertion in Equation 11 raises concerns regarding its mathematical validity and intuitive clarity. For example, the expression $A^{k+1}\_{u\_1,v} = A^{k+1}\_{u\_1, ;} \\cdot A^{k+1}\_{;, v}$ appears to be problematic.

From a path interpretation perspective, the original statement suggests that the number of paths of length $k+1$ between nodes $u_1$ and $v$ can be derived by aggregating information from $A^{k+1}\_{u_1, ;}$, which accounts for all paths of length $k+1$ originating from $u_1$, and $A^{k+1}{;, v}$, which encompasses all paths of length $k+1$ terminating at $v$ from any other node. Would it not be more accurate to represent this relationship as $A^{k+1}\_{u_1,v} = A^{k}\_{u_1, ;} \cdot A^{1}\_{;, v}$?

Additionally, I would like to point out the presence of redundant statements, such as $A^{k+1}\_{u_1, v} \geq A^{k+1}\_{u_1, v}$ found in line 961, which could benefit from clarification or removal.

### Questions
1. Could you clarify what new insights are brought by the results in Table 1 compared to those in Figure 6 and Figure 3? The distinction between these results is not immediately clear to me.
2. In line 247, you mention that the sampling strategy should "(...) alter the graph representations in a way that is consistent with the target space." This aligns with the theoretical concerns in point 1. However, the experimental study performed pertains to counting substructures, a task where the information extracted by SC is expected to be relevant. How do you expect this sampling method to compare to other centralities in tasks where SC may not encode the most relevant information, such as network flow characterization?
3. The results under OSAN represent 1-OSAN?
4. Proposition 2 - Appendix B relates MPNNs with SE to DSS-GNN. The complete proposed method shows better results than 1-OSAN (assumed, see above) which is upper-bounded by the 2-WL, and better results than policy-learn which is expected to be more expressive than 1-WL but less than 4-WL. Where is the complete proposed method positioned in the WL framework for expressivity?
5. In line 424, it is stated that Figure 3 compares random sampling with SC. However, earlier (line 51), you state that random sampling is suboptimal. Why were other sampling strategies not tested to fully validate SC?
6. What was the method used to select the hyperparameter T? A brief explanation in the text would provide more clarity.
7. It was not addressed why a higher T, meaning more subgraphs, meaning more information leads to worse results. Is the information not useful? Is it connected to the sampling procedure not taking into account already sampled subgraphs? This is unclear to me.
8. For proposition 1, Step 2, the block matrix $W^{(j+1)\_0}$ seems to select the first $k+j$ columns not $k+j+1$? Consider $k=3$ for $j=1$. The block matrix will have dimensions $7 \times 7$, with only the block in the first spot of the diagonal performing a selection since the second block $I_{j-1}$ is omitted.
9. For the same proposition, I would like to seek clarification regarding the explanation provided for $AXW^{(j+1)_1}$. Specifically, the identity matrix is described as having dimensions $k - (j - 1)$, yet the reference appears to describe the selection of the last $j$ columns. Additionally, the process by which the iterations from $j = 1$ to $j = k$ contribute to the formation of the final vector is not entirely clear. I would greatly appreciate any further elaboration on these points to enhance my understanding.
10. Considering choices of T > 1 in the experiments, for theorem 4, what is the impact of k>1 for top-k Subgraph GNN compared to MPNN+CSE?

Minor:
1. The phrase starting in line 214, "If instrumental (...)",  is confusing, consider rewriting it.
2. Quartic is often misspelled as "quatric".
3. Line 890, the expression "1-hop neighborhood" is imprecise, I recommend 1-hop induced subgraph.
4. Line 836, misspelled "the" as "he".
5. Missing subscript on line 952?
6. No identification of what $m$ denotes in equation 15.

Suggestions (Optional):
1. Line 228, "(...) untrained 3-layer GIN". By untrained I assume the weights were initialized at random. If this is the case, for GIN, different random initialization should lead to some variance, even if small, in the results, leading to variance in the perturbations. It would be more robust to report the mean difference in perturbations across multiple random initializations, as this would account for variance in the results.
2. There is no direct comparison with some relevant network architectures like $I^2$-GNN that are not captured by 1-OSAN. I understand that $I^2$-GNN was used as a comparison point in MAG-GNN, but the results were quite close in the referred work, hence, I believe it would be useful to add such comparison. More importantly, works like ESC-GNN introduce a faster alternative to $I^2$-GNN. Since the presented work also focuses on efficiency, I believe a comparison with ESC-GNN would be interesting.
3. Since much of the code used is based on GraphGPS, the configs could be more carefully described to be easier to match the experiments in the paper.

The paper presents contributions that may be of limited novelty while having some inconsistencies. However, I am **very** open to revisiting and potentially increasing my evaluation score, provided the authors effectively address the identified weaknesses and respond to the questions posed. I encourage the authors to consider these aspects to enhance the manuscript's impact and clarity.

### Soundness
2

### Presentation
3

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
This paper introduces a novel method called Hybrid Marked Network (HyMN), aimed at balancing 
computational efficiency and model expressiveness. The approach combines subgraph GNNs 
with structure encodings (SEs) based on walk centrality measures. The main innovation lies in 
utilizing subgraph centrality for subgraph sampling and structure encoding, allowing for the 
maintenance of prediction accuracy while reducing the required number of subgraphs, thus 
lowering computational costs while preserving the model's expressive capacity. Experimental 
validation on synthetic and real world tasks demonstrates that HyMN outperforms other state-of
the-art GNN methods while reducing computational demands

### Strengths
1. The walk centrality-based subgraph selection method achieves efficient subgraph sampling, 
simplifying the model while ensuring performance, making it suitable for larger subgraphs.
 2. Experiments on various tasks showcase HyMN's adaptability and performance.
 3. By ranking nodes based on their centrality values, and mark the top-scoring ones, the model 
reduces computation time, making it applicable to a wider spectrum of downstream tasks.

### Weaknesses
Abstract and Background Information:
The abstract is incomplete, as it lacks essential background information. This omission leaves me confused about the specific problem the paper aims to address. A well-defined problem statement is crucial for understanding the motivation behind the research and its significance.

Experimental Validation of CSEs:
I noticed that experiments on the effect of CSEs in the Peptides and Zinc datasets were not conducted. This absence raises questions about the impact of incorporating CSEs (Q4). Clarifying this point is essential for understanding how CSEs contribute to the overall findings of the study. Consider including experimental results or justifications for their omission.

Exploration of Selection Strategies:
The paper would benefit from exploring more complex selection strategies and different walk-based centrality measures. This exploration could provide deeper insights into the dynamics at play and strengthen the overall analysis.

Typos and Formatting Issues:
While typos and formatting are not the most critical issues, there are several areas that require attention. For instance, the indentation of the abstract does not align with the template requirements. Additionally, line 194 contains a grammatical error with "is are" used simultaneously, which should be corrected. Some equations lack punctuation at their end, where it is needed, leading to inconsistencies. Lines 206-210 stray from the scope of Definition 1 and should not be italicized. Furthermore, Algorithm 1 lacks a title, which is necessary for clarity and organization.

Section 2 - Clarification of Content:
Although Section 2 is titled “PRELIMINARIES AND RELATED WORK,” it primarily focuses on related work without adequately presenting the foundational definitions necessary for understanding the subsequent content. It would be beneficial to include a clearer explanation of the definitions and concepts that readers should be familiar with before delving into the related literature.

### Questions
Refer to Weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
To balance efficiency and expressiveness in subgraph GNNs, this paper proposes a novel framework that utilizes walk-based centrality measures for subgraph subsampling and structural encoding. The necessity of using centrality measures is demonstrated through theoretical analysis and experimental validation. Experimental results show the effectiveness of HyMN across various tasks and datasets.

### Strengths
1. The paper proposes a novel framework that balances the efficiency and expressiveness of subgraph GNNs.
2. This paper provides a comprehensive theoretical analysis and experimental validation of why the simple subgraph centrality measure is needed for subgraph GNN.
3. The experiment results demonstrate the effectiveness and efficiency of the proposed method.

### Weaknesses
1. The two main components (subsample subgraph using centrality measure, and centrality-based structural encoding) are similar with previous work.[1,2]
2. A more detailed introduction to the key background of this work would be helpful(i.e. the background and method of the node marking)
3. More backbone(except GIN) and more baseline should be preferred to be considered by authors.

[1] Sun, Qingyun, et al. "Sugar: Subgraph neural network with reinforcement pooling and self-supervised mutual information mechanism." *Proceedings of the web conference 2021*. 2021.

[2] Rampášek, Ladislav, et al. "Recipe for a general, powerful, scalable graph transformer." *Advances in Neural Information Processing Systems* 35 (2022): 14501-14515.

### Questions
1. Could the author provide more evidence to support the validity of the two claims in line 182?

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
3

### Summary
The paper proposes a model termed Hybrid Marketing Network (HyMN), which is designed based on two intuitions which may potentially improve the GNN performance. First, marking the nodes included in Subgraph GNNs using walk-based subgraph centrality measures as an efficient strategy. Second, augmenting the node features with the same centrality measures as Structureal Encodings (SEs).  The key insight is that walk-based centrality measures serve both as effective indicators for subgraph importance and as infrmative structrual features. The authors theoretically analysed the node marking strategy with graph perturbation theory and demonstrate that their approach effectively balance expressiveness and efficiency.

### Strengths
1. the theoretical analysis provide a solid foundation for their proposed method. The utilization of perturbation theory and expressive theory looks correct for me.
2. This method is neat in design, which only requires minimal computation overhead compared with baselines while maintaining competitive performance.
3. The experimental validation is comprehensive, besides comparing their method with SOTAs, they conduct synthetic experiments for counting substructures, detailed ablations and experiment time analysis.

### Weaknesses
1. The sampling strategy does not consider interactions between sampled subrgaphs that are already sampled, which can lead to potential redundancy.
2. The analysis focus primarily on single-node marking. While mentioned in the limitations, extending the analysis to multi-node marking could provide addiitonal insights. Imagine a social network representing employees in a company. In this network, there's a team of three senior managers (say Alice, Bob, and Carol) who work closely together and have very similar connections to other employees. They all interact with the same team members, attend the same meetings, and collaborate on similar projects. According to your approach, since all three managers have high centrality scores due to their positions, the algorithm might select both Alice and Bob for marking. However, because their network positions and connections are so similar, marking both of them provides largely redundant information. It would be more informative to mark Alice (representing the management team's perspective) and then perhaps mark someone from a different department or organizational level, like a developer or a project coordinator, to capture a different aspect of the company's structure.
3. The approach can be less effective on graphs where walk-based centrality measures don't align with task objectives. Consider a drug discovery task where we're trying to predict whether molecules will bind to a specific protein receptor. The binding activity often depends on specific functional groups located at the periphery of the molecule. Take acetylsalicylic acid (aspirin) as an example. The molecule's binding properties are largely determined by its acetyl group at the edge of the molecule, but the walk-based centrality measure would give more importance to the central benzene ring because it participates in more walks through the molecular graph. In this case, marking nodes based on centrality would emphasize the structurally central but functionally less relevant parts of the molecule, while potentially overlooking the peripheral functional groups that actually determine the binding behavior. This mismatch between structural centrality and functional importance could lead to suboptimal performance on specific prediction tasks.

### Questions
1. Have you investigated how the performance of you sampling strategy varies with graph density? Intuitively, in very sparse graphs, walk-based centrality might be less informative.
2. In the perturbation analysis, could the bound be tightened by considering specific graph properties or structures? For instance, does the bound become tighter for trees or graphs with bounded degree?
3. Is there any strategies to extend the perturbation analysis to handle edge features or different types of marking strategies?

### Soundness
3

### Presentation
3

### Contribution
3
