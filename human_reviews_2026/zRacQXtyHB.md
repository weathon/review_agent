# Out-of-Distribution Robust Explainer for Graph Neural Networks

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Graph Neural Networks (GNNs) are powerful tools for analyzing graph-structured data; however, their interpretability remains a challenge, leading to the growing use of eXplainable AI (XAI) methods. Most existing XAI models assume that GNNs are well-trained and that all nodes in the graph share similar data characteristics to those used during GNN training. In real-world applications, new nodes and edges are frequently added to the input graph during testing. 
This dynamic environment can introduce out-of-distribution (OOD) nodes, potentially undermining the reliability of XAI models.
To address this issue, we propose an OOD Robust Explainer (ORExplainer), a post-hoc, instance-level explanation model specifically designed to provide robust and reliable explanations in the presence of OOD nodes, noise, and outliers in graphs. 
ORExplainer incorporates Energy Scores to capture structural dependencies, allowing for prioritizing in-distribution nodes while reducing the impact of OOD nodes. 
We conduct experiments with varying types of OOD node inclusion. 
ORExplainer demonstrates superior robustness of generated explanations across synthetic and real-world datasets. 
Our code is available at https://anonymous.4open.science/r/ORExplainer-C52C/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an OOD robust explainer for GNNs and introduces a WEP mechanism that suppresses unreliable OOD nodes while emphasizing in-distribution information. The authors also provide a theoretical analysis connecting WEP to a diffusion process. Experiments on synthetic and citation datasets show improved explanation stability and robustness compared to existing explainers.

### Strengths
1. The presentation of the paper is clear.

2. In the theoretical section, they try to connect the proposed mechanism with diffusion-based energy minimization, showing some effort toward providing interpretability and analytical grounding.

### Weaknesses
1. While the paper identifies three types of OOD nodes in Introduction, their definitions appear somewhat overlapping, as all are described in the context of new or injected nodes. It is unclear whether these categories are mutually exclusive or represent different perspectives of the same phenomenon. Clarifying the conceptual distinctions would improve the overall clarity of the problem setup.

2. The authors mention that existing explainers assume a fixed graph, but the failure modes of prior works are not clearly articulated. A deeper analysis of why such assumptions reduce robustness, and how the proposed method directly addresses these issues, would make the motivation more convincing.

3. Eq(3) defines WEP as a simple average between a node’s own energy and its neighbors’ energies. However, authors did not explain why this linear diffusion operation enhances robustness, or how many propagation steps k are used.

4. The theoretical derivation assumes that ID and OOD nodes have disjoint energy intervals. This assumption seems strong, and it is unclear whether it holds for real-world graphs where ID and OOD distributions may overlap.

5. Theorem 5.2 sets WEP as a lazy substochastic diffusion, showing that minimizing propagated energy reduces visits to high-energy nodes. However, this finding largely restates the intuitive effect of averaging over neighbors and does not offer a deeper theoretical guarantee of robustness or faithfulness. This analysis reads more like a mathematical restatement of the mechanism than a theory.

6. The experiments are conducted on small-scale datasets, which may not adequately test scalability or robustness on larger and more complex graphs. Including larger datasets would make the evaluation more convincing.

### Questions
Please refer to Weaknesses part.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ORExplainer, a post-hoc explanation model designed to generate robust and reliable explanations for GNNs in the presence of out-of-distribution (OOD) nodes. The method introduces an Energy-Score mechanism to prioritize in-distribution (ID) nodes while suppressing OOD influence. The paper evaluates the approach on both synthetic and real-world datasets and reports improved robustness of explanations under several OOD settings. The topic is timely and relevant to trustworthy graph explainability. However, the paper contains several writing, methodological, and conceptual issues that need to be clarified before the contribution can be properly assessed.

### Strengths
1. Addresses an important and underexplored problem—robust explainability under graph OOD conditions.
2. The use of Energy Scores for node importance is intuitively reasonable.
3. The experiments include both synthetic and real-world datasets, demonstrating practical relevance.

### Weaknesses
1. Writing clarity issues.
Some expressions are grammatically or semantically unclear. For example: Line 126: should read “The graph used for explanation …” instead of the current phrasing. Line 129: should use “The model f is a node classifier …” rather than “The GNN f is a …”.

2. Unclear generation of OOD settings.
Lines 301–307 only describe how synthetic node OOD and real-world feature OOD are generated. It remains unclear:
How are node OODs created in real datasets?
How are feature OODs constructed in synthetic datasets?
For synthetic “unseen-label” OOD, is the largest label simply treated as unseen?

Furthermore, the experimental design is inconsistent: synthetic datasets only test structural OOD, while real datasets only test feature and unseen-label OOD. Why not evaluate all three OOD types on both domains to support the claim of general robustness?

3. Questionable assumption on excluding OOD nodes.
 Lines 199–202 state that explanations should consist mainly of ID nodes, with OOD nodes excluded or down-weighted. However, this assumption may fail when OOD nodes directly cause prediction errors. In such cases, removing them may hide the model’s failure mechanism rather than provide a faithful explanation. The authors should discuss this limitation explicitly.

4. Missing extension discussion.
Could the proposed ORExplainer framework be extended to graph-level classification tasks? Since the method currently focuses on node-level explanations, a discussion of potential extensions would be valuable.

5. Target-node OOD scenario.
If the target node itself is OOD, can ORExplainer still produce a reliable and meaningful explanation? This scenario seems practically important, but is not analyzed in the paper.

### Questions
See Weaknesses.

### Soundness
2

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
3

### Summary
This paper studied the problem of generating robust post-hoc, instance-level explanations for Graph Neural Networks (GNNs) in dynamic settings，where new nodes/edges at test time can introduce out-of-distribution (OOD) noise and outliers, undermining existing XAI methods that assume distributional consistency. To address this challenge, the authors propose ORExplainer, which incorporates Energy Scores to capture structural dependencies, prioritize in-distribution nodes, and suppress the influence of OOD nodes during explanation generation. Experiments across synthetic and real-world datasets with varied OOD injection strategies show that ORExplainer delivers more reliable and robust explanations than prior approaches, and the implementation is released for reproducibility.

### Strengths
1. This paper proposes ORExplainer, which provides robust and verifiable instance level explanations for GNNs under test time OOD scenarios in dynamic graphs. The problem setting is novel, and the method is supported by solid experimental results and theoretical analysis.

2. The paper is generally readable and reasonably well structured.

### Weaknesses
1. Some experimental results appear to be insufficiently comprehensive. For example, Figure 3 shows only the BA-Community and Cora datasets.

2. The paper appears not to report the accuracy metrics commonly used by prior GNNExplainer.

### Questions
1. Some experimental results appear insufficiently comprehensive. For example, Figure 3 includes only BA-Community and Cora; please expand to additional datasets or justify this selection.

2. The paper does not report the accuracy metrics commonly used by prior explainers (e.g., GNNExplainer). Please justify this choice and, if appropriate, include those metrics for comparison.

3. In Table 1 and Table 3, some results appear less than ideal. Please explain the underlying reasons.

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
This paper proposes ORExplainer, an out-of-distribution (OOD) robust post-hoc explainer for graph neural networks (GNNs). It introduces Weighted Energy Propagation (WEP) based on node energy scores to suppress unreliable OOD nodes and enhance explanation reliability. The method provides stable, in-distribution–focused subgraph explanations even under noisy or OOD conditions. Extensive experiments on synthetic and real-world datasets demonstrate that ORExplainer outperforms existing explainers in both robustness and fidelity.

### Strengths
1. The proposed Weighted Energy Propagation (WEP) effectively suppresses OOD influence, offering a simple yet principled way to enhance explainer robustness.

2. Extensive experiments on diverse datasets show consistent improvements in both robustness and fidelity over existing methods.

### Weaknesses
1. The overall writing quality could be improved. For example: (1) the caption of Figure 2 is not expressed clearly; (2) the font style of the embedding matrix Z in Preliminaries should be unified; (3) abbreviations such as “out-of-distribution (OOD)” appear repeatedly across Introduction, Related Work, and Our Proposed Method; and (4) subsection formatting should be made consistent.

2. In Figure 2, the CE loss should also have an arrow pointing to ORExplainer, and the main diagram could benefit from more detailed and polished visual design.

3. The experimental section could be strengthened by adding quantitative analyses of the energy mechanism to demonstrate its contribution and effectiveness.

### Questions
1.The method ensures robustness by suppressing OOD message passing, but such OOD information may sometimes carry useful signals. It would be valuable to discuss how these potentially informative OOD components could be better utilized.

### Soundness
2

### Presentation
2

### Contribution
2
