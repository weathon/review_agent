# BottleneckMLP: Graph Explanation via Implicit Information Bottleneck

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
The success of Graph Neural Networks (GNNs) in modeling unstructured data has heightened the demand for explainable AI (XAI) methods that provide transparent, interpretable rationales for their predictions. A prominent line of work leverages the Information Bottleneck (IB) principle, which frames explanation as optimizing for representations that maximize predictive information $I(Z;Y)$ while minimizing input dependence $I(X;Z)$. We show that explicit IB-based losses in GNN explainers provide little benefit beyond standard training: the fitting and compression phases of IB emerge naturally, whereas the variational bounds used in explicit objectives are too loose to meaningfully constrain mutual information. To address this, we propose BottleneckMLP, a simple architectural module that implicitly enforces the IB principle. By injecting Gaussian noise inversely scaled by node importance, followed by architectural compression, BottleneckMLP amplifies the reduction of $I(X;Z)$ while increasing $I(Z;Y)$. This yields embeddings where important nodes remain structured and clustered, while unimportant nodes drift toward Gaussianized, high-entropy distributions, consistent with progressive information loss under IB. BottleneckMLP integrates seamlessly with current explainers, as well as subgraph recognition tasks, replacing explicit IB terms and consistently improving predictive performance and explanation quality across diverse datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new architectural approach to improve the interpretability of Graph Neural Networks (GNNs) through implicit enforcement of the Information Bottleneck (IB) principle, without relying on explicit IB loss terms. Although the paper’s title sounds broad, the actual problem scope is much narrower, where they only focus on improving Ante-hoc and IB-based GNNExplainablity. Their goal is to show that explicit IB is unnecessary even within IB-based ante-hoc frameworks.

### Strengths
1. Their experiments demonstrate that adding BottleneckMLP often outperforms using explicit IB loss terms in existing ante-hoc explainers (GSAT, PGIB, TGIB) across multiple datasets. The results are effective. 
2. They provide analysis showing how unimportant node embeddings are pushed toward Gaussian, high-entropy distributions (i.e. “forgetting”), while important nodes remain structured, thereby aligning with the IB principle in a principled manner.

### Weaknesses
1. The claim that “a prominent line of work leverages the IB principle” is overclaim. IB-based explainers represent only a small subset of existing methods, and most state-of-the-art explainers (e.g., SubgraphX, GOAt, ReFine, PGExplainer++) do not rely on IB and typically perform better. 
2. By restricting comparisons only to IB-related baselines (GSAT, PGIB, TGIB) and including only one outdated non-IB baseline (PGExplainer, 2020), the experimental validation becomes narrow and unconvincing. Broader comparison with recent non-IB explainers is necessary to justify the contribution. 
3. Explaining temporal GNNs is within their scope, but the evaluation is incomplete. They do not clearly justify the pros and cons of applying IB to TGNNs, and they fail to compare with known temporal explainers like T-GNNExplainer.
4. They compare only to IB-based explainers, but they should also include other ante-hoc GNN explainers that are not IB-based.

### Questions
Why limit baseline to GCN? Why not test on GIN, which is stronger than GCN in many graph tasks? Their experimental setup seems biased by weak baselines.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the limitations of explicit Information Bottleneck objectives in GNN explainers and proposes a novel architectural module, BottleneckMLP, that implicitly enforces the IB without relying on auxiliary IB loss terms. First, the paper demonstrates that the existing explict IB losses are difficult to satisfy the i.i.d. assumption due to the structural dependencies inherent in graph data. Second, the paper achieves implicit extraction of critical information through importance-weighted Gaussian noise injection and additional MLP layers. Finally, the effectiveness of the proposed module is validated on both real-world and synthetic graph datasets.

### Strengths
1. The paper demonstrates that the existing explict IB losses are difficult to satisfy the i.i.d. assumption due to the structural dependencies inherent in graph data.
2. The paper proposes achieving implicit IB process through weighted noise injection and MLP layers, with certain theoretical guarantees.
3. The effectiveness of the proposed module is validated on both real-world and synthetic graph datasets.

### Weaknesses
1.	Figure 1 does not illustrate the crucial step that node embeddings are perturbed based on importance weights, the key question is whether the MLP in the subgraph extractor module and the MLP used to predict node importance weights share parameters and why.
2.	In experiment, Tables 1 and 2 lack validation on a broader range of real-world or synthetic datasets, as well as AUC/ROC of explanation subgraph, For example, on datasets such as Alkane-Carbonyl, Fluoride-Carbonyl, and Benzene from [1] ("Evaluating attribution for graph neural networks"), among others.
3.	The hyperparameter \sigma in Eq(6) also plays a crucial role in noise injection, should an additional parameter sensitivity analysis  be included in the experimental section?
4.	In presentation, the text in all figures is too small and causes some difficulty in reading.
5.	Adding an efficiency analysis and a detailed algorithmic description would make the method clearer in terms of computational efficiency and implementation details (or alternatively, providing concrete code implementation would be helpful).

### Questions
1. As shown in Figure 2(b), why does the CE loss alone for GNNs not exhibit the two distinct phases  similar to that in DNNs? and why are additional MLP layers introduced after adding weighted noise, and is this architecture necessary for achieving implicit IB? Please provide a detailed analysis rather than merely  based on visualization results.
2. Figure 3 lacks a more analysis, for example, why does I(Z;Y) and I(X;Z) at shallow layers (1 or 2) first decrease and then increase as training epochs progress? Please provide a more detailed analysis.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores whether the Graph Information Bottleneck (GIB) loss term is effective.
The authors argue that the i.i.d. assumption and the structural entanglement inherent in graph data violate the conditions required for these information-theoretic bounds to hold.
To address this issue, the paper proposes BottleneckMLP, which injects Gaussian noise into node embeddings and causes unimportant nodes to drift toward high-entropy, Gaussianized distributions.
Results across multiple graph explainers show that BottleneckMLP produces better explanatory subgraphs than GIB.

### Strengths
1. The novelty of this paper is strong. Although the GIB has been widely applied in graph explainers, there has been little analysis of its actual effectiveness. This work fills the gap.
2. This paper provides a rigorous theoretical foundation for deriving the proposed BottleneckMLP.
3. Extensive experiments of this paper support the method proposed in this paper.

### Weaknesses
1. The font size in Figure 1 of the paper is too small, and some elements lack legends for annotation.
2. Can this method be extended to post-hoc explainers, such as V-InFor [1]? Can the performance of post-hoc explainers be compared?
3. The paper notes the default MLP architecture is $h \rightarrow h/4 \rightarrow h$ and that finding the optimal architecture is like hyperparameter tuning. While Appendix G tests other configurations, a brief discussion behind a bottleneck-then-expansion structure versus a purely compressive one (e.g., $h \rightarrow h/4$) would be beneficial. It's unclear if the expansion phase plays a key role or if the compression is the only necessary component.
4. The paper reports Fidelity scores but does not provide results for sparsity, which is also an important property for explanatory subgraphs. Since the standard GIB can adaptively select the optimal budget, it remains unclear whether BottleneckMLP possesses this capability as well.

Reference:
[1] Wang, S., Yin, J., Li, C., Xie, X., & Wang, J. (2023). V-infor: A robust graph neural networks explainer for structurally corrupted graphs. Advances in Neural Information Processing Systems, 36, 56469-56487.

### Questions
Please see weaknesses.

### Soundness
4

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
The paper introduces BottleneckMLP, a new approach to induce information bottleneck (IB) in graph neural network (GNN) ante-hoc explainers (those that are trained together with the GNN downstream task). The paper shows, empirically and theoretically, why IB-based losses in existing ante-hoc explainers fail; then, it presents an theoretically-sound approach to circumvent this limitation (BottleneckMLP); finally, it provides empirical evidence of the success of this circumvention. The paper covers the preliminaries in IB theory, as well as related work, and the theoretical approach is based on proven lemmas and theorems. Overall, the proposed method to induce IB outperforms ante-hoc explainers without it (with and without IB-based losses proposed in previous work) in both explanation quality and accuracy. Additional experiments support additional claims, such as improved representation dynamics of BottleneckMLP over other methods. The paper ends with a summary of future directions.

### Strengths
The main strength of the paper is stating, measuring and mitigating a relevant limitation from previous work in IB approaches for GNN explainability. This is an important and significant issue in the field. In this case, the paper shows that current IB-based losses to induce better explanations lack soundness, and the paper proposes an alternative that is argued to mitigate this issue, both with theory and experiments to support this claim.

The paper is also strong in the sense of discussing a deeper theoretical analysis in a formulation that is rather simple: adding Gaussian noise and then using under-parametrised neural networks for compression. The simplicity of the method is also a highlight.

The paper is well organised and presented, it is very easy to follow the itemised contributions throughout the whole paper structure.

### Weaknesses
1. I missed a measure of the actual running time of BottleneckMLP when compared to other methods (with and without IB-based losses). Would it be possible to provide them, please?

2. The justification for the looseness of the variational upper bound (Sec. 4.1) did not convince me. Going away from an abstract approach, for instance, let’s consider GSAT. Which is the loss being used? And why exactly is it “loose”? What does it actually mean to be “loose”? Is it only because it relies on approximation? I believe that approaching these questions would clarify the looseness claim.

3. I don’t want to frame this as an issue of the paper itself as I believe this is actually a discussion about the GNN explainability field, but I miss a discussion in the paper about what *is* a graph explanation. Is it actually true that an information-bottlenecked-sub-graph is the sub-graph that correctly “explains” the downstream decision? What if the model, in its black-box architecture, uses other nodes that “are not meant to be used”? In some sense, this discussion collapses into the post-hoc vs ante-hoc discussion: should one “train” an explanation, which in this case loses the explainability role and becomes another model output?

4. The whole Sec. 4.3.1 develops on why variational bounds are used but in the end it simply says that the assumptions break with graphs. So why all the formulas? It could have been stated from the beginning, if the formulas do not contribute to the statement. I mean, it loses a special space in the paper.

5. When graphs are side-by-side, the default is sharing the same y-axis. Why is this not the case for Fig. 2?

6. The future directions could be more developed. I understand the limitation of space in the submission. Can you please develop more on those?

7. The abstract states that “explicit IB-based losses in GNN explainers provide little benefit beyond standard training: the fitting and compression phases of IB emerge naturally”. However, Fig. 2 supports that compression only happens with BottleneckMLP: “In Figure 2c, I(X; Z) rises early as task-relevant features are captured, then declines in later epochs, reflecting effective compression.” (L341). Can you please clarify? It could also be useful to see Fig. 3 for “Original” and “w/o IB Loss”, in addition to “BottleneckMLP”.

8. Very minor issues: Please take a look at the parenthesis in the citations. Almost all citations are with incorrect parenthesis. E.g., where it is “node classification Luo et al. (2020b)”, it should be “node classification (Luo et al., 2020b)”. Probably easy to change `\citet{}` to `\cite{}` in LaTeX. Also, please increase the font size of the text inside the figures. Finally, please take extra care to some parts of the text with typos (e.g., L178, L179, L229, L338).

### Questions
Please refer to the Weaknesses session.

### Soundness
3

### Presentation
4

### Contribution
3
