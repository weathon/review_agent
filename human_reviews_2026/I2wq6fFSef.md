# Fragment-Wise Interpretability in Graph Neural Networks via Molecule Decomposition and Contribution Analysis

- Decision: Reject
- Scores: 2, 2, 6, 2, 4

## Abstract
Graph neural networks (GNNs) are widely used in the field of predicting molecular properties. However, their black box nature limits their use in critical areas like drug discovery. Moreover, existing explainability methods often fail to reliably quantify the contribution of individual atoms or substructures due to the message-passing dynamics, which entangle local representations with information from the entire graph. As a remedy, we propose SEAL (Substructure Explanation via Attribution Learning), an interpretable GNN that divides the molecular graph into chemically meaningful fragments and limits information flow between them. As a result, contributions of individual substructures reflect the true influence of chemical fragments on prediction. Experiments on both synthetic and real molecular benchmarks demonstrate that SEAL consistently outperforms existing methods and produces explanations that chemists judge to be more intuitive and trustworthy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to build an interpretable GNN model that quantifies the contribution of chemically meaningful fragments, ensuring that the explanations better align with domain knowledge. The proposed method decomposes molecular graphs into sets of fragments using a decomposition procedure similar to the BRICS algorithm and evaluates each fragment’s contribution. To isolate the effects of different fragments, the authors introduce a message-passing layer with separate parameters to control information flow within and between fragments. Experiments on synthetic datasets, real-world datasets, and a user study demonstrate the effectiveness of the proposed approach.

### Strengths
1.	The paper addresses an important problem — ensuring that graph explanations are consistent with established scientific understanding. This is a valuable and meaningful research direction, and the paper makes a concrete step toward it.

2.	The paper conducts experiments on both synthetic and real-world datasets to demonstrate effectiveness. The results are clearly presented through figures and tables.

3.	A user study is included to examine the differences between human understanding and graph explanation techniques, providing an interesting perspective and direction for future research.

### Weaknesses
1. Limited novelty. The paper lacks significant novelty. Explaining graphs at the fragment level and using message-passing schemes have been studied in several prior works [1–4]. The proposed method does not appear to introduce a new concept or contribution beyond these approaches.

2. Questionable methodological claim. The authors claim that “a final embedding of each node reflects not only its own properties but also the cumulative properties of distant atoms.” However, Section 3.2 adopts a similar approach to existing methods, making this claim of methodological uniqueness questionable.

3. Lack of detailed explanations. Some statements lack sufficient detail, making the paper difficult to follow. For example, in Line 95, the authors state that in both Cao et al. (2024) and Wang et al. (2025), fragments may contain significant signals from other parts of the molecule, thereby reducing local interpretability. However, the proposed method still allows information flow between fragments, so the distinction between approaches is not clearly explained. Similarly, in Line 105, the authors claim that Wu et al. (2023) use the BRICS method for fragment-level explanations; however, that method is not included in the comparisons.

4. Unclear selection of baselines. The choice of baseline models is questionable. HiGNN is primarily designed for property prediction, yet it is used here for an explanation task. The authors do not compare their method with other interpretable GNN models but instead focus on post-hoc methods and a property prediction model, limiting the fairness of the evaluation.

5. Simplistic evaluation and overly strong results. The model assumes linear independence in the final step (Section 3.1) and aggregates via summation. The evaluation tasks appear overly simple: in Table 4 (AUROC) and Table 5 (F1), the classification models achieve scores of 1.0 or very close to it. This raises concerns about potential data leakage or task simplicity and whether the proposed method can generalize to more complex benchmarks [5].

6. Unsupported statements. Some claims lack experimental or theoretical justification. For instance, in Line 155, the authors state that “the lack of bias would cause an equal distribution of contributions across all fragments, diminishing interpretability.” It is unclear how the label distribution affects this behavior or why the absence of $b$ would reduce interpretability.

7. Inconsistent or unclear visualizations. Several visualizations conflict with or are difficult to reconcile with the fragment decomposition method. For example, in the right panel of Figure 1, it is unclear how the shown functional groups are derived from the decomposition process. Similar issues appear in other figures as well.

8. Questionable benchmark selection. The chosen benchmarks are not well justified. The synthetic dataset, B-XAIC, is not a widely recognized benchmark. For real-world datasets, explanation datasets such as Mutagenicity are available but not used. The rationale for benchmark selection and fairness of comparison should be clarified.

9. Minor errors and missing details.There are some minor issues in the manuscript. For instance, in Line 256, “Hierarchical GNN” should be “HiGNN.” Some figures lack column labels or sufficient annotation of important nodes or subgraphs in the appendix.

10. Inconsistent interpretation granularity. While the paper claims to provide fragment-level explanations, it also proposes atom-level explanations. In some cases, the atom-level model (SEALAtom) performs better than the fragment-level model (SEAL), which raises concerns about the claimed benefits of fragment-level reasoning.


1, Yu, Z. and Gao, H., 2024. MAGE: Model-level graph neural networks explanations via motif-based graph generation. arXiv preprint arXiv:2405.12519.

2, Yu, Z. and Gao, H., 2022. Motifexplainer: a motif-based graph neural network explainer. arXiv preprint arXiv:2202.00519.

3, Liu, X., Luo, D., Gao, W. and Liu, Y., 2025, July. 3dgraphx: Explaining 3d molecular graph models via incorporating chemical priors. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1 (pp. 859-870).

4, Gui, S., Yuan, H., Wang, J., Lao, Q., Li, K. and Ji, S., 2023. Flowx: Towards explainable graph neural networks via message flows. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(7), pp.4567-4578.

5, Bui, N., Nguyen, H.T., Nguyen, V.A. and Ying, R., 2024. Explaining graph neural networks via structure-aware interaction index. arXiv preprint arXiv:2405.14352.

### Questions
1.	In Figure 1, is the claim that “the most polar groups contribute positively and the hydrophobic groups contribute negatively” supported by existing scientific evidence? Does the contribution of each fragment align with known chemical findings?

2.	Could the authors provide a comparison of computational efficiency or runtime between the proposed method and the baselines?

3.	In Line 133, can the embeddings before the graph readout stage be analyzed to evaluate atom-level contributions?

4.	For the synthetic dataset, how are the ground-truth explanations obtained? In the user study, did participants provide what they believe the true explanations should be based on their expertise? Why are the ground-truth explanations not presented in the user study?

5.	The paper states that a high $\lambda$ leads to more interpretable results, but in some experiments, the chosen values are very small. Could the authors explain this inconsistency?

6.	Why is HiGNN used both as the explanation model and as a baseline model?

7.	How is the budget for the final explanation results determined? How do the authors decide how many fragments are selected for the final explanation? Additionally, could the authors explain the color-coding scheme used in Figure 4?

8.	SEALAtom treats a single atom as a functional group. Why does it outperform baselines such as GNNExplainer and PGExplainer?

9.	In Line 676, the paper states that “we decide to mask the same amount of atoms for each molecule among all methods.” Is this assumption reasonable for the budget settings?

### Soundness
2

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
5

### Summary
The paper proposes a novel fragment-wise approach to molecular property prediction, utilizing BRICS fragmentation and aggregating fragment-level contributions to enhance interpretability of Graph Neural Network models. The rationale is grounded in pharmaceutical chemistry, where molecular properties often relate to functional groups, and the model aims to mirror this intuition in predicting outcomes.

### Strengths
The work addresses an important gap in interpretable molecular property prediction by focusing on fragment-level explanations, which have strong chemical relevance.

The adoption of BRICS and fragment aggregation is conceptually appealing and could aid chemists in understanding model decisions in familiar terms.

The paper is well-motivated and offers thoughtful discussion about interpretability, with the potential to advance explainable AI in the molecular domain.

### Weaknesses
The main evaluation relies on synthetic datasets that are often trivial and do not sufficiently challenge the model’s capacity for real-world tasks. Only three real-world datasets are used for more robust assessment, limiting generalizability. There exist many other datasets which are not used in this work. For instance, in the paper of HiGNN (Zhu et al., 2022) there are 11 real world datasets and none of them is investigated in this paper.

A big part of the evaluation checks alignment with ground-truth substructures but does not verify faithfulness with respect to the model’s decision process, which is instead done only with 2 datasets. An explanation should first be evaluated for the faithfulness using standard metrics (Fidelity, Inv Fidelity): if a model learns a spurious bias, the correct explanation should reflect that bias. Only after confirming model alignment, comparisons to chemical ground truth are meaningful. Let me cite Agarwal et al, 2023: "trained GNN model may only capture one or an entirely different rationale. In such cases, evaluating the explanation output by a state-of-the-art method using the ground-truth explanation is incorrect because the underlying GNN model does not rely on that ground-truth explanation. In addition, even if a unique ground-truth explanation generates the correct class label, the GNN model trained on the data could be a weak predictor using an entirely different rationale for prediction. Therefore, the ground-truth explanation cannot be used to assess post hoc explanations of such models."

Fragment extraction is presented as a generalizable mechanism, but the experiments rely exclusively on pre-defined methods (BRICS), which fail to pinpoint atomic-level contributions (e.g., Boron atom in Figure 4), and do not capture long-range interactions crucial for complex properties.

Benchmarking against relevant self-explainable GNNs (GNAN, GIB, VIB, PiGNN, KerGNN) is scarce as only one method is used as comparison, weakening the claims of superiority or novelty in interpretability.

Comparison to leading molecular property predictors (e.g., Grover, MultiChem, HyperMol) is absent.

The user study is incorrectly posed and totally useless, as the question asks which part of the molecule is thought to better impact the solubility. However, the quality of explanations should be primarily checked against the model. The user cannot know whether the highlighted part is truly the one impacting the model’s prediction. Additionally, no information is provided on the number and specialty of the study participants. The reviewer would assume that they are chemists. However, it is obvious that a chemist would be biased toward using functional groups, as that is exactly how they reason, but there are no other methods among the techniques that actually use fragments.

### Questions
Can the authors address why evaluation against other self-explaining GNN methods was omitted?

Can the authors address why other real world datasets were omitted?

Why wasn't fidelity calculated on B-XAIC datasets?

How do the authors ensure that user studies accurately evaluate faithfulness of explanation with respect to the model (rather than human chemistry bias), and provide more details about study participants?

Why do the authors only do Figure 6 on Fidelity with 30% of masking?

### Soundness
3

### Presentation
2

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
The paper propose a new algorithm called SEAL that identifies fragments in molecule and constraints message passing architecture. It measures the influence of the fragments in the molecule to the prediction rather than focusing on node or edge level. This would increase the interpretability of GNN and generate more intuitive explanations.

### Strengths
- Good motivation and introduction to the problem. 
- Code sharing for reproducibility.
- User study to show the effectiveness of the model compared to the baselines.

### Weaknesses
- It is not clear if fragmentation algorithm is unique in the model, or just borrowed from the available methods.
- It is not clear how SEALAtom differs than the original graph without any fragmentation.
- Equation 3 requires more explanation.
- No theory behind the method. One can claim that the model should be able to learn the influential edges without separating intra and inter edges. Besides, what is the guarantee that inter-edges are not influential?

### Questions
- What is the guarantee that the initial fragments are well-decided?
- Can you summarize what novelty this paper brings to the field? 
- Can you explain what SEALAtom actually offers compared to the original graph? If every atom is a fragment, then that is already the original graph.
- Can you explain your GNN layer in more detailed with the size of each elements (i.e., trainable weight matrices, embedding size etc.)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes SEAL, an interpretable GNN for molecular property prediction that first decomposes a molecule into chemically meaningful fragments, aggregates node features within each fragment, and predicts as a sum of fragment contributions.

### Strengths
1. The prediction head is explicitly the sum of fragment MLP contributions, so each fragment’s scalar contribution is directly interpretable.
2. The paper is well-written and easy to follow.

### Weaknesses
1. The paper would benefit from a fuller survey of (i) fragment/motif/scaffold-based explanation methods and (ii) models that explicitly handle intra- vs. inter-fragment interactions in molecular representation learning. Please clarify how your design choices (fragmentation strategy, message-passing split, and regularization) differ from prior work, what problems they solve, and where the contribution sits in the existing taxonomy. A small comparison table contrasting assumptions, granularity, learning signals, and interpretability guarantees would clearly highlight the novelty and scope of the approach.
2. To support the interpretability claims, the paper needs to include head-to-head evaluations against widely used subgraph-level explanation approaches under a shared protocol: same trained backbone, matched sparsity/size budgets, and standard metrics (fidelity/accuracy drop, sufficiency/necessity, stability/consistency across seeds), plus runtime/overhead.
3. Beyond post-hoc explainers, it is also crucial to compare with representation learning methods that build on subgraphs/motifs/hierarchies.

### Questions
Please refer to the weaknesses.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The manuscript introduces SEAL (Substructure Explanation via Attribution Learning), an interpretable GNN for molecular property prediction. SEAL partitions a molecular graph into chemically meaningful fragments and constrains inter-fragment information flow, aiming to disentangle local representations from global message-passing effects so that attributions on substructures better reflect true causal influence. Across synthetic and real benchmarks, SEAL reportedly surpasses prior explainability methods and yields explanations rated more intuitive and trustworthy by chemists.

### Strengths
- Addressing a known limitation of message passing: attribution entanglement across the whole graph.

- Comprehensive results on both synthetic and real-world datasets.

### Weaknesses
1. Limited novelty. The core idea is not particularly new; many prior works have proposed similar frameworks and pipelines.


2.  The baseline set appears dated (e.g., ProtGNN from 2022). Please include more recent and stronger competitors.

3.  Report performance under Bemis–Murcko scaffold splits and strict OOD scaffold tests.

4.  Provide runtime/memory profiles vs. popular explainers: train/inference time per molecule, scaling with fragment count, and large-scale throughput.

5.   Missing ablations on key design choices.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2
