# Beyond Edge Deletion: A Comprehensive Approach to Counterfactual Explanation in Graph Neural Networks

- Decision: Reject
- Scores: 2, 4, 6, 8

## Abstract
Graph Neural Networks (GNNs) are increasingly adopted in domains like molecular biology and social network analysis, yet their black-box nature hinders interpretability and trust. This is especially problematic in high-stakes applications, such as predicting molecule toxicity, drug discovery, or guiding financial fraud detections, where transparent explanations are essential. Counterfactual explanations - minimal changes that flip a model's prediction - offer a transparent lens into GNNs behavior. In this work, we introduce XPlore, a novel technique that significantly broadens the counterfactual search space. It consists of gradient-guided perturbations to adjacency and node feature matrices. Unlike most prior methods, which focus solely on edge deletions, our approach belongs to the growing class of techniques that optimize edge insertions and node-feature perturbations, here jointly performed under a unified gradient-based framework, enabling a richer and more nuanced exploration of counterfactuals. To quantify both structural and semantic fidelity, we introduce a cosine similarity metric on learned graph embeddings, addressing a key limitation of traditional distance-based metrics, demonstrating that XPlore produces more coherent and minimal counterfactuals. Empirical results on nine real-world and five synthetic benchmarks show up to +56.3% improvement in validity and +52.8% in fidelity over state-of-the-art baselines, while retaining competitive runtime.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors present Xplore, a counterfactual (CF) explanation method for graph neural networks. The method is derived from CF-GNNexplainer (Lucic et al.), but with a diffusion twist that allows for edge addition. Experiments are performed on different datasets and baselines for the problem of finding counterfactual.

### Strengths
- The overall idea (of gradient guided optimization) is interesting.
- The experiments consider a good amount of datasets.
- The mathematical proof of convergence is sound, although very standard and too slow.

### Weaknesses
- Presentation: Some parts of the paper are hard to read. The literature citations completely stop the flow of the text, Appendix section are poorly referenced. The paper does not seem to have been proofread with the applied ICLR format.

- In the contribution stated claim 2) seem incorrect, claim 1) should be reformulated.

- The literature review is outdated, and omits works published after early 2024. The paper only uses work from 2023 and before, and ignores relevant baselines from 2023, 2024, and 2025.

- The claim of novelty of adding edges or perturbing features is false, as seen in works such as: (1) Empowering Counterfactual Reasoning over Graph Neural Networks through Inductivity, Verma et al, 2023; (2) Global Counterfactual Explainer for Graph Neural Networks, Kosan et al, 2022; (3) COMBINEX: A Unified Counterfactual Explainer for Graph Neural Networks via Node Feature and Structural Perturbations, Giogi et al, 2025

- Following these gaps in the literature review, important recent baselines are missing.

- The analysis of the experimental results is extremely lacking: in CF explanation, there is a trade-off between validity of the counterfactual and its distance to the original graphs. See Lucic et al. (2022), CF-GNNExplainer, or Ma et al. (2022), CLEAR. This trade-off is nowhere mentioned, and absent from the analysis. The good validity results of the method may be entirely explained by the algorithm not stopping until it finds any valid counterfactual.

### Questions
- Part 1: claim 1 seems to express that you are the first to consider edge and node deletion, but this is not true, as for instance the cited Ma et al.’s CLEAR already does this.

- Part 1: claim 2 appears incorrect as well, ”the closest counterfactual through directed modifications” is not mentioned in the rest of the paper, and in fact, as seen in Table 3 for GED, seems widely incorrect.

- Part 2: Since D4Explainer also uses diffusion to find CF through denoising diffusion, how does your method differ? Please add a deeper comparison with this paper.

- Part 3.1: Equation 2 is awkwardly introduced, and does not serve any purpose; the loss used is given in Equation 7. You should introduce the metrics for your objective here, not the loss.

- Part 3.1: the Node Counterfactual Explanation is unclear, and should go after your method or be more general. It fails to explain what Node Counterfactual Explanation is. Please state the objective.

- Part 3.2: As mentioned, equation (3) is a subcase of Lucic et al.’s work where the subgraph considered is the whole graph.

- Part 3.2: The idea of noisy perturbation then denoising is interesting, and very similar to diffusion. I would reframe the work this way.

- Part 4: Table 3 is misleading, and does not relate to the objective stated: you should compare fidelity/validity and GED/CS at the same time for each method. Fidelity aims at finding counterfactuals, GED looks for good, i.e close counterfactual. Hence both should be analyzed together, as there is a trade-off.

- Part 4: Table 3 and validity. Getting 100 % validity is not surprising since the algorithm stops when it finds a counterfactual. The comparison with baselines seems unfair. This is also NOT discussed anywhere in the paper, which is a major issue.

- Part 4: unsurprisingly, the GED/CS of the proposed method is much higher than that of other methods, since the algorithm.

- Part 4.3: I am not sure what is the purpose of this part, this is not introduced or mentioned in the paper, and poorly structured.

- Appendix A.1: I am puzzled as to why you rewrote the proof of an already proven theorem. It is sufficient to just cite a theorem and use its result.

### Soundness
1

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
2

### Summary
This paper proposed a gradient-based framework for graph counterfactual explanations that expands the search space beyond edge deletions to allow edge insertions and node feature perturbations. It optimizes a soft objective balancing prediction flip and distance to the original graph, and naturally extends to node-level counterfactuals.

### Strengths
1.	The proposed method XPlore achieves impressive improvements on both validity and fidelity metrics.
2.	The method’s performance was validated on 14 datasets, spanning multiple graph types.
3.	This paper explicitly acknowledged that the residual OOD effects remain open and links them to robustness of oracles.

### Weaknesses
1.	To my current knowledge, there exists prior work (e.g., C2Explainer) that has already enabled edge insertion and node feature perturbations; this might weaken Xplore’s claimed novelty unless positioned more precisely.
2.	While the evaluation was performed on 14 datasets, it is skewed towards molecular/biology category, with only one social network dataset (i.e., COLLAB). As social network analysis might be a key application area for GNN interpretability, adding more datasets in this area would benefit generality. 
3.	A few typos: (i) in Figure 2 (d), the caption reads “edge inserion” and should be “edge insertion”; (ii) in Section 4.2, “sparisity” should be “sparsity”.

### Questions
1.	Could you please situate the proposed work’s novelty against recent counterfactual explainers that support edge insertions and/or node feature perturbations, such as C2Explainer?
2.	Would you consider including recent baselines (2024-2025) that permit node perturbations or edge insertions?
3.	Would you consider adding more social-network datasets beyond COLLAB? This would help assess generality.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper studies counterfactual generation on graphs by allowing not only edge deletions but also edge additions and feature perturbations. The proposed gradient-based framework optimizes these operations in a unified manner. It replaces distance-based objectives with a cosine-similarity metric, yielding more coherent explanations. Experiments report substantially higher validity and fidelity than state-of-the-art baselines.

### Strengths
- Clear algorithmic description with well-structured steps; the method is easy to follow.
- Extensive experiments with comparisons against multiple competing approaches.
- Thoughtful discussion of future directions that can guide subsequent research.

### Weaknesses
- The contributions are repetitive and could be more concise. For example, contribution points 1 and 4 appear overlapping and could be merged.
- Positioning the work as an “extension” of a prior paper weakens the novelty message.
- Novelty is limited in parts; for instance, edge addition in counterfactual explanations has prior art.
- The motivation for feature perturbations is underdeveloped, and the ablation on this component is limited.
- The search space and resulting computational complexity are not sufficiently justified.
- It remains unclear why the method should yield better out-of-distribution robustness or influence.

### Questions
- Edge additions for counterfactual explanations have been studied. What is the specific new insight or advantage your approach provides over prior formulations?
- Complexity: With edge operations, a naive search could appear O(n^2). Please clarify why your algorithm remains O(|E| + n f)?
- Motivation for feature perturbation: Could you add concrete examples where edge edits alone fail but small feature changes produce plausible, faithful counterfactuals (e.g., molecular graphs where atom attributes change properties, or social/product graphs where node attributes shift recommendations)?
- Experimental protocol: How many runs were executed for the explanation module? You report standard deviations—does the table show mean ± std over k runs? Please state k and any fixed random seeds.

Editorial/Presentation Notes:
- Figure 2 colors are hard to distinguish; maybe increase line/marker thickness for clarity.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes XPlore, a counterfactual explainer for GNNs that can delete and insert edges and also perturb node features. The authors formulate counterfactual search as minimizing a prediction-change loss plus a distance loss. The method targets both graph-level and node-level counterfactuals and introduces a cosine-similarity-on-embeddings metric to capture semantic fidelity. Across 14 datasets, their method outperforms nearly all baselines, and OOD performance is discussed.

### Strengths
- XPlore covers graph modifications that most GCE methods don't. Insertions + feature shifts matter.
- Performance on benchmarks against baselines is very strong.
- The authors are honest about their OOD performance and highlight key challenges for all methods.

### Weaknesses
- The method still relies on an oracle model, which adds another degree of freedom for practitioners to consider and means XPlore also likely inherits the oracle's faults.
- The OOD discussion seems to suggest that XPlore focuses on model-flipping counterfactuals rather than plausible counterfactuals, which is a significant weakness.

### Questions
- Are there any ablations for examining deletions only, deletions+ insertions, and deletions+insertions+features performance?
- Have the authors examined examples for the molecular datasets to ensure that the generated counterfactuals are also chemically valid (e.g. does not violate valence rules)?
- Table 4 only considers CF-GNNExpl on one dataset. How about other baselines/datasets?

### Soundness
4

### Presentation
3

### Contribution
3
