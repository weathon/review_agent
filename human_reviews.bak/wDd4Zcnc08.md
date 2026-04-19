# HP$^3$-NS: Hybrid Perovskite Property Prediction Using Nested Subgraph

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 5

## Abstract
Many machine learning techniques have demonstrated superiority in large-scale material screening, enabling rapid and accurate estimation of material properties. However, data representation on hybrid organic-inorganic (HOI) crystalline materials poses a distinct challenge due to their intricate nature. Current graph-based representations often struggle to effectively capture the nuanced interactions between organic and inorganic components. Furthermore, these methods typically rely on detailed structural information that hinders the applications of the methods for novel material discovery. To address these, we propose a nested graph representation HP$^3$-NS (Hybrid Perovskite Property Prediction Using Nested Subgraph) that hierarchically encodes the distinct interactions within hybrid crystals. Our encoding scheme incorporates both intra- and inter-molecular interactions and distinguishes between the organic and inorganic components. This hierarchical representation also removes the dependence on detailed structural data, enabling the model application to newly designed materials. We demonstrate the effectiveness and significance of the method on hybrid perovskite datasets, wherein the proposed HP$^3$-NS achieves significant accuracy improvement compared to current state-of-the-art techniques for hybrid material property prediction tasks. Our method shows promising potential to accelerate hybrid perovskite development by enabling effective computational screening and analysis of HOI crystals.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a hybrid graph neural network (GNN) architecture to predict the properties of hybrid materials comprising both organic molecules and inorganic atoms. The proposed method shows effectiveness in predicting bandgap value for organic-inorganic perovskites.

### Strengths
1.	This work aims to leverage GNN to predict the properties of hybrid organic and inorganic materials. This is a novel task.

2.	This work proposes using a hybrid GNN to obtain high-quality graph-level embeddings for hybrid materials. The proposed hybrid GNN contains a nested GNN to specifically extract molecular representation for the organic molecule contained in the material. This model design is interesting.

3.	A new dataset of organic-inorganic perovskites is proposed, which can facilitate future ML works studying this task.

### Weaknesses
1.	Does the curated dataset provide atom coordinates?

2.	Why do authors follow classic ML methods to do structure-agnostic prediction? Currently, 3D structures are commonly considered by GNN methods developed to predict material properties. I don’t think “aiming to apply our model for analyzing large scale materials and discovering novel materials” and “using 3D structure” are contradictory. If the dataset contains material structures, and the proposed method can outperform SOTA methods like CGCNN (using structure). Then, it’s an exciting result.

3.	An ablation study is needed to confirm the effectiveness of the proposed edge design.

4.	Does the curated dataset provide more property targets? It’s encouraged to do experiments with more properties.

5.	In the last equation in Sec 3.4, what are node weights $n(i, j)$? Its definition is missing. 

6.	Minor: The serial numbers for equations are missing from Page 5.

7.	Minor: In Figure 1(b), the $e(i, j)$ between Cs and Sn is wrong. The ratios of H and N in $CH(NH_2)_2$ are switched.

8.	Typo: In Sec 3.2, “Each of the A, B, and C can be …” Here, C should be X?

### Questions
See in weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposed a method to construct graph connections and edge features for HOIPs, which can be represented as $ABX_3$, without structural information, to capture the relationships between atoms. The experiments show that the proposed graph construction is better than direct merging all features.

### Strengths
1. The authors claim to be the first work to utilize GNN and to learn separate representations for organic molecules.
2. They collected a small dataset with ~900 HOIP samples.

### Weaknesses
However, I hold concerns for the significance and generality of the proposed method.

1. The major contribution is a graph construction method without structural information, basically how to determine the edge connections and how to determine edge features, for a constrained  HOIPs system with format $ABX_3$. The solution is somehow straight forward and may not be able to extend to general cases.

2. Although there are limited approaches for HOIP systems, but constructing heterogeneous graphs to capture interactions between different components of chemical systems is not new, and this proposed method is far from the first.

### Questions
Is the constructed graph a fully connected one? The graph construction details are not very informative, and the figures in the paper is not very clear.

**I have read the authors’ rebuttal, and the concerns remain.**

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a nested graph representation for perovskite property prediction.

### Strengths
1. The investigated problem is important in the material science domain.
2. The figures in the experiment section are illustrative.

### Weaknesses
1. More baselines should be used and tested on more datasets.
2. More technical details should be included.
3. More theoretical contributions should be made.

### Questions
1. Are there any more baselines that should be compared?
2. What are the theoretical contributions of this work?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a novel graph representation for hybrid inorganic-organic materials targeted for perovskite bandgap prediction. Specifically, the proposed method considers the representation of molecules with nested graphs. And it attains the edge features from the chemical formula directly, eliminating the need for computationally intensive DFT calculations. Notably, the authors synthesized and tested 35 new perovskites, demonstrating a strong alignment between experiment results and model predictions.

### Strengths
1. The problem of hybrid organic-inorganic (HOI) crystalline materials is interesting, and the motivation is convincing.

2. The method is reasonable, and extracting the edge features directly from the chemical formula is a desirable property, which is necessary for material discovery. 

3. The synthesis experiment really helps validate the proposed method and make it more convincing.

### Weaknesses
1. The experiments are kind of inadequate, as the GNN baselines are too old and few (only 1 GNN...), and there are no ablation studies.

2. Writing typos. The paper needs more careful proofreading.

### Questions
Please answer the first question in the weakness part.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
