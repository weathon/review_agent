# Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Hypergraphs provide a natural way to represent higher-order interactions among multiple entities. While undirected hypergraphs have been extensively studied, the case of directed hypergraphs, which can model oriented group interactions, remains largely under-explored despite its relevance for many applications. Recent approaches in this direction often exhibit an implicit bias toward homophily, which limits their effectiveness in heterophilic settings. Rooted in the algebraic topology notion of Cellular Sheaves, Sheaf Neural Networks (SNNs) were introduced as an effective solution to circumvent such a drawback. While a generalization to hypergraphs is known, it is only suitable for undirected hypergraphs, failing to tackle the directed case. In this work, we introduce Directional Sheaf Hypergraph Networks (DSHN), a framework integrating sheaf theory with a principled treatment of asymmetric relations within a hypergraph. From it, we construct the Directed Sheaf Hypergraph Laplacian, a complex-valued operator by which we unify and generalize many existing Laplacian matrices proposed in the graph-and hypergraph-learning literature. Across 7 real-world datasets and against 13 baselines, DSHN achieves relative accuracy gains from 2% up to 20%, showing how a principled treatment of directionality in hypergraphs, combined with the expressive power of sheaves, can substantially improve performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the absence of a directed Sheaf Hypergraph Laplacian in the literature by proposing a novel definition, together with the introduction of corresponding Directed Hypergraph Cellular Sheaves and a Directional Sheaf Hypergraph Network. The authors motivate their approach by noting that existing work fails to properly address node classification in directed hypergraphs, particularly under heterophilic conditions. They evaluate the proposed model on hypergraph datasets with varying levels of homophily and further assess its effectiveness in a synthetic experimental setting.

### Strengths
- The work addresses a problem that was not thoroughly investigated in the literature before, providing a theoretically interesting solution.
- The proposed method is well justified theoretically, with several proofs regarding the properties of the newly defined Laplacian.
- The motivations, goal and method are overall clear. 
- The study on the effects of the parameter $q$ in Fig. 2 provides a nice and easily understandable justification for its introduction.

### Weaknesses
A major concern is the lack of an introduction to the Sheaf Hypergraph Network (SHN) [1] method, along with a comparison to your proposed method, in the main text. Strong claims are made in the introduction, stating that SHN does not actually possess some of the properties claimed by its authors, and that the proposed approach addresses this issue. However, in the main text there is no direct reference - even at an intuitive level - to these claims or to how the new method compares to SHN.

Additionally, the introduction of a sheaf structure generally adds computational and runtime overhead compared to a simple prediction model, so its inclusion should be well justified in two respects: (1) whether the additional cost is outweighed by the gain in prediction accuracy, and (2) whether there is a real practical need for a sheaf-based model. Regarding (1), a discussion of the computational cost, ideally comparing the model with non-sheaf baselines, is missing. Regarding (2), I refer to the absence of experiments on natively directed hypergraph datasets, since the considered ones are artificially derived from existing directed graph benchmarks. I understand that this limitation may be difficult to address experimentally due to time constraints, but it would still be helpful to provide some discussion about the availability of such datasets and whether they have been used in similar experimental settings, to clarify the data landscape and the actual necessity for directional hypergraph methods.

[1] Duta et al., Sheaf hypergaph networks, 2023

### Questions
**General concerns**
- Section 3.3 provides solid theoretical results, but two elements are missing: (1) a discussion of the practical meaning of these properties, how they relate to the model’s behavior when used to define a convolutional architecture, and (if possible) a connection to the experimental results; and (2) a comparison to SHN, as mentioned above, to explicitly support your claims.
- Could you be more specific regarding the novelty of the objects you use in your definitions? For example, it is not clear to me whether the object $\mathcal{S}^{(q)}$ represents a novel contribution or not.
- What is the computational and runtime overhead associated to your model with respect to SHN and other non-sheaf-based hypergraph networks?
- In the experimental section, there is no introduction or citation of some of the most relevant baselines (e.g., GeDi-HNN, DHGNN).

**Structure of the paper**
- Looking at the overall structure, I would argue that too little space is dedicated to properly introducing the background and previous methods that are central to the work (e.g., SHN), while a large portion focuses on detailed derivations of various sheaf objects. I suggest moving some of these explicit equations to the Appendix and keeping more intuitive explanations in the main text.

**Notation**
- I recommend to define more thoroughly the notation used in Section 2, for example specify how $| . |$ is used in different contexts, and unify the notations $\delta_e$ and $\delta(e)$.

### Soundness
3

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
2

### Summary
The paper introduces the concept of directed hypergraph cellular sheaves, extending sheaf neural networks to directed hypergraphs. The paper proposes a directed sheaf hypergraph Laplacian, a principled spectral operator. Finally, the paper introduces DSHNN, combining sheaf theory with directional information, enabling strong performance on homophilous/heterophilous graphs.

### Strengths
### Strengths

- **Interesting technical problem.** Extending sheaf neural networks to directed hypergraphs is an interesting technical problem.
- **Principled operator design and theoretical foundations.** The work provides a directed sheaf–hypergraph Laplacian with principled spectral properties and a unifying view that recovers prior (graph/hypergraph/magnetic) operators as special cases. The theoretical results give a well-posed basis for diffusion/convolution with per-incidence maps and directional phases.
- **Empirical competitiveness within the sheaf/hypergraph family.** The proposed models outperform or match prior sheaf/hypergraph baselines on a range of benchmarks.

### Weaknesses
### Weaknesses

- **High-level intuition is hard to understand.** For a reader unfamiliar with sheaf neural networks, I struggled to understand some of the motivation and would appreciate if the paper expanded on the intuition for the approach. One example of where I think expanding on intuition would help is why sheaves mitigate oversmoothing, oversquashing, and heterophily. This would help the reader appreciate the goal to extend them to directed hypergraph case. Another is if the paper could expand at a high level why the theoretical results are important. Why is it necessary that DSHN satisfies all the proven properties? Though each of these results are sound and clearly presented, I struggled to understand why each of these are desirable to have.  
- **Empirical setup and baselines.** I am not convinced by the decision to process datasets such as Cora, Chameleon, and Squirrel as directed hypergraphs. What does this star-expansion gain over modeling the dataset as an undirected graph (how the datasets are usually processed)? Additionally, if the authors are aware of or could evaluate on datasets with native directed-hypergraph representations (where one-to-many interactions are intrinsic), this would strengthen the empirical evaluation. As it stands now, the datasets evaluated on are usually processed as undirected graphs and I'm unsure what the issue is with these standard ways of processing them. Finally, I would appreciate if the authors could justify when to use DSHN over standard MPNNs and MPNNs designed for heterophily. Currently, there is no discussion of or empirical comparison against against strong heterophilous GNNs (e.g., H2GCN, GCNII). I think these should be discussed and ideally evaluated against since much effort has been dedicated to these types of GNNs for homophilous and heterophilous datasets.

### Questions
My questions are integrated with the weaknesses section.

### Soundness
3

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
3

### Summary
This paper extended Sheaf Neural Networks to work with directed hypergraphs. The authors define Cellular Sheaves for directed hypergraphs and then the Laplacian matrix associated with it. The authors show that the Laplacian matrix associated with directed sheaf hypergraphs has some favourable properties such as being diagonalizable with real eigenvalues, positive semidefinite, and has bounded spectrum. The authors also show that the new Laplacian matrix generalizes several previously considered special cases, including the classical graph Laplacian, the Magnetic Laplacian for directed graphs, and some existing hypergraph Laplacians. By parameterizing the heat diffusion induced by the new Laplacian operator, the authors obtain a neural network architecture which they call Directional Sheaf Hypergraph Network (DSHN). The authors show that DSHN outperforms several baselines over both real-world and synthetic datasets.

### Strengths
The Laplacian operator for directed sheaf hypergraphs is new and generalizes several previously considered special cases (but the extension is straightforward).

The experiments are reasonably comprehensive and well presented. Compared with existing baselines, overall there is a notable improvement in accuracy. Figure 2 clearly demonstrates the benefits of taking directionality into account and having a hyperparameter q which adjusts the level of directionality to be used.

Overall, the paper presents a simple yet seemingly effective idea to solve node classification tasks on directed hypergraphs.

### Weaknesses
The presentation and clarity can be improved. In particular, for readers who are not very familiar with cellular sheaves and Sheaf Neural Networks, the notations are a bit heavy and difficult to parse. The authors should try not to overload the text with unnecessary and complex notations. Maybe using different fonts for matrices/vector spaces/functions/maps could help. The choice of notations should be clearly stated at the beginning.

(This is not necessarily a weakness of this paper, but my rating is definitely affected by this) For several years I have been reviewing papers that propose new GNNs to solve node classifications tasks on standard benchmarks, and over time I start to get bored at reading papers that aim to improve node classification accuracy on standard benchmarks only. Indeed, with new methods like the one introduced in this paper, you can improve the performance on benchmarks. But really, how many of these new methods are practically relevant at all (i.e. being implemented in practice to solve actual tasks)? The benchmarks are so poor that many of the node features are still bag-of-words one-hot encodings. Even if a new method performs better on these benchmarks, I am not sure how well this will generalize to practical settings, where you can at least use a language model to generate much better features from raw texts. Because of this, I think the practical relevancy of this work is relatively low. Also see the recent position paper [1]. This work would have been great if the authors also develop some new benchmarks that are more aligned with the current ML practice (e.g. use better features).

Refs:\
[1] M. Bechler-Speicher et al. Position: Graph Learning Will Lose Relevance Due To Poor Benchmarks. ICML 2025

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
2
