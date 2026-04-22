# Bridging Input Feature Spaces Towards Graph Foundation Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 10

## Abstract
Unlike vision and language domains, graph learning lacks a shared input space, as input features differ across graph datasets not only in semantics, but also in value ranges and dimensionality. This misalignment prevents graph models from generalizing across datasets, limiting their use as foundation models.
In this work, we propose ALL-IN, a simple and theoretically grounded method that enables transferability across datasets with different input features. Our approach projects node features into a shared random space and constructs representations via covariance-based statistics, thus eliminating dependence on the original feature space. 
We show that the computed node-covariance operators and the resulting node representations are invariant in distribution to permutations of the input features. We further demonstrate that the expected operator exhibits invariance to general orthogonal transformations of the input features.
Empirically, ALL-IN achieves strong performance across diverse node- and graph-level tasks on unseen datasets with new input features, without requiring architecture changes or retraining. These results point to a promising direction for input-agnostic, transferable graph models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ALL-IN, an input-agnostic framework for transferring graph models across datasets with *different node and edge feature spaces*. It enforces both distributional and orthogonal invariance, supported by theoretical and empirical results. Experimental results show that a single instance of ALL-IN, jointly pre-trained on multiple heterogeneous datasets, performs competitively with per-dataset specialists and generalizes well to unseen datasets.

### Strengths
- Provides a clear and compelling principle for bridging feature spaces by operating on node–node covariance operators, which are independent of the original feature dimensionality. This method  integrate naturally into standard GNN layers.
- Introduces an interesting approach to achieve permutation and orthogonal invariance, effectively addressing a key bottleneck in building transferable graph foundation models.
- The method is simple and architecture-agnostic, making it easy to follow and utilize

### Weaknesses
- The use of  $n \times n$ covariance operators incurs $\mathcal{O}(n^2)$ memory and computation cost per operator. The paper could benefit from a more in-depth discussion of this limitation, ideally supported by experiments on larger-scale datasets, such as [OGB](https://ogb.stanford.edu/docs/nodeprop/).
- The current transfer setup is skewed toward small, homophilous benchmarks (e.g., MUTAG/PROTEINS contain fewer than 1,000 samples). Evaluating the method on larger-scale benchmarks would help demonstrate its viability as a foundation model.
-  While the method outlines an impact on the edge-feature processing pathway, it remains unclear whether this component was utilized in the experiments. Concrete results or ablation studies to support its effectiveness are missing.

### Questions
- Could the authors provide guidance or heuristics—possibly theory-backed—for selecting the projection dimension $h$ and number of propagation orders $k$?
- How does ALL-IN perform under strong heterophily benchmarks? Are the covariance operators dominated by local homophily, and do the propagated operators $K^{(p)}$ help mitigate or worsen this effect?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces ALL-IN, a method for cross-dataset transfer in graph learning that circumvents incompatible node feature spaces by projecting features into a shared random space and building representations from covariance-based statistics. The authors provide theory showing that the resulting node-covariance operators and representations are distributionally invariant to feature permutations, and that the expected operator is invariant to orthogonal transformations, which removes dependence on the original feature semantics, scales, and dimensionality. Practically, the approach requires no architecture changes or retraining and supports both node- and graph-level tasks on unseen datasets with new feature sets.

### Strengths
1. Study the important problem of feature space heterogeneity for GFMs.
2. Provide theoretical understanding of the proposed method.

### Weaknesses
1. The pretraining process of the proposed method is unclear. The manuscript does not clearly specify the pretraining objective, the supervision signals used, or how equation 8 aligns with the description in Section 5.1.
2. The data split for pretraining v.s. finetuning is confusing. The pretraining set has feature dimension up to 28, while the three citation datasets used for finetuning have hundreds or thousands of features. Also, the three citation networks have much more nodes and edges than the pretraining graphs. The rationale for this mismatch is not provided, and the effect on transfer is not analyzed.
3. In addition to bullet 2, the authors mention that the pretraining corpus is drawn from multiple domains, yet citation networks are not included despite being used for evaluation. This leads to some confusion about the choice of the pretraining datasets. I would appreciate if the authors can provide some discussion about the principle of dataset selection for pretraining.
4. The number of the downstream datasets is limited in the main text. Results focus on citation and bioinformatics datasets, with little evidence from other domains represented in the pretraining corpus. I would love to see the experimental results from domains other than citation networks and bioinformatics.
5. Table 12 reports results on heterophilic graphs, but the comparison lacks specialized supervised models that are designed for heterophily, which limits the strength of the claim on this setting.
6. The authors provide the complexity analysis in Appendix C.10. Given the quadratic complexity, it would be better to provide a comparison of the training and inference latency with other GFMs, if applicable.

### Questions
1. What is the pretraining task and loss? Are node classification and graph classification used in pretraining? According to section 5.1, seems that the model is also trained to perform regression. What is the purpose of including regression tasks in the supervised pretraining if the model is not tested for regression during the downstream validation?
2. Why are supervised tasks chosen for pretraining? Can the authors provide a minimal comparison between supervised and unsupervised pretraining on the proposed method to clarify the underlying principle of the choice of objective?
3. Can the method support edge level tasks such as link prediction? If so, what changes are needed in the representation or the predictor?

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
4

### Summary
This paper tackles a critical problem for graph models: graph datasets have inconsistent node features (like different dimensions or meanings), which stops a model trained on one dataset from working on another. The authors propose ALL-IN, a method that solves this by projecting features into a shared random space. Instead of using the original features, it computes a node-covariance matrix that captures node similarities. The GNN then uses this matrix to learn, making the model independent of the original, messy feature space. This allows ALL-IN to generalize to new, unseen datasets with completely different features, without needing to be retrained.

### Strengths
1. The paper correctly identifies input feature heterogeneity as a fundamental and persistent obstacle to cross-dataset generalization, and its focus on solving this core problem is a valuable contribution to the field.
2. The proposed ALL-IN method is supported by theoretical proofs demonstrating that its representations are robust to feature re-ordering (distributional invariance) and basis changes (orthogonal invariance), ensuring it captures stable properties of the data.
3. The model shows strong empirical results, not only maintaining competitive performance when jointly trained on diverse datasets but also, more importantly, successfully transferring to new, unseen datasets with entirely different features without retraining.

### Weaknesses
1. On the domain difference. The authors should explain how the domain-specific knowledge is modeled and preserved in ALL-IN. 
2. Another main concern is the computational complexity, as a general-purpose foundation model is usually trained on massive data at scale. Given that the covariance is expensive, complexity analysis and comparison against other GFMs or pre-training strategies are strongly encouraged.
3. This paper lacks some recent advances on GFM, and it will be more interesting if the following papers are well discussed and compared.  
a) NeurIPS24: GFT: Graph Foundation Model with Transferable Tree Vocabulary  
b) WWW25: SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation  
c) WWW25: Riemanngfm: Learning a graph foundation model from structural geometry   
d) OpenGraph: Towards open graph foundation models.   
e) ICML25: How Much Can Transfer? BRIDGE: Bounded Multi-Domain Graph Foundation Model with Generalization Guarantees   
f) ICML25: AutoGFM: Automated Graph Foundation Model with Adaptive Architecture Customization  
Also, the authors are required to specify difference between the aforementioned papers to defend their novelty.

### Questions
1. How can the $\mathcal{O}(n^2)$ complexity scale to foundation model-sized graphs? Are the sparse approximations you mentioned practical?
2. How critical is it to sample the projection matrix $C$ at each forward pass? Your theory (Prop 4.1, Thm 4.3) relies on this stochasticity. Would the method's benefits (especially distributional invariance) be lost if $C$ were sampled only once and then fixed throughout all training and inference?
3. The method achieves invariance by reducing features to their second-order statistics. This must inherently discard some information. Could you elaborate on what types of task-relevant information might be lost? For example, how would the model distinguish between two nodes with feature vectors $x$ and $-x$, given that they would contribute identically to the expected covariance matrix $\Pi_{c}XX^{T}\Pi_{c}$?
4. The proposed strategy for handling edge features (projecting, aggregating to nodes, then computing a new covariance operator $K_{edge}^{(p)}$) is not empirically ablated. Was this component used for pre-training on datasets like ZINC, which have edge attributes? How does the simple node-level aggregation avoid losing critical relational information, and can you provide an ablation study on this component's impact?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
Finding a good and general approach for graph foundation models is a relevant, important and actual topic in representation learning for structured data. The present article develops a really interesting approach in this direction, by proposing a novel way to think about misalignement between attributed graph datasets. The idea is to randomly project the features on the nodes and to compute the node-covariances operators though these projections. This ensures that key properties of the features are kept (e.g., strong relations between the features of some nodes) while providing a representation of these features which is invariant to permutations and more generally orthogonal rotations, of the features, and also which has the same dimension whatever the initial dataset.

The key idea in this work is very clever. The section 3 first details the method using this random projection and the concept of node-covariance operators, and secondly proposes a novel node representation method by using these elements inside a GNN (which encodes the structure, while the random features and node-covariance operators encode the attributes,  possibly supplemented to structural encodings if one wants). Then Section 4 proposes a thorough theoretical analysis of the key properties: invariance of the representation under orthogonal rotations ; cases of distinguishability ; proof that the training provides an upper bound of the training objectives formulated ; some elements on the transferability of the representation. All that are really good elements.

Numerical experiments follow and they are solid. The reader basically will agree to the steps and the conclusions obtained by the authors. The obtained performance are good, both as general purpose representions (Table 1), and for transfer to new datasets (Tables 2 and 3).

### Strengths
The strengths of the article are : 

1. A very good and novel proposition to build a transferable, general and learnable representation of attributed graphs with consistency across datasets, and transferability. Hence, this work is a very good step toward an efficient graph foundation model.

2. There are key theoretical insights, well proven in the article (or its Appendices) that show the impact and the importance of the various elements. 

3. Numerical experiments and conducted on several situations. The results are compared to a variety of state-of-the-art methods for graph learning (some supervised baselines ; some GFMs using LLM ; some GFMs built on top of GNN like the present method). The  complementary studies about the separability (linear or non-linear) of the obtained representation, and the ablation study are good also (for ablation: to study the impact of structural and positional encoding ; the impact of random projections ; the impact of using covariance operator ; of propagating it ; 	and so on). This is really a solid work. 

4. The writing of the article is globally exemplar.

### Weaknesses
I did not find any weakness in this paper.

### Questions
I have some questions or suggestions:

The step of equation (1) could be stressed a little bit more, possibly writing already here some insight about why you propose that. Also, could it be generalised to other types of random projections, or of sketching ?

* For theorem 4.5: I would put the remark in parentheses ("this holds,… ") outside of the theorem. Possibly just underneath.

* In 5.1 : it appears to be only 9 datasets, while D.1 writes 10 (we only have the results for 9 in the paper).

* Table 1: the order of the data is not natural. It would be easier to have it either ordered so that data with the same metric are adjacent, or ordered in the same manner than in the text (ModelNet should then be last). Still, having dataset where smaller is better first (ZINC, MOLSOL), then ROC-AUC (MOLHIV, MOLTOX21) and finally datasets measured with ACC (MOLHIV and the others) would be nice.

* Do you have a comment to compare the proposed ALL-IN to SCORE which appears to be a good competitors. Do you think that ALL-IN can be generally better to transfer in unseen graphs (as in Table 3) or do you currently know no more ?

* page 23, third bullet point: $+ S$ should be $ \oplus S $, shouldn't it ?

### Soundness
4

### Presentation
4

### Contribution
4
