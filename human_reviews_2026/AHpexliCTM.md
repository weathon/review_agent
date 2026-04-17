# Cooperative Sheaf Neural Networks

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Sheaf neural networks (SNNs) leverage cellular sheaves to induce flexible diffusion processes on graphs, generalizing the diffusion mechanism of classical graph neural networks. While SNNs have been shown to cope well with heterophilic tasks and alleviate oversmoothing, we show that there is further room for improving sheaf diffusion. More specifically, we argue that SNNs do not allow nodes to independently choose how they cooperate with their neighbors, i.e., whether they convey and/or gather information to/from their neighbors. To address this issue, we first introduce the notion of cellular sheaves over directed graphs and characterize their in- and out-degree Laplacians. We then leverage our construction to propose Cooperative Sheaf Neural Network (CSNN). Additionally, we formally characterize its receptive field and prove that it allows nodes to selectively attend (listen) to arbitrarily far nodes while ignoring all others in their path, which is key to alleviating oversquashing. Our results on synthetic data empirically substantiate our claims, showing that CSNN can handle long-range interactions while avoiding oversquashing. We also show that CSNN performs strongly in heterophilic node classification and long-range graph classification benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper argues that standard Sheaf Neural Networks (SNNs) on undirected graphs cannot realize node-level cooperative behavior (separating LISTEN vs PROPAGATE), because silencing incoming messages at a node also suppresses its outgoing influence.

### Strengths
### Clear motivation
* The limitation of undirected-sheaf diffusion for cooperative behavior is crisply identified and formalized. Proposition 3.1 explicitly shows PROPAGATE -> LISTEN coupling in classical SNNs; the directed-sheaf construction is a natural remedy

### Empirical breadth on heterophily
* Across 11 node-classification benchmarks (including the cleaned Squirrel/Chameleon), CSNN often outperforms both cooperative GNNs and prior SNNs

### Implementation details
* Parametrization for orthogonal maps, per-node conformal scaling, and a complexity discussion point to a careful engineering effort

### Weaknesses
### Novelty and Positioning
* Treating undirected edges as two directed edges plus per-node source/target maps is a natural extension but arguably an incremental one within the sheaf literature. The paper acknowledges related notions (e.g., quiver Laplacians), but the novelty boundary versus prior directed/sheaf constructions (and vector bundle variants on directed graphs) is not fully pinned down

### Theoretical Analysis
* Proposition 4.3 is an existence proof: there exist restriction maps for clean long-range, path-selective propagation. The paper does not analyze whether gradient-based training reliably finds such configurations in realistic, noisy data
* The paper notes their directed-sheaf Laplacians can have complex eigenvalues with negative real parts (unlike PSD Laplacians). Yet the stability of the discrete diffusionis not analyzed
* The model still performs local diffusion; the 2-hop per layer can accelerate reach but may also accelerate oversmoothing. There’s no theorem bounding oversmoothing for CSNN or showing improved expressivity beyond MPNN limitations

### Writing Clarifications
* A derivation/intuition for why this specific normalization (scalar multiples of identity under conformal maps) is preferred (vs. other block scalings) would help
* Regarding Proposition 4.3 
  * The constructive path-only propagation is compelling but assumes the ability to zero out many maps across layers. In practice, with shared parameters and noisy optimization, how often does training approximate this regime?

### Questions
Please see the above weaknesses

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
This paper extends Sheaf Neural Networks, a class of GNNs that use cellular sheaves to generalize message passing, to support cooperative communication between nodes. Classical SNNs effectively address heterophily and oversmoothing, but they lack the flexibility for nodes to independently choose how to exchange information (i.e., whether to propagate, listen, or isolate).
To overcome this, the authors propose the Cooperative Sheaf Neural Network (CSNN), which introduces directed cellular sheaves with separate in- and out-degree Laplacians, enabling asymmetric and selective communication.

### Strengths
Clear theoretical guarantees linked to long-range neighbors and over-squashing.

Empirical performance is compelling

### Weaknesses
Scalability of sheaf-based models on large-scale graphs remains to be tested.

### Questions
1. This paper seems to be an A+B work, with a combination of sheaf GNN + cooperative GNN, and some extension on directed graphs. And it is not well motivated or does not have some special designs for heterophily and over-squashing.

2. Why does the sheaf neural network have to achieve cooperative behavior?

3. "our model has the ability to reach longer distances." It is no surprise when you multiple your in-degree and out-degree sheaf Laplacian together. However, it is found that long-range information is harmful in many heterophilic datasets [1].

4. Missing comparison with some baseline models, e.g. ACMGCN [2], FSGNN [3], GloGNN [4]. More tests on malignant and ambiguous heterophilic datasets listed in [5]. Experiments on large scale datasets used in [4].




[1] Less is More: on the Over-Globalizing Problem in Graph Transformers. In Forty-first International Conference on Machine Learning 2024.

[2] Revisiting heterophily for graph neural networks. Advances in neural information processing systems. 2022 Dec 6;35:1362-75.

[3] Simplifying approach to node classification in graph neural networks. Journal of Computational Science, 62, 101695.

[4] Finding global homophily in graph neural networks when meeting heterophily. In International conference on machine learning (pp. 13242-13256). PMLR.

[5] The heterophilic graph learning handbook: Benchmarks, models, theoretical analysis, applications and challenges. arXiv preprint arXiv:2407.09618. 2024 Jul 12.

### Soundness
3

### Presentation
3

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
The work proposes a new Sheaf Neural Network model, called Cooperative Sheaf Neural Network (CSNN), with the goal of combining the benefits of Sheaf Neural Networks in tackling oversmoothing in GNNs and handling heterophilic data, with the property of performing selective communication between nodes, which is expected to lead to cooperative behaviour and avoid oversquashing problems. The model and its benefits are validated through synthetic and real-world experiments.

### Strengths
- The motivation, goal, and methodology are well formulated.
- The paper reads well, it has a nice structure, and the necessary background knowledge is well presented.

### Weaknesses
There are strong claims regarding (1) consistently outperforming SNNs and cooperative GNNs (both in introduction and in the experiments, see Results in Section 6.2), and (2) about not soccumbing to oversquashing (abstract). Looking at the results, it doesn't appear to improve substantially with respect to the competitors, so I would advise to reconsider the strength of the claims, which may lead to high expectations in the experiments.

Additionally, the introduction of a sheaf structure generally adds computational and runtime overhead compared to a GNN model, so its inclusion should be well justified in two respects: (1) whether the additional cost is outweighed by the gain in prediction accuracy, and (2) whether there is a real practical need for a sheaf-based model. Regarding (1), a discussion ideally comparing the model with non-sheaf baselines, is missing in the main part of the paper. Regarding (2), I would expect a convincing discussion of the advantages over other SNN methods in preventing oversquashing (due to the additional cooperative component), and over cooperative GNNs in mitigating oversmoothing and handling heterophilic datasets (thanks to the additional sheaf structure). These advantages are not evident from the experiments, as the relevant comparisons are either missing or the results/discussions are not sufficiently convincing.

### Questions
**General concerns** 
- In your introduction, you mention that selective communication is a desirable property for SNN in order to tackle oversquashing. Recently, Nonlinear Sheaf Neural Networks have shown a similar behavior, selectively exploiting information of neighbors in complex node interactions [1]. Do you think there may be a connection between your method and the employment of a nonlinear Laplacian? 
- After definition 2.3: "...however, when they publicly discuss this topic, they may prefer to not manifest their true opinion. ". In my understanding, since the edge stalks may be different from the node stalks, the individuals don't necessarily need to discuss the topics of the private opinion spaces. The topics in the public agreement space may also be different.
- It is not clear how your result in Proposition 4.3 relates to your discussion on the oversquashing behavior relying on the definitions in [2] and [3], which you reference after Example 4.4. These works formalize oversquashing as a bound on the Jacobian, as you also state, and this bound is strongly influenced by the presence of $A_{i,j}^{(t)}$. In Example 4.4, although you show that at each layer one node is influenced by only a single other node, when computing the Jacobian the update matrices $T$ and $S$ will still accumulate in the product across multiple layers. So, there is still a product of $O(t)$ matrices in the bound for the Jacobian. Wouldn’t this have the same effect as including the term  $A_{i,j}^{(t)}$? Could you please elaborate on this point, perhaps by explicitly showing how your method improves this bound compared to a sheaf model that does not perform selective communication?

**Experiments**
- It would be useful to have in Table 1 and 2 an homophily measure for each dataset, to intuitively understand the setting.
- I would recommend to rephrase the claims regarding the results. For example, in the "Results" of section 6.2, you state "We note CSNN often outperforms both NSD and CO-GNN by a significant margin", although looking at the results of Table 1 and 2, and considering the +/- confidence bound, the improvements are not as significant as stated. 
- As mentioned in the weaknesses, I believe the experiments would benefit from a direct comparison between the proposed method and its non-sheaf (CO-GNN) and non-cooperative (NSD/BuNN) counterparts - for example, by including these models in the synthetic experiments of Section 6.1 and adding CO-GNN in Table 2 to provide a more complete comparison and a clearer demonstration of the claims.

**Writing/Typos**
- In related works, in the paragraph related to Cooperative GNNs: "... that chooses the cooperation the action each node takes". There seems to be an error in the structure of this sentence.
- In Table 2, last column, the second-best result is not highlighted with grey color.



[1] Zaghen et al., Sheaf Diffusion Goes Nonlinear: Enhancing GNNs with Adaptive Sheaf Laplacians (2024)

[2] Di Giovanni et al., On over-squashing in message passing neural networks: The impact of width, depth, and topology. (2023)

[3] Topping et al., Understanding over-squashing and bottlenecks on graphs via curvature (2022)

### Soundness
2

### Presentation
3

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
The authors propose an approach for learning cooperative behavour between nodes, by treating undirected edges between nodes as pairs of directed edges and introducing directed sheaf neural networks to effectively deal with them.

### Strengths
The work is well motivated, addressing a clear limitation of sheaf neural networks in modeling cooperation patterns between nodes.

The proposed solution has solid theretical grounds.

Synthetic results clearly confirm the ability to mitigate oversquashing, and an extensive evaluation on real-world datasets shows competitive performance wrt the state of the art.

### Weaknesses
Experimental results on the classical datasets for heterophilic analysis (Table 2) show very marginal improvements (considering the high variance, most likely none of these is significant). This is not a novelty, and questions the appropriateness of these datasets as benchmarks (as Platonov et al already pointed out). I encourage the authors to briefly discuss this aspect, so as to direct further research towards more appropriate evaluation benchmarks.

### Questions
Is it possible to have a high-level illustration clarifying the problem with plain SNN and the advantage of CSNN? This would help interested readers not familiar with the math behind SNN to gather the intuition behind the approach.

### Soundness
4

### Presentation
3

### Contribution
4
