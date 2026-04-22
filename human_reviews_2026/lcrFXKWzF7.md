# $\textit{All the World's a Sphere}$: Learning Expressive Hierarchical Representations with Isotropic Hyperspherical Embeddings

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
Most existing embedding frameworks rely on Euclidean geometry, which, while effective for modeling symmetric similarity, struggle to represent richer relational structures such as asymmetry, hierarchy, and transitivity. Although alternatives like hypercubes and ellipsoids introduce containment-based semantics, they often suffer from axis-aligned rigidity, anisotropic bias, and high parameter overhead. To address these limitations, we propose SpheREx ($\textbf{Sphe}$rical $\textbf{R}$epresentations for Hierarchical $\textbf{Ex}$pressiveness), a geometric embedding framework that utilizes isotropic hyperspheres for hierarchical and asymmetrical relation representation. By representing entities as hyperspheres, SpheREx naturally models containment, intersection, and mutual exclusion while maintaining rotational invariance and closed-form inclusion criteria. We formally characterize the geometric and probabilistic properties of hyperspherical interactions and show that they capture desirable logical structures. To ensure stable optimization and prevent uncontrolled radius growth, we introduce a volume clipping and radius regularization strategy tailored for asymmetric tasks. We conduct extensive evaluations across four diverse real-world benchmarks, spanning both text and vision modalities. SpheREx consistently outperforms twelve competitive baselines, achieving statistically significant improvements across key evaluation measures. Ablations supported by qualitative analysis across benchmarks demonstrate the efficacy of hyperspheres over state-of-the-art geometric baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SpheREx, a geometric embedding model that represents each entity as a closed hypersphere (center + radius) in latent space. Relations such as containment, overlap, and disjointness are captured through simple geometric inequalities and a probabilistic overlap term. The framework unifies symbolic and data-driven reasoning across tasks including taxonomy expansion, knowledge graph completion, semantic similarity, and multimodal alignment. Experimental results show moderate improvements over box, cone, and hyperbolic embeddings.

### Strengths
1) the use of closed balls ensures isotropy and rotation-equivariance with low parameter cost.

2) Unified formulation across multiple relation types (containment, overlap, exclusion) and tasks.

3) Empirical evaluation is broad, spanning text, KG, and vision–language settings, with proper significance testing.

### Weaknesses
1) The core concept (representing entities as spheres/balls) is well established in prior work (e.g., EL Embeddings, N-ball embeddings, TranSHER). The differences, dual-ball relation templates and probabilistic loss, are modest extensions rather than conceptual innovations.

2) The claimed issues of box embeddings (orientation sensitivity, parameter inefficiency, local identifiability) are overstated and already mitigated in probabilistic and Gumbel-Box variants. The paper does not empirically demonstrate these weaknesses.

3) The “radius ratio” used as a proxy for inclusion probability is mathematically coarse. Actual intersection volume scales with r^d, not linearly with r. The model never validates that this surrogate correlates with true probabilistic inclusion. Using an “auxiliary intersection sphere” is not a true geometric intersection and can overestimate or underestimate overlap, particularly for near-tangent regions. No analysis quantifies the approximation error or its downstream effect.

4) Performance gains (2–5%) over strong baselines like GumbelBox and ConeE are modest and may result from regularization rather than the proposed geometry. There is no statistical or theoretical justification for superior expressiveness.

### Questions
1) How sensitive is performance to the radius-based probabilistic approximation? Does it correlate with true intersection volume?

2) Can the authors demonstrate a failure case for Gumbel-Box or Cone embeddings that SpheREx resolves?

3) Why is rotational invariance critical for the evaluated tasks, and can this benefit be empirically isolated?

4) Does the model generalize to non-tree or cyclic ontologies, where strict containment breaks down?

### Soundness
2

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
4

### Summary
This paper presents a new embedding framework for learning expressive hierarchical representations. The core idea is to embed entities as isotropic hyperspheres. Through experiments on a variety of tasks and datasets, the proposed SpheREx shows the state-of-the-art performance against several baselines.

### Strengths
I like the introduction paragraph where the authors use examples of real companies (Amazon, Alibaba, Pinterest) to motivate hiarachical structures. The visualization in Figure 1 is clear.

Source code and datasets are provided as supplementary and the authors commit to releasing them upon acceptance. 

The research idea and method part looks good to me, and the proposed SpheREx can achieve state-of-the-art results on a variety of datasets. The authors also provided statistical tests as a verification.

### Weaknesses
I believe this research is insightful, but the current manuscript is below the acceptance threshold from my perspective, due to the followings. Revisions are needed, and discussions in the rebuttal period are welcome.

1. There are several closely related works [1, 2, 3] from my perspective that are missing discussions. Specifically, the SpherE paper [2] introduces a similar core idea in knowledge graph settings. Also, discussions on hyperbolic embeddings considering hierarchical relations are missing [4, 5, and maybe more]

[1] Spherical Text Embedding. Meng et al. NeurIPS 2019.
[2] SpherE: Expressive and Interpretable Knowledge Graph Embedding for Set Retrieval, Li et al. SIGIR 2024.
[3] Sphere Embedding: An Application to Part-of-Speech Induction. Maron et al. NeurIPS 2010.
[4] Poincaré Embeddings for Learning Hierarchical Representations. Nickel et al. NeurIPS 2017.
[5] Hyperbolic Representation Learning: Revisiting and Advancing. Yang et al. ICML 2023.

2. (Hyper)Ellipsoids contribute a lot to the development of hyperspherical embeddings. The authors proposal to use isotropic hyperspheres as simplified hyperellipsoids and rejects ellipsoids due to parameter complexity. However, in practice, we can reduce the embedding dimension to make ellipsoids acceptable. I wonder if experiments using ellipsoids could be conducted to improve this part. 
(1) Will there be a small embedding dimension such that both hyperellipsoids and hyperspheres are runnable, and what will the performance be like? 
(2) Ellipsoids, as a generalized form of the proposed SpheREx, holds the potential to outperform SpheREx with the same embedding dimention.
(3) What is the number of parameters and complexity of using ellipsoids rather than spheres in theory and in practice given a specific embedding dimension? So that we can decide if the marginal gain of using ellipsoids is acceptable or not compared to the marginal cost of using ellipsoids.

Also, in Figures 6 and 8, the embedding dimension is actually smaller than I expected. I assume ellipsoids should be applicable?

3. How would SpheREx perform in larger embedding dimensions? (e.g., embedding dimension for modern LLMs is usually >=512)

4. Many hyperparameter studies are missing. There are many hyperparameters in Table 6-9, but the only hyperparameter study I see is Figure 9, on QQP dataset only and on varying thresholds only.

5. Could there be any ablations to show that the design components from sections 3.1 to 3.4 all contribute to the final performance? Or, please explain why SpheREx acts as a whole and is not composible.

6. The baselines being compared are quite outdated (most of them are before 2020). There are many new text encoders after BERT that could be compared with. 

7. The number of model parameters, training time, hardware, and GPU usage are not reported (or I missed somewhere).

8. From my perspective, there is no deeper understanding of how SpheREx works in practice beyond the overall performance numbers. I wonder if it would be possible to visualize (or maybe through other ways) some toy examples in 2d or 3d space, along with other baselines, to show that SpheREx indeed gives more expressive embeddings given the context of the task it is doing. Or in other words, if it is possible to provide some cases where SpheREx succeeds but other baselines fail.

### Questions
1. The descriptions of experiment tasks and datasets are not sufficient. I have no idea what SEMEVAL16 is about or what the scale of it is.

2. MovieLens is already quite a mature dataset. Why bother to construct a new MovieLens? This might also be applied to QQP.

3. Why does BERT perform so badly in Table 1 as a pre-trained model with a large number of parameters?

### Soundness
2

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
The authors present a method that uses isotrophic hyperspheres to embed hierarchical knowledge.
this could be seen as a simplification from earlier methods that used hyperellipsoids.

### Strengths
* There is an evaluation with multiple tasks. This is really important, as compared to papers that just present one (perhaps cherry picked) task.  
* For a fair share of the tasks, the results are pretty impressive
* The paper is overall well written and clear. It presents a rather simple idea in an elegant way.

### Weaknesses
* You claim that "axis-aligned geometric representations, such as boxes, are prone to high parameter complexity, orientation sensitivity, and local identifiability issues, wherein small changes to the parameters result
in invariant model behavior, leading to ambiguous gradients", without supporting evidence. There is some theoretical support, but it is not at all clear that the preconditions for these hold ion practical settings.

* I miss a comparison with the method introduced in Xiong, B., Cochez, M., Nayyeri, M., & Staab, S. (2022). Hyperbolic Embedding Inference for Structured Multi-Label Prediction. NeurIPS2022. It seems to have pretty similar properties. It also presents containment and disjointedness loss terms.

*Theorem 3 is presented as a strength, but it is a trivial result and if we continue the argument, we can go from hyperspheres to points and then to a null space where we the lowest possible capacity and parameter complexity. 

* The paper suffers a bit for a perceived need to over-complicate things. Some of the proofs are for rather trivial points that are written down with complex symbolic notation. Theorem 1-3 would have been more simple and intuitive to write them down in a less formal style.

* In your experiments, it is not clear to me why you have used different methods for different tasks. Many of them seem to be applicable to all of the tasks.

### Questions
* Why are the compared to methods used for the different tasks different?

### Soundness
3

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
This paper proposes a new embedding framework, SpheREx, that represents entities as isotropic hyperspheres in a latent space. The key idea is to model hierarchical and asymmetric relations (such as taxonomy parent-child relationships or set inclusions) through sphere containment. A smaller sphere contained inside a larger one can signify a subclass or “is-a” relationship, e.g, man <is-a> mammal, so the sphere of man should be inside mammal. 

Unlike prior region-based embeddings like axis-aligned boxes and ellipsoids, hyperspheres offer rotational invariance and lower parameter complexity. The paper provides a theoretical characterization of how spheres can capture logical relations such as containment for hierarchy, overlaps for intersection, and disjointness for mutual exclusivity, with simple distance and radius conditions. To address optimization challenges (like one sphere’s radius growing arbitrarily large), the authors introduce a specialized training regimen with volume clipping and radius regularization to stabilize learning.

### Strengths
The paper introduces hyperspherical embeddings as a new way to capture hierarchical relationships. This isotropic sphere representation is novel in that it combines the idea of region-based embeddings with rotational invariance.

### Weaknesses
1. The paper repeatedly argues that axis-aligned box embeddings have “high parameter overhead,” implying that hyperspheres (with one radius parameter) are inherently more efficient. While it is true that a naive box in $\mathbb{R}^d$ has 2$d$ parameters (min and max per dimension) versus $d+1$ for a sphere, this comparison is not entirely fair. Prior box-based models often employ regularization or tied parameters to effectively reduce complexity, and they can be made compact without sacrificing performance (e.g., by working in a lower-dimensional latent space or adding constraints on edge lengths). A simple modification would be to have box embeddings with the same width on each side.

2. Does rotational invariance help to represent hierarchical relationships or set inclusion? Also, the intersection operation is not faithfully encoded at all. Boxes are closed under intersection, and ellipsoids can approximate intersection closure reasonably. However, with spheres intersection is not closed, the intersection of two spheres is not a sphere. Even the approximation error grows as the dimension grows. Also, how does the method handle the intersection of multiple entities?

3. The authors highlight local identifiability issues in box embeddings – i.e., the problem that many different box parameter settings can yield equivalent overlaps or containments, leading to flat loss regions. However, this is a known issue in the literature, and there has been prior work explicitly aimed at mitigating it. For instance, Dasgupta et al. (2020) propose methods for improving local identifiability in probabilistic box embeddings by using Gumbel random variable-based parameterizations to ensure small parameter changes have observable effects. 

4. The Euclidean distance measure that is used in the paper would provide a concentric bias. A child entity would try to align the center with its parent. The design here would probably result in **vector embeddings with a learned threshold" rather than a true region-based embedding where inside the region the representation is position invariant. Conceptually, one can view the proposed model as a fairly incremental extension of standard vector embeddings. Each entity’s representation in SpheREx is essentially a point embedding (the center $\mathbf{c}$ in $\mathbb{R}^d$) augmented with a single additional parameter (the radius $r$). This “point + scope” formulation is certainly a form of region embedding, but it is a much simpler region than, say, a box that has $2d$ degrees of freedom or an ellipsoid with a full covariance matrix. The paper does not fully convince that this limited form of region is the key to its success.

5. Restricting embeddings to isotropic hyperspheres (one scalar radius for all directions), the model significantly reduces the number of parameters per concept, which the authors argue improves generalization. However, this comes with a classical bias–variance trade-off that the paper does not explicitly acknowledge.

### Questions
Same as weakness.

### Soundness
2

### Presentation
3

### Contribution
2
