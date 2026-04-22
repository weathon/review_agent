# GGBall: Graph Generative Model on Poincaré Ball

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Generating graphs with hierarchical structures remains a fundamental challenge due to the limitations of Euclidean geometry in capturing exponential complexity. 
        Here we introduce **GGBall**, a novel hyperbolic framework for graph generation that integrates geometric inductive biases with modern generative paradigms. 
        GGBall combines a Hyperbolic Vector-Quantized Autoencoder (HVQVAE) with a Riemannian flow matching prior defined via closed-form geodesics. This design enables flow-based priors to model complex latent distributions, while vector quantization helps preserve the curvature-aware structure of the hyperbolic space.
        We further develop a suite of hyperbolic GNN and Transformer layers that operate entirely within the manifold, ensuring stability and scalability.
        Empirically, GGBall establishes a new state-of-the-art across diverse benchmarks. On hierarchical graph datasets, it reduces the average generation error by up to 18\% compared to the strongest baselines.
        These results highlight the potential of hyperbolic geometry as a powerful foundation for the generative modeling of complex, structured, and hierarchical data domains. 
        Code is available at: https://github.com/AI4Science-WestlakeU/GGBall.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new generative framework for graph data based on a hyperbolic latent space. The authors motivate the use of hyperbolic geometry by highlighting its theoretical advantages for representing hierarchical and complex graph structures compared to standard Euclidean latent spaces.

The approach embeds graphs into a hyperbolic space through a Vector-Quantized Autoencoder (VQ-VAE) operating within the Poincaré ball model. The paper introduces dedicated architectural components, including a Poincaré Graph Neural Network (GNN), a Poincaré Transformer, and corresponding hyperbolic encoders and decoders that allow mapping graphs to and from the hyperbolic latent space.

The model is evaluated on three datasets containing relatively small graphs.

### Strengths
The paper is well-written and easy to follow, even though it introduces non-trivial concepts from hyperbolic geometry. The theoretical development is solid, clearly presented, and supported by extensive supplementary material.

The idea of leveraging a hyperbolic latent space for graph generation is both neat and original, and the proposed framework represents a meaningful conceptual step forward in the design of geometry-aware generative models.

I would like to particularly emphasize that the theoretical contribution and its presentation are of very high quality. The mathematical rigor and clarity are exemplary.

### Weaknesses
**Empirical Evaluation and Euclidean Baseline**

The paper’s main claim is the superiority of hyperbolic over Euclidean latent spaces for graph generation. Consequently, it is essential to provide clear empirical evidence supporting this claim.
Including an Euclidean VQ-VAE baseline in Table 1 would greatly strengthen the experimental section, as it would directly demonstrate the empirical advantages of the hyperbolic space. Similarly, presenting detailed results and experimental settings for Figure 1 would help assess the validity of the reported improvements.
Since the proposed model is theoretically and empirically more complex, it is important to show that it provides a clear and consistent performance advantage over its Euclidean counterpart, ideally across all reported datasets and experiments, including those in Tables 2 and 3.

**Evaluation Metrics and Protocol**

The evaluation protocol requires clarification.

* *Community-Small*: The results for the HVQVAE+Flow model appear to outperform the training set itself as given in SPECTRE, which should not occur in a properly calibrated evaluation. Moreover, the absence of standard deviations prevents assessment of variability or statistical significance. The paper refers to results from DiGress (which relies on SPECTRE), but the evaluation chain is not made explicit.

* *Ego-Small*: Similar concerns arise here, as the model’s results also systematically outperform the training set as given in DGAE (which seems to be the journal paper of VQGAE) , which appears inconsistent. Clarifying the evaluation setup is crucial to ensure reproducibility and interpretability of the reported findings.

**Evaluation on QM9**
The use of the *novelty* metric on QM9 is problematic. QM9 is an exhaustive enumeration of small organic molecules satisfying specific constraints; thus, generating molecules outside this set does not necessarily indicate successful generalization. This metric is therefore not commonly used for evaluation. It would be beneficial for the authors to justify its inclusion or reconsider its use.
For the same reason, achieving 100% *uniqueness* is not necessarily desirable. 

Therefore, the only meaningful metric in this setting is *validity*, which remains substantially below various baselines.
To provide a more comprehensive evaluation, including additional metrics such as FCD (Fréchet ChemNet Distance) or NSPDK similarity would be highly valuable.

**Graph Size and Dataset Diversity**
The experiments are limited to datasets with small graphs, with a maximum of around 20 nodes (e.g., Community-Small). Although this limitation is acknowledged in the paper, it remains a major limitation, as many contemporary models handle graphs with up to 200 nodes and report results on larger benchmarks such as Planar, SBM, Zinc250K, or Moses.

Including experiments on at least one larger-scale dataset would considerably strengthen the empirical evidence and demonstrate scalability.
This is particularly important because the Community-Small and Ego-Small datasets each contain only around 200 instances, making overfitting likely and reducing the statistical robustness of the results.


--------

**SPECTRE**: Karolis Martinkus, et al.. SPECTRE:
Spectral conditioning helps to overcome the expressivity limits of one-shot graph generators. In Proceedings of the 39th International Conference on Machine Learning. PMLR, 17–23 Jul 2022.
https://proceedings.mlr.press/v162/martinkus22a.html.

**DGAE**: Yoann Boget, et al.. Discrete graph auto-encoder. Transactions on Machine Learning Research, 2024. https://openreview.
net/forum?id=bZ80b0wb9d.

### Questions
The theoretical contribution of the paper is strong and well-founded, but the empirical evaluation does not yet convincingly demonstrate the claimed benefits of hyperbolic latent spaces.

I would encourage the authors to consider the following improvements:

1. Include a systematic ablation comparing hyperbolic and Euclidean latent spaces across all tasks and datasets.
2. Describe the experimental protocol in detail, including data splits, training configurations, and evaluation procedures (possibly in the supplementary material).
3. Expand the empirical evaluation by including larger datasets (e.g. Zinc250, Planar, SBM) and additional performance metrics (e.g., FCD, NSPDK).

Such additions would substantially reinforce the paper’s claims and make the overall contribution more compelling.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a hyperbolic framework for graph generation to address the limitations of Euclidean geometry in capturing hierarchical graph structures. It integrates a Hyperbolic Vector-Quantized Autoencoder with Riemannian flow matching based on closed-form geodesics in the Poincaré Ball model. Empirically, GGBall achieves state-of-the-art performance across benchmarks.

### Strengths
1. The proposed model is a fully hyperbolic graph generation framework using the Poincaré Ball, leveraging its exponential volume growth to naturally preserve hierarchical structures.

2. Combining HVQVAE with Riemannian flow matching (for flexible prior modeling) resolves stability issues of continuous hyperbolic VAEs and enhances generative capacity.

3. It outperforms SOTA baselines across abstract graph generation and molecular graph generation.

### Weaknesses
1. Experiments focus on small/medium graphs (e.g., QM9, small community graphs); performance on large-scale graphs is unproven.

2. While HVQVAE avoids HVAE’s KL issues, the paper does not explore alternative variational formulations to fully leverage hyperbolic probabilistic modeling. In addition, what is a L_degree in Line 273?

3. The Poincaré Ball’s fixed negative curvature may struggle with heterogeneous graphs (mixing hierarchical and non-hierarchical structures), unlike mixed-curvature alternatives not explored here.

4. The loss function has many hyperparameters. How are their values ​​determined? What are their values ​​for different datasets? What are the sensitivity experiments? Won't so many parameters increase the difficulty of hyperparameter tuning, thereby reducing the practical usability of the method? The inability to answer these questions may cause concern for the reader.

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a framework for generating graphs with hierarchical structure termed “GGBall”, consisting of

1. hyperbolic message passing and DiT layers which use
    
    1. message aggregation in the tangent space via log/exp maps (GNN)
        
    2. scale and shift values used for FiLM like message modulation derived from hyperbolic distances  (GNN)
        
    3. value aggregation using Möbius gyromidpoints (DiT)
        
    4. relevancy score calculation leveraging hyperbolic distances on poincare linear layer proejcted q,k values (DiT)
        
    5. hyperbolic auxillary operations (diT): layernorm aggregation in the tangent sapce like the GNN message aggregation, residual connections using möbius addition instead of standard addition and multi-head splitting and concatenation in a hyperbolic geometry perserving fashion
        
2. a novel graph auto encoder leveraging this hyperbolic latent space parametrization, trained to reconstruct node and edge types, regularized by matching degrees consistently + an l2 norm (evaluated in continuous, standard AE,variational AE and quantized VAE form, termed HGAE, HVAE and HVQVAE respectively, where quantized version was motivated by numerically unstable KL divergence for the HVAE)
    
3. a manifold flow matching method taken from [https://openreview.net/pdf?id=g7ohDlTITL](https://openreview.net/pdf?id=g7ohDlTITL)
    

The method is evaluated on community-small, ego-small and qm9, with the interpolation properties of the latent state being studied in particular

### Strengths
1. seemingly strong performance
2. overall very clear exposition of a complex, but theoretically well motivated approach

hitting the guides dimensions:

- originality: hyperbolic embeddings are well established, but creating end to end VQVAE+ flow models I haven't seen yet
- quality: well written, decent evaluation, proofs appear to be correct after a single close read
- clarity: fully understandable, with some small nits
- significance: solid incremental advance, evaluation on larger graphs/trees required to say more

### Weaknesses
1. The density of exposition and hyperbolic-geometry terms can make the paper somewhat difficult to follow (einstein midpoint, möbius gyromidpoint etc). I’d suggest the following two tweaks
    
    1. state direction around around 107 that all terms not immediately defined are defined in the appendix for space constraint reasons (to warn the reader some will be just mentioned)
        
    2. make use of latex’ glossary feature [https://www.overleaf.com/learn/latex/Glossaries](https://www.overleaf.com/learn/latex/Glossaries) and a reasonable link color/style, to allow hovering over the term to see the definition in modern browsers (I think this will help the definition quite a lot) + enabling backlinks (if readers click through)
        
2. should detail how  hyperparameter choices/tuning were performed
    
3. nice to have: consider adding a  test on larger graphs (guacamol,moses), as noted in appendix K
    
4. nit: shouldn’t it be $\frac{2}{\sqrt{c}\lambda_x^c}$ , see e.g. [https://arxiv.org/pdf/1805.09112](https://arxiv.org/pdf/1805.09112)  eq 12?
    
5. interpolation experiment needs reporting of a baseline (in the appendix)
    
6. not: I think $\lambda_{valid}$ is meant to be $\lambda_{degree}$ on eq 7?

### Questions
1. would it make sense to do do ablations over mechanisms in the poincare parametrization since a lot of them are introduced, or are they an all or nothing operation (I assume the lattern)?
    
2. why does HVQVAE change from e.g. 0.002 in table  1 to 0.0071 in table 2? should report mean/std across multiple rounds (at least for their method/closest baseline) for CIs
    
3. why no experiment on generating trees? this would seem like the clear “hello world” example for this?

+ address as many weaknesses as possible please

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
3

### Summary
The main goal of the paper is to develop a graph generative model that leverages hyperbolic geometry to naturally capture hierarchical and tree-like structures. Specifically, it aims to show that embedding graphs in the Poincaré ball and generating them through manifold-aware neural components (hyperbolic GNN, geodesic attention, HVQVAE, and Riemannian flow matching) can more effectively represent the relational geometry of graphs compared to traditional Euclidean models.

### Strengths
1. The work presents a coherent framework that combines hyperbolic graph neural networks, geodesic attention, vector quantization, and Riemannian flow matching in a single end-to-end model. This integration is technically nontrivial and demonstrates careful engineering of both discrete and continuous latent components.

2. Paper is well written and maintains consistent notation throughout to follow.

### Weaknesses
1. Authors assume node labels depend only on their own latent and edges depend only on pairwise hyperbolic relations which I agree cuts decoding complexity, but it also forbids higher-order dependencies (motifs, triads) and makes long-range constraints modelling not accounted. It can miss global combinatorial constraints that are not pairwise-decomposable.

2. Authors add a degree-edge consistency term aligning predicted degree to ground-truth degrees and it improved MMD on degree but might risk over-regularizing towards degree histograms at the expense of other structures (e.g., motif diversity), if authors can comment on it?

3. Authors shared the anonymous repo but all files on the anonymous link are not accessible. It compromises the reproducibility.

4. Hyperbolic geometry’s expressive power critically depends on curvature c. However, methodology fixes c a priori. Curvature effectively controls the “branching factor” of the embedding manifold. Without adaptive curvature learning: 1) Graphs of different hierarchy depths collapse into a single scale OR The model may under- or over-stretch distances, biasing flow priors and reconstruction.

5. No error bounds are shown for the tangent-space linearization (how much curvature is lost per layer). Similarly, is there any theoretical grounding that repeated log–exp projections preserve manifold consistency?

6. The results mainly demonstrate performance on small synthetic or molecular settings, leaving open whether the proposed hyperbolic framework generalizes to large-scale or sparse real-world graphs.

### Questions
Look in the Weaknesses Section. I am open to considering answers for concerns mentioned in the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
