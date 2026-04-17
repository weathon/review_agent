# RoBlock: Wide and Deep Scaling of Recommenders via Embedding Collapse Mitigation

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Scaling recommendation models has emerged as a promising direction for advancing recommender systems, yet they face a fundamental challenge: *embedding rank collapse*. 
This phenomenon, rooted in the intrinsic properties of feature interaction modules, causes embedding matrices to lose representational capacity as models scale, severely limiting their effectiveness. 
Existing solutions primarily focus on width-wise scaling via *multi-embedding*, which parallelizes multiple embedding tables and has shown success in alleviating collapse.
However, these methods configure only the initial embedding layer and fail to address *depth-wise* embedding collapse, which intensifies with increasing model depth and restricts the benefits of deeper architectures. 
We propose **RoBlock**, a stackable building block that delivers dimensionally-robust embeddings by mitigating collapse across width and depth. 
RoBlock integrates three key components: (1) **spectrum rebalancing** through rank-1 update normalization to restore the spectrum distribution of embedding matrices, (2) an **embedding decoupler** guided by the Hilbert–Schmidt Independence Criterion (HSIC) to extract independent embedding components while preserving spectrum (dimensional) robustness, and (3) **embedding regeneration** via a field-wise multi-head router to regenerate non-collapsed embedding sets, achieving the benefits of multi-embedding within each block.
Theoretical analysis establishes that RoBlock effectively mitigates embedding collapse, providing a principled foundation for scalable recommendation models. 
Extensive experiments across multiple datasets further demonstrate that RoBlock consistently alleviates embedding collapse across layers and delivers significant performance gains over the baselines, with improvements growing as model width and depth increase. 
The code is accessible at the anonymous link: https://anonymous.4open.science/r/RoBlock-2F8A

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the problem of embedding rank collapse in large-scale recommendation models, which manifests both in width (multi-embedding) and depth (layer stacking).
Existing works mainly address width-wise collapse (e.g., through multi-embedding fusion), while ignoring depth-wise degradation that occurs as the network deepens.
The authors propose RoBlock, a stackable modular block that mitigates rank collapse through three integrated mechanisms:
Rank-1 Update Normalization – balances the singular value spectrum of embeddings to prevent dominant directions.


HSIC-based Decoupler – enforces independence across embedding subspaces via Hilbert–Schmidt Independence Criterion(HSIC).


Field-wise Router – recombines embeddings across fields to regenerate non-collapsed components.


Theoretical analysis rigorously proves that these modules together preserve effective rank under both exact and approximate HSIC orthogonality. Empirical studies on Criteo and Avazu datasets show consistent improvements in AUC and information abundance (IA) across depth and width scaling.

### Strengths
**Design Strength**

- **Logical three-stage pipeline**: Clear progression from Stabilize (Rank-1 Normalization) → Decorrelate (HSIC Decoupler) → Regenerate (Field-wise Router)

- **Theory-practice alignment**: Theoretical goals directly map to implementation components

- **High interpretability**: Non-arbitrary, purposeful architecture design

**Theoretical Rigor**

- **Comprehensive proofs**: Each component mathematically justified (A.1: spectral convergence, A.2: rank additivity, A.3: router rank bounds)

- **Provable guarantees**: Positions RoBlock as a provably rank-preserving architecture

**Scalability Validation**

- **Consistent performance**: Maintains superior AUC and Information Abundance (IA) across depth and width scaling on Criteo and Avazu datasets

- **Resists degradation**: Successfully prevents IA decay observed in baseline models

**Appendix Completeness**

- **Thorough documentation**: Full proofs, robustness analysis, computational complexity (A.4.4), and standard deviation results (Tables 5–9)

- **Addresses credibility**: Covers stability, efficiency, and reproducibility concerns

### Weaknesses
**HSIC Regularization Sensitivity**

- **Limited exploration**: Only tests β ∈ {0, 0.1, 0.5, 1.0} without comprehensive sensitivity analysis

- **Unclear boundaries**: Performance behavior at extreme β values (→0 or large) unexplored

- **Missing guidance**: No clear recommendations for practitioners on optimal regularization strength selection

**Dataset Diversity and External Validity**

- **Narrow scope**: Limited to two CTR-focused tabular datasets (Criteo and Avazu)

- **Generalizability unknown**: Unclear if anti-collapse mechanisms transfer to:
  - Non-tabular data
  - Multi-modal embeddings (text, vision)
  - Other recommendation domains beyond CTR prediction

### Questions
none

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the embedding rank collapse issue that occurs when recommender models are scaled in both width and depth.
The authors propose RoBlock, a modular block composed of Rank-1 normalization, HSIC-based embedding decoupling, and a multi-head router, which jointly maintain high-rank embeddings during training.
Theoretical analysis and experiments on Criteo and Avazu show that RoBlock mitigates collapse and improves AUC over xDeepFM, DCNv2, and DHEN.

### Strengths
* Clearly identifies a key limitation in deep/wide scaling of recommenders.

* Provides a well-structured and interpretable module design.

* Strong empirical results that match theoretical motivation.

* Theoretical reasoning supports the design choices.

### Weaknesses
* Computational cost and ablation results are not thoroughly discussed.

* Released code appears incomplete ("The requested file is not found."), making reproduction difficult.

### Questions
How much additional cost does RoBlock introduce compared to other model architectures, such as DCNv2?

The reported DNN baseline on Avazu seems unusually low, was the baseline model properly tuned (e.g., embedding size, learning rate, dropout, optimizer)

### Soundness
3

### Presentation
3

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
This work addresses the problem of embedding rank collapse in large-scale recommendation systems where the embeddings lose expressive power as the model scales (both in depth and width). To solve this issue, the authors propose RoBlock, a modular design which helps maintain the dimensionality of the original embedding space and mitigate collapse. RoBlock adopts three core designs: (a) rank-1 update normalization to restore the singular values of the embeddings, (b) decoupled embeddings to encode different properties of the data, and (c) HSIC regularization to encourage diversity across the decoupled embeddings. Each of these has associated theoretical analysis on how they help preserve the rank of the embeddings. Empirically, the method is able to show robustness to different depths and widths, with some gains in performance.

### Strengths
I will start with the strengths of the work. 

1. I think the work studies an important problem on embedding collapse with clear motivation as to why we should care about it. 
2. The authors offer a series of modular designs within RoBlock which can be easily adapted to different settings. I also appreciate that each is motivated with theoretical analysis. 
3. The authors offer strong empirical evidence that they can remedy the embedding collapse problem (which is different than improving performance), particularly as the models scale across width or depth dimensions.

### Weaknesses
Despite the strengths, there are quite a few areas that lack justification or deeper analysis. 

1. Comparison to Previous Collapse Mitigation: The authors cite a variety of methods that study embedding collapse. Moreover, there are works in the space of recommenders that also work on this problem. Despite this, the authors do not offer any direct empirical analysis to them. Moreover, many of them tend to be much simpler compared to RoBlock, containing one component/loss, thus it is important to demonstrate that RoBlock warrants its multiple designs by out-performing them.   
2. Theory: While the authors provide theory to support the maintaining of rank, a clear gap is connecting rank to performance. While the authors intuitively argue that higher rank is better, it is not obvious to me that collapse is always problematic. It would be good for the authors to address this disconnect and make a rigorous argument connecting rank to performance. 
3. Runtime Analysis: No empirical runtime analysis is provided for the work. It would be great to have a sense of how much computational overhead the method requires as compared to the previous methods. 
4. Performance: This is more minor, but it can be hard to assess the significance of the performance improvements in Table 1. When reading the original paper that proposed Criteo and Avazu, it seems as though these numbers are comparable to what they original achieve. Can the authors further justify why improvements at the 3rd/4th decimal point are meaningful?

### Questions
1. Can the authors provide a deeper analysis to methods that mitigate embedding collapse? As of now, the baselines are relatively simple. 
2. Can the authors better connect matrix rank to performance? 
3. Can the authors provide an empirical runtime analysis compared to the other methods? 
4. Can the authors justify the significance of the performance? I will acknowledge that I see the standard deviations (I would like to see those in the main table, it is unclear why they are int he appendix), but that doesn't change the fact that the performance appears marginal.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The idea of RoBlock is quite clear — it treats the common embedding rank collapse problem in recommender systems as the main challenge to solve.
The authors argue that, after multiple layers of feature interactions, the embedding information tends to concentrate in a few dominant dimensions, leading to reduced rank and degraded representational capacity.
To address this, they design a modular structure where each layer is a RoBlock module that performs several operations:
first, Rank-1 Update Normalization to rebalance the singular values and prevent energy concentration;
second, HSIC regularization to encourage independence among multiple sub-embeddings and improve rank;
third, a field-wise router to dynamically combine features from different fields;
and finally, residual connections to stabilize information flow between layers.
Overall, the motivation is sound and the system design is consistent.
However, the novelty is somewhat limited — similar ideas such as spectral balancing, HSIC-based independence, router gating, and residual fusion have appeared in prior works.
RoBlock’s main contribution lies in integrating these mechanisms into a unified modular framework, which is well-structured and has some engineering value.

### Strengths
Clear problem definition and motivation.
The paper focuses on the embedding rank collapse issue in recommender systems, which is indeed an important challenge when scaling deep recommendation models.

Well-structured overall design.
By combining Rank-1 Update Normalization, HSIC-based decoupling, field-wise routing, and residual connections into a modular framework, the method provides a systematic structure for mitigating collapse.

Consistent theoretical and empirical support.
Theoretical analysis on spectral balancing and independence, together with the IA visualization results, reasonably supports the proposed design.

### Weaknesses
Limited novelty.
Most components of the proposed method have already appeared in prior works; the contribution mainly lies in integrating existing ideas rather than introducing a new algorithmic principle.

No discussion of efficiency.
The model involves computationally heavy operations (e.g., rank-1 update, HSIC regularization, dynamic routing), yet the paper does not evaluate or discuss their efficiency.

Unclear practical value.
Although the design may theoretically alleviate embedding degradation, the high computational cost raises concerns about its feasibility in real-world recommender systems.

### Questions
Could the authors provide an evaluation of the computational efficiency of RoBlock?
In particular, it would be helpful to report training time, inference latency, and memory usage compared to baselines such as xDeepFM or DHEN.
Since the proposed modules (Rank-1 Update, HSIC regularization, and field-wise router) are computationally intensive, understanding their actual runtime overhead is important for assessing the model’s practicality.

Have you considered using approximations or lightweight variants (e.g., simplified normalization or reduced HSIC sampling) to balance effectiveness and efficiency?

How does the model scale when applied to larger datasets or higher embedding dimensions?

### Soundness
3

### Presentation
3

### Contribution
2
