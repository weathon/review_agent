# GraphCliff: Short-Long Range Gating for Subtle Differences but Critical Changes

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Quantitative structure–activity relationship assumes a smooth relationship between molecular structure and biological activity. However, activity cliffs defined as pairs of structurally similar compounds with large potency differences break this continuity. Recent benchmarks targeting activity cliffs have revealed that classical machine learning models with extended connectivity fingerprints outperform graph neural networks. Our analysis shows that graph embeddings fail to adequately separate structurally similar molecules in the embedding space, making it difficult to distinguish between structurally similar but functionally different molecules. Despite this limitation, molecular graph structures are inherently expressive and attractive, as they preserve molecular topology. To preserve the structural representation of molecules as graphs, we propose a new model, GraphCliff, which integrates short- and long-range information through a gating mechanism. Experimental results demonstrate that GraphCliff consistently improves performance on both non-cliff and cliff compounds. Furthermore, layer-wise node embedding analyses reveal reduced over-smoothing and enhanced discriminative power relative to strong baseline graph models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes GraphCliff, a graph neural model for QSAR tasks with activity cliffs. The core idea is to explicitly combine short-range message passing (i.e. GINE) with long-range propagation (i.e. Chebnet) via a learnable sigmoid gate, followed by attention-based pooling for graph-level prediction. The authors motivate the work by showing that standard GNN embeddings under-emphasize local substructure differences compared to ECFP dissimilarities; GraphCliff is intended to preserve local sensitivity while providing global context. Experiments show lower RMSE overall and ablations indicate both filters and the gate contribute to performance.

### Strengths
S1: The paper targets activity cliffs, a well-known failure mode for deep models and matter for medicinal chemistry that generic molecular GNNs miss.

S2: The empirical study is extensive, with results on 30 MoleculeACE targets, along with targeted ablations such as removing short/long paths and gating, hop-wise sensitivity and Dirichlet energy estimation, which examine not just accuracy but also how information propagates through the network.

S3: The paper is well organized and readable. The modular architecture (short-range path, long-range path, and gating) is explained clearly, and the figures/equations make the design easy to follow.

### Weaknesses
W1: Section 5.2’s hop-wise sensitivity (perturbation of u affecting v at k hops) and Jacobian spectrum are closer to an over-squashing analysis than smoothing. Although Dirichlet energy is included for smoothing, the text and framing mix these notions, and it’s unclear whether the proposed gains primarily combat squashing (limited receptive field / information bottlenecks) or smoothing (Laplacian averaging). 

W2: Necessity of the mechanism vs. simpler baselines. If over-smoothing were the key issue motivating deep propagation, residual/skip connections, PairNorm/BatchNorm, JK-Net and APPNP/PPNP are standard, lightweight fixes. The paper does not compare GraphCliff against these strong anti over-smoothing baselines or against simple residual GNNs. This weakens the claim that the proposed mechanism is required.

W3: Chebyshev long-range propagation, gated/highway mechanisms, and attention pooling have all been in literature for other application; GINE is standard for molecular graphs. The specific combination is practical but feels like engineering glue more than a new principle. 

W4: Formal problem setup & notations are missing. There is no concise notation paragraph formally defining the input graph G=(V,E), node/edge features, and the prediction problem; the method section jumps directly into components/equations, which hurts clarity.

### Questions
Q1: Could you clarify why do deep GNNss are needed for this application ?

Q2: What are the training/inference runtime and memory costs of GraphCliff relative to strong baselines? How do these scale with graph size and Chebyshev order?

Q3: Did you try residual connections to the input or JK aggregation in lieu of gated fusion? Even a small study showing why they underperform would clarify the design choice.

Q4: Could you provide a few negative cases where GraphCliff fails on known cliff pairs and analyze whether the failure is due to insufficient local sensitivity, long-range diffusion, or data sparsity? This would sharpen the pros and cons of GraphCliff

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GraphCliff, a gated graph neural network that integrates short-range (GINE) and long-range (Chebyshev) filters to capture subtle structural variations, known as activity cliffs, in molecular property prediction. The model explicitly balances local substructural sensitivity and global molecular context, addressing over-smoothing issues in conventional GNNs. Extensive experiments across 30 MoleculeACE datasets and small-scale LSSNS benchmarks demonstrate consistent improvements and enhanced discriminative node embeddings.

### Strengths
- The paper tackles an important and challenging domain-specific issue—*activity cliffs*—that conventional GNNs struggle with.
- The proposed gating design effectively balances local and global information, reducing over-smoothing while preserving local sensitivity.
- The experiments are thorough, including 30 datasets, multiple baselines, ablation and interpretability analyses (e.g., Hop-wise sensitivity, Dirichlet energy, …).

### Weaknesses
- Although the overall idea of integrating short- and long-range information is reasonable, the novelty of the approach is somewhat limited, as similar hybrid architectures (e.g., GROVER, GraphTrans) have already been proposed. The paper should more clearly articulate how GraphCliff’s gating design provides advantages specific to molecular *activity cliff* prediction.
- Minor issues in figure and notation:
    - Figure 1 (Overall architecture of GraphCliff) is visually unrefined and lacks clear correspondence between visual components and the mathematical formulation, such as X, h, z.
    - Some notations and dimensional definitions are missing or ambiguous

GROVER : Rong, Yu, et al. "Self-supervised graph transformer on large-scale molecular data." *Advances in neural information processing systems* 33 (2020): 12559-12571.

Graphtrans : Wu, Zhanghao, et al. "Representing long-range context for graph neural networks with global attention." *Advances in neural information processing systems* 34 (2021): 13266-13279.

### Questions
The paper emphasizes integrating short- and long-range information. However, prior works such as GROVER and GraphTrans have already explored combining local message-passing GNNs with Transformer-based long-range modeling.
- Does it achieve superior results even when compared to such hybrid models? 
- What are the expected advantages of such hybrid models in terms of sensitivity, interpretability, and over-smoothing analyses?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper prosposes GraphClif,  a short–long range gating mechanism to explicitly integrate local substructural sensitivity and global molecular context, mitigating over-smoothing while preserving expressive molecular graph representations.

### Strengths
1. The empirical results are strong and were done on 30 benchmarks, nonetheless all of them stem from the same dataset.
2. The visualization obtained from the gating mechanism seems to provide interesting insights that align with domain priors
3. The approach to combine high and low frequency signals with gating is intuitive and simple.
4. The authors provide ablation studies.

### Weaknesses
1. The approach is tailored and demonstrated to molecules, and it is not clear whether other domain can benefit from it.
2. The empirical evaluation although very extensive, focuses only on ChEMBL, and it remains unknown if this method is also beneficial to other domains or datasets. Evaluating it on other diverse benchmarks from other domains and other tasks may be more convincing on the merits of this work.
3. Based on the two above comments, it is possible the contribution is incremental as it is beneficial only for very specific tasks and types of data. Nonetheless it is possible that this problem of its own is important enough to justify a tailored architecture. As I am not from the molecular field, I lack the ability to judge on the importance of this problem of its own, but rather commenting on the broad contribution of the method to the graph community. 
4. One thing that can strengthen the method is some theoretical example where it is provable that without this combination of high and low rank, the task is unrealizable e.g. with 1-WL GNN, but it is realizable with your method. This would motivate the generality of tis approach to solve general cases as shown here empirically, from the theoretical side.

### Questions
1. Could you provide other examples rather than molecular predictions where this approach is critical ?

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
2

### Summary
This paper aims to solve the discontinuity problem known as the activity cliff, where small structural differences significantly impact activity in QSAR. This paper proposes a new graph neural network called GraphCliff. The proposed method is inspired by StripedHyena2 and integrates local interactions with global interactions using gating mechanisms. The proposed method is applied to 39 QSAR tasks, including those addressing the activity cliff problem, and its predictive performance is evaluated. Additionally, ablation studies are conducted to analyze the impact of local structure on activity cliffs.

### Strengths
1. (Originality) As noted in this paper, methods for modeling long-range dependencies have been studied in several fields, such as genome language models. Approaches integrating local and global interactions have been studied, including the Hyena Hierarchy. However, to my knowledge, this paper is the first to apply this idea to the long-range dependency problem in graph learning.
2. (Quality) The paper deals with the important and specific problem of the activity cliff. Furthermore, the numerical experiments cover a wide range of experimental settings, including 16 baseline methods and 39 datasets, which strengthens the credibility of its claims.
3. (Clarity) The writing is clear. The paper's structure is appropriate, and the explanations in each section are straightforward. I had no significant difficulty in understanding the paper's main points.

### Weaknesses
1. The discussion of the numerical experiment results has room for improvement. Specifically, I question whether the presentation of Table 1 is appropriate. This table only highlights the datasets where the proposed method achieves the highest accuracy. Results for the remaining datasets are provided in the appendix. However, if I do not miss any information, no validation or discussion regarding them is presented.
2. While the paper claims the proposed method achieves the best overall performance (L.309), the basis for this claim is unclear. This claim should be substantiated through a quantitative evaluation using all 39 datasets.
3. Section 4 provides a detailed analysis of existing methods other than the proposed one (L.310--329). While this analysis is valuable, it deviates from the main focus of this paper, that is, validating the accuracy of the proposed method, and is therefore less important.
4. If I have not missed any information, the method for determining hyperparameters is not described.

### Questions
1. I would like to clarify the basis for claiming that the proposed method achieved the best overall performance.
2. I would like the authors to deepen the analysis of the causes for the poor prediction accuracy of the proposed method on certain datasets.

### Soundness
2

### Presentation
3

### Contribution
2
