# Multi-Graph Meta-Transformer (MGMT)

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Multi-graph learning is crucial for extracting meaningful signals from collections of heterogeneous graphs. However, effectively integrating information across graphs with differing topologies, scales, and semantics, often in the absence of shared node identities, remains a significant challenge. We present the Multi-Graph Meta-Transformer (MGMT), a unified, scalable, and interpretable framework for cross-graph learning. MGMT first applies Graph Transformer encoders to each graph, mapping structure and attributes into a shared latent space. It then selects task-relevant supernodes via attention and builds a meta-graph that connects functionally aligned supernodes across graphs using similarity in the latent space. Additional Graph Transformer layers on this meta-graph enable joint reasoning over intra- and inter-graph structure. The meta-graph provides built-in interpretability: supernodes and superedges highlight influential substructures and cross-graph alignments. Evaluating MGMT on both synthetic datasets and real-world neuroscience applications, we show that MGMT consistently outperforms existing state-of-the-art models in graph-level prediction tasks while offering interpretable representations that facilitate scientific discoveries. Our work establishes MGMT as a unified framework for structured multi-graph learning, advancing representation techniques in domains where graph-based data plays a central role.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces at the same time the task of multi-graph learning and a hierarchical graph model which incorporates 1-hop attention and multi-graph fusion to tackle this task.

The paper is well-written and easy to follow, and interesting mathematical motivations are provided for some design choices, especially the meta-graph construction.

The contributions appear however limited, and the lack of many influential graph transformer (GT) baselines weakens the experimental section.

### Strengths
The paper is well written and easy to read, and the task it proposed to address is original.
I found the theoretical foundation for meta-graph construction (A5) particularly informative.
The experiment on neural activity in rats' brain is original and constitutes a nice opening to more diverse and practical applications.

### Weaknesses
The main weakness I see is the superficial consideration of previous graph learning, and especially graph transformer literature.

This is visible in several parts of the manuscript, as detailed below.

The related works section is very short, and several influencial papers are missing, especially regarding the GT literature (see the references [1-5] below), and hierarchical graph models (e.g. [6]). This omission is felt in the method and experiments sections, and overall weakens this work's contribution.

The method section only considers graph attention through the prism of GAT, which is essentially a MPNN layer that uses attention to aggregate neighbors' features. This formulation has been shown to have several limitations, especially regarding its expressivity (it is no more expressive than 1-WL). Global attention, as used in the SOTA GT models listed below, enhances this expressivity beyond 1-WL. From that regard, the local attention mechanism used in this paper appears very limited, especially as memory-efficient GT models exist (see, e.g., [5]).
Several experimental findings hint at the limited expressivity of the 1-hop neighborhood attention:
- The initial increase in accuracy from small $\tau$ values in Figure A8 is likely caused by a larger attention receptive field, i.e. a more expressive attention layer;
- The decreased ablation performances when supernodes are removed highlight the gains from performing attention over larger neighborhoods.
Supernodes partly alleviate this issue, but to the cost of erasing structural information.

Graph model baselines are missing from the experiments. The SOTA multimodal learning baselines are not graph architectures, except one. However the datasets considered here, both real-world and synthetic, are intrinsically graph learning problems. This is highlighted for LFP which contains nodes that "vary in number and identity across subjects", calling for methods that are invariant to node permutations and encode graph structures. The graph nature of the Alzheimer's database is also highlighted by the drop in ablation performances when edges are discarded. In summary, MGMT should be compared with SOTA GT models, instead of generic multimodal fusion models. Or, alternatively, experiments should be conducted on the same dataset as those latter methods (e.g. MIMIC, which is used in both MedFuse and FlexCare).
The benefits of the proposed method could therefore be properly assessed.


[1] Do transformers really perform badly for graph representation?, Ying et al., NeurIPS 21.

[2] Global self-attention as a replacement for graph convolution, Hussain et al., KDD 22.

[3] A generalization of vit/mlp-mixer to graphs, He et al., ICML 23.

[4] Exphormer: Sparse transformers for graphs, Shirzad et al., ICML 23.

[5] Graph inductive biases in transformers without message passing, Ma et al., ICML 23.

[6] Enhancing Graph Transformers with Hierarchical Distance Structural Encoding, Luo et al., NeurIPS 24.

### Questions
Some additional questions:

1. You say in RW that MGMT is fundamentally different from multimodal graph learning, but when different graphs in a sample have different features dimension $d$, you will need separate encoders for each modality. How is it then different from multimodal graph learning?

2. How does Theorem 3.3 / L-hop mixing translate in terms of WL expressivity?

3. It seems from Equation 5 that metagraphs (graphs with supernodes) can become disconnected if $\tau$ is too high. Was it problematic in your experiments?

4. I don't understand the aggregation over the batch (sum over $k$) in Equation (A17), even when assuming all samples (graph collections) have the same number of graphs $n$. If the goal is to evaluate the relevance of depth $l$, why not aggregating over all $n$ graphs and $K$ collections?
The dependency on $i$ of $\Gamma_i^l$ in (A18) is unclear either.
It seems in Equation 3 that this dependency is absent, but the dependency on $k$ also seems to be omitted in that section. Can you clarify this?

### Soundness
3

### Presentation
3

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
This paper introduces a framework called Multi-Graph Meta-Transformer (MGMT), designed to effectively integrate information from heterogeneous graphs. The method first independently encodes each graph using specific depth-aware Graph Transformer layers. Subsequently, a "meta-graph" is constructed to capture inter-modal relationships. The meta-graph consists of super-nodes from each graph, extracted as nodes with high attention scores. The edges of the meta-graph comprise two parts: intra-modal edges and inter-modal edges, with the latter established based on similarity between super-nodes. Finally, an additional Graph Transformer is applied to this meta-graph to perform downstream tasks.

### Strengths
1. It’s a good attempt to build the "meta-graph" to explicitly model inter-graph relationships between heterogeneous graphs.

2. It’s meaningful and interesting to apply MGMT to neuroscience applications.

### Weaknesses
1. The motivation of the paper is not clear. Although the authors argue that multi-graph fusion is challenging, the paper does not clearly specify what exact limitations of existing approaches must be solved. It’s still confusing after reading the introduction. Is the main limitation the current multi-graph fusion methods cannot preserve both inter- and intra-graph structural information? In addition, in introduction part, the authors should summarize the contributions of the paper, which highlights the unique technical novelty and the significance.

2. The works mentioned in related works seem not highly related to this paper. Since the focus of the work is on learning across multi-graph (multiple heterogeneous graphs), the paper should compare against prior research specifically in multi-graph learning rather than primarily discuss multimodal graph learning or general multimodal fusion, which involve fundamentally different settings and assumptions. In addition, the paper structure feels somewhat unconventional, it would be better to have a separate section for related work. 

3. The authors claim that the graph-specific Transformer encoders produce depth-aware intra-graph representations. However, this appears to be essentially a form of multi-layer attention aggregation, which is conceptually similar to existing multi-layer aggregation techniques such as LayerMix. It remains unclear what exactly “depth-aware” means beyond weighted averaging across Transformer layers. There is currently no experimental or analytical evidence demonstrating that different layers contribute distinct or complementary information. For example, the authors didn't provide any visualization of the confidence scores $Γ^{(ℓ)}$, or an analysis showing different layers contribute meaningfully. Without such results, it is difficult to conclude that the model truly learns depth-specific structural semantics rather than merely fusing multi-level embeddings. Although the ablation experiment removing adaptive depth yields a performance degradation, this alone is insufficient to validate the claimed depth-aware representation learning.

### Questions
1. Why related work section does not mention about multi-graph learning, but instead of multimodal graph learning and multimodal fusion?

2. How does the depth-aware GT layers differ from existing multi-layer aggregation techniques?

3. In section 4.2, What do you mean multimodal classification? Also, in Figure 3, what’s the meaning of Modality 1, 2,…?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the Multi-Graph Meta-Transformer (MGMT), a unified, scalable, and interpretable framework for cross-graph learning, addressing the key challenge of effectively integrating information across heterogeneous graphs (with differing topologies, scales, semantics, and often no shared node identities). MGMT’s core pipeline includes three stages: (1) applying Graph Transformer (GT) encoders to each graph, with depth-aware aggregation of layer outputs to generate intra-graph latent representations; (2) extracting task-relevant supernodes via attention scores and constructing a meta-graph that connects functionally aligned supernodes across graphs using latent similarity; (3) applying additional GT layers on the meta-graph to enable joint reasoning over intra- and inter-graph structures. Evaluated on three synthetic datasets and two real-world neuroscience applications (hippocampal LFP-based stimulus prediction and Alzheimer’s disease detection), MGMT consistently outperforms state-of-the-art baselines (e.g., Meta-Transformer, MMGL) in graph-level prediction tasks while providing interpretable representations (via supernodes and superedges) that facilitate scientific discoveries.

### Strengths
This paper tackles a critical gap in multi-graph learning: existing fusion methods either assume node alignment (limiting applicability to heterogeneous graphs) or collapse graph topology into a single vector (losing structural information), while MGMT preserves both intra- and inter-graph structure—a highly valuable contribution.  

The proposed framework is notably versatile, unifying three common multi-graph scenarios (multi-modal, multi-view, multi-subject) under one umbrella, which broadens its practical utility across domains like neuroscience, molecular biology, and social networks.  

MGMT is supported by solid theoretical foundations: the paper formally proves that MGMT’s depth-aware GTs enable L-hop mixing (surpassing vanilla GTs) and that MGMT achieves a smaller approximation error than late fusion methods, enhancing the credibility of its design.  

Built-in interpretability is a key strength: supernodes highlight influential substructures (e.g., distal CA1 electrodes in neuroscience) and superedges reveal cross-graph alignments, making MGMT not only a predictive tool but also a means to drive scientific insights—rare in many graph learning models.  

The experimental design is thorough: it covers synthetic datasets (varying noise, node counts, and sample sizes) and real-world neuroscience tasks, includes comprehensive ablations (validating adaptive depth, supernode selection, and inter-graph edges), and compares against diverse baselines (single-source, early fusion, and state-of-the-art multimodal models), leaving little doubt about MGMT’s effectiveness.

### Weaknesses
MGMT relies on fixed thresholds (τ for supernode extraction, γ for inter-graph edge construction) that require cross-validation to tune. This manual thresholding lacks adaptability to dynamic or unseen data distributions, and the paper does not explore alternatives like learnable edge weights or data-driven dynamic thresholds, which could improve robustness.  

While the paper acknowledges rising computational complexity with more modalities/larger graphs (due to attention layers’ quadratic cost), it only mentions sparse/low-rank attention as a future direction without providing preliminary optimization results (e.g., runtime comparisons with sparse attention variants). This limits confidence in MGMT’s scalability for large-scale multi-graph scenarios (e.g., hundreds of graphs with thousands of nodes).  

The attention-based importance scores for supernode selection may fail to capture causal structures in noisy data—a limitation noted in the paper—but no mitigating strategies (e.g., robust attention, causal inference integration) are discussed or explored, leaving a gap in addressing real-world data imperfections.  
Experimental validation is concentrated on neuroscience applications. Although the paper claims MGMT is applicable to molecular biology and social networks, it provides no empirical results in these domains, weakening the evidence for its cross-domain generality.

### Questions
1. Why did you choose fixed thresholds (τ and γ) for supernode extraction and inter-graph edge construction instead of more adaptive approaches (e.g., learnable edge weights, dynamic thresholding based on data statistics)? How does threshold tuning affect MGMT’s performance across datasets with varying graph heterogeneity?  
2. For the computational complexity issue (quadratic in graph size/node count), have you conducted preliminary experiments with optimization techniques like sparse attention or low-rank attention? If so, could you share runtime and accuracy trade-off results; if not, what is the feasibility of these techniques for MGMT?  
3. Given that attention-based supernode scores struggle with causal structure in noisy data, do you have plans to integrate causal inference methods (e.g., causal attention, intervention-based importance scoring) to improve supernode selection accuracy? Could you outline potential design modifications for this?  
4. The paper asserts MGMT’s applicability to molecular biology and social networks, but only validates it in neuroscience. Can you provide preliminary experimental results in these other domains (e.g., molecular property prediction, social network classification) or explain how neuroscience findings generalize to them, given differences in graph structure (e.g., molecular graphs vs. brain connectivity graphs)?

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
2

### Summary
This paper proposes a unified framework for multi-graph learning, termed Multi-Graph Meta-Transformer (MGMT). Its core innovation lies in encoding multiple heterogeneous graphs using Graph Transformers, extracting representative supernodes via an attention mechanism, and constructing a meta-graph that captures inter-graph dependencies, thereby enabling information exchange and alignment among graphs within a shared latent space.

### Strengths
By stacking additional Transformer layers on the meta-graph, the model learns both intra- and inter-graph structural features, achieving depth-aware aggregation and hierarchical reasoning. This design improves the precision and robustness of cross-graph representations and provides intrinsic interpretability, leading to superior performance over existing fusion models in neuroscience and multimodal medical data analysis tasks.

### Weaknesses
The title only reflects the method name “Multi-Graph Meta-Transformer” and does not convey the core innovation or research objective, such as “cross-graph fusion” or “interpretable multimodal graph learning.” This makes the title insufficient to highlight the study’s contributions. It is recommended that the authors revise the title to better communicate the paper’s novelty and focus.


The positioning of the innovation is somewhat ambiguous. The differences between MGMT and existing Meta-Transformer and multimodal fusion models are insufficient to clearly define it as a “new framework.” MGMT is highly similar in naming and overall concept to Meta-Transformer (Ma et al., 2022). Although the paper introduces the concepts of meta-graph and supernodes, the approach essentially remains a variant of “cross-modal Transformer + graph structure reconstruction.” The paper does not adequately demonstrate substantial theoretical or representational improvements over previous work.


Minor language and formatting issues.
Section 2.1.1: “matrices and and bold lowercase …” – redundant “and.”
Proof of Theorem 3.4: “the desired results follows …” – subject-verb disagreement; should be results follow.
Appendix A1: “Shi et al. Shi et al. (2020)” – duplicated citation; should be Shi et al. (2020).
The cited literature is relatively outdated, lacking references to recent studies (within the past two years) on multi-graph learning and cross-modal Transformers. The experimental baselines are also old, mainly consisting of earlier methods and self-ablation experiments, which fail to demonstrate the method’s advancement and competitiveness.


Section A6 mainly describes the setup and results of the ablation experiments but does not analyze in detail how the absence of each module leads to performance degradation. It lacks an interpretation of the experimental results and explanations from a mechanistic perspective. It is suggested to include failure analysis and visualization to strengthen the discussion.


The paper does not provide a clear explanation for the runtime fluctuation observed in Figure A8 as the attention threshold τ increases, especially in the left panel where τ controls supernode selection. This non-monotonic trend is inconsistent with the theoretically smooth growth of MGMT’s computational complexity described in the paper. The authors are advised to further discuss possible reasons for this phenomenon.

### Questions
The experiments only verify graph-level classification tasks and lack extended experiments for node-level or edge-level tasks, which limits the demonstration of MGMT’s applicability to other graph learning scenarios such as link prediction, anomaly detection, or graph generation.

### Soundness
3

### Presentation
3

### Contribution
2
