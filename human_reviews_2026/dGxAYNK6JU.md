# PoinnCARE: Hyperbolic Multi-Modal Learning for Enzyme Classification

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 4, 8, 8

## Abstract
Enzyme Commission (EC) number prediction is vital for elucidating enzyme functions and advancing biotechnology applications. However, current methods struggle to capture the hierarchical relationships among enzymes and often overlook critical structural and active site features. To bridge this gap, we introduce PoinnCARE, a novel framework that jointly encodes and aligns multi-modal data from enzyme sequences, structures, and active sites in hyperbolic space. By integrating graph diffusion and alignment techniques, PoinnCARE mitigates data sparsity and enriches functional representations, while hyperbolic embedding preserves the intrinsic hierarchy of the EC system with theoretical guarantees in low-dimensional spaces. Extensive experiments on four datasets from the CARE benchmark demonstrate that PoinnCARE consistently and significantly outperforms state-of-the-art methods in EC number prediction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces PoinnCARE, a hyperbolic multi-modal learning framework for enzyme function prediction. By combining information from enzyme sequences, structures, and active sites, the approach leverages graph diffusion and dual-graph alignment within hyperbolic space to encode hierarchical relations informed by the Enzyme Commission (EC) system. Extensive experiments on the CARE benchmark demonstrate that PoinnCARE achieves improved accuracy over a strong set of baselines across several challenging test splits.

### Strengths
1. Integration of multiple modalities: The paper goes beyond sequence-based modeling by integrating experimentally determined or predicted enzyme structures and active site annotations, which addresses the intrinsic complexity of enzyme function prediction.
2. Hyperbolic embedding with theoretical justification: Embedding enzyme representations in hyperbolic space is both theoretically motivated  and practically validated. The paper rigorously discusses why hyperbolic geometry is ideal for modeling the EC system's hierarchy.
3. Comprehensive and robust empirical evaluation: The method is benchmarked across four different and increasingly challenging test scenarios from CARE, with comparisons to a wide range of recent and classical baselines, including CLEAN, CLEAN-Concat, various PLMs, and LLM approaches. Tables 2, 3, and supplemental metrics (precision, recall, F1; Tables 9–14) consistently show improvements of PoinnCARE, validating its robustness to sequence diversity and test set complexity.
4. Ablation and dimensionality studies: The results in Figure 6 and Table 16 demonstrate that performance gains are attributable to each design component, and PoinnCARE's effectiveness persists even at smaller embedding dimensions, reinforcing the theoretical claims regarding hyperbolic geometry.
5. Reproducibility efforts: The authors provide detailed data and implementation information, facilitating future reproduction.

### Weaknesses
1. Lack of comparison with advanced models: This paper uses enzyme structure and active site information, but the specialized models it compares against are limited to the sequence-based CLEAN and its derivatives. It does not include comparisons with structure-based models such as EnzymeCAGE (Liu et al. 2024) and GraphEC (Song et al. 2024).
(1) Liu, Yong, et al. "EnzymeCAGE: a geometric foundation model for enzyme retrieval with evolutionary insights." bioRxiv (2024): 2024-12.
(2) Song, Yidong, et al. "Accurately predicting enzyme functions through geometric graph learning on ESMFold-predicted structures." Nature Communications 15.1 (2024): 8180.
2. Questionable choice of hyperbolic GNN architecture: PoinnCARE adopts the Poincaré Ball Model and performs GNN aggregation and transformation in the tangent space at the origin ($o$). The authors state this choice was made "For simplicity". However, HGCN (Chami et al., 2019) explicitly argues for and demonstrates the superiority of aggregation in the local tangent space of each center node ($x_i^H$). PoinnCARE's architectural choice seems to be a simplification that may lead to suboptimal representation power. The paper does not provide sufficient justification that this "simple" choice is reasonable or equivalent to local-space aggregation.
(1) Chami, Ines, et al. "Hyperbolic graph convolutional neural networks." Advances in neural information processing systems 32 (2019).
3. Concerns about the efficacy of the active site modality: The paper admits that the active site data is extremely sparse, with "more than half of the enzymes have no annotated active site residues". Despite using graph diffusion, the information foundation for this modality is very weak. Furthermore, the input features $H^{(0)}$ for both hyperbolic GNNs are identical. This introduces a risk: the representation $H_{(a)}$ learned by the GNN on $G^{(a)}$ ($f_{hyp}^{(a)}$) might just be a variant of the $H^{(0)}$ features on a highly smoothed (post-diffusion) graph structure, without capturing much genuinely unique "active site" functional signal. The contribution of this modality may be overestimated.
4. Oversimplified modality fusion: The final fusion method for prediction is a simple weighted sum: $\beta_{s} \cdot H_{(s)} + \beta_{a} \cdot H_{(a)}$. Given the heterogeneous nature and different sparsity levels of the two modalities (structure vs. active site), this linear, static fusion method is likely suboptimal. The paper does not explore more dynamic fusion mechanisms (e.g., attention, gating).
5. Positioning in relation to closest prior work: Several highly relevant recent papers are not cited or discussed, particularly those specifically focusing on hyperbolic multimodal taxonomy or hierarchical enzyme function prediction. This weakens the contextualization and differentiation of the method’s originality.
(1) Gong, ZeMing, et al. "Hyperbolic Multimodal Representation Learning for Biological Taxonomies." arXiv preprint arXiv:2508.16744 (2025).
(2) Li, Nan, et al. "Hyperbolic hierarchical knowledge graph embeddings for biological entities." Journal of Biomedical Informatics 147 (2023): 104503.
6. Insufficient analysis of failure cases: The analysis leans heavily on aggregate metrics. It would be helpful to present examples or qualitative analysis, where the method fails, especially at deeper EC levels. There’s a lack of introspection on possible errors, such as misclassification of enzymes with convergent functions but divergent structures or sequences (a case hinted at by Figure 3, but not explored in error analysis).

### Questions
1. Could the authors please clarify the reasoning behind the exclusion of state-of-the-art specialized models from the evaluation?
2. What is the rationale for choosing to perform GNN operations in the origin's tangent space rather than the local node's tangent space? Beyond "simplicity," is there any theoretical or experimental evidence (e.g., a comparison of these two aggregation methods within the PoinnCARE framework) to support this decision?
3. Given the extreme sparsity of $G^{(a)}$ and the shared $H^{(0)}$ input, how can the authors demonstrate that $H_{(a)}$ has learned "active site"-specific information that is meaningful for prediction and distinct from $H_{(s)}$? Is there any qualitative or quantitative analysis of the differences between the $H_{(a)}$ and $H_{(s)}$ embedding spaces (e.g., t-SNE visualization or mutual information analysis)?
4. Why was a simple linear weighted sum chosen as the final modality fusion method? Did the authors experiment with more complex fusion mechanisms (e.g., concat + MLP, or cross-modal attention)?
5. In constructing $G^{(s)}$ and $G^{(a)}$ (Sections B.1 and B.2), how were the similarity thresholds $\delta^s=0.3$ and $\delta^a=0.05$ chosen? How sensitive is the model's performance to these thresholds?
6. Could the authors analyze cases where PoinnCARE makes errors at the deepest EC digit level (level 4), and provide qualitative examples? Are there biological patterns (e.g., convergent evolution) that systematically challenge the model?
7. How do the computational and memory costs of the hyperbolic GNN approach compare empirically to standard Euclidean GNNs, given the noted $O(nd)$ overhead?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents PoinnCARE, a novel framework for Enzyme Commission (EC) number prediction. It compellingly argues that existing methods fail by (1) not adequately modeling the hierarchical, tree-like structure of the EC classification system and (2) overlooking critical multi-modal data, specifically protein structure and active site information. The core proposal is to solve this by embedding multi-modal enzyme data (sequence, structure, and active site) into hyperbolic space, which is theoretically well-suited for hierarchical data.

### Strengths
(1) The paper's greatest strength is its clear and theoretically-grounded motivation. The introduction clearly identifies two major limitations of prior work: ignoring the EC hierarchy and overlooking multi-modal data . They also provide a strong theoretical case for why using hyperbolic space. 

(2) The experimental section is thorough, rigorous, and provides powerful evidence for the paper's claims. PoinnCARE consistently achieves the highest accuracy in nearly all cases. The authors have also provided extensive ablation studies to support the effectivness of their framework.

### Weaknesses
The most significant con I see of this paper is that it seems to require heavy hyperparmeters tuning. Table 8 lists the final hyperparameters, which is good for reproducibility. However, it reveals a potential weakness: the modality-weighting parameters are different for each test set. 

Before any learning occurs, the framework requires running multiple computationally intensive bioinformatics tools. This includes using Foldseek to compute all-vs-all structural similarity and Folddisco to compute all-vs-all active site motif similarity. This is more complex than a model that simply takes a protein sequence as input.

### Questions
Is it possible to further simplify the framework so that the overall complexity could be lower.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces PoinnCARE, a hyperbolic multi-modal learning framework for enzyme function classification. PoinnCARE integrates three modalities — protein sequence, 3D structure, and active site motif — into a unified representation. The representations are used to train a hyperbolic GNN in order to model the hierarchical relationships of the Enzyme Commission (EC) classification tree. Trained on an updated CARE benchmark dataset, PoinnCARE achieves state-of-the-art enzyme classification accuracy across several test sets, including low-homology and promiscuous enzymes.

### Strengths
- Originality: The integration of hyperbolic geometry with multi-modal structural representations (global structure + active site graphs) for enzyme classification is novel and elegant.
- The hierarchical nature of EC numbers fits naturally with hyperbolic geometry.
-  Combining sequence, structure, and active site information offers richer functional understanding than sequence-only baselines.
- Strong empirical performance: PoinnCARE consistently outperforms 12 state-of-the-art methods (ProtT5, ESM-2, CLEAN, etc.), especially on low-similarity and multi-functional enzyme subsets.
- Hyperbolic representations are robust and efficient: achieve comparable or better accuracy at much smaller dimensions (e.g., 32 vs. 512) compared to CLEAN representations.
Expanded the CARE dataset with active sites and structure for the protein-ml community.

### Weaknesses
Presentation clarity:
- While well-written, the methods section could better explain the intuition behind “graph diffusion” and “dual hyperbolic alignment” for readers unfamiliar with those concepts.

-Requiring active site motif limits generalization to unknown enzyme reactions or reactions that lack detailed enzymology studies.

### Questions
- How sensitive is the performance to the choice of hyperbolic dimension or curvature parameter?
- How does the model handle noisy or missing active-site data?
- What is the training time and computational cost compared to CLEAN?

### Soundness
3

### Presentation
3

### Contribution
3
