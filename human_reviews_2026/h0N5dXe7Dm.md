# Multi-Scale Adaptive Hypergraph Learning for High-Order Brain Connectivity Analysis

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Understanding complex interactions between brain regions is critical for early neurodegenerative disease classification such as Alzheimer’s Disease (AD) and Parkinson’s Disease (PD). Graph-based models are typically employed to investigate brain networks with regional features and their interconnectivity. However, traditional approaches primarily focus on pairwise node interactions between directly connected nodes, limiting their ability to capture higher-order dependencies from multiple brain regions. Although hypergraph-based approaches have been proposed to capture higher-order relations beyond pair-wise connectivity, many existing methods rely on predefined hyperedges or restrict learning to hyperedge weights, limiting their flexibility and ability to capture multi-resolution structural patterns. In this regard, we introduce an adaptive multi-scale hypergraph learning framework, i.e., MASH, which constructs hierarchical node features and dynamically learns high-order interaction through continuous hyperedge construction over multi-resolution graph signals. Through extensive experiments on brain network benchmarks, we demonstrate the superiority of MASH by improving classification of different disease stages. Our model further identifies key regions of interest (ROIs) and their group-wise interactions from the learned hyperedges that are associated with disease progression, highlighting its potential as a powerful tool for brain network analysis with neurodegenerative disorders.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes MASH (Multi-Scale Adaptive Hypergraph Learning), which couples learnable multi-scale spectral wavelet representations with a learnable hypergraph incidence to capture high-order (group-wise) dependencies in brain networks for disease staging. Experiments on ADNI and PPMI show improved classification and interpretable hyperedges that highlight biologically plausible ROIs.

### Strengths
1.The proposed MASH framework effectively integrates multi-scale spectral graph wavelet representations with a learnable hypergraph incidence matrix. This design enables the model to adaptively discover high-order relationships among brain regions while maintaining spectral interpretability.
2.The authors conduct extensive experiments on two benchmarks (ADNI and PPMI), including classification tasks, ablation studies, sensitivity analyses, and interpretability visualizations. The results demonstrate consistent improvements over strong baselines (e.g., GCN, HGNN, HyperGT, etc.), supporting the robustness and generalizability of the proposed approach.
3.Beyond performance gains, the paper provides interpretable insights by visualizing learned hyperedges and identifying biologically plausible regions of interest (ROIs). Additionally, it includes theoretical propositions linking spectral scales to hyperedge expansion, which enhances both conceptual clarity and credibility of the model.

### Weaknesses
1.The abstract currently motivates the work by stating that “earlier hypergraph methods were designed to overcome the limitations of pairwise relations,” while the actual contribution of this paper is an improvement upon existing hypergraph approaches. The authors should revise the abstract to (a) acknowledge that hypergraphs have already been used to capture higher-order relations, (b) explicitly identify the shortcomings of previous hypergraph methods, and (c) clearly position MASH as addressing these specific limitations.
2.The final paragraph of the introduction only cites 2016 and 2019 works as examples of “rigid” hypergraph constructions. However, numerous studies from 2020–2025 have proposed dynamic, adaptive, or learnable hyperedges and hyperedge-weight learning. The authors should update the introduction to reflect this recent progress and clearly distinguish MASH from these newer approaches.
3.In addition, the Related Work section should include representative dynamic/learnable hypergraph studies from 2025 and articulate how MASH differs from them—not only functionally but also in its relevance to brain-network analysis (e.g., scalability, interpretability, or multi-scale spectral modeling).
4.The experimental comparisons include baselines only up to 2024. Given that several 2025 works on dynamic or learnable hypergraphs have been released, the absence of these methods weakens the empirical credibility. The authors should either add these recent baselines or explicitly justify their exclusion.

### Questions
1.The method uses a combination of one low-pass and (J-1) band-pass kernels. Was an ablation study conducted to justify this specific design choice? For instance, what would be the performance impact of using only low-pass or only band-pass kernels across all scales? 
2.The paper claims a key difference from dwHGCN is the dynamic refinement of connectivity patterns rather than just updating hyperedge weights. Could the authors elaborate on this distinction with a more concrete example or analysis? How significant is the performance gain attributable specifically to this dynamic structure learning, as opposed to simply having multiple scales? 
3.The paper presents qualitative visualizations of learned hyperedges and associated ROIs, but does not quantify their significance. Without permutation or bootstrap tests, it is unclear whether the identified ROIs are statistically meaningful or could arise by chance. A quantitative validation (e.g., comparing with random or shuffled hyperedges) would make the interpretability claims more convincing.

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
3

### Summary
The authors propose an adaptive multi-scale hypergraph learning framework named MASH to address the limitations of existing brain network analysis methods in capturing high-order dependencies among multiple brain regions. By constructing hierarchical node features and dynamically learning high-order interactions among hyperedges, MASH captures complex interactions that go beyond pairwise structural connectivity. Extensive experiments demonstrate the effectiveness of MASH.

### Strengths
1. Solid theoretical foundation. The paper demonstrates considerable theoretical depth and presents a method of notable theoretical novelty.
2. Comprehensive experiments were conducted to validate the effectiveness of the proposed method. Beyond comparative experiments and ablation studies, the authors also discussed model complexity and provided biological interpretations, offering plausible explanations for the performance improvement from a neuroscience perspective.
3. The introduction is logically structured, with the research problem clearly articulated by the authors.

### Weaknesses
1. The authors claim that "graph convolution layers indirectly consider high-order interactions at the cost of the oversmoothing problem." However, numerous Transformer-based brain network representation learning methods have been proposed — and indeed summarized by the authors in the related work section — which effectively capture long-range dependencies and global graph structural information without suffering from oversmoothing. Could the authors clearly explain why these Transformer-based approaches remain inadequate for capturing high-order associations among brain regions?
2. The methodology section lacks clarity. Both Adaptive Multi-Scale Feature Filtering and Multi-Scale Hypergraph Structure Learning are crucial components of this work, yet the descriptions of these two modules remain somewhat vague.

### Questions
1. Could the authors provide a more detailed explanation of the workflow for Adaptive Multi-Scale Feature Filtering?
2. Based on the authors' descriptions in lines 132 and 182, where each wavelet basis possesses a specific scale $s$, how was the number of wavelet bases determined? Was the number of scales made learnable within the framework?

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
4

### Summary
Existing research primarily focuses on pairwise interactions between nodes and overlooks the challenge of capturing complex dependencies among multiple brain regions. To address this limitation, this paper proposes an adaptive multi-scale hypergraph learning framework. The proposed approach constructs hierarchical node features and dynamically learns high-order interactions through hyperedges. It is claimed that the proposed approach can effectively capture the complex relationships among multiple brain regions.

### Strengths
S1. The study innovatively introduces multi-scale wavelet coupling to capture how these higher-order dependencies evolve across different scales.

S2. The authors conducted extensive experiments. The results demonstrate both the effectiveness and interpretability of the proposed method.

### Weaknesses
S1. This paper focuses on capturing multi-scale high-order relationships in brain graphs. However, the existence of such multi-scale high-order relationships in brain graphs, as well as the precise definition of these relationships, remains unclear.

S2. Ref [1] also employed a hypergraph-based approach to capture high-order relationships in brain graphs. This paper doesn't include a comparison with [1] nor explicitly clarify the differences between the two methods.

S3. The paper’s use of the term “multi-scale” is potentially misleading. In brain network analysis, “multi-scale” typically refers to approaches that construct brain networks for the same subject across different atlases or spatial resolutions [2–4]. The authors should clarify their terminology to avoid any confusion.

S4. Although the authors emphasize the biological significance of hyperedges as "sets of co-varying ROIs," the paper lacks relevant case studies illustrating the key hyperedges learned by the model that possess significant classification power. It remains unclear whether these hyperedges align with known AD/PD-related neural circuits or correspond to clinical outcomes.

[1] Learning High-Order Relationships of Brain Regions. ICML2024.

[2] A mutual multi-scale triplet graph convolutional network for classification of brain disorders using functional or structural connectivity. TMI 2021.

[3] Mamf-gcn: Multi-scale adaptive multi-channel fusion deep graph convolutional network for predicting mental disorder. CBM2022.

[4] A multi-scale multi-hop graph convolution network for predicting fluid intelligence via functional connectivity. BIBM2022.

### Questions
Q1. How is the upper limit of the number of hyperedges defined across different datasets?

Q2. How is the initial matrix generated? Is it randomly generated or generated under other constraints? Will different generation methods affect the final performance?

Q3. Is it possible to use other backbone alternatives to the transformer architecture? 

Q4. It can be observed that on the PPMI dataset, the performance degradation is most significant when MST is removed. Does this mean that in this method, Transformer contributes more compared to wavelet transform?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces MASH, a novel framework for modeling high-order brain connectivity using an adaptive multi-scale hypergraph learning approach. It integrates graph wavelet-based multi-resolution filtering and dynamic hyperedge learning to capture both local and global structural relationships in brain networks. MASH is evaluated on two major neuroimaging datasets ADNI and PPMI demonstrating superior performance over 15 state-of-the-art graph and hypergraph baselines. The model further provides interpretability by identifying key brain regions (ROIs) and their group interactions associated with disease progression.

### Strengths
1. Innovative combination of graph wavelet transforms for adaptive multi-scale representation with learned hyperedges for high-order relational modeling.
2. Comprehensive experiments on two independent, large-scale datasets (ADNI, PPMI). Consistent and statistically significant performance improvements (2–7% absolute gains) across accuracy, precision, recall, and F1-score.
3. Identifies disease-relevant ROIs (e.g., hippocampus, thalamus, amygdala) with plausible neurological interpretations. Demonstrates hemispheric symmetry and subcortical prominence, aligning with known disease mechanisms.

### Weaknesses
1. The combination of multi-scale filtering, dynamic hyperedge construction, and transformer layers may be computationally expensive for larger brain graphs. The paper does not explicitly report runtime or memory overhead compared to baselines.
2. Evaluation is limited to ADNI and PPMI; additional validation on other disorders or multi-site datasets would strengthen claims of generality. External test sets or cross-study generalization are not explored.
3. Although ROI identification is discussed, causal or mechanistic interpretations of learned hyperedges are not deeply analyzed. Quantitative validation (e.g., comparison to known biomarkers or clinical scores) is limited.

### Questions
1. How does MASH scale computationally with increasing node counts or number of scales (J)? Are there mechanisms to limit the exponential growth of hyperedges?
2. How stable are the learned hyperedges across folds or random initializations?
3. Could MASH be adapted to handle temporal dynamics? For example, fMRI data?

### Soundness
3

### Presentation
4

### Contribution
3
