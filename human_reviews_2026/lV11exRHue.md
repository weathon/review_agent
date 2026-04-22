# StructLens: A Structural Lens for Language Models via Maximum Spanning Trees

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Language exhibits inherent structures,
a property that explains both language acquisition and language change. Given this characteristic, we expect language models to manifest internal structures as well.
While interpretability research has investigated the components of language models, existing approaches focus on local inter-token relationships within layers or modules (e.g., Multi-Head Attention), leaving global inter-layer relationships largely overlooked.
To address this gap, we introduce StructLens, an analytical framework designed to reveal how internal structures relate holistically through their inter-token connection within a layer.
StructLens constructs maximum spanning trees based on residual streams, analogous to dependency parsing, and leverages the tree properties to quantify inter-layer distance (or similarity) from a structural perspective.
Our findings demonstrate that StructLens yields an inter-layer similarity pattern that is distinctively different from conventional cosine similarity. Moreover, this structure-aware similarity proves to be beneficial for practical tasks, such as layer pruning, highlighting the effectiveness of structural analysis for understanding and optimizing language models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscript introduces StructLens, a graph-based analysis framework designed to uncover the internal structural organization of transformer-based language models. StructLens leverages the hidden representations of all tokens in a sentence to build a maximum spanning tree (MST) for each layer of the model. By analyzing inter-layer similarities among the MSTs, the authors identify three main processing stages -or "islands"-. Within each island, recurrent subtrees emerge across successive layers, revealing linguistically and functionally relevant structures characteristic of that region.
Furthermore, in the context of layer pruning, the same MST-derived observables enable a more principled and effective selection strategy than the simple cosine similarity, facilitating the identification of layers that can be ablated with minimal performance loss.

### Strengths
I find it interesting and powerful this way of aggregating the information of a whole sentence into a compact structured object as the MST, as most of the literature typically apply a much simpler pooling operation like mean, max or last. In this sense, the MST seems to be a compact and promising candidate summary statistics for the whole sentence's hidden states, as it potentially encodes meaningful language structures.

### Weaknesses
1. Is there a reason why the authors decided to inspect only C/MMLU entries? Since they are all similar to each other, I am afraid there might be some bias due to the specific structure of the inputs. A more diversified set (maybe like the one used for layer-removal calibration?) would imply a stronger robustness for the method. I also think the paper would benefit by the discussion of subtrees of more examples, which would allow to better inspect which language structures are captured by the framework.
2. The authors give a precise functional form for the operator $g(\cdot)$ used to create the adjacency graph. To which extent do the results depend on such a choice? i.e., what happens if one multiplies $||h_i^l-h_j^l||$ by a constant or uses a gaussian/power law distribution instead?

### Questions
1. The plots in Figures 1,2 (5,6) are obtained by averaging across the 50 samples the metrics computed for each sequence separately? Figure 3 and Table 4 are related to a single example or averaged across the 50 samples? In the latter case, please report error bars and standard deviations. The same goes for Figure 4: if this is the results across the 10 calibration examples, report the error bars. 
2.  Are the patterns of Tables 1 and 2 reproduced (at least partially or with similar characteristics) when other samples are considered? 
3. The layer-vs-layer patterns found in the study appear to closely match those presented in (at least) other two works, despite being extracted in a different way. I think it would be nice to mention them a possibly comment on such similarities.
	- Cheng et al. Emergence of a High-Dimensional Abstraction Phase in Language Transformers, 2025
	- Wolfram et al. Layers at Similar Depths Generate Similar Activations Across LLM Architectures, 2025
4. How do pruning results compare to the previous findings in literature achieved through different methods? Are they somewhat coherent or they are providing new insights?
5. I have some clarification questions concerning the clustering procedure (4.1 and C):
	- is the cluster partitioning of the layers computed for each MMLU sample separately using the various metrics as feature/affinity measure?
	- The ARI score is computed between two lists of indices. One is related to the clustering based on a specific metric. What about the second assumed as the reference/ground truth partition?
6. Table 9 is referred to Llama or Qwen tokenization? Since Table 1 and Table 2 have a different index-token pairings (for Llama 40 is "." while for Qwen it is "(B"  ), which I guess is due to the different tokenizers of the two models.

### Soundness
3

### Presentation
2

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
The paper addresses the problem that existing model analysis methods rely on partial modules or tokens, failing to provide global inter-layer relationships. To fill this gap, the authors propose StructLens, a novel framework that calculates the similarity between tokens at each layer, constructs MSTs to enable a comprehensive token analysis and proposes three-based metrics for global model analysis. Layer pruning experiments on Llama 3.1 and Qwen 2.5 demonstrate that these three metrics achieve better results than baseline (from ShortGPT).
The paper’s approach is novel, linking interpretability to language structure to provide more structured information to the model. It also uses frequent subtrees to partition the model into clusters, linking structured layers together and reducing the workload for model analysis.
However, the paper conducts relatively few experiments, and the authors do not conduct in-depth analysis of the experimental results. Although the paper positions its research within the context of interpretability, it lacks substantive discussion and analysis of model interpretability, which makes the conclusions less persuasive and less interpretable.

### Strengths
The paper is clearly structured, and the framework is highly practical. The choice of algorithms and the detailed methodological treatment are commendable, particularly the consideration of a single root node to ensure consistency. The tree-based indicators proposed in the paper are effectively applied and validated in these experiments. Additionally, the case study presented in *Section 4.2: FREQUENT SUBTREES* is a clever choice, effectively illustrating the relationship between language structure and hierarchical relationships within the model.

### Weaknesses
(1) The feasibility of MST calculation and its related algorithms requires further verification. For very large models or long token sequences, the computational cost of this algorithm can be substantial. Moreover, the **Edge-Edit** and **Tree-Edit** indicators involve multiple operations which may further reduce computational efficiency.

(2) The paper only conducts experiments only on **Llama 3.1 and Qwen 2.5**, limiting the number of models studied. The choice of models could be improved, especially for **Qwen 2.5**. In the *Section 5. LAYER PRUNING THROUGH* , after pruning some layers, the accuracy obtained deviates significantly from the baseline, with differences exceeding 10%.

(3) The experiments primarily focus on analyzing the three proposed metrics. The obtained model structure information is not connected with the language structure information. Tables 1-3 can be analyzed to show that the model reflects the language structure, but the article does not analyze the connection between these structures.

(4) The paper lacks sufficient experimental details, particularly regarding the implementation described in Section *4.2 FREQUENT SUBTREES.* For examples, it is unclear how the layer indices within each cluster are determined, which makes it difficult to fully reproduce or evaluate the proposed method.

### Questions
(1) In the *Section 4.2 FREQUENT SUBTREES*, “**islands”** are observed in the **"Edge-Edit"** similarity graph when dividing clusters. However, it remains unclear why the **“Cos-Struct”** indicator is used to determine the number of clusters.

(2) In the experiment, how is the model layers divided into clusters after the number of clusters is determined? The details of the division are not shown.

(3) The paper does not explicitly state whether the feasibility of the three selected metrics has been verified. Furthermore, the **“Tree-Edit”** metric requires computational complexity, relying on multiple operands. Did the authors discuss or experiment with its computational cost or possible optimization solutions?

(4) In the *Section 5. LAYER PRUNING THROUGH STRUCTLENS*, the authors present the results of layer pruning experiments on Llama 3.1 and Qwen 2.5, but lack in-depth analysis. Pruning experiments are often used in interpretability research to verify the function or contribution of each layer of the model. However, although the accuracy of the model is improved compared to the baseline(**CosBaseBI**) under the guidance of the three indicators(**CosStructBI, TreeBI and EdgeBI**), the authors do not further explore the reasons for this improvement, nor do they establish a connection between the performance changes after layer pruning and the interpretability of the model.

(5) In the experimental sections, the authors do not explain whether there is any prioritization or weighting among the three metrics. The results appear to be based on experimental performance, but the results vary widely across metrics, lacking a unified evaluation benchmark or clear comparison criteria. Was there a unified standard or prioritization for metric selection?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces StructLens, a method for analyzing LLMs based on constructing maximum spanning trees (MSTs) from token representations at each layer. The method builds trees using L2 distances between residual stream representations, and proposes three similarity metrics: Cos-Struct, Tree-Edit Distance, and Edge-Edit Distance. The authors apply the method to Qwen 2.5 7B and LLaMA 3.1 8B and  models, revealing distinctive "island" patterns of inter-layer similarity with a slight correlation model confidence after layer pruning. Layer pruning experiments demonstrate mixed results with different metrics leading at different pruning rates/languages.

### Strengths
- StructLens is an original approach to language model interpretability, offering a global structural perspective that complements existing token-level and attention-based analyses.
- The paper provides clear mathematical formulations for tree construction and for the presented similarity metrics.
- The exploration of structure-aware metrics for layer pruning is practical and connects interpretability with model compression.

### Weaknesses
- Only 50 instances per dataset is a small sample to obtain reliable or generalizable insights.
- The results obtained in Section 4.2 are on a single instance of MMLU, which is too limited to extract conclusions (and occupy an entire page).
- The layer pruning results in Table 5 are inconsistent. In some cases, structure-aware metrics underperform base cosine similarity. There is no statistical significance analysis.
- The findings presented in the paper, e.g., the "island" patterns in Edge-Edit similarity) are mostly qualitative and lack rigorous statistical validation.
- Results are limited to MMLU/CMMLU QA-style datasets.

### Questions
- Do we expect the different proposed similarity metrics to exhibit such different results (Figures 1 and 2)?
- Section 4.3 already deals with removing layers based on the magnitude of residual stream transformation. Should it be in Section 5 "Layer Pruning Through Structure Lens"?

 line 321: from left ot right ->  from left to right
 line 470: "hiest" -> "highest"

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes StructLens to construct structure-aware similarity metrics for layers in Large Language Models (LLM).  The method uses token representations at each transformer layer (taken from the residual stream) to define token distances and weights on forward edges on the graph of tokens, and then constructs a maximum spanning tree (MST). By comparing the MSTs of different layers, the authors propose three structure-aware distances (Cos-Struct, Tree-Edit, Edge-Edit). The approach reveals "islands" of similar layers and is used to guide layer pruning.

### Strengths
The idea of using a tree to define a structure summary of a layer's representation is interesting. The formulas and construction are clearly stated. The empirical patterns are visually compelling. The demonstrations of the correlation with confidence degradation and the pruning case study provide practical usefulness.

### Weaknesses
Some simple baselines are not compared, and key design choices are insufficiently justified. Without these pieces, claims about superiority and practical utility are not yet established.

### 1. Baseline to be compared

The paper contrasts StructLens primarily with token-aligned cosine (Eq. 6). However, global inter-layer similarity is standardly assessed with Centered Kernel Alignment (CKA) and close relatives SVCCA/PWCCA. For example, for two layer-representation matrices $X \in \mathbb{R}^{N \times d}$ and $Y \in \mathbb{R}^{N \times d}$, one may compute the linear CKA as 
$$\mathrm{CKA}(X, Y)=\frac{\left\|X^{\top} Y\right\|\_F^2}{\left\|X^{\top} X\right\|\_F\left\|Y^{\top} Y\right\|\_F},$$
and nonlinear CKA can also be defined by replacing the linear inner products by $K_{i j}=\exp \left(-\frac{\left\|x_i-x_j\right\|^2}{2 \sigma^2}\right)$ (see Kornblith et al, 2019). 
Without comparing with these methods, it is unclear whether StructLens provides any new insight than established global metrics.

Reference: _Kornblith, S., Norouzi, M., Lee, H. and Hinton, G., 2019, May. Similarity of neural network representations revisited. In International Conference on Machine Learning (pp. 3519-3529). PMlR._

### 2. Unjustified specifications

The method chooses to construct a single MST per layer, to restrict forward edges, and to use an L2-based affinity $\exp \left(-\left\|h_i-h_j\right\|\right)$. These are strong modeling choices (since attention patterns can exhibit multi-parent and backward dependencies). The paper does not motivate these choices before stating them. The paper may need ablations to show that these specifications are reasonable. If findings hinge on a particular set of specifications, readers should be explicitly informed about the scope and limitations.

### 3. Limited numerical comparisons 

In Table 4, the paper evaluates three StructLens heuristics (Cos-StructBI, TreeBI, EdgeBI) against a single cosine baseline (CosBaseBI). No single StructLens variant consistently dominates across models, removal ratios, and metrics. The narrative tends to highlight that "one of the three" beats cosine in each block, which risks post-hoc highlighting and can inflate apparent gains.

Furthermore, Table 4 lacks uncertainty estimates and stability checks, which undermines the claimed superiority of the proposed method.

### 4. Typos and wrong citations

Line 038: The paper on MST algorithm (Chu & Liu, 1965) is cited after the sentence "cosine similarity that is employed for inter-layer analysis"
Line 321: "left ot right"
Line 470: "Cos-Struct achieves hiest accuracy"

### Questions
1. Please add CKA (linear; RBF optional) as global, token-agnostic baselines computed on the same residual states and samples as StructLens. 

2. Please motivate and justify the choices of specification in your STUCTLENS, in particular, the function g, forward edges, and the MST. 

3. Table 4 shows no single StructLens variant consistently dominates CosBaseBI across models and removal ratios. How do you reconcile this with your superiority claim? If the contribution is conditional, please state the conditions explicitly and provide uncertainty quantifications. 

4. Please correct the typos and the wrong placement of citations.

### Soundness
2

### Presentation
3

### Contribution
3
