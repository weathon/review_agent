# Is Graph Mixup Beneficial? Investigating Interpolation And Empirical Performance of Graph Mixup Methods

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 4, 6, 8

## Abstract
Mixup is a widely used data augmentation technique that constructs new training
examples by interpolating between existing ones. While simple and effective in
domains like vision and language, applying mixup to graph data is non-trivial
and independent empirical evidence for its effectiveness is lacking. To fill this
gap, we conducted an independent evaluation following established evaluation
protocols for graph classification and found that none of the state-of-the-art mixup
methods yielded statistically significant improvements over the no-mixup baseline.
To obtain further insights, we analyzed the graphs generated from existing mixup
methods from an interpolation perspective using the graph edit distance. We found
that (i) many mixup methods failed to interpolate well, (ii) high interpolation error
led to performance degradation, and (iii) even optimal interpolation did not lead to
performance improvements. Our findings highlight the need for a more rigorous
exploration and evaluation of mixup for graphs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The work focuses on several interesting questions regarding graph mixup problems and methods: Does the interpolation align with the graph edit distance (GED)? Does misalignment lead to poor empirical performance? Does GED-aligned interpolation improve performance compared to the baseline? The paper first reviews different graph mixup methods, ranging from structure-based to embedding-based approaches, and then conducts empirical analyses to examine the benefits of mixup. Next, it analyzes the cause of performance differences by comparing interpolation errors with GED. The results are interesting. In summary, this is an engaging direction, and the paper provides valuable insights related to GED. However, it deserves clearer presentation to highlight its contributions—such as whether GED has been used before to evaluate or analyze mixup, whether GED-based mixup is newly proposed in this work, and what the generalizability and limitations of GED and the paper’s conclusions are in graph tasks.

### Strengths
1. The work focuses on several interesting questions regarding graph mixup problems and methods and is organized to address them step by step, making the paper relatively easy to follow.

2. The idea of GED-aligned mixup is interesting.

### Weaknesses
## 1. Evaluation tasks should be extended
One important application of mixup is data augmentation to address the issue of limited training data. This is a common scenario for molecular tasks, which can naturally be modeled as graphs. However, the current work only includes four datasets, with only one molecular dataset, and it is unclear what the data distributions look like—whether the labeled data are imbalanced and whether data augmentation is truly needed.

Conclusions such as “even optimal interpolation did not lead to performance improvement” and “graph mixup provided no significant improvement over the no-mixup baseline” are too arbitrary and dataset-specific. It is important to ensure that these conclusions are generalizable across different types of graph learning problems.

To make the conclusions more robust and convincing, the study should be extended to broader molecular benchmarks, such as those from OGB, MoleculeNet, or Polaris. Many molecular tasks are regression problems, which is another missing aspect in the paper’s current analysis. The tasks should also explicitly consider issues related to data imbalance and small-data regimes.

## 2. Insufficient coverage of existing mixup algorithms

The set of reviewed and evaluated mixup algorithms appears limited. Given the strong statement that “graph mixup provided no significant improvement over the no-mixup baseline, which questions the practical benefits of graph mixup,” several aspects remain unclear: What exactly is the no-mixup baseline? Which mixup variants are currently evaluated, and do they represent all major categories of mixup methods? Currently, only one embedding-based mixup approach is included. For a more systematic and fair evaluation, additional baselines should be considered. For example, there exists a line of research on graph rationale-based methods, which share conceptual similarities with mixup (e.g., selectively preserving interpretable parts of graphs). Including such methods would make the empirical comparison more comprehensive.


## 3. Limitations and generalizability of the mixup algorithms

Different mixup methods may vary in efficiency. How efficient is the GED-based mixup? An analysis of the training time and a comparison with other baselines should be included.

Another question concerns how different mixup methods generalize to molecular and other graph-structured data, since different graphs may have different node and edge features. When performing structural mixup, do the authors also mix the feature dimensions of nodes and edges? How are these aspects handled, and to what extent do these design choices affect the model’s performance?

### Questions
1. Is the GED-Mixup method newly proposed in this paper? How does it differ from or simplify the EPIC method?

2. How much computation time does the GED-Mixup process require? Is the runtime related to the dataset size or the graph size? How long does each training epoch take with this method?

3. Can GED-Mixup be applied to molecular graphs? Specifically, can it mix nodes and edges that contain multiple features, including both discrete and continuous values?

4. What types of atom and bond features are used in the MUTAG dataset?

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
3

### Summary
This paper revisits graph mixup techniques for graph classification. It conducts an empirical evaluation of several state-of-the-art graph mixup methods and analyzes their behavior through the lens of graph edit distance. Through both statistical testing and interpolation-based analysis, the study finds that current graph mixup methods provide no significant improvement over the no-mixup baseline, even when interpolation quality is high.

### Strengths
- The paper provides a unified and principled analysis of various graph mixup methods using edit distance as a common framework, which is conceptually elegant and insightful.
- The paper is well written and easy to follow. The discussion of related work is clear and well organized, providing an informative overview of existing graph mixup approaches.

### Weaknesses
- The experimental scope appears limited. The study evaluates only four datasets and contain relatively small graphs. This dataset selection may make it difficult to generalize the findings to other graph domains. Since the subsequent analyses rely heavily on these empirical results, expanding the dataset diversity would greatly strengthen the study’s conclusions.
- Figure 2 provides intriguing evidence that lower mIE values are associated with better accuracy; however, this relationship remains somewhat inconclusive, as different mixup methods vary in several aspects beyond mIE. To better support the argument, the authors could run an ablation study using GED-Mixup. They could create edit sets that are not perfectly optimal (with higher mIE values) and gradually adjust how suboptimal they are. Observing how performance changes in this setting would help clarify whether mIE actually affects accuracy.
- The domain composition of the datasets also warrants consideration. GED is particularly appropriate for molecular or bioinformatics graphs, where small structural edits correspond to meaningful chemical or biological variations. This might partly explain why interpolation quality correlates strongly with performance in these datasets. In contrast, on the IMDB-BINARY social-network dataset, methods with low mIE (e.g., GED-Mixup, SubMix, If-Mixup) do not exhibit clear performance advantages, suggesting that the observed trend could be domain-specific rather than universal.

### Questions
- The study evaluates only four small-scale TUDataset benchmarks. How confident can we be that the findings generalize to larger or more diverse graph domains, such as molecular graphs with higher node counts, biochemical interaction networks, or large social networks?
- Results on the IMDB-BINARY dataset appear inconsistent with the findings from molecular datasets. Is the link between interpolation fidelity and downstream accuracy dependent on domain semantics, such as chemically meaningful edit operations?

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
4

### Summary
The paper analyzes whether graph mixup actually improves graph classification performance. It first conducts an empirical comparison of existing graph mixup methods against baselines on four benchmark datasets, finding that none achieve statistically significant improvements. To further test generality, the authors perform a pooled analysis across datasets and models, showing that even when aggregated, mixup methods fail to yield consistent benefits. They then introduce an interpolation-based analysis framework using the graph edit distance to quantify interpolation error and assess the quality of augmented graphs. Additionally, they propose a mixup method based on optimal graph alignment, termed GED-Mixup. Their results show that most existing methods have high interpolation error, and that while better interpolation tends to correlate with improved performance, even optimal interpolation does not lead to significant gains.

### Strengths
1. The paper presents an independent and systematic empirical evaluation of existing graph mixup methods using a unified experimental setup and pooled analysis. This contributes to a clearer understanding of the empirical effectiveness of mixup in graph classification, addressing inconsistencies in prior studies.

2. The introduction of an interpolation-based metric using graph edit distance is a useful addition, providing a quantitative way to assess how well mixup outputs interpolate between input graphs. This analysis helps connect structural properties of the generated graphs with their empirical performance - an aspect that has been largely overlooked in earlier work.

### Weaknesses
1. The paper shows that existing graph mixup methods do not yield significant performance gains; however, it remains unclear why mixup fails. The authors demonstrate that many methods produce poor interpolations, yet even optimal interpolation (via GED-Mixup) does not improve accuracy significantly (Figure 2). This raises an unanswered question about the underlying cause of mixup’s ineffectiveness. The interpretation would benefit from a deeper diagnostic analysis.

2. The paper also overemphasizes negative results without exploring other potential benefits of mixup. Prior work suggests that mixup can improve robustness to topology perturbations and label noise, but this study focuses solely on classification accuracy. A discussion or evaluation of such alternative objectives would provide a more balanced perspective and clarify whether mixup is universally ineffective or only for accuracy metrics.

3. The evaluation scope is limited to relatively small TU datasets. Including larger and more diverse benchmarks (e.g., Reddit, DD) would strengthen the conclusions and assess generalizability to real-world or large-scale graph settings.

4. The proposed GED-Mixup method is interesting but computationally impractical for larger graphs as mentioned in the paper. The paper does not discuss viable approximations or scalable alternatives, leaving open the question of how GED-based interpolation could be applied in realistic scenarios.

5. Finally, as a suggestion, it would be valuable to compare newer methods such as MomentMixup (which mixes graph moment vectors and may reduce interpolation error) and SIGL (which modifies alignment in G-Mixup). Evaluating these under the proposed interpolation framework could yield further insights into the design of effective graph mixup strategies.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a graph mix-up method, a graph generator methodology which merges two input graphs, based on graph edit distance which is suitable for typically small graphs. 

The authors use a novel evaluation methodology focused on interpolation error. The authors demonstrate that prior mixup works failed to generate graphs which interpolate between their inputs.

### Strengths
1. The authors demonstrate an important empirical finding in this research area, using a novel evaluation methodology. Prior mix-up works are fairly niche as a graph generative model, however, within this prior work the negative finding of structural coherence is very significant.  

Furthermore, the presentation of the work is simple and understandable to a general AI research audience. This paper could be convincing for further work in this area. 

2. The GED-mixup method is well motivated and suitable in the many domains for small graphs. This seems like a reasonable assumption, where higher controllability is be better suited for smaller graphs; large graph generation could be bracketed as work of a graph foundational model, this is fine.

3. The authors scope their research questions well and empirically support each of them. The two contributions (Sec 1) are significant.

### Weaknesses
1. Over-reliance on fidelity: the authors argue but don't demonstrate the utility of measures such as mIE. That is, what is the qualitative impact of methods with similar ACC but higher mIE, e.g. in Fig 2? Similarly, the authors don't present an evaluation of downstream robustness, e.g. for distribution shift, etc, which are the common use-cases for graph augmentation. The same critique is true: the graph generator distribution need not necessarily have good mIE if it adds robustness along another problem dimension.

2. The three levels of pooled analysis are difficult to follow and are not well reflected in the figures. e.g. is fig 2 representative under A2 assumptions? Is Fig 3 presented under A3? More space could be dedicated to contrasting results at these pooling levels.

### Questions
1. What is the downstream effect of mixup with high ACC and high mIE (e.g. Fig 2)? 

2. Are there applications where high interpolation fidelity might not be best?

### Soundness
4

### Presentation
3

### Contribution
4
