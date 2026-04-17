# SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Real-world temporal data often consists of multiple signal types recorded at irregular, asynchronous intervals. For instance, in the medical domain, different types of blood tests can be measured at different times and frequencies, resulting in fragmented and unevenly scattered temporal data. Similar issues of irregular sampling occur in other domains, such as the monitoring of large systems using event log files. Effectively learning from such data requires handling sets of temporal sparse and heterogeneous signals. In this work, we propose Super Mixing Additive Networks (SuperMAN), a novel and interpretable-by-design framework for learning directly from such heterogeneous signals, by modeling them as sets of implicit graphs. SuperMAN provides diverse interpretability capabilities, including node-level, graph-level, and subset-level importance, and enables practitioners to trade finer-grained interpretability for greater expressivity when domain priors are available. SuperMAN achieves state-of-the-art performance in real-world high-stakes tasks, including predicting Crohn’s disease onset and hospital length of stay from routine blood test measurements and detecting fake news. Furthermore, we demonstrate how SuperMAN’s interpretability properties assist in revealing disease development phase transitions and provide crucial insights in the healthcare domain.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents GMAN (Graph additive mixing network) for learning on graphs that represent irregularly sampled time signals. Results show good empirical performance on different datasets.

### Strengths
- GMAN provides good empirical predictive performance and interpretability
- Domain prior can be incorporated into method
- Some theoretical analysis is included

### Weaknesses
- In the table 1 for example, GMAN is shown to beat other baselines, but the authors state that signals are grouped based on common biomarker domain knowledge, and tuning biomarker sets are also performed. My concern is that the benefit of GMAN is incremental in accuracy, and other methods do not use such specific domain knowledge, thus are more widely applicable and performant.
- Novelty is limited, and contribution seems incremental. Prior methods Raindrop uses GNNs and SeFT uses deep sets, and heavily relies on GNAN which also uses node level and graph level representations.
- Theoretical analysis is not extensive, seems elementary, and does not capture major elements of learning with temporal irregular signals

### Questions
See weaknesses.

### Soundness
3

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
4

### Summary
GMAN extends Graph Neural Additive Networks by operating on sets of graphs representing irregular, heterogeneous temporal signals.

### Strengths
1. Addresses an under-explored irregular-sampling problem.
2. Good design combining additive interpretability with graph-set flexibility.
3. Compelling real-world demonstrations (Crohn’s disease, LOS prediction).

### Weaknesses
1. Evaluation section briefly described; lacks detailed baselines beyond GNAN.
2. Interpretability examples are qualitative; quantitative faithfulness metrics missing.
3. Computational efficiency compared with standard GNNs unclear.

### Questions
1. For the "interpretability-expressivity trade-off," the biological grouping of biomarkers shows the best performance, but it’s unclear if data-driven grouping (e.g., clustering) can achieve similar results without domain priors. Please add experiments to validate this and discuss when domain knowledge is indispensable.
2. GMAN is applied to path graphs (medical data) and tree graphs (fake news), but it’s unclear how it performs on cyclic or disconnected graph, please add experiments on such graph types to verify generalizability beyond the tested structures.
3. The paper uses AUPRC for imbalanced medical tasks but doesn’t discuss how class imbalance is handled during training (e.g., sampling strategies, loss functions), please clarify this and provide additional metrics (e.g., AUROC) for comprehensive evaluation.
4. The ablation replaces ExtGNAN with MLPs/identity mappings, but it doesn’t test if simpler additive GNN variants (e.g., GNAN with minor modifications) can achieve comparable performance, please add such baselines to highlight GMAN’s unique value.

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
This paper proposes GMAN, a model for learning from a set of graphs that represent irregular, asynchronous signals (e.g., per-biomarker paths in EHR; propagation trees in social media). The key idea is to (i) encode each graph with an Extended GNAN (ExtGNAN) that allows grouped (multivariate) feature functions to trade fine-grained feature-level interpretability for higher expressivity, and then (ii) mix multiple graphs via a DeepSets aggregator at the subset level, yielding additive scores at the node/graph/subset levels and thus transparent importance attributions. The paper proves GMAN is strictly more expressive than GNAN and that grouping multiple graphs into subsets further strictly increases expressivity; it also gives a recoverability result (four-point condition) and a complexity analysis. Empirically, GMAN is evaluated on ICU length-of-stay (PhysioNet 2012), Crohn’s disease onset, and GossipCop fake-news detection, reporting competitive performance with built-in interpretability; ablations show non-trivial drops when removing the DeepSets mixer, the distance kernel ρ, or replacing ExtGNAN.

### Strengths
* Well-motivated representation of irregular, multi-signal data: Modeling each signal trajectory as a path graph and learning over a set of such graphs avoids time-gridding/imputation artifacts; the paper also justifies recoverability of structure under a tree-metric condition and gives a clear O($m·K·n²·d\psi$) complexity with GPU-friendly implementation notes.
* Theory for interpretability–expressivity trade-off: ExtGNAN enables multivariate within-graph feature grouping; the authors prove GMAN > GNAN in expressivity and that subset-mixing of multiple graphs (DeepSets) is strictly more expressive than singleton mixing, aligning with additive-model literature and permutation-invariant set learning.
* Built-in, multi-grain interpretability that surfaces plausible clinical signals: Node-, graph-, and subset-level importances are direct contributions (not post-hoc), and qualitative analyses on P12/CD highlight clinically coherent markers (e.g., F-Cal/CRP; renal and hepatic panels) and meaningful subset curves.
* Ablations substantiate design choices: Replacing DeepSets with mean pooling, removing ρ, or downgrading ExtGNAN causes notable AUPRC drops (e.g., −20% with mean pooling; −12% without ρ), supporting the necessity of the proposed components.
* Experiments span clinical time-series and social-media propagation graphs (GossipCop) with appropriate baselines per structure family (sequence-like vs. general graphs), and GMAN outperforms strong GNN baselines on GossipCop.

### Weaknesses
* Novelty relative to GNAN could be made crisper in the writing: While theorems show strict expressivity gains, the narrative at times reads like an incremental extension of GNAN; emphasize the two-level mixing (within-graph feature grouping and across-graph subset mixing) and the new recoverability/complexity results as core contributions.

* Scalability: ExtGNAN performs dense pairwise aggregation across nodes, giving per-graph O(n²) cost and overall O(m·K·n²·dψ); this may limit very long trajectories (ICU with minute-level vitals) unless optimized kernels or sparsification are used. Benchmarks on long sequences would strengthen the case.

* Limited evidence of robustness & calibration: Results focus on discrimination (e.g., AUPRC/accuracy). Reliability (calibration curves/ECE), OOD generalization (e.g., cross-hospital or temporal drift), and fairness stratifications are missing; these are common asks in clinical ML.

* Codes are not avaialble at this stage, limiting its reproducibility

### Questions
* Can you report calibration (ECE/Reliability) and robustness (noise stress tests, temporal shift) on P12/CD?
* How does performance/interpretability scale beyond 48-hour P12 windows—can GMAN handle much denser ICU streams without subsampling given the O(n²) component? Any sparse kernel or Nyström-like approximation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Graph Mixing Additive Networks (GMAN), a novel, interpretable framework for learning from sets of sparse, irregular temporal graph signals. Using Extended Graph Neural Additive Networks (ExtGNAN), GMAN models nonlinear interactions within signal groups while preserving additive interpretability at the node, graph, or subset level. This approach balances expressiveness and interpretability, allowing user-defined signal grouping (domain knowledge) to enhance representational power. GMAN is theoretically proven more expressive than prior models like GNAN. It achieves state-of-the-art performance on high-stakes tasks (e.g., Crohn’s disease prediction, fake news detection), delivering meaningful insights via built-in attributions.

### Strengths
1. It handles irregular, heterogeneous data: GMAN introduces a novel way to learn from sets of sparse, irregular time-series signals without any resampling or imputation. By converting each signal into a graph (e.g. linking events by time), the model preserves timing gaps and irregular patterns that other methods might obscure. This design avoids the information loss incurred by aligning or filling in missing data, enabling the model to exploit the full richness of the raw data (e.g. varying intervals between events can inform the prediction).

2. Interpretable at Multiple Granularities: A key strength of GMAN is its inherently interpretable architecture. The additive structure means one can decompose the prediction into contributions from individual nodes, entire graphs, or subsets of graphs. It retains the feature-level and node-level interpretability of GNAN for those parts of the model that remain ungrouped, and additionally provides graph-level and subset-level importance scores for grouped components. These transparent contributions are directly linked to the final output (since GMAN sums them linearly), making it straightforward to explain what drove a prediction. This multi-level interpretability is especially valuable in domains like healthcare, where practitioners might ask which biomarkers (or groups of biomarkers) were most influential in a diagnosis.

3. Flexible Expressiveness via Grouping: GMAN offers a trade-off between model complexity and interpretability. By grouping features or signals, domain experts can inject prior knowledge about which inputs actually interact, enabling the model to capture nonlinear interactions within those groups. The theory guarantees that any such grouping strictly increases the model’s expressive power compared to keeping everything separate. Practically, this flexibility led to state-of-the-art performance in diverse tasks – from clinical predictions to fake news detection – indicating that GMAN can adapt to different data structures and learn complex patterns when needed. Notably, its success across domains (time-series medicine and graph-structured social data) showcases the approach’s generality and robustness. Additionally, the use of a learned time-distance function ρ in ExtGNAN means GMAN can flexibly model temporal influence (e.g. decaying or long-range effects) rather than relying on fixed time windows, which is a powerful design for temporal graphs.

### Weaknesses
1. While GMAN’s grouping mechanism is powerful, it requires a priori decisions about how to partition features or signals. The model’s performance can depend on choosing sensible groups – a process that may need domain expertise or extensive tuning. In the experiments, the authors manually tried several grouping schemes (including no grouping vs. clinically guided groupings) and selected the best performing one. This indicates an added complexity: users must either have prior knowledge to guide grouping or resort to hyperparameter search to find an optimal grouping strategy. In scenarios with little known domain structure, finding the right grouping could be challenging. Moreover, if signals that truly interact are mistakenly kept separate (or vice versa), the additive structure of GMAN might fail to capture those cross-signal effects, potentially hurting performance in such cases.

2. GMAN cannot maximize expressiveness and maintain full fine-grain interpretability at the same time – there is an inherent trade-off. When features are grouped or signals are combined into a subset, the model loses the ability to attribute importance to each individual feature or each individual graph within that group. Interpretability is then only available at the aggregate subset level. In practice this means, for example, one might know a collection of biomarkers as a whole was important, but not the exact contribution of each single biomarker in that group. If a use case demands insight at the most granular level for every feature/signal, GMAN would need to forego grouping – but then it essentially reduces to GNAN, which might yield weaker accuracy if important interactions exist. Thus, in scenarios where every individual feature’s impact needs to be distinctly understood, GMAN’s strength (grouped nonlinear modeling) becomes limited unless one is willing to sacrifice some clarity.

3. The GMAN architecture introduces more components than a standard model, which could raise computational complexity. Each subset of graphs has its own ExtGNAN (each with multiple neural networks for feature groups) and possibly a DeepSets aggregator, and each node in a graph aggregates messages from all other nodes (weighted by ρ). This additive all-pairs message passing can be costly for large graphs, potentially $O(n^2)$ per graph in the worst case. In their experiments, this was manageable (e.g. path graphs or moderate-sized biomedical time series), but for very large graphs or very large numbers of signals, training and inference might slow down. Additionally, representing certain data as a set of many graphs could inflate memory or computation: for instance, breaking a big tree into hundreds of path graphs means running many small GNN instances. There is also a risk of model complexity in terms of parameters – with separate neural networks for each feature group and each subset, the number of parameters grows, which might require careful regularization and ample data to avoid overfitting. While not reported as an issue in the paper, these factors suggest that GMAN could be less sample-efficient or slower to train compared to simpler architectures, especially if used naively on very large-scale problems.

### Questions
The proofs demonstrate that grouping signals increases expressivity (Theorems 3.1–3.2). Could the authors clarify what classes of real-world functions this additional expressivity enables beyond XOR-type examples, perhaps with a more domain-relevant illustration (e.g., nonlinear biomarker interactions)?

The model’s flexibility depends on user-defined graph/feature grouping. Could the authors discuss automatic or data-driven methods for learning optimal groupings, and whether improper grouping choices can significantly degrade performance?

### Soundness
2

### Presentation
3

### Contribution
2
