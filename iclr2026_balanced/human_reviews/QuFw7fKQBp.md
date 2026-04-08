## Human Reviewer 1

### Summary
Test-time adaptation optimizes a source-trained model at inference to handle unseen distribution shifts. It typically minimizes prediction entropy directly or uses pseudo-labels, an implicit form of entropy minimization. The paper argues that these approaches overlook temporal reliability and the semantic structure of the label space, and proposes SURE, which regularizes predictions via a Prototype Reliability Graph (PRG). The PRG captures semantic affinity among classes and stabilizes confidence over time to improve reliability. Across benchmarks, the framework reports consistent gains over prior methods.

### Strengths
The claim that entropy minimization is not always a reliable signal for adaptation is reasonable; however, the argument that prototypes propagate noise and destabilize adaptation requires stronger evidence.

The integration of model predictions, prototypes, and a graph structure augmented with language-based semantics to correct outputs under distribution shift in VLMs is compelling and should be emphasized more clearly in the abstract and introduction.

### Weaknesses
The performance improvement is not significant or even as expected, given the proposal of the paper. This is further evident in the ablation study as well. 
There are several aspects of the paper that I was not able to follow, and these have been detailed in my questions below. I would also like to know why one of the baselines [1] was not included as a baseline.

### Questions
The method’s improvements over MAP-adjusted and pseudo-label baselines are modest on a per-dataset basis, which contradicts expectations for a framework that combines multiple techniques under SURE.

The introduction lacks citations for confidence thresholding, making it difficult to evaluate known limitations and design choices.
Lines 045–047 in the introduction are hard to parse, and “class-level prediction” should be defined precisely.

The paper introduces numerous new terms for the proposal, creating inconsistency and confusion across sections; keep the name and core terminology consistent across the abstract, introduction, related work, and methodology.

The motivation for using pseudo-label confidence at L082 is unclear, given the earlier critique that pseudo-labels can be inconsistent and that high-confidence misclassifications occur; reconcile this tension explicitly.

Add a citation at L90 to support the claim that the formulation reflects information-theoretic intuition.

In Section 4.2, specify precise graph notation, including symbols for nodes, edges, messages, and update rules.

From L066 onward, explain how pseudo labels are made reliable and how confidence values are computed and validated.
Despite leveraging graph structure, Table 4 shows limited gains on some datasets, with the prototypes-only variant being the strongest in several cases. An analysis is needed to determine when and why PRG helps or hurts.

Specify which datasets are included in Table 3 for test-time compute and mean accuracy, and provide a detailed table covering all datasets from Tables 1 and 2 to verify whether the reported gain percentage is consistent.

The consistently lower ECE scores in Table 10 are encouraging; however, recent methods from Tables 1 and 2 should also be included in this comparison.

Why was [1] not included in the comparison tables, given its relevance, non-gradient-based adaptation, and low ECE?

References:
[1] Niu, Shuaicheng, et al., “Test-time model adaptation with only forward passes,” ICML 2024.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper introduces SURE (Semantic Uncertainty Regularization), a novel approach for enhancing the stability of predictions in test-time adaptation (TTA) tasks. The core idea involves constructing and iteratively updating a Prototype-Reliability Graph (PRG), which captures inter-class semantic relationships derived from text embeddings. The PRG is dynamically refined based on the reliability of class-wise predictions during test time. Final predictions are obtained by combining the original model outputs with smoothed predictions informed by the PRG structure. The authors validate their method through comprehensive experiments under both natural distribution shifts and cross-dataset generalization scenarios, utilizing CLIP models with ResNet-50 and ViT-Base backbones.

### Strengths
- While TTA is known to be susceptible to noisy predictions, the authors propose a principled approach to mitigate this issue by leveraging statistical measures. Specifically, they downweight classes with high standard deviations (as shown in Equation 4), thereby enhancing the reliability of the constructed PRG.
- The experimental evaluation is comprehensive, considering both the diversity of test datasets and the inclusion of multiple backbone architectures, ResNet-50 and ViT-Base.
- The paper provides in-depth analyses through ablation studies, test-time inference behavior, and hyperparameter sensitivity, which collectively strengthen the empirical validity of the proposed method.

### Weaknesses
- The use of the term graph to describe inter-class relationships may be potentially misleading. In machine learning, graph typically refers to structures processed by specialized architectures such as GNNs. While the proposed representation can be interpreted as nodes and edges, the terminology might cause confusion for readers expecting conventional graph-based methods.
- The reliability estimation in Equation 4 could be further refined. Using the maximum standard deviation as a denominator may accompany instability, as this value itself can be noisy. Alternative formulations such as leveraging the cumulative distribution function (CDF) of each class could offer more robust normalization.
- Several baseline methods demonstrate performance comparable to SURE on specific test sets. For instance, DPE performs similarly in Table 1 (CLIP-RN50), and ZERO shows comparable results in Table 1 (CLIP-ViT-B). However, ZERO is omitted from the test-time inference comparison in Table 3, despite its relevance.
- In Table 4, the most substantial performance gain is attributed to the ProtoOnly variant, which is not the core contribution of the paper. Although the full PRG with regularization yields additional improvements, the gains appear modest compared to the use of prototype vectors alone in some test sets (e.g., on ImageNet-A).
- The final prediction is computed as a simple sum of the original model output and the PRG-based prediction. It remains unclear whether this combination is optimal. Exploring weighted combinations (e.g., p(y∣x)+α⋅p_graph(y∣x)) could potentially yield better results.
- Despite the concerns raised such as the use of terminology and certain methodological choices, I am open to further discussion with the authors. I am willing to increase my score if the authors provide clear and convincing responses during the rebuttal phase.

### Questions
Please refer to my weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 3

### Summary
SURE introduces a graph-based test-time adaptation framework for vision-language models (VLMs) under distribution shift. It constructs a dynamic Prototype-Reliability Graph (PRG) that integrates semantic similarity (from text embeddings) and temporal confidence stability of class prototypes. Predictions are refined via iterative logit propagation on PRG, prototype updates, and reliability tracking. This closed-loop mechanism suppresses error propagation and enforces semantic consistency.

### Strengths
1. Principled structured regularization: First to jointly model semantic affinity and class-wise reliability evolution in TTA, moving beyond instance-level confidence.
2. Closed-loop co-evolution of predictions, prototypes, and graph structure enables stable, error-resistant adaptation.
3. Strong empirical performance: Outperforms entropy minimization (TENT, SAR), prototype-based (Zanella & Ben Ayed, 2024), and recent SOTA across diverse shifts and backbones.

### Weaknesses
1. Figure 1 can be optimized: Text annotations overlap with black boxes, reducing clarity.
2. Eq(9) propagation process is only performed once — why not iterate to convergence? Justification for single-step sufficiency is missing.
3. Algorithm section shows no model parameter updates — does this mean adaptation is entirely prototype-driven? If so, clarify whether backbone features remain frozen and how this impacts representation drift.
4. Eq(12) uses f to update t — unclear why the current prediction f is used to update reliability τ; risks reinforcing early noise.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper proposes SURE (Semantic Uncertainty REgularization), a test-time adaptation (TTA) framework for vision-language models that addresses distribution shift without labeled data. The core contribution is a dynamically evolving Prototype-Reliability Graph (PRG) that combines semantic affinity between class prototypes with class-wise reliability scores based on temporal confidence stability. The method performs three key operations: (1) constructing a sparse graph where edges encode both semantic similarity and prediction reliability, (2) propagating logits through this graph to regularize predictions, and (3) updating prototypes and reliability estimates based on high-confidence pseudo-labels. Evaluated on ImageNet variants and 10 cross-dataset benchmarks using ResNet-50 and ViT-B/16 backbones, SURE achieves marginal but consistent improvements over baselines like DPE and BCA with competitive inference speed.

### Strengths
### Originality

The paper presents a reasonable integration of graph-based reasoning with test-time adaptation for VLMs. The concept of coupling semantic similarity with temporal reliability through a multiplicative joint reliability matrix (Eq. 5) is a sensible design choice. The use of sliding-window averaging for adjacency matrices (Eq. 8) to stabilize graph structure is practical.

### Quality

The experimental evaluation is comprehensive, covering 15 datasets across natural distribution shifts and cross-domain generalization. The ablation study (Table 4) systematically dissects component contributions, showing +1.05% gain from logit propagation and +1.24% from reliability modeling on OOD average. The stability analysis (Tables 7-8) demonstrates low variance across random seeds (<0.3% standard deviation). The efficiency analysis (Table 3) shows reasonable computational cost at 0.067s per sample.

### Clarity

The paper is generally well-structured with clear motivation in Section 1 and detailed methodology in Section 4. Figure 1 provides effective conceptual visualization of the framework. Algorithm 1 presents a clear procedural overview. The notation is mostly consistent, though the reliability score formulation (Eq. 4) could be better justified theoretically.

### Significance

The work addresses a practical problem of test-time adaptation under distribution shift without source data or labels. The framework achieves state-of-the-art results on multiple benchmarks, though improvements are often marginal. The method's applicability across different backbones (ResNet-50, ViT-B/16) and prompt configurations (handcrafted, ensemble, CoOp) demonstrates some generality.

### Weaknesses
### 1. Limited Novelty and Incremental Gains

The core mechanisms are not particularly novel: prototype adaptation follows Zhou et al. (2025) (Eq. 2), cosine similarity graphs are standard in graph-based learning, and confidence thresholding for pseudo-label filtering is widely used. The main contribution is combining these with a reliability weighting scheme, but the conceptual leap is limited. More critically, **empirical gains are marginal**: on ImageNet variants with ViT-B, SURE achieves 66.23% vs. DPE's 65.93% (+0.30%) and BCA's 65.37% (+0.86%). The authors acknowledge "the numerical margin over DPE appears modest" and attribute significance to "consistency across seeds," but this doesn't adequately address the limited practical impact.

### 2. Weak Theoretical Justification for Reliability Metric

The reliability score $R_j = \mu_j \cdot (1 - \frac{\sigma_j}{\sigma_{max}})$ (Eq. 4) is presented as an "information-theoretic intuition" for "inverse uncertainty," yet no formal connection to entropy or information theory is established. Why should the product of mean confidence and normalized inverse standard deviation be the optimal reliability measure? The paper would benefit from: (a) theoretical analysis showing this formulation minimizes adaptation error, (b) comparison with alternative reliability metrics (e.g., coefficient of variation, entropy-based measures, Bayesian credible intervals), or (c) ablation showing sensitivity to different formulations. The choice of $\sigma_{max} = 0.5$ appears arbitrary without justification.

### 3. Insufficient Analysis of Failure Cases and Limitations

The paper lacks a critical discussion of when and why SURE fails. For instance:

- **ImageNet-R saturation:** The authors note "performance tends to saturate" on ImageNet-R because "low-level style cues...are less influenced by semantic drift", but don't investigate whether this indicates fundamental limitations of semantic graph-based methods.
- **Modest gains on stable domains:** Improvements on Pets and Cars are described as "modest" because "prototypes...are already compact", but this raises the question: when should practitioners use SURE vs. simpler baselines?
- **Graph construction sensitivity:** What happens when semantic similarity is misleading (e.g., "hot dog" vs. "dog")? How does the method perform with class imbalance or rarely seen classes during the test stream?
- **Negative results:** The "+Graph w/o Rel" variant shows -0.24% on ImageNet-A (Table 4), suggesting graph smoothing can hurt without reliability gating. This deserves deeper analysis.

### 4. Hyperparameter Sensitivity and Tuning Protocol

While Figure 3 shows robustness across hyperparameters, several concerns remain:

- **Validation set usage:** The authors state "all hyperparameters are selected based on performance on the ImageNet validation set" (A.4). This is problematic for test-time adaptation, where access to validation labels violates the unsupervised assumption. How would practitioners tune $\theta$, k, and L in truly label-free scenarios?
- **Neighbor size scaling:** Setting $k = 3 log C$ is presented without justification. Why logarithmic scaling? How sensitive is performance to this choice across datasets with different C?
- **Initialization dependence:** The method initializes $N_i^{proto} = 30000$ confident samples following Zhou et al. (2025), but doesn't analyze sensitivity to this large prior. What if only 1000 or 100 samples are available?

### 5. Limited Baseline Comparisons and Missing Ablations

- **No comparison with uncertainty quantification methods:** The paper claims to address "semantic uncertainty" but doesn't compare with established uncertainty estimation techniques (e.g., temperature scaling, ensemble methods beyond ZERO, Monte Carlo dropout, evidential deep learning).
- **No analysis of graph structure alternatives:** Why top-k sparsification vs. threshold-based or learnable adjacency? How does performance compare to Graph Neural Network variants (e.g., GAT, GraphSAGE) or no sparsification?
- **Missing ablation on sliding window size L:** While Figure 3 shows performance stabilizes at $L \geq 3$, there's no analysis of computational vs. accuracy trade-offs or sensitivity to stream non-stationarity.

### 6. Calibration Analysis is Superficial

Table 10 shows SURE achieves 7.48 ECE on ImageNet-OOD vs. CLIP's 6.29, meaning SURE is **less calibrated** than the base model despite claims of preserving "trustworthy confidence estimation". The authors dismiss this by comparing only to adapted baselines (TPT, C-TPT), not addressing whether the graph regularization inherently degrades calibration. The calibration-accuracy trade-off deserves principled analysis, potentially through post-hoc calibration methods or uncertainty-aware loss terms.

### 7. Writing Quality Issues

- **Vague claims:** "Unlike these efforts, our approach dynamically constructs a class-level graph" (Section 2). How is this fundamentally different from other adaptive graph methods?
- **Over-claiming:** "SURE consistently outperforms prior methods" (Abstract) is misleading given marginal gains and mixed results (e.g., ImageNet-R).

### 8. Reproducibility Concerns

While the appendix provides implementation details, key aspects remain unclear:

- How are ties handled in top-k selection for graph construction?
- How does batch size affect the sliding window buffer updates in online settings?

### Questions
**Q1: Theoretical Justification**

Can you provide formal analysis showing $R_j = \mu_j \cdot (1 - \sigma_j/\sigma_{\max})$ minimizes expected adaptation error or connects to information-theoretic bounds? What about alternative reliability metrics (e.g., $R_j = \mu_j^2 / (\mu_j^2 + \sigma_j^2)$, inverse coefficient of variation)?

**Q2: Failure Mode Analysis**

Under what specific conditions does SURE underperform simpler baselines (e.g., ProtoOnly or BCA)? Can you characterize dataset properties (class count, domain gap, label granularity) where gains are minimal vs. substantial?

**Q3: Hyperparameter Tuning**

How should practitioners select $\theta$, $k$, and $L$ without validation labels? Can you propose unsupervised selection criteria (e.g., prediction consistency, entropy-based heuristics)?

**Q4: Graph Structure Alternatives**

Why is top-k sparsification optimal? Have you compared against threshold-based adjacency ($A_{jk} = W_{jk} \cdot \mathbb{1}(W_{jk} > \tau)$), fully connected graphs with learned attention (GAT-style), or no graph (direct prototype update)?

**Q5: Calibration Trade-off**

Table 10 shows SURE has higher ECE (7.48) than CLIP (6.29) on ImageNet-OOD. Can you incorporate calibration-aware objectives (e.g., temperature scaling, Dirichlet-based losses) or post-hoc calibration to improve trustworthiness without sacrificing accuracy?

**Q6: Class Imbalance and Rare Classes**

How does SURE perform when certain classes are rare or absent in the test stream? Does the initialization $\mu_j = 1.0, \sigma_j = 0.0$ cause over-reliance on initial prototypes for unseen classes?

**Q7: Computational Bottlenecks**

For datasets with large $C$ (e.g., ImageNet's 1000 classes), does the $O(C^2)$ similarity matrix computation become prohibitive? Can you analyze scaling to $C = 10{,}000$ or $C = 100{,}000$?

**Q8: Comparison with Uncertainty Quantification**

How does SURE compare to evidential deep learning (e.g., Dirichlet-based uncertainty), temperature scaling, or ensemble-based uncertainty estimation for identifying reliable predictions?

**Q9: Streaming vs. Batch Settings**

Algorithm 1 processes samples sequentially. How does performance change in batch settings (e.g., mini-batches of 32 or 64 samples)? Does batching improve stability or efficiency?

**Q10: Prompt Engineering Impact**

Table 9 shows CoOp prompts achieve 67.88% vs. Ensemble's 66.23%. Does SURE's reliability mechanism interact differently with learned vs. handcrafted prompts? Should $k$ or $\theta$ be adjusted based on prompt type?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 5

### Summary
Existing test-time adaptation (TTA) methods for vision–language models (VLM) heavily rely on model predictions, making them vulnerable to noisy pseudo-labels under distribution shifts. To address this limitation, the authors propose SURE (Semantic Uncertainty Regularization), a framework that regularizes model predictions via a dynamically evolving prototype–reliability graph (PRG). This graph enables the selective propagation of reliable predictions while suppressing erroneous ones. Extensive experiments on diverse domain-shift benchmarks demonstrate the effectiveness and robustness of the proposed approach.

### Strengths
1. The experimental section is comprehensive — the authors evaluate their method on multiple datasets and under different settings, achieving consistently strong results. The ablation studies further validate the effectiveness of the proposed approach.
2. The proposed method does not require updating model parameters and therefore incurs significantly lower computational overhead compared with baseline approaches.

### Weaknesses
1. Several existing works also incorporate graph structures with CLIP, such as GraphAdapter [1]. In addition, some test-time adaptation approaches have utilized graph-based mechanisms, for example PROGRAM [2]. The authors are encouraged to discuss these graph-related studies in the Related Work section to better position their method and highlight its unique advantages.
2. The design of the PRG appears heuristic. While it is empirically stable, the paper lacks a theoretical analysis or formal guarantee of its stability over time.
3. It would be helpful if the authors could provide a visualization of the proposed PRG — for instance, showing which classes are more closely connected and how the stability evolves over time. Such visualization would make the workflow of the proposed algorithm more intuitive and accessible to readers.

[1] GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph. NeurIPS'23

[2] PROGRAM: PROtotype GRAph Model based Pseudo-Label Learning for Test-Time Adaptation. ICLR'24

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
2