# Atlas Matters: Edge Quadratics for Consistent Brain Connectivity Prediction

- Avg Score: 4.40
- Decision: Reject
- Scores: 2, 4, 4, 6, 6

## Abstract
Functional connectivity from resting-state fMRI is a strong substrate for subject-level prediction, yet progress is held back by two issues. First, most architectures ingest FC via node-centric propagation or global attention, leaving higher-order edge interactions implicit. Second, evaluations are inconsistent across seeds, atlas choice, preprocessing, and hyperparameter budgets, which obscures true gains.

We propose a simple edge-image encoder that applies dual atrous spatial pyramid pooling to features and connectivity, coupled with a low-rank quadratic block that makes edge-edge effects explicit and efficient. Beyond design, we introduce a unified protocol with five fixed seeds, harmonized preprocessing, and multiple standard atlases, and we re-run recent GNN and transformer baselines under identical settings. Under this protocol, our model **EdgeQuad** attains the best mean performance on curated functional atlases for ABIDE and ADNI, while on unsupervised parcellations such as Ward and KMeans rankings are mixed, highlighting sensitivity to atlas construction. The quadratic block realizes localized degree-2 interactions with provable stability, explaining robustness. The model is lightweight and computationally efficient. To facilitate rigorous comparison, we release code, exact configs, and per-seed logs via an anonymous link.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes EdgeQuad, an edge-centric framework for brain functional connectivity modeling. The key idea is to treat the connectivity matrix as an edge image and introduce a low-rank quadratic block to capture second-order interactions between edges efficiently. A dual-branch atrous spatial pyramid pooling module is used to model multi-scale connectivity patterns. The authors also establish a standardized benchmark protocol with harmonized preprocessing, fixed random seeds, and consistent atlas settings to ensure reproducibility.

### Strengths
1. The paper tackles an important gap in brain network analysis by emphasizing reproducibility across datasets and atlases, which is often overlooked in the field.
2. The explicit quadratic formulation for edge–edge interactions provides a clean and efficient design compared to complex GNN architectures.

### Weaknesses
1. Confusing paper flow. The overall structure is difficult to follow. The Introduction reads like a method paper—emphasizing technical design and mathematical formulation—whereas the Experiments section reads more like a benchmark report with minimal methodological discussion. 
2. Doubtful experimental results. The reported AUC values are unusually higher than ACC across multiple datasets, which is atypical for balanced binary classification tasks. Normally, these metrics are relatively close, yet in this paper, the model with the highest AUC is often not the one with the highest ACC. This inconsistency raises concerns about the evaluation procedures and metric computation.
Furthermore, compared to prior works using the same datasets—e.g., [1] and subsequent follow-ups [2, 3]—the reported PPMI 4-class accuracy is notably lower than the >60% achieved in those studies. The authors should investigate possible causes and clarify how their setup diverges from established benchmarks.
3. Inconsistent performance advantage. The proposed EdgeQuad model does not consistently outperform baselines across datasets and atlases. Some competing models achieve similar or better results, calling into question whether the quadratic formulation provides a general performance advantage. 
4. Lack of interpretability analysis. A key shortcoming is the absence of ROI-level or network-level interpretation. Identifying discriminative brain regions or subnetworks is crucial for clinical translation. The paper should include visualization or attribution analyses to illustrate how EdgeQuad makes decisions and whether it highlights meaningful biomarkers related to the target conditions.
5. Missing computational efficiency comparison. Since EdgeQuad is described as “lightweight,” a concise table summarizing parameter counts, FLOPs, and inference time compared with baselines is necessary to substantiate this claim. Quantitative evidence would make the efficiency argument more convincing.
6. Lack of discussion and comparison with some existing pooling-based methods [2, 4, 5].

	
[1] Data-driven network neuroscience: On data collection and benchmark. NeurIPS 2023
[2] Contrastive Graph Pooling for Explainable Classification of Brain Networks. TMI 2024
[3] Multi-atlas brain network classification through consistency distillation and complementary information fusion. JBHI 2025
[4] BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis. MIA 2021
[5] Contrastive Brain Network Learning via Hierarchical Signed Graph Pooling Model. TNNLS 2022

### Questions
See Weaknesses

### Soundness
1

### Presentation
1

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
While existing methods for functional connectivities (FCs) modeling were mostly designed based on node-centric propagation and global attention, this paper proposes EdgeQuad to model FCs based on edge interactions. It directly models the FC as an edge image using dual atrous spatial pyramid pooling (ASPP) with a low-rank quadratic block. Moreover, the paper introduces a unified, standardized protocol for preprocessing and evaluation. Experiments were extensively performed on four brain network datasets, and the proposed model showed the best or on-par results compared to baselines.

### Strengths
1)	This paper proposes a novel method for graph learning. By using a dual ASPP scheme, multi-scale features are easily aggregated with a low computational cost.
2)	Extensive experiments were performed on 4 brain network datasets with multiple trainings to show the generalizability of the proposed method.

### Weaknesses
Weakness in Problem Definition 1) One of the main limitations of existing studies that the paper claims is reproducibility, that “the existing reported results often differ in atlas choice, preprocessing, splits …., which makes the results hard to compare” (line 42). However, although these settings may differ across papers, each study has performed experiments under a consistent configuration as in its own framework, allowing fair comparisons within that context. Therefore, the claim that existing results are neither reproducible nor comparable may be overstated. 

Weakness in Problem Definition 2) The current manuscript lacks a comprehensive discussion of related work, particularly regarding recent studies addressing similar problems or employing comparable methodologies. Specifically, the second existing limitation raised by the paper is Edge modeling, namely that recent studies mainly focus on node attributes and higher-order edge interactions are implicit. However, recent approaches [1,2,3] that rely solely on edge attributes have been actively studied, together with the analyses of their higher-order topological structures. Therefore, it is necessary to discuss how their formulation differs from or relates to the proposed edge-centric method and to provide experimental comparisons with these methods. Given the growing number of graph studies, I believe that the authors could find more edge-based methods with open source codes to strengthen the contribution of the proposed method.

[1] Park et al., “Convolving Directed Graph Edges via Hodge Laplacian for Brain Network Analysis”, MICCAI 2023.

[2] Fuchsgruber et al., “Graph Neural Networks for Edge Signals: Orientation Equivariance and Invariance”, ICLR 2025.

[3] Lecha et al., “Higher-Order Topological Directionality and Directed Simplicial Neural Networks”, ICASSP 2025.

Weakness in Presentation) In Fig. 1, it would be more intuitive to illustrate the conceptual mechanisms of each component rather than listing every model layer. Moreover, the formulations in the Method section are somewhat unclear; please refer to the related questions in Q2 and Q3.

Weakness in Lack of Experiment 1) This paper asserts that the proposed method with a low-rank quadratic interaction can avoid deep message passing and thereby handle over-fitting and over-squashing issues caused by deep layer stacks in existing studies. However, there is no experimental evidence to support this claim, e.g., performance comparison with baselines across the number of model layers. 

Weakness in Lack of Experiment 2) Moreover, since the paper accentuates that the model is ‘lightweight’, comparisons on computational efficiency (e.g., number of trainable parameters and training time) would be expected but are not provided.
Minor Weakness - Many notations (e.g., $\mathbb{E}, f(\cdot), h(\cdot), H^{l}, W^{l}$) in preliminary section 2.2 is not explicitly defined, which makes it difficult for readers to fully understand the mathematical formulation without background knowledge on GNNs and Transformers. Moreover, notations in the Method section have room to be improved. For example, the FC matrix $C$ is defined with a size of $H\times W$ in Section 3.1, but $N\times N$ in Section 3.2.

### Questions
1)	As explained in lines 30 and 75, fMRI basically measures brain regional BOLD signals, and FC is a set of Pearson correlations of these node signals, which means that edges are calculated based on node features. However, in lines 119-123 (‘Limits for FC graphs’), it is claimed that “explicit node attributes are scarce” and “FC is inherently edge-valued”, which raises the question of how the authors reconcile these two contradictory statements. 

2)	What is the exact formulation of $A_{edge}(\cdot)$ in Eq. 2? Is it different to $A_{feat}(\cdot)$?

3)	Is the $X$ in Eq. 3 the updated node features obtained from $F’$? Otherwise, at which stage is $F’$ actually used?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes EdgeQuad, a lightweight encoder for resting‑state fMRI functional connectivity (FC) that treats the correlation matrix as an edge image, processes it with dual atrous spatial pyramid pooling (ASPP) and introduces a low‑rank quadratic block to make edge–edge (degree‑2) interactions explicit. A content gate fuses first‑ and second‑order paths, followed by clustering-based readout. Alongside the model, the authors advocate a standardized evaluation protocol and re‑run several recent GNN/Transformer baselines under this setup. Across ABIDE and ADNI the method achieves the best mean AUC/ACC on curated/functional atlases, while rankings are mixed on clustering parcellations. They also provide propositions clarifying expressivity (rank‑k quadratics), locality induced by dual ASPP, and a Lipschitz‑type stability bound. Ablations test permutation sensitivity of ROI ordering and the effect of the quadratic rank.

### Strengths
1. Treating FC as an edge image with dual ASPP plus a low‑rank quadratic interaction is conceptually tidy and easy to implement
2. Section 4.2–4.3 fixes seeds/splits, aligns preprocessing, and re‑runs strong recent baselines under the same hyperparameter grids, then publishes per‑seed logs and configs (footnote/link), which gives transparent evaluation protocol

### Weaknesses
1. Many gains over strong baselines are small on curated atlases (Table 3) or even worse. Given the variance, it is not convinced that the solution is better.
2. Section 4.4 explores only ROI ordering and rank‑k. There is no ablation isolating the roles of feat vs. edge, the content gate, the cluster pooling, or degree normalization in Eq. (2). Without this, it’s hard to attribute where the gains come from.
3. The text mentions the model is “well‑calibrated”, but given the paper, I am not sure what does it mean and there is no quantitative substantiation

### Questions
In addition to the weaknesses above:
1. Since the method outputs a refined C′, it would be helpful to show qualitative examples (subject‑level or cohort‑average) illustrating how dual ASPP changes modular structure vs. raw C.
2. In Eq. (2), specify how D is computed for signed C′ (sum of absolute weights? positive part?)

### Soundness
2

### Presentation
2

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
This paper proposes EdgeQuad, which treats the functional connectivity matrix as an “edge image.” The method uses a dual-branch ASPP module (for feature and connectivity processing) together with a low-rank quadratic interaction to explicitly model edge-to-edge relationships. The authors re-implement a wide range of baselines under a unified evaluation protocol, and the results show good average performance on four datasets.

### Strengths
1. The paper discusses a broad spectrum of hyperparameter choices (atlas, random seed, and hyperparameter budget) and presents a benchmark comparison, which improves reproducibility and fairness.

2. The methodology is clearly described with intuitive explanations and theoretical grounding.

3. The ablation studies are comprehensive, and the discussion on ROI order invariance is particularly insightful, highlighting the model’s robustness to atlas indexing.

### Weaknesses
1. The paper lacks neuroscientific interpretation of the motivation and the proposed edge-to-edge interaction. Why is it necessary to explicitly model such quadratic relations? What biological or network-level insights (e.g., hubs, subnetworks) can this reveal? Also, why choose second-order interactions rather than higher-order ones? The authors should better justify this design choice as the key conceptual contribution.

2. The unified hyperparameter grid may constrain some baselines from achieving their best performance. It would be helpful to include an additional table showing each model’s best practice configuration and results for fair comparison.

3. Although algorithmic complexity is discussed, there is no empirical evidence of computational cost. Please provide training/inference time or FLOPs comparison to support the claim of efficiency.

4. The method section is somewhat repetitive, which makes the paper structure less concise. The authors could streamline the narrative and group the theoretical derivations more coherently.

### Questions
1. Why is the ADHD200 dataset evaluated on only one atlas? Is this due to computational constraints or does the model show limited generalization across atlases?

2. The title claims “Atlas Matters,” but the main text does not clearly emphasize or analyze why and how the atlas choice affects performance. It currently reads more like a regular hyperparameter factor. 

3. I'm worried about the justifiability of CNN that applied on FC. In FC matrices, each element represents the connection between two brain regions, so adjacent elements do not necessarily correspond to anatomically or functionally adjacent areas. The ROI order is usually defined by the atlas and does not guarantee true spatial continuity in brain, which makes the assumption of locality in 2D convolution questionable. Although 2D CNNs may still work empirically, since atlases like AAL or Schaefer are organized by anatomica/functional clusters. the authors should provide more neuroscientific explanations or visualizations to justify why applying 2D convolution on FC is reasonable and what biological meaning the learned local patterns might have.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents EdgeQuad, a framework for brain functional connectivity learning. The key idea is to treat edge connections as images, enabling convolutional architectures to process connectivity patterns directly. EdgeQuad integrates dual CNNs with atrous spatial pyramid pooling (ASPP) to capture both node-level features and inter-regional connections. Under a unified benchmark protocol, EdgeQuad achieves results that are competitive or superior to state-of-the-art methods across multiple brain atlases.

### Strengths
• The idea of modeling edges as images is novel and intuitive, offering a fresh perspective for FCN representation learning.

• The dual-ASPP and low-rank quadratic design is simple yet effective, and the accompanying theoretical analysis is sound.

• The unified experimental protocol, with harmonized preprocessing and consistent hyperparameter settings, represents a significant step toward reproducibility and fair comparison — addressing a longstanding issue in this research area.

• The experiments are comprehensive, spanning four cohorts and five atlases, which demonstrates strong empirical validation.

### Weaknesses
• The paper provides limited neuroscientific interpretation. It remains unclear which brain regions or subnetworks contribute most to model predictions, or whether the quadratic terms uncover interpretable motifs. Incorporating contrastive visualizations or saliency maps (as done in prior works like BQN) would add substantial value.

• The writing and organization require improvement. Several tables are unreferenced, and transitions between sections are occasionally abrupt, making it difficult to follow the narrative flow.

### Questions
It would be better if author could provide figures to visualize edge–edge interactions.

### Soundness
3

### Presentation
2

### Contribution
3
