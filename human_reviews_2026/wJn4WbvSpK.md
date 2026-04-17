# CAPSUL: A Comprehensive Human Protein Benchmark for Subcellular Localization

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4

## Abstract
Subcellular localization is a crucial biological task for drug target identification and function annotation. Although it has been biologically realized that subcellular localization is closely associated with protein structure, no existing dataset offers comprehensive 3D structural information with detailed subcellular localization annotations, thus severely hindering the application of promising structure-based models on this task. 
To address this gap, we introduce a new benchmark called $\textbf{CAPSUL}$, a $\textbf{C}$omprehensive hum$\textbf{A}$n $\textbf{P}$rotein benchmark for $\textbf{SU}$bcellular $\textbf{L}$ocalization. It features a dataset that integrates diverse 3D structural representations with fine-grained subcellular localization annotations carefully curated by domain experts. 
We evaluate this benchmark using a variety of state-of-the-art sequence-based and structure-based models, showcasing the importance of involving structural features in this task. Furthermore, we explore reweighting and single-label classification strategies to facilitate future investigation on structure-based methods for this task. 
Lastly, we showcase the powerful interpretability of structure-based methods through a case study on the Golgi apparatus, where we discover a decisive localization pattern $\alpha$-helix from attention mechanisms, demonstrating the potential for bridging the gap with intuitive biological interpretability and paving the way for data-driven discoveries in cell biology.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a subcellular localization dataset called CAPSUL that offers comprehensive 3D structural information with detailed subcellular localization annotated by domain experts. A variety of state-of-the-art methods are included to test the benchmark to demonstrate the importance of introducing structural information to identify the subcellular localization. A case study is also introduced to showcase the powerful interpretability of structure-based methods in cell biology.

### Strengths
1. The challenges and motivation behind the paper are well clarified and clearly demonstrated.

2. Extensive experiments and various tasks are conducted to prove the effectiveness of the proposed benchmarks. Both the quantitative and qualitative results are provided to showcase the importance and contribution of the proposed benchmarks in paving the way of cell biology.

3. The paper is well written and organized. Benchmarks and code are also available.

### Weaknesses
1. The proposed benchmark uses AlphaFold2 to extract the structural information for each protein. However, AlphaFold2 also has limitations on structural information. How will this affect the benchmarks and the following evaluations? When more accurate models emerge, will the benchmarks become outdated? And if the downstream tasks models are more accurate than AlphaFold2, such as AlphaFold3, will this benchmark downstream tasks evaluation be invalid?

2. The baselines in the paper have covered sequence and structure-based models. But there are more recent state-of-the-art models, for instance, OpenFold[1], Boltz[2], that should also be considered in the baselines.

[1] Gustav Ahdritz, Nazim Bouatta, Cristian Floristean, et al. Openfold: retraining alphafold2 yields new insights into its learning mechanisms and capacity for generalization. Nature Methods, 21: 1514–1524, 2024. doi: 10.1038/s41592-024-02272-z
[2] Jeremy Wohlwend, Gabriele Corso, Saro Passaro, Noah Getz, Mateo Reveiz, Ken Leidal, Wojtek Swiderski, Liam Atkinson, Tally Portnoi, Itamar Chinn, Jacob Silterra, Tommi Jaakkola, and Regina Barzilay. Boltz-1: Democratizing biomolecular interaction modeling. bioRxiv, 2024. doi: 10.1101/2024.11.19.624167.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
The paper presents CAPSUL, a benchmark of 20,181 human proteins with amino-acid sequences, Cα coordinates, and 3Di structural tokens, paired with fine-grained (20-category) subcellular localization labels aggregated from UniProt and HPA, along with evidence levels. It evaluates both sequence- and structure-based models (ESM-2/ESM-C, FoldSeek, GCN variants, Graph Transformer, Graph Mamba), explores reweighting and single-label strategies, and provides an interpretability case study (Golgi apparatus) via attention.

### Strengths
1.Unified access to structure (Cα, 3Di tokens) plus fine-grained labels and evidence levels; a clear advance beyond existing sequence-only/coarse-label datasets.
2.Broad coverage of representative sequence and structure baselines; reasonable class-imbalance mitigations (reweighting, focal, single-label) and a “randomized structure” ablation that is logically sound.

### Weaknesses
1.Evidence-level integration may introduce bias: treating non-experimental annotations as positives could inflate text biases.
2.Missing graph-construction details, i.e., edge criteria (kNN/sequence adjacencies), edge features (relative orientation/distance encodings), normalization, length truncation.
3.You are suggested that provide failure-case analyses (e.g., low pLDDT regions, disordered segments).
4.You should fix minor typos and keep notation consistent.
5.It is better to add fusion baselines (sequence+structure early/late fusion) to probe complementarity.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces CAPSUL, a new large-scale human protein benchmark for subcellular localization that integrates comprehensive 3D structural information (Cα coordinates and FoldSeek 3Di tokens) with fine-grained, expert-curated annotations across 20 subcellular compartments. The dataset unifies information from UniProt and the Human Protein Atlas and includes evidence-level labels for experimental support. The authors benchmark a diverse range of state-of-the-art sequence-based and structure-based models, analyse class imbalance, and explore interpretability via Transformer attention. Results demonstrate that explicit 3D geometry provides predictive power comparable to massive sequence pre-training, and that fine-grained compartmental labels uncover biologically meaningful hierarchical structure in localization processes. The paper thus establishes a strong and reproducible foundation for evaluating future multimodal protein models.

### Strengths
**Originality and significance**

* CAPSUL fills a clear and impactful gap in current bio-ML resources by providing the first benchmark where **structural and sequence modalities can be directly compared** for subcellular localization.
* The study yields a **quantitative insight into the trade-off between pre-training and structure**, showing that a modestly sized structure-aware model can match the performance of billion-parameter sequence-only models trained on hundreds of millions of proteins. This constitutes a valuable empirical reference for the community.
* The dataset’s **fine-grained, hierarchical labeling** reveals that localization is a multi-stage process, aligning with known biological transport hierarchies (e.g., generic targeting → organelle entry → sub-organelle retention). This suggests natural directions for **hierarchical or multi-task learning architectures**.

**Technical quality**

* The benchmark suite is broad and technically sound: eight representative models spanning sequence Transformers, geometric GNNs, and graph Transformers.
* Ablations on randomised coordinates convincingly isolate the contribution of geometric structure.
* The exploration of reweighting and single-label training provides concrete, actionable strategies for imbalance mitigation.
* Interpretability analyses identify α-helix transmembrane motifs consistent with established Golgi localization mechanisms—an impressive validation of biological fidelity.

**Clarity and reproducibility**

* The paper is well organised and carefully written; data processing steps are fully documented with evidence codes and validation statistics.
* Supplementary material provides sufficient implementation detail for replication.

**Impact**

* CAPSUL is likely to become a **standard benchmark** for structure-aware protein models. Its explicit link between geometry, hierarchy, and localization creates opportunities for causal and interpretable modeling in computational biology.

### Weaknesses
* The **structure–sequence trade-off** is not yet quantified in a controlled architectural setting. While the results suggest that explicit structure compensates for the absence of large-scale pre-training, a **single unified model trained in both modalities** (e.g., ESM backbone + structural tokens) would enable direct estimation of their relative contributions.
* The paper stops short of leveraging the **hierarchical organization of the labels**. Implementing or evaluating hierarchical loss functions or coarse-to-fine classifiers would make the biological interpretation stronger and could address imbalance more effectively.
* Although the case study on the Golgi is compelling, additional interpretability analyses across other organelles would reinforce the claim that structure-based attention consistently yields mechanistically meaningful motifs.
* The benchmark could report **sample efficiency curves** (performance vs number of training samples) to further substantiate the “structure as information equivalent to pre-training” argument.

### Questions
1. **Quantifying modality equivalence** – Could the authors train a unified model that accepts both sequence and structure to explicitly measure how much performance gain is attributable to each modality? This would clarify whether “one structured protein ≈ 10⁴–10⁵ sequences” holds empirically.
2. **Hierarchical classification** – Given that many compartments (e.g., nucleus → nucleolus, nucleoplasm) are nested, have the authors considered a hierarchical target formulation or conditional prediction pipeline? Such models might reflect biological transport stages and improve minority-class recall.
3. **Causal alignment** – Can the authors comment on whether CAPSUL could enable causal studies linking specific structural motifs (e.g., transmembrane helix length or charge distribution) to localization outcomes?
4. **Fusion and transfer learning** – Have experiments been attempted where pretrained sequence embeddings are fused with structural graphs? This could help quantify complementarity and guide future multimodal protein models.
5. **Dataset generality** – While CAPSUL focuses on human proteins, do the authors foresee extending it to other organisms or to dynamic (context-dependent) localization? Such extensions could test generalisation and evolutionary transfer.

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
3

### Summary
Protein subcellular localization is a fundamental aspect of cell biology, closely related to protein function and drug target identification. Although previous studies have shown that 3D protein structure plays a critical role in determining localization patterns, existing datasets provide sequence-level information, lacking structural data. This limits the development and evaluation of structure-aware models. This paper introduces CAPSUL, a comprehensive human protein benchmark for subcellular localization that integrates 3D structural information with fine-grained localization annotations.

### Strengths
1. CAPSUL combines 3D structural data with detailed subcellular localization annotations, enabling the development of structure-aware models.
2. The dataset includes 20 subcellular compartments, verified by domain experts, ensuring biological accuracy and interpretability.
3. The authors benchmark good structure-based models and propose reweighting and single-label classification strategies to mitigate class imbalance, showing improvements in underrepresented classes.

### Weaknesses
1. The current evaluation is limited to supervised multi-label classification. There is no attempt to leverage structural self-supervised learning or contrastive learning, which are promising directions for structure-aware protein modeling.
2. The structure encoders used are standard graph-based models. More advanced geometric deep learning methods (e.g., SE(3)-equivariant networks, structural diffusion models) are not explored, potentially limiting the upper bound of structural understanding.
3. Although evidence codes are provided, in the main experiments, all annotations are treated as positive, including non-experimental ones, which may introduce label noise. While ablations are provided, a more systematic analysis of how evidence levels affect model robustness is lacking.
4. Despite reweighting and single-label strategies, macro-averaged F1-scores remain low for rare classes, indicating that severe imbalance is still a fundamental challenge.

### Questions
Refer to weaknesses

### Soundness
2

### Presentation
2

### Contribution
3
