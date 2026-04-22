# BLUE: Bi-layer Heterogeneous Graph Fusion Network for Avian Influenza Forecasting

- Avg Score: 2.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2

## Abstract
Accurate forecasting of avian influenza outbreaks within wild bird populations requires models that account for complex, multi-scale transmission patterns driven by various factors. Spatio-temporal GNN-based models have recently gained traction for infection forecasting due to their ability to capture relations and flow between spatial regions, but most existing frameworks rely solely on spatial regions and their connections. This overlooks valuable genetic information at the case level, such as cases in one region being genetically descended from strains in another, which is essential for understanding how infectious diseases spread through epidemiological linkages beyond geography. We systemically formulate AIV forecasting problem by proposing a Bi-Layer heterogeneous graph fUsion pipEline (BLUE). This pipeline integrates genetic, spatial, and ecological data to achieve highly accurate outbreak forecasting. It 1) defines heterogeneous graphs from multiple information sources and multiple layers, 2) smooths across relation types, 3) performs fusion while retaining structural patterns, and 4) predicts future outbreaks via an autoregressive graph sequence model that captures transmission dynamics over time. To facilitate further research, we release the \textbf{Avian-US} dataset, the dataset for avian influenza outbreak forecasting in the United States, incorporating genetic, spatial, and ecological data across locations. BLUE achieves superior performance over existing baselines, highlighting the value of incorporating multi-layer information into infectious disease forecasting.
    The code is available at: https://anonymous.4open.science/r/BLUE-60F8/README.md.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents BLUE, a bi-layer heterogeneous graph fusion network for avian influenza (AIV)
outbreak forecasting. The method models both spatial and genetic relationships through two graph
layers, a location layer (geographical and ecological data) and a case layer (genetic data of virus samples).
A cross-layer smoothing block (inspired by MRFs) and an information-preserving graph fusion module
align heterogeneous information into a unified structure, optimized with a spectral regularizer ensuring
diffusion consistency. An autoregressive encoder–decoder then predicts future outbreaks. The authors
also introduce a new Avian-US dataset combining genetic, spatial, and ecological modalities. Experiments
show BLUE outperforms strong baselines (e.g., HGT, Cola-GNN, EpiGNN) on both Avian-US and Flu-Japan
datasets.

### Strengths
The paper tackles a scientifically meaningful and socially important problem with clear relevance to both
epidemiology and AI for science. The bi-layer heterogeneous graph design and spectral information-
preserving fusion are innovative and well-motivated. The theoretical analysis (Theorem 3.1) is a strong
addition. Combining spatial, genetic, and ecological information within a principled heterogeneous
framework is technically elegant. Besides, the dataset creation, which integrates genomic and spatial
ecological sources, is a valuable community contribution.

### Weaknesses
1. The model complexity (bi-layer + smoothing + fusion + autoregressive decoder) may be excessive for
datasets with limited sample size. The paper could benefit from a complexity–performance tradeoff
analysis. And the generalization analysis is limited in the paper. While the method is compelling for avian
influenza, the paper lacks experiments or discussion on whether the approach generalizes to other
diseases or epidemic structures (e.g., human COVID-19, plant viruses). 
2. The method is lack of novelty. The design and usage of heterogeneous gnn is standard. The
authors simply leverage it to solve a domain problem. I don't see novelty for the technical part.
1. where does the and in Eq. 3 come from?
2. Line 216, it seems that that the message-passing mechanism is mainly used to ensure "each node
embedding is influenced by its neighbors' semantics".

### Questions
see above.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces BLUE, a bi-layer heterogeneous graph fusion framework for avian influenza forecasting. The proposed model integrates spatial, genetic, and ecological data into a Graph Neural Network (GNN) pipeline that includes the following main components: (1) a bi-layer graph construction that integrates spatial, ecological, and genetic data; (2) a Cross-Layer Smoothing Block (inspired by MRFs) to refine node representations; (3) an Information-Preserving Fusion process that converts the heterogeneous graph into a homogeneous one; and (4) an autoregressive encoder–decoder for multi-step forecasting.

The authors also release a new dataset (Avian-US) combining spatial and genomic features and report performance gains over baselines such as HGT, EpiGNN, and STSGT.

The claimed contributions of this paper are as follows:
1. Propose BLUE, a pipeline that models heterogeneous nodes with multi-type information within a unified framework.
2. Provide a theoretical bound from a spectral perspective for the information-preserving graph fusion, which simplifies heterogeneous graphs without discarding their structural properties.
3. Publicly release the Avian-US dataset and empirically validate BLUE on it, demonstrating superior performance.

### Strengths
1. The paper addresses the critical and significant real-world problem of Avian Influenza forecasting, a task with important implications for global biosecurity and public health.

2. A key contribution is the introduction and public release of the Avian-US dataset, a new benchmark for AIV forecasting. This dataset integrates genetic, spatial, and ecological data across thousands of locations, providing a valuable resource for future research.

3. The proposed BLUE pipeline incorporates specific technical components to manage this heterogeneity, such as a cross-layer smoothing block and an information-preserving spectral regularizer.

### Weaknesses
1. My main concern is that  the novelty of the proposed model  is limited. The proposed approach, which combines many existing techniques into one large system, feels more like a complex engineering integration. It lacks the simplicity and fundamental novelty typically valued by the ICLR community.

2. The proposed model is overly complicated, with insufficient ablation to support each component. The BLUE pipeline combines many existing techniques (R-GCN-like smoothing, LSH sampling, attention gating, spectral loss, autoregressive decoder). The ablation study, while decent, does not disentangle these components.

3. The validation on Flu-Japan data is meaningless. The second experiment, on Flu-Japan, is invalid. By simulating a uniform, homogeneous “case layer,” the authors test a model component that is completely different from what they propose. This experiment fails to validate the paper’s core thesis about fusing genetic data and should be removed or completely reframed. As the authors mention: “We first construct the location layer based on the provided adjacency matrix. Each location node is assigned features based on the reported infection counts across prefectures. To represent case-level information, we simulate case nodes and associate them with their respective infected locations.”

Given this setup, the evaluation on Flu-Japan does not test real heterogeneous data. Thus, the evaluation relies on a single synthetic dataset that does not reflect the claimed capabilities of BLUE.

a. What is the scientific or empirical value of testing a heterogeneous fusion model on a homogeneous, simulated dataset?
b. How does BLUE generalize to other human epidemiological datasets (e.g., influenza-like illness or COVID-19 variants)?
c. How is “genetic similarity” translated into edge weights—through a fixed threshold, adaptive kernel, or another method?


4. The fusion process is unnecessarily complex (LSH sampling + attention gating + spectral regularization). Could the authors provide an ablation that isolates the contribution of the LSH and attention components? For example, what is the performance if one uses a simple k-NN graph on the fused node embeddings but keeps the spectral regularizer? This would clarify whether the complex edge-generation mechanism actually contributes to performance improvements.

### Questions
see weaknesses

### Soundness
1

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
This paper formulates avian influenza forecasting problem by proposing a Bi-Layer heterogeneous graph fUsion pipEline (BLUE). This pipeline integrates genetic, spatial, and ecological data to achieve accurate outbreak forecasting. The Avian-US dataset has been released, and BLUE achieves superior performance over existing baselines.

### Strengths
1. This paper deals with a very important problem in the real world.

2. The authors release a new dataset for benchmarking avian influenza forecasting. This contribution should be very meaningful for the advancement of this field.

3. The proposed framework, BLUE, shows good engineering with commonly-used techniques.

### Weaknesses
1. It is not very clear why the avian influenza forecasting needs more specialized methods while the conventional influenza-like and COVID-19 forecasting is not.

2. The proposed framework is a combination of existing techniques, which may not have sufficient novelty. As far as I understand, the novelty is importantly regarded in ICLR. This manuscript may not perfectly suit this venue.

3. The proposed framework seems to be overly complicated, in comparison to the size of the dataset that will be used for the framework. The framework’s complexity and heavy computation may also limit its scalability and real-time applicability in operational surveillance systems.

4. The practicability of the proposed framework is questionable, because it would be very hard to obtain the genetic information needed to build the case layer.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
