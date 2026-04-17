# HSG-12M: A Large-Scale Benchmark of Spatial Multigraphs from the Energy Spectra of Non-Hermitian Crystals

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
AI is transforming scientific research by revealing new ways to understand complex physical systems, but its impact remains constrained by the lack of large, high-quality domain-specific datasets. A rich, largely untapped resource lies in non-Hermitian quantum physics, where the energy spectra of crystals form intricate geometries on the complex plane—termed as $\textit{Hamiltonian spectral graphs}$. Despite their significance as fingerprints for electronic behavior, their systematic study has been intractable due to the reliance on manual extraction. To unlock this potential, we introduce $\textbf{Poly2Graph}$ (https://github.com/sarinstein-yan/Poly2Graph): a high-performance, open-source pipeline that automates the mapping of 1-D crystal Hamiltonians to spectral graphs. Using this tool, we present $\textbf{HSG-12M}$ (https://github.com/sarinstein-yan/HSG-12M): a dataset containing 11.6 million static and 5.1 million dynamic Hamiltonian spectral graphs across 1401 characteristic-polynomial classes, distilled from 177 TB of spectral potential data. Crucially, HSG-12M is the first large-scale dataset of $\textit{spatial multigraphs}$—graphs embedded in a metric space where multiple geometrically distinct trajectories between two nodes are retained as separate edges. This simultaneously addresses a critical gap, as existing graph benchmarks overwhelmingly assume simple, non-spatial edges, discarding vital geometric information. Benchmarks with popular GNNs expose new challenges in learning spatial multi-edges at scale. Beyond its practical utility, we show that spectral graphs serve as universal topological fingerprints of polynomials, vectors, and matrices, forging a new algebra-to-graph link. HSG-12M lays the groundwork for data-driven scientific discovery in condensed matter physics, new opportunities in geometry-aware graph learning and beyond.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper introduces Poly2Graph, an automated pipeline converting 1-D crystal Hamiltonians into spectral graphs. This enables the creation of HSG-12M, a large-scale dataset of spatial multigraphs, which is a new benchmark for geometry-aware graph learning.

### Strengths
The work introduces HSG-12M, a large-scale dataset. It addresses a clear gap by providing the first large-scale benchmark focused on spatial multigraphs , which are underrepresented in graph ML.

The authors release the Poly2Graph pipeline. This open-source tool enables researchers to generate new spectral graph datasets , promoting reproducibility and future study.

### Weaknesses
As the reviewer is not an expert in this area, the following points are offered from a more general perspective:

1. The paper’s heavy reliance on domain-specific concepts from non-Hermitian physics and algebraic geometry (e.g., Ronkin functions)  might make the data generation process inaccessible to the broader graph ML community. The physical motivation for the benchmark task (inverse design) feels highly specialized.

2. The benchmark evaluation focuses on standard GNNs (GCN, GAT, GIN)  not explicitly designed for spatial multigraphs. The paper identifies this as a new challenge but stops short of benchmarking against more suitable, geometry-aware architectures, weakening the analysis of the dataset's specific challenges.

3. A key feature is the rich, geometric nature of the multi-edges. However, the reference benchmark discards this by collapsing edge geometry into fixed-size summary features (e.g., length, midpoint). This featurization, while practical, may underutilize the dataset's novel geometric complexity.

4. The authors acknowledge a limitation in the Poly2Graph pipeline called "component fragmentation", where extremely low densities of states can cause graphs to be spuriously disconnected. This raises potential concerns about the topological fidelity of the generated graphs in complex cases.

### Questions
Please see the weaknesses.

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
3

### Summary
The paper presents a computational framework (Poly2Graph) and large-scale dataset that map non-Hermitian Hamiltonians of one-dimensional crystals into spectral graphs, i.e., geometric multigraphs embedded in the complex energy plane. The framework provides efficient construction of spectral graphs from Hamiltonians, enabling systematic dataset generation across varying scales. The largest constructed dataset, HSG-12M, contains 11.6 million graphs spanning 1,401 distinct Hamiltonian (characteristic polynomial) classes. The authors also benchmark multiple graph neural network (GNN) architectures on a graph-level classification task: predicting the underlying Hamiltonian class from the corresponding spectral graph.

### Strengths
- Non-Hermitian physics has emerged as an important area in condensed matter and atomic physics. This work establishes a physics-grounded benchmark for evaluating and developing machine learning models on spatial multigraphs derived from physical systems.
- The dataset is impressive in scale ($\approx$12 million graphs), and the automated spectral graph extraction pipeline is technically solid, with potential benefits for the broader physics community.

### Weaknesses
The paper does not clearly explain whether the proposed task, predicting the Hamiltonian class from its spectral graph, corresponds to a physically meaningful scenario. Specially, my question is two-fold:
- This formulation implicitly assumes situations where the Hamiltonian is unknown but the spectral graph itself can somehow be measured, possibly through experiments. A key question is whether the spectral graph, as defined in this work, represents a physically measurable quantity that can be constructed from experimental data.
- Also, the authors may elaborate on why predicting the Hamiltonian class represents a meaningful task in physics. 

I would be glad to raise my score if the authors can adequately address these concerns.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This submission introduces Poly2Graph, a high-throughput package that converts 1D crystal Hamiltonians into Hamiltonian Spectral Graphs (HSGs). Using Poly2Graph, the authors construct and release HSG-12M which is a large static spatial multigraph dataset with 11.6M static and 5.1M dynamic Hamiltonian spectral graphs across 1,401 classes; For the HSGs, each graph class corresponds to a Hamiltonian family (hopping pattern), the multi-edge spatial geometry is essential and cannot be simplified without loss, and  a GNN surrogate may enable inverse design from desired spectral graphs to candidate material structures (e.g., acoustic metamaterials, circuits, photonic crystals).

### Strengths
1. Clear problem framing & significance. Casting Hamiltonian systems as spatial multigraphs is compelling and bridges non-Hermitian quantum physics with graph representation learning.

2. Scale & engineering contribution. The scale (11.6M static; 5.1M dynamic; 1,401 classes) of the graph data is notable.

3. Breadth of tasks. Static graph classification plus temporal, graph-level tasks opens avenues beyond typical node/edge prediction settings in temporal graphs.

4. Inverse-design angle. The connection between hopping patterns ↔ spectral graphs ↔ structure motivates differentiable surrogates and downstream inverse design—a high-impact direction.

5. Writing & organization. The paper reads smoothly; motivation, pipeline, and applications are explained with good intuition.

### Weaknesses
1. What’s fundamentally new on the modeling side?
Beyond scale and the presence of multi-edges + spatial coordinates, it’s not yet clear which modeling challenges are truly novel relative to existing large-scale domains (molecules, traffic, social). For example: Do standard GNNs (e.g., message passing with geometric encoders) already handle these graphs well? Which failure modes emerge uniquely from spatial multigraphs that are not captured in simple GNNs?

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Poly2Graph, an open-source pipeline that converts one-dimensional non-Hermitian crystal Hamiltonians into Hamiltonian spectral graphs (HSGs): spatial multigraphs representing the geometry of complex energy spectra. The authors build HSG-12M, a dataset of 11.6 M static and 5.1 M dynamic multigraphs spanning 1,401 characteristic-polynomial classes. Each graph encodes spectral topology and geometry via node coordinates and multi-edge trajectories on the complex-energy plane. The dataset is positioned as the first large-scale benchmark for spatial multigraph learning, and as a bridge between algebraic physics data and graph machine learning. Benchmarks with popular GNNs (GCN, GAT, GINE, GraphSAGE, etc.) show that edge-aware models outperform edge-agnostic ones, while overall Top-1 accuracies remain moderate ( 30–60 %) and Top-10 accuracies high (95 %).

### Strengths
1. Impressive scale and engineering: Poly2Graph automates a previously manual physics workflow, generating > 10 M graphs and compressing 177 TB of raw spectra into 256 GB.
2. First large-scale multigraph dataset
3. Novel cross-disciplinary framing: Establishes a link between non-Hermitian band theory and graph representation learning, potentially inspiring new geometry-aware GNNs.
4. Solid baseline benchmarking: Eight GNNs are compared with consistent training budgets. Results reveal real performance gaps between edge-aware and edge-agnostic models.

### Weaknesses
1. Limited justification of scientific or ML value: The paper convincingly shows that the data can be generated, but not why learning from these graphs is necessary or insightful. The benchmark task, namely classifying Hamiltonian families, appears artificial, with no demonstrated physical or methodological payoff.
2. Unclear advantage of the graph representation: The authors do not compare to simpler baselines such as CNNs on spectral images, MLPs on polynomial coefficients, or models trained directly on spectral arrays. It remains unproven that representing spectra as graphs yields better or different information.
3. Representation loss and stability: The extraction from spectra to graph skeletons may discard quantitative details and can fragment edges (acknowledged in Appendix H). No analysis quantifies how much information or robustness is lost.
4. Synthetic and self-contained: All data are algorithmically generated. No connection is made to experimental measurements or to existing real-world multigraph domains.
5. Benchmark insight is shallow: Standard GNNs achieve modest accuracies, but this mainly reflects task complexity and training budget, not necessarily new modeling challenges.
6. Incremental as a benchmark contribution: The work is an engineering milestone rather than a conceptual one. Its usefulness for advancing ML is not clear.

### Questions
1. Why is a graph representation preferable to treating the spectra as 2-D arrays or polynomial coefficients?
2. Can you show any downstream task where learning on graphs leads to qualitatively different or improved results than learning on images?
3. How robust are the extracted graphs to numerical perturbations or thresholding choices?
4. Would CNNs or transformers on spectral images achieve comparable or better accuracy?
5. Are there physical or real-data applications planned where HSG-12M would provide measurable benefit?

### Soundness
3

### Presentation
3

### Contribution
2
